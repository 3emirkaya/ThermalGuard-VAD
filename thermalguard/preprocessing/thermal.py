"""
Specialized preprocessing pipeline for Long-Wave Infrared (LWIR) thermal imagery.

Why standard RGB pipelines are insufficient:
---------------------------------------------
* LWIR sensors (uncooled microbolometers) output 14/16-bit radiometric raw data;
  the scene dynamic range typically occupies only a narrow sub-range. Naive 8-bit
  quantization results in low-contrast, washed-out imagery.
* Infrared lacks RGB chromaticity and texture; distinguishing information resides
  almost exclusively in local temperature differentials (edges/contours). Global
  histogram equalization blows out large uniform backgrounds (e.g., clear night sky)
  while washing out faint hot human targets. Contrast-Limited Adaptive Histogram
  Equalization (CLAHE) is strictly required.
* Focal Plane Arrays (FPAs) suffer from Fixed Pattern Noise (FPN) — vertical/horizontal
  striping — as well as dead/hot defective pixels, which manifest as false positive
  anomalies in downstream reconstruction networks.
* Temporal noise (Noise Equivalent Temperature Difference, NETD) is high; since CLAHE
  amplifies high-frequency noise, denoising must precede CLAHE.

Pipeline sequence (each stage modularly toggleable):

    raw (8/16-bit) -> AGC (percentile + temporal EMA) -> dead pixel correction
        -> FPN destriping -> spatial resize (INTER_AREA)
        -> denoising (bilateral / NLM / median / gaussian)
        -> CLAHE -> [optional temporal EMA] -> float32 [0, 1]

Implemented with OpenCV/NumPy for embedded edge acceleration (NVIDIA Jetson / ARM CPU).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ThermalPreprocessConfig:
    """Preprocessing parameters. Defaults calibrated for KAIST LWIR benchmarks."""

    output_size: tuple[int, int] = (160, 128)  # (W, H) - model input resolution

    # --- Automatic Gain Control (AGC) ---------------------------------------
    agc: str = "percentile"          # 'percentile' | 'minmax' | 'none'
    agc_low: float = 0.5             # lower percentile (%)
    agc_high: float = 99.5           # upper percentile (%)
    agc_ema: float = 0.0             # 0 = per-frame; >0 -> temporal smoothing of bounds (prevents flicker)

    # --- Sensor defects -----------------------------------------------------
    dead_pixel: bool = True          # impulsive (salt & pepper) dead/hot pixel correction
    dead_pixel_thresh: float = 40.0  # deviation threshold from 3x3 median (8-bit scale)
    destripe: bool = True            # column/row fixed pattern noise (FPN) correction
    destripe_sigma: float = 8.0      # profile smoothing parameter (pixels)
    destripe_ema: float = 0.0        # FPN is static -> accumulate offsets temporally

    # --- Noise reduction ----------------------------------------------------
    denoise: str = "bilateral"       # 'none' | 'gaussian' | 'median' | 'bilateral' | 'nlm'
    bilateral_d: int = 5
    bilateral_sigma_color: float = 20.0
    bilateral_sigma_space: float = 3.0
    nlm_h: float = 7.0
    median_ksize: int = 3
    gaussian_sigma: float = 0.8

    # --- Contrast enhancement -----------------------------------------------
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)

    # --- Miscellaneous ------------------------------------------------------
    temporal_ema: float = 0.0        # >0 -> temporal recursive filter on output (static sensor)
    invert: bool = False             # black-hot -> white-hot

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ThermalPreprocessConfig":
        d = dict(d or {})
        for k in ("output_size", "clahe_grid"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def raw(cls, output_size=(160, 128)) -> "ThermalPreprocessConfig":
        """Ablation baseline: pure min-max normalization and resize only."""
        return cls(output_size=output_size, agc="minmax", dead_pixel=False, destripe=False,
                   denoise="none", clahe=False)


# --------------------------------------------------------------------------- #
#  Stateless individual operators
# --------------------------------------------------------------------------- #
def to_gray_float(frame: np.ndarray) -> np.ndarray:
    """Converts any thermal frame (8/16-bit, 1/3 channel) to single-channel float32."""
    if frame.ndim == 3:
        # Colormapped thermal frames: average across RGB channels
        frame = frame.mean(axis=2) if frame.shape[2] != 4 else frame[..., :3].mean(axis=2)
    return frame.astype(np.float32, copy=False)


def agc_bounds(img: np.ndarray, mode: str, lo: float, hi: float) -> tuple[float, float]:
    if mode == "minmax":
        return float(img.min()), float(img.max())
    if mode == "percentile":
        # Subsampled histogram for fast edge execution
        sub = img[::4, ::4]
        a, b = np.percentile(sub, (lo, hi))
        return float(a), float(b)
    raise ValueError(mode)


def apply_agc(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear gain/offset mapping to uint8 (linear AGC)."""
    scale = 255.0 / max(hi - lo, 1e-6)
    return np.clip((img - lo) * scale, 0, 255).astype(np.uint8)


def correct_dead_pixels(img: np.ndarray, thresh: float = 40.0) -> np.ndarray:
    """Replaces outlier pixels deviating excessively from 3x3 median (preserves structural edges)."""
    med = cv2.medianBlur(img, 3)
    diff = cv2.absdiff(img, med)
    mask = diff > thresh
    if not mask.any():
        return img
    out = img.copy()
    out[mask] = med[mask]
    return out


def estimate_stripe_offsets(img: np.ndarray, sigma: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates column and row FPN offsets.

    Column profile = median along rows. True scene geometry varies smoothly across the profile,
    whereas FPN manifests as high-frequency column-specific offsets.
    Offset = profile - Gaussian-smoothed profile.
    """
    f = img.astype(np.float32)
    col = np.median(f, axis=0)
    row = np.median(f, axis=1)
    k = int(6 * sigma) | 1
    col_s = cv2.GaussianBlur(col[None, :], (k, 1), sigma).ravel()
    row_s = cv2.GaussianBlur(row[:, None], (1, k), sigma).ravel()
    return (col - col_s).astype(np.float32), (row - row_s).astype(np.float32)


def remove_stripes(img: np.ndarray, col_off: np.ndarray, row_off: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) - col_off[None, :] - row_off[:, None]
    return np.clip(f, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray, cfg: ThermalPreprocessConfig) -> np.ndarray:
    m = cfg.denoise
    if m == "none":
        return img
    if m == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), cfg.gaussian_sigma)
    if m == "median":
        return cv2.medianBlur(img, cfg.median_ksize)
    if m == "bilateral":
        # Edge-preserving: suppresses NETD noise without blurring hot intruder contours
        return cv2.bilateralFilter(img, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)
    if m == "nlm":
        return cv2.fastNlMeansDenoising(img, None, h=cfg.nlm_h, templateWindowSize=5, searchWindowSize=11)
    raise ValueError(f"unknown denoise mode: {m}")


_CLAHE_CACHE: dict[tuple, "cv2.CLAHE"] = {}


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    key = (float(clip), tuple(grid))
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
        _CLAHE_CACHE[key] = clahe
    return clahe.apply(img)


# --------------------------------------------------------------------------- #
#  Stateful pipeline for video streaming
# --------------------------------------------------------------------------- #
class ThermalPreprocessor:
    """
    Streaming-compatible thermal preprocessor.

    Maintains internal AGC thresholds, FPN offsets, and temporal filter states
    across successive video frames to eliminate flickering and fixed sensor noise.
    """

    def __init__(self, cfg: Optional[ThermalPreprocessConfig] = None):
        self.cfg = cfg or ThermalPreprocessConfig()
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._agc: Optional[tuple[float, float]] = None
        self._col: Optional[np.ndarray] = None
        self._row: Optional[np.ndarray] = None
        self._ema: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Applies AGC and polarity only (used for raw stream inspection)."""
        c = self.cfg
        img = to_gray_float(frame)
        if c.agc == "none":
            u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            lo, hi = agc_bounds(img, c.agc, c.agc_low, c.agc_high)
            if c.agc_ema > 0 and self._agc is not None:
                a = c.agc_ema
                lo = a * self._agc[0] + (1 - a) * lo
                hi = a * self._agc[1] + (1 - a) * hi
            self._agc = (lo, hi)
            u8 = apply_agc(img, lo, hi)
        if c.invert:
            u8 = 255 - u8
        return u8

    # ------------------------------------------------------------------ #
    def __call__(self, frame: np.ndarray, return_stages: bool = False):
        c = self.cfg
        stages: dict[str, np.ndarray] = {}

        u8 = self.to_uint8(frame)
        if return_stages:
            stages["1_agc"] = u8

        if c.dead_pixel:
            u8 = correct_dead_pixels(u8, c.dead_pixel_thresh)
            if return_stages:
                stages["2_dead_pixel"] = u8

        if c.destripe:
            col, row = estimate_stripe_offsets(u8, c.destripe_sigma)
            if c.destripe_ema > 0 and self._col is not None and self._col.shape == col.shape:
                a = c.destripe_ema
                col = a * self._col + (1 - a) * col
                row = a * self._row + (1 - a) * row
            self._col, self._row = col, row
            u8 = remove_stripes(u8, col, row)
            if return_stages:
                stages["3_destripe"] = u8

        W, H = c.output_size
        if (u8.shape[1], u8.shape[0]) != (W, H):
            u8 = cv2.resize(u8, (W, H), interpolation=cv2.INTER_AREA)
            if return_stages:
                stages["4_resize"] = u8

        u8 = denoise(u8, c)
        if return_stages:
            stages["5_denoise"] = u8

        if c.clahe:
            u8 = apply_clahe(u8, c.clahe_clip, c.clahe_grid)
            if return_stages:
                stages["6_clahe"] = u8

        out = u8.astype(np.float32) * (1.0 / 255.0)
        if c.temporal_ema > 0:
            if self._ema is None or self._ema.shape != out.shape:
                self._ema = out
            else:
                self._ema = c.temporal_ema * self._ema + (1 - c.temporal_ema) * out
            out = self._ema

        if return_stages:
            stages["out"] = (out * 255).astype(np.uint8)
            return out, stages
        return out

    # ------------------------------------------------------------------ #
    def process_sequence(self, frames) -> np.ndarray:
        """Processes a sequence of frames sequentially -> (N, H, W) uint8."""
        self.reset()
        out = [np.round(self(f) * 255).astype(np.uint8) for f in frames]
        return np.stack(out) if out else np.zeros((0, *self.cfg.output_size[::-1]), np.uint8)


# --------------------------------------------------------------------------- #
#  Quality evaluation metrics
# --------------------------------------------------------------------------- #
def rms_contrast(img: np.ndarray) -> float:
    f = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    return float(f.std())


def entropy(img: np.ndarray) -> float:
    u8 = img if img.dtype == np.uint8 else np.clip(img * 255, 0, 255).astype(np.uint8)
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Immerkaer (1996) fast noise standard deviation estimation on 8-bit scale."""
    f = img.astype(np.float64)
    if img.dtype != np.uint8:
        f = f * 255.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    h, w = f.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))
