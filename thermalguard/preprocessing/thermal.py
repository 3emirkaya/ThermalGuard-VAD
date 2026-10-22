"""
Termal (LWIR) görüntülere özel ön işleme hattı.

Neden RGB hattı yetmez?
------------------------
* LWIR sensörleri (mikrobolometre) 14/16-bit radyometrik ham veri üretir; sahnenin
  dinamik aralığı çoğunlukla bu aralığın çok küçük bir kısmını kaplar →
  doğrudan 8-bit'e indirgenince **düşük kontrastlı, "sisli"** kareler oluşur.
* Renk ve doku yoktur; ayırt edici bilgi neredeyse tamamen **yerel sıcaklık
  farklarında** (kenar/kontur) saklıdır → global histogram eşitleme gökyüzü gibi
  büyük homojen bölgeleri patlatırken küçük sıcak hedefleri bastırır. Bu yüzden
  **CLAHE** (yerel, kırpma limitli) kullanılır.
* Odak düzlemi dizisi (FPA) kaynaklı **sabit desen gürültüsü (FPN)** — sütun/satır
  çizgileri — ve **ölü/sıcak pikseller** vardır; bunlar yeniden üretim
  (reconstruction) hatasında sahte anomali olarak görünür.
* Zamansal gürültü (NETD) yüksektir; CLAHE gürültüyü de yükselteceği için
  gürültü azaltma CLAHE'den **önce** yapılmalıdır.

Hat sırası (her adım ayrı ayrı açılıp kapatılabilir):

    ham (8/16-bit) ─► AGC (persentil + zamansal EMA) ─► ölü piksel düzeltme
        ─► FPN/çizgi giderme ─► yeniden boyutlandırma (INTER_AREA)
        ─► gürültü azaltma (bilateral / NLM / median / gaussian)
        ─► CLAHE ─► [opsiyonel zamansal EMA] ─► float32 [0, 1]

Tüm adımlar OpenCV/NumPy ile yazılmıştır → Jetson / ARM CPU üzerinde de çalışır.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Konfigürasyon
# --------------------------------------------------------------------------- #
@dataclass
class ThermalPreprocessConfig:
    """Ön işleme parametreleri. Varsayılanlar KAIST LWIR (640x512) için ayarlıdır."""

    output_size: tuple[int, int] = (160, 128)  # (W, H) — model girişi

    # --- Otomatik kazanç kontrolü (AGC) --------------------------------------
    agc: str = "percentile"          # 'percentile' | 'minmax' | 'none'
    agc_low: float = 0.5             # alt persentil (%)
    agc_high: float = 99.5           # üst persentil (%)
    agc_ema: float = 0.0             # 0 = kare bazlı; >0 → sınırlar zamanda yumuşatılır (titreme önler)

    # --- Sensör kusurları ----------------------------------------------------
    dead_pixel: bool = True          # dürtüsel (salt & pepper) ölü/sıcak piksel düzeltme
    dead_pixel_thresh: float = 40.0  # 3x3 medyandan bu kadar (8-bit) sapan pikseller değiştirilir
    destripe: bool = True            # sütun/satır sabit desen gürültüsü (FPN) giderme
    destripe_sigma: float = 8.0      # profil yumuşatma (piksel)
    destripe_ema: float = 0.0        # FPN statiktir → >0 ile ofsetler zamanda biriktirilir

    # --- Gürültü azaltma -----------------------------------------------------
    denoise: str = "bilateral"       # 'none' | 'gaussian' | 'median' | 'bilateral' | 'nlm'
    bilateral_d: int = 5
    bilateral_sigma_color: float = 20.0
    bilateral_sigma_space: float = 3.0
    nlm_h: float = 7.0
    median_ksize: int = 3
    gaussian_sigma: float = 0.8

    # --- Kontrast iyileştirme ------------------------------------------------
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)

    # --- Diğer ---------------------------------------------------------------
    temporal_ema: float = 0.0        # >0 → çıktı üzerinde zamansal gürültü bastırma (statik kamera)
    invert: bool = False             # black-hot → white-hot

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
        """Ablasyon için: yalnızca min-max normalizasyon + yeniden boyutlandırma."""
        return cls(output_size=output_size, agc="minmax", dead_pixel=False, destripe=False,
                   denoise="none", clahe=False)


# --------------------------------------------------------------------------- #
#  Tekil, durumsuz (stateless) operatörler — testlerde ve görselleştirmede de kullanılır
# --------------------------------------------------------------------------- #
def to_gray_float(frame: np.ndarray) -> np.ndarray:
    """Herhangi bir termal kareyi (8/16-bit, 1/3 kanal) float32 tek kanala çevirir."""
    if frame.ndim == 3:
        # Sahte renkli (pseudo-color) kaydedilmiş termal kareler: kanallar genelde eşit → ortalama.
        frame = frame.mean(axis=2) if frame.shape[2] != 4 else frame[..., :3].mean(axis=2)
    return frame.astype(np.float32, copy=False)


def agc_bounds(img: np.ndarray, mode: str, lo: float, hi: float) -> tuple[float, float]:
    if mode == "minmax":
        return float(img.min()), float(img.max())
    if mode == "percentile":
        # Hız için alt-örneklenmiş histogram (640x512 → ~20k piksel)
        sub = img[::4, ::4]
        a, b = np.percentile(sub, (lo, hi))
        return float(a), float(b)
    raise ValueError(mode)


def apply_agc(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Doğrusal kazanç/ofset → uint8. (Termal kameralardaki 'linear AGC' karşılığı.)"""
    scale = 255.0 / max(hi - lo, 1e-6)
    return np.clip((img - lo) * scale, 0, 255).astype(np.uint8)


def correct_dead_pixels(img: np.ndarray, thresh: float = 40.0) -> np.ndarray:
    """Yalnızca 3x3 medyandan aşırı sapan pikselleri düzeltir (kenarları korur)."""
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
    Sütun ve satır FPN ofsetlerini tahmin eder.

    Sütun profili = her sütunun (satırlar boyunca) medyanı. Gerçek sahne yapısı
    profilde yavaş değişir; FPN ise yüksek frekanslı, sütuna özgü ofsettir.
    Ofset = profil − Gauss ile yumuşatılmış profil.
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
        # Kenar koruyucu: sıcak hedef sınırlarını bulanıklaştırmadan NETD gürültüsünü bastırır.
        return cv2.bilateralFilter(img, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)
    if m == "nlm":
        # En iyi kalite, en yavaş (~3-6 ms @160x128). Kenar cihazda 'bilateral' önerilir.
        return cv2.fastNlMeansDenoising(img, None, h=cfg.nlm_h, templateWindowSize=5, searchWindowSize=11)
    raise ValueError(f"bilinmeyen denoise modu: {m}")


_CLAHE_CACHE: dict[tuple, "cv2.CLAHE"] = {}


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    key = (float(clip), tuple(grid))
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
        _CLAHE_CACHE[key] = clahe
    return clahe.apply(img)


# --------------------------------------------------------------------------- #
#  Durumlu (stateful) hat — video akışı için
# --------------------------------------------------------------------------- #
class ThermalPreprocessor:
    """
    Akış (stream) uyumlu termal ön işleyici.

    AGC sınırları, FPN ofsetleri ve zamansal filtre durumu kareler arasında
    saklanır; böylece canlı yayında parlaklık titremesi (AGC flicker) ve statik
    FPN etkili biçimde bastırılır. Toplu (batch) işlemde ``reset()`` çağırın.

    >>> pp = ThermalPreprocessor()
    >>> x = pp(frame_uint16)          # float32 (H, W) ∈ [0, 1]
    >>> stages = pp(frame, return_stages=True)   # görselleştirme için ara çıktılar
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
        """Yalnızca AGC + polarite (görselleştirme / 'ham' panel için)."""
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
        """Bir kare dizisini sırayla işler → (N, H, W) uint8 (önbellekleme için)."""
        self.reset()
        out = [np.round(self(f) * 255).astype(np.uint8) for f in frames]
        return np.stack(out) if out else np.zeros((0, *self.cfg.output_size[::-1]), np.uint8)


# --------------------------------------------------------------------------- #
#  Kalite metrikleri (README / ablasyon raporları için)
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
    """Immerkær (1996) hızlı gürültü std tahmini (8-bit ölçeğinde)."""
    f = img.astype(np.float64)
    if img.dtype != np.uint8:
        f = f * 255.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    h, w = f.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))
