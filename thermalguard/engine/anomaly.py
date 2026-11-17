"""
Thermal Anomaly Scoring and Reconstruction Heatmap Engine.

Principles of Anomaly Scoring in Infrared Video:
------------------------------------------------
1. **Spatial Aggregation (Local vs. Global Score)**:
   * In perimeter security, an intruder or vehicle occupies only 1%-3% of the frame area.
   * Standard mean squared error (MSE) causes massive normal background regions to wash out
     the small anomalous target, leading to false negatives.
   * We utilize **Top-K / Percentile Pooling** (e.g. 99.5th percentile or mean of top error patches).
2. **SSIM + MSE Fusion**:
   * MSE tracks absolute temperature/intensity deviation.
   * 1 - SSIM captures structural contour degradation (thermal boundary mismatch).
3. **Temporal Smoothing**:
   * To prevent transient single-frame sensor flicker from triggering false alarms,
     the raw score sequence is smoothed using a causal EMA filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class AnomalyScoreConfig:
    """Anomaly scoring configuration."""
    aggregation: str = "percentile"   # 'percentile' | 'top_k' | 'mean'
    percentile: float = 99.5          # Upper percentile (%) to capture compact targets
    top_k_ratio: float = 0.02         # Highest 2% error pixels
    ssim_weight: float = 0.2          # SSIM error weight (1 - ssim_weight: MSE)
    temporal_smoothing: float = 0.7   # EMA coefficient (0: disabled, 0.7: smooth stream)
    heatmap_blur: int = 5             # Gaussian blur kernel for heatmap rendering


class AnomalyScorer:
    """
    Computes pixel-level error heatmaps and frame-level anomaly scores from reconstruction errors.
    """

    def __init__(self, cfg: Optional[AnomalyScoreConfig] = None):
        self.cfg = cfg or AnomalyScoreConfig()
        self.smoothed_score: Optional[float] = None
        self.threshold: float = 0.5  # Calibrated via validation sequence

    def reset(self) -> None:
        self.smoothed_score = None

    def compute_error_map(
        self,
        original: np.ndarray,      # (H, W) float32 in [0, 1]
        reconstructed: np.ndarray, # (H, W) float32 in [0, 1]
    ) -> np.ndarray:
        """
        Pixel-level reconstruction error map (H, W) float32 in [0, 1].
        """
        # 1. Squared error (MSE map)
        diff = original.astype(np.float32) - reconstructed.astype(np.float32)
        mse_map = diff ** 2

        # 2. Optional Gaussian smoothing to suppress high-frequency sensor noise
        if self.cfg.heatmap_blur > 1:
            k = self.cfg.heatmap_blur | 1
            mse_map = cv2.GaussianBlur(mse_map, (k, k), 0)

        return np.clip(mse_map, 0.0, 1.0)

    def compute_frame_score(self, error_map: np.ndarray) -> float:
        """
        Computes a single scalar frame anomaly score from the 2D error map.
        """
        c = self.cfg
        if c.aggregation == "mean":
            raw_score = float(np.mean(error_map))
        elif c.aggregation == "percentile":
            raw_score = float(np.percentile(error_map, c.percentile))
        elif c.aggregation == "top_k":
            k = max(1, int(error_map.size * c.top_k_ratio))
            flat = error_map.ravel()
            partitioned = np.partition(flat, -k)[-k:]
            raw_score = float(np.mean(partitioned))
        else:
            raise ValueError(f"Unknown aggregation method: {c.aggregation}")

        # Temporal EMA smoothing (prevents transient false alarms)
        if c.temporal_smoothing > 0:
            if self.smoothed_score is None:
                self.smoothed_score = raw_score
            else:
                alpha = c.temporal_smoothing
                self.smoothed_score = alpha * self.smoothed_score + (1.0 - alpha) * raw_score
            final_score = self.smoothed_score
        else:
            final_score = raw_score

        return final_score

    def generate_heatmap(
        self,
        error_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        threshold_clip: Optional[float] = None,
    ) -> np.ndarray:
        """
        Converts the error map into a colorized heatmap (H, W, 3) uint8.
        """
        h_norm = error_map.copy()
        if threshold_clip is not None and threshold_clip > 0:
            h_norm = np.clip(h_norm / threshold_clip, 0.0, 1.0)
        else:
            p_max = max(float(np.percentile(h_norm, 99.8)), 1e-6)
            h_norm = np.clip(h_norm / p_max, 0.0, 1.0)

        u8 = (h_norm * 255.0).astype(np.uint8)
        color_map = cv2.applyColorMap(u8, colormap)
        return color_map

    def blend_heatmap(
        self,
        original_u8: np.ndarray,   # (H, W) or (H, W, 3) uint8
        heatmap: np.ndarray,       # (H, W, 3) uint8
        alpha: float = 0.55,
        threshold_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Blends anomalous regions with the original thermal frame using a colored heatmap.
        """
        if original_u8.ndim == 2:
            base_bgr = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = original_u8.copy()

        if threshold_mask is not None:
            # Colorize anomalous pixels only; preserve grayscale context for normal background
            mask_3c = np.repeat(threshold_mask[:, :, None], 3, axis=2).astype(bool)
            blended = base_bgr.copy()
            blended[mask_3c] = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)[mask_3c]
            return blended

        return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)
