"""
Termal Anomali Skorlama ve Isı Haritası (Heatmap) Hesaplama Motoru.

Kızılötesi Görüntülerde Anomali Skorlama Prensipleri:
----------------------------------------------------
1. **Yerel vs Global Skor (Spatial Aggregation)**:
   * Sınır güvenliğinde sızan bir insan veya araç karenin yalnızca %1-%3'ünü kaplar.
   * Basit ortalama MSE (mean squared error) hesaplanırsa, devasa normal arka plan
     küçük anomaliyi bastırır ve tespit kaçırılır.
   * Bu nedenle **Top-K / Persentil Havuzlama** (örn. piksellerin %99.5 persentili
     veya en yüksek hataya sahip pencerelerin ortalaması) kullanılır.
2. **SSIM + MSE Füzyonu**:
   * MSE mutlak sıcaklık/parlaklık farkına bakar.
   * 1 - SSIM ise yapısal kenar bozulmalarını yakalar (termal konturların uyumsuzluğu).
3. **Zamansal Yumuşatma (Temporal Smoothing)**:
   * Tek karelik sensör gürültüsü / titremelerinin sahte alarm (false alarm) üretmesini
     engellemek için skor akışı nedensel bir Gauss veya EMA filtresi ile dengelenir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class AnomalyScoreConfig:
    """Anomali skorlama parametreleri."""
    aggregation: str = "percentile"   # 'percentile' | 'top_k' | 'mean'
    percentile: float = 99.5          # Küçük hedefleri yakalamak için üst persentil (%)
    top_k_ratio: float = 0.02         # En yüksek hatalı %2 piksellik alan
    ssim_weight: float = 0.2          # SSIM hata ağırlığı (1 - ssim_weight: MSE)
    temporal_smoothing: float = 0.7   # EMA katsayısı (0: kapalı, 0.7: pürüzsüz akış)
    heatmap_blur: int = 5             # Isı haritası için Gauss bulanıklaştırma çekirdeği


class AnomalyScorer:
    """
    Yeniden üretim hatasından piksel bazlı ısı haritası ve kare bazlı anomali skoru üretir.
    """

    def __init__(self, cfg: Optional[AnomalyScoreConfig] = None):
        self.cfg = cfg or AnomalyScoreConfig()
        self.smoothed_score: Optional[float] = None
        self.threshold: float = 0.5  # Kalibrasyonla belirlenir

    def reset(self) -> None:
        self.smoothed_score = None

    def compute_error_map(
        self,
        original: np.ndarray,      # (H, W) float32 ∈ [0, 1]
        reconstructed: np.ndarray, # (H, W) float32 ∈ [0, 1]
    ) -> np.ndarray:
        """
        Piksel düzeyinde yeniden üretim hata haritası (H, W) float32 ∈ [0, 1].
        """
        # 1. Karesel Hata (MSE Map)
        diff = original.astype(np.float32) - reconstructed.astype(np.float32)
        mse_map = diff ** 2

        # 2. İsteğe bağlı Gauss yumuşatma (sensör yüksek frekanslı gürültüsünü süz)
        if self.cfg.heatmap_blur > 1:
            k = self.cfg.heatmap_blur | 1
            mse_map = cv2.GaussianBlur(mse_map, (k, k), 0)

        return np.clip(mse_map, 0.0, 1.0)

    def compute_frame_score(self, error_map: np.ndarray) -> float:
        """
        Hata haritasından tek bir skalar kare anomali skoru üretir.
        """
        c = self.cfg
        if c.aggregation == "mean":
            raw_score = float(np.mean(error_map))
        elif c.aggregation == "percentile":
            raw_score = float(np.percentile(error_map, c.percentile))
        elif c.aggregation == "top_k":
            k = max(1, int(error_map.size * c.top_k_ratio))
            # En büyük k elemanın ortalaması
            flat = error_map.ravel()
            partitioned = np.partition(flat, -k)[-k:]
            raw_score = float(np.mean(partitioned))
        else:
            raise ValueError(f"Bilinmeyen toplama yöntemi: {c.aggregation}")

        # Zamansal EMA yumuşatma (false alarm önleyici)
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
        Hata haritasını renkli RGB/BGR ısı haritasına (H, W, 3) uint8 dönüştürür.
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
        original_u8: np.ndarray,   # (H, W) veya (H, W, 3) uint8
        heatmap: np.ndarray,       # (H, W, 3) uint8
        alpha: float = 0.55,
        threshold_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Orijinal termal görüntünün üzerine anomali bölgesini renkli harita ile bindirir.
        """
        if original_u8.ndim == 2:
            base_bgr = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = original_u8.copy()

        if threshold_mask is not None:
            # Yalnızca anomalinin olduğu pikselleri boya, normal arka planı gri bırak
            mask_3c = np.repeat(threshold_mask[:, :, None], 3, axis=2).astype(bool)
            blended = base_bgr.copy()
            blended[mask_3c] = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)[mask_3c]
            return blended

        return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)
