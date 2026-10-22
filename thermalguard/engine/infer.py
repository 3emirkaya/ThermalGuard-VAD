"""
Gerçek Zamanlı Akış (Real-Time Streaming) Çıkarım Motoru.

Kenar cihaz (Jetson Orin, Xavier, gömülü PC) kısıtlarına göre optimize edilmiştir:
  * Rolling Buffer: Bellekte sadece son T kare tutulur.
  * Tekil Encoder Geçişi: Her yeni gelen kare yalnızca 1 kez encoder'dan geçirilir.
  * Durumlu Ön İşleme: AGC ve FPN ofsetleri kareler arasında saklanır.
  * Anlık Gecikme (Latency) & FPS takibi.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import torch

from ..models.stae import STAE
from ..preprocessing.thermal import ThermalPreprocessor
from .anomaly import AnomalyScorer


@dataclass
class InferenceResult:
    frame_idx: int
    raw_frame: np.ndarray          # Orijinal uint8 (H, W)
    preprocessed: np.ndarray       # Ön işlenmiş float32 [0, 1] (H, W)
    reconstructed: np.ndarray      # Model çıktısı float32 [0, 1] (H, W)
    error_map: np.ndarray          # Hata haritası float32 (H, W)
    anomaly_score: float           # Anlık anomali skoru
    is_anomaly: bool               # score >= threshold
    heatmap_bgr: np.ndarray        # Renkli ısı haritası uint8 (H, W, 3)
    overlay_bgr: np.ndarray        # Termal görüntü + Isı haritası bindirmesi uint8 (H, W, 3)
    latency_ms: float              # Çıkarım süresi (milisaniye)
    fps: float                     # Anlık FPS


class StreamingInferenceEngine:
    """
    Canlı video veya kare dizisi üzerinde kare-kare (frame-by-frame) çıkarım yapar.
    """

    def __init__(
        self,
        model: STAE,
        preprocessor: ThermalPreprocessor,
        scorer: Optional[AnomalyScorer] = None,
        clip_len: int = 8,
        threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        self.clip_len = clip_len
        self.threshold = threshold
        self.preprocessor = preprocessor
        self.scorer = scorer or AnomalyScorer()
        self.scorer.threshold = threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.buffer: Deque[torch.Tensor] = collections.deque(maxlen=clip_len)
        self.raw_buffer: Deque[np.ndarray] = collections.deque(maxlen=clip_len)
        self.frame_count = 0
        self._latencies: Deque[float] = collections.deque(maxlen=30)

    def reset(self) -> None:
        self.buffer.clear()
        self.raw_buffer.clear()
        self.preprocessor.reset()
        self.scorer.reset()
        self.frame_count = 0
        self._latencies.clear()

    @torch.no_grad()
    def step(self, raw_frame: np.ndarray) -> InferenceResult:
        """
        Yeni bir tekil kareyi işler ve anomali sonucunu döndürür.
        """
        t_start = time.perf_counter()
        idx = self.frame_count
        self.frame_count += 1

        # 1. Termal Ön İşleme (AGC, Gürültü Azaltma, CLAHE) -> float32 (H, W) ∈ [0, 1]
        prep_frame = self.preprocessor(raw_frame)
        H, W = prep_frame.shape

        # 2. Tensör formatına dönüştür: (B=1, T=1, C=1, H, W)
        tensor_frame = torch.from_numpy(prep_frame).view(1, 1, 1, H, W).to(self.device)

        self.buffer.append(tensor_frame)
        self.raw_buffer.append(raw_frame)

        # Buffer henüz dolmadıysa (ilk T-1 kare)
        if len(self.buffer) < self.clip_len:
            # Buffer'ı ilk kareyle doldurarak geçişi sağla
            pad_needed = self.clip_len - len(self.buffer)
            clip_tensor = torch.cat([self.buffer[0]] * pad_needed + list(self.buffer), dim=1)
        else:
            # (1, T, 1, H, W)
            clip_tensor = torch.cat(list(self.buffer), dim=1)

        # 3. Model İleri Geçiş (Spatio-Temporal Autoencoder)
        # last_only=True ile sadece son kareyi üretir (çıkarım hızlandırma)
        recon_tensor = self.model(clip_tensor, last_only=True)  # (1, 1, 1, H, W)
        recon_np = recon_tensor[0, 0, 0].cpu().numpy()

        # 4. Hata Haritası ve Anomali Skoru
        target_np = prep_frame
        error_map = self.scorer.compute_error_map(target_np, recon_np)
        score = self.scorer.compute_frame_score(error_map)
        is_anomaly = bool(score >= self.threshold)

        # 5. Görselleştirme Hazırlığı (Heatmap & Overlay)
        prep_u8 = (prep_frame * 255.0).astype(np.uint8)
        heatmap_bgr = self.scorer.generate_heatmap(error_map, colormap=cv2.COLORMAP_JET)

        # Anomali maskesi (belirli hata eşiğinin üstündeki pikseller)
        pixel_threshold = max(float(np.mean(error_map) + 2.0 * np.std(error_map)), 0.05)
        thresh_mask = (error_map > pixel_threshold).astype(np.uint8) if is_anomaly else None

        overlay_bgr = self.scorer.blend_heatmap(
            prep_u8,
            heatmap_bgr,
            alpha=0.6 if is_anomaly else 0.25,
            threshold_mask=thresh_mask,
        )

        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000.0
        self._latencies.append(lat_ms)
        avg_lat = np.mean(self._latencies) if self._latencies else lat_ms
        current_fps = 1000.0 / max(avg_lat, 1e-4)

        return InferenceResult(
            frame_idx=idx,
            raw_frame=raw_frame,
            preprocessed=prep_frame,
            reconstructed=recon_np,
            error_map=error_map,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            heatmap_bgr=heatmap_bgr,
            overlay_bgr=overlay_bgr,
            latency_ms=lat_ms,
            fps=current_fps,
        )
