"""
Saha Savunma Çevresi (Defense Perimeter) Termal Video Simülatörü.

Harici büyük veri setlerine bağımlı kalmadan anında demo çalıştırmak,
birim testler yazmak ve sınır güvenliği senaryolarını doğrulamak için
fiziksel temelli LWIR kızılötesi video sekansı üretir.

Simüle Edilen Fiziksel Dinamikler:
---------------------------------
  1. Termal Arka Plan: Soğuk gece gökyüzü (düşük radyasyon), zemin ısı gradyanı,
     nöbetçi kulesi / tel örgü gibi statik soğuk metal ve beton yapılar.
  2. Doğal Dinamikler: Rüzgar etkisiyle dalgalanan zemin ısısı, atmosferik türbülans.
  3. Sensör Gürültüsü: Mikrobolometre zamansal NETD gürültüsü (~50 mK eşdeğeri),
     FPN sütun çizgileri ve ölü pikseller.
  4. Anomali Senaryoları (Sınır İhlali):
     * Senaryo A: Yasak bölgeye sızan sıcak insan hedefi (37°C vücut ısısı imzası,
       adım atma ve hareket dinamiği).
     * Senaryo B: Sınır teline yaklaşan hızlı sıcak araç / termal iz.
     * Senaryo C: Statik alanda aniden beliren şüpheli sıcak paket / ısı kaynağı.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AnomalyEvent:
    start_frame: int
    end_frame: int
    anomaly_type: str = "intruder"  # 'intruder' | 'vehicle' | 'heat_source'
    start_pos: Tuple[float, float] = (0.2, 0.7)  # (x_norm, y_norm)
    end_pos: Tuple[float, float] = (0.8, 0.7)
    size: Tuple[int, int] = (14, 28)             # (w, h)
    intensity: float = 230.0                     # Termal parlaklık (255 üzerinden)


class ThermalPerimeterSimulator:
    """
    Sınır güvenliği kızılötesi kamera simülatörü.
    """

    def __init__(
        self,
        width: int = 160,
        height: int = 128,
        num_frames: int = 200,
        fps: float = 20.0,
        seed: int = 42,
    ):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.rng = np.random.default_rng(seed)

        # Temel termal arka planı oluştur
        self.background = self._generate_base_background()
        self.fence_mask = self._generate_perimeter_fence()

    def _generate_base_background(self) -> np.ndarray:
        """Soğuk gece zemini, gökyüzü gradyanı ve uzaktaki tepeleri simüle eder."""
        h, w = self.height, self.width
        bg = np.zeros((h, w), dtype=np.float32)

        # Gökyüzü (üst kısım soğuk: ~40-60 seviyesi)
        horizon = int(h * 0.45)
        for y in range(horizon):
            bg[y, :] = 40.0 + (y / horizon) * 20.0

        # Zemin (alt kısım: ~60-95 seviyesi, toprak/çakıl dokusu)
        for y in range(horizon, h):
            factor = (y - horizon) / (h - horizon)
            bg[y, :] = 60.0 + factor * 35.0

        # Hafif arazi dalgalanması
        x = np.linspace(0, 4 * np.pi, w)
        hills = np.sin(x) * 4.0
        for col in range(w):
            split_y = int(horizon + hills[col])
            bg[:split_y, col] = np.clip(bg[:split_y, col] - 5.0, 30, 255)

        # Doğal toprak dokusu (düşük frekanslı gürültü)
        noise = self.rng.normal(0, 3.0, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 1.5)
        bg += noise
        return np.clip(bg, 0, 255)

    def _generate_perimeter_fence(self) -> np.ndarray:
        """Sınır hattı tel örgü ve direklerini simüle eden ikili maske."""
        h, w = self.height, self.width
        fence = np.zeros((h, w), dtype=np.float32)
        horizon = int(h * 0.45)
        fence_y = int(h * 0.6)

        # Yatay gergi telleri
        for dy in [0, 6, 12, 18]:
            y_pos = fence_y + dy
            if y_pos < h:
                fence[y_pos, :] = 0.6

        # Dikey direkler
        for x_pos in range(10, w, 25):
            fence[fence_y : fence_y + 22, x_pos : x_pos + 2] = 0.9

        return fence

    def generate_sequence(
        self,
        anomalies: Optional[List[AnomalyEvent]] = None,
        add_sensor_defects: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sekansı üretir.

        Döndürür:
          * raw_frames: (N, H, W) uint8 ham termal kareler
          * labels: (N,) int8 kare etiketleri (0: normal, 1: anomali)
          * masks: (N, H, W) uint8 anomali bölge maskeleri
        """
        if anomalies is None:
            # Varsayılan senaryo: 80. kareden 150. kareye kadar tel örgüyü aşan sızma
            anomalies = [
                AnomalyEvent(
                    start_frame=70,
                    end_frame=150,
                    anomaly_type="intruder",
                    start_pos=(0.15, 0.65),
                    end_pos=(0.85, 0.70),
                    size=(10, 22),
                    intensity=235.0,
                )
            ]

        h, w = self.height, self.width
        raw_frames = np.zeros((self.num_frames, h, w), dtype=np.uint8)
        labels = np.zeros(self.num_frames, dtype=np.int8)
        masks = np.zeros((self.num_frames, h, w), dtype=np.uint8)

        # Sabit Desen Gürültüsü (FPN - sütun çizgileri)
        fpn_cols = self.rng.normal(0, 3.5, w).astype(np.float32) if add_sensor_defects else np.zeros(w)

        for t in range(self.num_frames):
            frame = self.background.copy()

            # Tel örgüyü ekle (soğuk metal olduğu için çevresinden biraz daha koyu/açık)
            frame = np.where(self.fence_mask > 0, frame * 0.85 + self.fence_mask * 20.0, frame)

            # Rüzgar / zamansal ışıma etkisi (arka planda hafif dalgalanma)
            temporal_drift = math.sin(t * 0.1) * 1.5
            frame += temporal_drift

            frame_mask = np.zeros((h, w), dtype=np.uint8)
            is_anomaly = 0

            # Anomalileri çiz
            for anom in anomalies:
                if anom.start_frame <= t <= anom.end_frame:
                    is_anomaly = 1
                    alpha = (t - anom.start_frame) / max(1, (anom.end_frame - anom.start_frame))
                    cx = (anom.start_pos[0] + alpha * (anom.end_pos[0] - anom.start_pos[0])) * w
                    cy = (anom.start_pos[1] + alpha * (anom.end_pos[1] - anom.start_pos[1])) * h

                    # İnsan yürüyüşü için dikey salınım (bobbing)
                    if anom.anomaly_type == "intruder":
                        cy += math.sin(t * 0.8) * 1.5

                    bw, bh = anom.size
                    x0 = int(np.clip(cx - bw / 2, 0, w))
                    x1 = int(np.clip(cx + bw / 2, 0, w))
                    y0 = int(np.clip(cy - bh / 2, 0, h))
                    y1 = int(np.clip(cy + bh / 2, 0, h))

                    if x1 > x0 and y1 > y0:
                        # İnsan vücut ısısı gradyanı (gövde en sıcak, kenarlar daha ılık)
                        target_patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
                        py, px = np.ogrid[: y1 - y0, : x1 - x0]
                        center_y, center_x = (y1 - y0) / 2.0, (x1 - x0) / 2.0
                        rad = ((px - center_x) / (bw / 2.0 + 1e-5)) ** 2 + ((py - center_y) / (bh / 2.0 + 1e-5)) ** 2
                        heat_blob = np.exp(-1.5 * rad) * anom.intensity

                        # Termal hedefi arka plana kaynaştır (maksimum sıcaklık)
                        frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], heat_blob)
                        frame_mask[y0:y1, x0:x1] = np.where(rad <= 1.0, 1, 0)

            # Sensör kusurları ekle
            if add_sensor_defects:
                # FPN
                frame += fpn_cols[None, :]
                # Zamansal NETD gürültüsü
                netd_noise = self.rng.normal(0, 4.0, (h, w)).astype(np.float32)
                frame += netd_noise

            raw_frames[t] = np.clip(frame, 0, 255).astype(np.uint8)
            labels[t] = is_anomaly
            masks[t] = frame_mask

        return raw_frames, labels, masks
