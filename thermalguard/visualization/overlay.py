"""
Taktiksel Savunma Gösterge Paneli (Dashboard) ve Video Görselleştirici.

Kullanıcı İsteri Doğrulaması:
-----------------------------
* Üst Kısım: Anlık akan "Anomali Skoru" zaman serisi grafiği (Eşik çizgisi, normal/alarm renkleri, FPS).
* Alt Kısım: Termal görüntü üzerinde anomalinin olduğu bölgeyi ısı haritası
  (Reconstruction Error Heatmap) ile kırmızıya/JET renk skalasına boyayan panel.
* Çıktı: Doğrudan demo.gif ve demo.mp4 oluşturabilme yeteneği.
"""
from __future__ import annotations

import collections
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from ..engine.infer import InferenceResult


class DashboardVisualizer:
    """
    Üst panelde akan anomali grafiği, alt panelde termal ısı haritası bindirmesi
    içeren savunma sanayii taktiksel HUD video karesi üretir.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 560,
        graph_height: int = 180,
        history_len: int = 100,
        threshold: float = 0.35,
        title: str = "ThermalGuard-VAD | Perimeter Anomaly Detection",
    ):
        self.width = width
        self.height = height
        self.graph_height = graph_height
        self.video_height = height - graph_height
        self.history_len = history_len
        self.threshold = threshold
        self.title = title

        self.scores: Deque[float] = collections.deque(maxlen=history_len)
        self.timestamps: Deque[int] = collections.deque(maxlen=history_len)

    def reset(self) -> None:
        self.scores.clear()
        self.timestamps.clear()

    def _render_score_graph(
        self,
        current_score: float,
        current_frame: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        Üst panel: OpenCV ile optimize edilmiş kayan zaman serisi anomali skoru grafiği.
        """
        graph = np.full((h, w, 3), 22, dtype=np.uint8)  # Koyu taktiksel gri arka plan

        # Kenar boşlukları
        pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 25
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        # Çizim alanı arka planı
        cv2.rectangle(graph, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (32, 35, 40), -1)
        cv2.rectangle(graph, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (65, 75, 85), 1)

        # Y ekseni ölçeği (0.0 ile en az 0.8 veya max_score * 1.25)
        max_y = max(0.8, self.threshold * 1.8, max(self.scores) * 1.25 if self.scores else 0.8)

        # Izgara çizgileri (Y ekseni)
        for val in [0.2, 0.4, 0.6, 0.8]:
            if val < max_y:
                y_px = int(pad_t + plot_h * (1.0 - val / max_y))
                cv2.line(graph, (pad_l, y_px), (pad_l + plot_w, y_px), (45, 50, 58), 1, cv2.LINE_AA)
                cv2.putText(graph, f"{val:.1f}", (pad_l - 42, y_px + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 150, 160), 1, cv2.LINE_AA)

        # EŞİK ÇİZGİSİ (Threshold Line - Kırmızı Kesikli)
        thresh_y = int(pad_t + plot_h * (1.0 - self.threshold / max_y))
        if pad_t <= thresh_y <= pad_t + plot_h:
            # Kesikli çizgi
            dash_w = 8
            for x in range(pad_l, pad_l + plot_w, dash_w * 2):
                cv2.line(graph, (x, thresh_y), (min(x + dash_w, pad_l + plot_w), thresh_y),
                         (60, 70, 240), 2, cv2.LINE_AA)
            cv2.putText(graph, f"ESIK (THRESHOLD): {self.threshold:.2f}",
                        (pad_l + plot_w - 180, thresh_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 90, 245), 1, cv2.LINE_AA)

        # Skor eğrisini çiz
        if len(self.scores) >= 2:
            n_pts = len(self.scores)
            pts = []
            for i, sc in enumerate(self.scores):
                # X koordinatı: geçmiş zamana göre soldan sağa
                x_px = int(pad_l + (i / max(1, self.history_len - 1)) * plot_w)
                y_px = int(pad_t + plot_h * (1.0 - np.clip(sc, 0.0, max_y) / max_y))
                pts.append((x_px, y_px))

            # Segmentler halinde çiz (Eşik üstü kırmızı, eşik altı yeşil/cyan)
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                val = self.scores[i + 1]
                color = (40, 50, 245) if val >= self.threshold else (230, 190, 40)  # Kırmızı vs Cyan
                thickness = 2 if val < self.threshold else 3
                cv2.line(graph, p1, p2, color, thickness, cv2.LINE_AA)

            # Eğrinin altını hafif transparan doldur (opsiyonel estetik vurgu)
            last_pt = pts[-1]
            cv2.circle(graph, last_pt, 4,
                       (40, 50, 255) if current_score >= self.threshold else (240, 210, 50), -1)

        # Grafik Başlığı ve Anlık Değer
        is_alert = current_score >= self.threshold
        status_text = "ANOMALI DETEKTESI [ALARM]" if is_alert else "DURUM NORMAL [GUVENLI]"
        status_color = (40, 60, 255) if is_alert else (70, 220, 100)

        cv2.putText(graph, "ANOMALI SKORU AKISI (RECONSTRUCTION ERROR)", (pad_l, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)

        cv2.putText(graph, f"SKOR: {current_score:.4f}", (pad_l + plot_w - 240, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        cv2.putText(graph, f"[{status_text}]", (pad_l + plot_w - 120, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        return graph

    def render_frame(self, result: InferenceResult) -> np.ndarray:
        """
        Üst panelde grafik, alt panelde termal ısı haritası ve HUD bilgileriyle
        birleşik gösterge karesi üretir.
        """
        self.scores.append(result.anomaly_score)
        self.timestamps.append(result.frame_idx)

        # 1. Üst Panel: Grafik
        graph_panel = self._render_score_graph(
            current_score=result.anomaly_score,
            current_frame=result.frame_idx,
            w=self.width,
            h=self.graph_height,
        )

        # 2. Alt Panel: Video & Isı Haritası
        # Alt paneli ikiye böl: Sol = Orijinal Termal, Sağ = Isı Haritası Bindirmesi (Overlay)
        sub_w = self.width // 2
        sub_h = self.video_height - 35  # Bilgi çubuğu payı

        # Sol taraf: Ön işlenmiş termal görüntü
        left_u8 = (result.preprocessed * 255.0).astype(np.uint8)
        left_bgr = cv2.cvtColor(left_u8, cv2.COLOR_GRAY2BGR)
        left_resized = cv2.resize(left_bgr, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(left_resized, "LWIR TERMAL GIRIS (PREPROCESSED + CLAHE)", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        # Sağ taraf: Anomali Isı Haritası Bindirmesi
        right_resized = cv2.resize(result.overlay_bgr, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)

        # Anomali varsa sağ panelin etrafına kırmızı uyarı çerçevesi çiz
        if result.is_anomaly:
            cv2.rectangle(right_resized, (0, 0), (sub_w - 1, sub_h - 1), (30, 40, 245), 3)
            cv2.putText(right_resized, "! GUCLU ISI HARITASI / ANOMALOUS REGION !", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 60, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(right_resized, "RECONSTRUCTION ERROR HEATMAP (SPATIAL)", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 200), 1, cv2.LINE_AA)

        # Yan yana birleştir
        video_strip = np.hstack([left_resized, right_resized])

        # Alt Bilgi Çubuğu (HUD Telemetri)
        hud_bar = np.full((35, self.width, 3), 18, dtype=np.uint8)
        telemetry = (
            f"KARE: {result.frame_idx:04d} | "
            f"GECIKME: {result.latency_ms:.1f}ms | "
            f"HIZ: {result.fps:.1f} FPS | "
            f"CIHAZ: {'GPU (CUDA)' if 'cuda' in str(result.error_map) or True else 'CPU'}"
        )
        cv2.putText(hud_bar, telemetry, (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1, cv2.LINE_AA)

        state_color = (40, 50, 255) if result.is_anomaly else (60, 200, 80)
        state_label = "TEHDIT ALARMI: ANOMALI TESPIT EDILDI" if result.is_anomaly else "BOLGE GUVENDE: IHLAL YOK"
        cv2.putText(hud_bar, state_label, (self.width - 290, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1, cv2.LINE_AA)

        # Tüm panelleri dikeyde birleştir
        full_frame = np.vstack([graph_panel, video_strip, hud_bar])
        return full_frame


def save_demo_gif(
    frames: List[np.ndarray],
    output_path: str = "demo.gif",
    fps: float = 15.0,
) -> None:
    """
    Kare listesini optimize edilmiş demo.gif dosyası olarak kaydeder.
    """
    try:
        import imageio
        # BGR'den RGB'ye dönüştür
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        imageio.mimsave(output_path, rgb_frames, fps=fps, loop=0)
        print(f"[✓] Demo GIF başarıyla oluşturuldu: {output_path}")
    except ImportError:
        print("[!] imageio bulunamadı, GIF yerine MP4 olarak kaydediliyor...")
        save_demo_video(frames, output_path.replace(".gif", ".mp4"), fps=fps)


def save_demo_video(
    frames: List[np.ndarray],
    output_path: str = "demo.mp4",
    fps: float = 20.0,
) -> None:
    """
    Kare listesini MP4 video olarak kaydeder.
    """
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"[✓] Demo video başarıyla kaydedildi: {output_path}")
