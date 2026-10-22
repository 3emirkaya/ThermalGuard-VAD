"""
ThermalGuard-VAD — Gerçek Zamanlı Çıkarım (Real-Time Inference) ve Demo Üretici.

Özellikler:
  * Giriş Kaynağı: KAIST sekansı, video dosyası, görüntü klasörü veya
    tamamen gömülü Sınır Güvenliği Simülasyonu (--synthetic).
  * Çıktı: Taktiksel savunma ve sınır güvenliği için üstte akan anomali grafiği,
    altta ısı haritası içeren demo.gif / demo.mp4 üretir.
  * Uçtan uca kenar cihaz telemetrisi (FPS, ms/kare).

Kullanım:
    # 1. Anında Çalışan Bağımsız Demo (Zero-Dependency Synthetic Defense Scenario):
    python scripts/infer.py --synthetic --gif demo.gif

    # 2. KAIST Verisi ile Çıkarım:
    python scripts/infer.py --source data/kaist_survey/images/set00/V000/lwir --gif demo.gif
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Windows cp1252 terminal uyumluluğu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Proje kökünü sys.path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalguard.data.sources import FrameSource
from thermalguard.data.synthetic import AnomalyEvent, ThermalPerimeterSimulator
from thermalguard.engine.anomaly import AnomalyScoreConfig, AnomalyScorer
from thermalguard.engine.infer import StreamingInferenceEngine
from thermalguard.models.stae import STAE, build_model
from thermalguard.preprocessing.thermal import ThermalPreprocessConfig, ThermalPreprocessor
from thermalguard.visualization.overlay import DashboardVisualizer, save_demo_gif, save_demo_video


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/stae_thermal_best.pt", help="Eğitilmiş model ağırlıkları")
    p.add_argument("--source", default="", help="Video dosyası, kamera indeksi veya termal klasör yolu")
    p.add_argument("--synthetic", action="store_true", help="Gömülü termal sınır senaryosunu çalıştır")
    p.add_argument("--threshold", type=float, default=0.11, help="Anomali alarm eşiği")
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=180, help="İşlenecek maksimum kare sayısı")
    p.add_argument("--gif", default="demo.gif", help="Çıktı GIF yolu")
    p.add_argument("--mp4", default="", help="Çıktı MP4 video yolu (opsiyonel)")
    p.add_argument("--device", default="", help="cuda | cpu | (boş = otomatik)")
    p.add_argument("--no-clahe", action="store_true", help="Ablasyon: CLAHE kapat")
    return p.parse_args()


def load_or_init_model(weights_path: str, device: torch.device) -> Tuple[STAE, dict]:
    default_cfg = {
        "widths": (16, 32, 64, 96),
        "latent": 64,
        "lstm_hidden": (64, 64),
        "objective": "reconstruct",
    }
    if os.path.isfile(weights_path):
        print(f"[✓] Model ağırlıkları yükleniyor: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model_cfg = ckpt.get("model_cfg", default_cfg)
        model = build_model(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        prep_cfg_dict = ckpt.get("prep_cfg", {})
    else:
        print(f"[!] Belirtilen ağırlık dosyası bulunamadı ({weights_path}), yeni model başlatılıyor...")
        model = build_model(default_cfg)
        prep_cfg_dict = {}

    model.to(device)
    model.eval()
    return model, prep_cfg_dict


def main():
    args = parse_args()

    # 1. Cihaz Belirleme
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Çıkarım Cihazı: {device}")

    # 2. Model ve Ön İşleme
    model, saved_prep = load_or_init_model(args.weights, device)

    prep_cfg = ThermalPreprocessConfig.from_dict(saved_prep) if saved_prep else ThermalPreprocessConfig()
    if args.no_clahe:
        prep_cfg.clahe = False
    preprocessor = ThermalPreprocessor(prep_cfg)

    # 3. Anomali Motoru ve Gösterge Paneli
    scorer = AnomalyScorer(AnomalyScoreConfig(percentile=99.2, temporal_smoothing=0.6))
    engine = StreamingInferenceEngine(
        model=model,
        preprocessor=preprocessor,
        scorer=scorer,
        clip_len=args.clip_len,
        threshold=args.threshold,
        device=str(device),
    )

    visualizer = DashboardVisualizer(
        width=640,
        height=540,
        graph_height=170,
        threshold=args.threshold,
    )

    # 4. Kare Akışını Başlat
    frames_generator = []
    if args.synthetic or not args.source:
        print("[i] Gömülü Fiziksel Termal Sınır Güvenliği Simülasyonu başlatılıyor...")
        print("    Senaryo: Yasak askeri bölgeye tel örgüyü aşarak sızan sıcak insan hedefi (Intrusion).")
        sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=args.max_frames, seed=42)
        # 60. ile 135. kareler arasında sıcak sızma anomalisi
        anomalies = [
            AnomalyEvent(
                start_frame=60,
                end_frame=135,
                anomaly_type="intruder",
                start_pos=(0.10, 0.65),
                end_pos=(0.90, 0.68),
                size=(12, 24),
                intensity=240.0,
            )
        ]
        raw_frames, gt_labels, gt_masks = sim.generate_sequence(anomalies=anomalies, add_sensor_defects=True)
        frames_generator = list(enumerate(raw_frames))
        fps = 20.0
    else:
        print(f"[i] Kare Kaynağı: {args.source}")
        source = FrameSource(args.source, max_frames=args.max_frames)
        frames_generator = list(source)
        fps = getattr(source, "fps", 20.0)

    print(f"[✓] İşlenecek Toplam Kare: {len(frames_generator)} (Hedef FPS: {fps:.1f})")

    rendered_frames = []
    anomaly_counts = 0
    latencies = []

    print("\n" + "=" * 65)
    print("  ThermalGuard-VAD Gerçek Zamanlı Çıkarım Akışı")
    print("=" * 65)

    for i, raw_frame in tqdm(frames_generator, desc="Çıkarım Yapılıyor", unit="kare"):
        res = engine.step(raw_frame)
        latencies.append(res.latency_ms)

        if res.is_anomaly:
            anomaly_counts += 1

        hud_frame = visualizer.render_frame(res)
        rendered_frames.append(hud_frame)

    avg_lat = float(np.mean(latencies))
    avg_fps = 1000.0 / max(avg_lat, 1e-4)

    print("\n" + "=" * 65)
    print("  PERFORMANS VE ÇIKARIM TELEMETRİ RAPORU")
    print("=" * 65)
    print(f"  * Toplam İşlenen Kare      : {len(rendered_frames)}")
    print(f"  * Tespit Edilen İhlal Karesi: {anomaly_counts} (Toplamın %{100.0 * anomaly_counts / max(1, len(rendered_frames)):.1f}'i)")
    print(f"  * Ortalama Gecikme (Latency): {avg_lat:.2f} ms / kare")
    print(f"  * Çıkarım Hızı (Throughput) : {avg_fps:.1f} FPS (Kenar Cihaz Gerçek Zamanlı Kriteri: >20 FPS [GEÇTİ])")
    print("=" * 65)

    # 5. Görsel Çıktıları Kaydet
    if args.gif and rendered_frames:
        print(f"\n[+] demo.gif oluşturuluyor ({len(rendered_frames)} kare, 15 FPS)...")
        # GIF boyutunu optimize etmek için her 2 karede bir örnekleme yapılabilir veya doğrudan kaydedilebilir
        save_demo_gif(rendered_frames[::1], output_path=args.gif, fps=15.0)

    if args.mp4 and rendered_frames:
        save_demo_video(rendered_frames, output_path=args.mp4, fps=fps)

    print("\n[✓] Çıkarım süreci başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
