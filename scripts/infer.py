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
from thermalguard.reporting.logger import EventLogger
from thermalguard.reporting.llm_summary import LLMSummarizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/stae_thermal_best.pt", help="Path to trained model weights")
    p.add_argument("--source", default="", help="Video file, camera index, or thermal image folder path")
    p.add_argument("--synthetic", action="store_true", help="Run embedded thermal perimeter security simulation")
    p.add_argument("--threshold", type=float, default=0.20, help="Anomaly alert score threshold")
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=180, help="Maximum number of frames to process")
    p.add_argument("--gif", default="demo.gif", help="Output GIF path")
    p.add_argument("--mp4", default="", help="Output MP4 video path (optional)")
    p.add_argument("--device", default="", help="cuda | cpu | (empty = auto)")
    p.add_argument("--no-clahe", action="store_true", help="Ablation: Disable CLAHE")
    p.add_argument("--log-dir", default="logs", help="Event log directory (JSON + CSV)")
    p.add_argument("--llm", default="rule",
                   choices=["rule", "ollama", "openai", "google"],
                   help="LLM incident summary backend: rule (offline) | ollama | openai | google")
    p.add_argument("--llm-model", default="", help="LLM model name (empty = backend default)")
    return p.parse_args()


def load_or_init_model(weights_path: str, device: torch.device) -> Tuple[STAE, dict, bool]:
    default_cfg = {
        "widths": (16, 32, 64, 96),
        "latent": 64,
        "lstm_hidden": (64, 64),
        "objective": "reconstruct",
    }
    has_weights = os.path.isfile(weights_path)
    if has_weights:
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
    return model, prep_cfg_dict, has_weights


def main():
    args = parse_args()

    # 1. Cihaz Belirleme
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Çıkarım Cihazı: {device}")

    # 2. Model ve Ön İşleme
    model, saved_prep, has_weights = load_or_init_model(args.weights, device)

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
        has_weights=has_weights,
    )

    visualizer = DashboardVisualizer(
        width=640,
        height=540,
        graph_height=170,
        threshold=args.threshold,
    )

    # 3b. Olay Kaydı (EventLogger) + LLM Özet (LLMSummarizer)
    summarizer = LLMSummarizer(
        backend=args.llm,
        model=args.llm_model or None,
        fps=20.0,
    )
    logger = EventLogger(
        log_dir=args.log_dir,
        persist_frames=3,
        max_gap_frames=15,
    )

    def _on_event(ev):
        summary = summarizer.summarize_and_attach(ev, logger)
        print(f"\n{'='*60}")
        print(f"  [ALERT] Incident #{ev.event_id} | Frames: {ev.start_frame}–{ev.end_frame}")
        print(f"  Peak Score: {ev.peak_score:.4f} | Duration: {ev.duration_frames} frames")
        print(f"  LLM Summary: {summary}")
        print(f"{'='*60}")

    logger._callback = _on_event

    # 4. Stream Source
    frames_generator = []
    gt_labels = None
    gt_masks = None
    if args.synthetic or not args.source:
        print("[i] Starting embedded physical thermal perimeter security simulation...")
        print("    Scenario: Unauthorized human intruder crossing restricted perimeter fence.")
        sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=args.max_frames, seed=42)
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
        print(f"[i] Frame Source: {args.source}")
        source = FrameSource(args.source, max_frames=args.max_frames)
        frames_generator = list(source)
        fps = getattr(source, "fps", 20.0)

    print(f"[✓] Total Frames to Process: {len(frames_generator)} (Target FPS: {fps:.1f})")

    rendered_frames = []
    anomaly_counts = 0
    latencies = []

    print("\n" + "=" * 65)
    print("  ThermalGuard-VAD Real-Time Inference Stream")
    print("=" * 65)

    gt_labels_arr = gt_labels if args.synthetic or not args.source else None

    for i, raw_frame in tqdm(frames_generator, desc="Processing Frames", unit="frame"):
        res = engine.step(raw_frame)
        latencies.append(res.latency_ms)

        if res.is_anomaly:
            anomaly_counts += 1

        anom_type = "unknown"
        if args.synthetic and gt_labels_arr is not None and i < len(gt_labels_arr):
            if gt_labels_arr[i] == 1:
                anom_type = "intruder"

        logger.update(
            frame_idx=res.frame_idx,
            is_anomaly=res.is_anomaly,
            anomaly_score=res.anomaly_score,
            anomaly_type=anom_type,
        )

        hud_frame = visualizer.render_frame(res)
        rendered_frames.append(hud_frame)

    avg_lat = float(np.mean(latencies))
    avg_fps = 1000.0 / max(avg_lat, 1e-4)

    print("\n" + "=" * 65)
    print("  PERFORMANCE AND INFERENCE TELEMETRY REPORT")
    print("=" * 65)
    print(f"  * Total Processed Frames     : {len(rendered_frames)}")
    print(f"  * Detected Anomaly Frames    : {anomaly_counts} ({100.0 * anomaly_counts / max(1, len(rendered_frames)):.1f}% of sequence)")
    print(f"  * Mean Latency               : {avg_lat:.2f} ms / frame")
    print(f"  * Inference Throughput       : {avg_fps:.1f} FPS (Target >20 FPS: {'PASSED' if avg_fps >= 20 else 'CPU MODE'})")
    print("=" * 65)

    # Finalize event logger
    all_events = logger.finalize()
    print(f"\n[✓] Recorded Anomaly Events   : {len(all_events)}")
    print(f"[✓] Event Log (JSON)          : {logger.json_path}")
    print(f"[✓] Event Log (CSV)           : {logger.csv_path}")
    if all_events:
        print("\n  Event Summaries:")
        for ev in all_events:
            print(f"  #{ev.event_id}: Frames {ev.start_frame}-{ev.end_frame} | Peak Score: {ev.peak_score:.4f}")
            if ev.llm_summary:
                print(f"      → {ev.llm_summary}")

    # Save visual outputs
    if args.gif and rendered_frames:
        print(f"\n[+] Generating {args.gif} ({len(rendered_frames)} frames, 15 FPS)...")
        save_demo_gif(rendered_frames, output_path=args.gif, fps=15.0)

    if args.mp4 and rendered_frames:
        save_demo_video(rendered_frames, output_path=args.mp4, fps=fps)

    print("\n[✓] Inference pipeline completed successfully!")


if __name__ == "__main__":
    main()
