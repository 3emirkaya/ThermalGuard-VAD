"""
ThermalGuard-VAD — Saha Kısıtları, Termal Ön İşleme ve Donanım Karşılaştırma Betiği (Benchmark).

Saha Kısıtları ve Mühendislik Karşılaştırmaları:
------------------------------------------------
1. **Ön İşleme Ablasyonu (Preprocessing Ablation)**:
   * Düzenek A (Ham / Raw): Yalnızca Min-Max normalizasyon (Standard RGB yaklaşımı hatası).
   * Düzenek B (Yalnızca CLAHE): Gürültü filtresiz CLAHE (gürültüyü de büyütür).
   * Düzenek C (ThermalGuard Tam Hat): AGC + Ölü Piksel + FPN Çizgi Giderme + Bilateral Filtre + CLAHE.
2. **Sensör Gürültü Dayanımı (Noise Robustness)**:
   * Temiz Sensör vs Saha Koşulları (Sis / Düşük Kontrast, 60mK NETD gürültüsü, Sabit Desen Gürültüsü).
3. **Kenar Cihaz Çıkarım Telemetrisi**:
   * Ortalama Gecikme (Latency ms) ve FPS analizi.
"""
from __future__ import annotations

import argparse
import json
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

import numpy as np
import torch
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt="github"):
        res = " | ".join(headers) + "\n" + " | ".join(["---"] * len(headers)) + "\n"
        for r in rows:
            res += " | ".join(str(x) for x in r) + "\n"
        return res

# Proje kökünü sys.path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalguard.data.synthetic import AnomalyEvent, ThermalPerimeterSimulator
from thermalguard.engine.anomaly import AnomalyScoreConfig, AnomalyScorer
from thermalguard.engine.eval import compute_frame_metrics, compute_pixel_auroc
from thermalguard.engine.infer import StreamingInferenceEngine
from thermalguard.models.stae import build_model
from thermalguard.preprocessing.degradation import DegradationConfig, ThermalDegrader
from thermalguard.preprocessing.thermal import (
    ThermalPreprocessConfig,
    ThermalPreprocessor,
    entropy,
    estimate_noise_sigma,
    rms_contrast,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/stae_thermal_best.pt")
    p.add_argument("--num-frames", type=int, default=160)
    p.add_argument("--out-json", default="benchmark_results.json")
    p.add_argument("--device", default="")
    return p.parse_args()


def run_single_experiment(
    pipeline_name: str,
    prep_cfg: ThermalPreprocessConfig,
    degrader: Optional[ThermalDegrader],
    raw_frames: np.ndarray,
    gt_labels: np.ndarray,
    gt_masks: np.ndarray,
    model,
    device: torch.device,
) -> dict:
    preprocessor = ThermalPreprocessor(prep_cfg)
    scorer = AnomalyScorer(AnomalyScoreConfig(percentile=99.2, temporal_smoothing=0.5))
    engine = StreamingInferenceEngine(
        model=model,
        preprocessor=preprocessor,
        scorer=scorer,
        clip_len=8,
        threshold=0.30,
        device=str(device),
    )

    scores = []
    error_maps = []
    latencies = []
    contrast_list = []
    noise_list = []
    entropy_list = []

    for t in range(len(raw_frames)):
        frame = raw_frames[t].copy()
        if degrader is not None:
            frame = degrader(frame)

        t0 = time.perf_counter()
        res = engine.step(frame)
        lat = (time.perf_counter() - t0) * 1000.0

        latencies.append(lat)
        scores.append(res.anomaly_score)
        error_maps.append(res.error_map)

        # Ön işleme kalite metrikleri
        contrast_list.append(rms_contrast(res.preprocessed))
        noise_list.append(estimate_noise_sigma(res.preprocessed))
        entropy_list.append(entropy(res.preprocessed))

    scores_arr = np.array(scores)
    maps_arr = np.array(error_maps)

    frame_metrics = compute_frame_metrics(scores_arr, gt_labels)
    pixel_auroc = compute_pixel_auroc(maps_arr, gt_masks, sample_stride=4)

    return {
        "pipeline": pipeline_name,
        "auroc": frame_metrics["auroc"],
        "pr_auc": frame_metrics["pr_auc"],
        "f1_score": frame_metrics["best_f1"],
        "pixel_auroc": pixel_auroc,
        "rms_contrast": float(np.mean(contrast_list)),
        "noise_sigma": float(np.mean(noise_list)),
        "entropy": float(np.mean(entropy_list)),
        "latency_ms": float(np.mean(latencies[5:])),  # Isınma sonrası
        "fps": 1000.0 / max(float(np.mean(latencies[5:])), 1e-4),
    }


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Benchmark Başlatıldı | Donanım: {device}")

    # Model başlat (Ağırlık varsa yükle, yoksa sıfırdan mimariyi test et)
    model_cfg = {"widths": (16, 32, 64, 96), "latent": 64, "lstm_hidden": (64, 64), "objective": "reconstruct"}
    model = build_model(model_cfg).to(device)
    if os.path.isfile(args.weights):
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    # Test Sekansı Üret (Sınır ihlali senaryosu)
    sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=args.num_frames, seed=42)
    anomalies = [
        AnomalyEvent(start_frame=50, end_frame=115, anomaly_type="intruder", start_pos=(0.1, 0.65), end_pos=(0.9, 0.68))
    ]
    raw_frames, gt_labels, gt_masks = sim.generate_sequence(anomalies=anomalies, add_sensor_defects=False)

    print("\n" + "=" * 80)
    print("  1. TERMAL ÖN İŞLEME VE SAHA DAYANIKLILIK ABLASYON TESTİ")
    print("=" * 80)

    # Karşılaştırılacak Düzenekler
    pipelines = [
        ("A: Ham (Raw Normalization)", ThermalPreprocessConfig.raw(), None),
        ("B: Yalnızca CLAHE (Filtresiz)", ThermalPreprocessConfig(clahe=True, denoise="none", dead_pixel=False, destripe=False), None),
        ("C: ThermalGuard Tam Hat", ThermalPreprocessConfig(clahe=True, denoise="bilateral", dead_pixel=True, destripe=True), None),
        # Saha Bozulması Eklenmiş Zorlayıcı Test (Sis, Düşük Kontrast, Yüksek NETD Gürültüsü)
        ("A*: Ham + Saha Gürültüsü", ThermalPreprocessConfig.raw(), ThermalDegrader(DegradationConfig.level(2, seed=12))),
        ("C*: ThermalGuard + Saha Gürültüsü", ThermalPreprocessConfig(clahe=True, denoise="bilateral", dead_pixel=True, destripe=True),
         ThermalDegrader(DegradationConfig.level(2, seed=12))),
    ]

    results = []
    table_rows = []

    for name, cfg, degrader in pipelines:
        print(f"[...] Test ediliyor: {name}...")
        res = run_single_experiment(name, cfg, degrader, raw_frames, gt_labels, gt_masks, model, device)
        results.append(res)
        table_rows.append([
            name,
            f"{res['auroc']:.4f}",
            f"{res['pr_auc']:.4f}",
            f"{res['pixel_auroc']:.4f}",
            f"{res['rms_contrast']:.3f}",
            f"{res['noise_sigma']:.2f}",
            f"{res['latency_ms']:.1f} ms",
            f"{res['fps']:.1f}",
        ])

    headers = ["Düzenek / Hat", "AUROC", "PR-AUC", "Piksel-AUC", "Kontrast", "Gürültü σ", "Gecikme", "FPS"]
    table_str = tabulate(table_rows, headers=headers, tablefmt="github")
    print("\n" + table_str + "\n")

    # JSON olarak kaydet
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[✓] Detaylı benchmark sonuçları kaydedildi: {args.out_json}")


if __name__ == "__main__":
    main()
