"""
ThermalGuard-VAD — Spatio-Temporal Autoencoder (STAE) Training Script.

Unsupervised Training Protocol:
  * The model is trained EXCLUSIVELY on nominal normal conditions (thermal clips
    devoid of pedestrians or intruders).
  * Loss Function: L = MSE(x, x_hat) + lambda_ssim * (1 - SSIM(x, x_hat)).
  * Data Sources: KAIST LWIR benchmarks or embedded physical thermal perimeter simulations (--synthetic).

Usage:
    # 1. Fast training with embedded physical thermal simulator (zero external download required):
    python scripts/train.py --synthetic --epochs 12 --batch-size 8

    # 2. Training on downloaded KAIST LWIR benchmark sequences:
    python scripts/train.py --data data/kaist_survey --seqs set00/V000 --epochs 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Terminal utf-8 configuration
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalguard.data.dataset import ThermalClipDataset, build_dataset_from_sequences
from thermalguard.data.kaist import find_sequences, load_sequence
from thermalguard.data.synthetic import ThermalPerimeterSimulator
from thermalguard.models.stae import ReconstructionLoss, STAE, build_model, count_parameters
from thermalguard.preprocessing.thermal import ThermalPreprocessConfig, ThermalPreprocessor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="Train on embedded physical thermal simulator")
    p.add_argument("--data", default="data/kaist_survey", help="KAIST dataset root directory")
    p.add_argument("--seqs", nargs="+", default=["set00/V000"], help="Training sequences")
    p.add_argument("--out-dir", default="weights", help="Directory to save model checkpoints")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--no-clahe", action="store_true", help="Ablation: Disable CLAHE")
    p.add_argument("--augment", action="store_true", help="Apply sensor degradation augmentation during training")
    p.add_argument("--device", default="", help="cuda | cpu | (empty = auto)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Device Selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Execution Device: {device}")

    # 2. Thermal Preprocessing Configuration
    prep_cfg = ThermalPreprocessConfig(
        output_size=(160, 128),
        clahe=not args.no_clahe,
        denoise="bilateral",
        dead_pixel=True,
        destripe=True,
    )
    preprocessor = ThermalPreprocessor(prep_cfg)

    # 3. Dataset Preparation
    if args.synthetic or not os.path.isdir(args.data):
        print("[i] Using embedded physical thermal perimeter security simulator...")
        sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=350, seed=101)
        # Generate 350 purely normal frames devoid of anomalies
        raw_frames, labels, _ = sim.generate_sequence(anomalies=[])
        prep_frames = preprocessor.process_sequence(raw_frames)

        dataset = ThermalClipDataset(
            frames=prep_frames,
            clip_len=args.clip_len,
            stride=2,
            labels=labels,
            only_normal=True,
            augment=args.augment,
        )
    else:
        print(f"[i] Loading KAIST dataset: {args.data}")
        sequences = []
        for s in args.seqs:
            try:
                seq = load_sequence(args.data, s)
                sequences.append(seq)
                print(f"    - {seq.summary()}")
            except Exception as e:
                print(f"    ! Failed to load sequence: {s} ({e})")

        if not sequences:
            print("[!] No valid sequence found, falling back to simulator...")
            sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=300, seed=42)
            raw_frames, labels, _ = sim.generate_sequence(anomalies=[])
            prep_frames = preprocessor.process_sequence(raw_frames)
            dataset = ThermalClipDataset(prep_frames, clip_len=args.clip_len, stride=2, labels=labels, only_normal=True)
        else:
            dataset = build_dataset_from_sequences(
                sequences=sequences,
                preprocessor=preprocessor,
                clip_len=args.clip_len,
                clip_stride=2,
                only_normal=True,
                augment=args.augment,
            )

    print(f"[✓] Available Normal Training Clips: {len(dataset)}")
    if len(dataset) == 0:
        print("[X] Error: No normal training clips available!")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 4. Model and Loss Setup
    model_cfg = {
        "widths": (16, 32, 64, 96),
        "latent": 64,
        "lstm_hidden": (64, 64),
        "objective": "reconstruct",
    }
    model = build_model(model_cfg).to(device)
    print(f"[✓] Spatio-Temporal Autoencoder parameter count: {count_parameters(model):,}")

    criterion = ReconstructionLoss(ssim_weight=args.ssim_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 5. Training Loop
    print("\n" + "=" * 65)
    print("  ThermalGuard-VAD Model Training Started (Unsupervised)")
    print("=" * 65)

    best_loss = float("inf")
    best_path = Path(args.out_dir) / "stae_thermal_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        t0 = time.time()

        for batch in loader:
            clips = batch["clip"].to(device)  # (B, T, 1, H, W)
            optimizer.zero_grad()

            recon = model(clips)  # (B, T, 1, H, W)
            losses = criterion(recon, clips)

            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += losses["mse"].item()

        scheduler.step()
        n_b = max(1, len(loader))
        avg_loss = epoch_loss / n_b
        avg_mse = epoch_mse / n_b
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - "
              f"Total Loss: {avg_loss:.5f} | "
              f"MSE: {avg_mse:.5f} | "
              f"Time: {elapsed:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "prep_cfg": prep_cfg.to_dict(),
                "best_loss": best_loss,
            }, best_path)
            print(f"  --> [*] Updated best model checkpoint: {best_path} (Loss: {best_loss:.5f})")

    print("\n" + "=" * 65)
    print(f"[✓] Training Completed! Best Checkpoint: {best_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
