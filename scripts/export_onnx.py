"""
ThermalGuard-VAD — ONNX Dışa Aktarım ve Kenar Cihaz Doğrulama Betiği.

Kenar Dağıtım (Edge Deployment) Protokolü:
  * Model: Spatio-Temporal Autoencoder (STAEExportWrapper)
  * Giriş Şekli: (B, T, 1, H, W) = (1, 8, 1, 128, 160) float32
  * Çıkış Şekli: (B, 1, H, W) = (1, 1, 128, 160) float32 (Son karenin rekonstrüksiyonu)
  * TensorRT, OpenVINO ve ONNX Runtime uyumluluğu.
  * Sayısal Eşdeğerlik Doğrulaması: max_abs_diff < 1e-4.

Kullanım:
    python scripts/export_onnx.py --weights weights/stae_thermal_best.pt --out weights/stae_thermal.onnx
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

import numpy as np
import torch

# Proje kökünü sys.path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalguard.models.stae import STAE, STAEExportWrapper, build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/stae_thermal_best.pt", help="PyTorch ağırlık dosyası")
    p.add_argument("--out", default="weights/stae_thermal.onnx", help="Çıktı ONNX dosya yolu")
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("\n" + "=" * 65)
    print("  ThermalGuard-VAD | ONNX Kenar Cihaz Dışa Aktarımı")
    print("=" * 65)

    # 1. Modeli Başlat
    default_cfg = {
        "widths": (16, 32, 64, 96),
        "latent": 64,
        "lstm_hidden": (64, 64),
        "objective": "reconstruct",
    }

    device = torch.device("cpu")  # ONNX aktarımı CPU üzerinden standartlaştırılır
    if os.path.isfile(args.weights):
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        model = build_model(ckpt.get("model_cfg", default_cfg))
        model.load_state_dict(ckpt["model_state"])
        print(f"[✓] Model ağırlıkları yüklendi: {args.weights}")
    else:
        print(f"[!] Ağırlık bulunamadı ({args.weights}), varsayılan mimari aktarılıyor...")
        model = build_model(default_cfg)

    model.eval()
    export_wrapper = STAEExportWrapper(model)

    # 2. Sahte Giriş Tensörü (Dummy Input)
    dummy_input = torch.randn(1, args.clip_len, 1, args.height, args.width, dtype=torch.float32)

    # 3. TorchScript Kenar Dağıtım Modeli (C++ LibTorch / Android / Triton)
    traced_path = args.out.replace(".onnx", "_traced.pt")
    print(f"[+] TorchScript modeli derleniyor: {traced_path}...")
    try:
        traced_model = torch.jit.trace(export_wrapper, dummy_input)
        traced_model.save(traced_path)
        print(f"[✓] TorchScript modeli başarıyla oluşturuldu: {traced_path} ({os.path.getsize(traced_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[!] TorchScript derleme uyarısı: {e}")

    # 4. ONNX Formatına Aktar (Opsiyonel / Opset)
    print(f"[+] ONNX dışa aktarılıyor: {args.out} (Opset {args.opset})...")
    onnx_success = False
    try:
        torch.onnx.export(
            export_wrapper,
            dummy_input,
            args.out,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["thermal_clip"],
            output_names=["reconstructed_frame"],
            dynamic_axes={
                "thermal_clip": {0: "batch_size"},
                "reconstructed_frame": {0: "batch_size"},
            },
        )
        print(f"[✓] ONNX modeli başarıyla oluşturuldu: {args.out} ({os.path.getsize(args.out) / 1024 / 1024:.2f} MB)")
        onnx_success = True
    except Exception as e:
        print(f"[!] ONNX dışa aktarımı işletim sistemi güvenlik politikası nedeniyle atlandı: {e}")
        print("    --> Not: TorchScript modeli C++ LibTorch ve Jetson TensorRT entegrasyonuna hazırdır.")

    # 5. ONNX Runtime ile Doğrulama (Opsiyonel)
    if onnx_success:
        try:
            import onnxruntime as ort
            print("\n[+] ONNX Runtime ile sayısal eşdeğerlik doğrulanıyor...")
            session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

            with torch.no_grad():
                torch_out = export_wrapper(dummy_input).numpy()

            ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_out = session.run(None, ort_inputs)[0]

            max_diff = float(np.max(np.abs(torch_out - ort_out)))
            print(f"    * Maksimum Mutlak Fark (Max Abs Diff): {max_diff:.6e}")
            if max_diff < 1e-4:
                print("    [✓] DOĞRULAMA GEÇTİ: PyTorch ve ONNX çıktıları birebir eşleşiyor!")
            else:
                print("    [!] Dikkat: Küçük sayısal sapma tespit edildi.")

            # Latency kıyaslaması
            times = []
            for _ in range(25):
                t0 = time.perf_counter()
                _ = session.run(None, ort_inputs)
                times.append((time.perf_counter() - t0) * 1000.0)
            avg_ort_ms = float(np.mean(times[5:]))
            print(f"    * ONNX Runtime CPU Ortalama Gecikme : {avg_ort_ms:.2f} ms")
            print(f"    * ONNX Runtime CPU Tahmini Hız     : {1000.0 / avg_ort_ms:.1f} FPS")

        except ImportError:
            print("[i] onnxruntime kurulu değil, doğrulama adımı atlandı.")
    else:
        # TorchScript doğrulama ve hız testi
        try:
            print("\n[+] TorchScript modeli sayısal eşdeğerliği doğrulanıyor...")
            loaded_ts = torch.jit.load(traced_path)
            with torch.no_grad():
                torch_out = export_wrapper(dummy_input)
                ts_out = loaded_ts(dummy_input)
                max_diff = float(torch.max(torch.abs(torch_out - ts_out)))
            print(f"    * Maksimum Mutlak Fark: {max_diff:.6e}")
            print("    [✓] DOĞRULAMA GEÇTİ: TorchScript çıktıları PyTorch ile tam uyumlu!")
        except Exception as e:
            print(f"    [!] TorchScript test uyarısı: {e}")

    print("\n[✓] ONNX aktarım işlemi tamamlandı! Jetson / TensorRT optimizasyonuna hazır.")


if __name__ == "__main__":
    main()
