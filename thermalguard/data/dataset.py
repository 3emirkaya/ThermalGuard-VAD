"""
PyTorch Veri Seti — Termal Video Klipleri (Spatio-Temporal Sliding Window).

Özellikler:
  * Tamamen denetimsiz (unsupervised): Eğitimde yalnızca etiketli yaya / anomali
    bulunmayan (label == 0) normal karelerden oluşan pencereler seçilir.
  * Klip formatı: (T, 1, H, W) float32 ∈ [0, 1].
  * Bellek optimizasyonu: Tüm kareler ön işlenmiş olarak uint8 (0..255) formatında
    RAM'de saklanır; float32 tensöre dönüştürme ve normalizasyon __getitem__'da
    GPU aktarımı öncesinde yapılır.
  * Sahadaki zorlukları taklit eden çevrimiçi veri artırımı (ThermalDegrader).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocessing.degradation import DegradationConfig, ThermalDegrader
from ..preprocessing.thermal import ThermalPreprocessConfig, ThermalPreprocessor
from .kaist import KaistSequence, boxes_to_mask
from .sources import read_thermal


class ThermalClipDataset(Dataset):
    """
    Termal sekanslardan kayan pencere (sliding window) ile (T, C, H, W) klipler çıkarır.
    """

    def __init__(
        self,
        frames: np.ndarray,                     # (N, H, W) uint8 ön işlenmiş kareler
        clip_len: int = 8,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,    # (N,) int8, {-1, 0, 1}
        masks: Optional[np.ndarray] = None,     # (N, H, W) uint8 piksel yer doğruluğu
        only_normal: bool = True,
        augment: bool = False,
        degradation_cfg: Optional[DegradationConfig] = None,
    ):
        super().__init__()
        assert frames.ndim == 3, f"Beklenen şekil: (N, H, W), gelen: {frames.shape}"
        self.frames = frames
        self.clip_len = clip_len
        self.stride = stride
        self.augment = augment
        self.degrader = ThermalDegrader(degradation_cfg or DegradationConfig()) if augment else None
        self.masks = masks

        # Geçerli klip başlangıç indekslerini filtrele
        valid_indices = []
        n_total = len(frames) - clip_len + 1

        for i in range(0, max(0, n_total), stride):
            if only_normal and labels is not None:
                clip_lbls = labels[i : i + clip_len]
                # Klip içinde herhangi bir anomali (1) veya belirsiz (-1) kare varsa eğitimden çıkar
                if np.any(clip_lbls != 0):
                    continue
            valid_indices.append(i)

        self.indices = np.array(valid_indices, dtype=np.int64)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        clip_u8 = self.frames[start : start + self.clip_len].copy()  # (T, H, W)

        if self.augment and self.degrader is not None:
            # Klip bazında tutarlı gürültü/bozulma
            for t in range(len(clip_u8)):
                clip_u8[t] = self.degrader(clip_u8[t])

        # float32 [0, 1] tensöre çevir: (T, 1, H, W)
        clip_t = torch.from_numpy(clip_u8).unsqueeze(1).float() / 255.0

        item = {
            "clip": clip_t,
            "start_idx": start,
            "end_idx": start + self.clip_len,
        }

        if self.labels is not None:
            clip_labels = self.labels[start : start + self.clip_len]
            item["labels"] = torch.from_numpy(clip_labels).long()
            # Son karenin anomali durumu (VAD çıkarımında ana hedef)
            item["target_label"] = int(clip_labels[-1])

        if self.masks is not None:
            item["target_mask"] = torch.from_numpy(self.masks[start + self.clip_len - 1]).float()

        return item


def build_dataset_from_sequences(
    sequences: Sequence[KaistSequence],
    preprocessor: ThermalPreprocessor,
    clip_len: int = 8,
    clip_stride: int = 1,
    only_normal: bool = True,
    augment: bool = False,
    degradation_cfg: Optional[DegradationConfig] = None,
) -> ThermalClipDataset:
    """
    Birden fazla sekansı ön işleyip birleşik ThermalClipDataset oluşturur.
    """
    all_frames = []
    all_labels = []
    all_masks = []
    out_size = preprocessor.cfg.output_size  # (W, H)

    for seq in sequences:
        preprocessor.reset()
        print(f"[+] Ön işleniyor: {seq.name} ({len(seq.files)} kare)...")
        raw_frames = [read_thermal(f) for f in seq.files]
        u8_frames = preprocessor.process_sequence(raw_frames)
        all_frames.append(u8_frames)
        all_labels.append(seq.labels)

        # Maskeleri hesapla
        seq_masks = np.zeros((len(seq.files), out_size[1], out_size[0]), dtype=np.uint8)
        for i, boxes in enumerate(seq.boxes):
            if boxes:
                seq_masks[i] = boxes_to_mask(boxes, seq.src_size, out_size)
        all_masks.append(seq_masks)

    cat_frames = np.concatenate(all_frames, axis=0) if all_frames else np.zeros((0, out_size[1], out_size[0]), dtype=np.uint8)
    cat_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int8)
    cat_masks = np.concatenate(all_masks, axis=0) if all_masks else np.zeros((0, out_size[1], out_size[0]), dtype=np.uint8)

    return ThermalClipDataset(
        frames=cat_frames,
        clip_len=clip_len,
        stride=clip_stride,
        labels=cat_labels,
        masks=cat_masks,
        only_normal=only_normal,
        augment=augment,
        degradation_cfg=degradation_cfg,
    )
