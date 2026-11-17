"""
PyTorch Dataset — Thermal Video Clips (Spatio-Temporal Sliding Window).

Key features:
  * Completely unsupervised: Training selects only normal clips devoid of
    annotated pedestrians or anomalies (label == 0).
  * Clip format: (T, 1, H, W) float32 in [0, 1].
  * Memory optimization: All frames are stored preprocessed as uint8 (0..255)
    in RAM; conversion to float32 tensors and normalization occurs in __getitem__
    right before GPU transfer.
  * Online thermal data augmentations simulating harsh field conditions (ThermalDegrader).
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
    Extracts (T, C, H, W) clips from thermal sequences using a sliding window.
    """

    def __init__(
        self,
        frames: np.ndarray,                     # (N, H, W) uint8 preprocessed frames
        clip_len: int = 8,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,    # (N,) int8, {-1, 0, 1}
        masks: Optional[np.ndarray] = None,     # (N, H, W) uint8 pixel ground truth
        only_normal: bool = True,
        augment: bool = False,
        degradation_cfg: Optional[DegradationConfig] = None,
    ):
        super().__init__()
        assert frames.ndim == 3, f"Expected shape: (N, H, W), got: {frames.shape}"
        self.frames = frames
        self.clip_len = clip_len
        self.stride = stride
        self.augment = augment
        self.degrader = ThermalDegrader(degradation_cfg or DegradationConfig()) if augment else None
        self.masks = masks

        # Filter valid clip start indices
        valid_indices = []
        n_total = len(frames) - clip_len + 1

        for i in range(0, max(0, n_total), stride):
            if only_normal and labels is not None:
                clip_lbls = labels[i : i + clip_len]
                # If clip contains any anomaly (1) or ambiguous (-1) frame, exclude from training
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
            # Consistent degradation across the clip
            for t in range(len(clip_u8)):
                clip_u8[t] = self.degrader(clip_u8[t])

        # Convert to float32 [0, 1] tensor: (T, 1, H, W)
        clip_t = torch.from_numpy(clip_u8).unsqueeze(1).float() / 255.0

        item = {
            "clip": clip_t,
            "start_idx": start,
            "end_idx": start + self.clip_len,
        }

        if self.labels is not None:
            clip_labels = self.labels[start : start + self.clip_len]
            item["labels"] = torch.from_numpy(clip_labels).long()
            # Anomaly state of the target (last) frame (primary VAD target)
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
    Preprocesses multiple sequences and constructs a unified ThermalClipDataset.
    """
    all_frames = []
    all_labels = []
    all_masks = []
    out_size = preprocessor.cfg.output_size  # (W, H)

    for seq in sequences:
        preprocessor.reset()
        print(f"[+] Preprocessing: {seq.name} ({len(seq.files)} frames)...")
        raw_frames = [read_thermal(f) for f in seq.files]
        u8_frames = preprocessor.process_sequence(raw_frames)
        all_frames.append(u8_frames)
        all_labels.append(seq.labels)

        # Compute ground truth masks
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
