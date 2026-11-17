"""
KAIST Multispectral Pedestrian Benchmark — LWIR thermal channel only.

Anomaly protocol ("Pedestrian-free normality / Perimeter Intrusion"):
  * **Normal (0)**  : Frame contains no annotated objects.
  * **Anomaly (1)** : At least one ``person / people / cyclist`` box with height
    greater than ``min_height`` pixels (visible real-world intruder).
  * **Ambiguous (-1)**: Only distant/tiny or ``person?`` boxes — excluded from both
    training and evaluation (eliminates label ambiguity).

Training is strictly conducted on purely normal (0) clips -> unsupervised.
Annotations are exclusively utilized for downstream validation (AUROC metrics).

Supported directory structures:
  * HF mirror  : images/setXX/VYYY/lwir/I*.jpg  + annotations/setXX/VYYY/I*.xml
  * Original   : images/setXX/VYYY/lwir/I*.jpg  + annotations/setXX/VYYY/I*.txt (bbGt v3)
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sources import frame_index_from_path, list_images

ANOMALY_CLASSES = {"person", "people", "cyclist"}


@dataclass
class Box:
    cls: str
    x: float
    y: float
    w: float
    h: float
    occlusion: int = 0


def parse_annotation(path: str) -> list[Box]:
    if not os.path.exists(path):
        return []
    if path.endswith(".xml"):
        root = ET.parse(path).getroot()
        out = []
        for o in root.findall("object"):
            bb = o.find("bndbox")
            occ = o.find("occlusion")
            out.append(Box(o.findtext("name", "person").strip(),
                           float(bb.findtext("x")), float(bb.findtext("y")),
                           float(bb.findtext("w")), float(bb.findtext("h")),
                           int(occ.text) if occ is not None and occ.text else 0))
        return out
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            p = line.split()
            out.append(Box(p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                           int(p[5]) if len(p) > 5 else 0))
    return out


def frame_label(boxes: list[Box], min_height: float) -> int:
    if not boxes:
        return 0
    if any(b.cls in ANOMALY_CLASSES and b.h >= min_height for b in boxes):
        return 1
    return -1


@dataclass
class KaistSequence:
    name: str               # 'set00/V000'
    files: list[str]
    frame_idx: np.ndarray   # original frame index (I01234 -> 1234)
    labels: np.ndarray      # int8, {-1, 0, 1}
    boxes: list[list[Box]]  # in original (640x512) coordinates
    src_size: tuple[int, int] = (640, 512)

    @property
    def stride(self) -> int:
        if len(self.frame_idx) < 2:
            return 1
        return int(np.median(np.diff(self.frame_idx)))

    def summary(self) -> str:
        l = self.labels
        return (f"{self.name}: {len(l)} frames | normal={int((l == 0).sum())} "
                f"anomaly={int((l == 1).sum())} ambiguous={int((l == -1).sum())} | stride={self.stride}")


def find_sequences(root: str) -> list[str]:
    """List all 'setXX/VYYY' sequences under root/images."""
    seqs = []
    for d in sorted(glob.glob(os.path.join(root, "images", "set*", "*", "lwir"))):
        seqs.append(Path(os.path.relpath(Path(d).parent, os.path.join(root, "images"))).as_posix())
    return seqs


def load_sequence(root: str, name: str, min_height: float = 30.0) -> KaistSequence:
    img_dir = os.path.join(root, "images", *name.split("/"), "lwir")
    ann_dir = os.path.join(root, "annotations", *name.split("/"))
    files = list_images(img_dir)
    idx, labels, boxes = [], [], []
    for f in files:
        stem = Path(f).stem
        ann = os.path.join(ann_dir, stem + ".xml")
        if not os.path.exists(ann):
            ann = os.path.join(ann_dir, stem + ".txt")
        b = parse_annotation(ann)
        idx.append(frame_index_from_path(f))
        boxes.append(b)
        labels.append(frame_label(b, min_height) if os.path.exists(ann) else -1)
    return KaistSequence(name, files, np.asarray(idx, np.int64), np.asarray(labels, np.int8), boxes)


def boxes_to_mask(boxes: list[Box], src_size: tuple[int, int], out_size: tuple[int, int],
                  min_height: float = 0.0) -> np.ndarray:
    """Converts bounding boxes to a binary ground truth mask (for pixel-level AUROC)."""
    W, H = out_size
    sx, sy = W / src_size[0], H / src_size[1]
    m = np.zeros((H, W), np.uint8)
    for b in boxes:
        if b.cls not in ANOMALY_CLASSES or b.h < min_height:
            continue
        x0, y0 = int(np.floor(b.x * sx)), int(np.floor(b.y * sy))
        x1, y1 = int(np.ceil((b.x + b.w) * sx)), int(np.ceil((b.y + b.h) * sy))
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 1
    return m
