"""
Kare kaynakları — tek bir arayüzle video dosyası, görüntü klasörü, KAIST sekansı,
canlı kamera (USB/RTSP) okuma.

16-bit radyometrik TIFF/PNG kareler ``IMREAD_UNCHANGED`` ile bit derinliği
korunarak okunur (AGC ön işleme modülünde yapılır).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_thermal(path: str | os.PathLike) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[..., 0]
    return img


def list_images(folder: str | os.PathLike) -> list[str]:
    files = [f for f in glob.glob(os.path.join(str(folder), "*")) if f.lower().endswith(IMG_EXT)]
    return sorted(files)


@dataclass
class FrameSource:
    """
    Birleşik kare kaynağı.

    ``spec`` örnekleri:
      * ``data/kaist/images/set00/V000/lwir``  (klasör)
      * ``video.mp4``                          (video dosyası)
      * ``0``                                   (USB kamera indeksi)
      * ``rtsp://192.168.1.10/stream``          (IP termal kamera)
    """

    spec: str
    stride: int = 1
    start: int = 0
    max_frames: int = 0
    loop: bool = False
    fps: float = field(default=20.0, init=False)

    def __post_init__(self):
        self._files: Optional[list[str]] = None
        s = str(self.spec)
        if os.path.isdir(s):
            files = list_images(s)
            if not files and os.path.isdir(os.path.join(s, "lwir")):
                files = list_images(os.path.join(s, "lwir"))
            if not files:
                raise FileNotFoundError(f"klasörde görüntü yok: {s}")
            self._files = files[self.start::max(1, self.stride)]
            if self.max_frames:
                self._files = self._files[: self.max_frames]
            self.fps = 20.0 / max(1, self.stride)  # KAIST 20 Hz

    @property
    def files(self) -> Optional[list[str]]:
        return self._files

    def __len__(self) -> int:
        if self._files is not None:
            return len(self._files)
        cap = cv2.VideoCapture(self._cap_arg())
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n = max(0, (n - self.start)) // max(1, self.stride)
        return min(n, self.max_frames) if self.max_frames else n

    def _cap_arg(self):
        s = str(self.spec)
        return int(s) if s.isdigit() else s

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """(kare_indeksi, ham_kare) üretir."""
        if self._files is not None:
            while True:
                for i, f in enumerate(self._files):
                    yield i, read_thermal(f)
                if not self.loop:
                    return
        cap = cv2.VideoCapture(self._cap_arg())
        if not cap.isOpened():
            raise IOError(f"kaynak açılamadı: {self.spec}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.fps = fps / max(1, self.stride)
        i = n = 0
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    if self.loop and not str(self.spec).isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        i = 0
                        continue
                    break
                if i >= self.start and (i - self.start) % max(1, self.stride) == 0:
                    if fr.ndim == 3:
                        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    yield n, fr
                    n += 1
                    if self.max_frames and n >= self.max_frames:
                        break
                i += 1
        finally:
            cap.release()


def frame_index_from_path(path: str) -> int:
    """'I01234.jpg' → 1234 ; sayı bulunamazsa -1."""
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1
