"""
Frame sources — unified interface for reading video files, image sequences, KAIST benchmarks,
and live cameras (USB/RTSP).

16-bit radiometric TIFF/PNG frames are read with ``IMREAD_UNCHANGED`` preserving bit depth
(handled downstream by the AGC preprocessing module).
"""
from __future__ import annotations

import glob
import os
import sys
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
    Unified frame source.

    ``spec`` examples:
      * ``data/kaist/images/set00/V000/lwir``  (folder)
      * ``video.mp4``                          (video file)
      * ``0``                                  (USB camera index)
      * ``rtsp://192.168.1.10/stream``         (IP thermal camera)
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
                raise FileNotFoundError(f"no images found in directory: {s}")
            self._files = files[self.start::max(1, self.stride)]
            if self.max_frames:
                self._files = self._files[: self.max_frames]
            self.fps = 20.0 / max(1, self.stride)  # KAIST benchmark 20 Hz

    @property
    def files(self) -> Optional[list[str]]:
        return self._files

    def __len__(self) -> int:
        if self._files is not None:
            return len(self._files)
        if str(self.spec).isdigit():
            return 0  # Live camera stream
        cap = self._open_capture()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n = max(0, (n - self.start)) // max(1, self.stride)
        return min(n, self.max_frames) if self.max_frames else n

    def _cap_arg(self):
        s = str(self.spec)
        return int(s) if s.isdigit() else s

    def _open_capture(self) -> cv2.VideoCapture:
        arg = self._cap_arg()
        if sys.platform == "win32" and isinstance(arg, int):
            return cv2.VideoCapture(arg, cv2.CAP_DSHOW)
        return cv2.VideoCapture(arg)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yields (frame_index, raw_frame)."""
        if self._files is not None:
            while True:
                for i, f in enumerate(self._files):
                    yield i, read_thermal(f)
                if not self.loop:
                    return
        cap = self._open_capture()
        if not cap.isOpened():
            raise IOError(f"cannot open video source: {self.spec}")
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
    """'I01234.jpg' -> 1234 ; -1 if no digits found."""
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1
