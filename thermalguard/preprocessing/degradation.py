"""
Field condition simulator — thermal sensor degradation models.

In real-world deployment (border perimeter, nocturnal, fog, rain, low-cost uncooled microbolometers),
LWIR imagery can be significantly worse than pristine training datasets. This module injects
controlled degradations to support:

1. Training **augmentation** — enhances model robustness against sensor noise.
2. Evaluation **stress testing** — enables rigorous "preprocessing ON vs. OFF"
   AUROC benchmarking under severe atmospheric and hardware degradation.

All parameters operate in the standard 8-bit [0, 255] domain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DegradationConfig:
    contrast: float = 1.0        # < 1 -> low contrast (fog, thermal crossover hours)
    brightness: float = 0.0      # offset (8-bit)
    gauss_sigma: float = 0.0     # temporal noise (NETD) std, 8-bit
    stripe_sigma: float = 0.0    # column fixed pattern noise (FPN) std
    dead_rate: float = 0.0       # dead/hot pixel defect ratio
    seed: int | None = None

    @classmethod
    def level(cls, k: int, seed: int | None = None) -> "DegradationConfig":
        """Predefined severity levels from 0 (clean) to 4 (severe)."""
        presets = [
            dict(),
            dict(contrast=0.6, gauss_sigma=3, stripe_sigma=1.5, dead_rate=1e-4),
            dict(contrast=0.4, gauss_sigma=6, stripe_sigma=3, dead_rate=5e-4),
            dict(contrast=0.25, gauss_sigma=9, stripe_sigma=4.5, dead_rate=1e-3),
            dict(contrast=0.15, gauss_sigma=12, stripe_sigma=6, dead_rate=2e-3),
        ]
        return cls(seed=seed, **presets[int(np.clip(k, 0, len(presets) - 1))])


class ThermalDegrader:
    """Applies LWIR sensor degradation to frames. FPN stripe patterns remain fixed across a sequence."""

    def __init__(self, cfg: DegradationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._stripes: dict[int, np.ndarray] = {}
        self._dead: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg
        f = img.astype(np.float32)
        if c.contrast != 1.0 or c.brightness:
            m = f.mean()
            f = (f - m) * c.contrast + m + c.brightness
        if c.stripe_sigma > 0:
            w = f.shape[1]
            if w not in self._stripes:
                self._stripes[w] = self.rng.normal(0, c.stripe_sigma, w).astype(np.float32)
            f += self._stripes[w][None, :]
        if c.gauss_sigma > 0:
            f += self.rng.normal(0, c.gauss_sigma, f.shape).astype(np.float32)
        out = np.clip(f, 0, 255).astype(np.uint8)
        if c.dead_rate > 0:
            key = out.shape
            if key not in self._dead:
                n = int(c.dead_rate * out.size)
                idx = self.rng.choice(out.size, n, replace=False)
                val = self.rng.choice(np.array([0, 255], np.uint8), n)
                self._dead[key] = (idx, val)
            idx, val = self._dead[key]
            out.reshape(-1)[idx] = val
        return out
