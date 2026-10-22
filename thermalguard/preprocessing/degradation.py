"""
Saha koşulu simülatörü — termal sensör bozulmaları.

Gerçek sahada (sınır hattı, gece, sis, yağmur, ucuz/sıcak kamera) LWIR görüntüsü
eğitim verisinden çok daha kötü olabilir. Bu modül kontrollü bozulmalar ekleyerek:

1. Eğitimde **artırım (augmentation)** — modelin gürültüye karşı sağlamlığı,
2. Değerlendirmede **dayanıklılık (robustness) testi** — "ön işleme açık / kapalı"
   AUROC karşılaştırması

yapmayı sağlar. Tüm parametreler 8-bit uint8 ölçeğindedir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DegradationConfig:
    contrast: float = 1.0        # <1 → düşük kontrast (sis, termal geçiş saatleri / "thermal crossover")
    brightness: float = 0.0      # ofset (8-bit)
    gauss_sigma: float = 0.0     # zamansal gürültü (NETD) std, 8-bit
    stripe_sigma: float = 0.0    # sütun FPN std
    dead_rate: float = 0.0       # ölü/sıcak piksel oranı
    seed: int | None = None

    @classmethod
    def level(cls, k: int, seed: int | None = None) -> "DegradationConfig":
        """0 (temiz) … 4 (çok ağır) arası hazır seviyeler."""
        presets = [
            dict(),
            dict(contrast=0.6, gauss_sigma=3, stripe_sigma=1.5, dead_rate=1e-4),
            dict(contrast=0.4, gauss_sigma=6, stripe_sigma=3, dead_rate=5e-4),
            dict(contrast=0.25, gauss_sigma=9, stripe_sigma=4.5, dead_rate=1e-3),
            dict(contrast=0.15, gauss_sigma=12, stripe_sigma=6, dead_rate=2e-3),
        ]
        return cls(seed=seed, **presets[int(np.clip(k, 0, len(presets) - 1))])


class ThermalDegrader:
    """Kareye LWIR'e özgü bozulmalar uygular. FPN deseni sekans boyunca sabittir (gerçekçi)."""

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
