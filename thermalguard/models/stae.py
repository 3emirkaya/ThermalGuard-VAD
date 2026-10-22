"""
Spatio-Temporal Autoencoder (STAE) — ConvLSTM darboğazlı.

Mimari (varsayılan, giriş 1×128×160, T=8):

    kare_t ─► [2D CNN Encoder, zaman boyunca ağırlık paylaşımlı]  (1/16 çözünürlük, 8×10)
                │
                ▼
          ConvLSTM ×2  (zamansal dinamik: hareket deseni, ısı değişimi)
                │   h_t
                ▼
          [2D CNN Decoder]  ─►  x̂_t   (yeniden üretilen kare)

* **Atlamalı bağlantı (skip) YOK** — aksi halde ağ anomaliyi de "kopyalar" ve
  yeniden üretim hatası anlamını yitirir. Darboğaz (64×8×10 = 5120) girişin
  (20480) %25'i kadardır.
* İki hedef desteklenir:
    - ``reconstruct``: x_{1..T} → x̂_{1..T}  (nedensel; x̂_t yalnızca ≤t karelere bakar)
    - ``predict``    : x_{1..T} → x̂_{T+1}   (gelecek kare tahmini)
* **Akış (streaming) optimizasyonu**: encoder kare bazlı olduğu için canlı
  çıkarımda her yeni karede yalnızca 1 encoder geçişi yapılır; son T karenin
  öznitelikleri önbellekte tutulur (bkz. ``encode_frame`` / ``forward_from_features``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTM


def _conv(cin, cout, k=3, s=1, norm=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _down(cin, cout):
    return nn.Sequential(_conv(cin, cout, 3, 2), _conv(cout, cout, 3, 1))


def _up(cin, cout):
    # Upsample + conv: ConvTranspose'a göre damalı (checkerboard) artefakt üretmez.
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         _conv(cin, cout, 3, 1), _conv(cout, cout, 3, 1))


class FrameEncoder(nn.Module):
    def __init__(self, in_ch=1, widths=(16, 32, 64, 96), latent=64):
        super().__init__()
        c0 = widths[0]
        self.stem = _conv(in_ch, c0, 3, 1)
        blocks, c = [], c0
        for w in widths:
            blocks.append(_down(c, w))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(c, latent, 1)

    def forward(self, x):
        return self.proj(self.blocks(self.stem(x)))


class FrameDecoder(nn.Module):
    def __init__(self, out_ch=1, widths=(96, 64, 32, 16), latent=64):
        super().__init__()
        self.inp = _conv(latent, widths[0], 3, 1)
        blocks, c = [], widths[0]
        for w in widths:
            blocks.append(_up(c, w))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(c, out_ch, 3, 1, 1)

    def forward(self, z):
        return torch.sigmoid(self.head(self.blocks(self.inp(z))))


class STAE(nn.Module):
    def __init__(self, in_ch: int = 1, widths=(16, 32, 64, 96), latent: int = 64,
                 lstm_hidden=(64, 64), objective: str = "reconstruct"):
        super().__init__()
        assert objective in ("reconstruct", "predict")
        self.objective = objective
        self.encoder = FrameEncoder(in_ch, widths, latent)
        self.temporal = ConvLSTM(latent, list(lstm_hidden), kernel_size=3)
        self.decoder = FrameDecoder(in_ch, tuple(reversed(widths)), lstm_hidden[-1])
        self.downsample = 2 ** len(widths)

    # ------------------------------------------------------------------ #
    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, T, latent, h, w)"""
        B, T = x.shape[:2]
        z = self.encoder(x.flatten(0, 1))
        return z.view(B, T, *z.shape[1:])

    def forward_from_features(self, z: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """ConvLSTM + decoder. ``last_only`` → yalnızca son zaman adımını çöz (çıkarım)."""
        h_seq, _ = self.temporal(z)
        if last_only or self.objective == "predict":
            return self.decoder(h_seq[:, -1]).unsqueeze(1)          # (B, 1, C, H, W)
        B, T = h_seq.shape[:2]
        y = self.decoder(h_seq.flatten(0, 1))
        return y.view(B, T, *y.shape[1:])                           # (B, T, C, H, W)

    def forward(self, x: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        return self.forward_from_features(self.encode_frames(x), last_only=last_only)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Tek kare (B, C, H, W) → latent (B, latent, h, w) — akış önbelleği için."""
        return self.encoder(frame)


class STAEExportWrapper(nn.Module):
    """ONNX/TensorRT dışa aktarımı için: (B, T, 1, H, W) → son karenin yeniden üretimi (B, 1, H, W)."""

    def __init__(self, model: STAE):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, last_only=True)[:, 0]


def build_model(cfg: dict) -> STAE:
    m = dict(cfg or {})
    return STAE(in_ch=1,
                widths=tuple(m.get("widths", (16, 32, 64, 96))),
                latent=int(m.get("latent", 64)),
                lstm_hidden=tuple(m.get("lstm_hidden", (64, 64))),
                objective=m.get("objective", "reconstruct"))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
#  Kayıp fonksiyonları
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int, sigma: float, device, dtype):
    g = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x: torch.Tensor, y: torch.Tensor, size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """(N, 1, H, W) ∈ [0,1] → SSIM haritası (N, 1, H, W)."""
    w = _gaussian_window(size, sigma, x.device, x.dtype)
    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad)
    mu_y = F.conv2d(y, w, padding=pad)
    sxx = F.conv2d(x * x, w, padding=pad) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


class ReconstructionLoss(nn.Module):
    """L = MSE + λ_ssim · (1 − SSIM). SSIM terimi bulanık (ortalama) çıktıları cezalandırır."""

    def __init__(self, ssim_weight: float = 0.2):
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        p = pred.flatten(0, 1).float()
        t = target.flatten(0, 1).float()
        mse = F.mse_loss(p, t)
        out = {"mse": mse}
        loss = mse
        if self.ssim_weight > 0:
            s = 1 - ssim_map(p, t).mean()
            out["ssim"] = s
            loss = loss + self.ssim_weight * s
        out["loss"] = loss
        return out
