"""Convolutional LSTM (Shi et al., NeurIPS 2015) — ONNX dışa aktarımına uygun, saf PyTorch."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hid_ch = hid_ch
        # Tek konvolüsyonla 4 kapı (i, f, o, g) — çekirdek füzyonu → daha hızlı çıkarım.
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=kernel_size // 2)
        # GroupNorm: küçük batch'lerde (kenar cihaz, B=1) BatchNorm'dan daha kararlı.
        self.norm = nn.GroupNorm(4, 4 * hid_ch)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():  # unutma kapısı ön yargısı = 1 → erken eğitimde gradyan akışı
            self.gates.bias[hid_ch:2 * hid_ch].fill_(1.0)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        g = self.norm(self.gates(torch.cat([x, h], dim=1)))
        i, f, o, u = torch.chunk(g, 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, x: torch.Tensor):
        B, _, H, W = x.shape
        z = x.new_zeros(B, self.hid_ch, H, W)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """Yığılmış ConvLSTM. Giriş (B, T, C, H, W) → çıkış dizisi (B, T, C_h, H, W) ve son durumlar."""

    def __init__(self, in_ch: int, hidden: list[int], kernel_size: int = 3):
        super().__init__()
        chans = [in_ch] + list(hidden)
        self.cells = nn.ModuleList(ConvLSTMCell(chans[i], chans[i + 1], kernel_size)
                                   for i in range(len(hidden)))

    def forward(self, x: torch.Tensor, states: Optional[list] = None):
        T = x.shape[1]
        if states is None:
            states = [cell.init_state(x[:, 0]) if i == 0 else None for i, cell in enumerate(self.cells)]
        out = x
        new_states = []
        for li, cell in enumerate(self.cells):
            st = states[li]
            seq = []
            for t in range(T):
                inp = out[:, t]
                if st is None:
                    st = cell.init_state(inp)
                st = cell(inp, st)
                seq.append(st[0])
            out = torch.stack(seq, dim=1)
            new_states.append(st)
        return out, new_states
