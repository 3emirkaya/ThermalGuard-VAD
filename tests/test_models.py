import numpy as np
import pytest
import torch

from thermalguard.models.convlstm import ConvLSTM, ConvLSTMCell
from thermalguard.models.stae import (
    ReconstructionLoss,
    STAE,
    STAEExportWrapper,
    build_model,
    count_parameters,
    ssim_map,
)


def test_convlstm_cell_and_stack():
    cell = ConvLSTMCell(in_ch=16, hid_ch=32, kernel_size=3)
    x = torch.randn(2, 16, 16, 20)
    st = cell.init_state(x)
    h, c = cell(x, st)

    assert h.shape == (2, 32, 16, 20)
    assert c.shape == (2, 32, 16, 20)

    # 2 katmanlı ConvLSTM
    lstm = ConvLSTM(in_ch=16, hidden=[32, 32], kernel_size=3)
    seq_in = torch.randn(2, 6, 16, 16, 20)  # (B, T, C, H, W)
    out_seq, states = lstm(seq_in)
    assert out_seq.shape == (2, 6, 32, 16, 20)
    assert len(states) == 2


def test_stae_forward_and_reconstruct():
    cfg = {
        "widths": (16, 32, 64),
        "latent": 32,
        "lstm_hidden": (32,),
        "objective": "reconstruct",
    }
    model = build_model(cfg)
    model.eval()

    # (B=2, T=4, C=1, H=64, W=64)
    x = torch.rand(2, 4, 1, 64, 64)

    # 1. Tam dizi rekonstrüksiyonu
    with torch.no_grad():
        y = model(x, last_only=False)
    assert y.shape == (2, 4, 1, 64, 64)
    assert 0.0 <= y.min() and y.max() <= 1.0

    # 2. Yalnızca son kare çıkarımı (kenar hızlandırma)
    with torch.no_grad():
        y_last = model(x, last_only=True)
    assert y_last.shape == (2, 1, 1, 64, 64)


def test_reconstruction_loss():
    crit = ReconstructionLoss(ssim_weight=0.2)
    pred = torch.rand(2, 4, 1, 32, 32)
    target = pred.clone()

    loss_dict = crit(pred, target)
    assert loss_dict["mse"].item() < 1e-6
    assert loss_dict["loss"].item() < 0.05

    # Farklı görüntülerde kayıp pozitif olmalı
    noisy = target + 0.2 * torch.randn_like(target)
    loss_noisy = crit(noisy, target)["loss"]
    assert loss_noisy.item() > loss_dict["loss"].item()


def test_export_wrapper():
    model = build_model({"widths": (16, 32), "latent": 16, "lstm_hidden": (16,)})
    wrapper = STAEExportWrapper(model)
    dummy = torch.randn(1, 4, 1, 32, 32)
    out = wrapper(dummy)
    assert out.shape == (1, 1, 32, 32)
