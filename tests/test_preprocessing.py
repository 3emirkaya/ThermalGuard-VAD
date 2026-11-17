import numpy as np
import pytest

from thermalguard.preprocessing.degradation import DegradationConfig, ThermalDegrader
from thermalguard.preprocessing.thermal import (
    ThermalPreprocessConfig,
    ThermalPreprocessor,
    apply_clahe,
    correct_dead_pixels,
    entropy,
    estimate_noise_sigma,
    estimate_stripe_offsets,
    remove_stripes,
    rms_contrast,
)


def test_thermal_preprocessor_shape_and_range():
    # 640x512 random thermal frame
    frame = np.random.randint(50, 180, (512, 640), dtype=np.uint8)
    cfg = ThermalPreprocessConfig(output_size=(160, 128), clahe=True, denoise="bilateral")
    pp = ThermalPreprocessor(cfg)

    out = pp(frame)
    assert out.shape == (128, 160), f"Expected shape (128, 160), got: {out.shape}"
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_16bit_radiometric_support():
    # 14/16-bit raw radiometric camera output (e.g. 8000 - 12000 digital levels)
    raw_16bit = np.random.randint(8000, 12000, (512, 640), dtype=np.uint16)
    pp = ThermalPreprocessor(ThermalPreprocessConfig(output_size=(160, 128)))
    out = pp(raw_16bit)
    assert out.shape == (128, 160)
    assert out.dtype == np.float32
    assert out.max() > 0.5  # AGC must scale dynamically to [0, 1]


def test_clahe_contrast_enhancement():
    # Very low contrast thermal fog scene
    low_contrast = np.full((128, 160), 100, dtype=np.uint8)
    low_contrast[40:80, 50:110] = 104  # Only 4 counts temperature difference

    c_before = rms_contrast(low_contrast)
    clahe_out = apply_clahe(low_contrast, clip=3.0, grid=(8, 8))
    c_after = rms_contrast(clahe_out)

    assert c_after > c_before, "CLAHE must improve local contrast!"


def test_dead_pixel_correction():
    img = np.full((64, 64), 100, dtype=np.uint8)
    # Inject 3 dead/hot defective pixels
    img[10, 10] = 255  # Hot
    img[20, 20] = 0    # Dead
    img[30, 30] = 250  # Hot

    corrected = correct_dead_pixels(img, thresh=40.0)
    assert corrected[10, 10] == 100
    assert corrected[20, 20] == 100
    assert corrected[30, 30] == 100


def test_destripe_fpn():
    img = np.full((64, 64), 120, dtype=np.uint8)
    # Add column-based FPN stripe artifacts
    img[:, 15] = 180
    img[:, 35] = 70

    col_off, row_off = estimate_stripe_offsets(img, sigma=8.0)
    destriped = remove_stripes(img, col_off, row_off)

    # Defective columns must be restored close to background
    assert abs(int(destriped[0, 15]) - 120) < 15
    assert abs(int(destriped[0, 35]) - 120) < 15


def test_degrader_simulator():
    img = np.full((128, 160), 120, dtype=np.uint8)
    deg = ThermalDegrader(DegradationConfig.level(2, seed=42))
    degraded = deg(img)

    assert degraded.shape == img.shape
    assert estimate_noise_sigma(degraded) > estimate_noise_sigma(img)
