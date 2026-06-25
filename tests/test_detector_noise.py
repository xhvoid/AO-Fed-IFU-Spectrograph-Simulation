"""Tests for the detector noise model."""

import numpy as np
import pytest

from ao_fed_ifu.detector_model import DetectorConfig, apply_poisson_noise, expose


def test_expose_shapes_and_variance_floor():
    signal = np.full((20, 30), 500.0)
    cfg = DetectorConfig(read_noise_e=10.0, dark_current_e_s_pix=0.0, bad_pixel_fraction=0.0,
                         flat_field_sigma=0.0)
    out = expose(signal, cfg, exposure_s=100.0, rng=0)
    assert out.image_e.shape == signal.shape
    assert out.variance_e2.shape == signal.shape
    # Variance >= read noise^2 + signal (shot).
    assert np.all(out.variance_e2 >= cfg.read_noise_e**2)
    assert out.variance_e2.mean() == pytest.approx(500.0 + 100.0, rel=1e-6)  # signal + RN^2


def test_more_read_noise_increases_variance():
    signal = np.full((10, 10), 100.0)
    lo = expose(signal, DetectorConfig(read_noise_e=5.0, bad_pixel_fraction=0.0, flat_field_sigma=0.0),
                100.0, rng=1)
    hi = expose(signal, DetectorConfig(read_noise_e=20.0, bad_pixel_fraction=0.0, flat_field_sigma=0.0),
                100.0, rng=1)
    assert hi.variance_e2.mean() > lo.variance_e2.mean()


def test_poisson_mean_matches_signal():
    rng = np.random.default_rng(0)
    mean = np.full((200, 200), 25.0)
    noisy = apply_poisson_noise(mean, rng)
    # Sample mean of many Poisson draws ~ the true mean.
    assert noisy.mean() == pytest.approx(25.0, rel=0.02)
    # Variance of Poisson ~ mean.
    assert noisy.var() == pytest.approx(25.0, rel=0.1)


def test_bad_pixels_masked_and_infinite_variance():
    signal = np.full((40, 40), 300.0)
    cfg = DetectorConfig(bad_pixel_fraction=0.1, flat_field_sigma=0.0)
    out = expose(signal, cfg, exposure_s=50.0, rng=3)
    assert out.bad_pixel_mask.any()
    # Bad pixels are zeroed in the image and carry infinite variance.
    assert np.all(out.image_e[out.bad_pixel_mask] == 0.0)
    assert np.all(np.isinf(out.variance_e2[out.bad_pixel_mask]))


def test_full_well_saturation():
    signal = np.full((5, 5), 1.0e6)  # way above the well
    cfg = DetectorConfig(full_well_e=5.0e4, read_noise_e=0.0, dark_current_e_s_pix=0.0,
                         bad_pixel_fraction=0.0, flat_field_sigma=0.0)
    out = expose(signal, cfg, exposure_s=1.0, rng=0)
    assert np.all(out.image_e <= cfg.full_well_e)


def test_dark_current_adds_with_exposure():
    signal = np.zeros((10, 10))
    cfg = DetectorConfig(read_noise_e=0.0, dark_current_e_s_pix=1.0, bad_pixel_fraction=0.0,
                         flat_field_sigma=0.0)
    short = expose(signal, cfg, exposure_s=10.0, rng=0).variance_e2.mean()
    long = expose(signal, cfg, exposure_s=100.0, rng=0).variance_e2.mean()
    # Variance is dominated by dark = rate * exposure, so it grows with time.
    assert long > short
    assert long == pytest.approx(100.0, rel=1e-6)
