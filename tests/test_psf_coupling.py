"""Tests for PSF proxies and spaxel encircled-energy coupling."""

import numpy as np
import pytest

from ao_fed_ifu import psf_coupling as psf


def test_psfs_are_normalised():
    airy = psf.airy_psf_proxy(2.2, 8.0, grid_mas=10.0, n_pix=99)
    seeing = psf.seeing_gaussian_psf(500.0, grid_mas=20.0, n_pix=99)
    ao = psf.ao_corrected_psf_proxy(0.4, 2.2, 8.0, 600.0, grid_mas=10.0, n_pix=99)
    for p in (airy, seeing, ao):
        assert np.sum(p) == pytest.approx(1.0, rel=1e-6)
        assert np.all(p >= 0)


def test_airy_fwhm_matches_diffraction_scale():
    lam, diam = 2.2, 8.0
    # Sample finely so the FWHM estimate is accurate.
    p = psf.airy_psf_proxy(lam, diam, grid_mas=2.0, n_pix=201)
    fwhm = psf.psf_fwhm_mas(p, 2.0)
    expected = psf.diffraction_fwhm_mas(lam, diam)
    assert fwhm == pytest.approx(expected, rel=0.15)


def test_higher_strehl_increases_encircled_energy():
    common = dict(lambda_um=2.2, telescope_diameter_m=8.0, seeing_fwhm_mas=600.0,
                  grid_mas=4.0, n_pix=161)
    lo = psf.ao_corrected_psf_proxy(strehl=0.05, **common)
    hi = psf.ao_corrected_psf_proxy(strehl=0.7, **common)
    ee1_lo, ee3_lo, _ = psf.encircled_energy_per_spaxel(lo, 4.0, 20.0)
    ee1_hi, ee3_hi, _ = psf.encircled_energy_per_spaxel(hi, 4.0, 20.0)
    assert ee1_hi > ee1_lo
    assert ee3_hi > ee3_lo
    for ee in (ee1_lo, ee3_lo, ee1_hi, ee3_hi):
        assert 0.0 <= ee <= 1.0


def test_convolution_conserves_plane_flux():
    from functools import partial

    cube = np.zeros((11, 11, 3))
    cube[5, 5, :] = 100.0  # a point source in every plane
    wl = np.array([2.0, 2.2, 2.4])
    builder = partial(psf.ao_corrected_psf_proxy, strehl=0.3,
                      telescope_diameter_m=8.0, seeing_fwhm_mas=400.0)
    out = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=20.0, kernel_n_pix=11)
    for il in range(3):
        assert out[:, :, il].sum() == pytest.approx(cube[:, :, il].sum(), rel=1e-3)


def test_metrics_flags_undersampling():
    # A narrow diffraction PSF sampled by a coarse spaxel -> undersampled.
    p = psf.airy_psf_proxy(1.0, 39.0, grid_mas=1.0, n_pix=201)  # tiny core (ELT)
    metrics = psf.psf_metrics(p, fine_scale_mas=1.0, spaxel_scale_mas=50.0)
    assert metrics.undersampled is True
    # A well-sampled case is not flagged.
    p2 = psf.seeing_gaussian_psf(800.0, grid_mas=10.0, n_pix=201)
    metrics2 = psf.psf_metrics(p2, fine_scale_mas=10.0, spaxel_scale_mas=50.0)
    assert metrics2.undersampled is False


def test_centroid_shift_small_for_centred_psf():
    p = psf.seeing_gaussian_psf(300.0, grid_mas=10.0, n_pix=99)
    m = psf.psf_metrics(p, 10.0, 30.0)
    assert m.centroid_shift_mas < 1.0  # symmetric PSF is centred
