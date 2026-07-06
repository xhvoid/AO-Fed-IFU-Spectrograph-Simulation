"""Tests for the covariance-aware SNR and Gaussian line-fit diagnostics."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S
from ao_fed_ifu import diagnostics as diag


# ---------------------------------------------------------------------------
# Covariance-aware uncertainty (task 1).
# ---------------------------------------------------------------------------

def test_covariance_inflation_factor_basic():
    # rho=0 recovers the diagonal budget; N=1 has no off-diagonal terms.
    assert diag.covariance_inflation_factor(0.0, 8) == pytest.approx(1.0)
    assert diag.covariance_inflation_factor(0.9, 1) == pytest.approx(1.0)
    # Positive correlation inflates (>1) and grows with rho; negative deflates.
    f_lo = diag.covariance_inflation_factor(0.3, 8)
    f_hi = diag.covariance_inflation_factor(0.7, 8)
    assert 1.0 < f_lo < f_hi
    assert diag.covariance_inflation_factor(-0.5, 8) < 1.0
    # AR(1) asymptote: f -> (1+rho)/(1-rho) for large N.
    assert diag.covariance_inflation_factor(0.5, 500) == pytest.approx(3.0, rel=0.02)


def _aperture_cube(continuum_1d, wl, lam0, line_sigma=0.01, line_amp=50.0):
    """A (5,5,nl) cube: a 3x3 aperture carrying continuum + a Gaussian line."""
    line = line_amp * np.exp(-0.5 * ((wl - lam0) / line_sigma) ** 2)
    spec = continuum_1d + line
    cube = np.zeros((5, 5, wl.size))
    cube[1:4, 1:4, :] = spec[None, None, :]
    ap = np.zeros((5, 5), dtype=bool)
    ap[1:4, 1:4] = True
    return cube, ap


def test_positive_spectral_correlation_lowers_covariance_aware_snr():
    nl = 80
    wl = np.linspace(2.0, 2.4, nl)
    lam0 = 2.2
    rng = np.random.default_rng(0)
    # Strongly smoothed continuum -> positive lag-1 autocorrelation off the line.
    corr = gaussian_filter1d(rng.normal(0.0, 1.0, nl), sigma=3.0)
    cube, ap = _aperture_cube(100.0 + 5.0 * corr, wl, lam0)
    var = np.ones_like(cube)

    d = diag.aperture_spectrum_uncertainty(cube, var, ap, wl, lam0, window_um=0.03)
    assert d["lag1_autocorr"] > 0.3            # genuinely correlated continuum
    assert d["covariance_inflation_factor"] > 1.0
    # Positive correlation inflates the noise, so covariance-aware SNR is lower.
    assert d["snr_covariance_aware"] < d["snr_diagonal"]
    # The signal (line flux) is unchanged between the two SNRs.
    assert d["noise_covariance_aware"] > d["noise_diagonal"]


def test_white_continuum_keeps_covariance_aware_close_to_diagonal():
    nl = 80
    wl = np.linspace(2.0, 2.4, nl)
    lam0 = 2.2
    rng = np.random.default_rng(1)
    cube, ap = _aperture_cube(100.0 + 5.0 * rng.normal(0.0, 1.0, nl), wl, lam0)
    var = np.ones_like(cube)
    d = diag.aperture_spectrum_uncertainty(cube, var, ap, wl, lam0, window_um=0.03)
    # Uncorrelated continuum -> rho ~ 0 -> inflation ~ 1 -> the two SNRs coincide.
    assert abs(d["covariance_inflation_factor"] - 1.0) < 0.4
    assert d["snr_covariance_aware"] == pytest.approx(d["snr_diagonal"], rel=0.2)


def test_isolated_noise_reference_recovers_white_and_correlated():
    nl = 120
    wl = np.linspace(2.0, 2.4, nl)
    lam0 = 2.2
    rng = np.random.default_rng(2)
    # A noiseless reference reconstruction (smooth continuum + line), and a
    # "measured" cube = reference + noise. The diagnostic must recover the noise
    # correlation from (measured - reference), NOT the continuum shape.
    cont = 100.0 + 20.0 * np.sin(np.linspace(0, 4 * np.pi, nl))   # strongly shaped continuum
    ref_cube, ap = _aperture_cube(cont, wl, lam0)
    var = np.ones_like(ref_cube)

    # White noise -> isolated-noise rho ~ 0 -> inflation ~ 1 (diagonal ~ valid),
    # even though the continuum shape is large.
    white = ref_cube + rng.normal(0.0, 1.0, ref_cube.shape) * (ref_cube > 0)
    dw = diag.aperture_spectrum_uncertainty(white, var, ap, wl, lam0, 0.03, noise_reference_cube=ref_cube)
    assert dw["lag1_autocorr_method"] == "isolated_noise"
    assert abs(dw["lag1_autocorr"]) < 0.2
    assert dw["covariance_inflation_factor"] == pytest.approx(1.0, abs=0.4)

    # Spectrally-correlated noise -> isolated-noise rho > 0 -> inflation > 1.
    corr_noise = gaussian_filter1d(rng.normal(0.0, 1.0, (5, 5, nl)), sigma=2.0, axis=2)
    correlated = ref_cube + corr_noise * (ref_cube > 0)
    dc = diag.aperture_spectrum_uncertainty(correlated, var, ap, wl, lam0, 0.03, noise_reference_cube=ref_cube)
    assert dc["lag1_autocorr"] > 0.3
    assert dc["covariance_inflation_factor"] > 1.0
    assert dc["snr_covariance_aware"] < dc["snr_diagonal"]


def test_bad_pixel_voxels_excluded_from_aperture_uncertainty():
    nl = 40
    wl = np.linspace(2.0, 2.4, nl)
    lam0 = 2.2
    cube, ap = _aperture_cube(np.full(nl, 100.0), wl, lam0)
    var = np.ones_like(cube)
    var[2, 2, :] = np.inf          # one aperture spaxel fully masked
    d = diag.aperture_spectrum_uncertainty(cube, var, ap, wl, lam0, window_um=0.03)
    # A masked spaxel must not send the aperture SNR to zero/NaN.
    assert np.isfinite(d["snr_diagonal"]) and d["snr_diagonal"] > 0


# ---------------------------------------------------------------------------
# Gaussian line fitting (task 2).
# ---------------------------------------------------------------------------

def _line_spectrum(wl, lam0, amp, mu, sigma, c0=10.0, slope=0.0):
    return c0 + slope * (wl - lam0) + amp * np.exp(-0.5 * ((wl - mu) / sigma) ** 2)


def test_gaussian_fit_noiseless_recovers_truth():
    wl = np.linspace(2.0, 2.4, 400)
    lam0 = 2.2
    amp, mu, sigma = 50.0, 2.205, 0.004
    spec = _line_spectrum(wl, lam0, amp, mu, sigma, c0=10.0, slope=3.0)
    f = diag.fit_gaussian_line(spec, wl, lam0, window_um=0.06)
    assert f["success"]
    assert f["line_flux"] == pytest.approx(amp * sigma * np.sqrt(2 * np.pi), rel=1e-4)
    assert f["velocity_kms"] == pytest.approx(SPEED_OF_LIGHT_KM_S * (mu - lam0) / lam0, rel=1e-4)
    assert f["sigma_um"] == pytest.approx(sigma, rel=1e-4)
    assert f["fwhm_um"] == pytest.approx(2.3548 * sigma, rel=1e-3)
    assert f["fit_r2"] > 0.999


def test_gaussian_fit_recovers_continuum_slope_without_biasing_flux():
    wl = np.linspace(2.0, 2.4, 400)
    lam0 = 2.2
    amp, mu, sigma = 40.0, 2.2, 0.005
    flat = diag.fit_gaussian_line(_line_spectrum(wl, lam0, amp, mu, sigma, slope=0.0), wl, lam0, window_um=0.06)
    sloped = diag.fit_gaussian_line(_line_spectrum(wl, lam0, amp, mu, sigma, slope=8.0), wl, lam0, window_um=0.06)
    # A strong continuum slope must not change the recovered line flux.
    assert sloped["line_flux"] == pytest.approx(flat["line_flux"], rel=1e-3)
    assert sloped["success"] and flat["success"]


def test_gaussian_fit_noisy_within_tolerance():
    wl = np.linspace(2.0, 2.4, 400)
    lam0 = 2.2
    amp, mu, sigma = 50.0, 2.205, 0.004
    truth_flux = amp * sigma * np.sqrt(2 * np.pi)
    rng = np.random.default_rng(0)
    var = np.full_like(wl, 4.0)
    hits = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noisy = _line_spectrum(wl, lam0, amp, mu, sigma) + rng.normal(0.0, 2.0, wl.size)
        f = diag.fit_gaussian_line(noisy, wl, lam0, window_um=0.06, variance_1d=var)
        assert f["success"]
        # Fitted flux within 3 formal sigma of truth, reduced chi2 order unity.
        if abs(f["line_flux"] - truth_flux) <= 3.0 * f["line_flux_err"]:
            hits += 1
        assert 0.3 < f["reduced_chi2"] < 3.0
    assert hits >= 8   # ~3sigma coverage across seeds


def test_gaussian_fit_fails_on_pure_noise():
    wl = np.linspace(2.0, 2.4, 400)
    lam0 = 2.2
    rng = np.random.default_rng(3)
    var = np.full_like(wl, 4.0)
    noise_only = 10.0 + rng.normal(0.0, 2.0, wl.size)
    f = diag.fit_gaussian_line(noise_only, wl, lam0, window_um=0.06, variance_1d=var, min_snr=3.0)
    assert not f["success"]


def test_gaussian_fit_too_short_region_returns_failure():
    wl = np.linspace(2.0, 2.4, 400)
    f = diag.fit_gaussian_line(np.ones_like(wl), wl, 2.2, window_um=1e-5)
    assert not f["success"]
    assert np.isnan(f["line_flux"])


def test_gaussian_fit_map_flags_lit_spaxels():
    wl = np.linspace(2.0, 2.4, 120)
    lam0 = 2.2
    cube = np.zeros((4, 4, wl.size))
    # Two spaxels carry a real line; the rest are flat continuum only.
    for iy, ix in [(1, 1), (2, 2)]:
        cube[iy, ix, :] = _line_spectrum(wl, lam0, 60.0, lam0, 0.005)
    cube[0, 0, :] = 10.0
    maps = diag.fit_gaussian_line_map(cube, wl, lam0, window_um=0.06, min_snr=3.0)
    assert maps["success"].shape == (4, 4)
    assert maps["success"][1, 1] and maps["success"][2, 2]
    assert not maps["success"][0, 0]
    assert np.isfinite(maps["line_flux"][1, 1])
    assert np.isnan(maps["line_flux"][0, 0])
