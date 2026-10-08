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
    # Positive correlation inflates (>1) and grows with rho.
    f_lo = diag.covariance_inflation_factor(0.3, 8)
    f_hi = diag.covariance_inflation_factor(0.7, 8)
    assert 1.0 < f_lo < f_hi
    # Negative (anti-)correlation is CLAMPED: an "inflation" factor must never
    # credit a noise reduction, so it stays exactly at the diagonal budget.
    assert diag.covariance_inflation_factor(-0.5, 8) == pytest.approx(1.0)
    assert diag.covariance_inflation_factor(-0.9, 64) >= 1.0
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
    assert 1.0 <= d["covariance_inflation_factor"] < 1.4
    assert d["snr_covariance_aware"] == pytest.approx(d["snr_diagonal"], rel=0.2)
    # Invariant of a conservative budget: never above the diagonal SNR.
    assert d["snr_covariance_aware"] <= d["snr_diagonal"]


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
    assert dw["lag1_autocorr_method"] == "reference_residual"
    assert abs(dw["lag1_autocorr"]) < 0.2
    assert 1.0 <= dw["covariance_inflation_factor"] < 1.4

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


def test_masked_spaxel_flux_does_not_enter_signal():
    """A bad spaxel (non-finite variance) must be excluded from the FLUX sum
    too, not only from the variance sum -- otherwise a hot spaxel with huge
    line flux and zero contributed noise inflates the aperture SNR."""
    nl = 40
    wl = np.linspace(2.0, 2.4, nl)
    lam0 = 2.2
    cube, ap = _aperture_cube(np.full(nl, 100.0), wl, lam0)
    var = np.ones_like(cube)

    hot = cube.copy()
    hot[2, 2, :] *= 100.0          # a hot spaxel with a huge (bogus) line flux...
    hot_var = var.copy()
    hot_var[2, 2, :] = np.inf      # ...flagged bad via non-finite variance
    d = diag.aperture_spectrum_uncertainty(hot, hot_var, ap, wl, lam0, window_um=0.03)

    # Reference: the same clean aperture with that spaxel excluded outright.
    ap_ref = ap.copy()
    ap_ref[2, 2] = False
    ref = diag.aperture_spectrum_uncertainty(cube, var, ap_ref, wl, lam0, window_um=0.03)

    # Flux, noise and SNR must all match the excluded-spaxel reference: none of
    # the masked spaxel's flux may leak into the line flux or the SNR.
    assert d["line_flux"] == pytest.approx(ref["line_flux"])
    assert d["noise_diagonal"] == pytest.approx(ref["noise_diagonal"])
    assert d["snr_diagonal"] == pytest.approx(ref["snr_diagonal"])
    assert d["snr_covariance_aware"] == pytest.approx(ref["snr_covariance_aware"])


def test_masked_spaxel_excluded_from_isolated_noise_spectrum():
    """With a noise_reference_cube, the isolated-noise spectrum must use the
    same valid-voxel mask as the flux/variance sums."""
    nl = 120
    wl = np.linspace(2.0, 2.4, nl)
    lam0 = 2.2
    rng = np.random.default_rng(7)
    ref_cube, ap = _aperture_cube(np.full(nl, 100.0), wl, lam0)
    var = np.ones_like(ref_cube)

    measured = ref_cube + rng.normal(0.0, 1.0, ref_cube.shape) * (ref_cube > 0)
    measured[2, 2, :] += 1e6       # a wild bad spaxel...
    var[2, 2, :] = np.inf          # ...flagged via non-finite variance
    d = diag.aperture_spectrum_uncertainty(
        measured, var, ap, wl, lam0, 0.03, noise_reference_cube=ref_cube)
    # The bogus spaxel would dominate the noise spectrum if it leaked in; with
    # it excluded the noise stays white (rho ~ 0) and the SNR stays finite.
    assert abs(d["lag1_autocorr"]) < 0.3
    assert np.isfinite(d["snr_diagonal"]) and d["snr_diagonal"] > 0
    assert d["snr_covariance_aware"] <= d["snr_diagonal"]


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
    # Noiseless spectra with a known per-channel variance: an exact fit with no
    # supplied variance has zero residual-scaled covariance, so its SNR (and
    # hence detection) is undefined -- numerical recovery and detection are
    # separate contracts.
    var = np.full_like(wl, 1.0)
    flat = diag.fit_gaussian_line(_line_spectrum(wl, lam0, amp, mu, sigma, slope=0.0), wl, lam0,
                                  window_um=0.06, variance_1d=var)
    sloped = diag.fit_gaussian_line(_line_spectrum(wl, lam0, amp, mu, sigma, slope=8.0), wl, lam0,
                                    window_um=0.06, variance_1d=var)
    # A strong continuum slope must not change the recovered line flux.
    assert sloped["line_flux"] == pytest.approx(flat["line_flux"], rel=1e-3)
    assert sloped["success"] and flat["success"]
    # The fitted slope c1 is returned, so callers can render the FULL model.
    assert sloped["continuum_slope"] == pytest.approx(8.0, rel=1e-3)
    assert flat["continuum_slope"] == pytest.approx(0.0, abs=1e-3)


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
        assert f["fit_converged"] and f["fit_detected"]
        # Fitted flux within 3 formal sigma of truth, reduced chi2 order unity.
        if abs(f["line_flux"] - truth_flux) <= 3.0 * f["line_flux_err"]:
            hits += 1
        assert 0.3 < f["reduced_chi2"] < 3.0
        # A correct error model passes the quality gate...
        assert f["fit_quality_ok"]
        # ...and the chi2 scaling never credits chi2 < 1 (scale >= 1).
        assert f["error_scale"] >= 1.0
        assert f["snr_fit_scaled"] <= f["snr_fit"]
    assert hits >= 8   # ~3sigma coverage across seeds


def test_gaussian_fit_quality_flag_catches_underestimated_variance():
    """A fit can converge and detect the line while its error model is wrong:
    variance 25x too small gives reduced chi2 ~ 25, which must clear neither
    the quality gate nor leave the formal SNR unqualified -- the scaled SNR
    deflates it by sqrt(reduced chi2)."""
    wl = np.linspace(2.0, 2.4, 400)
    lam0 = 2.2
    rng = np.random.default_rng(5)
    spec = _line_spectrum(wl, lam0, 50.0, 2.205, 0.004) + rng.normal(0.0, 2.0, wl.size)
    var_too_small = np.full_like(wl, 4.0 / 25.0)
    f = diag.fit_gaussian_line(spec, wl, lam0, window_um=0.06, variance_1d=var_too_small)
    assert f["fit_converged"] and f["fit_detected"] and f["success"]
    assert f["reduced_chi2"] > diag.FIT_QUALITY_CHI2_RANGE[1]
    assert not f["fit_quality_ok"]
    assert f["error_scale"] == pytest.approx(np.sqrt(f["reduced_chi2"]))
    assert f["snr_fit_scaled"] == pytest.approx(f["snr_fit"] / f["error_scale"])
    assert f["line_flux_err_scaled"] == pytest.approx(f["line_flux_err"] * f["error_scale"])
    assert f["snr_fit_scaled"] < f["snr_fit"]


def test_gaussian_fit_fails_on_pure_noise():
    wl = np.linspace(2.0, 2.4, 400)
    lam0 = 2.2
    rng = np.random.default_rng(3)
    var = np.full_like(wl, 4.0)
    noise_only = 10.0 + rng.normal(0.0, 2.0, wl.size)
    f = diag.fit_gaussian_line(noise_only, wl, lam0, window_um=0.06, variance_1d=var, min_snr=3.0)
    assert not f["success"]
    assert not f["fit_detected"]


def test_gaussian_fit_too_short_region_returns_failure():
    wl = np.linspace(2.0, 2.4, 400)
    f = diag.fit_gaussian_line(np.ones_like(wl), wl, 2.2, window_um=1e-5)
    assert not f["success"]
    assert not f["fit_converged"] and not f["fit_detected"] and not f["fit_quality_ok"]
    assert np.isnan(f["line_flux"])
    assert np.isnan(f["continuum_slope"])


def test_gaussian_fit_masks_invalid_variance_without_dropping_weights():
    wl = np.linspace(2.1, 2.3, 301)
    spec = _line_spectrum(wl, 2.2, 60, 2.201, 0.004)
    var = np.linspace(1, 9, wl.size)
    bad = np.array([0, 146, 153, 163])
    var[bad] = [np.nan, np.inf, 0, -1]
    spec[bad] = 1e8  # excluded channels must not influence any fit parameter
    valid = np.isfinite(var) & (var > 0)
    actual = diag.fit_gaussian_line(spec, wl, 2.2, variance_1d=var)
    expected = diag.fit_gaussian_line(spec[valid], wl[valid], 2.2, variance_1d=var[valid])
    assert actual["success"]
    for key in ("line_flux", "line_flux_err", "velocity_kms", "reduced_chi2"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-10, abs=1e-10)
    assert not diag.fit_gaussian_line(spec, wl, 2.2, variance_1d=np.full_like(wl, np.inf))["success"]


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


def test_aperture_uncertainty_uses_exact_spatial_covariance_when_model_given():
    """With an extraction model the covariance-aware budget propagates the
    joint extraction's spatial covariance exactly (checked against Monte
    Carlo); without it the independent-spaxel sum is used and labelled."""
    from ao_fed_ifu import cube_reconstruction as recon
    from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
    from ao_fed_ifu.spectrograph_model import SpectrographConfig

    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.1, resolving_power_R=2000.0)
    ifu = IFUConfig(nx_spaxel=6, ny_spaxel=6, cross_talk_sigma_pix=0.6)
    wl = np.linspace(2.0, 2.1, spectro.n_detector_pixels)
    mapping = build_slicer_mapping(ifu, spectro, wl)
    lam0 = 2.05
    cube = np.full((6, 6, wl.size), 30.0)
    cube[2:4, 2:4, :] += 400.0 * np.exp(-0.5 * ((wl[None, None, :] - lam0) / 0.002) ** 2)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    sigma = 6.0
    var = np.full_like(frame, sigma**2)
    truth, fvar, model = recon.reconstruct_cube_joint_with_model(frame, var, mapping)
    ap = np.zeros((6, 6), dtype=bool)
    ap[1:5, 1:5] = True

    with_model = diag.aperture_spectrum_uncertainty(
        truth, fvar, ap, wl, lam0, window_um=0.01, noise_reference_cube=truth, extraction_model=model
    )
    without = diag.aperture_spectrum_uncertainty(
        truth, fvar, ap, wl, lam0, window_um=0.01, noise_reference_cube=truth
    )
    assert with_model["spatial_covariance_method"] == "exact_extraction_model"
    assert without["spatial_covariance_method"] == "independent_spaxels"
    assert without["spatial_covariance_factor"] == pytest.approx(1.0)
    assert with_model["snr_diagonal"] == pytest.approx(without["snr_diagonal"])
    # deblended neighbours are anti-correlated: the exact aperture noise is
    # BELOW the independent-spaxel sum here
    assert with_model["spatial_covariance_factor"] < 0.8
    # ... and it is the right number: Monte Carlo of the windowed line flux
    rng = np.random.default_rng(0)
    in_win = np.abs(wl - lam0) <= 0.01
    fluxes = []
    for _ in range(300):
        noisy = recon.reconstruct_cube_weighted(frame + rng.normal(0.0, sigma, frame.shape), var, mapping)
        aperture_spec = noisy[ap, :].sum(axis=0)
        fluxes.append(aperture_spec[in_win].sum() - in_win.sum() * np.median(aperture_spec[~in_win]))
    mc_noise = float(np.std(fluxes, ddof=1))
    assert with_model["noise_covariance_aware"] == pytest.approx(mc_noise, rel=0.12)
    # ... while the independent-spaxel sum overstates the measured noise.
    assert without["noise_covariance_aware"] > 1.2 * mc_noise


def test_line_window_maps_distinguish_full_and_partial_with_coverage():
    wl = np.arange(11.0)
    cube = np.full((1, 2, 11), 10.0)
    cube[:, :, 4:7] += [2.0, 10.0, 2.0]
    var = np.ones_like(cube)
    cube[0, 1, 5] = 1e9   # an invalid bright pixel must enter neither sum
    var[0, 1, 5] = np.inf
    maps = diag.line_window_maps(cube, var, wl, 5.0, 1.0)
    assert maps["line_flux"][0, 0] == 14.0
    assert maps["partial_line_flux"][0, 0] == 14.0
    assert np.isnan(maps["line_flux"][0, 1])
    assert np.isnan(maps["snr"][0, 1])
    assert maps["partial_line_flux"][0, 1] == 4.0
    assert maps["measured_channel_fraction"][0, 1] == pytest.approx(2 / 3)
    # Channel coverage cannot repair a missing line peak: 4/(2/3) != 14.
    assert maps["partial_line_flux"][0, 1] / maps["measured_channel_fraction"][0, 1] != 14.0
    np.testing.assert_allclose(diag.snr_map(cube, var, wl, 5.0, 1.0), maps["snr"], equal_nan=True)
    np.testing.assert_allclose(
        diag.line_flux_map(cube, wl, 5.0, 1.0, variance_cube=var), maps["line_flux"], equal_nan=True)


@pytest.mark.parametrize("invalid_variance", [np.nan, np.inf, 0.0, -1.0])
def test_line_continuum_masks_invalid_variance_and_flux(invalid_variance):
    wl = np.arange(9.0)
    cube = np.full((1, 1, 9), 10.0)
    cube[:, :, 4] += 5.0
    var = np.ones_like(cube)
    cube[:, :, :3] = 1e9
    var[:, :, :3] = invalid_variance
    cube[:, :, 8] = np.nan  # finite variance alone does not make data usable
    m = diag.line_window_maps(cube, var, wl, 4.0, 0.1)
    assert m["continuum"][0, 0] == 10.0
    assert m["n_continuum_channels"][0, 0] == 4
    assert m["line_flux"][0, 0] == 5.0
    assert m["continuum_variance"][0, 0] == pytest.approx(np.pi / 8)


def test_missing_line_or_continuum_stays_unavailable():
    wl = np.arange(9.0)
    cube = np.ones((1, 1, 9))
    var = np.full_like(cube, np.inf)
    all_bad = diag.line_window_maps(cube, var, wl, 4.0, 1.0)
    for key in ("line_flux", "variance", "snr", "partial_line_flux", "partial_variance", "partial_snr"):
        assert np.isnan(all_bad[key][0, 0])
    assert all_bad["measured_channel_fraction"][0, 0] == 0
    var[:, :, 3:6] = 1.0
    no_cont = diag.line_window_maps(cube, var, wl, 4.0, 1.0)
    assert no_cont["complete_window"][0, 0]
    assert np.isnan(no_cont["line_flux"][0, 0])
    no_subtract = diag.line_window_maps(cube, var, wl, 4.0, 1.0, subtract_continuum=False)
    assert no_subtract["line_flux"][0, 0] == 3.0
    assert no_subtract["variance"][0, 0] == 3.0
    assert np.isnan(diag.line_flux_map(cube, wl, 4.0, 20.0)[0, 0])
    empty_window = diag.line_window_maps(cube, np.ones_like(cube), wl, 20.0, 0.1)
    assert np.isnan(empty_window["partial_snr"][0, 0])
    assert empty_window["measured_channel_fraction"][0, 0] == 0.0


def test_gaussian_median_continuum_variance_analytic_and_monte_carlo():
    wl = np.arange(200.0)
    cube = np.full((1, 1, 200), 10.0)
    var = np.ones_like(cube)
    maps = diag.line_window_maps(cube, var, wl, 100.0, 7.0)
    expected = 15.0 + 15.0**2 * np.pi / (2 * 185)
    assert maps["variance"][0, 0] == pytest.approx(expected)
    rng = np.random.default_rng(9)
    samples = rng.normal(size=(18000, 200))
    mask = np.abs(wl - 100.0) <= 7.0
    measured = samples[:, mask].sum(axis=1) - 15 * np.median(samples[:, ~mask], axis=1)
    assert np.var(measured, ddof=1) == pytest.approx(expected, rel=0.035)
    aperture = diag.aperture_spectrum_uncertainty(cube, var, np.ones((1, 1), bool),
                                                wl, 100.0, 7.0, noise_reference_cube=cube)
    assert aperture["noise_diagonal"]**2 == pytest.approx(expected)
    assert aperture["noise_covariance_aware"]**2 == pytest.approx(expected)


def test_heteroskedastic_and_small_sample_median_variance():
    wl = np.arange(5.0)
    cube = np.full((1, 1, 5), 10.0)
    var = np.array([[[1., 4., 9., 16., 25.]]])
    maps = diag.line_window_maps(cube, var, wl, 2.0, 0.1)
    expected_cont = np.pi * 4 / (2 * (1 + 1/2 + 1/4 + 1/5)**2)
    assert maps["continuum_variance"][0, 0] == pytest.approx(expected_cont)
    var[:, :, [0, 4]] = np.inf
    maps = diag.line_window_maps(cube, var, wl, 2.0, 0.1)
    assert maps["continuum_variance"][0, 0] == 5.0  # median of two is their mean
    var[:, :, 1] = np.inf
    maps = diag.line_window_maps(cube, var, wl, 2.0, 0.1)
    assert maps["continuum_variance"][0, 0] == 16.0


def test_aperture_partial_channels_keep_fixed_spatial_support():
    wl = np.arange(11.0)
    cube = np.full((1, 2, 11), 10.0)
    cube[0, 1, :] = 100.0  # varying spatial support must not change continuum
    cube[:, :, 4:7] += 5.0
    var = np.ones_like(cube)
    var[0, 1, [1, 5]] = np.inf  # one off-line and one in-line aperture gap
    d = diag.aperture_spectrum_uncertainty(cube, var, np.ones((1, 2), bool), wl, 5., 1.)
    assert d["line_flux"] == 20.0  # two complete channels, two spaxels, 5e each
    assert d["n_measured_window_channels"] == 2
    assert d["n_continuum_channels"] == 7
    assert d["measured_channel_fraction"] == pytest.approx(2/3)
    assert np.isnan(d["full_window_line_flux"])
    assert np.isfinite(d["partial_snr_diagonal"])
    assert d["line_flux_definition"] == "measured_partial_window_fixed_effective_aperture"


@pytest.mark.parametrize("bad_kind", ["all", "continuum", "nan_flux", "negative_variance"])
def test_aperture_undefined_measurements_are_nan(bad_kind):
    wl = np.arange(11.0)
    cube = np.full((1, 2, 11), 10.0)
    var = np.ones_like(cube)
    if bad_kind == "all":
        var[:] = np.inf
    elif bad_kind == "continuum":
        var[:, :, np.abs(wl - 5.0) > 1.0] = np.inf
    elif bad_kind == "nan_flux":
        cube[:] = np.nan
    else:
        var[:] = -1.0
    d = diag.aperture_spectrum_uncertainty(cube, var, np.ones((1, 2), bool), wl, 5., 1.)
    for key in ("line_flux", "snr_diagonal", "snr_covariance_aware", "full_window_line_flux"):
        assert np.isnan(d[key])


def test_residual_autocorrelation_does_not_bridge_missing_channels():
    residual = np.array([1.0, np.nan, 3.0, np.nan, 2.0, np.nan, 4.0])
    assert np.isnan(diag._lag1_autocorr(residual))
    assert np.isnan(diag._detrended_lag1_autocorr(residual))


def test_velocity_moment_returns_nan_for_masked_line_channel():
    wl = np.arange(11.0) + 100.0
    cube = np.full((1, 1, 11), 10.0)
    cube[:, :, 5] += 20.0
    var = np.ones_like(cube)
    var[:, :, 5] = np.inf
    assert np.isnan(diag.velocity_centroid_map(cube, wl, 105., 1., variance_cube=var)[0, 0])


def test_gaussian_fit_requires_finite_uncertainty_for_detection(monkeypatch):
    import scipy.optimize

    def undefined_fit_covariance(*args, **kwargs):
        return np.array([50., 2.2, .005, 10., 0.]), np.full((5, 5), np.inf)

    monkeypatch.setattr(scipy.optimize, "curve_fit", undefined_fit_covariance)
    wl = np.linspace(2.0, 2.4, 400)
    f = diag.fit_gaussian_line(_line_spectrum(wl, 2.2, 50., 2.2, .005), wl, 2.2, .05)
    assert f["fit_converged"]
    assert not f["fit_detected"]
    assert not f["success"]
