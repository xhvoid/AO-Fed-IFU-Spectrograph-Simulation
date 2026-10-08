"""Science diagnostics computed from a reconstructed datacube.

These are the "what did we actually measure" products an IFU pipeline delivers,
reduced here to compact, inspectable functions:

    line_flux_map        integrated, continuum-subtracted line flux per spaxel
    velocity_centroid_map line-of-sight velocity per spaxel (first moment)
    snr_map              signal-to-noise of the line per spaxel (diagonal only)
    aperture_spectrum_uncertainty aperture SNR with a diagonal AND an approximate
                         covariance-aware (AR(1)) uncertainty
    fit_gaussian_line    Gaussian+continuum fit of a 1-D line (flux, velocity,
                         width, fitted R, formal errors, fit quality)
    fit_gaussian_line_map per-spaxel Gaussian fit -> flux/velocity/sigma maps
    resolving_power_proxy R = lambda / FWHM recovered from a line profile
    cross_talk_fraction  flux leaked out of an isolated illuminated spaxel
    calibration_bias_summary line-flux / velocity / SNR bias from a perturbation

Physics recap
    A spectral emission line carries both flux (how bright) and a Doppler shift
    (how fast the gas moves along the line of sight): an observed centroid
    lambda_obs maps to velocity ``v = c (lambda_obs - lambda_rest) / lambda_rest``.
    The first spectral moment of the (continuum-subtracted) line gives lambda_obs,
    and its second moment gives the line width, hence the resolving power.
"""

from __future__ import annotations

import numpy as np

from . import SPEED_OF_LIGHT_KM_S


def _window_mask(wavelength_grid_um: np.ndarray, line_center_um: float, window_um: float) -> np.ndarray:
    """Boolean mask of wavelength channels within +/- ``window_um`` of the line."""
    wl = np.asarray(wavelength_grid_um, dtype=float)
    return np.abs(wl - line_center_um) <= window_um


def _median_continuum(cube, variance, valid):
    """Median and its Gaussian, independent-channel variance approximation.

    For N independent Gaussian samples with a common continuum level and known,
    possibly unequal standard deviations, the asymptotic median variance is
    ``pi*N / (2*(sum(1/sigma))**2)``. For N=1 and N=2 the sample median is a
    linear mean, so its exact variance is used. This does not account for a
    sloping/misspecified continuum or spectral covariance, and is approximate
    for small N>=3. Unmeasured channels never enter either quantity.
    """
    n = np.sum(valid, axis=-1)
    continuum = np.ma.median(np.ma.array(cube, mask=~valid), axis=-1).filled(np.nan)
    inv_sigma = np.divide(1.0, np.sqrt(np.where(valid, variance, 1.0)),
                          out=np.zeros_like(variance, dtype=float), where=valid)
    precision_sum = inv_sigma.sum(axis=-1)
    out = np.full(n.shape, np.nan, dtype=float)
    np.divide(np.pi * n, 2.0 * precision_sum**2, out=out, where=n > 0)
    small = (n > 0) & (n <= 2)
    np.divide(np.where(valid, variance, 0.0).sum(axis=-1), np.maximum(n, 1)**2,
              out=out, where=small)
    return np.asarray(continuum), out, n


def _diagnostic_inputs(cube, variance_cube, wavelength_grid_um):
    cube = np.asarray(cube, dtype=float)
    variance = np.asarray(variance_cube, dtype=float)
    wl = np.asarray(wavelength_grid_um, dtype=float)
    if cube.ndim != 3 or variance.shape != cube.shape:
        raise ValueError("cube and variance_cube must have matching (ny, nx, nlambda) shapes")
    if wl.ndim != 1 or wl.size != cube.shape[-1] or not np.all(np.isfinite(wl)):
        raise ValueError("wavelength grid must be finite and match the cube spectral axis")
    valid = np.isfinite(cube) & np.isfinite(variance) & (variance > 0)
    return cube, variance, wl, valid


def line_window_maps(
    reconstructed_cube: np.ndarray,
    variance_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
    subtract_continuum: bool = True,
) -> dict[str, np.ndarray]:
    """Full-window and explicitly partial line measurements, with coverage.

    ``line_flux``, ``variance`` and ``snr`` require every requested line channel
    to be measured. Missing/invalid data or an unavailable continuum produce
    NaN, not a zero non-detection. ``partial_*`` sum only valid channels and
    describe a DIFFERENT, incomplete-window quantity; no coverage rescaling or
    spectral interpolation is performed. Channel coverage is not flux coverage.

    The off-line median ignores invalid flux and nonpositive/nonfinite variance.
    Its uncertainty uses :func:`_median_continuum` and contributes
    ``N_measured**2 * Var(continuum)`` to the line variance, since the SAME
    continuum estimate is subtracted from every line channel. Spectral
    covariance, continuum-model error and detector calibration systematics are
    outside this diagonal, Gaussian-approximate budget. With no usable off-line
    channels continuum-subtracted measurements are unavailable; explicitly set
    ``subtract_continuum=False`` only when no continuum estimate is intended.
    """
    cube, variance, wl, valid = _diagnostic_inputs(
        reconstructed_cube, variance_cube, wavelength_grid_um)
    in_window = _window_mask(wl, line_center_um, window_um)
    n_window = int(in_window.sum())
    line_valid = valid & in_window[None, None, :]
    n_measured = line_valid.sum(axis=-1)
    cont, cont_var, n_cont = _median_continuum(
        cube, variance, valid & ~in_window[None, None, :])
    if not subtract_continuum:
        cont = np.zeros(cube.shape[:2])
        cont_var = np.zeros(cube.shape[:2])
    available = (n_measured > 0) & np.isfinite(cont) & np.isfinite(cont_var)
    partial_flux = np.where(line_valid, cube, 0.0).sum(axis=-1) - n_measured * cont
    partial_var = (np.where(line_valid, variance, 0.0).sum(axis=-1)
                   + n_measured**2 * cont_var)
    partial_flux = np.where(available, partial_flux, np.nan)
    partial_var = np.where(available, partial_var, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        partial_snr = partial_flux / np.sqrt(partial_var)
    complete = (n_window > 0) & (n_measured == n_window)
    return {
        "line_flux": np.where(complete, partial_flux, np.nan),
        "variance": np.where(complete, partial_var, np.nan),
        "snr": np.where(complete, partial_snr, np.nan),
        "partial_line_flux": partial_flux,
        "partial_variance": partial_var,
        "partial_snr": partial_snr,
        "measured_channel_fraction": n_measured / n_window if n_window else np.zeros_like(cont),
        "complete_window": complete,
        "n_measured_channels": n_measured,
        "continuum": cont,
        "continuum_variance": cont_var,
        "n_continuum_channels": n_cont,
    }


def _continuum_subtract(cube: np.ndarray, in_window: np.ndarray,
                        valid: np.ndarray | None = None) -> np.ndarray:
    """Subtract the valid off-line median; no available continuum gives NaN."""
    cube = np.asarray(cube, dtype=float)
    valid = np.isfinite(cube) if valid is None else valid & np.isfinite(cube)
    continuum, _, _ = _median_continuum(
        cube, np.ones_like(cube), valid & ~in_window[None, None, :])
    return np.where(valid, cube - continuum[:, :, None], np.nan)


def line_flux_map(
    reconstructed_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
    subtract_continuum: bool = True,
    *,
    variance_cube: np.ndarray | None = None,
) -> np.ndarray:
    """Integrated line flux per spaxel, in electrons.

    Sums the (optionally continuum-subtracted) cube over the wavelength channels
    within +/- ``window_um`` of ``line_center_um``.  Returns a 2-D map
    ``(ny, nx)``. An incomplete window or unavailable continuum gives NaN.
    Supply ``variance_cube`` to recognize zero-filled missing measurements;
    without it, only nonfinite flux is recognized as invalid. For partial-window
    measurements and coverage use :func:`line_window_maps` explicitly.
    """
    variance = np.ones_like(reconstructed_cube) if variance_cube is None else variance_cube
    return line_window_maps(reconstructed_cube, variance, wavelength_grid_um,
                            line_center_um, window_um, subtract_continuum)["line_flux"]


def velocity_centroid_map(
    reconstructed_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
    *,
    variance_cube: np.ndarray | None = None,
) -> np.ndarray:
    """Line-of-sight velocity per spaxel from the first spectral moment (km/s).

    For each spaxel the intensity-weighted mean wavelength (the first moment) of
    the continuum-subtracted line is converted to a velocity relative to the rest
    wavelength:

        lambda_cen = sum(lambda * I) / sum(I)
        v          = c * (lambda_cen - lambda_rest) / lambda_rest

    This robust moment clips ALL negative continuum-subtracted samples, not just
    numerical roundoff. That creates a positive noise pedestal which tends to
    pull weak-line centroids toward the centre of a symmetric window; it is
    biased at low SNR and is not a detection statistic. Nonuniform channel
    noise can produce other centroid biases. No positive clipped flux, missing
    line channels, or an unavailable continuum produces NaN. Supply variance
    to recognize zero-filled missing voxels.
    """
    wl = np.asarray(wavelength_grid_um, dtype=float)
    in_window = _window_mask(wl, line_center_um, window_um)
    variance = np.ones_like(reconstructed_cube) if variance_cube is None else variance_cube
    raw, _, _, valid = _diagnostic_inputs(reconstructed_cube, variance, wl)
    cube = _continuum_subtract(raw, in_window, valid)

    wl_win = wl[in_window]
    intens = cube[:, :, in_window]
    # Robust but biased positive-part moment; see the low-SNR caveat above.
    intens = np.clip(intens, 0.0, None)
    weight = intens.sum(axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_cen = np.where(
            weight > 0,
            (intens * wl_win[None, None, :]).sum(axis=2) / weight,
            np.nan,
        )
    return SPEED_OF_LIGHT_KM_S * (lambda_cen - line_center_um) / line_center_um


def snr_map(
    reconstructed_cube: np.ndarray,
    variance_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
) -> np.ndarray:
    """Full-window line SNR, including approximate median-continuum variance.

    Missing line data or an undefined continuum yield NaN, not a zero physical
    non-detection. See :func:`line_window_maps` for explicit partial-window SNR,
    coverage and assumptions of the diagonal uncertainty budget.
    """
    return line_window_maps(reconstructed_cube, variance_cube, wavelength_grid_um,
                            line_center_um, window_um)["snr"]


def covariance_inflation_factor(lag1_autocorr: float, n_channels: int) -> float:
    """Variance inflation for summing ``n_channels`` correlated channels (AR(1)).

    A diagonal (independent-channel) error budget -- what :func:`snr_map` uses --
    sums per-channel variances as if ``Cov(i, j) = 0`` for ``i != j``.  Real IFU
    reductions can correlate spectral noise through interpolation/resampling.
    An optical LSF smooths the expected signal; it does not itself correlate
    independent photon/read noise drawn AFTER that convolution. Spatial trace
    cross-talk/deblending likewise need not correlate wavelength channels.

    Approximating the channel-to-channel correlation as an AR(1) process with
    lag-1 coefficient ``rho`` (the spectral-autocorrelation proxy), so
    ``Corr(i, j) = rho**|i-j|``, the variance of the ``N``-channel sum is inflated
    relative to the diagonal budget by

        f = Var(sum) / (N * v) = 1 + (2 / N) * sum_{k=1}^{N-1} (N - k) * rho**k.

    ``rho = 0`` -> ``f = 1`` (recovers the diagonal budget); ``rho > 0`` -> ``f > 1``
    (positive correlation inflates the uncertainty).  A measured NEGATIVE lag-1
    autocorrelation would mathematically deflate the summed variance (``f < 1``),
    but this factor is a conservative error-budget INFLATION: crediting a noise
    reduction from a one-parameter proxy is not defensible, so negative ``rho``
    is clamped to zero and the returned factor is always ``>= 1``.  Callers that
    want the signed measurement should report ``rho`` itself alongside.

    This is a deliberately APPROXIMATE, single-parameter covariance model -- one
    lag-1 number stretched over all lags -- NOT a full spectral covariance matrix.
    """
    n = int(n_channels)
    if n <= 1:
        return 1.0
    rho = float(np.clip(lag1_autocorr, 0.0, 0.999))
    k = np.arange(1, n)
    f = 1.0 + (2.0 / n) * float(np.sum((n - k) * rho**k))
    # rho >= 0 makes every off-diagonal term non-negative, so f >= 1 exactly;
    # the max() is a numerical safety net that also documents the invariant.
    return float(max(f, 1.0))


def _lag1_autocorr(values: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D sequence (mean-removed); NaN if undefined."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() < 3:
        return float("nan")
    v = v - np.mean(v[finite])
    pairs = finite[:-1] & finite[1:]
    if pairs.sum() < 2:
        return float("nan")
    a, b = v[:-1][pairs], v[1:][pairs]
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def _detrended_lag1_autocorr(values: np.ndarray, degree: int = 3) -> float:
    """Lag-1 autocorrelation of a sequence after removing a smooth polynomial.

    The off-line continuum of an IFU aperture spectrum is itself smooth (the
    telluric/filter/continuum SHAPE), so a raw lag-1 autocorrelation would be
    dominated by that shape (~1) rather than by the channel-to-channel NOISE
    correlation. Removing a low-order polynomial from LINE-EXCLUDED channels
    reduces smooth contamination but does not isolate pure noise or guarantee
    an upper bound. Masked channel gaps retain their original positions.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() < max(degree + 3, 4):
        return _lag1_autocorr(v)
    x = np.arange(v.size, dtype=float)
    coeffs = np.polyfit(x[finite], v[finite], min(degree, int(finite.sum()) - 2))
    return _lag1_autocorr(v - np.polyval(coeffs, x))


def aperture_spectrum_uncertainty(
    reconstructed_cube: np.ndarray,
    variance_cube: np.ndarray,
    aperture_mask: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
    noise_reference_cube: np.ndarray | None = None,
    extraction_model=None,
) -> dict:
    """Measured-partial-window aperture SNR, coverage, and full-window results.

    Wholly unmeasured spaxels are dropped ONCE from a fixed effective aperture
    and its spatial coverage is reported. Only channels where that aperture is
    complete enter the line or off-line median; an aperture that changes with
    wavelength would bias its continuum. The historical ``line_flux`` and SNR
    keys describe this explicitly partial spectral-window measurement, never
    a coverage-corrected total. ``full_window_*`` keys are NaN unless the full
    requested aperture and line window are available. All-missing data or no
    usable continuum yield NaN, not false zero detections.

    The diagonal budget includes the shared median-continuum variance from
    :func:`_median_continuum`. With an extraction model, each complete channel's
    spatial aperture variance is exact (u^T Cov u), and those variances feed the
    SAME Gaussian median approximation. Spatial anti-correlations can reduce
    aperture noise. The continuum model must be approximately constant.

    Spectral covariance remains an approximate AR(1) budget: positive residual
    lag-1 correlation inflates the line sum and median-continuum terms using
    their respective channel counts. Cross-covariance between the line and
    off-line median is not modelled. Masked gaps preserve adjacency when rho is
    measured; the count-based AR(1) approximation does not model those gaps or
    arbitrary heteroskedastic covariance exactly.

    With ``noise_reference_cube``, rho measures a MATCHED REFERENCE RESIDUAL,
    which can contain calibration mismatch as well as random noise. Removing
    its mean does not remove structured residuals, so neither smallness nor an
    unbiased noise-correlation estimate is guaranteed. Without a reference,
    a line-excluded, polynomial-detrended residual is used; it is not a proven
    upper bound. An optical LSF alone smooths signal, not independent detector
    noise generated after it. These are conditional internal uncertainties,
    not an externally calibrated error budget.
    """
    cube, variance, wl, valid = _diagnostic_inputs(
        reconstructed_cube, variance_cube, wavelength_grid_um)
    mask_ap = np.asarray(aperture_mask, dtype=bool)
    if mask_ap.shape != cube.shape[:2]:
        raise ValueError("aperture_mask must match the cube spatial shape")
    effective_ap = mask_ap & np.any(valid, axis=-1)
    n_ap = int(mask_ap.sum())
    n_effective = int(effective_ap.sum())
    complete_channel = (np.all(valid[effective_ap, :], axis=0) if n_effective
                        else np.zeros(wl.size, dtype=bool))
    flux_spec = np.where(effective_ap[:, :, None] & valid, cube, 0.0).sum(axis=(0, 1))
    var_spec = np.where(effective_ap[:, :, None] & valid, variance, 0.0).sum(axis=(0, 1))
    flux_spec = np.where(complete_channel, flux_spec, np.nan)
    var_spec = np.where(complete_channel, var_spec, np.nan)
    if extraction_model is not None:
        spatial_var_spec = np.asarray(extraction_model.aperture_variance(effective_ap), dtype=float)
        if spatial_var_spec.shape != wl.shape:
            raise ValueError("extraction model aperture variance must match the spectral axis")
        spatial_var_spec = np.where(complete_channel, spatial_var_spec, np.nan)
        spatial_method = "exact_extraction_model"
    else:
        spatial_var_spec = var_spec
        spatial_method = "independent_spaxels"
    # A malformed/unavailable model channel must not contribute signal while
    # disappearing only from its propagated covariance.
    common = complete_channel & np.isfinite(spatial_var_spec) & (spatial_var_spec > 0)
    flux_spec = np.where(common, flux_spec, np.nan)
    var_spec = np.where(common, var_spec, np.nan)
    spatial_var_spec = np.where(common, spatial_var_spec, np.nan)
    maps = line_window_maps(flux_spec[None, None, :], var_spec[None, None, :],
                            wl, line_center_um, window_um)
    spatial_maps = line_window_maps(flux_spec[None, None, :], spatial_var_spec[None, None, :],
                                    wl, line_center_um, window_um)
    scalar = lambda key: float(maps[key][0, 0])
    in_window = _window_mask(wl, line_center_um, window_um)
    signal = scalar("partial_line_flux")
    n_win = int(in_window.sum())
    n_measured = int(scalar("n_measured_channels"))
    n_cont = int(scalar("n_continuum_channels"))
    diag_var = scalar("partial_variance")
    spatial_var = float(spatial_maps["partial_variance"][0, 0])
    cont_var = scalar("continuum_variance")
    spatial_cont_var = float(spatial_maps["continuum_variance"][0, 0])
    spatial_factor = spatial_var / diag_var if diag_var > 0 else float("nan")
    if noise_reference_cube is not None:
        reference = np.asarray(noise_reference_cube, dtype=float)
        if reference.shape != cube.shape:
            raise ValueError("noise_reference_cube must match the reconstructed cube")
        ref_complete = np.all(np.isfinite(reference[effective_ap, :]), axis=0)
        residual = np.where(effective_ap[:, :, None], cube - reference, 0.0).sum(axis=(0, 1))
        rho = _lag1_autocorr(np.where(common & ref_complete, residual, np.nan))
        rho_method = "reference_residual"
    else:
        rho = _detrended_lag1_autocorr(np.where(common & ~in_window, flux_spec, np.nan))
        rho_method = "detrended_offline_proxy"
    rho_for_budget = rho if np.isfinite(rho) else 0.0
    infl = covariance_inflation_factor(rho_for_budget, n_measured)
    cont_infl = covariance_inflation_factor(rho_for_budget, n_cont)
    line_spatial_var = float(np.nansum(spatial_var_spec[in_window]))
    covariance_var = line_spatial_var * infl + n_measured**2 * spatial_cont_var * cont_infl
    noise_diag = float(np.sqrt(diag_var)) if diag_var > 0 else float("nan")
    noise_cov = float(np.sqrt(covariance_var)) if covariance_var > 0 else float("nan")
    snr_diag = signal / noise_diag
    snr_cov = signal / noise_cov
    full_available = bool(maps["complete_window"][0, 0]) and n_effective == n_ap and n_ap > 0

    return {
        "aperture_spaxels": n_ap,
        "effective_aperture_spaxels": n_effective,
        "aperture_measured_spaxel_fraction": n_effective / n_ap if n_ap else 0.0,
        "n_window_channels": n_win,
        "n_measured_window_channels": n_measured,
        "n_continuum_channels": n_cont,
        "measured_channel_fraction": scalar("measured_channel_fraction"),
        "line_flux_definition": "measured_partial_window_fixed_effective_aperture",
        "line_flux": signal,
        "partial_line_flux": signal,
        "partial_snr_diagonal": float(snr_diag),
        "partial_snr_covariance_aware": float(snr_cov),
        "full_window_line_flux": signal if full_available else float("nan"),
        "full_window_snr_diagonal": float(snr_diag) if full_available else float("nan"),
        "full_window_snr_covariance_aware": float(snr_cov) if full_available else float("nan"),
        "continuum_variance_diagonal": cont_var,
        "continuum_variance_spatial": spatial_cont_var,
        "continuum_variance_method": "gaussian_independent_sample_median_approximation",
        "lag1_autocorr": float(rho),
        "lag1_autocorr_method": rho_method,
        "covariance_inflation_factor": float(infl),
        "continuum_covariance_inflation_factor": float(cont_infl),
        "spatial_covariance_factor": float(spatial_factor),
        "spatial_covariance_method": spatial_method,
        "noise_diagonal": noise_diag,
        "noise_covariance_aware": noise_cov,
        "snr_diagonal": float(snr_diag),
        "snr_covariance_aware": float(snr_cov),
    }


def resolving_power_proxy(
    spectrum_1d: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
) -> float:
    """Recover R = lambda / FWHM from a 1-D line profile via its half-maximum width.

    The line FWHM is measured directly from the continuum-subtracted profile by
    finding the wavelengths at which it crosses half of its peak, on either side
    of the peak, with linear interpolation between samples:

        FWHM = lambda_right_half - lambda_left_half
        R    = lambda_rest / FWHM

    A half-maximum width is used (rather than the second moment) because it is
    *local* to the line core and therefore far more robust to noise in the line
    wings -- important when measuring R on a single, real (noisy) spaxel.

    This "recovered R" tracks the spectrograph's configured R, broadened by any
    intrinsic line width and the pixel sampling; the validation suite uses it to
    check that larger configured R yields a narrower recovered line.  Returns NaN
    if no clean half-maximum crossing exists (e.g. an undetected line).
    """
    wl = np.asarray(wavelength_grid_um, dtype=float)
    spec = np.asarray(spectrum_1d, dtype=float)
    in_window = _window_mask(wl, line_center_um, window_um)
    out = ~in_window
    cont = float(np.median(spec[out])) if np.any(out) else 0.0

    wl_win = wl[in_window]
    prof = np.clip(spec[in_window] - cont, 0.0, None)
    if prof.size < 3 or prof.max() <= 0:
        return float("nan")

    peak = int(np.argmax(prof))
    half = prof[peak] / 2.0

    # Walk left and right from the peak to the first sub-half-maximum sample.
    li = peak
    while li > 0 and prof[li] > half:
        li -= 1
    ri = peak
    while ri < prof.size - 1 and prof[ri] > half:
        ri += 1
    if li == peak or ri == peak:
        return float("nan")  # line not bracketed within the window

    # Linear interpolation of the exact half-maximum crossing wavelengths.
    lam_left = wl_win[li] + (half - prof[li]) * (wl_win[li + 1] - wl_win[li]) / (prof[li + 1] - prof[li])
    lam_right = wl_win[ri - 1] + (half - prof[ri - 1]) * (wl_win[ri] - wl_win[ri - 1]) / (prof[ri] - prof[ri - 1])
    fwhm = lam_right - lam_left
    if fwhm <= 0:
        return float("nan")
    return float(line_center_um / fwhm)


# ----------------------------------------------------------------------------
# Gaussian-plus-continuum line fitting.
#
# The maps above are FIXED-WINDOW moment/integration diagnostics (integrate a
# few resolution elements, first moment for velocity).  A parametric Gaussian
# fit is a complementary estimator: it uses the whole line profile, returns a
# model line flux / centroid / width with formal uncertainties, and degrades
# gracefully (a clear failure flag) when the line is undetected.  Neither is
# "the truth"; comparing the two is itself a useful recovery diagnostic.
# ----------------------------------------------------------------------------

# Loose fit-quality window on the reduced chi^2 of a variance-weighted fit:
# outside it the error model (or the Gaussian + linear-continuum model itself)
# does not describe the data, so the FORMAL parameter errors are not
# trustworthy -- convergence and detection are then not enough.
FIT_QUALITY_CHI2_RANGE = (0.2, 5.0)
# Fallback when no variance is given (reduced chi^2 undefined): require the
# model to explain most of the variance inside the fit window.
FIT_QUALITY_MIN_R2 = 0.5


def _failed_line_fit(line_center_um: float) -> dict:
    """A fit-result dict marking a failed / undetected line fit."""
    nan = float("nan")
    return {
        "success": False,
        "fit_converged": False, "fit_detected": False, "fit_quality_ok": False,
        "line_flux": nan, "line_flux_err": nan, "line_flux_err_scaled": nan,
        "centroid_um": nan, "velocity_kms": nan, "velocity_err_kms": nan,
        "sigma_um": nan, "fwhm_um": nan, "resolving_power": nan,
        "amplitude": nan, "continuum": nan, "continuum_slope": nan,
        "fit_r2": nan, "reduced_chi2": nan, "error_scale": nan,
        "snr_fit": nan, "snr_fit_scaled": nan,
    }


def fit_gaussian_line(
    spectrum_1d: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float | None = None,
    variance_1d: np.ndarray | None = None,
    min_snr: float = 3.0,
) -> dict:
    """Fit a Gaussian emission line on a linear continuum to a 1-D spectrum.

    Model: ``f(lambda) = c0 + c1 (lambda - lambda0) + amp * exp(-0.5 ((lambda -
    mu)/sigma)^2)`` with ``lambda0 = line_center_um``.  Fitted with
    ``scipy.optimize.curve_fit`` (bounded: ``amp >= 0``, ``sigma > 0``, ``mu``
    inside the fit region).  When ``variance_1d`` is given it is used as the
    per-channel weight and the parameter covariance is in absolute units
    (``absolute_sigma=True``), so the reported uncertainties are propagated
    noise; otherwise they come from the residual scatter. Non-finite flux or
    non-positive/non-finite supplied variance excludes that channel; a masked
    variance must never silently switch the fit to unweighted estimation.

    Parameters
    ----------
    window_um
        Half-width (um) of the fit region around ``line_center_um``.  ``None``
        fits the whole spectrum.  A window a few resolution elements wide keeps
        the continuum well-behaved and the fit stable.
    min_snr
        A converged fit whose chi2-scaled flux SNR (``snr_fit_scaled``) falls
        below this is flagged ``fit_detected=False`` (and hence
        ``success=False``) -- an undetected / unreliable line.

    Returns
    -------
    dict with three separate status flags plus the fit parameters:

    * ``fit_converged`` -- the optimiser returned parameters;
    * ``fit_detected`` -- converged with positive flux/amplitude and
      chi2-scaled flux SNR >= ``min_snr``;
    * ``fit_quality_ok`` -- loose goodness/error-model gate:
      ``reduced_chi2`` inside :data:`FIT_QUALITY_CHI2_RANGE` when a variance is
      given, else ``fit_r2 > FIT_QUALITY_MIN_R2``.  A detection with
      ``fit_quality_ok=False`` converged on a line, but the model/variance do
      not describe the data and the formal errors are not trustworthy;
    * ``success`` -- backward-compatible alias of
      ``fit_converged and fit_detected`` (it does NOT certify fit quality);
    * ``line_flux`` (= ``amp * sigma * sqrt(2 pi)``), ``line_flux_err``
      (FORMAL, propagated), ``line_flux_err_scaled``
      (= ``line_flux_err * error_scale``), ``centroid_um``, ``velocity_kms``
      (from ``mu`` vs ``lambda0``), ``velocity_err_kms``, ``sigma_um``,
      ``fwhm_um``, ``resolving_power`` (= ``lambda0 / FWHM``), ``amplitude``,
      ``continuum`` (``c0``), ``continuum_slope`` (``c1``), ``fit_r2``,
      ``reduced_chi2`` (NaN if no variance),
      ``error_scale`` (= ``sqrt(max(1, reduced_chi2))``, 1 if no variance),
      ``snr_fit`` (FORMAL flux SNR -- optimistic when ``reduced_chi2 >> 1``),
      ``snr_fit_scaled`` (= ``snr_fit / error_scale``, the conservative SNR).

    On non-convergence or a too-narrow fit region the failed-fit dict (all NaN,
    all flags False) is returned rather than raising.
    """
    from scipy.optimize import curve_fit

    wl = np.asarray(wavelength_grid_um, dtype=float)
    spec = np.asarray(spectrum_1d, dtype=float)
    lam0 = float(line_center_um)

    sel = np.abs(wl - lam0) <= window_um if window_um is not None else np.ones_like(wl, dtype=bool)
    sel &= np.isfinite(wl) & np.isfinite(spec)
    variance = None
    if variance_1d is not None:
        variance = np.asarray(variance_1d, dtype=float)
        if variance.shape != spec.shape:
            raise ValueError("variance_1d must match spectrum_1d")
        sel &= np.isfinite(variance) & (variance > 0)
    x, y = wl[sel], spec[sel]
    if x.size <= 5:
        return _failed_line_fit(lam0)

    def model(lam, c0, c1, amp, mu, sigma):
        return c0 + c1 * (lam - lam0) + amp * np.exp(-0.5 * ((lam - mu) / sigma) ** 2)

    order = np.argsort(x)
    xs = x[order]
    span = float(xs[-1] - xs[0])
    dlam = float(np.median(np.diff(xs))) if xs.size > 1 else span
    c0_0 = float(np.median(y))
    amp_0 = float(np.max(y) - c0_0)
    if amp_0 <= 0:
        amp_0 = float(np.std(y)) or 1.0
    sigma_0 = max(2.0 * dlam, span / 10.0)
    p0 = [c0_0, 0.0, amp_0, lam0, sigma_0]
    lo = [-np.inf, -np.inf, 0.0, float(xs[0]), 0.3 * dlam]
    hi = [np.inf, np.inf, np.inf, float(xs[-1]), span]

    sigma_arg = None if variance is None else np.sqrt(variance[sel])

    try:
        popt, pcov = curve_fit(
            model, x, y, p0=p0, bounds=(lo, hi),
            sigma=sigma_arg, absolute_sigma=sigma_arg is not None, maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return _failed_line_fit(lam0)

    c0, c1, amp, mu, sigma = (float(v) for v in popt)
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    root2pi = np.sqrt(2.0 * np.pi)

    flux = amp * sigma * root2pi
    # Propagate the flux error through amp and sigma, including their covariance.
    d_amp, d_sig = sigma * root2pi, amp * root2pi
    flux_var = (d_amp * perr[2]) ** 2 + (d_sig * perr[4]) ** 2 + 2.0 * d_amp * d_sig * pcov[2, 4]
    flux_err = float(np.sqrt(flux_var)) if flux_var > 0 else float("nan")

    vel = SPEED_OF_LIGHT_KM_S * (mu - lam0) / lam0
    vel_err = SPEED_OF_LIGHT_KM_S * perr[3] / lam0
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    resolving_power = lam0 / fwhm if fwhm > 0 else float("nan")

    yhat = model(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if sigma_arg is not None:
        dof = max(x.size - len(popt), 1)
        reduced_chi2 = float(np.sum(((y - yhat) / sigma_arg) ** 2) / dof)
    else:
        reduced_chi2 = float("nan")

    snr_fit = float(flux / flux_err) if (np.isfinite(flux_err) and flux_err > 0) else float("nan")
    # reduced_chi2 >> 1 means the propagated variance (or the Gaussian + linear
    # continuum model itself) underestimates the residuals, so the formal errors
    # are too small; the scaled variants deflate them by sqrt(reduced_chi2),
    # never crediting reduced_chi2 < 1.
    error_scale = float(np.sqrt(max(1.0, reduced_chi2))) if np.isfinite(reduced_chi2) else 1.0
    flux_err_scaled = float(flux_err * error_scale) if np.isfinite(flux_err) else float("nan")
    snr_scaled = float(snr_fit / error_scale) if np.isfinite(snr_fit) else float("nan")

    detected = bool(np.isfinite(flux) and flux > 0 and amp > 0
                    and np.isfinite(snr_scaled) and snr_scaled >= min_snr)
    if np.isfinite(reduced_chi2):
        quality_ok = FIT_QUALITY_CHI2_RANGE[0] < reduced_chi2 < FIT_QUALITY_CHI2_RANGE[1]
    else:
        quality_ok = bool(np.isfinite(r2) and r2 > FIT_QUALITY_MIN_R2)

    return {
        "success": detected,
        "fit_converged": True, "fit_detected": detected, "fit_quality_ok": bool(quality_ok),
        "line_flux": float(flux), "line_flux_err": flux_err, "line_flux_err_scaled": flux_err_scaled,
        "centroid_um": float(mu), "velocity_kms": float(vel), "velocity_err_kms": float(vel_err),
        "sigma_um": float(sigma), "fwhm_um": float(fwhm), "resolving_power": float(resolving_power),
        "amplitude": float(amp), "continuum": float(c0), "continuum_slope": float(c1),
        "fit_r2": float(r2), "reduced_chi2": reduced_chi2, "error_scale": error_scale,
        "snr_fit": snr_fit, "snr_fit_scaled": snr_scaled,
    }


def fit_gaussian_line_map(
    reconstructed_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float | None = None,
    variance_cube: np.ndarray | None = None,
    min_snr: float = 3.0,
) -> dict:
    """Per-spaxel Gaussian line fit -> maps of flux, velocity, sigma, success.

    Applies :func:`fit_gaussian_line` to every spaxel spectrum.  This is a
    per-spaxel loop (one bounded least-squares per spaxel), so it is slower than
    the moment maps -- use it on modest fields or an aperture subset.  Returns a
    dict of ``(ny, nx)`` arrays: ``line_flux``, ``velocity_kms``, ``sigma_um``,
    ``resolving_power``, ``snr_fit`` (NaN where the fit failed) and a boolean
    ``success`` map.
    """
    ny, nx, _ = reconstructed_cube.shape
    flux = np.full((ny, nx), np.nan)
    vel = np.full((ny, nx), np.nan)
    sig = np.full((ny, nx), np.nan)
    rpow = np.full((ny, nx), np.nan)
    snr = np.full((ny, nx), np.nan)
    ok = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            var1d = None if variance_cube is None else variance_cube[iy, ix, :]
            fit = fit_gaussian_line(
                reconstructed_cube[iy, ix, :], wavelength_grid_um, line_center_um,
                window_um=window_um, variance_1d=var1d, min_snr=min_snr,
            )
            ok[iy, ix] = fit["success"]
            if fit["success"]:
                flux[iy, ix] = fit["line_flux"]
                vel[iy, ix] = fit["velocity_kms"]
                sig[iy, ix] = fit["sigma_um"]
                rpow[iy, ix] = fit["resolving_power"]
                snr[iy, ix] = fit["snr_fit"]
    return {
        "line_flux": flux, "velocity_kms": vel, "sigma_um": sig,
        "resolving_power": rpow, "snr_fit": snr, "success": ok,
    }


def cross_talk_fraction(
    reconstructed_cube: np.ndarray,
    illuminated_ix: int,
    illuminated_iy: int,
) -> float:
    """Fraction of recovered flux that leaked out of an isolated spaxel.

    For an input cube where only spaxel ``(illuminated_ix, illuminated_iy)``
    carried flux, collapse the reconstruction over wavelength and compute

        leakage = (total_flux - central_spaxel_flux) / total_flux

    A value of 0 means perfect isolation; larger values mean more cross-talk into
    neighbours.  Increasing the modelled ``cross_talk_sigma_pix`` should increase
    this fraction monotonically (a validation trend check).
    """
    image = reconstructed_cube.sum(axis=2)
    total = float(image.sum())
    if total <= 0:
        return float("nan")
    central = float(image[illuminated_iy, illuminated_ix])
    return (total - central) / total


def calibration_bias_summary(
    reference_cube: np.ndarray,
    perturbed_cube: np.ndarray,
    variance_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
) -> dict:
    """Summarise how a calibration perturbation biases the science products.

    Compares a ``perturbed_cube`` (e.g. reconstructed with a wavelength-offset or
    extra read noise) against a clean ``reference_cube``:

        line_flux_bias_frac  median fractional change in line flux,
        velocity_bias_kms    median change in velocity centroid (km/s),
        snr_loss_frac        median fractional loss of SNR.

    Returns a dict of these scalars -- the calibration-sensitivity notebook and
    the validation suite use it to confirm biases have the expected sign/size.
    """
    flux_ref = line_flux_map(reference_cube, wavelength_grid_um, line_center_um, window_um)
    flux_pert = line_flux_map(perturbed_cube, wavelength_grid_um, line_center_um, window_um)
    vel_ref = velocity_centroid_map(reference_cube, wavelength_grid_um, line_center_um, window_um)
    vel_pert = velocity_centroid_map(perturbed_cube, wavelength_grid_um, line_center_um, window_um)
    snr_ref = snr_map(reference_cube, variance_cube, wavelength_grid_um, line_center_um, window_um)
    snr_pert = snr_map(perturbed_cube, variance_cube, wavelength_grid_um, line_center_um, window_um)

    with np.errstate(divide="ignore", invalid="ignore"):
        flux_bias = np.where(flux_ref != 0, (flux_pert - flux_ref) / flux_ref, np.nan)
        snr_loss = np.where(snr_ref != 0, (snr_ref - snr_pert) / snr_ref, np.nan)

    return {
        "line_flux_bias_frac": float(np.nanmedian(flux_bias)),
        "velocity_bias_kms": float(np.nanmedian(vel_pert - vel_ref)),
        "snr_loss_frac": float(np.nanmedian(snr_loss)),
    }
