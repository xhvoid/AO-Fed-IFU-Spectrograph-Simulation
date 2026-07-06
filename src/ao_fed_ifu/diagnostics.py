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


def _continuum_subtract(
    cube: np.ndarray, in_window: np.ndarray
) -> np.ndarray:
    """Subtract a per-spaxel continuum estimated from *outside* the line window.

    The continuum level at each spaxel is the median of the channels that lie
    outside the line window (robust to the line itself).  Returns a cube of the
    same shape with the continuum removed.
    """
    out_window = ~in_window
    if not np.any(out_window):
        # Degenerate case: window covers everything -> no continuum estimate.
        return cube
    continuum = np.median(cube[:, :, out_window], axis=2, keepdims=True)
    return cube - continuum


def line_flux_map(
    reconstructed_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
    subtract_continuum: bool = True,
) -> np.ndarray:
    """Integrated line flux per spaxel, in electrons.

    Sums the (optionally continuum-subtracted) cube over the wavelength channels
    within +/- ``window_um`` of ``line_center_um``.  Returns a 2-D map
    ``(ny, nx)``; brighter line emission -> larger values.
    """
    in_window = _window_mask(wavelength_grid_um, line_center_um, window_um)
    cube = _continuum_subtract(reconstructed_cube, in_window) if subtract_continuum else reconstructed_cube
    return cube[:, :, in_window].sum(axis=2)


def velocity_centroid_map(
    reconstructed_cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
) -> np.ndarray:
    """Line-of-sight velocity per spaxel from the first spectral moment (km/s).

    For each spaxel the intensity-weighted mean wavelength (the first moment) of
    the continuum-subtracted line is converted to a velocity relative to the rest
    wavelength:

        lambda_cen = sum(lambda * I) / sum(I)
        v          = c * (lambda_cen - lambda_rest) / lambda_rest

    Spaxels with non-positive line flux are returned as NaN (no measurable line).
    """
    wl = np.asarray(wavelength_grid_um, dtype=float)
    in_window = _window_mask(wl, line_center_um, window_um)
    cube = _continuum_subtract(reconstructed_cube, in_window)

    wl_win = wl[in_window]
    intens = cube[:, :, in_window]
    # Clip tiny negatives from continuum subtraction so the moment is stable.
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
    """Signal-to-noise ratio of the line per spaxel.

    Signal = integrated line flux (continuum-subtracted) in the window.
    Noise  = sqrt(sum of per-channel variance over the same window), the
    standard propagation for a summed quantity with independent channels.
    A bad voxel carries infinite variance, so it drives that spaxel's noise to
    infinity and its SNR cleanly to zero.

    Scope: the variance cube propagates Poisson + dark + read noise only.
    Flat-field response error and saturation are applied to the data but treated
    as calibratable systematics (not in the variance), and spatial/spectral
    covariance is not propagated -- so this is a *propagated* SNR, not a full
    error budget.
    """
    in_window = _window_mask(wavelength_grid_um, line_center_um, window_um)
    signal = line_flux_map(reconstructed_cube, wavelength_grid_um, line_center_um, window_um)

    noise = np.sqrt(variance_cube[:, :, in_window].sum(axis=2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(noise > 0, signal / noise, 0.0)


def covariance_inflation_factor(lag1_autocorr: float, n_channels: int) -> float:
    """Variance inflation for summing ``n_channels`` correlated channels (AR(1)).

    A diagonal (independent-channel) error budget -- what :func:`snr_map` uses --
    sums per-channel variances as if ``Cov(i, j) = 0`` for ``i != j``.  Real IFU
    spectra are correlated by the LSF and cross-talk, so neighbouring channels
    covary and the true variance of a summed line flux is larger.

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
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    v = v - np.mean(v)
    a, b = v[:-1], v[1:]
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def _detrended_lag1_autocorr(values: np.ndarray, degree: int = 3) -> float:
    """Lag-1 autocorrelation of a sequence after removing a smooth polynomial.

    The off-line continuum of an IFU aperture spectrum is itself smooth (the
    telluric/filter/continuum SHAPE), so a raw lag-1 autocorrelation would be
    dominated by that shape (~1) rather than by the channel-to-channel NOISE
    correlation the LSF and cross-talk imprint.  Removing a low-order polynomial
    trend first isolates the noise correlation, which is what the covariance
    inflation should be built from.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < max(degree + 3, 4):
        return _lag1_autocorr(v)
    x = np.arange(v.size, dtype=float)
    coeffs = np.polyfit(x, v, min(degree, v.size - 2))
    return _lag1_autocorr(v - np.polyval(coeffs, x))


def aperture_spectrum_uncertainty(
    reconstructed_cube: np.ndarray,
    variance_cube: np.ndarray,
    aperture_mask: np.ndarray,
    wavelength_grid_um: np.ndarray,
    line_center_um: float,
    window_um: float,
    noise_reference_cube: np.ndarray | None = None,
) -> dict:
    """Aperture line-flux SNR with a diagonal AND an approximate covariance-aware
    uncertainty.

    Sums the aperture into a 1-D flux spectrum and a 1-D variance spectrum
    (variances add over spaxels assumed spatially independent), continuum-
    subtracts the flux, and integrates the line over ``+/- window_um``.  Two
    uncertainties are reported:

      * **diagonal** -- ``sqrt(sum of per-channel variance in the window)``, the
        independent-channel budget :func:`snr_map` also assumes;
      * **covariance-aware** -- the diagonal variance inflated by
        :func:`covariance_inflation_factor` ``(rho, N)``, where ``rho`` is the
        lag-1 spectral autocorrelation of the reconstructed NOISE and ``N`` is
        the number of channels in the line window.

    Estimating ``rho`` -- how ``rho`` is measured is set by
    ``noise_reference_cube``:

      * given a noiseless reference reconstruction of the SAME source through the
        SAME reduction (e.g. the pipeline's ``observed_truth_cube``),
        ``reconstructed_cube - noise_reference_cube`` approximately isolates the
        reconstructed noise and ``rho`` is its lag-1 autocorrelation (method
        ``"isolated_noise"``, the preferred path).  This is *exact* only for a
        linear detector chain; the default nonlinear/reduction terms (dark
        current, multiplicative flat-field, a deliberate sky-subtraction scale
        error) add a small DETERMINISTIC residual on top of the noise.  That
        residual is sub-dominant to the stochastic noise and its mean is removed
        by the autocorrelation, so ``rho`` (and hence the inflation) stays
        essentially unbiased -- but the difference is not *purely* noise;
      * without a reference, ``rho`` falls back to the lag-1 autocorrelation of
        the DETRENDED off-line continuum residual (method
        ``"detrended_offline_proxy"``) -- a rough upper bound that can be
        contaminated by residual continuum shape.

    Positive spectral correlation (``rho > 0``) inflates the uncertainty and
    lowers the covariance-aware SNR relative to the diagonal SNR; a negative
    measured ``rho`` is clamped to zero inside
    :func:`covariance_inflation_factor` (a conservative budget never credits a
    noise reduction), so ``snr_covariance_aware <= snr_diagonal`` always.  This
    is an APPROXIMATE covariance-aware diagnostic (a single lag-1 AR(1) proxy),
    NOT a full spectral covariance matrix.

    Voxels with non-finite (bad-pixel) variance are excluded from BOTH the flux
    and the variance sums (and from the isolated-noise spectrum), so a masked
    voxel neither sends the whole aperture SNR to zero nor leaks its flux into
    the signal with no matching noise term.  Bad pixels are typically flagged
    as whole spaxels, where this exclusion is exact; a mask that varies per
    channel also varies the number of spaxels entering each channel, which the
    flat aperture continuum estimate does not model.

    Returns a dict of scalars.
    """
    mask_ap = np.asarray(aperture_mask, dtype=bool)
    # A voxel contributes only if it is inside the aperture AND carries finite
    # (good-pixel) variance: flux and variance must be excluded TOGETHER, or a
    # masked voxel's flux would enter the signal with no matching noise term.
    valid = mask_ap[:, :, None] & np.isfinite(variance_cube)
    flux_spec = np.where(valid, reconstructed_cube, 0.0).sum(axis=(0, 1))  # (nlambda,)
    var_spec = np.where(valid, variance_cube, 0.0).sum(axis=(0, 1))        # (nlambda,)

    in_window = _window_mask(wavelength_grid_um, line_center_um, window_um)
    out_window = ~in_window
    cont = float(np.median(flux_spec[out_window])) if np.any(out_window) else 0.0
    line = flux_spec - cont

    signal = float(np.sum(line[in_window]))
    n_win = int(np.count_nonzero(in_window))
    diag_var = float(np.sum(var_spec[in_window]))

    if noise_reference_cube is not None:
        # The same reduction applied to (data) and (noiseless source) differ by
        # the reconstructed noise plus a small deterministic residual from the
        # nonlinear/reduction terms (dark, flat-field, any sky-scale error). The
        # residual is sub-dominant and its mean drops out of the autocorrelation,
        # so this approximately isolates the noise without de-trending; it is
        # exact only for a linear chain.
        noise_diff = (np.asarray(reconstructed_cube, dtype=float)
                      - np.asarray(noise_reference_cube, dtype=float))
        noise_ap = np.where(valid, noise_diff, 0.0).sum(axis=(0, 1))
        rho = _lag1_autocorr(noise_ap)
        rho_method = "isolated_noise"
    else:
        rho = _detrended_lag1_autocorr(line[out_window])
        rho_method = "detrended_offline_proxy"
    if not np.isfinite(rho):
        rho = 0.0
    infl = covariance_inflation_factor(rho, n_win)

    noise_diag = float(np.sqrt(diag_var)) if diag_var > 0 else float("inf")
    noise_cov = noise_diag * float(np.sqrt(infl))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_diag = signal / noise_diag if np.isfinite(noise_diag) and noise_diag > 0 else 0.0
        snr_cov = signal / noise_cov if np.isfinite(noise_cov) and noise_cov > 0 else 0.0

    return {
        "aperture_spaxels": int(mask_ap.sum()),
        "n_window_channels": n_win,
        "line_flux": signal,
        "lag1_autocorr": float(rho),
        "lag1_autocorr_method": rho_method,
        "covariance_inflation_factor": float(infl),
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
    noise; otherwise they come from the residual scatter.

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
    x, y = wl[sel], spec[sel]
    if x.size < 5 or not np.all(np.isfinite(y)):
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

    sigma_arg = None
    if variance_1d is not None:
        v = np.asarray(variance_1d, dtype=float)[sel]
        if np.all(np.isfinite(v)) and np.all(v > 0):
            sigma_arg = np.sqrt(v)

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

    detected = bool(np.isfinite(flux) and flux > 0 and amp > 0)
    if np.isfinite(snr_scaled) and snr_scaled < min_snr:
        detected = False
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
