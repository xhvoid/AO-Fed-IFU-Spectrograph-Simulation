"""Science diagnostics computed from a reconstructed datacube.

These are the "what did we actually measure" products an IFU pipeline delivers,
reduced here to compact, inspectable functions:

    line_flux_map        integrated, continuum-subtracted line flux per spaxel
    velocity_centroid_map line-of-sight velocity per spaxel (first moment)
    snr_map              signal-to-noise of the line per spaxel
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
