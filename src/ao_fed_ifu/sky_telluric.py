"""Sky-background and telluric handling -- a controlled, public-data experiment.

This module turns the cached ESO SkyCalc public data into an inspectable
sky-subtraction / telluric-correction demonstration.  It is **not** a real
reduction pipeline: it lets us *inject* a known sky background and telluric
absorption, then *remove* them imperfectly (with a controllable scale error), so
the residual's effect on the science can be measured.

Physical picture
    observed = source x telluric_transmission + sky_emission
    reduced  = (observed - sky_estimate) / telluric_estimate

Provenance
    The SkyCalc radiance / transmission are ``direct_public_data`` when cached
    (else ``synthetic_proxy``).  The radiance -> electrons conversion is a
    physically-motivated but ``engineering_estimate`` scaling, never a calibrated
    background prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SkyTelluricConfig:
    """Scalars needed to turn SkyCalc radiance into a sky-background electron rate.

    All physically meaningful, but the absolute level is an engineering_estimate
    (the throughput is a scalar and the conversion is not a calibrated ETC).
    """

    collecting_area_m2: float
    exposure_s: float
    throughput_scalar: float
    spaxel_scale_mas: float
    delta_lambda_um: float
    sky_scaling: float = 1.0

    @property
    def spaxel_solid_angle_arcsec2(self) -> float:
        """Solid angle of one spaxel in arcsec^2 (mas/1000 -> arcsec, squared)."""
        return (self.spaxel_scale_mas / 1000.0) ** 2


def build_sky_background_spectrum(
    wavelength_grid_um: np.ndarray,
    sky_model: dict,
    config: SkyTelluricConfig,
    filter_transmission: np.ndarray | None = None,
) -> np.ndarray:
    """Sky-background electrons per spaxel per wavelength bin.

    Converts SkyCalc radiance ``[ph / s / m^2 / um / arcsec^2]`` to electrons via

        sky_e = radiance * area * exposure * throughput * filter * dlambda * Omega

    where ``Omega`` is the spaxel solid angle.  ``filter_transmission`` (if given)
    must already be sampled on ``wavelength_grid_um``.  Returns a 1-D array the
    length of the wavelength grid (uniform across spaxels).
    """
    radiance = np.clip(np.asarray(sky_model["radiance"], dtype=float), 0.0, None)
    filt = np.ones_like(wavelength_grid_um) if filter_transmission is None else np.clip(filter_transmission, 0.0, 1.0)
    return (
        radiance
        * config.collecting_area_m2
        * config.exposure_s
        * config.throughput_scalar
        * filt
        * config.delta_lambda_um
        * config.spaxel_solid_angle_arcsec2
        * config.sky_scaling
    )


def apply_telluric_absorption(array: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    """Attenuate the source by the telluric transmission along the wavelength axis.

    ``array`` is a spectrum ``(nlambda,)`` or a cube ``(ny, nx, nlambda)``;
    ``transmission`` is ``(nlambda,)`` in [0, 1].
    """
    transmission = np.clip(np.asarray(transmission, dtype=float), 0.0, 1.0)
    return array * _broadcast_spectral(array, transmission)


def correct_telluric(
    array: np.ndarray,
    transmission: np.ndarray,
    transmission_error: float = 0.0,
    floor: float = 0.02,
) -> np.ndarray:
    """Divide out an (imperfectly known) telluric transmission.

    The *assumed* transmission is ``transmission * (1 + transmission_error)``,
    clipped at ``floor`` so deep telluric troughs do not blow the correction up.
    A non-zero ``transmission_error`` leaves a residual that the science metrics
    then quantify.
    """
    transmission = np.asarray(transmission, dtype=float)
    assumed = np.clip(transmission * (1.0 + transmission_error), floor, None)
    return array / _broadcast_spectral(array, assumed)


def subtract_sky(data: np.ndarray, sky_estimate: np.ndarray, scale_error: float = 0.0) -> np.ndarray:
    """Subtract a (possibly mis-scaled) sky estimate from the data.

    ``scale_error`` models an imperfect sky-subtraction: the subtracted sky is
    ``sky_estimate * (1 + scale_error)``, so ``scale_error != 0`` leaves an
    over/under-subtraction residual.  ``sky_estimate`` broadcasts along the
    wavelength axis of ``data``.
    """
    return data - (1.0 + scale_error) * _broadcast_spectral(data, np.asarray(sky_estimate, dtype=float))


def sky_residual_metrics(
    residual: np.ndarray,
    sky_injected: np.ndarray,
    transmission: np.ndarray,
) -> dict:
    """Summarise the leftover sky and the telluric depth.

    * ``residual_sky_fraction`` -- ``sum|residual| / sum(injected sky)`` (0 = perfect
      subtraction; the absolute value captures both over- and under-subtraction);
    * ``telluric_min_transmission`` -- the deepest telluric trough in band.

    Works for a 1-D spectrum or a 3-D cube: the injected sky (which may be a 1-D
    per-spaxel spectrum) is broadcast to the *full* shape of ``residual`` before
    summing, so the numerator and denominator span the same number of voxels.
    """
    residual = np.asarray(residual, dtype=float)
    sky_spectral = _broadcast_spectral(residual, np.asarray(sky_injected, dtype=float))
    sky_full = np.broadcast_to(sky_spectral, residual.shape)
    sky_total = float(np.sum(np.abs(sky_full)))
    resid_total = float(np.sum(np.abs(residual)))
    frac = resid_total / sky_total if sky_total > 0 else float("nan")
    return {
        "residual_sky_fraction": frac,
        "telluric_min_transmission": float(np.min(transmission)),
    }


def _broadcast_spectral(array: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Broadcast a 1-D spectrum onto a spectrum/cube along the wavelength axis."""
    array = np.asarray(array)
    if array.ndim == 1:
        return spectrum
    if array.ndim == 3:
        return spectrum[None, None, :]
    raise ValueError(f"expected a (nlambda,) spectrum or (ny, nx, nlambda) cube, got ndim={array.ndim}")


# ---------------------------------------------------------------------------
# Sky / telluric time variability and residual realism (Priority 6).
# Controlled residual experiments, NOT a real telluric reduction.  The SkyCalc
# radiance/transmission stay direct_public_data; the variability/mismatch
# scalings are engineering_estimate.
# ---------------------------------------------------------------------------

def split_sky_continuum_lines(radiance: np.ndarray, smooth_size: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Separate a sky radiance spectrum into a smooth continuum and OH-line spikes.

    The continuum is a median-filtered version of the radiance (robust to the
    narrow OH lines); the lines are the positive residual above it.
    """
    from scipy.ndimage import median_filter

    radiance = np.asarray(radiance, dtype=float)
    continuum = median_filter(radiance, size=max(int(smooth_size), 3), mode="nearest")
    lines = np.clip(radiance - continuum, 0.0, None)
    return continuum, lines


def vary_sky_radiance(
    radiance: np.ndarray,
    continuum_scale: float = 1.0,
    oh_line_scale: float = 1.0,
    residual_frac: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply sky time-variability: scale the continuum and OH lines separately.

    ``continuum_scale`` and ``oh_line_scale`` independently scale the smooth and
    OH-line components; ``residual_frac`` adds a smooth seeded wavelength-dependent
    perturbation (fraction of the continuum).  Returns a non-negative radiance.
    All scalings are engineering estimates of night-to-night sky variability.
    """
    continuum, lines = split_sky_continuum_lines(radiance)
    varied = continuum * continuum_scale + lines * oh_line_scale
    if residual_frac and seed is not None:
        from scipy.ndimage import gaussian_filter1d

        rng = np.random.default_rng(seed)
        noise = gaussian_filter1d(rng.normal(0.0, 1.0, size=len(varied)), sigma=max(len(varied) // 50, 1))
        varied = varied + residual_frac * continuum * noise
    return np.clip(varied, 0.0, None)


def telluric_with_airmass(
    transmission: np.ndarray,
    airmass_ratio: float = 1.0,
    pwv_ratio: float = 1.0,
    floor: float = 1.0e-3,
) -> np.ndarray:
    """Scale telluric transmission for a different airmass / water-vapour column.

    Beer-Lambert: optical depth scales with airmass and precipitable water vapour,
    so ``T' = T ** (airmass_ratio * pwv_ratio)`` -- higher airmass/PWV deepens the
    troughs.  ``airmass_ratio = pwv_ratio = 1`` returns the input unchanged.
    Clipped to ``floor`` to avoid divide-by-zero downstream.
    """
    t = np.clip(np.asarray(transmission, dtype=float), floor, 1.0)
    return np.clip(t ** (float(airmass_ratio) * float(pwv_ratio)), floor, 1.0)


def sky_line_mask(radiance: np.ndarray, threshold_factor: float = 3.0) -> np.ndarray:
    """Boolean mask of wavelengths where sky radiance exceeds ``threshold_factor`` x median.

    Marks the bright OH-airglow regions that are usually masked in science
    spectra.  Lowering the threshold masks more wavelengths.
    """
    radiance = np.asarray(radiance, dtype=float)
    return radiance > threshold_factor * float(np.median(radiance))


def masked_fraction(mask: np.ndarray) -> float:
    """Fraction of wavelengths flagged by a sky-line mask."""
    return float(np.mean(np.asarray(mask, dtype=bool)))


def science_window_masked_fraction(
    wavelength_um: np.ndarray, mask: np.ndarray, line_center_um: float, window_um: float
) -> float:
    """Fraction of a science window (+/- ``window_um`` of the line) that is masked."""
    wl = np.asarray(wavelength_um, dtype=float)
    in_window = np.abs(wl - line_center_um) <= window_um
    if not np.any(in_window):
        return 0.0
    return float(np.mean(np.asarray(mask, dtype=bool)[in_window]))
