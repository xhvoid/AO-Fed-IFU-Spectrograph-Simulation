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
    """
    sky_total = float(np.sum(np.abs(_broadcast_spectral(residual, np.asarray(sky_injected, dtype=float)))))
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
