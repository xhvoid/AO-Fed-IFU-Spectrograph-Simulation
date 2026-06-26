"""A simple single-order spectrograph: wavelength solution, LSF and sampling.

A spectrograph disperses light so that wavelength maps to position along the
detector.  This module models that map in the simplest useful way:

    * a **linear wavelength solution** lambda(x) along the dispersion axis,
    * a Gaussian **line-spread function** (LSF) that blurs the spectrum to the
      instrument's spectral resolution,
    * the **resolving power** R = lambda / delta_lambda and the associated
      sampling (pixels per resolution element).

Key physical quantities
    R                         resolving power (dimensionless); larger R resolves
                              finer spectral detail.
    delta_lambda = lambda / R width of one resolution element (a spectral "FWHM"
                              the instrument can deliver).
    pixels_per_resel          how many detector pixels sample one resolution
                              element; >= 2 is required to avoid undersampling.

Everything here is a ``synthetic_proxy`` / ``literature_derived`` construction:
mode presets resemble published R and wavelength ranges, but no real grating or
optical design is reproduced.

References (anchors only -- see docs/references.md):
    * HARMONI overview, arXiv:2103.11215; MUSE, arXiv:astro-ph/0606329 --
      R and wavelength-range presets.
    * BlueMUSE pipeline design, arXiv:2209.06022 -- LSF / wavelength-calibration
      importance.
    * PypeIt, arXiv:2005.06505 and PyWiFeS, arXiv:1311.2666 -- trace / wavelength /
      extraction vocabulary; ESO ETC 2.0, arXiv:2407.02972 -- ETC thinking.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class SpectrographConfig:
    """Configuration of the single-order spectrograph.

    Parameters
    ----------
    lambda_min_um, lambda_max_um
        Wavelength range covered along the dispersion axis, in micrometres.
    resolving_power_R
        Target resolving power R = lambda / delta_lambda (mode preset).
    pixels_per_resolution_element
        Detector pixels per resolution element (the sampling).  Must be >= 2 for
        a Nyquist-sampled spectrum; presets typically use ~2.0-2.5.
    dispersion_axis
        Which detector axis carries wavelength; only ``"x"`` is implemented in
        this first milestone (dispersion along detector columns).
    lsf_fwhm_pix
        FWHM of the Gaussian LSF in pixels.  Defaults to
        ``pixels_per_resolution_element`` (the resolution element *is* the LSF
        FWHM by definition).
    """

    lambda_min_um: float = 1.95
    lambda_max_um: float = 2.45
    resolving_power_R: float = 3000.0
    pixels_per_resolution_element: float = 2.5
    dispersion_axis: str = "x"
    lsf_fwhm_pix: float | None = None

    def __post_init__(self) -> None:
        if self.dispersion_axis != "x":
            raise NotImplementedError(
                "Only dispersion_axis='x' is implemented in this milestone."
            )
        if self.lsf_fwhm_pix is None:
            self.lsf_fwhm_pix = float(self.pixels_per_resolution_element)

    # -- Wavelength sampling -------------------------------------------------

    @property
    def lambda_mid_um(self) -> float:
        """Band-centre wavelength in micrometres."""
        return 0.5 * (self.lambda_min_um + self.lambda_max_um)

    def delta_lambda_resel_um(self, lambda_um: float | None = None) -> float:
        """Width of one resolution element, ``delta_lambda = lambda / R`` (um).

        Evaluated at ``lambda_um`` (defaults to the band centre).
        """
        lam = self.lambda_mid_um if lambda_um is None else lambda_um
        return lam / self.resolving_power_R

    @property
    def n_detector_pixels(self) -> int:
        """Number of detector pixels along the dispersion axis.

        Chosen so the band is sampled at ``pixels_per_resolution_element`` pixels
        per resolution element:

            n_resel = (lambda_max - lambda_min) / delta_lambda(band centre)
            n_pix   = ceil(n_resel * pixels_per_resolution_element)

        Tying the detector width to R and the sampling keeps the model
        self-consistent: the recovered spectrum is neither over- nor
        under-resolved relative to the stated R.
        """
        n_resel = (self.lambda_max_um - self.lambda_min_um) / self.delta_lambda_resel_um()
        return int(np.ceil(n_resel * self.pixels_per_resolution_element))

    @property
    def dispersion_um_per_pix(self) -> float:
        """Linear dispersion: micrometres of wavelength per detector pixel."""
        return (self.lambda_max_um - self.lambda_min_um) / max(self.n_detector_pixels - 1, 1)

    def actual_pixels_per_resolution_element(self, lambda_um: float | None = None) -> float:
        """Realised sampling = delta_lambda_resel / dispersion-per-pixel."""
        return self.delta_lambda_resel_um(lambda_um) / self.dispersion_um_per_pix

    def is_well_sampled(self) -> bool:
        """True if the spectrum is sampled at >= 2 pixels per resolution element."""
        return self.actual_pixels_per_resolution_element() >= 2.0

    # -- Wavelength solution (linear, optionally perturbed) ------------------

    def wavelength_to_detector_x(self, lambda_um) -> np.ndarray:
        """Map wavelength(s) to fractional detector column index.

        Linear solution: ``x = (lambda - lambda_min) / dispersion``.
        Returns floats (callers round/interpolate as needed).
        """
        lambda_um = np.asarray(lambda_um, dtype=float)
        return (lambda_um - self.lambda_min_um) / self.dispersion_um_per_pix

    def detector_x_to_wavelength(self, x) -> np.ndarray:
        """Map detector column index(es) to wavelength in micrometres.

        Inverse of :meth:`wavelength_to_detector_x`:
        ``lambda = lambda_min + x * dispersion``.
        """
        x = np.asarray(x, dtype=float)
        return self.lambda_min_um + x * self.dispersion_um_per_pix

    def wavelength_grid_um(self) -> np.ndarray:
        """Wavelength at every detector column, shape ``(n_detector_pixels,)``."""
        return self.detector_x_to_wavelength(np.arange(self.n_detector_pixels))

    def with_wavelength_offset(self, delta_lambda_um: float) -> "SpectrographConfig":
        """Return a copy whose wavelength solution is shifted by a constant.

        This models a **wavelength-calibration error**: if reconstruction uses a
        solution offset by ``delta_lambda_um`` from the truth, every line appears
        Doppler-shifted by ``delta_v = c * delta_lambda / lambda``.  The
        validation suite uses exactly this to check the predicted velocity bias.

        The shift is applied by moving ``lambda_min`` (and ``lambda_max``) so the
        dispersion is unchanged -- a pure zero-point offset.
        """
        shifted = copy.deepcopy(self)
        shifted.lambda_min_um = self.lambda_min_um + delta_lambda_um
        shifted.lambda_max_um = self.lambda_max_um + delta_lambda_um
        return shifted

    def with_dispersion_stretch(self, fractional_stretch: float) -> "SpectrographConfig":
        """Return a copy whose dispersion is stretched by ``(1 + fractional_stretch)``.

        Models a first-order (linear) wavelength-scale error, the next term
        beyond a pure offset.  Useful for calibration-sensitivity tests where the
        velocity bias varies across the band.
        """
        stretched = copy.deepcopy(self)
        span = (self.lambda_max_um - self.lambda_min_um) * (1.0 + fractional_stretch)
        stretched.lambda_max_um = self.lambda_min_um + span
        return stretched

    # -- Line-spread function ------------------------------------------------

    @property
    def lsf_sigma_pix(self) -> float:
        """Gaussian LSF sigma in pixels (from the configured FWHM)."""
        return float(self.lsf_fwhm_pix) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def apply_lsf(self, array: np.ndarray, axis: int = -1) -> np.ndarray:
        """Convolve ``array`` with the Gaussian LSF along ``axis``.

        Used to blur a noiseless spectrum / detector image to the instrument's
        spectral resolution before detection.  Gaussian convolution conserves
        total flux along the convolved axis.
        """
        if self.lsf_sigma_pix <= 0:
            return np.asarray(array, dtype=float)
        return gaussian_filter1d(
            np.asarray(array, dtype=float), sigma=self.lsf_sigma_pix, axis=axis, mode="nearest"
        )


# Instrument-inspired presets (literature_derived: plausible mode parameters,
# NOT reproductions of any real instrument's as-built configuration).
SPECTROGRAPH_PRESETS: dict[str, SpectrographConfig] = {
    # Near-IR, AO-fed, moderate resolution -- HARMONI-*inspired* K-band-ish mode.
    "nir_R3000": SpectrographConfig(
        lambda_min_um=1.95, lambda_max_um=2.45, resolving_power_R=3000.0,
        pixels_per_resolution_element=2.5,
    ),
    # Visible, MUSE-*inspired* wide-field mode (single synthetic order here).
    "vis_R3000": SpectrographConfig(
        lambda_min_um=0.48, lambda_max_um=0.93, resolving_power_R=3000.0,
        pixels_per_resolution_element=2.5,
    ),
    # Higher-resolution NIR mode for line-kinematics demonstrations.
    "nir_R7000": SpectrographConfig(
        lambda_min_um=1.95, lambda_max_um=2.45, resolving_power_R=7000.0,
        pixels_per_resolution_element=2.5,
    ),
}
