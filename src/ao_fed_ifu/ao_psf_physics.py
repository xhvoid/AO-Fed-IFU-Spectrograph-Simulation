"""Wavelength-dependent AO PSF physics (a sharper proxy, still not a phase screen).

The base ``psf_coupling`` core+halo PSF takes a *fixed* Strehl at a *fixed*
wavelength.  Real AO behaves more interestingly with wavelength: the
diffraction-limited core grows as ``lambda / D`` while, for a fixed residual
wavefront error, the Strehl ratio *rises* toward longer wavelength.  This module
adds that wavelength dependence and exposes encircled-energy / sampling
diagnostics as first-class outputs.

It is still a **synthetic proxy / engineering estimate** -- there is no
phase-screen or AO-telemetry simulation.  The Strehl-vs-wavelength law is the
Marechal approximation

    S(lambda) = exp( -(2 pi sigma_wfe / lambda)^2 )

with ``sigma_wfe`` a residual-wavefront-error proxy in nm.  The core+halo PSF from
``psf_coupling`` is preserved and simply evaluated at the wavelength-dependent
Strehl (or at a fixed Strehl as a fallback).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .psf_coupling import (
    ao_corrected_psf_proxy,
    diffraction_fwhm_mas,
    psf_fwhm_mas,
    spaxelize_psf,
)


def strehl_from_wavefront_error(wavefront_error_nm: float, lambda_um: float) -> float:
    """Marechal Strehl: ``S = exp(-(2 pi sigma / lambda)^2)`` (sigma, lambda in same unit).

    Returns a Strehl in [0, 1] that increases with wavelength for a fixed residual
    wavefront error -- an ``engineering_estimate`` proxy, not an AO measurement.
    """
    sigma_um = float(wavefront_error_nm) * 1e-3
    phase_var = (2.0 * np.pi * sigma_um / float(lambda_um)) ** 2
    return float(np.clip(np.exp(-phase_var), 0.0, 1.0))


@dataclass
class AOPSFConfig:
    """Configuration for the wavelength-dependent AO PSF proxy.

    If ``wavefront_error_nm`` is set, the Strehl varies with wavelength via the
    Marechal law; otherwise ``fixed_strehl`` is used at every wavelength (the
    original fixed-Strehl fallback).
    """

    telescope_diameter_m: float = 8.0
    wavefront_error_nm: float | None = 150.0
    fixed_strehl: float = 0.3
    seeing_fwhm_mas: float = 600.0
    spaxel_scale_mas: float = 10.0
    fine_scale_mas: float = 2.0
    n_pix: int = 161

    def strehl_at(self, lambda_um: float) -> float:
        """Strehl at ``lambda_um`` (wavelength-dependent, or the fixed fallback)."""
        if self.wavefront_error_nm is None:
            return float(np.clip(self.fixed_strehl, 0.0, 1.0))
        return strehl_from_wavefront_error(self.wavefront_error_nm, lambda_um)


def ao_psf_at_wavelength(lambda_um: float, cfg: AOPSFConfig) -> dict:
    """Build the AO PSF at one wavelength and return its coupling diagnostics.

    Returns a dict with the Strehl, the diffraction-core FWHM (lambda/D), the
    measured PSF FWHM, the encircled energy in the central spaxel (``ee1_spaxel``)
    and central 3x3 block (``ee3x3_spaxel``) as a fraction of the *total* PSF
    energy, the in-field fraction (``ee_field`` = energy captured by the 7x7 spaxel
    field, i.e. the field throughput after PSF coupling), and the undersampling
    flag (PSF FWHM spans fewer than two spaxels).
    """
    strehl = cfg.strehl_at(lambda_um)
    psf = ao_corrected_psf_proxy(
        strehl, lambda_um, cfg.telescope_diameter_m, cfg.seeing_fwhm_mas,
        cfg.fine_scale_mas, cfg.n_pix,
    )  # normalised to unit total flux
    spaxel_psf = spaxelize_psf(psf, cfg.fine_scale_mas, cfg.spaxel_scale_mas, n_spaxel=7)
    c = spaxel_psf.shape[0] // 2
    fwhm = psf_fwhm_mas(psf, cfg.fine_scale_mas)
    return {
        "wavelength_um": float(lambda_um),
        "strehl": float(strehl),
        "diffraction_fwhm_mas": float(diffraction_fwhm_mas(lambda_um, cfg.telescope_diameter_m)),
        "psf_fwhm_mas": float(fwhm),
        "ee1_spaxel": float(spaxel_psf[c, c]),                                  # fraction of TOTAL PSF
        "ee3x3_spaxel": float(spaxel_psf[c - 1 : c + 2, c - 1 : c + 2].sum()),
        "ee_field": float(spaxel_psf.sum()),                                   # in-field fraction
        "undersampled": bool(np.isfinite(fwhm) and fwhm < 2.0 * cfg.spaxel_scale_mas),
    }


def ao_psf_wavelength_sweep(wavelengths_um, cfg: AOPSFConfig) -> list[dict]:
    """Evaluate :func:`ao_psf_at_wavelength` over a wavelength grid.

    Deterministic (no random numbers): returns one diagnostic dict per wavelength.
    """
    return [ao_psf_at_wavelength(float(w), cfg) for w in np.asarray(wavelengths_um, dtype=float)]
