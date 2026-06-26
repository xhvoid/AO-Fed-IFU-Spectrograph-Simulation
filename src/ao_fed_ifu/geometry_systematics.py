"""Spectrograph / IFU geometry systematics (small, controlled, illustrative).

The base image-slicer map is perfectly rectangular and deterministic.  Real
spectrographs have small geometric imperfections; this module adds compact,
controlled proxies for the most common ones:

    * **trace curvature** -- the spectral trace bows in the cross-dispersion
      direction as a function of detector column (wavelength);
    * **polynomial wavelength solution** -- ``lambda(col)`` with small zero-point,
      stretch and curvature terms, instead of a pure line;
    * **slice-dependent LSF** -- the line-spread-function width varies slightly
      from slice to slice;
    * **flexure** -- a small, exposure-dependent trace offset.

All are illustrative ``synthetic_proxy`` / ``engineering_estimate`` constructions
with zero-amplitude defaults (so the nominal map is reproduced exactly).  This is
**not** an as-built slicer or optical model.
"""

from __future__ import annotations

import numpy as np


def trace_curvature_offsets(n_cols: int, amplitude_pix: float) -> np.ndarray:
    """Parabolic cross-dispersion row offset per detector column.

    The trace sits on its nominal row at the band centre and bows by
    ``amplitude_pix`` rows at the band edges:
    ``offset(col) = amplitude * (2*col/(n-1) - 1)^2``.  Returns integer row
    offsets (the detector grid is integer).  ``amplitude_pix = 0`` -> all zeros.
    """
    n = int(n_cols)
    if amplitude_pix == 0 or n <= 1:
        return np.zeros(n, dtype=int)
    x = 2.0 * np.arange(n) / (n - 1) - 1.0  # -1 .. +1 across the band
    return np.round(amplitude_pix * x**2).astype(int)


def polynomial_wavelength_solution(
    n_cols: int, lambda_min_um: float, lambda_max_um: float,
    stretch_frac: float = 0.0, curvature_frac: float = 0.0,
) -> np.ndarray:
    """Wavelength at each detector column for a (mildly) polynomial solution.

    Starts from the linear solution and adds a fractional linear *stretch* and a
    quadratic *curvature* term (both fractions of the band span):

        lambda(col) = lambda_min + span * [ u (1 + stretch) + curvature (u^2 - u) ]

    where ``u = col/(n-1)`` in [0, 1].  ``stretch=curvature=0`` reproduces the
    linear solution exactly.  Monotonic for small terms.
    """
    n = int(n_cols)
    u = np.arange(n) / max(n - 1, 1)
    span = lambda_max_um - lambda_min_um
    return lambda_min_um + span * (u * (1.0 + stretch_frac) + curvature_frac * (u**2 - u))


def invert_wavelength_solution(lambdas: np.ndarray, query_lambda_um) -> np.ndarray:
    """Fractional detector column for a wavelength, by interpolating ``lambda(col)``.

    ``lambdas`` is the monotonic per-column wavelength array; returns the
    (fractional) column index of ``query_lambda_um``.
    """
    cols = np.arange(len(lambdas))
    return np.interp(np.asarray(query_lambda_um, dtype=float), lambdas, cols)


def slice_lsf_fwhm_pix(n_slices: int, base_fwhm_pix: float, variation_frac: float = 0.0,
                       seed: int = 0) -> np.ndarray:
    """Per-slice LSF FWHM with a small deterministic-ish variation.

    Returns ``base_fwhm_pix * (1 + variation_frac * w_slice)`` where ``w_slice`` is
    a fixed, seeded pattern in [-1, 1] (deterministic for a given seed).
    ``variation_frac = 0`` -> all slices equal.
    """
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, size=int(n_slices))
    return base_fwhm_pix * (1.0 + variation_frac * w)


def recovered_R_from_lsf(lambda_um: float, fwhm_pix: float, dispersion_um_per_pix: float) -> float:
    """Resolving power R = lambda / (FWHM_pix * dispersion) implied by an LSF width.

    Wider LSF (more pixels) -> lower recovered R, the expected direction.
    """
    delta_lambda = fwhm_pix * dispersion_um_per_pix
    return float(lambda_um / delta_lambda) if delta_lambda > 0 else float("nan")


def flexure_offset_pix(amplitude_pix: float, seed: int = 0) -> tuple[int, int]:
    """A small seeded flexure trace offset ``(d_row, d_col)`` in integer pixels.

    Models the spectral trace drifting slightly between exposures.
    ``amplitude_pix = 0`` -> (0, 0).
    """
    if amplitude_pix == 0:
        return (0, 0)
    rng = np.random.default_rng(seed)
    drow = int(np.round(rng.uniform(-amplitude_pix, amplitude_pix)))
    dcol = int(np.round(rng.uniform(-amplitude_pix, amplitude_pix)))
    return (drow, dcol)
