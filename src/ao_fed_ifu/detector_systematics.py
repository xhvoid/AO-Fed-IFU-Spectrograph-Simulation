"""Engineering-like IR-detector systematics (compact, testable proxies).

A small set of HAWAII-like effects, each a standalone function so it can be
tested and visualised on its own:

    * up-the-ramp / Fowler **sampling** -- effective read noise drops with the
      number of non-destructive reads (engineering_estimate scaling formulas);
    * **nonlinearity** -- a signal-dependent compression of the response near the
      full well, with an approximate correction (engineering_estimate);
    * **correlated read noise** -- a row/column/frame common-mode, plus a
      reference-pixel-style common-mode removal (synthetic_proxy);
    * **inter-pixel capacitance (IPC)** -- a small 3x3 charge-sharing convolution
      (synthetic_proxy).

These are labelled proxies / engineering estimates inspired by the detector
literature (Rauscher arXiv:1509.06264; MOSFIRE arXiv:1208.0314; JWST/NIRSpec
arXiv:2208.12686; GPI arXiv:1407.2302) -- **not** a measured device model.
"""

from __future__ import annotations

import numpy as np


# --- Up-the-ramp / Fowler sampling -----------------------------------------

def effective_read_noise(read_noise_e: float, n_reads: int, mode: str = "fowler") -> float:
    """Effective read noise after ``n_reads`` non-destructive reads.

    * ``mode="fowler"``  -- averaging N reads reduces white read noise as
      ``sigma / sqrt(N)`` (Fowler-N / MCDS).
    * ``mode="up_the_ramp"`` -- the slope-fit (SUTR) read noise scales as
      ``sigma * sqrt(12 (N-1) / (N (N+1)))`` for N>=2.

    ``n_reads <= 1`` returns ``read_noise_e`` unchanged (single read).  Both are
    engineering-estimate scalings, not a calibrated detector measurement.
    """
    n = max(int(n_reads), 1)
    if n == 1:
        return float(read_noise_e)
    if mode == "fowler":
        return float(read_noise_e / np.sqrt(n))
    if mode == "up_the_ramp":
        return float(read_noise_e * np.sqrt(12.0 * (n - 1) / (n * (n + 1))))
    raise ValueError(f"unknown sampling mode {mode!r}; expected 'fowler' or 'up_the_ramp'")


# --- Nonlinearity ----------------------------------------------------------

def apply_nonlinearity(signal_e: np.ndarray, full_well_e: float, coeff: float = 0.05) -> np.ndarray:
    """Signal-dependent compression: ``measured = s (1 - coeff * s / full_well)``.

    High-count pixels are compressed more than low-count pixels (the fractional
    loss grows linearly with signal).  ``coeff`` is the fractional nonlinearity at
    the full well (engineering estimate).
    """
    s = np.clip(np.asarray(signal_e, dtype=float), 0.0, None)
    return s * (1.0 - coeff * s / float(full_well_e))


def correct_nonlinearity(measured_e: np.ndarray, full_well_e: float, coeff: float = 0.05) -> np.ndarray:
    """Approximate inverse of :func:`apply_nonlinearity` (solve the quadratic)."""
    a = float(coeff) / float(full_well_e)
    m = np.clip(np.asarray(measured_e, dtype=float), 0.0, None)
    if a <= 0:
        return m
    disc = np.clip(1.0 - 4.0 * a * m, 0.0, None)
    return (1.0 - np.sqrt(disc)) / (2.0 * a)


# --- Correlated read noise + reference-pixel removal -----------------------

def apply_correlated_read_noise(frame: np.ndarray, sigma_e: float, rng, mode: str = "row") -> np.ndarray:
    """Add a common-mode read-noise offset per row / column / frame.

    ``mode="row"`` adds one offset per detector row (the dominant HxRG pattern),
    ``"column"`` per column, ``"frame"`` a single picture-frame offset.
    """
    frame = np.asarray(frame, dtype=float)
    if mode == "row":
        offset = rng.normal(0.0, sigma_e, size=(frame.shape[0], 1))
    elif mode == "column":
        offset = rng.normal(0.0, sigma_e, size=(1, frame.shape[1]))
    elif mode == "frame":
        offset = float(rng.normal(0.0, sigma_e))
    else:
        raise ValueError(f"unknown correlated-noise mode {mode!r}")
    return frame + offset


def subtract_reference_pixels(frame: np.ndarray, n_ref: int = 2) -> np.ndarray:
    """Reference-pixel-style per-row common-mode removal.

    Real IR arrays carry non-illuminated border pixels that measure only the
    electronics offset.  Here the per-row offset is estimated as the median of the
    first/last ``n_ref`` columns (assumed source-free) and subtracted from each
    row -- a proxy for the reference-pixel correction.
    """
    frame = np.asarray(frame, dtype=float)
    ref = np.concatenate([frame[:, :n_ref], frame[:, -n_ref:]], axis=1)
    row_offset = np.median(ref, axis=1, keepdims=True)
    return frame - row_offset


def row_common_mode_metric(frame: np.ndarray) -> float:
    """RMS of the per-row means -- a measure of residual row common-mode."""
    return float(np.std(np.mean(np.asarray(frame, dtype=float), axis=1)))


# --- Inter-pixel capacitance (IPC) -----------------------------------------

def apply_ipc(frame: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Inter-pixel capacitance: a 3x3 charge-sharing convolution.

    Each pixel leaks a fraction ``alpha`` of its signal to each of its four
    nearest neighbours (a small, flux-conserving kernel).  Synthetic proxy.
    """
    from scipy.ndimage import convolve

    a = float(alpha)
    kernel = np.array([[0.0, a, 0.0], [a, 1.0 - 4.0 * a, a], [0.0, a, 0.0]])
    return convolve(np.asarray(frame, dtype=float), kernel, mode="nearest")
