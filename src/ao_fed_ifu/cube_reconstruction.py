"""Inverse mapping: rebuild the datacube from the detector frame.

Reconstruction is the inverse of :mod:`ao_fed_ifu.ifu_mapping`.  Because the
forward model gives every spaxel a unique detector row and every wavelength a
unique column, the inverse is mostly a look-up:

    cube[iy, ix, il] = detector[ det_row(spaxel), col(wavelength) ]

Two reconstructors are provided:

    ``reconstruct_cube_nearest``  -- read the single trace pixel (fast, exact for
                                     the noiseless, cross-talk-free case).
    ``reconstruct_cube_weighted`` -- inverse-variance ("optimal", Horne 1986)
                                     extraction across a cross-dispersion window,
                                     which recovers light that cross-talk has
                                     spread into neighbouring rows.

Each reconstructor has a **matching variance estimate** so SNRs are internally
consistent: ``estimate_variance_cube`` (nearest look-up) pairs with the nearest
flux, and ``reconstruct_cube_weighted_with_variance`` /
``estimate_variance_cube_weighted`` propagate the Horne estimator's variance
``Var(f) = 1 / sum(P^2 / V)`` for the weighted flux.  Mixing the estimators
(weighted flux with nearest variance) understates the noise of the wider
extraction window and inflates SNR.

The module also tracks **flux conservation** -- comparing the total electrons in
vs out (excluding bad pixels) -- which is the headline internal-validation check.
"""

from __future__ import annotations

import numpy as np

from .ifu_mapping import SlicerMapping


def _voxel_row_indices(mapping: SlicerMapping) -> np.ndarray:
    """Detector row read by each cube voxel, shape ``(ny, nx, nlambda)``.

    Accounts for trace curvature (a per-column row shift); when the trace is
    straight (the default) this is just each spaxel's constant detector row
    broadcast over wavelength -- identical to the nominal nearest look-up.
    """
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    n_rows = mapping.det_shape[0]
    det_row_grid = mapping.det_row_of_spaxel.reshape(ny, nx)
    row_shift = mapping.lambda_row_shift()  # (nlambda,), zeros if straight
    return np.clip(det_row_grid[:, :, None] + row_shift[None, None, :], 0, n_rows - 1)


def reconstruct_cube_nearest(detector_e: np.ndarray, mapping: SlicerMapping) -> np.ndarray:
    """Nearest-pixel reconstruction: pure inverse look-up.

    For each spaxel, read the detector row it was projected onto; for each
    wavelength, read the column it was dispersed onto.  Vectorised with fancy
    indexing, so it is fast and fully deterministic.

    Returns a cube of shape ``(ny, nx, nlambda)``.
    """
    cols = mapping.col_of_lambda
    return detector_e[_voxel_row_indices(mapping), cols[None, None, :]]


def _extraction_profile(profile_sigma_pix: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised Gaussian cross-dispersion profile and its row offsets.

    Window half-width: 3 sigma, collapsing to a single row for sigma -> 0.
    """
    half = int(np.ceil(3.0 * profile_sigma_pix)) if profile_sigma_pix > 0 else 0
    offsets = np.arange(-half, half + 1)
    if profile_sigma_pix > 0:
        profile = np.exp(-0.5 * (offsets / profile_sigma_pix) ** 2)
    else:
        profile = np.ones(1)
    return offsets, profile / profile.sum()


def _on_detector_window(
    r0: int, offsets: np.ndarray, profile: np.ndarray, n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """On-detector rows of a trace window, with the profile subset renormalised.

    Near the top/bottom of the array some window rows fall past the detector
    edge.  Those rows carry no signal, so they are **discarded** -- not clipped
    onto the edge row.  Clipping would read the one edge pixel several times, as
    if it were several independent measurements, which duplicates its flux and
    (through ``sum(P^2/V)``) mis-states its variance for a boundary trace.

    The extraction profile is restricted to the surviving rows and renormalised
    to sum to 1, so the Horne estimator stays a proper inverse-variance
    weighting over the *observed* part of the trace.  ``r0`` is always a valid
    detector row (it is a spaxel's own trace row), so the central row always
    survives and the renormalisation never divides by zero.

    Returns ``(rows, profile_subset)`` where ``rows`` are the in-bounds detector
    rows and ``profile_subset`` is the matching renormalised profile.
    """
    raw = r0 + offsets
    valid = (raw >= 0) & (raw < n_rows)
    rows = raw[valid]
    p = profile[valid]
    return rows, p / p.sum()


def reconstruct_cube_weighted_with_variance(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Horne optimal extraction returning a **matching** (flux, variance) pair.

    Follows the Horne (1986) optimal-extraction estimator.  For each spaxel we
    assume a normalised cross-dispersion profile ``P_r`` (a Gaussian of width
    ``profile_sigma_pix``) spanning a small window of rows around the trace, and
    estimate the trace flux at each column as

        f      = sum_r ( P_r * D_r / V_r ) / sum_r ( P_r^2 / V_r )
        Var(f) = 1 / sum_r ( P_r^2 / V_r )

    where ``D_r`` and ``V_r`` are the data and variance in row ``r``.  The
    variance is the standard propagation of the same linear estimator, so SNRs
    built from this pair are internally consistent (using the nearest-pixel
    variance with the weighted flux would understate the noise of the wider
    extraction window and inflate SNR).

    Down-weights noisy/bad pixels (``V_r = inf`` -> zero weight); a column whose
    whole window is bad gets flux 0 and variance ``inf`` (SNR cleanly 0).

    When ``profile_sigma_pix`` is 0 the window collapses to a single row and
    both flux and variance equal the nearest-pixel reconstruction.

    Detector edges: window rows that fall past the top/bottom of the array are
    discarded and the profile is renormalised over the surviving rows (see
    :func:`_on_detector_window`), so a boundary trace measures the flux/variance
    of its *observed* rows under the profile model rather than re-reading the
    edge pixel.  Interior traces (whole window on-detector) are unchanged.

    Parameters
    ----------
    profile_sigma_pix
        Width of the assumed extraction profile.  Defaults to the IFU's
        ``cross_talk_sigma_pix`` so the extraction matches the modelled spread.
    """
    if profile_sigma_pix is None:
        profile_sigma_pix = mapping.ifu.cross_talk_sigma_pix

    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    nlam = mapping.col_of_lambda.size
    n_rows = mapping.det_shape[0]
    cube = np.zeros((ny, nx, nlam), dtype=float)
    var_cube = np.zeros((ny, nx, nlam), dtype=float)

    offsets, profile = _extraction_profile(profile_sigma_pix)

    cols = mapping.col_of_lambda
    for iy in range(ny):
        for ix in range(nx):
            sid = iy * nx + ix
            r0 = mapping.det_row_of_spaxel[sid]
            # Discard off-detector window rows (no edge-row duplication); the
            # profile subset is renormalised over the surviving rows.
            rows, p = _on_detector_window(r0, offsets, profile, n_rows)
            D = detector_e[rows][:, cols]              # (nwin, nlambda)
            V = detector_variance[rows][:, cols]       # (nwin, nlambda)
            P = p[:, None]                             # (nwin, 1)
            inv_v = 1.0 / V                            # inf-variance -> 0 weight
            denom = np.sum(P**2 * inv_v, axis=0)
            numer = np.sum(P * D * inv_v, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                flux = np.where(denom > 0, numer / denom, 0.0)
                var = np.where(denom > 0, 1.0 / denom, np.inf)
            # The Horne estimator already returns the *total* trace flux, which
            # for a single-row (cross-talk-free) trace equals the nearest-pixel
            # value -- so the two reconstructors share one flux normalisation.
            cube[iy, ix, :] = flux
            var_cube[iy, ix, :] = var
    return cube, var_cube


def reconstruct_cube_weighted(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> np.ndarray:
    """Inverse-variance optimal extraction (flux only).

    Convenience wrapper around :func:`reconstruct_cube_weighted_with_variance`;
    use that function when a consistent variance is needed for SNR work.
    """
    cube, _ = reconstruct_cube_weighted_with_variance(
        detector_e, detector_variance, mapping, profile_sigma_pix
    )
    return cube


def estimate_variance_cube(
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
) -> np.ndarray:
    """Map the detector variance frame back to a per-voxel variance cube.

    Uses the same nearest look-up as :func:`reconstruct_cube_nearest`, so the
    returned variance cube is the propagated noise of the **nearest**
    reconstruction and can be turned into a per-voxel error or SNR for that
    estimator.  For the weighted reconstruction use the matching variance from
    :func:`reconstruct_cube_weighted_with_variance` (or
    :func:`estimate_variance_cube_weighted`) instead -- pairing the nearest
    variance with the weighted flux mixes estimators and inflates SNR.
    """
    cols = mapping.col_of_lambda
    return detector_variance[_voxel_row_indices(mapping), cols[None, None, :]]


def estimate_variance_cube_weighted(
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> np.ndarray:
    """Propagated variance of the Horne weighted extraction, ``1 / sum(P^2/V)``.

    The variance of the weighted estimator does not depend on the data values,
    only on the profile and the detector variance frame, so it can be computed
    stand-alone.  Bit-for-bit identical to the variance returned by
    :func:`reconstruct_cube_weighted_with_variance` (both evaluate the
    denominator with the same ``sum(P^2 * (1/V))`` expression).
    """
    if profile_sigma_pix is None:
        profile_sigma_pix = mapping.ifu.cross_talk_sigma_pix

    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    nlam = mapping.col_of_lambda.size
    n_rows = mapping.det_shape[0]
    var_cube = np.zeros((ny, nx, nlam), dtype=float)

    offsets, profile = _extraction_profile(profile_sigma_pix)

    cols = mapping.col_of_lambda
    for iy in range(ny):
        for ix in range(nx):
            sid = iy * nx + ix
            r0 = mapping.det_row_of_spaxel[sid]
            # Same on-detector windowing AND the same denominator expression as
            # the flux path (inv_v = 1/V, then sum(P^2 * inv_v)), so the two
            # weighted-variance cubes are bit-for-bit identical, not just equal
            # to floating-point tolerance.
            rows, p = _on_detector_window(r0, offsets, profile, n_rows)
            V = detector_variance[rows][:, cols]       # (nwin, nlambda)
            P = p[:, None]                             # (nwin, 1)
            inv_v = 1.0 / V                            # inf-variance -> 0 weight
            denom = np.sum(P**2 * inv_v, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                var_cube[iy, ix, :] = np.where(denom > 0, 1.0 / denom, np.inf)
    return var_cube


def extract_1d_spectrum(cube: np.ndarray, aperture_mask: np.ndarray) -> np.ndarray:
    """Sum a cube spatially over a boolean aperture into a 1-D spectrum.

    Parameters
    ----------
    cube
        Datacube ``(ny, nx, nlambda)``.
    aperture_mask
        Boolean array ``(ny, nx)``; True spaxels are summed.

    Returns
    -------
    spectrum
        1-D array ``(nlambda,)`` of summed flux over the aperture.
    """
    aperture_mask = np.asarray(aperture_mask, dtype=bool)
    if aperture_mask.shape != cube.shape[:2]:
        raise ValueError("aperture_mask shape must match the cube's spatial shape")
    return cube[aperture_mask, :].sum(axis=0)


def bad_voxel_mask(detector_bad_mask: np.ndarray, mapping: SlicerMapping) -> np.ndarray:
    """Project the detector bad-pixel mask into cube (voxel) space.

    A voxel is "bad" if the detector pixel it reads from is bad.  Used to exclude
    those voxels from the flux-conservation accounting.
    """
    cols = mapping.col_of_lambda
    return detector_bad_mask[_voxel_row_indices(mapping), cols[None, None, :]]


def flux_conservation(
    input_cube: np.ndarray,
    reconstructed_cube: np.ndarray,
    bad_voxels: np.ndarray | None = None,
) -> dict:
    """Compare total electrons in vs out, excluding bad voxels.

    Returns a dict with ``input_total``, ``reconstructed_total`` and ``ratio =
    reconstructed/input``.  For a noiseless, cross-talk-free pass the ratio should
    be 1 to within floating-point tolerance -- the flux-conservation validation
    check asserts exactly that.

    Parameters
    ----------
    bad_voxels
        Optional boolean cube; True voxels are excluded from both totals so that
        masked bad pixels do not look like lost flux.
    """
    if bad_voxels is not None:
        good = ~np.asarray(bad_voxels, dtype=bool)
        in_total = float(np.sum(input_cube[good]))
        out_total = float(np.sum(reconstructed_cube[good]))
    else:
        in_total = float(np.sum(input_cube))
        out_total = float(np.sum(reconstructed_cube))
    ratio = out_total / in_total if in_total != 0 else float("nan")
    return {
        "input_total": in_total,
        "reconstructed_total": out_total,
        "ratio": ratio,
    }


# ----------------------------------------------------------------------------
# Reconstruction-limitation demonstrators (direction: "reconstruction limits").
# These show how reconstruction *choices* bias the result -- they are controlled
# illustrations, not a real reduction pipeline.
# ----------------------------------------------------------------------------

def reconstruct_cube_trace_offset(
    detector_e: np.ndarray, mapping: SlicerMapping, row_offset: int = 1
) -> np.ndarray:
    """Nearest reconstruction reading the *wrong* detector row (a trace error).

    Models an imperfect trace solution: every spaxel is read ``row_offset`` rows
    away from its true trace.  The recovered flux then comes partly from a
    neighbouring spaxel/slice, biasing line flux and (where the neighbour has a
    different velocity) the velocity centroid.
    """
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    n_rows = mapping.det_shape[0]
    det_row_grid = np.clip(mapping.det_row_of_spaxel.reshape(ny, nx) + int(row_offset), 0, n_rows - 1)
    cols = mapping.col_of_lambda
    return detector_e[det_row_grid[:, :, None], cols[None, None, :]]


def reconstruct_cube_with_bad_pixel_mask(
    detector_e: np.ndarray,
    detector_bad_mask: np.ndarray,
    mapping: SlicerMapping,
) -> np.ndarray:
    """Nearest reconstruction that repairs bad pixels by spectral interpolation.

    Where a spaxel's trace pixel is flagged bad, the value is linearly
    interpolated from the nearest good wavelength channels in the same trace row
    (rather than left as a 0 / NaN spike).  Guarantees a finite cube with no
    NaN/inf, so downstream diagnostics stay well-defined.
    """
    cube = reconstruct_cube_nearest(detector_e, mapping).astype(float)
    bad_cube = bad_voxel_mask(detector_bad_mask, mapping)
    nlam = cube.shape[2]
    lam_idx = np.arange(nlam)
    ny, nx = cube.shape[:2]
    for iy in range(ny):
        for ix in range(nx):
            bad = bad_cube[iy, ix, :]
            if not bad.any():
                continue
            good = ~bad
            if good.sum() >= 2:
                cube[iy, ix, bad] = np.interp(lam_idx[bad], lam_idx[good], cube[iy, ix, good])
            elif good.sum() == 1:
                cube[iy, ix, bad] = cube[iy, ix, good][0]
            else:
                cube[iy, ix, bad] = 0.0
    return cube


def spectral_correlation_proxy(cube: np.ndarray, aperture_mask: np.ndarray | None = None) -> float:
    """Mean lag-1 spectral autocorrelation -- an illustrative covariance proxy.

    The LSF and cross-talk correlate neighbouring wavelength channels, so adjacent
    reconstructed channels are not independent.  This returns the average
    correlation coefficient between channel ``l`` and ``l+1`` over the aperture
    (continuum-removed per spaxel).  It is **illustrative only**, not a calibrated
    covariance matrix.
    """
    cube = np.asarray(cube, dtype=float)
    ny, nx, nlam = cube.shape
    if aperture_mask is None:
        aperture_mask = np.ones((ny, nx), dtype=bool)
    spaxels = cube[np.asarray(aperture_mask, dtype=bool), :]
    corrs = []
    for spec in spaxels:
        s = spec - np.mean(spec)
        a, b = s[:-1], s[1:]
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if denom > 0:
            corrs.append(float(np.sum(a * b) / denom))
    return float(np.mean(corrs)) if corrs else float("nan")


def estimate_covariance_proxy(cube: np.ndarray, aperture_mask: np.ndarray | None = None, n_lag: int = 4) -> np.ndarray:
    """Average normalised spectral autocovariance out to ``n_lag`` lags.

    Returns a length-``n_lag+1`` array (lag 0 = 1.0); a slowly-decaying tail
    indicates channel-to-channel correlation from the LSF/cross-talk.  Illustrative
    only.
    """
    cube = np.asarray(cube, dtype=float)
    ny, nx, _ = cube.shape
    if aperture_mask is None:
        aperture_mask = np.ones((ny, nx), dtype=bool)
    spaxels = cube[np.asarray(aperture_mask, dtype=bool), :]
    acov = np.zeros(n_lag + 1)
    count = 0
    for spec in spaxels:
        s = spec - np.mean(spec)
        var = np.sum(s * s)
        if var <= 0:
            continue
        for lag in range(n_lag + 1):
            acov[lag] += np.sum(s[: len(s) - lag] * s[lag:]) / var
        count += 1
    return acov / count if count else acov


def compare_reconstruction_methods(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    detector_bad_mask: np.ndarray,
    mapping: SlicerMapping,
    input_cube: np.ndarray,
    trace_offset: int = 1,
) -> dict:
    """Reconstruct with several methods and return ``{method: (cube, metrics)}``.

    Methods: ``nearest``, ``weighted``, ``nearest_bad_pixel_mask``,
    ``trace_offset``.  Metrics per method: flux-conservation ratio (vs the input),
    finite-output flag, and the spectral-correlation proxy.  Each method also
    carries its **matching** ``variance_cube`` (nearest look-up for the
    nearest-based methods, Horne-propagated for ``weighted``, and the wrong-row
    look-up for ``trace_offset``) so callers can build internally consistent
    SNRs per method.
    """
    var_nearest = estimate_variance_cube(detector_variance, mapping)
    weighted_cube, weighted_var = reconstruct_cube_weighted_with_variance(
        detector_e, detector_variance, mapping
    )
    methods = {
        "nearest": (reconstruct_cube_nearest(detector_e, mapping), var_nearest),
        "weighted": (weighted_cube, weighted_var),
        # The repaired cube keeps the nearest variance: interpolated voxels
        # retain their (infinite) bad-pixel variance, so their SNR is cleanly 0.
        "nearest_bad_pixel_mask": (
            reconstruct_cube_with_bad_pixel_mask(detector_e, detector_bad_mask, mapping),
            var_nearest,
        ),
        # The wrong-row extraction reads the wrong row's noise too.
        "trace_offset": (
            reconstruct_cube_trace_offset(detector_e, mapping, trace_offset),
            reconstruct_cube_trace_offset(detector_variance, mapping, trace_offset),
        ),
    }
    out = {}
    for name, (cube, var_cube) in methods.items():
        fc = flux_conservation(input_cube, cube)
        out[name] = {
            "cube": cube,
            "variance_cube": var_cube,
            "flux_conservation_ratio": fc["ratio"],
            "all_finite": bool(np.all(np.isfinite(cube))),
            "spectral_correlation_proxy": spectral_correlation_proxy(cube),
        }
    return out
