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

The module also tracks **flux conservation** -- comparing the total electrons in
vs out (excluding bad pixels) -- which is the headline internal-validation check.
"""

from __future__ import annotations

import numpy as np

from .ifu_mapping import SlicerMapping


def reconstruct_cube_nearest(detector_e: np.ndarray, mapping: SlicerMapping) -> np.ndarray:
    """Nearest-pixel reconstruction: pure inverse look-up.

    For each spaxel, read the detector row it was projected onto; for each
    wavelength, read the column it was dispersed onto.  Vectorised with fancy
    indexing, so it is fast and fully deterministic.

    Returns a cube of shape ``(ny, nx, nlambda)``.
    """
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    det_row_grid = mapping.det_row_of_spaxel.reshape(ny, nx)
    cols = mapping.col_of_lambda
    # Broadcast (ny, nx, 1) detector rows against (1, 1, nlambda) columns.
    return detector_e[det_row_grid[:, :, None], cols[None, None, :]]


def reconstruct_cube_weighted(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> np.ndarray:
    """Inverse-variance optimal extraction across the cross-dispersion profile.

    Follows the Horne (1986) optimal-extraction estimator.  For each spaxel we
    assume a normalised cross-dispersion profile ``P_r`` (a Gaussian of width
    ``profile_sigma_pix``) spanning a small window of rows around the trace, and
    estimate the trace flux at each column as

        f = sum_r ( P_r * D_r / V_r ) / sum_r ( P_r^2 / V_r )

    where ``D_r`` and ``V_r`` are the data and variance in row ``r``.  This
    down-weights noisy/bad pixels (which carry ``V_r = inf``) and gathers flux
    that cross-talk has smeared across rows.

    When ``profile_sigma_pix`` is 0 the window collapses to a single row and the
    result equals :func:`reconstruct_cube_nearest`.

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

    # Window half-width: 3 sigma, at least 0 (single row) for sigma -> 0.
    half = int(np.ceil(3.0 * profile_sigma_pix)) if profile_sigma_pix > 0 else 0
    offsets = np.arange(-half, half + 1)
    if profile_sigma_pix > 0:
        profile = np.exp(-0.5 * (offsets / profile_sigma_pix) ** 2)
    else:
        profile = np.ones(1)
    profile = profile / profile.sum()

    cols = mapping.col_of_lambda
    for iy in range(ny):
        for ix in range(nx):
            sid = iy * nx + ix
            r0 = mapping.det_row_of_spaxel[sid]
            rows = np.clip(r0 + offsets, 0, n_rows - 1)
            D = detector_e[rows][:, cols]              # (nwin, nlambda)
            V = detector_variance[rows][:, cols]       # (nwin, nlambda)
            P = profile[:, None]                       # (nwin, 1)
            inv_v = 1.0 / V                            # inf-variance -> 0 weight
            denom = np.sum(P**2 * inv_v, axis=0)
            numer = np.sum(P * D * inv_v, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                flux = np.where(denom > 0, numer / denom, 0.0)
            # The Horne estimator already returns the *total* trace flux, which
            # for a single-row (cross-talk-free) trace equals the nearest-pixel
            # value -- so the two reconstructors share one flux normalisation.
            cube[iy, ix, :] = flux
    return cube


def estimate_variance_cube(
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
) -> np.ndarray:
    """Map the detector variance frame back to a per-voxel variance cube.

    Uses the same nearest look-up as :func:`reconstruct_cube_nearest`, so the
    returned variance cube is the propagated noise of the nearest reconstruction
    and can be turned into a per-voxel error or SNR.
    """
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    det_row_grid = mapping.det_row_of_spaxel.reshape(ny, nx)
    cols = mapping.col_of_lambda
    return detector_variance[det_row_grid[:, :, None], cols[None, None, :]]


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
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    det_row_grid = mapping.det_row_of_spaxel.reshape(ny, nx)
    cols = mapping.col_of_lambda
    return detector_bad_mask[det_row_grid[:, :, None], cols[None, None, :]]


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
