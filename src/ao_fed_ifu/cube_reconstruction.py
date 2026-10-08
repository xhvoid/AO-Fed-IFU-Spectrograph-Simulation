"""Inverse mapping: rebuild the datacube from the detector frame.

Reconstruction is the inverse of :mod:`ao_fed_ifu.ifu_mapping`.  Because the
forward model gives every spaxel a unique detector row and every wavelength a
unique column, the inverse is mostly a look-up:

    cube[iy, ix, il] = detector[ det_row(spaxel), col(wavelength) ]

Two reconstructors are provided:

    ``reconstruct_cube_nearest``  -- read the single trace pixel (fast, exact for
                                     the noiseless, cross-talk-free case).
    ``reconstruct_cube_weighted`` -- joint inverse-variance least-squares
                                     extraction of ALL traces at once, per
                                     detector column, using the forward
                                     cross-talk kernel as the trace profile.

Why joint, not per-trace Horne
------------------------------
On an image slicer the traces of neighbouring spaxels occupy *adjacent*
detector rows, and cross-dispersion cross-talk spreads each trace over its
neighbours' rows.  A per-trace Horne (1986) window then reads every pixel
several times -- once for each trace whose window covers it -- so neighbouring
flux is re-used instead of shared out: for the default profile a uniform
field comes back amplified by ``1 / sum(P^2) ~ 2.02`` and its nominal S/N by
``1.4``.  The estimator here solves the full linear model of a detector column,

    d_r = sum_s A[r, s] f_s + noise,   A[r, s] = K(r - row_s),

for all trace fluxes ``f_s`` simultaneously by weighted least squares
(``f = (A^T W A)^-1 A^T W d``, ``W = 1/V``).  With the exact forward kernel
``K`` (:func:`ao_fed_ifu.ifu_mapping.cross_talk_kernel`) it inverts the
cross-talk blur exactly in the noiseless case (flux conservation ratio 1),
reduces to the nearest-pixel read-out when the profile width is zero, and
coincides with Horne's estimator for an isolated trace.  Its **matching
covariance** is ``(A^T W A)^-1``: the diagonal is the per-voxel variance and
the off-diagonal terms -- neighbouring extractions share detector pixels, so
their errors are correlated (usually anti-correlated after deblending) -- are
kept in banded form
(:class:`TraceExtractionModel`) so an aperture sum's variance can be
propagated exactly instead of adding spaxel variances as if independent.

Each reconstructor therefore has a **matching variance estimate**:
``estimate_variance_cube`` (nearest look-up) pairs with the nearest flux, and
``reconstruct_cube_weighted_with_variance`` / ``estimate_variance_cube_weighted``
carry the least-squares covariance diagonal for the weighted flux.  Mixing
the estimators (weighted flux with nearest variance) mis-states the noise.
The historical per-trace estimator is kept as
:func:`reconstruct_cube_horne_per_trace_with_variance` purely as a
reconstruction-limitation demonstrator.

The module also tracks **flux conservation** -- comparing the total electrons in
vs out (excluding bad pixels) -- which is the headline internal-validation check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, solveh_banded

from .ifu_mapping import SlicerMapping, cross_talk_kernel


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


# A voxel whose deblended variance exceeds this multiple of the variance its
# own observed taps would give an ISOLATED trace (Horne, 1/sum(K^2 w)) is
# reported as unmeasured (variance inf): separable traces cost O(1-10) in
# amplification, near-null-space members >> 1e6.  Scale-free, so bright
# (Poisson-dominated) voxels are never mistaken for degenerate ones.
UNMEASURED_VARIANCE_FACTOR = 1.0e3

# A coefficient is individually identifiable only when it has no component in
# a discarded null eigenmode.  Numerical eigensolvers can leak round-off-level
# amplitude between otherwise disconnected trace blocks; this multiplier sets
# the amplitude tolerance relative to ``n * eps`` before it is squared for the
# null-space projection (which is a sum of squared amplitudes).
NULL_AMPLITUDE_ROUNDOFF = 100.0


@dataclass
class TraceExtractionModel:
    """Banded normal matrices of the joint trace extraction, per detector column.

    Holds everything needed to propagate the extraction covariance without
    storing every inverse densely.  For a full-rank wavelength bin, the
    symmetric positive-definite normal matrix ``N = A^T W A`` is kept in the
    upper banded storage used by :func:`scipy.linalg.solveh_banded`.
    ``aperture_variance`` solves ``N x = u`` and returns
    ``u . x = u^T N^-1 u``.  A genuinely rank-deficient bin instead carries its
    Moore--Penrose covariance in ``covariance_overrides``; coefficients that
    participate in a null mode are marked unobserved and omitted from ``u``.
    Thus no artificial prior enters either the estimate or its covariance.

    Attributes
    ----------
    order
        Spaxel ids (``iy * nx + ix``) in detector-row order; ``normal_banded``
        rows/columns follow this order.
    bandwidth
        Number of super-diagonals kept (``2 x`` the kernel radius).
    normal_banded
        Array ``(bandwidth + 1, n_spaxel, nlambda)``; ``normal_banded[:, :, l]``
        is the upper-form banded ``N`` of wavelength bin ``l``.  A trace with
        no data support has its otherwise-zero row/column replaced by a scaled
        unit diagonal so full-rank trace blocks remain directly solvable.
    unobserved
        Boolean ``(ny, nx, nlambda)``: voxels with no usable estimate (infinite
        variance in the returned cube), including null-mode participants and
        extremely variance-amplified deblends.
    covariance_overrides
        Dense Moore--Penrose covariance, keyed by wavelength index, only for
        rank-deficient normal matrices that cannot use the banded SPD solve.
    """

    order: np.ndarray
    bandwidth: int
    normal_banded: np.ndarray
    unobserved: np.ndarray
    covariance_overrides: dict[int, np.ndarray]

    def _covariance_action(self, il: int, rhs: np.ndarray) -> np.ndarray:
        """Apply the channel covariance to one or more right-hand sides."""
        if il in self.covariance_overrides:
            return self.covariance_overrides[il] @ rhs
        return solveh_banded(self.normal_banded[:, :, il], rhs, lower=False)

    def aperture_variance(self, aperture_mask: np.ndarray) -> np.ndarray:
        """Exact variance of ``sum(cube[aperture])`` per wavelength bin.

        Unobserved (infinite-variance) voxels inside the aperture are dropped
        from the sum in that bin, matching the masked-voxel convention of the
        diagnostics; a bin with no observed aperture voxel returns ``inf``.
        """
        mask = np.asarray(aperture_mask, dtype=bool).ravel()
        n_spax = self.order.size
        if mask.size != n_spax:
            raise ValueError(f"aperture_mask has {mask.size} elements; expected {n_spax}")
        nlam = self.normal_banded.shape[2]
        u_all = mask[self.order].astype(float)  # in row order
        unobs = self.unobserved.reshape(n_spax, nlam)[self.order]
        var = np.empty(nlam)
        for il in range(nlam):
            u = np.where(unobs[:, il], 0.0, u_all)
            if not np.any(u):
                var[il] = np.inf
                continue
            x = self._covariance_action(il, u)
            var[il] = float(u @ x)
        return var

    def aperture_variance_diagonal(self, aperture_mask: np.ndarray) -> np.ndarray:
        """Same aperture sum with spaxels treated as independent (for comparison)."""
        mask = np.asarray(aperture_mask, dtype=bool).ravel()
        n_spax = self.order.size
        if mask.size != n_spax:
            raise ValueError(f"aperture_mask has {mask.size} elements; expected {n_spax}")
        nlam = self.normal_banded.shape[2]
        u_all = mask[self.order].astype(float)
        unobs = self.unobserved.reshape(n_spax, nlam)[self.order]
        var = np.empty(nlam)
        for il in range(nlam):
            u = np.where(unobs[:, il], 0.0, u_all)
            if not np.any(u):
                var[il] = np.inf
                continue
            inv = self._covariance_action(il, np.diag(u))
            var[il] = float(np.sum(np.diag(inv) * u))
        return var


def _joint_extraction(
    detector_e: np.ndarray | None,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float,
) -> tuple[np.ndarray, np.ndarray, TraceExtractionModel]:
    """Joint weighted least-squares extraction of all traces, per column.

    Returns ``(flux_cube, variance_cube, model)``; with ``detector_e=None`` the
    flux cube is all zeros. Variance and model depend on the usable-data mask,
    including non-finite detector values when a data frame is supplied.
    """
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    n_spax = ny * nx
    nlam = mapping.col_of_lambda.size
    n_rows = mapping.det_shape[0]
    cols = mapping.col_of_lambda
    row_shift = mapping.lambda_row_shift().astype(int)      # (nlam,)

    offsets, kernel = cross_talk_kernel(profile_sigma_pix)  # (2R+1,)
    radius = int(offsets.max())
    bandwidth = 2 * radius

    # Spaxels in detector-row order (rows are unique per spaxel).
    r0 = mapping.det_row_of_spaxel.astype(int)
    order = np.argsort(r0, kind="stable")
    r0_sorted = r0[order]

    # Pixel rows each ordered spaxel's kernel touches, per wavelength bin
    # (curvature shifts them per column); off-detector taps get zero weight.
    rows = r0_sorted[:, None, None] + offsets[None, :, None] + row_shift[None, None, :]
    valid = (rows >= 0) & (rows < n_rows)
    rows_c = np.clip(rows, 0, n_rows - 1)
    col_idx = np.broadcast_to(cols[None, None, :], rows.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / detector_variance[rows_c, col_idx]          # inf variance -> 0
    # A non-positive or non-finite variance is not a valid statistical weight;
    # treat it exactly like a masked detector pixel instead of making the Gram
    # matrix indefinite with a negative weight.
    usable = valid & np.isfinite(w) & (w > 0.0)
    d_pix = None
    if detector_e is not None:
        d_raw = detector_e[rows_c, col_idx]
        # Bad pixels are normally identified by V=inf, but accepting NaN-valued
        # detector masks as well avoids the undefined 0 * NaN operation in the
        # weighted RHS and makes the normal matrix reflect the actual data used.
        usable &= np.isfinite(d_raw)
        d_pix = np.where(usable, d_raw, 0.0)
    w = np.where(usable, w, 0.0)                               # (n_spax, 2R+1, nlam)

    # Banded normal matrix N[i, i+d] = sum_k K(k) K(k - delta_{i,d}) w[i, k]
    # with delta_{i,d} = row(i+d) - row(i) (>= d; zero coupling beyond 2R).
    normal = np.zeros((bandwidth + 1, n_spax, nlam))
    k_axis = offsets
    for d in range(bandwidth + 1):
        i_idx = np.arange(n_spax - d)
        delta = r0_sorted[i_idx + d] - r0_sorted[i_idx]        # (n_spax-d,)
        shifted = k_axis[None, :] - delta[:, None]              # (n_spax-d, 2R+1)
        in_range = np.abs(shifted) <= radius
        k_shift = np.where(in_range, kernel[np.clip(shifted + radius, 0, 2 * radius)], 0.0)
        coeff = kernel[None, :] * k_shift                       # (n_spax-d, 2R+1)
        # upper banded storage: ab[bandwidth - d, j] = N[j - d, j]
        normal[bandwidth - d, d:, :] = np.einsum("ik,ikl->il", coeff, w[i_idx])

    # Best-case (isolated-trace Horne) variance of each unknown from its own
    # observed taps, 1 / sum_k K(k)^2 w_k -- the scale-free reference for the
    # variance amplification the deblend costs (brightness cancels out).
    own_weight = np.einsum("k,ikl->il", kernel**2, w)           # (n_spax, nlam)
    with np.errstate(divide="ignore"):
        var_isolated = np.where(own_weight > 0.0, 1.0 / own_weight, np.inf)

    # A wholly unsupported design column has an exactly zero normal-matrix
    # row/column.  Replace only that decoupled zero diagonal by the column's
    # measurement scale: the corresponding coefficient is already marked
    # unobserved below, its RHS is zero, and observed trace blocks are unchanged.
    # This makes the common sigma=0 + bad-pixel case cheaply solvable without a
    # dense fallback.  Unlike the old global ridge/interpolation constraint, it
    # cannot influence any measured coefficient.
    no_information = own_weight <= 0.0
    diag = normal[bandwidth]                                    # (n_spax, nlam)
    col_scale = np.max(diag, axis=0)
    col_scale = np.where(col_scale > 0.0, col_scale, 1.0)
    diag[no_information] = np.broadcast_to(col_scale, diag.shape)[no_information]

    if d_pix is not None:
        rhs = np.einsum("k,ikl->il", kernel, w * d_pix)         # (n_spax, nlam)
    else:
        rhs = None

    flux_sorted = np.zeros((n_spax, nlam))
    var_sorted = np.empty((n_spax, nlam))
    nonidentifiable = np.zeros((n_spax, nlam), dtype=bool)
    covariance_overrides: dict[int, np.ndarray] = {}
    eye = np.eye(n_spax)
    for il in range(nlam):
        ab = normal[:, :, il]
        try:
            inv = solveh_banded(ab, eye, lower=False)
            if not np.all(np.isfinite(inv)) or np.any(np.diag(inv) <= 0.0):
                raise LinAlgError("invalid inverse from banded normal matrix")
        except LinAlgError:
            # A positive-semidefinite data normal can be exactly rank deficient
            # after masking.  Its Moore--Penrose inverse is both the covariance
            # of the identifiable least-squares projections and the minimum-
            # norm solve.  Mark every individual coefficient with a component
            # in a discarded null mode: its value is not unique and must never
            # be presented as a measured neighbour.
            dense = _banded_to_dense(ab, bandwidth)
            eigval, eigvec = np.linalg.eigh(dense)
            scale = float(np.max(np.abs(eigval))) if eigval.size else 0.0
            eig_tol = np.finfo(float).eps * max(n_spax, 1) * scale
            keep = eigval > eig_tol
            if np.any(keep):
                vk = eigvec[:, keep]
                inv = (vk / eigval[keep]) @ vk.T
            else:
                inv = np.zeros_like(dense)
            if np.any(~keep):
                null_projection = np.sum(eigvec[:, ~keep] ** 2, axis=1)
                amplitude_tol = (
                    NULL_AMPLITUDE_ROUNDOFF * np.finfo(float).eps * max(n_spax, 1)
                )
                nonidentifiable[:, il] = null_projection > amplitude_tol**2
            covariance_overrides[il] = inv
        var_sorted[:, il] = np.diag(inv)
        if rhs is not None:
            flux_sorted[:, il] = inv @ rhs[:, il]

    # Unmeasured: no observed tap at all; any coefficient that participates in
    # an exact numerical null mode; a deblending variance amplification above
    # UNMEASURED_VARIANCE_FACTOR relative to the trace's own isolated-Horne
    # variance; or a non-positive/non-finite covariance diagonal.  There is no
    # interpolation prior: identifiable estimates remain the unbiased data-only
    # least-squares solution, while unreliable/null-linked neighbours are masked.
    with np.errstate(invalid="ignore", divide="ignore"):
        unobserved_sorted = (
            ~np.isfinite(var_sorted) | (var_sorted <= 0.0)
            | no_information | nonidentifiable
            | (var_sorted > UNMEASURED_VARIANCE_FACTOR * var_isolated)
        )
    var_sorted[unobserved_sorted] = np.inf
    flux_sorted[unobserved_sorted] = 0.0

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n_spax)
    cube = flux_sorted[inv_order].reshape(ny, nx, nlam)
    var_cube = var_sorted[inv_order].reshape(ny, nx, nlam)
    unobserved = unobserved_sorted[inv_order].reshape(ny, nx, nlam)
    model = TraceExtractionModel(
        order=order, bandwidth=bandwidth, normal_banded=normal, unobserved=unobserved,
        covariance_overrides=covariance_overrides,
    )
    return cube, var_cube, model


def _banded_to_dense(ab: np.ndarray, bandwidth: int) -> np.ndarray:
    n = ab.shape[1]
    dense = np.zeros((n, n))
    for d in range(bandwidth + 1):
        for j in range(d, n):
            dense[j - d, j] = ab[bandwidth - d, j]
            dense[j, j - d] = ab[bandwidth - d, j]
    return dense


def reconstruct_cube_joint_with_model(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray, TraceExtractionModel]:
    """Joint least-squares extraction returning ``(flux, variance, model)``.

    The flux and variance cubes are those of
    :func:`reconstruct_cube_weighted_with_variance`; the
    :class:`TraceExtractionModel` additionally lets callers propagate the
    spatial covariance of the extraction (see
    :meth:`TraceExtractionModel.aperture_variance`).
    """
    if profile_sigma_pix is None:
        profile_sigma_pix = mapping.ifu.cross_talk_sigma_pix
    return _joint_extraction(detector_e, detector_variance, mapping, float(profile_sigma_pix))


def reconstruct_cube_weighted_with_variance(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint inverse-variance extraction returning a **matching** (flux, variance) pair.

    Solves every detector column's linear trace model for all spaxels at once
    (module docstring): ``f = (A^T W A)^-1 A^T W d`` with the forward
    cross-talk kernel of width ``profile_sigma_pix`` as the trace profile and
    ``W = 1/V``.  The variance is the diagonal of ``(A^T W A)^-1`` -- the exact
    propagation of the same linear estimator -- so SNRs built from this pair
    are internally consistent.  Neighbouring traces are deblended rather than
    double-counted; a uniform field is recovered with ratio 1, and an isolated
    trace reproduces Horne's optimal extraction.

    Bad pixels (``V = inf``) get zero weight.  A voxel the remaining pixels
    cannot constrain -- every contributing pixel bad, or a trace that two
    nearby bad rows leave degenerate with its neighbours, or one seen only
    through its faintest taps (variance amplification above
    ``UNMEASURED_VARIANCE_FACTOR`` relative to its own isolated-trace Horne
    variance) -- is reported as
    unmeasured: flux 0 and variance ``inf`` (SNR cleanly 0).  When
    ``profile_sigma_pix`` is 0 the model is the identity and both flux and
    variance equal the nearest-pixel reconstruction.  Off-detector kernel taps
    are simply absent from the model (the forward blur loses that light off
    the array), so an edge trace's flux is estimated from its observed rows
    without renormalising the profile.

    Parameters
    ----------
    profile_sigma_pix
        Width of the trace profile.  Defaults to the IFU's
        ``cross_talk_sigma_pix`` so the extraction inverts the modelled spread.
    """
    cube, var_cube, _ = reconstruct_cube_joint_with_model(
        detector_e, detector_variance, mapping, profile_sigma_pix
    )
    return cube, var_cube


def reconstruct_cube_horne_per_trace_with_variance(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    mapping: SlicerMapping,
    profile_sigma_pix: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace Horne extraction -- kept ONLY as a reconstruction-limitation demo.

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

    **Not a valid estimator for adjacent traces.**  The windows of neighbouring
    spaxels overlap on an image slicer, so every pixel is read by several
    traces and the flux of an extended source is amplified by ``1/sum(P^2)``
    (~2.02 for the default profile) with the nominal S/N inflated by ~1.4.
    The science reconstructors use the joint estimator instead; this function
    exists so the bias can be demonstrated (``run_reconstruction_limitations``).

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
    *,
    detector_e: np.ndarray | None = None,
) -> np.ndarray:
    """Propagated variance of the joint weighted extraction, ``diag((A^T W A)^-1)``.

    The covariance depends on the trace model, detector variance, and usable
    data mask, not on finite data values. With the same mask this is identical
    to the combined flux/variance path. If the detector data contain NaN/inf
    despite finite supplied variance, pass ``detector_e`` here to exclude those
    same pixels. Without it, only the variance frame defines usable samples.
    """
    if profile_sigma_pix is None:
        profile_sigma_pix = mapping.ifu.cross_talk_sigma_pix
    if detector_e is not None:
        detector_e = np.asarray(detector_e)
        if detector_e.shape != np.shape(detector_variance):
            raise ValueError("detector_e shape must match detector_variance")
        detector_variance = np.where(np.isfinite(detector_e), detector_variance, np.inf)
    _, var_cube, _ = _joint_extraction(None, detector_variance, mapping, float(profile_sigma_pix))
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

def mapping_with_trace_offset(mapping: SlicerMapping, row_offset: int) -> SlicerMapping:
    """A copy of ``mapping`` whose trace solution is wrong by ``row_offset`` rows.

    Every spaxel's detector row is shifted by ``row_offset`` (clipped to the
    array), i.e. an imperfect trace calibration.  Feeding this mapping to the
    joint weighted extraction models a trace error under the SAME estimator as
    the reference reduction, so a perturbation scan isolates the trace error
    instead of an estimator switch.
    """
    from dataclasses import replace

    n_rows = mapping.det_shape[0]
    rows = np.clip(mapping.det_row_of_spaxel.astype(int) + int(row_offset), 0, n_rows - 1)
    return replace(mapping, det_row_of_spaxel=rows)


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


def _spectral_proxy_samples(cube, aperture_mask, reference_cube=None,
                            line_free_mask=None, valid_mask=None):
    """Prepare centred samples without connecting across excluded channels."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("cube must have shape (ny, nx, nlambda)")
    ny, nx, nlam = cube.shape
    if aperture_mask is None:
        aperture_mask = np.ones((ny, nx), dtype=bool)
    aperture_mask = np.asarray(aperture_mask, dtype=bool)
    if aperture_mask.shape != (ny, nx):
        raise ValueError("aperture_mask shape must match the cube's spatial shape")
    residual = cube
    if reference_cube is not None:
        reference_cube = np.asarray(reference_cube, dtype=float)
        if reference_cube.shape != cube.shape:
            raise ValueError("reference_cube shape must match cube")
        residual = cube - reference_cube
    valid = np.isfinite(residual)
    if valid_mask is not None:
        valid &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), cube.shape)
    if line_free_mask is not None:
        line_free_mask = np.asarray(line_free_mask, dtype=bool)
        if line_free_mask.shape != (nlam,):
            raise ValueError("line_free_mask must have one entry per wavelength channel")
        valid &= line_free_mask[None, None, :]
    samples = residual[aperture_mask].copy()
    masks = valid[aperture_mask]
    indices = np.arange(nlam, dtype=float)
    for spec, mask in zip(samples, masks):
        if np.count_nonzero(mask) < 3:
            mask[:] = False
            continue
        if reference_cube is None and line_free_mask is not None:
            # Only explicitly line-free data determine this trend. Fitting a
            # trend through the emission line would leave correlated signal.
            coefficients = np.polyfit(indices[mask], spec[mask], 1)
            spec -= np.polyval(coefficients, indices)
        spec -= np.mean(spec[mask])
    return samples, masks


def _mean_spectral_lag_correlation(samples, masks, lag):
    corrs = []
    for spec, mask in zip(samples, masks):
        left = slice(None, -lag) if lag else slice(None)
        pair = mask[left] & mask[lag:]
        if np.count_nonzero(pair) < 3:
            continue
        a, b = spec[left][pair], spec[lag:][pair]
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if denom > 0:
            corrs.append(float(np.sum(a * b) / denom))
    return float(np.mean(corrs)) if corrs else float("nan")


def spectral_correlation_proxy(cube: np.ndarray, aperture_mask: np.ndarray | None = None,
                               *, valid_mask: np.ndarray | None = None) -> float:
    """Mean lag-1 SIGNAL/STRUCTURE correlation, not a noise-covariance estimate.

    Mean removal leaves emission-line and continuum shape intact. The LSF
    smooths that signal, but does not by itself correlate independent detector
    noise drawn afterwards. Use :func:`spectral_noise_correlation_proxy` with
    matched truth or explicitly line-free channels for a noise diagnostic.
    """
    samples, masks = _spectral_proxy_samples(cube, aperture_mask, valid_mask=valid_mask)
    return _mean_spectral_lag_correlation(samples, masks, 1)


def spectral_noise_correlation_proxy(
    cube: np.ndarray, aperture_mask: np.ndarray | None = None, *,
    reference_cube: np.ndarray | None = None,
    line_free_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Lag-1 residual/noise proxy with original channel adjacency preserved.

    Supply a noiseless reference through the SAME reduction, or explicitly
    line-free channels on which a linear continuum trend can be removed.
    ``valid_mask`` can exclude non-finite-variance zero placeholders. Only pairs
    valid at both original adjacent indices contribute; gaps are never joined.
    This remains a residual proxy, not a calibrated covariance/error budget.
    """
    if reference_cube is None and line_free_mask is None:
        raise ValueError("noise correlation needs reference_cube or line_free_mask")
    samples, masks = _spectral_proxy_samples(
        cube, aperture_mask, reference_cube, line_free_mask, valid_mask
    )
    return _mean_spectral_lag_correlation(samples, masks, 1)


def estimate_covariance_proxy(
    cube: np.ndarray, aperture_mask: np.ndarray | None = None, n_lag: int = 4, *,
    reference_cube: np.ndarray | None = None,
    line_free_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mean normalised spectral correlation out to ``n_lag`` original lags.

    Without a reference or line-free selection this measures SIGNAL STRUCTURE,
    not noise. With either, samples follow :func:`spectral_noise_correlation_proxy`.
    Lag zero is 1 for nonconstant usable samples; unsupported lags return NaN.
    Excluded channels never create artificial neighbours across a gap.
    """
    if not isinstance(n_lag, (int, np.integer)) or n_lag < 0:
        raise ValueError("n_lag must be a nonnegative integer")
    samples, masks = _spectral_proxy_samples(
        cube, aperture_mask, reference_cube, line_free_mask, valid_mask
    )
    return np.array([_mean_spectral_lag_correlation(samples, masks, lag)
                     for lag in range(n_lag + 1)])


def compare_reconstruction_methods(
    detector_e: np.ndarray,
    detector_variance: np.ndarray,
    detector_bad_mask: np.ndarray,
    mapping: SlicerMapping,
    input_cube: np.ndarray,
    trace_offset: int = 1,
) -> dict:
    """Reconstruct with several methods and return ``{method: (cube, metrics)}``.

    Methods: ``nearest``, ``weighted`` (joint least-squares deblend),
    ``horne_per_trace`` (the per-trace Horne window -- kept ONLY to show how it
    over-counts flux shared between adjacent traces), ``nearest_bad_pixel_mask``,
    ``trace_offset``.  Metrics per method include two complementary flux ratios:
    ``flux_conservation_ratio`` compares every input voxel with the returned
    cube, so an unmeasured voxel's zero placeholder honestly counts as lost;
    ``measured_flux_conservation_ratio`` excludes exactly the voxels whose
    matching variance is non-finite, and ``measured_voxel_fraction`` reports the
    retained coverage.  Each method also carries its **matching**
    ``variance_cube`` (nearest look-up for the nearest-based methods, the
    least-squares / Horne propagation for the weighted methods, and the
    wrong-row look-up for ``trace_offset``) so callers can build internally
    consistent SNRs per method.
    """
    var_nearest = estimate_variance_cube(detector_variance, mapping)
    weighted_cube, weighted_var = reconstruct_cube_weighted_with_variance(
        detector_e, detector_variance, mapping
    )
    horne_cube, horne_var = reconstruct_cube_horne_per_trace_with_variance(
        detector_e, detector_variance, mapping
    )
    methods = {
        "nearest": (reconstruct_cube_nearest(detector_e, mapping), var_nearest),
        "weighted": (weighted_cube, weighted_var),
        "horne_per_trace": (horne_cube, horne_var),
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
        unmeasured = ~np.isfinite(var_cube)
        fc_measured = flux_conservation(input_cube, cube, bad_voxels=unmeasured)
        structure_correlation = spectral_correlation_proxy(cube, valid_mask=~unmeasured)
        out[name] = {
            "cube": cube,
            "variance_cube": var_cube,
            "flux_conservation_ratio": fc["ratio"],
            "measured_flux_conservation_ratio": fc_measured["ratio"],
            "measured_voxel_fraction": float(np.mean(~unmeasured)),
            "all_finite": bool(np.all(np.isfinite(cube))),
            # Legacy name retained, explicitly structure rather than noise.
            "spectral_correlation_proxy": structure_correlation,
            "spectral_structure_correlation_proxy": structure_correlation,
        }
    return out
