"""Multi-exposure observing strategy: dither, sky frames, coadd (a demonstrator).

Real near-IR IFU observing takes many short exposures, dithered on the sky, with
interleaved sky frames, then rejects cosmic rays and coadds with inverse-variance
weights.  This module reproduces that *structure* on top of the existing pipeline:

    for each exposure:  dither the source -> run the pipeline (fresh noise)
                        -> optionally subtract a sky model -> inject CR proxy
                        -> align (shift back by the known dither)
    coadd:              sigma-clip across exposures (reject CRs) -> inverse-variance
                        weighted mean, with 1/sum(1/var) coadd variance.

It is an **observing-strategy demonstrator**, not a real scheduler or reduction
pipeline.  The cosmic-ray injection/rejection is a ``synthetic_proxy``; the sky
model and noise come from the existing (proxy/engineering-estimate) pipeline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import diagnostics as diag


@dataclass
class ExposurePlan:
    """An observing plan: how many exposures, the dither pattern, sky frames, CRs."""

    n_exposures: int = 4
    dither_spaxels: list[tuple[int, int]] | None = None  # per-exposure integer (dy, dx)
    sky_frames: bool = True
    cosmic_ray_rate: float = 0.0      # fraction of voxels hit per exposure (0 = off)
    cosmic_ray_value: float = 1.0e6   # injected electron value of a CR hit
    sigma_clip: float = 5.0
    # Use the nearest reconstructor by default: it is consistent with the
    # nearest-based variance cube, so the sigma clip and flux accounting are clean
    # (the weighted/optimal reconstruction over-counts relative to that variance).
    reconstructor: str = "nearest"    # "nearest" | "weighted"
    align: bool = True                # shift exposures back by the known dither
    seed: int = 0

    def offsets(self) -> list[tuple[int, int]]:
        """Integer dither offsets, one per exposure (default small cycling pattern)."""
        if self.dither_spaxels is not None:
            base = list(self.dither_spaxels)
        else:
            base = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)]
        return [base[i % len(base)] for i in range(self.n_exposures)]


@dataclass
class ObservingStrategyResult:
    """Products of an observing-strategy run."""

    reference: object                 # a representative PipelineResult (truth, grids)
    exposure_cubes: np.ndarray        # (N, ny, nx, nlambda), aligned, sky-subtracted
    exposure_variances: np.ndarray    # (N, ny, nx, nlambda)
    offsets: list
    coadd_cube: np.ndarray
    coadd_variance: np.ndarray
    rejected_fraction: float
    finite_fraction: float
    extras: dict = field(default_factory=dict)


def _clone(cfg, **changes):
    new = copy.deepcopy(cfg)
    for k, v in changes.items():
        setattr(new, k, v)
    return new


def _sigma_clip_coadd(cubes: np.ndarray, variances: np.ndarray, sigma: float):
    """Reject cosmic rays vs the propagated noise, then inverse-variance coadd.

    Returns ``(coadd, coadd_variance, rejected_fraction)``.

    Cosmic rays only ever ADD charge, so the rejection is one-sided.  Each valid
    sample is compared with a reference built from the OTHER valid samples of
    the same voxel -- their median, in which every exposure counts once, so it is
    robust both to further cosmic rays (even ones hitting the most precise
    exposures, which would dominate a variance-weighted median) and to a stray
    low sample.  A sample is rejected only if it lies more than ``sigma`` times
    the COMBINED uncertainty ``sqrt(var_sample + var_reference)`` above that
    reference.  With unequal noise this matters twice over: a precise exposure
    is never rejected because a noisy one moved the reference, and a sample
    below its reference (a precise sample under a noisy one, or any sample under
    a cosmic ray) is never clipped.  The smallest valid sample therefore always
    survives, so every measured voxel keeps at least one observation without a
    special fallback; with two exposures a cosmic ray is assigned to the higher
    sample, where the extra charge is.  Using the *propagated* noise rather than
    the empirical scatter keeps a single extreme outlier from inflating its own
    threshold.  Identical noiseless frames are all kept (zero deviation).

    ``var_reference`` is the variance of the exposure chosen as the median, or a
    quarter of the summed variances of the two middle samples when there is an
    even number of them (their mean is the median).  This is conservative: a
    cosmic ray must exceed ~``sigma * sqrt(2)`` single-exposure sigmas in equal-
    noise data to be clipped.

    Missing samples -- the pipeline's zero-flux / infinite-variance placeholders,
    or any nonfinite flux or variance -- are not observations: they carry no
    weight in any reference, are never kept, and never count as rejected.  A
    voxel with no valid sample in any exposure stays unmeasured (flux 0,
    variance inf), distinct from a measured zero.  ``rejected_fraction`` counts
    clipped samples among the valid ones only.
    """
    valid = np.isfinite(cubes) & np.isfinite(variances) & (variances >= 0)
    n_exp = cubes.shape[0]
    weight = np.where(valid, 1.0 / np.clip(np.where(valid, variances, 1.0), 1e-30, None), 0.0)
    order = np.argsort(np.where(valid, cubes, np.inf), axis=0, kind="stable")
    x_sorted = np.take_along_axis(np.where(valid, cubes, 0.0), order, axis=0)
    v_sorted = np.take_along_axis(np.where(valid, variances, 0.0), order, axis=0)

    def take(values, index):
        return np.take_along_axis(values, index[None], axis=0)[0]

    keep = np.zeros_like(valid)
    valid_sorted = np.take_along_axis(valid, order, axis=0).astype(float)
    for i in range(n_exp):
        # Median of the other valid samples: the first sorted entry at which the
        # running count reaches half the count, or -- for an even count -- the
        # midpoint of that entry and the next valid one.
        counted = np.where(order == i, 0.0, valid_sorted)
        cumulative = np.cumsum(counted, axis=0)
        total = cumulative[-1]
        half = 0.5 * total
        lower = np.argmax(cumulative >= half[None], axis=0)
        upper = np.argmax(cumulative > half[None], axis=0)
        tie = take(cumulative, lower) == half
        ref = np.where(tie, 0.5 * (take(x_sorted, lower) + take(x_sorted, upper)), take(x_sorted, lower))
        ref_var = np.where(tie, 0.25 * (take(v_sorted, lower) + take(v_sorted, upper)), take(v_sorted, lower))
        has_reference = total > 0
        own_var = np.where(valid[i], variances[i], 0.0)
        with np.errstate(invalid="ignore"):
            limit = sigma * np.sqrt(own_var + np.where(has_reference, ref_var, 0.0)) + 1e-9
            keep[i] = valid[i] & (~has_reference | (cubes[i] - ref <= limit))

    w = np.where(keep, weight, 0.0)
    wsum = np.sum(w, axis=0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        weighted = np.sum(np.where(keep, w * np.where(valid, cubes, 0.0), 0.0), axis=0)
        coadd = np.where(wsum > 0, weighted / np.where(wsum > 0, wsum, 1.0), 0.0)
        coadd_var = np.where(wsum > 0, 1.0 / np.where(wsum > 0, wsum, 1.0), np.inf)
    n_valid = int(np.sum(valid))
    rejected_fraction = float(1.0 - np.sum(keep) / n_valid) if n_valid else 0.0
    return coadd, coadd_var, rejected_fraction


def run_observing_strategy(base_config, plan: ExposurePlan) -> ObservingStrategyResult:
    """Run ``plan.n_exposures`` dithered exposures and coadd them."""
    from .integration import run_pipeline

    rng = np.random.default_rng(plan.seed)
    offsets = plan.offsets()
    cubes, variances = [], []
    reference = None

    for i, (dy, dx) in enumerate(offsets):
        changes = dict(seed=plan.seed + i + 1, source_dither_spaxels=(int(dy), int(dx)))
        if plan.sky_frames:
            # Sky frames are subtracted in the SAME measurement coordinates as the
            # science: the pipeline forward-models the injected sky through the
            # LSF/cross-talk/edge path and reconstructs it with the selected
            # estimator.  Subtracting the raw injected spectrum from a
            # reconstructed cube would ignore those losses.
            changes["subtract_sky"] = True
        cfg = _clone(base_config, **changes)
        res = run_pipeline(cfg)
        if reference is None:
            reference = res
        # Pair the flux and variance from the SAME estimator (nearest look-up vs
        # Horne-propagated), so the coadd's inverse-variance weights and SNR are
        # internally consistent with the chosen reconstructor.
        if plan.reconstructor == "weighted":
            cube = res.rebuilt_weighted.astype(float)
            var = np.asarray(res.variance_cube_weighted, dtype=float).copy()
        else:
            cube = res.rebuilt_nearest.astype(float)
            var = np.asarray(res.variance_cube_nearest, dtype=float).copy()

        if plan.sky_frames:
            # Subtract a sky model (a sky "frame"); a controlled proxy, not a real
            # interleaved off-sky reduction.  The pipeline produces no
            # sky-subtracted product only when no sky was injected, in which
            # case there is nothing to subtract.
            skysub = (res.rebuilt_weighted_skysub if plan.reconstructor == "weighted"
                      else res.rebuilt_nearest_skysub)
            if skysub is not None:
                cube = skysub.astype(float)
        if plan.cosmic_ray_rate > 0:
            hit = rng.random(cube.shape) < plan.cosmic_ray_rate
            cube = cube.copy()
            cube[hit] += plan.cosmic_ray_value

        # Align: shift back by the known dither so all exposures register. When
        # alignment is off (for the before/after comparison) the dithered source
        # is left scattered across positions, so the coadd smears it.
        if plan.align:
            cube = np.roll(cube, shift=(-int(dy), -int(dx)), axis=(0, 1))
            var = np.roll(var, shift=(-int(dy), -int(dx)), axis=(0, 1))
        cubes.append(cube)
        variances.append(var)

    cubes = np.asarray(cubes)
    variances = np.asarray(variances)
    coadd, coadd_var, rejected = _sigma_clip_coadd(cubes, variances, plan.sigma_clip)
    finite_fraction = float(np.mean(np.isfinite(coadd)))
    return ObservingStrategyResult(
        reference=reference, exposure_cubes=cubes, exposure_variances=variances,
        offsets=offsets, coadd_cube=coadd, coadd_variance=coadd_var,
        rejected_fraction=rejected, finite_fraction=finite_fraction)


def coadd_first_k(result: ObservingStrategyResult, k: int, sigma: float = 5.0):
    """Coadd only the first ``k`` aligned exposures (for the SNR-vs-N curve)."""
    return _sigma_clip_coadd(result.exposure_cubes[:k], result.exposure_variances[:k], sigma)


def snr_vs_nexp(result: ObservingStrategyResult, line_center_um: float, window_um: float):
    """Median line SNR of the coadd as a function of number of exposures coadded.

    Returns ``(n_values, snr_values, sqrt_n_slope)`` where the slope is the fit of
    log(SNR) vs log(N) -- expected near 0.5 in the independent-noise regime.
    """
    grid = result.reference.wavelength_grid_um
    ap = result.reference.aperture_mask
    ns, snrs = [], []
    for k in range(1, result.exposure_cubes.shape[0] + 1):
        coadd, coadd_var, _ = coadd_first_k(result, k)
        snr = diag.snr_map(coadd, coadd_var, grid, line_center_um, window_um)
        ns.append(k)
        snrs.append(float(np.nanmedian(snr[ap])))
    ns = np.array(ns, dtype=float)
    snrs = np.array(snrs, dtype=float)
    good = np.isfinite(snrs) & (snrs > 0)
    slope = float(np.polyfit(np.log(ns[good]), np.log(snrs[good]), 1)[0]) if good.sum() >= 2 else float("nan")
    return ns, snrs, slope
