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

    Returns ``(coadd, coadd_variance, rejected_fraction)``.  A voxel is rejected
    in an exposure if it sits more than ``sigma`` * (propagated noise) above/below
    the across-exposure median.  Using the *propagated* noise (sqrt of the
    variance) rather than the empirical scatter is robust to a single extreme
    cosmic-ray outlier (which would otherwise inflate the scatter and survive).
    Identical noiseless frames are all kept (zero deviation).

    Degenerate all-rejected voxels are rescued via the per-voxel **minimum**.
    With N=2 the median is the midpoint of the pair, so a cosmic ray makes *both*
    exposures deviate equally and both get clipped -- a naive fallback (keep the
    contaminated midpoint with infinite variance) would silently poison the
    coadd and zero that spaxel's SNR.  Because cosmic rays only ever ADD
    electrons, the minimum across exposures is the least-contaminated sample;
    it is used, with its own (finite) variance, wherever every exposure was
    clipped.
    """
    med = np.median(cubes, axis=0)
    noise = np.sqrt(np.clip(variances, 0.0, None))  # inf for bad pixels -> always kept
    keep = np.abs(cubes - med[None]) <= sigma * noise + 1e-9
    w = keep.astype(float) / np.clip(variances, 1e-30, None)  # inverse-variance, 0 if rejected
    wsum = np.sum(w, axis=0)

    # CR-robust fallback for voxels where every exposure was clipped (see above).
    min_idx = np.argmin(cubes, axis=0)
    min_val = np.take_along_axis(cubes, min_idx[None], axis=0)[0]
    min_var = np.take_along_axis(variances, min_idx[None], axis=0)[0]
    fallback = wsum <= 0
    if np.any(fallback):
        # Count the rescued minimum sample as kept, so rejected_fraction reflects
        # the samples actually excluded from the coadd.
        rescued = np.zeros_like(keep)
        np.put_along_axis(rescued, min_idx[None], True, axis=0)
        keep = keep | (rescued & fallback[None])

    with np.errstate(invalid="ignore", divide="ignore"):
        coadd = np.where(wsum > 0, np.sum(w * cubes, axis=0) / np.where(wsum > 0, wsum, 1.0), min_val)
        coadd_var = np.where(wsum > 0, 1.0 / np.where(wsum > 0, wsum, 1.0), min_var)
    rejected_fraction = float(1.0 - np.mean(keep))
    return coadd, coadd_var, rejected_fraction


def run_observing_strategy(base_config, plan: ExposurePlan) -> ObservingStrategyResult:
    """Run ``plan.n_exposures`` dithered exposures and coadd them."""
    from .integration import run_pipeline

    rng = np.random.default_rng(plan.seed)
    offsets = plan.offsets()
    cubes, variances = [], []
    reference = None

    for i, (dy, dx) in enumerate(offsets):
        cfg = _clone(base_config, seed=plan.seed + i + 1, source_dither_spaxels=(int(dy), int(dx)))
        res = run_pipeline(cfg)
        if reference is None:
            reference = res
        cube = (res.rebuilt_weighted if plan.reconstructor == "weighted" else res.rebuilt_nearest).astype(float)
        var = np.asarray(res.variance_cube, dtype=float).copy()

        if plan.sky_frames and res.sky_background_spectrum is not None:
            # Subtract a sky model (a sky "frame"); a controlled proxy, not a real
            # interleaved off-sky reduction.
            cube = cube - res.sky_background_spectrum[None, None, :]
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
