"""Instrument-systematics sensitivity scans (controlled, internal trend checks).

Given a base pipeline configuration, this module perturbs one instrument /
calibration parameter at a time and measures how the recovered science products
move.  It is a *controlled perturbation sanity check*, **not** a real tolerance
budget: the perturbation amplitudes are engineering estimates and the instrument
internals are synthetic proxies.

Perturbation kinds (sky-subtraction and telluric-correction residuals are
demonstrated separately in ``examples/run_sky_telluric_residuals.py``):
    wavelength_offset      reconstruction-time wavelength zero-point error
    wavelength_stretch     reconstruction-time linear wavelength-scale error
    trace_row_offset       reading the wrong detector trace row
    cross_talk_sigma       cross-dispersion leakage width
    read_noise_e           detector read noise
    flat_field_sigma       flat-field response error
    bad_pixel_fraction     dead/hot pixel fraction

Reported per perturbation level: median velocity bias [km/s], median line-flux
bias fraction, median SNR-loss fraction, and recovered-R bias.

Every perturbation is measured against the reference with the SAME estimator
(the pipeline's ``science_extraction`` and its matching variance): a trace-row
error is applied as a shifted trace solution fed to that estimator, and the
wavelength zero-point / stretch errors evaluate all products -- recovered R
included -- on the perturbed grid the reduction would believe.  Switching
estimators between reference and perturbation would mostly measure the
estimator change, not the perturbation.  The cross-talk scan stays within the
regime where a deblend can still separate adjacent traces (sigma <~ 1.2 px);
voxels an estimator cannot measure are recorded as ``unmeasured_fraction``.
"""

from __future__ import annotations

import copy

import numpy as np

from . import cube_reconstruction as recon
from . import diagnostics as diag
from .detector_model import DetectorConfig
from .ifu_mapping import build_slicer_mapping, forward_project_cube

ALL_KINDS = [
    "wavelength_offset", "wavelength_stretch", "trace_row_offset",
    "cross_talk_sigma", "read_noise_e", "flat_field_sigma", "bad_pixel_fraction",
]

# Default perturbation-level grids (engineering_estimate amplitudes).
_DEFAULT_LEVELS = {
    "wavelength_offset": (1.0, 2.0, 3.0),     # in detector-pixel widths
    "wavelength_stretch": (0.0005, 0.001, 0.002),
    "trace_row_offset": (1, 2, 3),            # detector rows
    "cross_talk_sigma": (0.8, 1.0, 1.2),      # separable regime (see module docstring)
    "read_noise_e": (12.0, 24.0, 40.0),
    "flat_field_sigma": (0.02, 0.04, 0.08),
    "bad_pixel_fraction": (0.02, 0.05, 0.10),
}
_UNITS = {
    "wavelength_offset": "pix", "wavelength_stretch": "frac", "trace_row_offset": "rows",
    "cross_talk_sigma": "pix", "read_noise_e": "e-", "flat_field_sigma": "frac",
    "bad_pixel_fraction": "frac",
}


def _clone(cfg, **changes):
    """Deep-copy a PipelineConfig, applying detector-field or top-level overrides."""
    new = copy.deepcopy(cfg)
    det_changes = {k: changes.pop(k) for k in list(changes) if k in DetectorConfig.__dataclass_fields__}
    if det_changes:
        new.detector = DetectorConfig(**{**new.detector.__dict__, **det_changes})
    for k, v in changes.items():
        setattr(new, k, v)
    return new


def _aperture_biases(ref, p_lf, p_vel, p_snr) -> tuple[float, float, float]:
    """Median velocity / line-flux / SNR-loss biases over the reference aperture."""
    ap = ref.aperture_mask
    eps = 1e-30
    valid_v = ap & np.isfinite(p_vel) & np.isfinite(ref.velocity_map)
    v = (float(np.median(p_vel[valid_v] - ref.velocity_map[valid_v]))
         if np.any(valid_v) else float("nan"))
    lf = float(np.nanmedian((p_lf[ap] - ref.line_flux_map[ap]) / (ref.line_flux_map[ap] + eps)))
    sl = float(np.nanmedian((ref.snr_map[ap] - p_snr[ap]) / (ref.snr_map[ap] + eps)))
    return v, lf, sl


def _recovered_R(cube, grid, line_center, window) -> float:
    cy, cx = cube.shape[0] // 2, cube.shape[1] // 2
    return diag.resolving_power_proxy(cube[cy, cx, :], grid, line_center, window)


def _maps(cube, variance_cube, grid, lc, window):
    return (diag.line_flux_map(cube, grid, lc, window, variance_cube=variance_cube),
            diag.velocity_centroid_map(cube, grid, lc, window),
            diag.snr_map(cube, variance_cube, grid, lc, window))


def compute_systematics(base_config, n_levels: int = 3, kinds=None, levels=None) -> dict:
    """Run the perturbation scans.

    Returns ``{kind: {x, units, velocity, lineflux, snr, recovered_R_bias,
    [crosstalk_leakage]}}``, where each value list runs over the perturbation
    levels.  ``run_pipeline`` is imported lazily to avoid a circular import.
    """
    from .integration import run_pipeline

    kinds = list(kinds) if kinds is not None else list(ALL_KINDS)
    levels = levels or {}

    ref = run_pipeline(base_config)
    grid, lc = ref.wavelength_grid_um, ref.line_center_um
    window = 3.0 * ref.spectrograph.delta_lambda_resel_um(lc)
    ref_R = _recovered_R(ref.science_cube, grid, lc, window)
    mapping = build_slicer_mapping(ref.ifu_config, ref.spectrograph, grid)

    out: dict = {}
    for kind in kinds:
        lv = list(levels.get(kind, _DEFAULT_LEVELS[kind]))[:n_levels]
        vel, lf, sl, rbias, leak, unmeasured = [], [], [], [], [], []
        for x in lv:
            p_lf, p_vel, p_snr, p_cube, p_grid, p_unmeasured = _perturbed_maps(
                kind, x, ref, base_config, mapping, grid, lc, window, run_pipeline)
            b = _aperture_biases(ref, p_lf, p_vel, p_snr)
            vel.append(b[0]); lf.append(b[1]); sl.append(b[2])
            # recovered R on the grid the (perturbed) reduction believes
            rbias.append(_recovered_R(p_cube, p_grid, lc, window) - ref_R)
            unmeasured.append(p_unmeasured)
            if kind == "cross_talk_sigma":
                leak.append(_crosstalk_leakage(base_config, float(x)))
        entry = {"x": lv, "units": _UNITS[kind], "velocity": vel, "lineflux": lf,
                 "snr": sl, "recovered_R_bias": rbias, "unmeasured_fraction": unmeasured}
        if leak:
            entry["crosstalk_leakage"] = leak
        out[kind] = entry
    return out


def _unmeasured_fraction(variance_cube) -> float:
    return float(np.mean(~np.isfinite(np.asarray(variance_cube, dtype=float))))


def _perturbed_maps(kind, x, ref, base_config, mapping, grid, lc, window, run_pipeline):
    """Return ``(line_flux_map, velocity_map, snr_map, cube, grid_used,
    unmeasured_fraction)`` for one perturbation, always with the reference's
    science estimator so that only the perturbation differs from the reference."""
    if kind == "wavelength_offset":
        g = grid + float(x) * ref.spectrograph.dispersion_um_per_pix
        cube = ref.science_cube
        return (*_maps(cube, ref.variance_cube, g, lc, window), cube, g,
                _unmeasured_fraction(ref.variance_cube))
    if kind == "wavelength_stretch":
        g = grid[0] + (grid - grid[0]) * (1.0 + float(x))
        cube = ref.science_cube
        return (*_maps(cube, ref.variance_cube, g, lc, window), cube, g,
                _unmeasured_fraction(ref.variance_cube))
    if kind == "trace_row_offset":
        # A wrong trace solution: the SAME science estimator run with a mapping
        # whose traces sit ``x`` rows away from the truth, on the same noisy
        # frame and variance frame as the reference.
        shifted = recon.mapping_with_trace_offset(mapping, int(x))
        detector_science = ref.detector_calibrated
        if ref.metrics["sky_subtraction_applied"]:
            # Keep the baseline's optical sky model and scale error, but read
            # both data and model through the perturbed trace solution.  The
            # reconstruction is linear for the fixed variance weights, so
            # subtracting this model before extraction is equivalent to the
            # pipeline's post-extraction subtraction; its shot noise remains.
            sky_cube = np.broadcast_to(ref.sky_background_spectrum, ref.coupled_cube.shape)
            detector_sky = forward_project_cube(
                sky_cube, mapping, apply_lsf=False, apply_cross_talk=True
            )
            detector_science = detector_science - (
                1.0 + ref.config.sky_subtraction_scale_error
            ) * detector_sky
        cube, var = recon.reconstruct_cube_with_variance(
            detector_science, ref.variance_frame, shifted, method=ref.science_extraction
        )
        return (*_maps(cube, var, grid, lc, window), cube, grid, _unmeasured_fraction(var))
    # Remaining kinds are genuine pipeline re-runs with one changed parameter.
    changes = {kind: x}
    if kind == "flat_field_sigma":
        # A sensitivity scan of an UNCORRECTED response error, not known
        # detector nonuniformity that the nominal calibration removes.
        changes["correct_flat_field"] = False
    pert = run_pipeline(_clone(base_config, **changes))
    return (pert.line_flux_map, pert.velocity_map, pert.snr_map, pert.science_cube,
            pert.wavelength_grid_um, _unmeasured_fraction(pert.variance_cube))


def _crosstalk_leakage(base_config, sigma: float) -> float:
    """Leakage fraction out of an isolated spaxel at a given cross-talk sigma.

    The AO PSF is disabled here so the metric isolates *cross-dispersion
    cross-talk* (otherwise the PSF spreading a point source across spaxels would
    dominate and swamp the cross-talk trend).
    """
    from .integration import run_pipeline

    cfg = _clone(base_config, source_kind="point", cross_talk_sigma_pix=sigma,
                 use_public_cache=False, apply_psf=False)
    res = run_pipeline(cfg)
    cx, cy = cfg.nx // 2, cfg.ny // 2
    return diag.cross_talk_fraction(res.rebuilt_nearest, cx, cy)


def systematics_summary_rows(results: dict) -> list[dict]:
    """Flatten the scan into (kind, metric, value, source_class, notes) records.

    SNR response is reported as BOTH a signed ``snr_change_max_frac`` (the change
    with the largest magnitude; positive = loss, negative = a small SNR *gain*
    from favourable noise) and a non-negative ``snr_loss_max_frac`` (clamped at
    0).  A perturbation that leaves the SNR essentially unchanged can move it up
    a hair by chance; reporting only a "loss" would then show a misleading
    negative number, so the signed change is kept explicit and the loss is
    clamped.
    """
    rows = []
    for kind, e in results.items():
        snr = list(e["snr"])                       # signed loss fractions per level
        i_max = int(np.argmax(np.abs(snr)))        # largest-magnitude change (signed)
        snr_change_signed = float(snr[i_max])
        snr_loss_clamped = float(max(0.0, max(snr)))
        rows.append({"kind": kind, "metric": "velocity_bias_max_kms",
                     "value": float(np.max(np.abs(e["velocity"]))), "source_class": "internal_validation", "notes": ""})
        rows.append({"kind": kind, "metric": "lineflux_bias_max_frac",
                     "value": float(np.max(np.abs(e["lineflux"]))), "source_class": "internal_validation", "notes": ""})
        rows.append({"kind": kind, "metric": "snr_change_max_frac", "value": snr_change_signed,
                     "source_class": "internal_validation",
                     "notes": "signed SNR change (positive = loss, negative = small noise-driven gain)"})
        rows.append({"kind": kind, "metric": "snr_loss_max_frac", "value": snr_loss_clamped,
                     "source_class": "internal_validation",
                     "notes": "max SNR loss, clamped at 0 (a tiny SNR gain reports 0, not a negative loss)"})
        rows.append({"kind": kind, "metric": "recovered_R_bias_max",
                     "value": float(np.max(np.abs(e["recovered_R_bias"]))), "source_class": "internal_validation", "notes": ""})
        rows.append({"kind": kind, "metric": "assumed_amplitude_max",
                     "value": float(np.max(np.abs(e["x"]))), "source_class": "engineering_estimate", "notes": ""})
        if "crosstalk_leakage" in e:
            rows.append({"kind": kind, "metric": "crosstalk_leakage_max_frac",
                         "value": float(np.max(e["crosstalk_leakage"])), "source_class": "synthetic_proxy", "notes": ""})
        if "unmeasured_fraction" in e:
            rows.append({"kind": kind, "metric": "unmeasured_voxel_fraction_max",
                         "value": float(np.max(e["unmeasured_fraction"])), "source_class": "internal_validation",
                         "notes": "fraction of voxels the science extraction reports as unmeasured (inf variance)"})
    return rows
