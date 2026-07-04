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
"""

from __future__ import annotations

import copy

import numpy as np

from . import cube_reconstruction as recon
from . import diagnostics as diag
from .detector_model import DetectorConfig
from .ifu_mapping import build_slicer_mapping

ALL_KINDS = [
    "wavelength_offset", "wavelength_stretch", "trace_row_offset",
    "cross_talk_sigma", "read_noise_e", "flat_field_sigma", "bad_pixel_fraction",
]

# Default perturbation-level grids (engineering_estimate amplitudes).
_DEFAULT_LEVELS = {
    "wavelength_offset": (1.0, 2.0, 3.0),     # in detector-pixel widths
    "wavelength_stretch": (0.0005, 0.001, 0.002),
    "trace_row_offset": (1, 2, 3),            # detector rows
    "cross_talk_sigma": (0.8, 1.2, 1.6),
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
    v = float(np.nanmedian(p_vel[ap]) - np.nanmedian(ref.velocity_map[ap]))
    lf = float(np.nanmedian((p_lf[ap] - ref.line_flux_map[ap]) / (ref.line_flux_map[ap] + eps)))
    sl = float(np.nanmedian((ref.snr_map[ap] - p_snr[ap]) / (ref.snr_map[ap] + eps)))
    return v, lf, sl


def _recovered_R(cube, grid, line_center, window) -> float:
    cy, cx = cube.shape[0] // 2, cube.shape[1] // 2
    return diag.resolving_power_proxy(cube[cy, cx, :], grid, line_center, window)


def _maps(cube, variance_cube, grid, lc, window):
    return (diag.line_flux_map(cube, grid, lc, window),
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
    ref_R = _recovered_R(ref.rebuilt_weighted, grid, lc, window)
    mapping = build_slicer_mapping(ref.ifu_config, ref.spectrograph, grid)

    out: dict = {}
    for kind in kinds:
        lv = list(levels.get(kind, _DEFAULT_LEVELS[kind]))[:n_levels]
        vel, lf, sl, rbias, leak = [], [], [], [], []
        for x in lv:
            p_lf, p_vel, p_snr, p_cube = _perturbed_maps(
                kind, x, ref, base_config, mapping, grid, lc, window, run_pipeline)
            b = _aperture_biases(ref, p_lf, p_vel, p_snr)
            vel.append(b[0]); lf.append(b[1]); sl.append(b[2])
            rbias.append(_recovered_R(p_cube, grid, lc, window) - ref_R)
            if kind == "cross_talk_sigma":
                leak.append(_crosstalk_leakage(base_config, float(x)))
        entry = {"x": lv, "units": _UNITS[kind], "velocity": vel, "lineflux": lf,
                 "snr": sl, "recovered_R_bias": rbias}
        if leak:
            entry["crosstalk_leakage"] = leak
        out[kind] = entry
    return out


def _perturbed_maps(kind, x, ref, base_config, mapping, grid, lc, window, run_pipeline):
    """Return (line_flux_map, velocity_map, snr_map, cube) for one perturbation."""
    if kind == "wavelength_offset":
        g = grid + float(x) * ref.spectrograph.dispersion_um_per_pix
        cube = ref.rebuilt_weighted
        return (*_maps(cube, ref.variance_cube, g, lc, window), cube)
    if kind == "wavelength_stretch":
        g = grid[0] + (grid - grid[0]) * (1.0 + float(x))
        cube = ref.rebuilt_weighted
        return (*_maps(cube, ref.variance_cube, g, lc, window), cube)
    if kind == "trace_row_offset":
        # Nearest-style wrong-row extraction -> pair with the nearest variance
        # (ref.variance_cube is the weighted-estimator variance).
        cube = recon.reconstruct_cube_trace_offset(ref.detector_noisy, mapping, int(x))
        return (*_maps(cube, ref.variance_cube_nearest, grid, lc, window), cube)
    # Remaining kinds are genuine pipeline re-runs with one changed parameter.
    pert = run_pipeline(_clone(base_config, **{kind: x}))
    return pert.line_flux_map, pert.velocity_map, pert.snr_map, pert.rebuilt_weighted


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
    """Flatten the scan into (kind, metric, value, source_class) records (max level)."""
    rows = []
    for kind, e in results.items():
        rows.append({"kind": kind, "metric": "velocity_bias_max_kms",
                     "value": float(np.max(np.abs(e["velocity"]))), "source_class": "internal_validation"})
        rows.append({"kind": kind, "metric": "lineflux_bias_max_frac",
                     "value": float(np.max(np.abs(e["lineflux"]))), "source_class": "internal_validation"})
        rows.append({"kind": kind, "metric": "snr_loss_max_frac",
                     "value": float(np.max(e["snr"])), "source_class": "internal_validation"})
        rows.append({"kind": kind, "metric": "recovered_R_bias_max",
                     "value": float(np.max(np.abs(e["recovered_R_bias"]))), "source_class": "internal_validation"})
        rows.append({"kind": kind, "metric": "assumed_amplitude_max",
                     "value": float(np.max(np.abs(e["x"]))), "source_class": "engineering_estimate"})
        if "crosstalk_leakage" in e:
            rows.append({"kind": kind, "metric": "crosstalk_leakage_max_frac",
                         "value": float(np.max(e["crosstalk_leakage"])), "source_class": "synthetic_proxy"})
    return rows
