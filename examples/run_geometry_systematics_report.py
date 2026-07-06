#!/usr/bin/env python3
"""Spectrograph / IFU geometry systematics report (illustrative).

Shows small, controlled departures from the perfectly rectangular slicer: trace
curvature on the detector, a polynomial wavelength-solution residual, a
slice-dependent LSF (and the recovered-R variation it implies), and a flexure
trace offset's effect on line flux and velocity.

Illustrative `synthetic_proxy` geometry, **not** an as-built slicer or optical
model.

Run:
    python examples/run_geometry_systematics_report.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S  # noqa: E402
from ao_fed_ifu import cube_reconstruction as recon  # noqa: E402
from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu import geometry_systematics as geo  # noqa: E402
from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube  # noqa: E402
from ao_fed_ifu.integration import make_toy_config, run_pipeline  # noqa: E402
from ao_fed_ifu.source_cube import (  # noqa: E402
    SourceCubeConfig, create_emission_line_cube, scale_cube_to_total_electrons,
)
from ao_fed_ifu.spectrograph_model import SpectrographConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_geometry_systematics_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    curvature_pix = 4.0

    # A small cube + mapping to visualise curved traces and the curvature bias.
    sp = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=1200.0 if quick else 1800.0)
    cfg = SourceCubeConfig(nx=8, ny=6, nlambda=sp.n_detector_pixels, lambda_min_um=2.0, lambda_max_um=2.3)
    cube = scale_cube_to_total_electrons(create_emission_line_cube(cfg, line_flux_rel=4.0, continuum_level=1.0), 1.0e5)
    mc = build_slicer_mapping(IFUConfig(nx_spaxel=8, ny_spaxel=6, trace_curvature_pix=curvature_pix), sp, cfg.wavelength_grid_um)
    frame = forward_project_cube(cube, mc, apply_lsf=False, apply_cross_talk=False)

    # --- Figure 1: trace geometry map. ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    offsets = mc.row_offsets()
    ax[0].plot(offsets); ax[0].set_xlabel("detector column"); ax[0].set_ylabel("trace row offset [pix]")
    ax[0].set_title(f"Trace curvature (amplitude {curvature_pix:.0f} pix)"); ax[0].grid(alpha=0.3)
    im = ax[1].imshow(frame, aspect="auto", origin="lower", cmap="magma")
    ax[1].set_title("Curved traces on the detector"); ax[1].set_xlabel("det_x"); ax[1].set_ylabel("det_y")
    fig.colorbar(im, ax=ax[1])
    p1 = _save(fig, outdir / "trace_geometry_map.png")

    # --- Figure 2: polynomial wavelength-solution residuals. ---
    n_cols = sp.n_detector_pixels
    linear = geo.polynomial_wavelength_solution(n_cols, 2.0, 2.3, 0.0, 0.0)
    poly = geo.polynomial_wavelength_solution(n_cols, 2.0, 2.3, stretch_frac=0.002, curvature_frac=0.006)
    resid_um = poly - linear
    vel_resid = SPEED_OF_LIGHT_KM_S * resid_um / linear
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(linear, resid_um * 1e4); ax[0].set_xlabel("wavelength [um]"); ax[0].set_ylabel("residual [1e-4 um]")
    ax[0].set_title("Polynomial - linear wavelength solution"); ax[0].grid(alpha=0.3)
    ax[1].plot(linear, vel_resid, color="C1"); ax[1].set_xlabel("wavelength [um]"); ax[1].set_ylabel("velocity residual [km/s]")
    ax[1].set_title("Implied velocity residual"); ax[1].grid(alpha=0.3)
    p2 = _save(fig, outdir / "wavelength_solution_residuals.png")

    # --- Figure 3: slice-dependent LSF -> recovered R. ---
    n_slices = cfg.ny
    fwhm = geo.slice_lsf_fwhm_pix(n_slices, base_fwhm_pix=2.5, variation_frac=0.25, seed=0)
    R = [geo.recovered_R_from_lsf(sp.lambda_mid_um, f, sp.dispersion_um_per_pix) for f in fwhm]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].bar(range(n_slices), fwhm, color="C0"); ax[0].set_xlabel("slice"); ax[0].set_ylabel("LSF FWHM [pix]")
    ax[0].set_title("Slice-dependent LSF width")
    ax[1].plot(range(n_slices), R, "o-", color="C2"); ax[1].set_xlabel("slice"); ax[1].set_ylabel("recovered R")
    ax[1].set_title("Recovered R per slice (wider LSF -> lower R)"); ax[1].grid(alpha=0.3)
    p3 = _save(fig, outdir / "slice_lsf_variation.png")

    # --- Flexure: reconstruct the curved detector with the nominal (straight) map. ---
    straight_read = dataclasses.replace(mc, col_row_offset=np.zeros_like(mc.row_offsets()))
    fc_corrected = recon.flux_conservation(cube, recon.reconstruct_cube_nearest(frame, mc))["ratio"]
    fc_uncorrected = recon.flux_conservation(cube, recon.reconstruct_cube_nearest(frame, straight_read))["ratio"]

    # Flexure trace offset's effect on a toy line (reuse the trace-offset reconstructor).
    res = run_pipeline(make_toy_config(nx=8, ny=8, resolving_power_R=1500.0))
    grid, lc = res.wavelength_grid_um, res.line_center_um
    window = 3.0 * res.spectrograph.delta_lambda_resel_um(lc)
    tmap = build_slicer_mapping(res.ifu_config, res.spectrograph, grid)
    drow, _dcol = geo.flexure_offset_pix(2.0, seed=3)
    nominal = res.rebuilt_nearest
    flexed = recon.reconstruct_cube_trace_offset(res.detector_noisy, tmap, drow)
    ap = res.aperture_mask
    nlf = diag.line_flux_map(nominal, grid, lc, window); flf = diag.line_flux_map(flexed, grid, lc, window)
    nvel = diag.velocity_centroid_map(nominal, grid, lc, window); fvel = diag.velocity_centroid_map(flexed, grid, lc, window)
    flex_lf_bias = float(np.nanmedian((flf[ap] - nlf[ap]) / (nlf[ap] + 1e-30)))
    flex_vel_bias = float(np.nanmedian((fvel - nvel)[ap]))

    rows = [
        rep.metric_row("trace_curvature_pix", curvature_pix, "engineering_estimate", "trace curvature"),
        rep.metric_row("max_trace_row_offset_pix", float(np.max(np.abs(offsets))), "synthetic_proxy", "trace curvature"),
        rep.metric_row("curvature_flux_ratio_corrected", fc_corrected, "internal_validation", "curve-aware recon", "~1"),
        rep.metric_row("curvature_flux_ratio_uncorrected", fc_uncorrected, "internal_validation", "nominal recon", "<1 (bias)"),
        rep.metric_row("wavelength_residual_max_um", float(np.max(np.abs(resid_um))), "synthetic_proxy", "poly wavelength"),
        rep.metric_row("velocity_residual_max_kms", float(np.max(np.abs(vel_resid))), "internal_validation", "poly wavelength"),
        rep.metric_row("lsf_fwhm_min_pix", float(np.min(fwhm)), "synthetic_proxy", "slice LSF"),
        rep.metric_row("lsf_fwhm_max_pix", float(np.max(fwhm)), "synthetic_proxy", "slice LSF"),
        rep.metric_row("recovered_R_min", float(np.min(R)), "internal_validation", "slice LSF"),
        rep.metric_row("recovered_R_max", float(np.max(R)), "internal_validation", "slice LSF"),
        rep.metric_row("flexure_offset_pix", float(drow), "engineering_estimate", "flexure"),
        rep.metric_row("flexure_line_flux_bias_frac", flex_lf_bias, "internal_validation", "flexure"),
        rep.metric_row("flexure_velocity_bias_kms", flex_vel_bias, "internal_validation", "flexure"),
    ]
    csv = rep.write_metrics_csv(outdir / "geometry_systematics_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "geometry_systematics_summary.json", {
        "trace_curvature_pix": curvature_pix,
        "curvature_flux_ratio": {"corrected": fc_corrected, "uncorrected": fc_uncorrected},
        "wavelength_residual_max_um": float(np.max(np.abs(resid_um))),
        "recovered_R_range": [float(np.min(R)), float(np.max(R))],
        "flexure": {"offset_pix": drow, "line_flux_bias_frac": flex_lf_bias, "velocity_bias_kms": flex_vel_bias},
        "limitations": [
            "Illustrative geometry proxies, NOT an as-built slicer or optical model.",
            "Trace curvature is a parabolic integer row offset; wavelength solution and "
            "slice LSF are small synthetic variations.",
        ],
    })

    classes = ["engineering_estimate", "synthetic_proxy", "internal_validation"]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "synthetic_proxy", classes, "curved slicer map", "Trace geometry map."),
        rep.manifest_entry(p2.name, "figure", GEN, "synthetic_proxy", classes, "poly wavelength", "Wavelength-solution residuals."),
        rep.manifest_entry(p3.name, "figure", GEN, "synthetic_proxy", classes, "slice LSF", "Slice LSF -> recovered R."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes, "geometry systematics", "Curvature/wavelength/LSF/flexure metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes, "geometry systematics", "Geometry effects + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"trace_map": p1, "wavelength_resid": p2, "slice_lsf": p3, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote geometry-systematics artifacts:")
    for k, p in paths.items():
        print(f"  {k:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
