#!/usr/bin/env python3
"""Science case: recover a high-z emission-line galaxy from the simulated cube.

A complete, reproducible "truth -> instrument -> recovery" story:

    synthetic high-z galaxy cube (redshifted H-alpha, rotation, intrinsic
    dispersion, public-conditioned)  ->  AO PSF + image-slicer + dispersion +
    detector noise  ->  reconstruction  ->  recovered line flux / velocity / SNR
    ->  recovery-vs-truth bias.

This is a **synthetic science-recovery test**, not a real galaxy observation and
not a validated redshift/flux measurement.

Run:
    python examples/run_science_case_highz_galaxy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu.integration import make_highz_galaxy_config, run_pipeline  # noqa: E402
from ao_fed_ifu.source_cube import REST_FRAME_LINES_UM, observed_wavelength_um  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures" / "generated"
GEN = "examples/run_science_case_highz_galaxy.py"


def generate(outdir: Path, line: str = "Halpha", redshift: float = 2.30, quick: bool = False) -> dict:
    outdir = Path(outdir)
    cfg = make_highz_galaxy_config(line=line, redshift=redshift)
    if quick:
        cfg.nx = cfg.ny = 10
        cfg.resolving_power_R = 2000.0
    result = run_pipeline(cfg)
    grid, lc = result.wavelength_grid_um, result.line_center_um
    window = 4.0 * result.spectrograph.delta_lambda_resel_um(lc)
    obs = observed_wavelength_um(REST_FRAME_LINES_UM[line], redshift)

    # Truth maps from the (conditioned) input cube vs recovered from reconstruction.
    truth_lf = diag.line_flux_map(result.source_cube, grid, lc, window)
    truth_vel = diag.velocity_centroid_map(result.source_cube, grid, lc, window)
    rec_lf, rec_vel, rec_snr = result.line_flux_map, result.velocity_map, result.snr_map
    ap = result.aperture_mask

    # --- Figure 1: truth cube summary. ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    _img(fig, ax[0], truth_lf, "truth line flux [e-]")
    _img(fig, ax[1], truth_vel, "truth velocity [km/s]", symmetric=True, cmap="coolwarm")
    ax[2].plot(grid, result.source_cube[cfg.ny // 2, cfg.nx // 2, :])
    ax[2].axvline(obs, color="r", ls="--", lw=0.8, label=f"{line} @ z={redshift}")
    ax[2].set_title("central-spaxel truth spectrum"); ax[2].set_xlabel("wavelength [um]"); ax[2].legend(fontsize=8)
    fig.suptitle(f"High-z galaxy truth ({line}, z={redshift}, observed {obs:.3f} um)")
    p1 = _save(fig, outdir / "highz_truth_cube_summary.png")

    # --- Figure 2: detector frame. ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    _img(fig, ax[0], result.detector_clean, "detector (noiseless)", aspect="auto", cmap="magma")
    _img(fig, ax[1], result.detector_noisy, "detector (noisy)", aspect="auto", cmap="magma")
    fig.suptitle("High-z galaxy on the image-slicer detector")
    p2 = _save(fig, outdir / "highz_detector_frame.png")

    # --- Figure 3: recovered products. ---
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    _img(fig, ax[0, 0], rec_lf, "recovered line flux [e-]")
    _img(fig, ax[0, 1], rec_vel, "recovered velocity [km/s]", symmetric=True, cmap="coolwarm")
    _img(fig, ax[1, 0], rec_snr, "recovered SNR", cmap="cividis")
    ax[1, 1].plot(grid, result.extracted_spectrum); ax[1, 1].axvline(obs, color="r", ls="--", lw=0.8)
    ax[1, 1].set_title("aperture-extracted spectrum"); ax[1, 1].set_xlabel("wavelength [um]")
    fig.suptitle("High-z galaxy recovered products")
    p3 = _save(fig, outdir / "highz_recovered_products.png")

    # --- Figure 4: recovery vs truth bias. ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].scatter(truth_lf[ap], rec_lf[ap], s=8, alpha=0.6)
    _diag_line(ax[0], truth_lf[ap], rec_lf[ap]); ax[0].set_xlabel("truth line flux"); ax[0].set_ylabel("recovered"); ax[0].set_title("line flux")
    good = np.isfinite(truth_vel[ap]) & np.isfinite(rec_vel[ap])
    ax[1].scatter(truth_vel[ap][good], rec_vel[ap][good], s=8, alpha=0.6)
    _diag_line(ax[1], truth_vel[ap][good], rec_vel[ap][good]); ax[1].set_xlabel("truth velocity [km/s]"); ax[1].set_ylabel("recovered"); ax[1].set_title("velocity")
    resid = rec_vel - truth_vel
    _img(fig, ax[2], resid, "velocity residual [km/s]", symmetric=True, cmap="coolwarm")
    fig.suptitle("High-z galaxy recovery vs truth")
    p4 = _save(fig, outdir / "highz_recovery_bias.png")

    # --- Metrics + biases. ---
    eps = 1e-30
    lf_bias = float(np.nanmedian((rec_lf[ap] - truth_lf[ap]) / (truth_lf[ap] + eps)))
    vel_bias = float(np.nanmedian((rec_vel - truth_vel)[ap]))
    m = result.metrics
    rows = [
        rep.metric_row("line", line, "literature_derived", "rest-frame line", "nebular line id"),
        rep.metric_row("rest_wavelength_um", REST_FRAME_LINES_UM[line], "literature_derived", "atomic constant"),
        rep.metric_row("redshift", redshift, "engineering_estimate", "scenario", "chosen to place line in band"),
        rep.metric_row("observed_wavelength_um", obs, "literature_derived", "redshifted line"),
        rep.metric_row("median_snr", m["median_snr"], "synthetic_proxy", "recovered"),
        rep.metric_row("recovered_R_bright_spaxel", m["recovered_R_bright_spaxel"], "synthetic_proxy", "central spaxel"),
        rep.metric_row("sky_to_source_ratio", m["sky_to_source_ratio"], "engineering_estimate", "public sky budget"),
        rep.metric_row("filter_weighted_throughput", m["filter_weighted_throughput"], "engineering_estimate", "SVO Ks"),
        rep.metric_row("psf_field_throughput", m["psf_field_throughput"], "synthetic_proxy", "PSF coupling"),
        rep.metric_row("line_flux_recovery_bias_frac", lf_bias, "internal_validation", "recovered vs truth"),
        rep.metric_row("velocity_recovery_bias_kms", vel_bias, "internal_validation", "recovered vs truth"),
        rep.metric_row("public_data_components_used", m["public_data_components_used"], "internal_validation", "sky+source+filter"),
    ]
    csv = rep.write_metrics_csv(outdir / "highz_science_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "highz_science_summary.json", {
        "mode": cfg.mode, "line": line, "rest_wavelength_um": REST_FRAME_LINES_UM[line],
        "redshift": redshift, "observed_wavelength_um": obs,
        "band_um": [cfg.lambda_min_um, cfg.lambda_max_um],
        "metrics": m, "provenance": result.provenance.to_records(),
        "provenance_counts": result.provenance.summary_counts(),
        "public_data_components": result.conditioning_info["components"],
        "limitations": [
            "Synthetic high-z galaxy, not a real observation.",
            "Sky background level is an engineering_estimate, not a calibrated ETC.",
            "Weighted reconstruction can over-count flux from closely packed traces.",
        ],
    })

    entries = [
        rep.manifest_entry(p.name, "figure", GEN, "synthetic_proxy",
                           ["direct_public_data", "engineering_estimate", "synthetic_proxy", "internal_validation"],
                           "high-z galaxy public-conditioned run", note)
        for p, note in [(p1, "Truth cube summary."), (p2, "Detector frame."),
                        (p3, "Recovered products."), (p4, "Recovery vs truth.")]
    ]
    entries.append(rep.manifest_entry(csv.name, "table", GEN, "internal_validation",
                   ["literature_derived", "engineering_estimate", "synthetic_proxy", "internal_validation"],
                   "high-z science run", "Science metrics with provenance."))
    entries.append(rep.manifest_entry(summ.name, "table", GEN, "internal_validation",
                   ["direct_public_data", "literature_derived", "engineering_estimate", "synthetic_proxy"],
                   "high-z science run", "Config, line, redshift, provenance, limitations."))
    rep.update_manifest(outdir, entries)
    return {"truth": p1, "detector": p2, "recovered": p3, "bias": p4, "metrics_csv": csv, "summary_json": summ}


def _img(fig, ax, data, title, cmap="viridis", symmetric=False, aspect=None):
    kw = {"origin": "lower", "cmap": cmap}
    if symmetric:
        vmax = np.nanmax(np.abs(data)) or 1.0
        kw.update(vmin=-vmax, vmax=vmax)
    if aspect:
        kw["aspect"] = aspect
    im = ax.imshow(data, **kw); ax.set_title(title); fig.colorbar(im, ax=ax)


def _diag_line(ax, x, y):
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)])); hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote high-z science artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
