#!/usr/bin/env python3
"""Observing-strategy report: dither, sky frames, cosmic-ray reject, coadd.

Runs several dithered exposures through the pipeline, subtracts a sky model,
injects a cosmic-ray proxy, aligns and inverse-variance coadds with sigma
clipping, and shows the SNR gain (~sqrt(N)), the cosmic-ray cleanup and the
single-vs-coadd products.

This is an observing-strategy **demonstrator**, not a real scheduler or reduction
pipeline.

Run:
    python examples/run_observing_strategy_report.py
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
from ao_fed_ifu.integration import make_toy_config  # noqa: E402
from ao_fed_ifu.observing_strategy import (  # noqa: E402
    ExposurePlan, run_observing_strategy, snr_vs_nexp,
)

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_observing_strategy_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    # A compact (galaxy) source with bad pixels + cosmic rays, so dithering and
    # rejection visibly help.
    n = 10 if quick else 14
    base = make_toy_config(nx=n, ny=n, resolving_power_R=1500.0 if quick else 2000.0,
                           source_kind="galaxy")
    n_exp = 4 if quick else 6
    plan = ExposurePlan(n_exposures=n_exp, cosmic_ray_rate=0.01, seed=0)
    res = run_observing_strategy(base, plan)
    misaligned = run_observing_strategy(base, ExposurePlan(
        n_exposures=n_exp, cosmic_ray_rate=0.01, seed=0, align=False))

    grid, lc = res.reference.wavelength_grid_um, res.reference.line_center_um
    window = 3.0 * res.reference.spectrograph.delta_lambda_resel_um(lc)
    ns, snrs, slope = snr_vs_nexp(res, lc, window)
    ap = res.reference.aperture_mask

    single_lf = diag.line_flux_map(res.exposure_cubes[0], grid, lc, window)
    coadd_lf = diag.line_flux_map(res.coadd_cube, grid, lc, window)
    single_snr = diag.snr_map(res.exposure_cubes[0], res.exposure_variances[0], grid, lc, window)
    coadd_snr = diag.snr_map(res.coadd_cube, res.coadd_variance, grid, lc, window)

    # --- Figure 1: dither pattern. ---
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    offs = np.array(res.offsets)
    ax.plot(offs[:, 1], offs[:, 0], "o-", color="C0")
    for i, (dy, dx) in enumerate(res.offsets):
        ax.annotate(str(i + 1), (dx, dy), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("dx [spaxels]"); ax.set_ylabel("dy [spaxels]"); ax.grid(alpha=0.3)
    ax.set_title(f"Dither pattern ({len(res.offsets)} exposures)"); ax.set_aspect("equal")
    p1 = _save(fig, outdir / "dither_pattern.png")

    # --- Figure 2: coadd SNR gain vs N. ---
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(ns, snrs, "o-", label="measured median SNR")
    ax.plot(ns, snrs[0] * np.sqrt(ns / ns[0]), "k--", label="sqrt(N) reference")
    ax.set_xlabel("number of exposures coadded"); ax.set_ylabel("median line SNR")
    ax.set_title(f"Coadd SNR gain (slope {slope:.2f} vs 0.5 expected)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p2 = _save(fig, outdir / "coadd_snr_gain.png")

    # --- Figure 3: single vs coadd products. ---
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for a, data, title in [(axes[0, 0], single_lf, "single: line flux"),
                           (axes[0, 1], coadd_lf, "coadd: line flux"),
                           (axes[1, 0], single_snr, "single: SNR"),
                           (axes[1, 1], coadd_snr, "coadd: SNR")]:
        im = a.imshow(data, origin="lower", cmap="cividis" if "SNR" in title else "viridis")
        a.set_title(title, fontsize=9); fig.colorbar(im, ax=a)
    fig.suptitle("Single exposure vs dithered coadd (cosmic rays + bad pixels cleaned)")
    p3 = _save(fig, outdir / "single_vs_coadd_products.png")

    # --- Metrics. ---
    snr_single = float(np.nanmedian(single_snr[ap]))
    snr_coadd = float(np.nanmedian(coadd_snr[ap]))
    peak_aligned = float(np.nanmax(coadd_lf))
    peak_misaligned = float(np.nanmax(diag.line_flux_map(misaligned.coadd_cube, grid, lc, window)))
    rows = [
        rep.metric_row("n_exposures", plan.n_exposures, "internal_validation", "plan"),
        rep.metric_row("snr_single", snr_single, "synthetic_proxy", "single exposure"),
        rep.metric_row("snr_coadd", snr_coadd, "synthetic_proxy", "dithered coadd"),
        rep.metric_row("snr_gain", snr_coadd / snr_single if snr_single else float("nan"), "internal_validation", "coadd/single"),
        rep.metric_row("sqrt_n_slope", slope, "internal_validation", "d log SNR / d log N", "expect ~0.5"),
        rep.metric_row("cosmic_ray_rejected_fraction", res.rejected_fraction, "synthetic_proxy", "sigma clip", "CR proxy"),
        rep.metric_row("coadd_finite_fraction", res.finite_fraction, "internal_validation", "coadd"),
        rep.metric_row("flux_peak_recovery_aligned_over_misaligned",
                       peak_aligned / peak_misaligned if peak_misaligned else float("nan"),
                       "internal_validation", "alignment", ">1 means alignment concentrates the source"),
    ]
    csv = rep.write_metrics_csv(outdir / "observing_strategy_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "observing_strategy_summary.json", {
        "n_exposures": plan.n_exposures, "offsets": res.offsets,
        "snr_vs_n": {int(k): float(v) for k, v in zip(ns, snrs)},
        "sqrt_n_slope": slope, "cosmic_ray_rejected_fraction": res.rejected_fraction,
        "coadd_finite_fraction": res.finite_fraction,
        "limitations": [
            "Observing-strategy demonstrator, not a real scheduler or reduction pipeline.",
            "Cosmic-ray injection/rejection is a synthetic proxy; sky model is the pipeline proxy.",
            "Integer dither only; sub-spaxel dithering is not modelled.",
        ],
    })

    classes = ["internal_validation", "synthetic_proxy"]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "internal_validation", classes, "exposure plan", "Dither pattern."),
        rep.manifest_entry(p2.name, "figure", GEN, "internal_validation", classes, "coadd sweep", "SNR gain vs N."),
        rep.manifest_entry(p3.name, "figure", GEN, "synthetic_proxy", classes, "single vs coadd", "Single vs coadd products."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes, "observing strategy", "SNR gain / rejection / recovery metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes, "observing strategy", "SNR-vs-N + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"dither": p1, "snr_gain": p2, "single_vs_coadd": p3, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote observing-strategy artifacts:")
    for k, p in paths.items():
        print(f"  {k:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
