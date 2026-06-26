#!/usr/bin/env python3
"""Reconstruction-method comparison and limitations (controlled illustration).

Compares nearest / weighted / weighted+bad-pixel-mask / trace-offset
reconstruction and shows their honest limitations -- in particular that optimal
(weighted) extraction can OVER-count flux shared between closely packed IFU
traces, while a trace offset biases the result.

Not a real reduction pipeline; the covariance proxy is illustrative only.

Run:
    python examples/run_reconstruction_limitations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import cube_reconstruction as recon  # noqa: E402
from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu.detector_model import DetectorConfig  # noqa: E402
from ao_fed_ifu.ifu_mapping import build_slicer_mapping  # noqa: E402
from ao_fed_ifu.integration import make_toy_config, run_pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures" / "generated"
GEN = "examples/run_reconstruction_limitations.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    n = 8 if quick else 12
    cfg = make_toy_config(nx=n, ny=n, resolving_power_R=1500.0 if quick else 2000.0,
                          detector=DetectorConfig(read_noise_e=10.0, dark_current_e_s_pix=0.01, bad_pixel_fraction=0.05))
    res = run_pipeline(cfg)
    grid, lc = res.wavelength_grid_um, res.line_center_um
    window = 3.0 * res.spectrograph.delta_lambda_resel_um(lc)
    mapping = build_slicer_mapping(res.ifu_config, res.spectrograph, grid)

    methods = recon.compare_reconstruction_methods(
        res.detector_noisy, res.variance_frame, res.bad_pixel_mask, mapping, res.coupled_cube, trace_offset=2)
    truth_lf = diag.line_flux_map(res.coupled_cube, grid, lc, window)
    truth_v = diag.velocity_centroid_map(res.coupled_cube, grid, lc, window)
    bad_frac = float(np.mean(recon.bad_voxel_mask(res.bad_pixel_mask, mapping)))
    names = list(methods)

    # --- Figure 1: method comparison (flux conservation + extracted spectra). ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    fc = [methods[m]["flux_conservation_ratio"] for m in names]
    bars = ax[0].bar(range(len(names)), fc, color=["C0", "C1", "C2", "C3"])
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("flux conservation ratio"); ax[0].set_title("flux conservation by method (1.0 = ideal)")
    for b, v in zip(bars, fc):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    aperture = truth_lf > 0.2 * np.nanmax(truth_lf)
    for m in names:
        ax[1].plot(grid, recon.extract_1d_spectrum(methods[m]["cube"], aperture), lw=0.8, label=m)
    ax[1].axvline(lc, color="r", ls="--", lw=0.7); ax[1].legend(fontsize=7)
    ax[1].set_title("aperture-extracted spectrum by method"); ax[1].set_xlabel("wavelength [um]")
    p1 = _save(fig, outdir / "reconstruction_method_comparison.png")

    # --- Figure 2: residual maps (recovered line flux - truth). ---
    fig, ax = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4))
    for a, m in zip(np.atleast_1d(ax), names):
        rec_lf = diag.line_flux_map(methods[m]["cube"], grid, lc, window)
        resid = rec_lf - truth_lf
        vmax = np.nanmax(np.abs(resid)) or 1.0
        im = a.imshow(resid, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        a.set_title(f"{m}\nline-flux residual", fontsize=8); fig.colorbar(im, ax=a)
    fig.suptitle("Line-flux residual maps (recovered - truth)")
    p2 = _save(fig, outdir / "reconstruction_residual_maps.png")

    # --- Figure 3: covariance proxy. ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in ["nearest", "weighted"]:
        acov = recon.estimate_covariance_proxy(methods[m]["cube"], aperture, n_lag=5)
        ax.plot(range(len(acov)), acov, "o-", label=f"{m} (lag-1 corr {methods[m]['spectral_correlation_proxy']:.2f})")
    ax.set_xlabel("spectral lag [channels]"); ax.set_ylabel("normalised autocovariance")
    ax.set_title("Spectral covariance proxy (illustrative only)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p3 = _save(fig, outdir / "reconstruction_covariance_proxy.png")

    # --- Tables. ---
    rows = [rep.metric_row("bad_pixel_masked_fraction", bad_frac, "synthetic_proxy", "detector", "voxel fraction")]
    for m in names:
        rec_lf = diag.line_flux_map(methods[m]["cube"], grid, lc, window)
        rec_v = diag.velocity_centroid_map(methods[m]["cube"], grid, lc, window)
        ap = truth_lf > 0.2 * np.nanmax(truth_lf)
        lf_bias = float(np.nanmedian((rec_lf[ap] - truth_lf[ap]) / (truth_lf[ap] + 1e-30)))
        v_bias = float(np.nanmedian((rec_v - truth_v)[ap]))
        rows += [
            rep.metric_row(f"{m}.flux_conservation_ratio", methods[m]["flux_conservation_ratio"], "internal_validation", m),
            rep.metric_row(f"{m}.line_flux_bias_frac", lf_bias, "internal_validation", m),
            rep.metric_row(f"{m}.velocity_bias_kms", v_bias, "internal_validation", m),
            rep.metric_row(f"{m}.all_finite", float(methods[m]["all_finite"]), "internal_validation", m),
            rep.metric_row(f"{m}.spectral_correlation_proxy", methods[m]["spectral_correlation_proxy"], "synthetic_proxy", m),
        ]
    csv = rep.write_metrics_csv(outdir / "reconstruction_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "reconstruction_summary.json", {
        "methods": names,
        "flux_conservation_ratio": {m: methods[m]["flux_conservation_ratio"] for m in names},
        "bad_pixel_masked_fraction": bad_frac,
        "limitations": [
            "Controlled inverse mapping, not a real reduction pipeline.",
            "Weighted/optimal extraction can over-count flux from closely packed traces (fc > 1).",
            "Trace offset biases flux and velocity; covariance proxy is illustrative only.",
        ],
    })

    entries = [
        rep.manifest_entry(p.name, "figure", GEN, "internal_validation",
                           ["internal_validation", "synthetic_proxy"], "toy pipeline reconstruction", note)
        for p, note in [(p1, "Method comparison."), (p2, "Residual maps."), (p3, "Covariance proxy.")]
    ]
    entries += [
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", ["internal_validation", "synthetic_proxy"], "reconstruction comparison", "Per-method metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", ["internal_validation", "synthetic_proxy"], "reconstruction comparison", "Methods + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"comparison": p1, "residuals": p2, "covariance": p3, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote reconstruction-limitation artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
