#!/usr/bin/env python3
"""Instrument-systematics sensitivity report (controlled trend checks).

Perturbs one instrument/calibration parameter at a time and shows how the
recovered velocity, line flux, SNR and resolving power respond.  These are
controlled perturbation sanity checks, **not** a real tolerance budget.

Run:
    python examples/run_systematics_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import systematics as sysm  # noqa: E402
from ao_fed_ifu.detector_model import DetectorConfig  # noqa: E402
from ao_fed_ifu.integration import make_toy_config, run_pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures" / "generated"
GEN = "examples/run_systematics_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    base = make_toy_config(nx=8 if quick else 10, ny=8 if quick else 10, resolving_power_R=1500.0 if quick else 2000.0)
    n_levels = 2 if quick else 3
    kinds = ["wavelength_offset", "trace_row_offset", "cross_talk_sigma", "read_noise_e"] if quick else sysm.ALL_KINDS
    results = sysm.compute_systematics(base, n_levels=n_levels, kinds=kinds)

    # --- Figure 1: bias matrix (perturbation x response, at the max level). ---
    perts = list(results)
    responses = ["velocity", "lineflux", "snr"]
    rlabels = ["|velocity bias|\n[km/s]", "|line-flux bias|\n[frac]", "SNR loss\n[frac]"]
    M = np.array([[np.max(np.abs(results[p][r])) for r in responses] for p in perts])
    Mn = M / (np.max(np.abs(M), axis=0, keepdims=True) + 1e-30)  # normalise per column for colour
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(perts) + 2))
    im = ax.imshow(Mn, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(responses))); ax.set_xticklabels(rlabels, fontsize=8)
    ax.set_yticks(range(len(perts))); ax.set_yticklabels(perts, fontsize=8)
    for i in range(len(perts)):
        for j in range(len(responses)):
            ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center", color="w", fontsize=7)
    ax.set_title("Systematics bias matrix (max-level magnitude; colour = per-response norm)")
    p1 = _save(fig, outdir / "systematics_bias_matrix.png")

    # --- Figure 2: waterfall of extracted spectra vs read noise. ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    rn_levels = [8, 20, 40] if not quick else [8, 40]
    for k, rn in enumerate(rn_levels):
        cfg = make_toy_config(nx=base.nx, ny=base.ny, resolving_power_R=base.resolving_power_R,
                              detector=DetectorConfig(read_noise_e=float(rn), dark_current_e_s_pix=0.01))
        r = run_pipeline(cfg)
        spec = r.extracted_spectrum / np.nanmax(r.extracted_spectrum)
        ax.plot(r.wavelength_grid_um, spec + 1.2 * k, lw=0.9, label=f"read noise {rn} e-")
    ax.set_title("Extracted-spectrum waterfall vs read noise (offset for clarity)")
    ax.set_xlabel("wavelength [um]"); ax.set_ylabel("normalised flux + offset"); ax.legend(fontsize=8)
    p2 = _save(fig, outdir / "systematics_waterfall_spectrum.png")

    # --- Figure 3: metric trends vs perturbation level. ---
    ncol = 2
    nrow = int(np.ceil(len(perts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.2 * nrow))
    for a, pert in zip(np.atleast_1d(axes).ravel(), perts):
        e = results[pert]
        a.plot(e["x"], e["velocity"], "o-", label="velocity [km/s]")
        a.plot(e["x"], np.array(e["lineflux"]) * 100, "s-", label="line-flux [%]")
        a.plot(e["x"], np.array(e["snr"]) * 100, "^-", label="SNR loss [%]")
        a.set_title(pert); a.set_xlabel(f"level [{e['units']}]"); a.grid(alpha=0.3); a.legend(fontsize=6)
    for a in np.atleast_1d(axes).ravel()[len(perts):]:
        a.axis("off")
    fig.suptitle("Systematics metric trends (internal trend checks, not a tolerance budget)")
    p3 = _save(fig, outdir / "systematics_metric_trends.png")

    # --- Tables. ---
    raw = sysm.systematics_summary_rows(results)
    rows = [rep.metric_row(f"{r['kind']}.{r['metric']}", r["value"], r["source_class"], r["kind"], r.get("notes", ""))
            for r in raw]
    csv = rep.write_metrics_csv(outdir / "systematics_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "systematics_summary.json", {
        "perturbations": perts,
        "trends": {k: {kk: (vv if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in results.items()},
        "limitations": [
            "Controlled single-parameter perturbations, not a real tolerance budget.",
            "Perturbation amplitudes are engineering estimates.",
            "Flat-field error and saturation are not in the propagated SNR variance.",
        ],
    })

    classes = ["internal_validation", "engineering_estimate", "synthetic_proxy"]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "internal_validation", classes, "toy pipeline perturbations", "Bias matrix."),
        rep.manifest_entry(p2.name, "figure", GEN, "synthetic_proxy", ["synthetic_proxy", "engineering_estimate"], "read-noise scan", "Spectrum waterfall."),
        rep.manifest_entry(p3.name, "figure", GEN, "internal_validation", classes, "toy pipeline perturbations", "Metric trends."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes, "systematics scan", "Per-perturbation max biases."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes, "systematics scan", "Trends + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"bias_matrix": p1, "waterfall": p2, "trends": p3, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote systematics artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
