#!/usr/bin/env python3
"""Detector-systematics report: sampling, nonlinearity, correlated noise, IPC.

Visualises four engineering-like IR-array effects on their own: the up-the-ramp /
Fowler read-noise reduction vs number of reads, a signal-dependent nonlinearity,
correlated row read-noise with reference-pixel removal, and inter-pixel
capacitance (IPC) charge sharing.

These are labelled proxies / engineering estimates, **not** a measured detector
model.

Run:
    python examples/run_detector_systematics_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import detector_systematics as ds  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_detector_systematics_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    rng = np.random.default_rng(0)
    read_noise = 12.0
    full_well = 1.0e5
    nl_coeff = 0.05
    ipc_alpha = 0.02
    corr_sigma = 20.0  # row common-mode dominates the per-pixel read noise here

    # --- Figure 1: noise components (nonlinearity, correlated, IPC). ---
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) nonlinearity response.
    s = np.linspace(0, full_well, 200)
    ax[0, 0].plot(s, ds.apply_nonlinearity(s, full_well, nl_coeff), label="measured")
    ax[0, 0].plot(s, s, "k--", lw=0.8, label="ideal (linear)")
    ax[0, 0].set_title(f"Nonlinearity ({nl_coeff:.0%} at full well)")
    ax[0, 0].set_xlabel("input electrons"); ax[0, 0].set_ylabel("measured electrons"); ax[0, 0].legend(fontsize=8)
    # (b) IPC effect on a point source (radial-ish profile).
    pt = np.zeros((21, 21)); pt[10, 10] = 100.0
    ipc = ds.apply_ipc(pt, ipc_alpha)
    ax[0, 1].plot(pt[10], "o-", label="before IPC")
    ax[0, 1].plot(ipc[10], "s-", label="after IPC")
    ax[0, 1].set_title(f"IPC charge sharing (alpha={ipc_alpha})")
    ax[0, 1].set_xlabel("column"); ax[0, 1].set_ylabel("electrons"); ax[0, 1].legend(fontsize=8)
    # (c)/(d) correlated row noise before/after reference-pixel removal.
    frame = np.zeros((48, 48)); frame[:, 3:45] = 400.0  # interior illuminated, edge columns = reference
    noisy = ds.apply_correlated_read_noise(frame, corr_sigma, rng, "row")
    noisy = noisy + rng.normal(0, read_noise, size=noisy.shape)
    corrected = ds.subtract_reference_pixels(noisy)
    vlo, vhi = np.percentile(noisy, [5, 95])
    im = ax[1, 0].imshow(noisy, cmap="magma", vmin=vlo, vmax=vhi); ax[1, 0].set_title("correlated row noise (before)")
    fig.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(corrected, cmap="magma", vmin=vlo, vmax=vhi); ax[1, 1].set_title("after reference-pixel removal")
    fig.colorbar(im, ax=ax[1, 1])
    fig.suptitle("Detector systematics proxies (engineering estimate / synthetic proxy)")
    p1 = _save(fig, outdir / "detector_noise_components.png")

    # --- Figure 2: sampling noise curve. ---
    n_reads = np.arange(1, 9 if quick else 33)
    fowler = [ds.effective_read_noise(read_noise, n, "fowler") for n in n_reads]
    sutr = [ds.effective_read_noise(read_noise, n, "up_the_ramp") for n in n_reads]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(n_reads, fowler, "o-", label="Fowler-N (sigma/sqrt(N))")
    ax.plot(n_reads, sutr, "s-", label="up-the-ramp slope fit")
    ax.axhline(read_noise, color="k", ls=":", lw=0.8, label="single read")
    ax.set_xlabel("number of non-destructive reads"); ax.set_ylabel("effective read noise [e-]")
    ax.set_title("Sampling read-noise reduction (engineering estimate)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p2 = _save(fig, outdir / "detector_sampling_noise_curve.png")

    # --- Metrics. ---
    interior = slice(3, 45)
    cm_before = ds.row_common_mode_metric(noisy[:, interior])
    cm_after = ds.row_common_mode_metric(corrected[:, interior])
    lo, hi = 1000.0, 0.9 * full_well
    loss_lo = 1 - ds.apply_nonlinearity(np.array([lo]), full_well, nl_coeff)[0] / lo
    loss_hi = 1 - ds.apply_nonlinearity(np.array([hi]), full_well, nl_coeff)[0] / hi
    rows = [
        rep.metric_row("effective_read_noise_fowler_16", ds.effective_read_noise(read_noise, 16, "fowler"), "engineering_estimate", "Fowler-16"),
        rep.metric_row("effective_read_noise_uptheramp_32", ds.effective_read_noise(read_noise, 32, "up_the_ramp"), "engineering_estimate", "SUTR-32"),
        rep.metric_row("nonlinearity_frac_loss_low", float(loss_lo), "engineering_estimate", "low count"),
        rep.metric_row("nonlinearity_frac_loss_high", float(loss_hi), "engineering_estimate", "high count"),
        rep.metric_row("nonlinearity_high_over_low", float(loss_hi / loss_lo), "internal_validation", "high vs low", "high pixels change more"),
        rep.metric_row("correlated_common_mode_before", cm_before, "synthetic_proxy", "row noise"),
        rep.metric_row("correlated_common_mode_after", cm_after, "synthetic_proxy", "after ref-pixel removal"),
        rep.metric_row("correlated_reduction_factor", float(cm_before / cm_after) if cm_after > 0 else float("inf"), "internal_validation", "removal"),
        rep.metric_row("ipc_neighbor_fraction", float(4 * ipc_alpha), "synthetic_proxy", "IPC", "fraction shared to neighbours"),
        rep.metric_row("ipc_flux_conserved", float(np.isclose(ipc.sum(), pt.sum())), "internal_validation", "IPC"),
    ]
    csv = rep.write_metrics_csv(outdir / "detector_systematics_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "detector_systematics_summary.json", {
        "effects": ["up_the_ramp/fowler_sampling", "nonlinearity", "correlated_read_noise", "ipc"],
        "provenance": {
            "sampling": "engineering_estimate (Fowler/SUTR scaling formulas)",
            "nonlinearity": "engineering_estimate (quadratic compression)",
            "correlated_read_noise": "synthetic_proxy (row common-mode + reference-pixel removal)",
            "ipc": "synthetic_proxy (3x3 charge-sharing kernel)",
        },
        "limitations": [
            "Compact IR-array proxies, NOT a measured detector model or device cosmetics.",
            "Up-the-ramp slope-fit read noise only beats a single read for many samples; "
            "Fowler averaging reduces it monotonically.",
        ],
    })

    classes = ["engineering_estimate", "synthetic_proxy", "internal_validation"]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "synthetic_proxy", classes, "detector proxies", "Nonlinearity / correlated noise / IPC."),
        rep.manifest_entry(p2.name, "figure", GEN, "engineering_estimate", ["engineering_estimate"], "sampling formulas", "Read-noise vs number of reads."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes, "detector systematics", "Sampling / nonlinearity / correlated / IPC metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes, "detector systematics", "Effects + provenance + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"noise_components": p1, "sampling_curve": p2, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote detector-systematics artifacts:")
    for k, p in paths.items():
        print(f"  {k:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
