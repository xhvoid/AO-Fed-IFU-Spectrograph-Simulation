#!/usr/bin/env python3
"""Throughput budget report: decompose where the photons are lost (offline).

Builds a decomposed end-to-end throughput budget (atmosphere -> telescope -> AO
relay -> IFU slicer -> spectrograph -> filter -> detector QE) from cached public
data (SkyCalc telluric, SVO Ks) plus labelled engineering-estimate / synthetic
proxies, and writes a waterfall, a combined-curve figure, a components CSV and a
summary JSON.

This is a decomposed engineering estimate for visualisation, **not** a calibrated
throughput model or an exposure-time calculator.

Run:
    python examples/run_throughput_budget_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ao_fed_ifu.throughput_budget import build_default_budget  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_throughput_budget_report.py"

# A fixed provenance colour map for the bars.
_PROV_COLOR = {
    "direct_public_data": "C0", "literature_derived": "C2",
    "engineering_estimate": "C1", "synthetic_proxy": "C3", "internal_validation": "C4",
}


def generate(outdir: Path, quick: bool = False, cache_dir=None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    grid = np.linspace(1.95, 2.45, 120 if quick else 400)
    budget = build_default_budget(grid, cache_dir=cache_dir)
    combined = budget.combined(grid)
    cascade = budget.cumulative_cascade(grid)
    table = budget.component_table(grid)
    effective = budget.effective_throughput(grid)

    # --- Figure 1: waterfall (cumulative cascade + per-component effective). ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    names = [n for n, _ in cascade]
    cum = [v for _, v in cascade]
    ax[0].bar(range(len(names)), cum, color="C0")
    ax[0].plot(range(len(names)), cum, "k.-", lw=0.8)
    for i, v in enumerate(cum):
        ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("cumulative throughput"); ax[0].set_title("Throughput waterfall (remaining after each stage)")
    ax[0].set_ylim(0, 1)

    eff = [r["value_or_effective_throughput"] for r in table]
    colors = [_PROV_COLOR.get(r["source_class"], "C7") for r in table]
    comp_names = [r["component"] for r in table]
    ax[1].barh(range(len(comp_names)), eff, color=colors)
    ax[1].set_yticks(range(len(comp_names))); ax[1].set_yticklabels(comp_names, fontsize=8)
    ax[1].invert_yaxis(); ax[1].set_xlabel("effective transmission"); ax[1].set_xlim(0, 1)
    ax[1].set_title("Per-component (colour = provenance)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _PROV_COLOR.values()]
    ax[1].legend(handles, list(_PROV_COLOR), fontsize=6, loc="lower right")
    fig.suptitle(f"End-to-end throughput budget (effective {effective:.3f}) -- engineering estimate, not an ETC")
    p1 = _save(fig, outdir / "throughput_budget_waterfall.png")

    # --- Figure 2: combined curve + per-component curves vs wavelength. ---
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for c in budget.components:
        ax.plot(grid, c.on_grid(grid), lw=0.8, alpha=0.7, label=c.name)
    ax.plot(grid, combined, "k", lw=2.0, label="combined")
    ax.set_xlabel("wavelength [um]"); ax.set_ylabel("transmission"); ax.set_ylim(0, 1)
    ax.set_title("Throughput components and combined curve"); ax.legend(fontsize=7, ncol=2)
    p2 = _save(fig, outdir / "throughput_combined_curve.png")

    # --- CSV: component,value_or_effective_throughput,source_class,reference,notes ---
    for r in table:
        rep.validate_class(r["source_class"])
    csv = outdir / "throughput_budget_components.csv"
    pd.DataFrame(table, columns=["component", "value_or_effective_throughput",
                                 "source_class", "reference", "notes"]).to_csv(csv, index=False)

    # --- JSON summary. ---
    counts = budget.provenance_counts()
    summ = rep.write_summary_json(outdir / "throughput_budget_summary.json", {
        "components_used": [r["component"] for r in table],
        "effective_throughput": effective,
        "cumulative_cascade": {n: v for n, v in cascade},
        "provenance_counts": counts,
        "n_public": counts.get("direct_public_data", 0),
        "n_proxy_or_estimate": counts.get("synthetic_proxy", 0) + counts.get("engineering_estimate", 0),
        "limitations": [
            "Decomposed engineering estimate, not a calibrated throughput or ETC.",
            "Atmosphere and filter use cached public data; instrument stages are "
            "engineering-estimate / synthetic proxies.",
        ],
    })

    classes = sorted(set(r["source_class"] for r in table))
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "engineering_estimate", classes, "SkyCalc+SVO cache + proxies", "Throughput waterfall."),
        rep.manifest_entry(p2.name, "figure", GEN, "engineering_estimate", classes, "SkyCalc+SVO cache + proxies", "Combined + component curves."),
        rep.manifest_entry(csv.name, "table", GEN, "engineering_estimate", classes, "throughput components", "component,value,source_class,reference,notes."),
        rep.manifest_entry(summ.name, "table", GEN, "engineering_estimate", classes, "throughput budget", "Effective throughput + provenance + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"waterfall": p1, "combined_curve": p2, "components_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote throughput-budget artifacts:")
    for k, p in paths.items():
        print(f"  {k:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
