#!/usr/bin/env python3
"""Benchmark trend-validation report (offline).

Runs the compact benchmark suite -- seven internal trend / order-of-magnitude
checks over the pipeline and its physics/engineering extensions (SNR scaling,
recovered resolving power, sky-subtraction residual, throughput monotonicity, AO
PSF trends, detector noise regimes, reconstruction flux conservation) -- and
writes a matrix CSV, a summary JSON and a compact pass/fail dashboard.

These are ``internal_validation`` trend checks against the model's own physics.
They are **not** a calibrated validation, **not** an exposure-time calculator, and
**not** an observatory-performance prediction.

Run:
    python examples/run_benchmark_validation_report.py
    ao-ifu benchmark --out figures/generated
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402

from ao_fed_ifu import benchmark_validation as bv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures" / "generated"
GEN = "examples/run_benchmark_validation_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig

    # The suite is closed-form / cheap, so ``quick`` is a no-op here -- accepted
    # for CLI parity with the other figure-suite generators.
    results = bv.run_all_benchmarks()
    df = bv.benchmark_dataframe(results)
    n_pass = int(df["passed"].sum())

    # --- Matrix CSV (one row per benchmark; every row carries a valid source_class). ---
    csv = outdir / "benchmark_validation_matrix.csv"
    df.to_csv(csv, index=False)

    # --- Dashboard figure (compact: pass/fail + key measured values). ---
    png = _dashboard(results, n_pass, outdir / "benchmark_trend_dashboard.png")

    # --- Summary JSON (author + non-overclaiming note injected by reporting). ---
    summ = rep.write_summary_json(outdir / "benchmark_validation_summary.json", {
        "n_benchmarks": len(results),
        "n_passed": n_pass,
        "all_passed": bool(n_pass == len(results)),
        "benchmarks": [
            {"benchmark": r.name, "passed": r.passed, "measured": r.measured,
             "expected": r.expected, "criterion": r.criterion,
             "source_class": r.source_class, "notes": r.notes}
            for r in results
        ],
        "limitations": [
            "Trend / order-of-magnitude checks only; each row is an internal trend check "
            "(source_class is internal_validation / engineering_estimate / synthetic_proxy).",
            "NOT a calibrated validation against any real instrument.",
            "NOT an exposure-time calculator and NOT an observatory-performance prediction.",
        ],
    })

    # --- Manifest (source_classes_used reflects the provenance mix in the matrix). ---
    classes = sorted({r.source_class for r in results})
    entries = [
        rep.manifest_entry(png.name, "figure", GEN, "internal_validation", classes,
                           "benchmark trend suite", "Pass/fail dashboard + key measured values."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes,
                           "benchmark trend matrix", "One row per benchmark: passed, measured, expected, criterion."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes,
                           "benchmark trend suite", "Benchmark outcomes + criteria + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"dashboard": png, "matrix_csv": csv, "summary_json": summ}


def _dashboard(results, n_pass, path):
    """Compact two-panel dashboard: pass/fail markers + key measured values."""
    n = len(results)
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.7 * n + 1.6), gridspec_kw={"width_ratios": [1.0, 1.5]})

    # Panel 1: PASS/FAIL per benchmark.
    for i, r in enumerate(reversed(results)):
        colour = "#1b7837" if r.passed else "#b2182b"
        ax[0].plot(0.06, i, marker="o", ms=11, color=colour, zorder=3)
        ax[0].text(0.13, i, "PASS" if r.passed else "FAIL", va="center", fontsize=8.5,
                   color=colour, fontweight="bold")
        ax[0].text(0.30, i, r.name, va="center", fontsize=8.5)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-0.6, n - 0.4)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    for s in ax[0].spines.values():
        s.set_visible(False)
    ax[0].set_title(f"Benchmark trend checks  ({n_pass}/{n} pass)", fontsize=10)

    # Panel 2: the key measured value(s) behind each verdict.
    ax[1].axis("off")
    lines = []
    for r in reversed(results):
        measured = r.measured if isinstance(r.measured, dict) else {"value": r.measured}
        summary = ", ".join(f"{k}={_fmt(v)}" for k, v in measured.items())
        lines.append(f"{r.name}:\n    {summary}")
    ax[1].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.8,
               family="monospace", transform=ax[1].transAxes)
    ax[1].set_title("Key measured values", fontsize=10)

    fig.suptitle("Benchmark trend validation — internal order-of-magnitude checks, NOT instrument validation",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _fmt(v):
    """Format a measured scalar compactly."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> int:
    paths = generate(OUTDIR)
    results = bv.run_all_benchmarks()
    n_pass = sum(1 for r in results if r.passed)
    print("=" * 80)
    print(" Benchmark trend validation (internal order-of-magnitude checks, not instrument validation)")
    print("=" * 80)
    for r in results:
        print(f" [{'PASS' if r.passed else 'FAIL'}] {r.name}")
        print(f"        expects: {r.expected}")
    print("-" * 80)
    print("Wrote benchmark artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    print(f" {n_pass}/{len(results)} benchmarks passed")
    print("=" * 80)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
