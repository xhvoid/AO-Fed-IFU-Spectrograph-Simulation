#!/usr/bin/env python3
"""Run the internal validation suite and write a results table.

Executes the seven trend/consistency checks (flux conservation, SNR vs exposure,
resolving-power scaling, read-noise vs SNR, cross-talk leakage, wavelength-offset
velocity bias, AO PSF encircled energy) and writes ``validation_results.csv``
under ``data/reference_metrics/``.

These are ``internal_validation`` checks against the model's own physics, not a
validation against any real instrument.

Run:
    python examples/run_validation_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu.validation import run_all_validations, validation_dataframe  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "data" / "reference_metrics"


def main() -> int:
    results = run_all_validations()

    print("=" * 80)
    print(" Internal validation checks (trend/consistency, not instrument validation)")
    print("=" * 80)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f" [{status}] {r.name}")
        print(f"        {r.detail}")
    print("-" * 80)

    df = validation_dataframe(results)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "validation_results.csv"
    df.to_csv(out, index=False)
    print(f" wrote {out.relative_to(Path(__file__).resolve().parents[1])}")

    n_pass = int(df["passed"].sum())
    print(f" {n_pass}/{len(df)} checks passed")
    print("=" * 80)
    # Non-zero exit if anything failed, so CI fails loudly.
    return 0 if n_pass == len(df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
