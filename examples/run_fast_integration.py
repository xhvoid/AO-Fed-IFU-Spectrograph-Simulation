#!/usr/bin/env python3
"""Run the pipeline and write figures, a CSV metrics table and a JSON summary.

This is the artifact-producing entry point used by CI.  It runs the toy preset
and writes, under ``figures/generated/``:

    * ``fast_detector_frame.png``           -- the (noiseless & noisy) detector frame
    * ``fast_reconstructed_diagnostics.png`` -- line-flux / velocity / SNR / spectrum
    * ``fast_metrics.csv``                  -- a flat metrics table
    * ``fast_summary.json``                 -- metrics + full provenance log

Run:
    python examples/run_fast_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu.integration import make_toy_config, run_pipeline, write_artifacts  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "figures" / "generated"


def main() -> None:
    result = run_pipeline(make_toy_config())
    paths = write_artifacts(result, OUTDIR, prefix="fast")

    print("Wrote artifacts:")
    for name, path in paths.items():
        # Show paths relative to the repo root for readability.
        try:
            rel = path.relative_to(Path(__file__).resolve().parents[1])
        except ValueError:
            rel = path
        print(f"  {name:28s} -> {rel}")

    # Sanity: the acceptance criteria require at least one of each artifact type.
    assert paths["detector_frame"].exists()
    assert paths["reconstructed_diagnostics"].exists()
    assert paths["metrics_csv"].exists()
    assert paths["summary_json"].exists()
    print("All required artifacts present.")


if __name__ == "__main__":
    main()
