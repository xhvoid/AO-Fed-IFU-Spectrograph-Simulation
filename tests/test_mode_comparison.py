"""Tests for the mode-comparison dashboard (direction 4)."""

import json

import pandas as pd

from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def test_dashboard_generates_with_tiny_configs(tmp_path):
    mod = _load_example("run_mode_comparison_dashboard")
    paths = mod.generate(tmp_path, quick=True)
    assert paths["dashboard"].exists()
    assert paths["metrics_csv"].exists()
    assert paths["summary_json"].exists()


def test_metrics_csv_columns_and_every_mode_present(tmp_path):
    mod = _load_example("run_mode_comparison_dashboard")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"])
    assert set(["mode", "metric", "value", "source_class", "notes"]).issubset(df.columns)
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
    # Every mode contributes at least one metric row.
    for mode in ["toy", "instrument_inspired", "public_conditioned", "highz_galaxy"]:
        assert (df["mode"] == mode).sum() >= 1


def test_summary_json_lists_modes_and_provenance(tmp_path):
    mod = _load_example("run_mode_comparison_dashboard")
    paths = mod.generate(tmp_path, quick=True)
    summary = json.loads(paths["summary_json"].read_text())
    assert len(summary["modes"]) == 4
    assert set(summary["provenance_counts"]) == set(summary["modes"])
