"""Tests for the throughput budget module, pipeline thread, and report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ao_fed_ifu import throughput_budget as tb
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.integration import make_public_conditioned_config, run_pipeline

CACHE = Path(__file__).resolve().parents[1] / "data" / "public_cache"
GRID = np.linspace(1.95, 2.45, 100)


def test_combined_throughput_in_unit_interval():
    comb = tb.build_default_budget(GRID).combined(GRID)
    assert np.all(comb >= 0.0) and np.all(comb <= 1.0)
    assert 0.0 < tb.build_default_budget(GRID).effective_throughput(GRID) < 1.0


def test_extra_lossy_component_never_increases_throughput():
    budget = tb.build_default_budget(GRID)
    base = budget.combined(GRID)
    budget.components.append(tb.ThroughputComponent(
        name="extra_loss", kind="test", transmission=0.5, source_class="engineering_estimate"))
    with_extra = budget.combined(GRID)
    assert np.all(with_extra <= base + 1e-12)
    assert with_extra.max() < base.max()


def test_instrument_only_is_higher_than_full_budget():
    full = tb.build_default_budget(GRID).effective_throughput(GRID)
    instr = tb.build_default_budget(GRID, include=tb.INSTRUMENT_STAGES).effective_throughput(GRID)
    assert instr > full  # fewer lossy stages -> higher throughput


def test_public_components_labelled_direct_when_cached():
    if not ((CACHE / "skycalc.dat").exists() and (CACHE / "K.dat").exists()):
        pytest.skip("public cache not present")
    table = {r["component"]: r["source_class"] for r in tb.build_default_budget(GRID).component_table(GRID)}
    assert table["atmosphere"] == "direct_public_data"
    assert table["filter_Ks"] == "direct_public_data"
    assert table["ifu"] == "synthetic_proxy"


def test_empty_cache_falls_back_to_synthetic(tmp_path):
    table = {r["component"]: r["source_class"] for r in tb.build_default_budget(GRID, cache_dir=tmp_path).component_table(GRID)}
    assert table["atmosphere"] == "synthetic_proxy"
    assert table["filter_Ks"] == "synthetic_proxy"


def test_component_table_provenance_valid():
    for r in tb.build_default_budget(GRID).component_table(GRID):
        assert r["source_class"] in VALID_SOURCE_CLASSES


def test_pipeline_thread_is_backward_compatible_and_deterministic():
    # Default: no budget keys, behaviour unchanged.
    m0 = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0)).metrics
    assert "effective_throughput_used" not in m0
    # Enabled: reports both throughputs and is deterministic for a fixed seed.
    cfg = dict(nx=8, ny=8, resolving_power_R=1500.0, use_throughput_budget=True)
    m1 = run_pipeline(make_public_conditioned_config(**cfg)).metrics
    m2 = run_pipeline(make_public_conditioned_config(**cfg)).metrics
    assert 0.0 < m1["effective_throughput_used"] < 1.0
    assert "configured_throughput_scalar" in m1
    assert m1["median_snr"] == m2["median_snr"]


def test_report_csv_has_valid_provenance(tmp_path):
    mod = _load_example("run_throughput_budget_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["waterfall", "combined_curve", "components_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["components_csv"])
    assert set(["component", "value_or_effective_throughput", "source_class", "reference", "notes"]).issubset(df.columns)
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
