"""Tests for the benchmark trend-validation suite (internal order-of-magnitude checks)."""

import json
import math

import pandas as pd

from ao_fed_ifu import benchmark_validation as bv
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def _all_numbers_finite(measured) -> bool:
    values = measured.values() if isinstance(measured, dict) else [measured]
    return all((not isinstance(v, (int, float))) or math.isfinite(v) for v in values)


def test_run_all_benchmarks_non_empty():
    results = bv.run_all_benchmarks()
    assert isinstance(results, list) and len(results) >= 7


# Benchmarks are synthetic trend checks: their provenance must be one of these,
# and in particular never direct_public_data (they are not measured data).
_ALLOWED_BENCHMARK_CLASSES = {"internal_validation", "engineering_estimate", "synthetic_proxy"}


def test_each_result_is_well_formed():
    for r in bv.run_all_benchmarks():
        assert isinstance(r.name, str) and r.name.strip()
        assert isinstance(r.passed, bool)
        assert r.source_class in VALID_SOURCE_CLASSES
        assert r.source_class in _ALLOWED_BENCHMARK_CLASSES  # never direct_public_data
        assert isinstance(r.expected, str) and r.expected.strip()
        assert isinstance(r.criterion, str) and r.criterion.strip()
        assert isinstance(r.notes, str)
        # measured must be a scalar or a JSON-serialisable dict of finite numbers.
        assert isinstance(r.measured, (int, float, dict))
        assert _all_numbers_finite(r.measured)
        json.dumps(r.measured)  # raises if not serialisable


def test_benchmark_dataframe_columns_and_provenance():
    df = bv.benchmark_dataframe()
    assert list(df.columns) == bv.BENCHMARK_COLUMNS
    assert "criterion" in df.columns
    assert df["criterion"].map(lambda c: isinstance(c, str) and bool(c.strip())).all()
    assert len(df) >= 7
    assert set(df["source_class"]).issubset(_ALLOWED_BENCHMARK_CLASSES)
    assert "direct_public_data" not in set(df["source_class"])
    assert df["passed"].map(lambda x: isinstance(x, bool)).all()


def test_core_benchmarks_pass():
    results = bv.run_all_benchmarks()
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"benchmark trend checks failed: {failed}"


def test_report_writes_artifacts_with_provenance(tmp_path):
    mod = _load_example("run_benchmark_validation_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ("dashboard", "matrix_csv", "summary_json"):
        assert paths[key].exists()

    # CSV: every row carries a valid provenance class and an explicit criterion.
    df = pd.read_csv(paths["matrix_csv"])
    assert {"source_class", "criterion"}.issubset(df.columns)
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
    assert df["criterion"].map(lambda c: isinstance(c, str) and bool(c.strip())).all()

    # Summary JSON: carries the shared non-overclaiming note + per-benchmark criterion.
    payload = json.loads(paths["summary_json"].read_text())
    assert "note" in payload and "NOT" in payload["note"]
    assert payload["all_passed"] is True
    assert all("criterion" in b and b["criterion"] for b in payload["benchmarks"])

    # Manifest: includes the three benchmark artifacts.
    manifest = pd.read_csv(tmp_path / "artifact_manifest.csv")
    tracked = set(manifest["artifact_path"])
    assert {"benchmark_validation_matrix.csv", "benchmark_validation_summary.json",
            "benchmark_trend_dashboard.png"}.issubset(tracked)
