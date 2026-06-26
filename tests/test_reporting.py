"""Tests for the shared reporting helpers and the merge-able manifest."""

import pandas as pd
import pytest

from ao_fed_ifu import reporting as rep
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def test_invalid_source_class_raises():
    with pytest.raises(ValueError):
        rep.validate_class("mixed")
    with pytest.raises(ValueError):
        rep.metric_row("m", 1.0, "not_a_class")


def test_join_source_classes_dedups_and_validates():
    s = rep.join_source_classes(["synthetic_proxy", "synthetic_proxy", "internal_validation"])
    assert s == "synthetic_proxy;internal_validation"
    with pytest.raises(ValueError):
        rep.join_source_classes(["bogus"])


def test_write_metrics_csv_columns(tmp_path):
    rows = [rep.metric_row("a", 1.0, "internal_validation", "comp", "note")]
    path = rep.write_metrics_csv(tmp_path / "m.csv", rows)
    df = pd.read_csv(path)
    assert list(df.columns) == ["metric", "value", "source_class", "component", "notes"]


def test_update_manifest_merges_by_path(tmp_path):
    e1 = [rep.manifest_entry("a.png", "figure", "gen1", "synthetic_proxy",
                             ["synthetic_proxy", "internal_validation"], "in", "first")]
    rep.update_manifest(tmp_path, e1)
    e2 = [rep.manifest_entry("b.csv", "table", "gen2", "internal_validation",
                             "internal_validation", "in", "second")]
    rep.update_manifest(tmp_path, e2)

    df = pd.read_csv(tmp_path / "artifact_manifest.csv")
    assert set(df["artifact_path"]) == {"a.png", "b.csv"}  # both generators' rows kept
    assert set(rep.MANIFEST_COLUMNS).issubset(df.columns)
    # Re-writing the same path replaces (does not duplicate) its row.
    rep.update_manifest(tmp_path, [rep.manifest_entry("a.png", "figure", "gen1", "internal_validation",
                                                      ["internal_validation"], "in", "updated")])
    df2 = pd.read_csv(tmp_path / "artifact_manifest.csv")
    assert (df2["artifact_path"] == "a.png").sum() == 1
    assert df2.loc[df2["artifact_path"] == "a.png", "source_class"].iloc[0] == "internal_validation"
    # Every source_classes_used token is a valid class.
    for cell in df2["source_classes_used"]:
        for tok in str(cell).split(";"):
            assert tok in VALID_SOURCE_CLASSES
