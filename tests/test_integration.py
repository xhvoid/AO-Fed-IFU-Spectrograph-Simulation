"""Tests for the end-to-end pipeline and the provenance of output tables.

These guard the two project requirements that unit tests of individual stages
do not otherwise cover:
  * every output table carries provenance labels, and
  * PSF coupling reports its field throughput honestly (no silent flux loss).
"""

import json

import pandas as pd
import pytest

from ao_fed_ifu import validation
from ao_fed_ifu.integration import (
    make_toy_config,
    metric_provenance,
    run_pipeline,
    write_artifacts,
)
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def test_pipeline_reports_psf_field_throughput():
    result = run_pipeline(make_toy_config())
    m = result.metrics
    # The input (pre-PSF) electrons and the throughput are both reported.
    assert "input_source_electrons" in m
    assert "coupled_electrons" in m
    assert "psf_field_throughput" in m
    # With the default flux-conserving convolution, throughput is ~1 (no silent loss).
    assert 0.97 <= m["psf_field_throughput"] <= 1.03


def test_metrics_csv_has_provenance_column(tmp_path):
    result = run_pipeline(make_toy_config())
    paths = write_artifacts(result, tmp_path, prefix="t")
    df = pd.read_csv(paths["metrics_csv"])
    # The metrics table must be self-describing: metric, value, source_class.
    assert set(["metric", "value", "source_class"]).issubset(df.columns)
    assert df["source_class"].notna().all()
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_validation_csv_has_provenance_column(tmp_path):
    df = validation.validation_dataframe()
    assert "source_class" in df.columns
    assert (df["source_class"] == "internal_validation").all()


def test_summary_json_carries_provenance_log(tmp_path):
    result = run_pipeline(make_toy_config())
    paths = write_artifacts(result, tmp_path, prefix="t")
    summary = json.loads(paths["summary_json"].read_text())
    # The shared writer must inject author / version / non-overclaiming note and
    # keep the run fields, so CLI `run` summaries are consistent with the rest.
    for key in ["author", "package_version", "note", "mode", "metrics",
                "provenance", "provenance_counts"]:
        assert key in summary, key
    assert summary["author"] == "Xu Han"
    assert "NOT" in summary["note"]
    assert len(summary["provenance"]) > 0
    for record in summary["provenance"]:
        assert record["source_class"] in VALID_SOURCE_CLASSES


def test_every_metric_has_a_known_provenance_class():
    result = run_pipeline(make_toy_config())
    for key in result.metrics:
        assert metric_provenance(key) in VALID_SOURCE_CLASSES


def test_pipeline_snr_uses_matching_weighted_variance():
    import numpy as np

    result = run_pipeline(make_toy_config())
    # The diagnostics variance is the weighted-estimator variance, and both
    # per-estimator variances are exposed.
    assert result.variance_cube is result.variance_cube_weighted
    assert result.variance_cube_nearest.shape == result.variance_cube_weighted.shape
    # With cross-talk sigma 0.6 the Horne window spans several rows, so the
    # propagated weighted variance exceeds the single-pixel nearest variance.
    finite = np.isfinite(result.variance_cube_weighted) & np.isfinite(result.variance_cube_nearest)
    assert np.median(result.variance_cube_weighted[finite]) > np.median(
        result.variance_cube_nearest[finite]
    )
    # Both estimator SNR baselines are reported.
    assert "median_snr_nearest" in result.metrics
    assert np.isfinite(result.metrics["median_snr_nearest"])


def test_pipeline_snr_pairs_agree_without_crosstalk():
    import numpy as np

    # With no cross-talk the weighted extraction collapses to the nearest pixel,
    # so the two (flux, variance) pairs -- and their SNRs -- must coincide.
    result = run_pipeline(make_toy_config(cross_talk_sigma_pix=0.0))
    assert np.allclose(result.rebuilt_weighted, result.rebuilt_nearest)
    assert np.allclose(result.variance_cube_weighted, result.variance_cube_nearest)
    assert result.metrics["median_snr"] == pytest.approx(result.metrics["median_snr_nearest"])
