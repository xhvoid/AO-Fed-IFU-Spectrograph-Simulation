"""Tests for public-data loading, public-conditioned physics, and artifacts.

Covers the loader provenance (public vs synthetic), the public-conditioned
budget metrics, and that the public-data figure tables carry source_class.
All checks use tiny configs / internal functions so they stay fast.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ao_fed_ifu.data_sources import (
    VALID_SOURCE_CLASSES,
    load_filter_curve,
    load_sky_spectrum,
    load_source_spectrum,
)
from ao_fed_ifu.integration import (
    make_public_conditioned_config,
    metric_provenance,
    run_pipeline,
)

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "public_cache"
NEW_METRICS = [
    "filter_weighted_throughput", "source_electrons", "sky_background_electrons",
    "sky_to_source_ratio", "public_data_components_used",
]


def _has_cache() -> bool:
    return (CACHE / "skycalc.dat").exists() and (CACHE / "source.dat").exists() and (CACHE / "K.dat").exists()


def test_loaders_return_public_when_cache_present():
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    wl = np.linspace(2.0, 2.4, 40)
    assert load_filter_curve("K")[1].source_class == "direct_public_data"
    assert load_sky_spectrum(wl)[1].source_class == "direct_public_data"
    assert load_source_spectrum(wl)[1].source_class == "direct_public_data"


def test_loaders_fall_back_to_synthetic_with_empty_cache(tmp_path):
    wl = np.linspace(2.0, 2.4, 40)
    assert load_filter_curve("K", cache_dir=tmp_path)[1].source_class == "synthetic_proxy"
    assert load_sky_spectrum(wl, cache_dir=tmp_path)[1].source_class == "synthetic_proxy"
    assert load_source_spectrum(wl, cache_dir=tmp_path)[1].source_class == "synthetic_proxy"


def test_public_conditioned_provenance_has_public_components():
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    result = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0))
    by_name = {r.name: r.source_class for r in result.provenance.records}
    assert by_name.get("sky_spectrum") == "direct_public_data"
    assert by_name.get("source_spectrum") == "direct_public_data"
    assert by_name.get("filter_K") == "direct_public_data"
    assert result.metrics["public_data_components_used"] == 3


def test_public_conditioned_falls_back_with_empty_cache(tmp_path):
    result = run_pipeline(make_public_conditioned_config(
        nx=8, ny=8, resolving_power_R=1500.0, public_cache_dir=str(tmp_path)))
    by_name = {r.name: r.source_class for r in result.provenance.records}
    assert by_name.get("sky_spectrum") == "synthetic_proxy"
    assert by_name.get("source_spectrum") == "synthetic_proxy"
    assert result.metrics["public_data_components_used"] == 0


def test_new_public_metrics_have_provenance_and_values():
    result = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0))
    for k in NEW_METRICS:
        assert k in result.metrics
        assert metric_provenance(k) in VALID_SOURCE_CLASSES
    # Sky background entered the budget and is reported, not hidden.
    assert result.metrics["sky_background_electrons"] > 0
    assert result.metrics["source_electrons"] > 0
    assert result.metrics["sky_to_source_ratio"] > 0


def test_public_conditioning_changes_the_result(tmp_path):
    # Public data must actually change the physics, not just the provenance log.
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    pub = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0)).metrics
    syn = run_pipeline(make_public_conditioned_config(
        nx=8, ny=8, resolving_power_R=1500.0, public_cache_dir=str(tmp_path))).metrics
    # The real Ks filter and SkyCalc sky differ from the synthetic proxies.
    assert pub["filter_weighted_throughput"] != pytest.approx(syn["filter_weighted_throughput"], rel=1e-3)


def test_psf_field_throughput_no_silent_loss():
    result = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0))
    # The uniform emission field uses edge-normalised (flux-conserving) convolution.
    assert 0.95 <= result.metrics["psf_field_throughput"] <= 1.05


def _load_figures_module():
    path = REPO / "examples" / "run_public_data_figures.py"
    spec = importlib.util.spec_from_file_location("run_public_data_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_public_data_figure_tables_have_source_class(tmp_path):
    mod = _load_figures_module()
    paths = mod.generate_all(tmp_path, quick=True)
    # All four PNGs, the two tables and the manifest exist.
    for key in ["public_inputs", "comparison", "psf_grid", "calibration",
                "metrics_csv", "summary_json", "manifest"]:
        assert paths[key].exists()
    # The metrics CSV is self-describing.
    df = pd.read_csv(paths["metrics_csv"])
    assert set(["metric", "value", "source_class", "component"]).issubset(df.columns)
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
    # The manifest carries the required columns and a valid source_class per row.
    man = pd.read_csv(paths["manifest"])
    assert set(["artifact_path", "artifact_type", "generated_by", "source_class",
                "inputs_used", "notes"]).issubset(man.columns)
    assert set(man["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
