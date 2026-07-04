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
    "sky_subtraction_applied", "sky_subtraction_scale_error",
    "sky_line_masked_fraction", "sky_line_masked_in_window_fraction",
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


# ---------------------------------------------------------------------------
# Cache-coverage hardening: no silent extrapolation beyond the cached range.
# ---------------------------------------------------------------------------

def test_sky_loader_refuses_out_of_range_extrapolation():
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    # skycalc.dat covers ~1.90-2.50 um; ask beyond both ends.
    wl = np.linspace(1.5, 2.6, 60)
    sky, prov = load_sky_spectrum(wl)
    assert prov.source_class == "synthetic_proxy"
    assert "extrapolate" in prov.notes
    assert "1.90" in prov.notes and "2.60" in prov.notes  # coverage vs request
    # The synthetic proxy still returns usable arrays on the requested grid.
    assert sky["radiance"].shape == wl.shape
    assert np.all(np.isfinite(sky["radiance"]))


def test_source_loader_refuses_out_of_range_extrapolation():
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    # source.dat covers ~0.30-2.60 um; ask beyond the red end.
    wl = np.linspace(2.0, 3.0, 60)
    spec, prov = load_source_spectrum(wl)
    assert prov.source_class == "synthetic_proxy"
    assert "extrapolate" in prov.notes
    assert spec["flux"].shape == wl.shape


def test_loaders_accept_exact_boundary_range():
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    # A grid whose endpoints equal the cache endpoints must still be served
    # from the cache (the coverage check needs numerical slack, not strictness).
    wl = np.linspace(1.90, 2.50, 100)
    assert load_sky_spectrum(wl)[1].source_class == "direct_public_data"
    wl_src = np.linspace(0.32, 2.59, 100)
    assert load_source_spectrum(wl_src)[1].source_class == "direct_public_data"


def test_pipeline_band_outside_cache_flips_to_proxy_with_note():
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")
    # A public-conditioned run over a band the sky cache cannot cover must not
    # silently reuse cache-edge values: it flips to the synthetic proxy and the
    # provenance says why.
    result = run_pipeline(make_public_conditioned_config(
        nx=8, ny=8, resolving_power_R=1500.0,
        lambda_min_um=1.60, lambda_max_um=2.30))
    by_name = {r.name: r for r in result.provenance.records}
    assert by_name["sky_spectrum"].source_class == "synthetic_proxy"
    assert "extrapolate" in by_name["sky_spectrum"].notes
    # The source cache (0.30-2.60 um) still covers this band.
    assert by_name["source_spectrum"].source_class == "direct_public_data"


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


# ---------------------------------------------------------------------------
# Sky-subtracted products (public-conditioned reduction stage).
# ---------------------------------------------------------------------------

def test_public_conditioned_subtracts_sky_by_default():
    result = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0))
    assert result.metrics["sky_subtraction_applied"] is True
    assert result.rebuilt_weighted_skysub is not None
    assert result.rebuilt_nearest_skysub is not None
    assert result.sky_cube_estimate_weighted is not None
    # Exact linear identity: skysub + sky model == raw sky-added cube.
    assert np.allclose(
        result.rebuilt_weighted_skysub + result.sky_cube_estimate_weighted,
        result.rebuilt_weighted,
    )
    # The diagnostics were computed on the REDUCED cube.
    from ao_fed_ifu import diagnostics as diag

    window = 3.0 * result.spectrograph.delta_lambda_resel_um(result.line_center_um)
    lf = diag.line_flux_map(
        result.rebuilt_weighted_skysub, result.wavelength_grid_um,
        result.line_center_um, window,
    )
    assert np.allclose(lf, result.line_flux_map, equal_nan=True)


def test_sky_added_products_remain_available():
    result = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0))
    # Raw (sky-added) and reduced cubes genuinely differ when a sky was injected.
    assert result.metrics["sky_background_electrons"] > 0
    assert not np.allclose(result.rebuilt_weighted, result.rebuilt_weighted_skysub)
    # The sky model is non-negative and spectrally structured (OH lines).
    sky = result.sky_cube_estimate_weighted
    assert np.nanmin(sky) >= -1e-9
    assert np.nanmax(sky) > np.nanmedian(sky)


def test_sky_scale_error_leaves_linear_residual():
    kw = dict(nx=8, ny=8, resolving_power_R=1500.0, seed=3)
    perfect = run_pipeline(make_public_conditioned_config(**kw))
    over = run_pipeline(make_public_conditioned_config(sky_subtraction_scale_error=0.2, **kw))
    # Same seed -> same data; over-subtraction removes exactly 0.2x more sky.
    expected = perfect.rebuilt_weighted_skysub - 0.2 * perfect.sky_cube_estimate_weighted
    assert np.allclose(over.rebuilt_weighted_skysub, expected)
    assert over.metrics["sky_subtraction_scale_error"] == pytest.approx(0.2)


def test_subtract_sky_off_keeps_raw_diagnostics():
    result = run_pipeline(make_public_conditioned_config(
        nx=8, ny=8, resolving_power_R=1500.0, subtract_sky=False))
    assert result.metrics["sky_subtraction_applied"] is False
    assert result.rebuilt_weighted_skysub is None
    from ao_fed_ifu import diagnostics as diag

    window = 3.0 * result.spectrograph.delta_lambda_resel_um(result.line_center_um)
    lf = diag.line_flux_map(
        result.rebuilt_weighted, result.wavelength_grid_um, result.line_center_um, window)
    assert np.allclose(lf, result.line_flux_map, equal_nan=True)


def test_toy_mode_has_no_sky_and_no_subtraction():
    from ao_fed_ifu.integration import make_toy_config

    result = run_pipeline(make_toy_config())
    assert result.metrics["sky_subtraction_applied"] is False
    assert result.metrics["sky_line_masked_fraction"] == 0.0
    assert result.rebuilt_weighted_skysub is None
    assert result.observed_truth_cube is not None  # observed truth exists for all modes


def test_sky_line_mask_metrics_in_range():
    result = run_pipeline(make_public_conditioned_config(nx=8, ny=8, resolving_power_R=1500.0))
    frac = result.metrics["sky_line_masked_fraction"]
    in_win = result.metrics["sky_line_masked_in_window_fraction"]
    assert 0.0 <= frac <= 1.0
    assert 0.0 <= in_win <= 1.0
    # A near-IR sky has OH lines, so a 3x-median mask flags a non-trivial fraction.
    assert frac > 0.0


def test_observed_truth_is_deterministic_and_source_only():
    from ao_fed_ifu.detector_model import DetectorConfig

    # Disable bad pixels: with them, the seed-dependent inf-variance rows change
    # the extraction weights near bad pixels, which is expected behaviour.
    det = DetectorConfig(read_noise_e=8.0, dark_current_e_s_pix=0.005, bad_pixel_fraction=0.0)
    kw = dict(nx=8, ny=8, resolving_power_R=1500.0, detector=det)
    a = run_pipeline(make_public_conditioned_config(seed=1, **kw))
    b = run_pipeline(make_public_conditioned_config(seed=2, **kw))
    # The observed truth is built from the NOISELESS source-only frame with
    # deterministic variance weights, so it is seed-independent -- unlike the
    # noisy reconstructions.
    assert np.all(np.isfinite(a.observed_truth_cube))
    assert np.allclose(a.observed_truth_cube, b.observed_truth_cube)
    assert not np.allclose(a.rebuilt_weighted, b.rebuilt_weighted)


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
    # The manifest carries the required columns; source_class is a single valid
    # primary label, and source_classes_used lists the full (possibly mixed)
    # provenance, each token a valid source class.
    man = pd.read_csv(paths["manifest"])
    assert set(["artifact_path", "artifact_type", "generated_by", "source_class",
                "source_classes_used", "inputs_used", "notes"]).issubset(man.columns)
    assert set(man["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
    for cell in man["source_classes_used"]:
        tokens = str(cell).split(";")
        assert tokens, "source_classes_used must not be empty"
        for tok in tokens:
            assert tok in VALID_SOURCE_CLASSES, f"invalid source class token {tok!r}"
