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
from ao_fed_ifu.spectrograph_model import (
    SpectrographConfig,
    wavelength_bin_edges,
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
    sky, sky_provenance = load_sky_spectrum(wl)
    assert sky_provenance.source_class == "direct_public_data"
    assert sky["native_edge_clipped"]
    assert "zero-filled" in sky_provenance.notes
    wl_src = np.linspace(0.32, 2.59, 100)
    assert load_source_spectrum(wl_src)[1].source_class == "direct_public_data"


def test_sky_radiance_bin_means_preserve_narrow_line_flux_at_subpixel_phases(tmp_path):
    """A narrow native line must not depend on where it falls inside a coarse bin."""
    science = np.linspace(1.99, 2.01, 21)
    edges = wavelength_bin_edges(science)
    native = np.arange(1.98, 2.0200001, 2.0e-6)
    sigma = 2.0e-5
    recovered = []
    point_sampled = []
    for phase in (0.08, 0.47, 0.86):
        centre = 2.0 + phase * (science[1] - science[0])
        radiance = np.exp(-0.5 * ((native - centre) / sigma) ** 2)
        np.savetxt(
            tmp_path / "skycalc.dat",
            np.column_stack((native, radiance, np.ones_like(native))),
        )
        sky, provenance = load_sky_spectrum(science, cache_dir=tmp_path)
        assert provenance.source_class == "direct_public_data"
        np.testing.assert_array_equal(sky["native_wavelength_um"], native)
        np.testing.assert_array_equal(sky["native_radiance"], radiance)
        recovered.append(float(np.sum(sky["radiance"] * np.diff(edges))))
        point_sampled.append(float(np.sum(np.interp(science, native, radiance) * np.diff(edges))))

    analytic_flux = np.sqrt(2.0 * np.pi) * sigma
    np.testing.assert_allclose(recovered, analytic_flux, rtol=2e-12, atol=0.0)
    assert np.ptp(recovered) < analytic_flux * 2e-12
    # The old point-sampling path misses this sub-pixel line almost completely.
    assert max(point_sampled) < analytic_flux * 0.01


def test_native_lsf_then_detector_integration_matches_gaussian_reference():
    """The single-LSF helper preserves photons and reproduces Gaussian broadening."""
    from scipy.special import ndtr

    spectro = SpectrographConfig(
        lambda_min_um=1.99,
        lambda_max_um=2.01,
        resolving_power_R=2000.0,
        pixels_per_resolution_element=2.5,
    )
    detector_edges = wavelength_bin_edges(spectro.wavelength_grid_um())
    native = np.arange(1.98, 2.0200001, 2.0e-6)
    intrinsic_sigma_um = 2.0e-5
    output_totals = []
    for phase in (-0.41, 0.07, 0.43):
        centre = 2.0 + phase * spectro.dispersion_um_per_pix
        density = np.exp(-0.5 * ((native - centre) / intrinsic_sigma_um) ** 2)
        density /= np.sqrt(2.0 * np.pi) * intrinsic_sigma_um
        integrated = spectro.integrate_spectral_density(native, density, apply_lsf=True)
        output_totals.append(float(np.sum(integrated)))

        broadened_sigma = np.hypot(
            intrinsic_sigma_um,
            spectro.lsf_sigma_pix * spectro.dispersion_um_per_pix,
        )
        reference = np.diff(ndtr((detector_edges - centre) / broadened_sigma))
        np.testing.assert_allclose(integrated, reference, rtol=0.0, atol=4e-5)

    np.testing.assert_allclose(output_totals, 1.0, rtol=0.0, atol=2e-12)
    assert np.ptp(output_totals) < 2e-12


def test_cached_sky_native_lsf_flux_matches_independent_fine_grid_reference():
    """The delivered fine SkyCalc cache must bypass the lossy point-sample path."""
    if not _has_cache():
        pytest.skip("public cache not present in this checkout")

    from scipy.integrate import cumulative_trapezoid
    from scipy.ndimage import gaussian_filter1d

    # The high-z showcase band exposed the original sub-pixel line loss.
    spectro = SpectrographConfig(
        lambda_min_um=1.9863,
        lambda_max_um=2.3463,
        resolving_power_R=4000.0,
        pixels_per_resolution_element=2.5,
    )
    science = spectro.wavelength_grid_um()
    sky, _ = load_sky_spectrum(science)
    filt, _ = load_filter_curve("K")
    native = sky["native_wavelength_um"]
    native_filter = np.clip(
        np.interp(
            native,
            filt["wavelength_um"],
            filt["transmission"],
            left=0.0,
            right=0.0,
        ),
        0.0,
        1.0,
    )
    native_density = sky["native_radiance"] * native_filter
    integrated = spectro.integrate_spectral_density(native, native_density)

    # Independent reference: explicitly oversample the native piecewise-linear
    # spectrum by eight, convolve it, integrate its numerical antiderivative at
    # detector edges, and compare total/L1 flux rather than stored outputs.
    reference_step = float(np.median(np.diff(native))) / 8.0
    reference_grid = np.arange(
        native[0], native[-1] + 0.5 * reference_step, reference_step
    )
    reference_density = np.interp(
        reference_grid, native, native_density, left=0.0, right=0.0
    )
    reference_density = gaussian_filter1d(
        reference_density,
        sigma=(spectro.lsf_sigma_pix * spectro.dispersion_um_per_pix) / reference_step,
        mode="constant",
        cval=0.0,
    )
    cumulative = cumulative_trapezoid(
        reference_density, reference_grid, initial=0.0
    )
    edges = wavelength_bin_edges(science)
    reference = np.diff(np.interp(edges, reference_grid, cumulative))
    assert np.sum(integrated) == pytest.approx(np.sum(reference), rel=1e-7)
    assert np.sum(np.abs(integrated - reference)) < 1e-4 * np.sum(reference)

    # The superseded ordering (point sample first, then coarse LSF) loses a
    # material fraction for this same cache, while the native path does not.
    science_filter = np.clip(
        np.interp(
            science,
            filt["wavelength_um"],
            filt["transmission"],
            left=0.0,
            right=0.0,
        ),
        0.0,
        1.0,
    )
    point_density = np.interp(science, native, sky["native_radiance"]) * science_filter
    point_integrated = spectro.apply_lsf(point_density) * np.diff(edges)
    old_fractional_error = abs(np.sum(point_integrated) / np.sum(reference) - 1.0)
    new_fractional_error = abs(np.sum(integrated) / np.sum(reference) - 1.0)
    assert old_fractional_error > 0.05
    assert new_fractional_error < old_fractional_error / 1000.0


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
    # Pass the matching variance so zero-filled missing samples stay masked,
    # exactly as the pipeline's own coverage rules do.
    lf = diag.line_flux_map(
        result.rebuilt_weighted_skysub, result.wavelength_grid_um,
        result.line_center_um, window, variance_cube=result.variance_cube_weighted,
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
        result.rebuilt_weighted, result.wavelength_grid_um, result.line_center_um, window,
        variance_cube=result.variance_cube_weighted)
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


# ---------------------------------------------------------------------------
# Cache-table validation: ordering, duplicates and physical ranges.
# ---------------------------------------------------------------------------

def _write_table(path, rows):
    np.savetxt(path, np.asarray(rows, dtype=float))


def test_reversed_source_cache_gives_the_same_photon_shape(tmp_path):
    rows = [[2.0, 1.0], [2.2, 0.6], [2.4, 0.4]]
    (tmp_path / "asc").mkdir()
    (tmp_path / "desc").mkdir()
    _write_table(tmp_path / "asc" / "source.dat", rows)
    _write_table(tmp_path / "desc" / "source.dat", rows[::-1])
    wl = np.array([2.0, 2.2, 2.4])
    asc, _ = load_source_spectrum(wl, cache_dir=tmp_path / "asc")
    desc, prov = load_source_spectrum(wl, cache_dir=tmp_path / "desc")
    assert np.allclose(desc["flux"], asc["flux"])
    assert prov.source_class == "direct_public_data" and "re-sorted" in prov.notes


def test_reversed_filter_cache_keeps_in_band_transmission(tmp_path):
    rows = [[1.9, 0.0], [2.0, 0.9], [2.2, 0.95], [2.4, 0.9], [2.5, 0.0]]
    _write_table(tmp_path / "K.dat", rows[::-1])
    curve, _ = load_filter_curve("K", cache_dir=tmp_path)
    assert np.all(np.diff(curve["wavelength_um"]) > 0)
    assert np.interp(2.2, curve["wavelength_um"], curve["transmission"]) == pytest.approx(0.95)


def test_duplicate_or_unphysical_cache_rows_are_rejected(tmp_path):
    _write_table(tmp_path / "K.dat", [[2.0, 0.5], [2.0, 0.6], [2.2, 0.7]])
    with pytest.raises(ValueError, match="duplicate wavelengths"):
        load_filter_curve("K", cache_dir=tmp_path)
    _write_table(tmp_path / "K.dat", [[2.0, 50.0], [2.2, 90.0]])   # percent, not 0..1
    with pytest.raises(ValueError, match="physical range"):
        load_filter_curve("K", cache_dir=tmp_path)
    _write_table(tmp_path / "source.dat", [[2.0, 1.0], [2.2, np.nan], [2.4, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        load_source_spectrum(np.array([2.1]), cache_dir=tmp_path)
    _write_table(tmp_path / "skycalc.dat", [[2.0, 1.0, 1.2], [2.4, 1.0, 0.9]])
    with pytest.raises(ValueError, match="transmission"):
        load_sky_spectrum(np.array([2.1, 2.2]), cache_dir=tmp_path)


def test_relative_cache_dir_outside_repo_is_hashed_from_the_file_actually_read(tmp_path, monkeypatch):
    # A relative public_cache_dir resolved from a cwd outside the repository
    # must be recorded (and hashed) as that file, not as a repo-relative path.
    from ao_fed_ifu import reporting as rep

    if not (CACHE / "K.dat").exists():
        pytest.skip("public cache not present in this checkout")
    (tmp_path / "cache").mkdir()
    altered = np.loadtxt(CACHE / "K.dat")
    altered[:, 1] *= 0.5
    np.savetxt(tmp_path / "cache" / "K.dat", altered)
    monkeypatch.chdir(tmp_path)
    _, prov = load_filter_curve("K", cache_dir="cache")
    identity = rep.input_cache_identity([prov.as_dict()])
    [(label, digest)] = identity["files_sha256"].items()
    assert label == str((tmp_path / "cache" / "K.dat").resolve())
    assert digest == rep.file_sha256(tmp_path / "cache" / "K.dat")
