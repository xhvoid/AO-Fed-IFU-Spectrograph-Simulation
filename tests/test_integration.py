"""Tests for the end-to-end pipeline and the provenance of output tables.

These guard the two project requirements that unit tests of individual stages
do not otherwise cover:
  * every output table carries provenance labels, and
  * PSF coupling reports its field throughput honestly (no silent flux loss).
"""

import json
from pathlib import Path

import numpy as np
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


def test_native_sky_is_broadened_once_before_detector_projection():
    import numpy as np
    from ao_fed_ifu.integration import make_public_conditioned_config
    from ao_fed_ifu.ifu_mapping import build_slicer_mapping, forward_project_cube

    result = run_pipeline(make_public_conditioned_config(
        nx=3, ny=3, resolving_power_R=2000, apply_psf=False
    ))
    mapping = build_slicer_mapping(result.ifu_config, result.spectrograph, result.wavelength_grid_um)
    source = forward_project_cube(result.coupled_cube, mapping, apply_lsf=True)
    sky_cube = np.broadcast_to(result.sky_background_spectrum, result.coupled_cube.shape)
    sky = forward_project_cube(sky_cube, mapping, apply_lsf=False)
    np.testing.assert_allclose(result.detector_clean, source + sky, rtol=1e-13)
    assert result.conditioning_info["sky_lsf_applied"] is True
    # A second LSF measurably changes OH profiles, so the check would reject
    # simply adding the post-LSF sky cube to the source before its projection.
    double_lsf_sky = forward_project_cube(sky_cube, mapping, apply_lsf=True)
    assert np.linalg.norm(double_lsf_sky - sky) > 0.01 * np.linalg.norm(sky)


def test_masked_line_windows_have_explicit_coverage_and_usable_snr():
    import numpy as np
    result = run_pipeline(make_toy_config(nx=4, ny=4))
    window = 3 * result.spectrograph.delta_lambda_resel_um(result.line_center_um)
    selected = np.abs(result.wavelength_grid_um - result.line_center_um) <= window
    valid = np.all(np.isfinite(result.variance_cube_weighted[:, :, selected]), axis=2)
    assert result.metrics["line_window_measured_spaxel_fraction"] == pytest.approx(valid.mean())
    assert result.metrics["unmeasured_voxel_fraction"] == pytest.approx(result.extraction_model.unobserved.mean())
    assert result.metrics["median_snr_measured_spaxels"] == pytest.approx(np.median(result.snr_map[valid]))


def _reject_constant(token):
    raise ValueError(f"non-standard JSON constant {token}")


def test_summary_json_is_strict_and_reproducible(tmp_path):
    import dataclasses

    from ao_fed_ifu import reporting as rep

    cfg = make_toy_config(seed=7, exposure_s=1234.0)
    result = run_pipeline(cfg)
    result.metrics["recovered_R_bright_spaxel"] = float("nan")   # an unavailable metric
    paths = write_artifacts(result, tmp_path, prefix="t")
    # Strict parsing: NaN/Infinity would be rejected here.
    summary = json.loads(paths["summary_json"].read_text(), parse_constant=_reject_constant)
    assert summary["metrics"]["recovered_R_bright_spaxel"] is None
    assert summary["unavailable_values"]["metrics.recovered_R_bright_spaxel"] == "nan"
    # The complete nested configuration round-trips (seed, exposure, detector...).
    assert summary["configuration"] == json.loads(json.dumps(rep.to_strict_json(dataclasses.asdict(cfg))))
    assert summary["configuration"]["seed"] == 7
    assert summary["configuration"]["detector"]["read_noise_e"] == cfg.detector.read_noise_e
    assert summary["software"]["source_sha256"] == rep.source_sha256()
    assert summary["input_caches"] == {"used": False, "files_sha256": {}}


def test_public_run_records_input_cache_hashes(tmp_path):
    from ao_fed_ifu import reporting as rep
    from ao_fed_ifu.integration import make_public_conditioned_config

    result = run_pipeline(make_public_conditioned_config(nx=4, ny=4, resolving_power_R=200.0))
    summary = json.loads(write_artifacts(result, tmp_path, prefix="p")["summary_json"].read_text(),
                         parse_constant=_reject_constant)
    caches = summary["input_caches"]
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "public_cache"
    if not (cache_dir / "skycalc.dat").exists():
        pytest.skip("public cache not present in this checkout")
    assert caches["used"] is True
    # Only the files actually served as public data, repo-relative, by content.
    assert set(caches["files_sha256"]) == {"data/public_cache/K.dat", "data/public_cache/skycalc.dat",
                                           "data/public_cache/source.dat"}
    assert caches["files_sha256"]["data/public_cache/skycalc.dat"] == rep.file_sha256(cache_dir / "skycalc.dat")
    assert "/Users/" not in paths_text(summary)


def paths_text(summary):
    return json.dumps(summary["provenance"])


def test_rejected_cache_is_not_reported_as_an_input(tmp_path):
    # An empty cache directory: every component falls back to a synthetic proxy.
    from ao_fed_ifu import reporting as rep
    from ao_fed_ifu.integration import make_public_conditioned_config

    result = run_pipeline(make_public_conditioned_config(nx=4, ny=4, resolving_power_R=200.0,
                                                         public_cache_dir=str(tmp_path)))
    assert rep.input_cache_identity(result.provenance.to_records()) == {"used": False, "files_sha256": {}}


@pytest.mark.parametrize("spaxel_scale_mas", [10.0, 50.0])
def test_reported_psf_fwhm_resolves_the_core_at_any_spaxel_scale(spaxel_scale_mas):
    # A 1 um, 39 m diffraction core (~5.4 mas) must not be reported as the
    # spaxel/5 pixelisation (it used to read ~29 mas at 50 mas spaxels).
    from ao_fed_ifu.psf_coupling import diffraction_fwhm_mas

    cfg = make_toy_config(nx=6, ny=6, lambda_min_um=0.98, lambda_max_um=1.02, telescope_diameter_m=39.0,
                          strehl=0.5, spaxel_scale_mas=spaxel_scale_mas, resolving_power_R=500.0)
    core = diffraction_fwhm_mas(1.0, 39.0)
    assert run_pipeline(cfg).metrics["psf_fwhm_mas"] == pytest.approx(core, rel=0.01)


def test_halo_dominated_psf_fwhm_is_independent_of_spaxel_scale():
    # ELT K band at Strehl 3e-4: the core sets the peak but the halo sets the
    # half-maximum crossing. Compare with the analytic two-Gaussian FWHM.
    from scipy.optimize import brentq

    from ao_fed_ifu.psf_coupling import diffraction_fwhm_mas

    strehl, seeing = 3e-4, 600.0
    sc, sh = diffraction_fwhm_mas(2.2, 39.0) / 2.3548, seeing / 2.3548

    def intensity(r):
        return strehl / sc**2 * np.exp(-r * r / 2 / sc**2) + (1 - strehl) / sh**2 * np.exp(-r * r / 2 / sh**2)

    analytic = 2 * brentq(lambda r: intensity(r) - intensity(0) / 2, 1e-6, 5 * seeing)
    reported = [run_pipeline(make_toy_config(
        nx=4, ny=4, lambda_min_um=2.18, lambda_max_um=2.22, telescope_diameter_m=39.0, strehl=strehl,
        seeing_fwhm_mas=seeing, spaxel_scale_mas=scale, resolving_power_R=500.0)).metrics["psf_fwhm_mas"]
        for scale in (10.0, 50.0)]
    assert reported[0] == pytest.approx(reported[1], rel=1e-3)
    assert reported[0] == pytest.approx(analytic, rel=0.01)


def _line_electrons_in_cube(cfg):
    from dataclasses import replace as dc_replace

    with_line = run_pipeline(cfg)
    without = run_pipeline(dc_replace(cfg, line_flux_cgs=0.0))
    return with_line, float((with_line.intrinsic_source_cube - without.intrinsic_source_cube).sum())


def test_physical_line_at_band_edge_keeps_only_its_captured_fraction():
    # Line centred on the first channel: about half the profile is off band and
    # must be lost, not redistributed into the surviving channels (1.84x).
    from scipy.special import ndtr

    cfg = make_toy_config(nx=2, ny=2, source_kind="point", lambda_min_um=2.0, lambda_max_um=2.3,
                          line_center_um=2.0, line_sigma_um=0.002, resolving_power_R=2000,
                          apply_psf=False, photon_budget_mode="continuum_line", line_flux_cgs=1e-16)
    result, in_cube = _line_electrons_in_cube(cfg)
    fraction = ndtr(0.5 * result.spectrograph.dispersion_um_per_pix / cfg.line_sigma_um)
    total = result.metrics["line_total_electrons"]
    assert result.metrics["line_in_band_electrons"] == pytest.approx(total * fraction, rel=1e-3)
    assert in_cube == pytest.approx(total * fraction, rel=1e-3)


def test_physical_line_doppler_shifted_off_band_loses_its_wing():
    # A velocity gradient pushes some spaxels' lines past the red band edge;
    # each spaxel keeps only its own captured fraction of the line.
    from scipy.special import ndtr

    from ao_fed_ifu import SPEED_OF_LIGHT_KM_S

    cfg = make_toy_config(nx=6, ny=2, source_kind="emission_line", lambda_min_um=2.0, lambda_max_um=2.3,
                          line_center_um=2.297, line_sigma_um=0.002, resolving_power_R=2000,
                          velocity_gradient_kms_per_spaxel=400.0, apply_psf=False,
                          photon_budget_mode="continuum_line", line_flux_cgs=1e-16)
    result, in_cube = _line_electrons_in_cube(cfg)
    x = np.arange(cfg.nx) - (cfg.nx - 1) / 2.0
    centres = cfg.line_center_um * (1 + cfg.velocity_gradient_kms_per_spaxel * x / SPEED_OF_LIGHT_KM_S)
    upper_edge = cfg.lambda_max_um + 0.5 * result.spectrograph.dispersion_um_per_pix
    fraction = np.mean(ndtr((upper_edge - centres) / cfg.line_sigma_um))   # uniform spatial profile
    assert 0.2 < fraction < 0.9                                             # really truncated
    assert in_cube == pytest.approx(result.metrics["line_total_electrons"] * fraction, rel=2e-3)


def test_physical_line_wholly_in_band_keeps_its_full_budget():
    cfg = make_toy_config(nx=2, ny=2, source_kind="point", lambda_min_um=2.0, lambda_max_um=2.3,
                          line_center_um=2.15, line_sigma_um=0.002, resolving_power_R=2000,
                          apply_psf=False, photon_budget_mode="continuum_line", line_flux_cgs=1e-16)
    result, in_cube = _line_electrons_in_cube(cfg)
    assert in_cube == pytest.approx(result.metrics["line_total_electrons"], rel=1e-9)


def test_saved_configuration_is_the_one_the_run_used(tmp_path):
    cfg = make_toy_config(seed=7, exposure_s=600.0)
    result = run_pipeline(cfg)
    cfg.seed, cfg.exposure_s = 8, 1200.0               # reuse the object, e.g. in a sweep
    summary = json.loads(write_artifacts(result, tmp_path, prefix="s")["summary_json"].read_text())
    assert (summary["configuration"]["seed"], summary["configuration"]["exposure_s"]) == (7, 600.0)
    assert result.config is not cfg


def test_cache_identity_is_bound_to_the_bytes_the_run_loaded(tmp_path):
    from ao_fed_ifu import reporting as rep
    from ao_fed_ifu.integration import make_public_conditioned_config

    source = Path(__file__).resolve().parents[1] / "data" / "public_cache"
    if not (source / "skycalc.dat").exists():
        pytest.skip("public cache not present in this checkout")
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("K.dat", "skycalc.dat", "source.dat"):
        (cache / name).write_bytes((source / name).read_bytes())
    original = rep.file_sha256(cache / "K.dat")
    result = run_pipeline(make_public_conditioned_config(nx=4, ny=4, resolving_power_R=200.0,
                                                         public_cache_dir=str(cache)))
    altered = np.loadtxt(cache / "K.dat")
    altered[:, 1] *= 0.2                               # replace the file after the run
    np.savetxt(cache / "K.dat", altered)
    caches = rep.input_cache_identity(result.provenance.to_records())
    [k_label] = [label for label in caches["files_sha256"] if label.endswith("K.dat")]
    assert caches["files_sha256"][k_label] == original
