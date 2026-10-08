"""Tests for the reconstruction-limitation extensions (direction 5)."""

import numpy as np
import pytest
import pandas as pd

from ao_fed_ifu import cube_reconstruction as recon
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.detector_model import DetectorConfig, expose
from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
from ao_fed_ifu.source_cube import SourceCubeConfig, create_emission_line_cube, scale_cube_to_total_electrons
from ao_fed_ifu.spectrograph_model import SpectrographConfig


def _setup(cross_talk=0.0):
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=2000.0)
    cfg = SourceCubeConfig(nx=6, ny=5, nlambda=spectro.n_detector_pixels, lambda_min_um=2.0, lambda_max_um=2.3)
    ifu = IFUConfig(nx_spaxel=6, ny_spaxel=5, cross_talk_sigma_pix=cross_talk)
    cube = scale_cube_to_total_electrons(create_emission_line_cube(cfg, line_flux_rel=5.0), 1.0e5)
    mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
    return cube, mapping


def test_nearest_no_noise_no_offset_is_exact():
    cube, mapping = _setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_trace_offset(frame, mapping, row_offset=0)
    assert np.allclose(rebuilt, cube, atol=1e-6)


def test_trace_offset_introduces_bias():
    cube, mapping = _setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    exact = recon.reconstruct_cube_nearest(frame, mapping)
    offset = recon.reconstruct_cube_trace_offset(frame, mapping, row_offset=2)
    # A non-zero trace offset must change the recovered cube.
    assert not np.allclose(offset, exact)


def test_bad_pixel_mask_prevents_nan_inf():
    cube, mapping = _setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    readout = expose(frame, DetectorConfig(bad_pixel_fraction=0.2), exposure_s=1.0, rng=0)
    repaired = recon.reconstruct_cube_with_bad_pixel_mask(readout.image_e, readout.bad_pixel_mask, mapping)
    assert np.all(np.isfinite(repaired))


def test_compare_methods_returns_all_finite():
    cube, mapping = _setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping)
    readout = expose(frame, DetectorConfig(bad_pixel_fraction=0.05), exposure_s=1.0, rng=0)
    methods = recon.compare_reconstruction_methods(
        readout.image_e, readout.variance_e2, readout.bad_pixel_mask, mapping, cube)
    assert set(methods) == {"nearest", "weighted", "horne_per_trace", "nearest_bad_pixel_mask", "trace_offset"}
    # The raw ratio includes honest zero placeholders for unstable deblends, so
    # it must not be asserted near unity under a 5% bad-pixel mask.  The measured
    # ratio excludes exactly those non-finite-variance voxels and reports its
    # reduced coverage separately.  (Noise and the LSF mean even that ratio is
    # not an exact-inverse test; the noiseless/no-LSF tests cover that invariant.)
    weighted = methods["weighted"]
    assert weighted["measured_voxel_fraction"] < 1.0
    assert weighted["measured_flux_conservation_ratio"] > weighted["flux_conservation_ratio"]
    expected = recon.flux_conservation(
        cube, weighted["cube"], bad_voxels=~np.isfinite(weighted["variance_cube"])
    )["ratio"]
    assert weighted["measured_flux_conservation_ratio"] == expected
    # the per-trace window still over-counts shared detector flux
    assert methods["horne_per_trace"]["flux_conservation_ratio"] > 1.5
    for name, d in methods.items():
        assert d["all_finite"], name


def test_example_csv_provenance_valid(tmp_path):
    mod = _load_example("run_reconstruction_limitations")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_reconstruction_snr_metrics_are_interpretable(tmp_path):
    import json

    mod = _load_example("run_reconstruction_limitations")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"]).set_index("metric")
    methods = ["nearest", "weighted", "horne_per_trace", "nearest_bad_pixel_mask", "trace_offset"]
    for m in methods:
        # The interpretability companions to median_line_snr all exist.
        frac = float(df.loc[f"{m}.finite_snr_fraction", "value"])
        median_all = float(df.loc[f"{m}.median_line_snr", "value"])
        median_finite = float(df.loc[f"{m}.median_line_snr_finite_only", "value"])
        n_ap = float(df.loc[f"{m}.aperture_spaxels", "value"])
        assert 0.0 <= frac <= 1.0
        assert n_ap > 0
        # Excluding bad-pixel-zeroed spaxels can only raise the median.
        if np.isfinite(median_finite):
            assert median_finite >= median_all - 1e-9
        # A 0.0 aperture-median must be explained by a sub-majority usable fraction,
        # never presented as a physical non-detection.
        if median_all == 0.0:
            assert frac < 0.5

    # The summary explains the 0.0 case and carries the per-method breakdown.
    summary = json.loads(paths["summary_json"].read_text())
    assert "snr_interpretation" in summary
    assert "not a physical non-detection" in summary["snr_interpretation"]
    assert set(summary["snr_by_method"]) == set(methods)
