"""Tests for the reconstruction-limitation extensions (direction 5)."""

import numpy as np
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
    assert set(methods) == {"nearest", "weighted", "nearest_bad_pixel_mask", "trace_offset"}
    for name, d in methods.items():
        assert d["all_finite"], name


def test_example_csv_provenance_valid(tmp_path):
    mod = _load_example("run_reconstruction_limitations")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
