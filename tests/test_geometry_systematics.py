"""Tests for the spectrograph / IFU geometry systematics (Priority 5)."""

import numpy as np
import pandas as pd

from ao_fed_ifu import cube_reconstruction as recon
from ao_fed_ifu import geometry_systematics as geo
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
from ao_fed_ifu.source_cube import SourceCubeConfig, create_emission_line_cube, scale_cube_to_total_electrons
from ao_fed_ifu.spectrograph_model import SpectrographConfig


def _setup(curvature=0.0):
    sp = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=1800.0)
    cfg = SourceCubeConfig(nx=6, ny=5, nlambda=sp.n_detector_pixels, lambda_min_um=2.0, lambda_max_um=2.3)
    cube = scale_cube_to_total_electrons(create_emission_line_cube(cfg, line_flux_rel=4.0), 1.0e5)
    mapping = build_slicer_mapping(IFUConfig(nx_spaxel=6, ny_spaxel=5, trace_curvature_pix=curvature), sp, cfg.wavelength_grid_um)
    return cube, mapping


def test_zero_curvature_reproduces_nominal_mapping():
    cube, straight = _setup(curvature=0.0)
    assert np.all(straight.row_offsets() == 0)
    frame = forward_project_cube(cube, straight, apply_lsf=False, apply_cross_talk=False)
    fc = recon.flux_conservation(cube, recon.reconstruct_cube_nearest(frame, straight))
    assert fc["ratio"] == 1.0


def test_small_curvature_stays_in_bounds():
    _, mc = _setup(curvature=4.0)
    n_rows = mc.det_shape[0]
    max_row = int(mc.det_row_of_spaxel.max() + mc.row_offsets().max())
    assert 0 <= max_row < n_rows
    assert np.all(mc.lambda_row_shift() >= 0)


def test_curvature_round_trip_and_uncorrected_bias():
    import dataclasses
    cube, mc = _setup(curvature=4.0)
    frame = forward_project_cube(cube, mc, apply_lsf=False, apply_cross_talk=False)
    corrected = recon.flux_conservation(cube, recon.reconstruct_cube_nearest(frame, mc))["ratio"]
    straight_read = dataclasses.replace(mc, col_row_offset=np.zeros_like(mc.row_offsets()))
    uncorrected = recon.flux_conservation(cube, recon.reconstruct_cube_nearest(frame, straight_read))["ratio"]
    assert np.isclose(corrected, 1.0, atol=1e-9)     # curve-aware recovers flux
    assert uncorrected < 0.99                         # nominal recon is biased


def test_trace_curvature_offsets_parabolic_and_zero_default():
    assert np.all(geo.trace_curvature_offsets(100, 0.0) == 0)
    off = geo.trace_curvature_offsets(101, 5.0)
    assert off[50] == 0 and off[0] == 5 and off[-1] == 5   # 0 at centre, max at edges


def test_polynomial_wavelength_solution_monotonic():
    lam = geo.polynomial_wavelength_solution(500, 2.0, 2.3, stretch_frac=0.01, curvature_frac=0.02)
    assert np.all(np.diff(lam) > 0)
    # Linear case reproduces the linear solution.
    lin = geo.polynomial_wavelength_solution(500, 2.0, 2.3)
    assert np.allclose(lin, np.linspace(2.0, 2.3, 500))


def test_wider_lsf_lowers_recovered_R():
    r_narrow = geo.recovered_R_from_lsf(2.2, 2.0, 0.0005)
    r_wide = geo.recovered_R_from_lsf(2.2, 4.0, 0.0005)
    assert r_wide < r_narrow


def test_flexure_offset_creates_finite_bias():
    from ao_fed_ifu import diagnostics as diag
    from ao_fed_ifu.integration import make_toy_config, run_pipeline

    res = run_pipeline(make_toy_config(nx=8, ny=8, resolving_power_R=1500.0))
    grid, lc = res.wavelength_grid_um, res.line_center_um
    win = 3.0 * res.spectrograph.delta_lambda_resel_um(lc)
    mp = build_slicer_mapping(res.ifu_config, res.spectrograph, grid)
    drow, _ = geo.flexure_offset_pix(2.0, seed=3)
    assert drow != 0
    flexed = recon.reconstruct_cube_trace_offset(res.detector_noisy, mp, drow)
    ap = res.aperture_mask
    bias = np.nanmedian((diag.velocity_centroid_map(flexed, grid, lc, win)
                         - res.velocity_map)[ap])
    assert np.isfinite(bias)


def test_report_csv_provenance_valid(tmp_path):
    mod = _load_example("run_geometry_systematics_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["trace_map", "wavelength_resid", "slice_lsf", "metrics_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
