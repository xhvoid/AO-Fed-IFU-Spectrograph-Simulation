"""Tests for the image-slicer IFU remapping and detector projection."""

import numpy as np
import pytest

from ao_fed_ifu.ifu_mapping import (
    IFUConfig,
    build_slicer_mapping,
    forward_project_cube,
    image_slicer_map,
)
from ao_fed_ifu.source_cube import (
    SourceCubeConfig,
    create_emission_line_cube,
    create_point_source_cube,
    scale_cube_to_total_electrons,
)
from ao_fed_ifu.spectrograph_model import SpectrographConfig


def _setup(nx=6, ny=5, R=2000.0, lam=(2.0, 2.3)):
    spectro = SpectrographConfig(lambda_min_um=lam[0], lambda_max_um=lam[1], resolving_power_R=R)
    cfg = SourceCubeConfig(nx=nx, ny=ny, nlambda=spectro.n_detector_pixels,
                           lambda_min_um=lam[0], lambda_max_um=lam[1])
    ifu = IFUConfig(nx_spaxel=nx, ny_spaxel=ny)
    return cfg, ifu, spectro


def test_each_spaxel_maps_to_unique_detector_row():
    cfg, ifu, spectro = _setup()
    mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
    rows = mapping.det_row_of_spaxel
    # No two spaxels share a detector row -> the map is invertible row-wise.
    assert len(np.unique(rows)) == ifu.n_spaxel
    assert mapping.det_shape[1] == spectro.n_detector_pixels


def test_noiseless_forward_projection_conserves_flux():
    cfg, ifu, spectro = _setup()
    cube = create_emission_line_cube(cfg, line_flux_rel=5.0)
    cube = scale_cube_to_total_electrons(cube, 1.0e5)
    mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
    # Pure rearrangement: no LSF, no cross-talk.
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    assert frame.sum() == pytest.approx(cube.sum(), rel=1e-9)


def test_cross_talk_conserves_total_but_spreads_rows():
    cfg, ifu, spectro = _setup(nx=9, ny=9)
    ifu_xt = IFUConfig(nx_spaxel=cfg.nx, ny_spaxel=cfg.ny, cross_talk_sigma_pix=1.0)
    # A point source lands on a single detector row, so cross-dispersion blur
    # visibly lowers the peak while ~conserving the total.
    cube = create_point_source_cube(cfg, line_flux_rel=5.0)
    cube = scale_cube_to_total_electrons(cube, 1.0e5)
    mapping = build_slicer_mapping(ifu_xt, spectro, cfg.wavelength_grid_um)
    plain = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    smeared = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    # Total flux ~conserved by the blur, but the per-row peak is reduced.
    assert smeared.sum() == pytest.approx(plain.sum(), rel=2e-2)
    assert smeared.max() < plain.max()


def test_trace_tables_have_expected_columns():
    cfg, ifu, spectro = _setup()
    mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
    spaxel_tbl = mapping.spaxel_trace_table()
    assert set(["spaxel_id", "ix", "iy", "slice_id", "slit_y", "det_y"]).issubset(spaxel_tbl.columns)
    assert len(spaxel_tbl) == ifu.n_spaxel
    full = mapping.detector_trace_table(lambda_index=[0, 1])
    assert set(["spaxel_id", "slice_id", "slit_y", "lambda_index", "det_x", "det_y"]).issubset(full.columns)


def test_image_slicer_map_convenience_matches_manual():
    cfg, ifu, spectro = _setup()
    cube = create_emission_line_cube(cfg, line_flux_rel=5.0)
    frame, mapping = image_slicer_map(cube, ifu, spectro, cfg.wavelength_grid_um,
                                      apply_lsf=False, apply_cross_talk=False)
    manual = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    assert np.allclose(frame, manual)


def test_lenslet_mode_is_deferred():
    with pytest.raises(NotImplementedError):
        IFUConfig(slicer_mode="lenslet")
