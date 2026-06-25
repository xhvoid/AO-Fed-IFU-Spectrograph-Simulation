"""Tests for cube reconstruction (inverse mapping) and flux conservation."""

import numpy as np
import pytest

from ao_fed_ifu import cube_reconstruction as recon
from ao_fed_ifu.detector_model import DetectorConfig, expose
from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
from ao_fed_ifu.source_cube import SourceCubeConfig, create_emission_line_cube, scale_cube_to_total_electrons
from ao_fed_ifu.spectrograph_model import SpectrographConfig


def _pipeline_setup(nx=6, ny=5, cross_talk=0.0):
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=2000.0)
    cfg = SourceCubeConfig(nx=nx, ny=ny, nlambda=spectro.n_detector_pixels,
                           lambda_min_um=2.0, lambda_max_um=2.3)
    ifu = IFUConfig(nx_spaxel=nx, ny_spaxel=ny, cross_talk_sigma_pix=cross_talk)
    cube = create_emission_line_cube(cfg, line_flux_rel=5.0, velocity_gradient_kms_per_spaxel=10.0)
    cube = scale_cube_to_total_electrons(cube, 1.0e5)
    mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
    return cfg, cube, mapping


def test_nearest_reconstruction_is_exact_inverse():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
    assert rebuilt.shape == cube.shape
    # With no LSF/cross-talk the round trip is exact.
    assert np.allclose(rebuilt, cube, rtol=1e-9, atol=1e-6)


def test_flux_conservation_ratio_unity_noiseless():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
    fc = recon.flux_conservation(cube, rebuilt)
    assert fc["ratio"] == pytest.approx(1.0, abs=1e-9)


def test_weighted_equals_nearest_when_no_crosstalk():
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.0)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    var = np.ones_like(frame)
    nearest = recon.reconstruct_cube_nearest(frame, mapping)
    weighted = recon.reconstruct_cube_weighted(frame, var, mapping, profile_sigma_pix=0.0)
    assert np.allclose(nearest, weighted)


def test_extract_1d_spectrum_sums_aperture():
    cfg, cube, mapping = _pipeline_setup()
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    mask[1:3, 1:3] = True
    spec = recon.extract_1d_spectrum(cube, mask)
    assert spec.shape == (cfg.nlambda,)
    assert np.allclose(spec, cube[1:3, 1:3, :].sum(axis=(0, 1)))


def test_bad_voxels_excluded_from_conservation():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    cfgd = DetectorConfig(read_noise_e=0.0, dark_current_e_s_pix=0.0, bad_pixel_fraction=0.05,
                          flat_field_sigma=0.0)
    out = expose(frame, cfgd, exposure_s=1.0, rng=0)
    rebuilt = recon.reconstruct_cube_nearest(out.image_e, mapping)
    bad_vox = recon.bad_voxel_mask(out.bad_pixel_mask, mapping)
    fc = recon.flux_conservation(cube, rebuilt, bad_voxels=bad_vox)
    # Excluding the masked voxels, the (Poisson-only) flux is conserved on average.
    assert fc["ratio"] == pytest.approx(1.0, abs=0.05)


def test_variance_cube_shape_matches_cube():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    out = expose(frame, DetectorConfig(), exposure_s=1.0, rng=0)
    var_cube = recon.estimate_variance_cube(out.variance_e2, mapping)
    assert var_cube.shape == cube.shape
    assert np.all(var_cube > 0)
