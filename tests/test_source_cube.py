"""Tests for source datacube construction and the photon budget."""

import numpy as np
import pytest

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S
from ao_fed_ifu.source_cube import (
    SourceCubeConfig,
    create_emission_line_cube,
    create_extended_galaxy_cube,
    create_point_source_cube,
    photon_budget_from_abmag,
    scale_cube_to_total_electrons,
)


def test_cube_shape_and_grid():
    cfg = SourceCubeConfig(nx=8, ny=6, nlambda=20)
    cube = create_emission_line_cube(cfg)
    # Convention: cube is (ny, nx, nlambda).
    assert cube.shape == (6, 8, 20)
    grid = cfg.wavelength_grid_um
    assert grid.shape == (20,)
    assert grid[0] == pytest.approx(cfg.lambda_min_um)
    assert grid[-1] == pytest.approx(cfg.lambda_max_um)


def test_point_source_is_localised():
    cfg = SourceCubeConfig(nx=9, ny=9, nlambda=10)
    cube = create_point_source_cube(cfg)
    image = cube.sum(axis=2)
    # All the flux should sit in the central spaxel.
    iy, ix = np.unravel_index(np.argmax(image), image.shape)
    assert (iy, ix) == (cfg.ny // 2, cfg.nx // 2)
    assert image[iy, ix] > 0
    assert np.count_nonzero(image) == 1


def test_photon_budget_scales_with_area_and_exposure():
    base = SourceCubeConfig(exposure_s=100.0, telescope_diameter_m=8.0)
    more_time = SourceCubeConfig(exposure_s=200.0, telescope_diameter_m=8.0)
    bigger = SourceCubeConfig(exposure_s=100.0, telescope_diameter_m=16.0)
    e_base = photon_budget_from_abmag(base, ab_magnitude=20.0)
    e_time = photon_budget_from_abmag(more_time, ab_magnitude=20.0)
    e_area = photon_budget_from_abmag(bigger, ab_magnitude=20.0)
    assert e_base > 0
    # Double exposure -> double electrons.
    assert e_time == pytest.approx(2.0 * e_base, rel=1e-6)
    # Double diameter -> 4x area -> 4x electrons.
    assert e_area == pytest.approx(4.0 * e_base, rel=1e-6)


def test_photon_budget_brighter_is_more_electrons():
    cfg = SourceCubeConfig()
    bright = photon_budget_from_abmag(cfg, ab_magnitude=18.0)
    faint = photon_budget_from_abmag(cfg, ab_magnitude=22.0)
    # Smaller AB magnitude = brighter = more electrons; 4 mag = factor ~40.
    assert bright > faint
    assert bright / faint == pytest.approx(10 ** (0.4 * 4), rel=1e-6)


def test_scale_to_total_electrons():
    cfg = SourceCubeConfig(nx=5, ny=5, nlambda=8)
    cube = create_emission_line_cube(cfg)
    scaled = scale_cube_to_total_electrons(cube, 1.0e5)
    assert np.sum(scaled) == pytest.approx(1.0e5, rel=1e-9)


def test_velocity_gradient_shifts_line_centroid():
    # A pure velocity gradient should Doppler-shift the line centre across x.
    # Use a finely-sampled, well-resolved line so the discrete centroid is unbiased.
    cfg = SourceCubeConfig(nx=11, ny=3, nlambda=400, lambda_min_um=2.0, lambda_max_um=2.4)
    centre = 2.2
    grad = 50.0  # km/s per spaxel
    cube = create_emission_line_cube(
        cfg, line_center_um=centre, line_flux_rel=20.0, line_sigma_um=0.003,
        velocity_gradient_kms_per_spaxel=grad, continuum_level=0.0,
    )
    wl = cfg.wavelength_grid_um
    # Centroid wavelength in the leftmost vs rightmost spaxel.
    def centroid(ix):
        spec = cube[1, ix, :]
        return np.sum(wl * spec) / np.sum(spec)
    left = centroid(0)
    right = centroid(cfg.nx - 1)
    # Rightmost spaxel is red-shifted relative to leftmost (positive gradient).
    assert right > left
    measured_dv = SPEED_OF_LIGHT_KM_S * (right - left) / centre
    expected_dv = grad * (cfg.nx - 1)
    assert measured_dv == pytest.approx(expected_dv, rel=0.05)


def test_extended_galaxy_is_centrally_peaked():
    cfg = SourceCubeConfig(nx=15, ny=15, nlambda=10)
    cube = create_extended_galaxy_cube(cfg, half_light_radius_spaxel=3.0)
    image = cube.sum(axis=2)
    # Brightest spaxel should be at (or adjacent to) the centre.
    iy, ix = np.unravel_index(np.argmax(image), image.shape)
    assert abs(iy - cfg.ny // 2) <= 1 and abs(ix - cfg.nx // 2) <= 1
