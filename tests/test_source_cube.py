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


# ---------------------------------------------------------------------------
# Continuum/line photon-budget semantics (photon_budget_mode="continuum_line").
# ---------------------------------------------------------------------------

def test_line_budget_analytic_conversion():
    from ao_fed_ifu.source_cube import photon_budget_line_electrons

    cfg = SourceCubeConfig(exposure_s=900.0, telescope_diameter_m=8.0, throughput_scalar=0.25)
    f_line = 1.0e-16  # erg/s/cm^2
    lam = 2.2  # um
    e = photon_budget_line_electrons(cfg, f_line, lam)
    # electrons = F[J/s/m^2] / (hc/lambda) * area * t * throughput
    h, c = 6.62607015e-34, 2.99792458e8
    expected = (f_line * 1e-3) / (h * c / (lam * 1e-6)) * cfg.collecting_area_m2 * 900.0 * 0.25
    assert e == pytest.approx(expected, rel=1e-12)
    # Linear in flux, exposure-scaling like any photon count.
    assert photon_budget_line_electrons(cfg, 2 * f_line, lam) == pytest.approx(2 * e, rel=1e-12)


def test_equivalent_width_line_electrons():
    from ao_fed_ifu.source_cube import line_electrons_from_equivalent_width

    # 100 e-/bin continuum with 0.001-um bins and EW = 0.01 um -> 1000 e- of line.
    assert line_electrons_from_equivalent_width(100.0, 0.001, 0.01) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        line_electrons_from_equivalent_width(100.0, 0.001, -1.0)


def test_surface_brightness_budget_equals_effective_magnitude():
    from ao_fed_ifu.source_cube import photon_budget_from_surface_brightness

    cfg = SourceCubeConfig(nx=20, ny=20, spaxel_scale_mas=10.0)
    sb = 20.0  # AB mag / arcsec^2
    omega = 20 * 20 * (10.0 / 1000.0) ** 2  # 0.04 arcsec^2
    expected = photon_budget_from_abmag(cfg, sb - 2.5 * np.log10(omega))
    assert photon_budget_from_surface_brightness(cfg, sb) == pytest.approx(expected, rel=1e-12)


def test_default_budget_mode_is_shape_only():
    # Documents the default semantics: changing line_flux_rel redistributes a
    # FIXED AB-magnitude total; it does not add photons.
    from ao_fed_ifu.integration import make_toy_config, run_pipeline

    weak = run_pipeline(make_toy_config(line_flux_rel=4.0))
    strong = run_pipeline(make_toy_config(line_flux_rel=32.0))
    assert weak.metrics["photon_budget_mode"] == "total_abmag"
    assert weak.metrics["input_source_electrons"] == pytest.approx(
        strong.metrics["input_source_electrons"], rel=1e-9)


def test_continuum_line_mode_adds_physical_line_photons():
    from ao_fed_ifu.integration import make_toy_config, run_pipeline

    kw = dict(photon_budget_mode="continuum_line", seed=0)
    no_line = run_pipeline(make_toy_config(line_flux_cgs=1.0e-17, **kw))
    more = run_pipeline(make_toy_config(line_flux_cgs=2.0e-17, **kw))
    m0, m1 = no_line.metrics, more.metrics
    assert m0["photon_budget_mode"] == "continuum_line"
    # Continuum budget unchanged; line electrons double with the line flux.
    assert m0["continuum_total_electrons"] == pytest.approx(m1["continuum_total_electrons"], rel=1e-9)
    assert m1["line_total_electrons"] == pytest.approx(2.0 * m0["line_total_electrons"], rel=1e-9)
    # Total input electrons = continuum + line (additive, not redistributive).
    assert m0["input_source_electrons"] == pytest.approx(
        m0["continuum_total_electrons"] + m0["line_total_electrons"], rel=1e-6)


def test_equivalent_width_matches_direct_line_flux_route():
    from ao_fed_ifu.integration import make_toy_config, run_pipeline
    from ao_fed_ifu.source_cube import line_electrons_from_equivalent_width

    ew = 0.002  # um
    res = run_pipeline(make_toy_config(
        photon_budget_mode="continuum_line", line_equivalent_width_um=ew))
    m = res.metrics
    per_bin = m["continuum_total_electrons"] / res.wavelength_grid_um.size
    expected = line_electrons_from_equivalent_width(
        per_bin, res.source_config.delta_lambda_um, ew)
    assert m["line_total_electrons"] == pytest.approx(expected, rel=1e-6)


def test_budget_mode_misconfiguration_raises():
    from ao_fed_ifu.integration import make_toy_config, run_pipeline

    # Physical line controls demand the continuum_line mode...
    with pytest.raises(ValueError, match="continuum_line"):
        run_pipeline(make_toy_config(line_flux_cgs=1e-17))
    # ...which in turn demands exactly one line-strength control.
    with pytest.raises(ValueError, match="exactly one"):
        run_pipeline(make_toy_config(photon_budget_mode="continuum_line"))
    with pytest.raises(ValueError, match="exactly one"):
        run_pipeline(make_toy_config(
            photon_budget_mode="continuum_line",
            line_flux_cgs=1e-17, line_equivalent_width_um=0.001))
    with pytest.raises(ValueError, match="photon_budget_mode"):
        run_pipeline(make_toy_config(photon_budget_mode="nonsense"))
