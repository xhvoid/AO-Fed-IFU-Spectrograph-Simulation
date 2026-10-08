"""Tests for the configurable high-z galaxy source (Priority 7)."""

import numpy as np
import pandas as pd
import pytest

from ao_fed_ifu import galaxy_kinematics as gk
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.source_cube import REST_FRAME_LINES_UM, observed_wavelength_um


def test_face_on_has_lower_velocity_span_than_edge_on():
    face = gk.projected_velocity_field(gk.GalaxySourceConfig(inclination_deg=5.0))
    edge = gk.projected_velocity_field(gk.GalaxySourceConfig(inclination_deg=85.0))
    assert np.ptp(face) < np.ptp(edge)


def test_larger_sersic_radius_broadens_light():
    small = gk.sersic_light_profile(gk.GalaxySourceConfig(half_light_radius_spaxel=2.0))
    large = gk.sersic_light_profile(gk.GalaxySourceConfig(half_light_radius_spaxel=6.0))
    # Fraction of spaxels above half-peak is larger for the bigger galaxy.
    assert (large > 0.5).mean() > (small > 0.5).mean()


def test_arctan_rotation_curve_rises_and_flattens():
    r = np.array([0.0, 1.0, 2.0, 10.0, 100.0])
    v = gk.arctan_rotation_curve(r, v_max=200.0, r_turnover=2.0)
    assert v[0] == 0.0
    assert np.all(np.diff(v) >= 0)          # monotonically rising
    assert v[-1] < 200.0 and v[-1] > 0.9 * 200.0   # flattens toward v_max


def test_multiline_cube_places_lines_at_redshifted_wavelengths():
    z = 2.30
    wl = np.linspace(2.0, 2.4, 1000)
    # Face-on (no rotation shift), small dispersion -> sharp lines at the centres.
    cfg = gk.GalaxySourceConfig(nx=8, ny=8, inclination_deg=0.0, velocity_dispersion_kms=20.0)
    cube = gk.build_galaxy_source_cube(cfg, wl, z, lines=["Halpha", "NII6583"])
    spec = cube.sum(axis=(0, 1))
    for ln in ("Halpha", "NII6583"):
        obs = observed_wavelength_um(REST_FRAME_LINES_UM[ln], z)
        peak_lambda = wl[np.argmax(np.where(np.abs(wl - obs) < 0.003, spec, 0))]
        assert abs(peak_lambda - obs) < 0.001


def test_dust_attenuates_blue_more_than_red():
    cfg = gk.GalaxySourceConfig(dust_attenuation_av_mag=1.0, dust_wavelength_law=True)
    blue = gk.dust_attenuation_factor(cfg, REST_FRAME_LINES_UM["Hbeta"])   # 0.486 um
    red = gk.dust_attenuation_factor(cfg, REST_FRAME_LINES_UM["Halpha"])   # 0.656 um
    assert blue < red < 1.0
    # Grey dust (no wavelength law) attenuates both equally.
    grey = gk.GalaxySourceConfig(dust_attenuation_av_mag=1.0, dust_wavelength_law=False)
    assert gk.dust_attenuation_factor(grey, 0.4) == gk.dust_attenuation_factor(grey, 0.7)


def test_out_of_band_or_unknown_line_raises():
    wl = np.linspace(2.0, 2.4, 200)
    with pytest.raises(ValueError, match="no requested line"):
        gk.build_galaxy_source_cube(gk.GalaxySourceConfig(nx=6, ny=6), wl, redshift=0.1)
    with pytest.raises(ValueError, match="unknown line"):
        gk.build_galaxy_source_cube(gk.GalaxySourceConfig(nx=6, ny=6), wl, 2.3, lines=["NotALine"])


def test_report_csv_provenance_valid(tmp_path):
    mod = _load_example("run_galaxy_source_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["truth_maps", "multiline_spectrum", "recovery", "metrics_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_line_ratios_are_integrated_flux_ratios_across_wide_lines():
    # A common velocity dispersion makes Hbeta/[O III] narrower in wavelength
    # than H-alpha; configured ratios must still hold for INTEGRATED flux.
    z = 0.0
    wl = np.linspace(0.47, 0.67, 40000)
    cfg = gk.GalaxySourceConfig(nx=3, ny=3, inclination_deg=0.0, velocity_dispersion_kms=60.0)
    lines = ["Halpha", "Hbeta", "OIII5007"]
    cube = gk.build_galaxy_source_cube(cfg, wl, z, lines=lines)
    spec = cube[1, 1]
    flux = {}
    for ln in lines:
        obs = observed_wavelength_um(REST_FRAME_LINES_UM[ln], z)
        flux[ln] = spec[np.abs(wl - obs) < 0.004].sum()
    assert flux["Hbeta"] / flux["Halpha"] == pytest.approx(gk.DEFAULT_LINE_RATIOS["Hbeta"], rel=1e-3)
    assert flux["OIII5007"] / flux["Halpha"] == pytest.approx(gk.DEFAULT_LINE_RATIOS["OIII5007"], rel=1e-3)
    # Unit-area profiles: the spaxel's H-alpha flux equals its Sersic weight.
    assert flux["Halpha"] == pytest.approx(gk.sersic_light_profile(cfg)[1, 1], rel=1e-6)


def test_truncated_line_keeps_off_band_loss():
    z = 0.0
    obs = REST_FRAME_LINES_UM["Halpha"]
    wl = np.linspace(obs, obs + 0.01, 2000)   # band starts AT the line centre
    cfg = gk.GalaxySourceConfig(nx=1, ny=1, inclination_deg=0.0, velocity_dispersion_kms=60.0)
    cube = gk.build_galaxy_source_cube(cfg, wl, z, lines=["Halpha"])
    # Only the red half (plus half a channel) lands in band; it is not renormalised.
    half_bin = 0.5 * (wl[1] - wl[0])
    sigma = obs * 60.0 / 299792.458
    from scipy.special import ndtr
    assert cube.sum() == pytest.approx(1.0 - ndtr(-half_bin / sigma), rel=1e-6)
