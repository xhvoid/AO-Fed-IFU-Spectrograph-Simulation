"""Tests for the wavelength-dependent AO PSF proxy (Priority 2)."""

import numpy as np
import pandas as pd

from ao_fed_ifu.ao_psf_physics import (
    AOPSFConfig,
    ao_psf_at_wavelength,
    ao_psf_wavelength_sweep,
    strehl_from_wavefront_error,
)
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def test_strehl_rises_with_wavelength():
    s_blue = strehl_from_wavefront_error(150.0, 1.0)
    s_red = strehl_from_wavefront_error(150.0, 2.4)
    assert 0.0 <= s_blue <= s_red <= 1.0
    assert s_red > s_blue


def test_diffraction_core_fwhm_increases_with_wavelength():
    cfg = AOPSFConfig(telescope_diameter_m=8.0)
    rows = ao_psf_wavelength_sweep([1.0, 1.6, 2.2], cfg)
    diff = [r["diffraction_fwhm_mas"] for r in rows]
    assert diff[0] < diff[1] < diff[2]


def test_higher_strehl_increases_encircled_energy_at_fixed_wavelength():
    common = dict(telescope_diameter_m=8.0, wavefront_error_nm=None, seeing_fwhm_mas=600.0,
                  spaxel_scale_mas=20.0, fine_scale_mas=3.0, n_pix=121)
    lo = ao_psf_at_wavelength(2.2, AOPSFConfig(fixed_strehl=0.1, **common))
    hi = ao_psf_at_wavelength(2.2, AOPSFConfig(fixed_strehl=0.7, **common))
    assert hi["ee1_spaxel"] > lo["ee1_spaxel"]
    assert hi["ee3x3_spaxel"] > lo["ee3x3_spaxel"]


def test_smaller_spaxel_reduces_undersampling():
    # A sharp ELT-like core: coarse spaxels undersample, fine spaxels do not.
    coarse = ao_psf_at_wavelength(1.2, AOPSFConfig(telescope_diameter_m=39.0, wavefront_error_nm=None,
                                                   fixed_strehl=0.7, spaxel_scale_mas=20.0, fine_scale_mas=1.0, n_pix=161))
    fine = ao_psf_at_wavelength(1.2, AOPSFConfig(telescope_diameter_m=39.0, wavefront_error_nm=None,
                                                 fixed_strehl=0.7, spaxel_scale_mas=2.0, fine_scale_mas=1.0, n_pix=161))
    assert coarse["undersampled"] and not fine["undersampled"]


def test_metrics_finite_and_in_range():
    rows = ao_psf_wavelength_sweep(np.linspace(1.0, 2.4, 6), AOPSFConfig())
    for r in rows:
        assert np.isfinite(r["psf_fwhm_mas"]) and r["psf_fwhm_mas"] > 0
        assert 0.0 <= r["ee1_spaxel"] <= 1.0
        assert 0.0 <= r["ee_field"] <= 1.0


def test_sweep_is_deterministic():
    cfg = AOPSFConfig()
    a = ao_psf_wavelength_sweep([1.5, 2.0], cfg)
    b = ao_psf_wavelength_sweep([1.5, 2.0], cfg)
    assert a == b


def test_report_csv_has_per_row_source_class(tmp_path):
    mod = _load_example("run_ao_psf_wavelength_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["wavelength_grid", "ee_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["ee_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
    assert {"wavelength_um", "strehl", "ee1_spaxel", "ee_field", "undersampled"}.issubset(df.columns)
