"""Tests for the sky / telluric module and example (direction 3)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ao_fed_ifu import sky_telluric as skt
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES, load_sky_spectrum

CACHE = Path(__file__).resolve().parents[1] / "data" / "public_cache"


def _cfg():
    return skt.SkyTelluricConfig(collecting_area_m2=50.0, exposure_s=600.0, throughput_scalar=0.3,
                                 spaxel_scale_mas=10.0, delta_lambda_um=0.001)


def test_empty_cache_returns_synthetic(tmp_path):
    wl = np.linspace(2.0, 2.4, 40)
    _, prov = load_sky_spectrum(wl, cache_dir=tmp_path)
    assert prov.source_class == "synthetic_proxy"


def test_public_cache_returns_direct_public():
    if not (CACHE / "skycalc.dat").exists():
        pytest.skip("public cache not present")
    wl = np.linspace(2.0, 2.4, 40)
    _, prov = load_sky_spectrum(wl)
    assert prov.source_class == "direct_public_data"


def test_nonzero_sky_scale_error_gives_nonzero_residual():
    wl = np.linspace(2.0, 2.4, 50)
    sky, _ = load_sky_spectrum(wl)
    sky_e = skt.build_sky_background_spectrum(wl, sky, _cfg())
    observed = 100.0 + sky_e  # flat source + sky
    perfect = skt.subtract_sky(observed, sky_e, scale_error=0.0)
    imperfect = skt.subtract_sky(observed, sky_e, scale_error=0.05)
    # Perfect subtraction leaves the flat source; imperfect leaves a sky residual.
    assert np.allclose(perfect, 100.0)
    assert np.max(np.abs(imperfect - perfect)) > 0


def test_residual_sky_fraction_cube_is_not_inflated():
    # Regression: the denominator must span the full cube, not a single spectrum.
    nlam = 30
    sky_e = np.linspace(1.0, 5.0, nlam)
    trans = np.full(nlam, 0.9)
    residual = -0.05 * sky_e[None, None, :] * np.ones((4, 3, nlam))
    frac = skt.sky_residual_metrics(residual, sky_e, trans)["residual_sky_fraction"]
    assert frac == pytest.approx(0.05, rel=1e-6)  # ~0.05, NOT 0.05 * nx * ny


def test_residual_sky_fraction_spectrum():
    nlam = 30
    sky_e = np.linspace(1.0, 5.0, nlam)
    trans = np.full(nlam, 0.9)
    residual = -0.10 * sky_e
    frac = skt.sky_residual_metrics(residual, sky_e, trans)["residual_sky_fraction"]
    assert frac == pytest.approx(0.10, rel=1e-6)


def test_telluric_correction_inverts_absorption():
    wl = np.linspace(2.0, 2.4, 50)
    sky, _ = load_sky_spectrum(wl)
    trans = np.clip(sky["transmission"], 1e-3, 1.0)
    source = np.full(wl.size, 50.0)
    absorbed = skt.apply_telluric_absorption(source, trans)
    corrected = skt.correct_telluric(absorbed, trans, transmission_error=0.0)
    assert np.allclose(corrected, source, rtol=1e-6)


def test_example_csv_provenance_valid(tmp_path):
    mod = _load_example("run_sky_telluric_residuals")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
