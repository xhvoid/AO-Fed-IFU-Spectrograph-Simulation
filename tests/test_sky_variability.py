"""Tests for sky/telluric time variability and residual realism (Priority 6)."""

import numpy as np
import pandas as pd

from ao_fed_ifu import sky_telluric as skt
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def _sky():
    # Deterministic synthetic sky: flat continuum + a few OH-line spikes.
    rad = np.full(200, 100.0)
    for i in (40, 90, 140):
        rad[i] += 1500.0
    return rad


def test_higher_oh_scale_increases_residual_near_lines():
    rad = _sky()
    cont, lines = skt.split_sky_continuum_lines(rad)
    line_px = lines > 3 * np.median(lines)
    diff = skt.vary_sky_radiance(rad, oh_line_scale=2.0) - skt.vary_sky_radiance(rad, oh_line_scale=1.0)
    assert np.mean(diff[line_px]) > 0
    assert np.mean(diff[line_px]) > np.mean(diff[~line_px]) + 1.0


def test_continuum_scale_scales_continuum():
    rad = _sky()
    cont, _ = skt.split_sky_continuum_lines(rad)
    varied = skt.vary_sky_radiance(rad, continuum_scale=2.0, oh_line_scale=1.0)
    cont_px = cont > 0
    # On continuum pixels the varied radiance is ~2x the continuum.
    assert np.allclose(varied[cont_px][:5], 2.0 * cont[cont_px][:5], rtol=0.05)


def test_telluric_mismatch_worse_in_low_transmission():
    trans = np.full(200, 0.95); trans[80:90] = 0.3       # a deep telluric trough
    src = np.full(200, 100.0)
    actual = skt.telluric_with_airmass(trans, airmass_ratio=1.4)
    resid = (src * actual) / trans - src
    assert np.mean(np.abs(resid[80:90])) > 5 * np.mean(np.abs(resid[trans > 0.9]))


def test_perfect_match_returns_input_transmission():
    trans = np.linspace(0.2, 1.0, 100)
    assert np.allclose(skt.telluric_with_airmass(trans, 1.0, 1.0), np.clip(trans, 1e-3, 1.0))


def test_mask_fraction_increases_when_threshold_lowered():
    rad = _sky()
    assert skt.masked_fraction(skt.sky_line_mask(rad, 2.0)) >= skt.masked_fraction(skt.sky_line_mask(rad, 5.0))
    # The bright lines are flagged.
    assert skt.sky_line_mask(rad, 3.0)[40]


def test_science_window_masked_fraction_bounds():
    wl = np.linspace(2.0, 2.4, 200)
    mask = np.zeros(200, dtype=bool); mask[100] = True
    frac = skt.science_window_masked_fraction(wl, mask, wl[100], 0.02)
    assert 0.0 <= frac <= 1.0 and frac > 0.0


def test_report_csv_provenance_valid(tmp_path):
    mod = _load_example("run_sky_variability_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["variability", "telluric_mismatch", "sky_mask", "metrics_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
