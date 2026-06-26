"""Tests for the systematics sensitivity module (direction 2)."""

import numpy as np

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S
from ao_fed_ifu import systematics as sysm
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.integration import make_toy_config


def _base():
    return make_toy_config(nx=8, ny=8, resolving_power_R=1500.0)


def test_wavelength_offset_matches_doppler_relation():
    from ao_fed_ifu.integration import run_pipeline

    base = _base()
    res = sysm.compute_systematics(base, n_levels=2, kinds=["wavelength_offset"])
    e = res["wavelength_offset"]
    # A wavelength offset of N pixels shifts velocity by c * (N * dispersion) / lambda.
    ref = run_pipeline(base)
    disp = ref.spectrograph.dispersion_um_per_pix
    predicted_1pix = SPEED_OF_LIGHT_KM_S * (e["x"][0] * disp) / ref.line_center_um
    assert e["velocity"][0] > 0
    assert abs(e["velocity"][0] - predicted_1pix) < 0.05 * predicted_1pix
    # Linear: the 2-pixel offset bias is ~2x the 1-pixel one.
    assert abs(e["velocity"][1] - 2 * e["velocity"][0]) < 0.2 * abs(e["velocity"][0])


def test_higher_read_noise_lowers_snr():
    res = sysm.compute_systematics(_base(), n_levels=3, kinds=["read_noise_e"])
    snr_loss = res["read_noise_e"]["snr"]
    # SNR loss should grow with read noise.
    assert snr_loss[-1] > snr_loss[0]
    assert snr_loss[-1] > 0


def test_higher_crosstalk_increases_leakage():
    res = sysm.compute_systematics(_base(), n_levels=3, kinds=["cross_talk_sigma"])
    leak = res["cross_talk_sigma"]["crosstalk_leakage"]
    assert leak[-1] > leak[0]


def test_summary_rows_have_valid_provenance():
    res = sysm.compute_systematics(_base(), n_levels=2, kinds=["wavelength_offset", "read_noise_e"])
    rows = sysm.systematics_summary_rows(res)
    assert rows
    for r in rows:
        assert r["source_class"] in VALID_SOURCE_CLASSES
