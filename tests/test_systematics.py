"""Tests for the systematics sensitivity module (direction 2)."""

import numpy as np
import pytest

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S
from ao_fed_ifu import systematics as sysm
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.integration import make_toy_config


def _base():
    return make_toy_config(nx=8, ny=8, resolving_power_R=1500.0)


def test_wavelength_offset_matches_doppler_relation():
    from ao_fed_ifu.integration import make_toy_config, run_pipeline

    # The Doppler relation is exact for a noiseless cube; a deep exposure keeps
    # the noise contribution to the windowed first-moment centroid small
    # (the deblending extraction carries more per-voxel noise than a direct
    # read, so the default toy depth leaves a ~7 % noise-driven scatter).
    base = make_toy_config(nx=8, ny=8, resolving_power_R=1500.0, exposure_s=3000.0)
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


def test_snr_loss_is_clamped_but_change_is_signed():
    # wavelength_offset shifts the grid only, leaving the SNR essentially
    # unchanged -- noise can nudge it up, which must NOT show as a negative loss.
    res = sysm.compute_systematics(_base(), n_levels=3,
                                   kinds=["wavelength_offset", "read_noise_e"])
    rows = {(r["kind"], r["metric"]): r for r in sysm.systematics_summary_rows(res)}
    for kind in ("wavelength_offset", "read_noise_e"):
        loss = rows[(kind, "snr_loss_max_frac")]
        change = rows[(kind, "snr_change_max_frac")]
        assert loss["value"] >= 0.0                       # never a negative "loss"
        assert "clamped" in loss["notes"]
        assert "signed" in change["notes"]
        # The clamped loss is the non-negative part of the signed change set.
        assert loss["value"] == pytest.approx(max(0.0, max(res[kind]["snr"])))
    # Read noise genuinely lowers SNR -> a real positive loss.
    assert rows[("read_noise_e", "snr_loss_max_frac")]["value"] > 0.0


def test_trace_offset_scan_uses_the_reference_estimator():
    """A zero-row trace offset must reproduce the reference exactly (no
    estimator switch), and a real offset must bias the line flux."""
    from ao_fed_ifu.integration import run_pipeline

    base = _base()
    res = sysm.compute_systematics(base, n_levels=2, kinds=["trace_row_offset"],
                                   levels={"trace_row_offset": (0, 1)})
    e = res["trace_row_offset"]
    assert e["velocity"][0] == pytest.approx(0.0, abs=1e-9)
    assert e["lineflux"][0] == pytest.approx(0.0, abs=1e-9)
    assert e["snr"][0] == pytest.approx(0.0, abs=1e-9)
    # a one-row trace error under the SAME estimator is a percent-level
    # line-flux bias (the old estimator switch reported ~50 %)
    assert 0.002 < abs(e["lineflux"][1]) < 0.2
    ref = run_pipeline(base)
    assert e["unmeasured_fraction"][0] == pytest.approx(
        float(np.mean(~np.isfinite(ref.variance_cube)))
    )


def test_wavelength_stretch_recovered_R_evaluated_on_perturbed_grid():
    base = _base()
    res = sysm.compute_systematics(base, n_levels=3, kinds=["wavelength_stretch"],
                                   levels={"wavelength_stretch": (0.0, 0.001, 0.002)})
    e = res["wavelength_stretch"]
    # no stretch -> no R bias; a stretch of the believed grid changes the
    # recovered R (it is no longer forced to zero by evaluating on the
    # unperturbed grid) and does so linearly in the stretch
    assert e["recovered_R_bias"][0] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(e["recovered_R_bias"][1]) and abs(e["recovered_R_bias"][1]) > 0.0
    assert e["recovered_R_bias"][2] == pytest.approx(2.0 * e["recovered_R_bias"][1], rel=0.1)
