"""Tests for the engineering-like detector systematics (Priority 4)."""

import numpy as np
import pandas as pd

from ao_fed_ifu import detector_systematics as ds
from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.detector_model import DetectorConfig, expose


def test_more_reads_lowers_effective_read_noise():
    rn = 12.0
    fowler = [ds.effective_read_noise(rn, n, "fowler") for n in (1, 2, 4, 16)]
    assert fowler == sorted(fowler, reverse=True)   # monotonically decreasing
    assert ds.effective_read_noise(rn, 1) == rn
    # Up-the-ramp slope fit beats a single read for many samples.
    assert ds.effective_read_noise(rn, 32, "up_the_ramp") < rn


def test_nonlinearity_changes_high_more_than_low():
    fw = 1.0e5
    loss_lo = 1 - ds.apply_nonlinearity(np.array([1000.0]), fw, 0.05)[0] / 1000.0
    loss_hi = 1 - ds.apply_nonlinearity(np.array([9.0e4]), fw, 0.05)[0] / 9.0e4
    assert loss_hi > 10 * loss_lo


def test_nonlinearity_correction_inverts():
    fw, coeff = 1.0e5, 0.05
    s = np.linspace(0, 0.9 * fw, 50)
    recovered = ds.correct_nonlinearity(ds.apply_nonlinearity(s, fw, coeff), fw, coeff)
    assert np.allclose(recovered, s, rtol=1e-6, atol=1e-3)


def test_reference_pixel_removal_lowers_row_common_mode():
    rng = np.random.default_rng(0)
    frame = np.zeros((40, 40)); frame[:, 3:37] = 400.0          # interior + reference edges
    noisy = ds.apply_correlated_read_noise(frame, 20.0, rng, "row")
    noisy = noisy + rng.normal(0, 5.0, size=noisy.shape)
    before = ds.row_common_mode_metric(noisy[:, 3:37])
    after = ds.row_common_mode_metric(ds.subtract_reference_pixels(noisy)[:, 3:37])
    assert after < before


def test_ipc_conserves_flux_and_shares_to_neighbours():
    pt = np.zeros((11, 11)); pt[5, 5] = 100.0
    out = ds.apply_ipc(pt, 0.02)
    assert np.isclose(out.sum(), 100.0)
    assert out[5, 5] < 100.0 and out[5, 4] > 0.0


def test_expose_n_reads_backward_compatible_and_caps_full_well():
    low = np.full((20, 20), 50.0)  # read-noise-dominated, so the floor reduction shows
    cfg1 = DetectorConfig(read_noise_e=10.0, bad_pixel_fraction=0.0, flat_field_sigma=0.0)
    base = expose(low, cfg1, 100.0, rng=0)
    same = expose(low, DetectorConfig(read_noise_e=10.0, n_reads=1, bad_pixel_fraction=0.0, flat_field_sigma=0.0), 100.0, rng=0)
    assert np.array_equal(base.image_e, same.image_e)          # n_reads=1 unchanged
    more = expose(low, DetectorConfig(read_noise_e=10.0, n_reads=16, bad_pixel_fraction=0.0, flat_field_sigma=0.0), 100.0, rng=0)
    assert more.variance_e2.min() < base.variance_e2.min()      # more reads -> lower read-noise floor
    # Full-well clipping still caps the collected charge, independently of the
    # sampling proxy; only (reduced) read noise is added on top.
    capped_cfg = DetectorConfig(full_well_e=8.0e4, n_reads=16, bad_pixel_fraction=0.0, flat_field_sigma=0.0)
    capped = expose(np.full((10, 10), 5.0e5), capped_cfg, 1.0, rng=0)
    assert capped.saturation_mask.all()
    assert np.all(np.abs(capped.image_e - 8.0e4) < 6 * capped_cfg.read_noise_e)


def test_variance_finite_except_bad_pixels():
    out = expose(np.full((30, 30), 300.0), DetectorConfig(bad_pixel_fraction=0.1, n_reads=4), 50.0, rng=1)
    finite = np.isfinite(out.variance_e2)
    assert np.all(finite[~out.bad_pixel_mask])                 # finite where not masked
    assert np.all(np.isinf(out.variance_e2[out.bad_pixel_mask]))


def test_report_csv_provenance_valid(tmp_path):
    mod = _load_example("run_detector_systematics_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["noise_components", "sampling_curve", "metrics_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))
