"""Raw detector semantics and ideal-known calibration invariants."""

import numpy as np
import pytest

from ao_fed_ifu import detector_model as detector


def _fixed_response(monkeypatch, response):
    def apply(frame, cfg, rng=None):
        flat = np.broadcast_to(response, frame.shape).copy()
        return frame * flat, flat

    monkeypatch.setattr(detector, "apply_flat_field_error", apply)


def _expectation_only(monkeypatch):
    monkeypatch.setattr(
        detector, "apply_poisson_noise", lambda mean, rng=None: mean.copy()
    )


def test_calibration_recovers_signal_but_retains_dark_shot_variance(monkeypatch):
    signal = np.array([[2., 10., 30.], [50., 80., 150.]])
    response = np.array([[.5, 1., 1.5], [2., .8, 1.2]])
    cfg = detector.DetectorConfig(
        read_noise_e=0, dark_current_e_s_pix=.3, bias_e=17,
        flat_field_sigma=.2, bad_pixel_fraction=0,
    )
    exposure = 20.
    dark = cfg.dark_current_e_s_pix * exposure
    _fixed_response(monkeypatch, response)
    _expectation_only(monkeypatch)

    raw = detector.expose(signal, cfg, exposure, rng=123)
    raw_before = raw.image_e.copy()
    variance_before = raw.variance_e2.copy()
    np.testing.assert_allclose(raw.image_e, response * (signal + dark) + cfg.bias_e)
    np.testing.assert_allclose(raw.variance_e2, response * (signal + dark))
    np.testing.assert_array_equal(raw.flat_response, response)
    assert not raw.saturation_mask.any()

    calibrated = detector.calibrate_readout(raw, cfg, exposure)
    np.testing.assert_allclose(calibrated.image_e, signal)
    np.testing.assert_allclose(calibrated.variance_e2, (signal + dark) / response)
    assert not calibrated.invalid_pixel_mask.any()
    np.testing.assert_array_equal(raw.image_e, raw_before)
    np.testing.assert_array_equal(raw.variance_e2, variance_before)

    # Explicit residual-flat mode still removes the response-scaled dark; it
    # intentionally retains the sensitivity error on astronomical electrons.
    residual = detector.calibrate_readout(raw, cfg, exposure, correct_flat=False)
    np.testing.assert_allclose(residual.image_e, response * signal)
    np.testing.assert_allclose(residual.variance_e2, raw.variance_e2)


def test_response_scaled_poisson_variance_includes_effective_read_noise(monkeypatch):
    signal = np.array([[20., 70.]])
    response = np.array([[.4, 1.7]])
    _fixed_response(monkeypatch, response)
    cfg = detector.DetectorConfig(
        read_noise_e=12, n_reads=9, read_mode="fowler",
        dark_current_e_s_pix=.2, bias_e=41,
        flat_field_sigma=.2, bad_pixel_fraction=0,
    )
    raw = detector.expose(signal, cfg, exposure_s=10, rng=10)
    expected_raw = response * (signal + 2.) + (12. / np.sqrt(9))**2
    np.testing.assert_allclose(raw.variance_e2, expected_raw)
    calibrated = detector.calibrate_readout(raw, cfg, exposure_s=10)
    np.testing.assert_allclose(calibrated.variance_e2, expected_raw / response**2)


@pytest.mark.parametrize("response", [.6, 1.4])
def test_calibrated_mean_and_variance_match_bounded_monte_carlo(monkeypatch, response):
    # Independent detector pixels provide 120,000 draws without running a
    # pipeline or growing a covariance matrix. Keep the realized flat fixed:
    # these are conditional Poisson/read variances, not flat uncertainty.
    signal_e = 40.
    exposure = 20.
    signal = np.full((300, 400), signal_e)
    cfg = detector.DetectorConfig(
        read_noise_e=5, dark_current_e_s_pix=.3, bias_e=31,
        flat_field_sigma=.1, bad_pixel_fraction=0,
    )
    _fixed_response(monkeypatch, response)
    raw = detector.expose(signal, cfg, exposure, rng=891)
    calibrated = detector.calibrate_readout(raw, cfg, exposure)
    expected_variance = (
        response * (signal_e + cfg.dark_current_e_s_pix * exposure)
        + cfg.read_noise_e**2
    ) / response**2
    assert calibrated.image_e.mean() == pytest.approx(signal_e, abs=.18)
    assert calibrated.image_e.var(ddof=1) == pytest.approx(expected_variance, rel=.02)
    np.testing.assert_allclose(calibrated.variance_e2, expected_variance)


def test_expose_retains_clipping_mask_and_masks_zero_response(monkeypatch):
    cfg = detector.DetectorConfig(
        read_noise_e=0, dark_current_e_s_pix=0, bias_e=5,
        full_well_e=50, flat_field_sigma=.2, bad_pixel_fraction=0,
    )
    _fixed_response(monkeypatch, np.array([[1., 1., 0.]]))
    _expectation_only(monkeypatch)
    raw = detector.expose(np.array([[10., 100., 20.]]), cfg, 1, rng=0)
    # The CHARGE clips at the full well (50), then the bias (5) is added.
    np.testing.assert_array_equal(raw.image_e, [[15., 55., 5.]])
    np.testing.assert_array_equal(raw.saturation_mask, [[False, True, False]])
    np.testing.assert_array_equal(raw.bad_pixel_mask, [[False, False, False]])
    np.testing.assert_array_equal(np.isinf(raw.variance_e2), [[False, True, True]])
    calibrated = detector.calibrate_readout(raw, cfg, 1)
    np.testing.assert_array_equal(calibrated.image_e, [[10., 0., 0.]])
    np.testing.assert_array_equal(calibrated.invalid_pixel_mask, [[False, True, True]])
    np.testing.assert_array_equal(np.isinf(calibrated.variance_e2), [[False, True, True]])


@pytest.mark.parametrize("correct_flat", [True, False])
def test_calibration_masks_all_unusable_samples(correct_flat):
    raw = detector.DetectorReadout(
        image_e=np.array([[20., 20., 20., 20.], [np.nan, 20., 20., 20.]]),
        variance_e2=np.array([[8., 8., 8., 8.], [8., -1., 8., np.inf]]),
        bad_pixel_mask=np.array([[False, True, False, False], [False] * 4]),
        flat_response=np.array([[1., 1., 1., 0.], [1., 1., np.inf, 1.]]),
        saturation_mask=np.array([[False, False, True, False], [False] * 4]),
    )
    cfg = detector.DetectorConfig(dark_current_e_s_pix=1, bias_e=3)
    calibrated = detector.calibrate_readout(raw, cfg, 2, correct_flat=correct_flat)
    expected_invalid = np.ones((2, 4), dtype=bool)
    expected_invalid[0, 0] = False
    np.testing.assert_array_equal(calibrated.invalid_pixel_mask, expected_invalid)
    assert calibrated.image_e[0, 0] == 15.
    assert calibrated.variance_e2[0, 0] == 8.
    assert np.all(calibrated.image_e[expected_invalid] == 0)
    assert np.all(np.isinf(calibrated.variance_e2[expected_invalid]))


def test_legacy_readout_requires_a_known_response():
    # Existing three-positional-field construction is still supported. Only a
    # configuration without response variation supplies a known missing map.
    # Without a saturation mask, saturation is inferred from the bias-
    # subtracted charge (23 - 3 = 20 reaches the 20 e- well; 15 - 3 does not).
    raw = detector.DetectorReadout(
        np.array([[15., 23.]]), np.full((1, 2), 7.), np.zeros((1, 2), dtype=bool)
    )
    cfg = detector.DetectorConfig(
        flat_field_sigma=0, dark_current_e_s_pix=1, bias_e=3, full_well_e=20, read_noise_e=0
    )
    calibrated = detector.calibrate_readout(raw, cfg, 2)
    np.testing.assert_array_equal(calibrated.image_e, [[10., 0.]])
    np.testing.assert_array_equal(calibrated.invalid_pixel_mask, [[False, True]])
    cfg.flat_field_sigma = .01
    with pytest.raises(ValueError, match="realized flat_response"):
        detector.calibrate_readout(raw, cfg, 2)


def test_nonrepresentable_response_correction_is_masked():
    raw = detector.DetectorReadout(
        np.array([[1.]]), np.array([[1.]]), np.array([[False]]),
        flat_response=np.array([[1e-200]]), saturation_mask=np.array([[False]]),
    )
    calibrated = detector.calibrate_readout(raw, detector.DetectorConfig(), 1)
    assert calibrated.invalid_pixel_mask[0, 0]
    assert calibrated.image_e[0, 0] == 0
    assert np.isinf(calibrated.variance_e2[0, 0])


@pytest.mark.parametrize("field", ["variance_e2", "bad_pixel_mask", "flat_response", "saturation_mask"])
def test_calibration_rejects_mismatched_array_shapes(field):
    values = dict(
        image_e=np.ones((2, 3)), variance_e2=np.ones((2, 3)),
        bad_pixel_mask=np.zeros((2, 3), dtype=bool),
        flat_response=np.ones((2, 3)), saturation_mask=np.zeros((2, 3), dtype=bool),
    )
    values[field] = np.ones((1, 3))
    with pytest.raises(ValueError, match="match"):
        detector.calibrate_readout(detector.DetectorReadout(**values), detector.DetectorConfig(), 1)


def test_bias_does_not_saturate_the_charge_well():
    # Identical photon draws: a large electronic bias must not change which
    # pixels saturate (it used to clip 51% of a 1000 e- scene at bias 99,000).
    scene = np.full((40, 40), 1000.0)
    common = dict(read_noise_e=5.0, dark_current_e_s_pix=0.0, full_well_e=1.0e5,
                  bad_pixel_fraction=0.0, flat_field_sigma=0.0)
    zero = detector.expose(scene, detector.DetectorConfig(bias_e=0.0, **common), 1, rng=3)
    high = detector.expose(scene, detector.DetectorConfig(bias_e=99_000.0, **common), 1, rng=3)
    assert not zero.saturation_mask.any() and not high.saturation_mask.any()
    np.testing.assert_allclose(high.image_e - 99_000.0, zero.image_e)
    calibrated = detector.calibrate_readout(high, detector.DetectorConfig(bias_e=99_000.0, **common), 1)
    assert not calibrated.invalid_pixel_mask.any()


def test_legacy_readout_saturation_follows_charge_semantics():
    # A mask-less readout from the current expose: a large bias must not be
    # mistaken for saturation, while clipped pixels are still caught.
    scene = np.full((60, 60), 1000.0)
    scene[:5] = 5.0e5
    cfg = detector.DetectorConfig(read_noise_e=5.0, dark_current_e_s_pix=0.0, full_well_e=1.0e5,
                                  bias_e=99_000.0, bad_pixel_fraction=0.0, flat_field_sigma=0.0)
    full = detector.expose(scene, cfg, 1, rng=4)
    legacy = detector.DetectorReadout(full.image_e, np.where(full.saturation_mask, 7.0, full.variance_e2),
                                      full.bad_pixel_mask)
    calibrated = detector.calibrate_readout(legacy, cfg, 1)
    np.testing.assert_array_equal(calibrated.invalid_pixel_mask, full.saturation_mask)


def test_legacy_saturation_survives_bias_rounding_without_read_noise():
    # (full_well + bias) - bias can round just below full_well; with zero read
    # noise there is no margin, so a floating-point floor must catch it.
    cfg = detector.DetectorConfig(read_noise_e=0.0, dark_current_e_s_pix=0.0, full_well_e=1.0e5,
                                  bias_e=0.3, bad_pixel_fraction=0.0, flat_field_sigma=0.0)
    raw = np.array([[1.0e5 + 0.3, 500.3]])
    legacy = detector.DetectorReadout(raw, np.ones_like(raw), np.zeros_like(raw, dtype=bool))
    np.testing.assert_array_equal(detector.calibrate_readout(legacy, cfg, 1).invalid_pixel_mask,
                                  [[True, False]])
