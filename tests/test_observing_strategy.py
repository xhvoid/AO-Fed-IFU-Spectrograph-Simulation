"""Tests for the multi-exposure observing-strategy layer (Priority 3)."""

import numpy as np
import pandas as pd

from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.integration import make_toy_config
from ao_fed_ifu.observing_strategy import (
    ExposurePlan, _sigma_clip_coadd, run_observing_strategy, snr_vs_nexp,
)


def test_coadd_snr_improves_over_single_exposure():
    base = make_toy_config(nx=6, ny=6, resolving_power_R=1500.0)
    res = run_observing_strategy(base, ExposurePlan(n_exposures=4, seed=0))
    lc = res.reference.line_center_um
    win = 3.0 * res.reference.spectrograph.delta_lambda_resel_um(lc)
    ns, snrs, slope = snr_vs_nexp(res, lc, win)
    assert snrs[-1] > snrs[0]            # coadd improves SNR
    assert 0.4 <= slope <= 0.85          # roughly sqrt(N)


def test_equal_weight_coadd_of_identical_cubes_conserves_flux():
    cubes = np.full((5, 4, 4, 8), 3.0)
    coadd, _, rejected = _sigma_clip_coadd(cubes, np.ones_like(cubes), sigma=5.0)
    assert np.allclose(coadd, 3.0)
    assert rejected == 0.0


def test_variance_decreases_as_one_over_n():
    rng = np.random.default_rng(0)
    for n in (2, 4, 8):
        cubes = rng.normal(10.0, 1.0, size=(n, 3, 3, 6))
        _, coadd_var, _ = _sigma_clip_coadd(cubes, np.ones_like(cubes), sigma=6.0)
        assert np.isclose(np.mean(coadd_var), 1.0 / n, rtol=0.05)


def test_dither_align_recovers_source_centroid():
    base = make_toy_config(nx=11, ny=11, resolving_power_R=1500.0, source_kind="point", apply_psf=False)
    plan = ExposurePlan(n_exposures=4, dither_spaxels=[(0, 0), (2, 0), (0, 2), (2, 2)], seed=1)
    res = run_observing_strategy(base, plan)
    image = res.coadd_cube.sum(axis=2)
    iy, ix = np.unravel_index(int(np.nanargmax(image)), image.shape)
    assert (iy, ix) == (5, 5)  # back at the field centre after alignment


def test_cosmic_ray_rejection_matches_rate_and_caps_snr():
    base = make_toy_config(nx=6, ny=6, resolving_power_R=1500.0)
    res = run_observing_strategy(base, ExposurePlan(n_exposures=6, cosmic_ray_rate=0.01, seed=2))
    assert res.rejected_fraction < 0.03            # ~ the 1% CR rate, not wildly over-rejecting
    assert res.finite_fraction == 1.0
    lc = res.reference.line_center_um
    win = 3.0 * res.reference.spectrograph.delta_lambda_resel_um(lc)
    _, snrs, _ = snr_vs_nexp(res, lc, win)
    assert np.all(np.isfinite(snrs)) and snrs[-1] < 1e4   # CRs did not blow up the SNR


def test_report_csv_provenance_valid(tmp_path):
    mod = _load_example("run_observing_strategy_report")
    paths = mod.generate(tmp_path, quick=True)
    for key in ["dither", "snr_gain", "single_vs_coadd", "metrics_csv", "summary_json"]:
        assert paths[key].exists()
    df = pd.read_csv(paths["metrics_csv"])
    assert "source_class" in df.columns
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_missing_placeholder_cannot_reject_valid_pair_member():
    # Two exposures: one unmeasured (zero placeholder, infinite variance), one
    # valid.  The valid sample must survive, not the placeholder.
    cubes = np.array([0.0, 100.0]).reshape(2, 1, 1, 1)
    variances = np.array([np.inf, 1.0]).reshape(2, 1, 1, 1)
    coadd, coadd_var, rejected = _sigma_clip_coadd(cubes, variances, sigma=5.0)
    assert coadd[0, 0, 0] == 100.0
    assert coadd_var[0, 0, 0] == 1.0
    assert rejected == 0.0


def test_missing_majority_does_not_drag_median_onto_placeholder():
    # Two valid samples, so the valid-minimum fallback cannot hide a median
    # dragged onto the placeholders (which would clip BOTH valid samples).
    cubes = np.array([0.0, 0.0, 100.0, 102.0]).reshape(4, 1, 1, 1)
    variances = np.array([np.inf, np.inf, 1.0, 1.0]).reshape(4, 1, 1, 1)
    coadd, coadd_var, rejected = _sigma_clip_coadd(cubes, variances, sigma=5.0)
    assert coadd[0, 0, 0] == 101.0 and coadd_var[0, 0, 0] == 0.5
    assert rejected == 0.0


def test_nan_exposure_does_not_contaminate_valid_samples():
    cubes = np.array([np.nan, 10.0, 12.0]).reshape(3, 1, 1, 1)
    variances = np.array([1.0, 4.0, 4.0]).reshape(3, 1, 1, 1)
    coadd, coadd_var, _ = _sigma_clip_coadd(cubes, variances, sigma=5.0)
    assert coadd[0, 0, 0] == 11.0 and coadd_var[0, 0, 0] == 2.0


def test_fully_unmeasured_voxel_stays_unmeasured_and_cr_fallback_uses_valid_min():
    # voxel 0: never measured; voxel 1: a CR-contaminated valid pair plus a
    # placeholder -- the fallback must pick the valid minimum, not the placeholder.
    cubes = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 1.0e6]]).reshape(3, 1, 2, 1)
    variances = np.array([[np.inf, np.inf], [np.inf, 1.0], [np.inf, 1.0]]).reshape(3, 1, 2, 1)
    coadd, coadd_var, _ = _sigma_clip_coadd(cubes, variances, sigma=5.0)
    assert coadd[0, 0, 0] == 0.0 and np.isinf(coadd_var[0, 0, 0])
    assert coadd[0, 1, 0] == 100.0 and coadd_var[0, 1, 0] == 1.0


def test_sky_frames_use_matched_pipeline_subtraction():
    # Sky frames must subtract the sky in the SAME reconstructed coordinates as
    # the pipeline's matched subtraction, whatever the base subtract_sky flag.
    from ao_fed_ifu.integration import make_public_conditioned_config

    kw = dict(nx=4, ny=4, resolving_power_R=200.0)
    plan = ExposurePlan(n_exposures=1, sky_frames=True, seed=0)
    fallback = run_observing_strategy(make_public_conditioned_config(subtract_sky=False, **kw), plan)
    matched = run_observing_strategy(make_public_conditioned_config(subtract_sky=True, **kw), plan)
    assert fallback.reference.metrics["sky_background_electrons"] > 0
    assert np.allclose(fallback.exposure_cubes, matched.exposure_cubes)
    # ... and that product IS the pipeline's sky-subtracted cube (one undithered
    # exposure), not the sky-added reconstruction.
    reference = fallback.reference
    assert np.allclose(fallback.exposure_cubes[0], reference.rebuilt_nearest_skysub)
    assert not np.allclose(fallback.exposure_cubes[0], reference.rebuilt_nearest)
