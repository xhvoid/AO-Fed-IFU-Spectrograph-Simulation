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
