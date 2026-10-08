"""Regression gates on the generated artifacts and core physical consistency.

These read ``AO_IFU_ARTIFACT_DIR`` (locally, ``figures/`` by default) and assert physically
meaningful thresholds, so a future change that silently breaks the science
(sky contamination, inflated SNR, a lost interpretability metric, a negative
"loss", a broken manifest) fails CI rather than passing unnoticed.  They are
deliberately TOLERANT of stochastic simulation scatter but strict enough to
catch regressions. CI generates a new artifact directory before running these
gates, and missing files or legacy missing metrics FAIL there. Locally, optional
missing artifacts skip unless ``AO_IFU_REQUIRE_ARTIFACTS=1`` is set; a few cheap
fresh-run gates check the pipeline itself rather than a stored file.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = Path(os.environ.get("AO_IFU_ARTIFACT_DIR", REPO / "figures"))


def _missing_artifact(message: str) -> None:
    strict = any(os.environ.get(key, "").lower() in {"1", "true", "yes"}
                 for key in ("CI", "AO_IFU_REQUIRE_ARTIFACTS"))
    if strict:
        pytest.fail(message)
    pytest.skip(message)


def _csv(name: str) -> pd.DataFrame:
    p = GEN / name
    if not p.exists():
        _missing_artifact(f"required artifact {name} not generated in {GEN}")
    return pd.read_csv(p)


def _json(name: str) -> dict:
    p = GEN / name
    if not p.exists():
        _missing_artifact(f"required artifact {name} not generated in {GEN}")
    return json.loads(p.read_text())


def _has(df: pd.DataFrame, metric: str) -> bool:
    return bool((df["metric"] == metric).any())


def _metric(df: pd.DataFrame, metric: str) -> float:
    row = df[df["metric"] == metric]
    assert not row.empty, f"metric {metric!r} missing from table"
    return float(row["value"].iloc[0])


@pytest.mark.parametrize("loader,name", [(_csv, "absent.csv"), (_json, "absent.json")])
@pytest.mark.parametrize("strict_key", ["CI", "AO_IFU_REQUIRE_ARTIFACTS"])
def test_missing_artifact_is_an_error_when_required(monkeypatch, tmp_path, loader, name, strict_key):
    monkeypatch.setitem(globals(), "GEN", tmp_path)
    monkeypatch.setenv(strict_key, "true")
    with pytest.raises(pytest.fail.Exception, match="required artifact"):
        loader(name)


def test_missing_artifact_remains_optional_locally(monkeypatch, tmp_path):
    monkeypatch.setitem(globals(), "GEN", tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("AO_IFU_REQUIRE_ARTIFACTS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="required artifact"):
        _csv("absent.csv")


# ---------------------------------------------------------------------------
# High-z science recovery gates.
# ---------------------------------------------------------------------------

def test_highz_line_flux_bias_vs_observed_truth_small():
    df = _csv("highz_science_metrics.csv")
    # Window-sum recovery: percent-ish bias vs the matched observed truth.
    assert abs(_metric(df, "line_flux_bias_vs_observed_truth_frac")) < 0.15
    # Parametric (Gaussian-fit) recovery: a slightly looser gate.
    if _has(df, "fit_line_flux_bias_vs_observed_truth_frac"):
        assert abs(_metric(df, "fit_line_flux_bias_vs_observed_truth_frac")) < 0.20
    else:
        _missing_artifact("Gaussian-fit bias metric not in this artifact")


def test_highz_velocity_bias_small():
    df = _csv("highz_science_metrics.csv")
    assert abs(_metric(df, "velocity_bias_vs_observed_truth_kms")) < 15.0


def test_highz_sky_subtraction_applied():
    d = _json("highz_science_summary.json")
    assert d["metrics"]["sky_subtraction_applied"] is True
    assert d["bias_reference_surface"] == "observed"


def test_highz_aperture_covariance_metrics_consistent():
    df = _csv("highz_science_metrics.csv")
    if not _has(df, "aperture_snr_diagonal"):
        _missing_artifact("covariance-aware aperture metrics not in this artifact")
    diag = _metric(df, "aperture_snr_diagonal")
    cov = _metric(df, "aperture_snr_covariance_aware")
    infl = _metric(df, "aperture_covariance_inflation_factor")
    spatial = _metric(df, "aperture_spatial_covariance_factor")
    assert diag > 0 and cov > 0
    # The spectral term is an INFLATION that never credits a noise reduction
    # (factor >= 1); the spatial term is the exact covariance of the joint
    # deblend, whose anti-correlated neighbours make an aperture sum LESS noisy
    # than the independent-spaxel budget (factor < 1) -- so the covariance-
    # aware SNR may exceed the diagonal one.
    assert infl >= 1.0
    assert 0.0 < spatial < 1.0
    # covariance-aware SNR = diagonal / sqrt(spatial x inflation) by construction.
    assert cov == pytest.approx(diag / np.sqrt(spatial * infl), rel=0.05)
    # For this forward model the reconstructed noise is ~white -> inflation ~1.
    assert infl < 2.5


def test_highz_gaussian_fit_converged():
    df = _csv("highz_science_metrics.csv")
    if not _has(df, "fit_success"):
        _missing_artifact("Gaussian-fit metrics not in this artifact")
    assert _metric(df, "fit_success") == 1.0
    # Fitted and window-sum aperture line flux agree to order unity.
    fit_e = _metric(df, "fit_line_flux_recovered_e")
    win_e = _metric(df, "window_line_flux_recovered_e")
    assert 0.5 < fit_e / win_e < 2.0


def test_highz_gaussian_fit_quality_flagged_honestly():
    """``fit_success`` alone is not enough: a reduced chi2 far from 1 means the
    formal errors are not trustworthy, so the artifact must carry an explicit
    quality flag consistent with chi2, and -- when quality is poor -- a
    conservative chi2-scaled SNR next to the optimistic formal one, plus an
    explicit poor-quality marker in the summary."""
    from ao_fed_ifu import diagnostics as diag

    df = _csv("highz_science_metrics.csv")
    if not _has(df, "fit_reduced_chi2"):
        _missing_artifact("Gaussian-fit metrics not in this artifact")
    chi2 = _metric(df, "fit_reduced_chi2")
    assert _has(df, "fit_quality_ok"), "fit_quality_ok flag missing from artifact"
    quality = _metric(df, "fit_quality_ok")
    lo, hi = diag.FIT_QUALITY_CHI2_RANGE
    assert quality == (1.0 if lo < chi2 < hi else 0.0)

    assert _has(df, "fit_snr_formal") and _has(df, "fit_snr_chi2_scaled")
    snr_formal = _metric(df, "fit_snr_formal")
    snr_scaled = _metric(df, "fit_snr_chi2_scaled")
    assert snr_scaled == pytest.approx(snr_formal / np.sqrt(max(1.0, chi2)), rel=1e-6)

    if quality == 0.0:
        # Poor fit quality must be marked in the summary, not silently passed.
        d = _json("highz_science_summary.json")
        q = d["gaussian_fit_quality"]
        assert q["quality_ok"] is False
        assert any("reduced_chi2" in lim for lim in d["limitations"])


# ---------------------------------------------------------------------------
# Reconstruction interpretability + systematics semantics gates.
# ---------------------------------------------------------------------------

def test_reconstruction_finite_snr_interpretability_present():
    df = _csv("reconstruction_metrics.csv")
    for m in ("nearest", "weighted", "nearest_bad_pixel_mask", "trace_offset"):
        assert _has(df, f"{m}.finite_snr_fraction"), m
        assert _has(df, f"{m}.median_line_snr_finite_only"), m
        frac = _metric(df, f"{m}.finite_snr_fraction")
        assert 0.0 <= frac <= 1.0
    d = _json("reconstruction_summary.json")
    assert "snr_interpretation" in d
    assert "not a physical non-detection" in d["snr_interpretation"]


def test_systematics_snr_loss_non_negative():
    df = _csv("systematics_metrics.csv")
    loss = df[df["metric"].str.endswith("snr_loss_max_frac")]
    assert not loss.empty
    # A "loss" metric must never be presented as negative (a tiny SNR gain).
    assert (loss["value"] >= 0.0).all(), loss[loss["value"] < 0.0].to_dict("records")
    # The signed change companion must exist so gains are still reported.
    assert not df[df["metric"].str.endswith("snr_change_max_frac")].empty


# ---------------------------------------------------------------------------
# Manifest integrity gate.
# ---------------------------------------------------------------------------

def test_manifest_no_duplicates_and_all_listed_artifacts_exist():
    df = _csv("artifact_manifest.csv")
    paths = df["artifact_path"].astype(str).tolist()
    assert len(paths) == len(set(paths)), "duplicate artifact_path rows in the manifest"
    missing = [p for p in paths if not (GEN / p).exists()]
    assert not missing, f"manifest lists missing artifacts: {missing}"


# ---------------------------------------------------------------------------
# Cheap fresh-run gates on the pipeline itself (not a stored artifact).
# ---------------------------------------------------------------------------

def test_weighted_variance_and_snr_internally_consistent():
    from ao_fed_ifu import diagnostics as diag
    from ao_fed_ifu.integration import make_toy_config, run_pipeline

    r = run_pipeline(make_toy_config(nx=8, ny=8, resolving_power_R=1500.0))
    w = 3.0 * r.spectrograph.delta_lambda_resel_um(r.line_center_um)
    manual = diag.snr_map(r.rebuilt_weighted, r.variance_cube_weighted,
                          r.wavelength_grid_um, r.line_center_um, w)
    # The reported SNR pairs weighted flux with the MATCHING weighted variance.
    assert np.allclose(manual, r.snr_map, equal_nan=True)
    fin = np.isfinite(r.variance_cube_weighted) & np.isfinite(r.variance_cube_nearest)
    # The windowed weighted extraction carries at least the single-pixel noise.
    assert np.median(r.variance_cube_weighted[fin]) >= np.median(r.variance_cube_nearest[fin])


def test_sky_subtraction_residual_small_unless_deliberately_mis_scaled():
    from ao_fed_ifu.integration import make_public_conditioned_config, run_pipeline

    # Sky-dominant regime (sky_scaling boosts the sky well above the noise) so
    # the sky-subtraction effect is unambiguous and the gate is robust.  The
    # joint deblending extraction carries ~2.5-3.5x the per-voxel variance of a
    # direct read, so the sky must be boosted further than the old per-trace
    # estimator needed for the median residual to sit below the sky removed.
    kw = dict(nx=8, ny=8, resolving_power_R=1500.0, seed=0, sky_scaling=20.0)
    perfect = run_pipeline(make_public_conditioned_config(**kw))
    mis = run_pipeline(make_public_conditioned_config(sky_subtraction_scale_error=0.5, **kw))
    assert perfect.metrics["sky_subtraction_applied"] is True

    ap = perfect.aperture_mask
    grid, lc = perfect.wavelength_grid_um, perfect.line_center_um
    w = 3.0 * perfect.spectrograph.delta_lambda_resel_um(lc)
    off = np.abs(grid - lc) > w
    truth = perfect.observed_truth_cube[ap][:, off]                  # source-only (noiseless)

    resid_perfect = float(np.median(np.abs(perfect.rebuilt_weighted_skysub[ap][:, off] - truth)))
    resid_mis = float(np.median(np.abs(mis.rebuilt_weighted_skysub[ap][:, off] - truth)))
    sky_removed = float(np.median(np.abs(
        perfect.rebuilt_weighted[ap][:, off] - perfect.rebuilt_weighted_skysub[ap][:, off])))

    # A real amount of sky was removed, and the perfectly-subtracted residual is
    # smaller than the sky it removed (no large sky left in the science product).
    assert sky_removed > 0.0
    assert resid_perfect < sky_removed
    # A deliberate 50% mis-scale leaves a clearly larger residual -> the gate
    # catches a broken (mis-scaled) sky subtraction.
    assert resid_mis > 1.3 * resid_perfect


def test_saved_summaries_were_generated_by_the_current_source():
    """A saved artifact is only evidence for the code that produced it.

    Every summary records ``software.source_sha256``; one that differs from the
    current package source predates the implementation (regenerate the figures).
    """
    from ao_fed_ifu.reporting import source_sha256

    summaries = sorted(GEN.glob("*_summary.json"))
    if not summaries:
        _missing_artifact(f"no *_summary.json artifacts in {GEN}")
    current = source_sha256()
    stale = [p.name for p in summaries
             if json.loads(p.read_text()).get("software", {}).get("source_sha256") != current]
    assert not stale, (f"{len(stale)} summaries in {GEN} were generated by different source code "
                       f"(regenerate them): {stale}")
