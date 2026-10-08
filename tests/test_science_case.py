"""Tests for the high-z emission-line galaxy science case (direction 1)."""

import json

import pandas as pd

from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.integration import make_highz_galaxy_config
from ao_fed_ifu.source_cube import REST_FRAME_LINES_UM, observed_wavelength_um


def test_aperture_gaussian_fits_use_covariance_and_common_complete_channels():
    from types import SimpleNamespace
    import numpy as np
    import pytest
    from ao_fed_ifu import diagnostics as diag

    grid = np.linspace(2.16, 2.24, 161)
    signal = 20 + 100 * np.exp(-0.5 * ((grid - 2.201) / 0.003)**2)
    variance = np.full((1, 2, grid.size), 10.0)
    variance[0, 1, 85] = np.inf
    aperture = np.ones((1, 2), dtype=bool)
    # A two-trace covariance [[10,-8],[-8,10]] has sum variance 4, NOT 20.
    model = SimpleNamespace(aperture_variance=lambda ap: np.full(grid.size, 4.0))
    result = SimpleNamespace(
        aperture_mask=aperture, extraction_model=model, variance_cube_weighted=variance,
        wavelength_grid_um=grid, line_center_um=2.2, extracted_spectrum=signal.copy(),
        observed_truth_cube=np.broadcast_to(signal / 2, variance.shape).copy(),
    )
    result.extracted_spectrum[85] = 1e6
    result.observed_truth_cube[:, :, 85] = 1e6
    recovered, truth = _load_example("run_science_case_highz_galaxy")._fit_aperture_lines(result, 0.03)
    expected_variance = np.full(grid.size, 4.0)
    expected_variance[85] = np.inf
    expected = diag.fit_gaussian_line(signal, grid, 2.2, window_um=0.03, variance_1d=expected_variance)
    for fit in (recovered, truth):
        assert fit["success"]
        assert fit["line_flux"] == pytest.approx(expected["line_flux"], rel=1e-9)
        assert fit["line_flux_err"] == pytest.approx(expected["line_flux_err"], rel=1e-9)


def test_observed_wavelength_lies_in_band():
    for line in ("Halpha", "OIII", "OII"):
        # Pick a redshift that puts the line near 2.2 um, then check the band.
        z = 2.2 / REST_FRAME_LINES_UM[line] - 1.0
        cfg = make_highz_galaxy_config(line=line, redshift=z)
        obs = observed_wavelength_um(REST_FRAME_LINES_UM[line], z)
        assert cfg.lambda_min_um <= obs <= cfg.lambda_max_um


def test_out_of_band_redshift_raises():
    import pytest

    # H-alpha at z=0.5 -> ~0.98 um, far below the supported 2.0-2.4 um band.
    with pytest.raises(ValueError, match="outside the supported"):
        make_highz_galaxy_config(line="Halpha", redshift=0.5)
    # [O II] at z=10 -> ~4.1 um, far above the band.
    with pytest.raises(ValueError, match="outside the supported"):
        make_highz_galaxy_config(line="OII", redshift=10.0)


def test_generate_runs_with_tiny_config(tmp_path):
    mod = _load_example("run_science_case_highz_galaxy")
    paths = mod.generate(tmp_path, line="Halpha", redshift=2.30, quick=True)
    for key in ["truth", "detector", "recovered", "bias", "metrics_csv", "summary_json"]:
        assert paths[key].exists()


def test_metrics_csv_has_valid_source_class(tmp_path):
    mod = _load_example("run_science_case_highz_galaxy")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"])
    assert set(["metric", "value", "source_class", "component", "notes"]).issubset(df.columns)
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_summary_json_has_non_overclaiming_note(tmp_path):
    mod = _load_example("run_science_case_highz_galaxy")
    paths = mod.generate(tmp_path, quick=True)
    summary = json.loads(paths["summary_json"].read_text())
    assert "note" in summary and "NOT" in summary["note"]
    assert any("not a real" in lim.lower() for lim in summary["limitations"])
    assert "observed_wavelength_um" in summary and "redshift" in summary


def test_summary_declares_truth_levels_and_bias_surface(tmp_path):
    mod = _load_example("run_science_case_highz_galaxy")
    paths = mod.generate(tmp_path, quick=True)
    summary = json.loads(paths["summary_json"].read_text())
    # The four truth levels are declared, and the bias reference is the
    # physically matching observed surface (post-PSF, not deconvolved).
    assert set(summary["truth_levels"]) == {
        "intrinsic", "conditioned", "observed", "reduced_recovery"}
    assert summary["bias_reference_surface"] == "observed"
    assert "not deconvolved" in summary["truth_levels"]["observed"]
    # The reduction stage ran: diagnostics are sky-subtracted products.
    assert summary["metrics"]["sky_subtraction_applied"] is True


def test_bias_metrics_compare_matching_surfaces(tmp_path):
    mod = _load_example("run_science_case_highz_galaxy")
    paths = mod.generate(tmp_path, quick=True)
    df = pd.read_csv(paths["metrics_csv"]).set_index("metric")
    # Renamed metrics: quoted against the observed truth, never implying a
    # deconvolved intrinsic recovery.
    assert "line_flux_bias_vs_observed_truth_frac" in df.index
    assert "velocity_bias_vs_observed_truth_kms" in df.index
    assert "line_flux_recovery_bias_frac" not in df.index
    # On the matched surface the sky-subtracted recovery is unbiased to within
    # noise (the old sky-added, wrong-surface comparison sat at ~ +0.8).
    lf_bias = float(df.loc["line_flux_bias_vs_observed_truth_frac", "value"])
    assert abs(lf_bias) < 0.25
    # The degradation-chain metrics document the truth-level ratios explicitly.
    assert "line_flux_conditioning_ratio" in df.index
    assert "line_flux_observed_over_conditioned" in df.index
