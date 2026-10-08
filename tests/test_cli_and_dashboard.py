"""Tests for the CLI (direction 6) and the optional dashboard (direction 7)."""

import pandas as pd
import pytest

from ao_fed_ifu.cli import _load_example, main
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES


def test_cli_validate_returns_zero(capsys):
    assert main(["validate"]) == 0
    assert "validation checks passed" in capsys.readouterr().out


def test_cli_run_toy_writes_csv_with_source_class(tmp_path):
    assert main(["run", "--mode", "toy", "--out", str(tmp_path), "--quick"]) == 0
    csv = tmp_path / "toy_metrics.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_cli_public_data_quick(tmp_path):
    assert main(["public-data", "--out", str(tmp_path), "--quick"]) == 0
    assert (tmp_path / "artifact_manifest.csv").exists()


def test_cli_invalid_mode_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["run", "--mode", "bogus"])
    assert exc.value.code != 0


def test_cli_run_highz_is_package_native(tmp_path):
    # `run --mode highz` must not depend on examples/ (package-native path).
    assert main(["run", "--mode", "highz", "--out", str(tmp_path), "--quick"]) == 0
    assert (tmp_path / "highz_metrics.csv").exists()
    df = pd.read_csv(tmp_path / "highz_metrics.csv")
    assert set(df["source_class"]).issubset(set(VALID_SOURCE_CLASSES))


def test_load_example_missing_raises_helpful():
    with pytest.raises(SystemExit) as exc:
        _load_example("definitely_not_a_real_generator")
    assert "source checkout" in str(exc.value)


@pytest.mark.parametrize("cmd,artifact", [
    ("sky-telluric", "sky_telluric_inputs.png"),
    ("compare", "mode_comparison_dashboard.png"),
    ("systematics", "systematics_bias_matrix.png"),
    ("reconstruct-demo", "reconstruction_method_comparison.png"),
    # Physics / engineering extension figure-suites (P1-P8).
    ("throughput", "throughput_budget_waterfall.png"),
    ("ao-psf", "ao_psf_wavelength_grid.png"),
    ("observing", "coadd_snr_gain.png"),
    ("detector", "detector_noise_components.png"),
    ("geometry", "trace_geometry_map.png"),
    ("sky-variability", "sky_variability_examples.png"),
    ("galaxy", "galaxy_source_truth_maps.png"),
    ("benchmark", "benchmark_trend_dashboard.png"),
])
def test_cli_figure_suites_create_missing_outdir(cmd, artifact, tmp_path):
    # Regression: --out pointing at a directory that does NOT yet exist must work
    # (the generators previously saved figures before creating the directory), and
    # every figure-suite command must run in --quick mode and write its manifest.
    out = tmp_path / f"{cmd.replace('-', '_')}_new"
    assert not out.exists()
    assert main([cmd, "--out", str(out), "--quick"]) == 0
    assert out.exists()
    assert (out / artifact).exists()
    assert (out / "artifact_manifest.csv").exists()


def test_dashboard_imports_safely_and_formats_metrics():
    mod = _load_example("run_interactive_dashboard")
    # Formatting helpers work without a Streamlit runtime.
    assert "median_snr" in mod.format_metric("median_snr", 12.3456)
    assert "yes" in mod.format_metric("flag", True)
    assert "e+06" in mod.format_metric("electrons", 3.0e6)
    assert "direct_public_data=3" in mod.provenance_summary({"direct_public_data": 3, "synthetic_proxy": 0})


def test_dashboard_main_gracefully_skips_without_streamlit():
    try:
        import streamlit  # noqa: F401
        pytest.skip("Streamlit installed; rendering needs a Streamlit runtime")
    except ImportError:
        mod = _load_example("run_interactive_dashboard")
        assert mod.main() == 0  # prints the install hint and returns 0


def test_cli_benchmark_failure_sets_exit_status(tmp_path, monkeypatch):
    # A failed benchmark must fail the CLI, as the standalone script already does.
    from ao_fed_ifu import benchmark_validation as bv

    real = bv.run_all_benchmarks

    def one_failure():
        results = real()
        results[0].passed = False
        return results

    monkeypatch.setattr(bv, "run_all_benchmarks", one_failure)
    assert main(["benchmark", "--out", str(tmp_path)]) == 1
