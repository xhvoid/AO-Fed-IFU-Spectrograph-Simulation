"""Tests for the high-z emission-line galaxy science case (direction 1)."""

import json

import pandas as pd

from ao_fed_ifu.cli import _load_example
from ao_fed_ifu.data_sources import VALID_SOURCE_CLASSES
from ao_fed_ifu.integration import make_highz_galaxy_config
from ao_fed_ifu.source_cube import REST_FRAME_LINES_UM, observed_wavelength_um


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
