"""Tests that the internal validation suite runs and all trend checks pass."""

import pandas as pd

from ao_fed_ifu import validation


def test_all_validation_checks_pass():
    results = validation.run_all_validations()
    # Seven checks are expected; every one must report ``passed``.
    assert len(results) == 7
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"validation checks failed: {failed}"


def test_validation_dataframe_structure():
    df = validation.validation_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert set(["check", "passed", "detail"]).issubset(df.columns)
    assert len(df) == 7
    assert df["passed"].all()


def test_individual_checks_expose_measurements():
    # Each check should carry the numbers behind its verdict for the report.
    fc = validation.check_flux_conservation()
    assert "ratio" in fc.measured
    bias = validation.check_wavelength_offset_velocity_bias()
    assert "predicted_bias_kms" in bias.measured and "measured_bias_kms" in bias.measured
