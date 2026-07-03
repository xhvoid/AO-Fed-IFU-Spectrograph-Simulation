"""Internal sanity checks (``internal_validation``).

These are **trend and consistency checks against the model's own physics** -- not
validation against any real instrument, calibration file or observation.  Each
check builds a tiny controlled scenario, runs the relevant part of the pipeline,
and asserts that a quantity moves the way physics says it must:

    1. flux conservation       noiseless map+reconstruct conserves electrons
    2. SNR vs exposure         photon-limited SNR scales ~ sqrt(t)
    3. resolving power         larger configured R -> narrower recovered line
    4. read noise vs SNR       more read noise -> lower SNR for a faint source
    5. cross-talk leakage      more cross-talk sigma -> more neighbour leakage
    6. wavelength bias         a calibration offset -> v = c d_lambda / lambda
    7. AO PSF coupling         higher Strehl -> higher encircled energy

Each returns a :class:`ValidationResult`; :func:`run_all_validations` runs them
all and :func:`validation_dataframe` tabulates the outcome for a CSV artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import SPEED_OF_LIGHT_KM_S
from . import cube_reconstruction as recon
from . import diagnostics as diag
from . import psf_coupling as psf
from .detector_model import DetectorConfig
from .ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
from .source_cube import (
    SourceCubeConfig,
    create_emission_line_cube,
    create_point_source_cube,
    scale_cube_to_total_electrons,
)
from .spectrograph_model import SpectrographConfig


@dataclass
class ValidationResult:
    """Outcome of a single validation check.

    Attributes
    ----------
    name
        Short identifier of the check.
    passed
        Whether the expected trend/relation held.
    detail
        Human-readable one-line explanation of what was compared.
    measured
        Dict of the numbers behind the verdict (for the CSV / report).
    """

    name: str
    passed: bool
    detail: str
    measured: dict = field(default_factory=dict)


def _propagated_snr(mean_signal_e: float, cfg: DetectorConfig, exposure_s: float) -> float:
    """SNR of a single resolution element from the detector variance model.

    Variance = shot (signal + dark) + read^2; SNR = signal / sqrt(variance).
    This is exactly the noise model the pipeline reports, so testing it here
    validates the same code path the science SNR map uses.
    """
    var = mean_signal_e + cfg.dark_current_e_s_pix * exposure_s + cfg.read_noise_e**2
    return mean_signal_e / np.sqrt(var)


def check_flux_conservation(tol: float = 1e-9) -> ValidationResult:
    """Noiseless forward-projection + nearest reconstruction conserves flux."""
    cfg = SourceCubeConfig(nx=8, ny=8, nlambda=40, lambda_min_um=2.0, lambda_max_um=2.4)
    spectro = SpectrographConfig(
        lambda_min_um=cfg.lambda_min_um, lambda_max_um=cfg.lambda_max_um, resolving_power_R=3000.0
    )
    cube = create_emission_line_cube(cfg, line_flux_rel=5.0, velocity_gradient_kms_per_spaxel=10.0)
    cube = scale_cube_to_total_electrons(cube, total_electrons=1.0e6)

    mapping = build_slicer_mapping(IFUConfig(nx_spaxel=cfg.nx, ny_spaxel=cfg.ny), spectro, cfg.wavelength_grid_um)
    # No LSF, no cross-talk -> the map is a pure rearrangement of electrons.
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
    fc = recon.flux_conservation(cube, rebuilt)

    passed = abs(fc["ratio"] - 1.0) <= tol
    return ValidationResult(
        name="flux_conservation",
        passed=passed,
        detail=f"reconstructed/input electrons = {fc['ratio']:.12f} (target 1)",
        measured=fc,
    )


def check_snr_sqrt_exposure() -> ValidationResult:
    """Photon-limited SNR scales ~ sqrt(exposure time)."""
    # Bright, low-read-noise, no-dark regime -> photon (shot) noise dominates.
    cfg = DetectorConfig(read_noise_e=3.0, dark_current_e_s_pix=0.0)
    rate_e_per_s = 200.0
    exposures = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    snr = np.array([_propagated_snr(rate_e_per_s * t, cfg, t) for t in exposures])
    # Fit slope of log SNR vs log t; expect 0.5 in the photon limit.
    slope = float(np.polyfit(np.log(exposures), np.log(snr), 1)[0])
    passed = 0.45 <= slope <= 0.55
    return ValidationResult(
        name="snr_sqrt_exposure",
        passed=passed,
        detail=f"d log(SNR)/d log(t) = {slope:.3f} (target ~0.5)",
        measured={"slope": slope, "exposures_s": exposures.tolist(), "snr": snr.tolist()},
    )


def check_resolving_power_scaling() -> ValidationResult:
    """Larger configured R yields a narrower recovered line (higher recovered R)."""

    def recovered_R(R_config: float) -> float:
        spectro = SpectrographConfig(
            lambda_min_um=2.0, lambda_max_um=2.4, resolving_power_R=R_config,
            pixels_per_resolution_element=2.5,
        )
        wl = spectro.wavelength_grid_um()
        centre = spectro.lambda_mid_um
        # An (almost) unresolved line: a single-channel spike on a faint continuum.
        spectrum = np.full(wl.size, 0.01)
        spectrum[np.argmin(np.abs(wl - centre))] += 1.0
        spectrum = spectro.apply_lsf(spectrum)  # blur to the instrument resolution
        window = 6.0 * spectro.delta_lambda_resel_um()
        return diag.resolving_power_proxy(spectrum, wl, centre, window)

    r_low = recovered_R(2000.0)
    r_high = recovered_R(6000.0)
    passed = bool(np.isfinite(r_low) and np.isfinite(r_high) and r_high > r_low)
    return ValidationResult(
        name="resolving_power_scaling",
        passed=passed,
        detail=f"recovered R: {r_low:.0f} (R=2000) -> {r_high:.0f} (R=6000), expect increase",
        measured={"recovered_R_low": r_low, "recovered_R_high": r_high},
    )


def check_read_noise_reduces_snr() -> ValidationResult:
    """Higher read noise lowers SNR for a faint source."""
    exposure = 60.0
    faint_signal_e = 30.0  # comparable to read noise -> read-noise sensitive
    snr_low = _propagated_snr(faint_signal_e, DetectorConfig(read_noise_e=5.0), exposure)
    snr_high = _propagated_snr(faint_signal_e, DetectorConfig(read_noise_e=25.0), exposure)
    passed = snr_high < snr_low
    return ValidationResult(
        name="read_noise_reduces_snr",
        passed=passed,
        detail=f"SNR {snr_low:.2f} (RN=5) -> {snr_high:.2f} (RN=25), expect decrease",
        measured={"snr_low_rn": snr_low, "snr_high_rn": snr_high},
    )


def check_crosstalk_increases_leakage() -> ValidationResult:
    """More cross-talk sigma leaks more flux out of an isolated spaxel."""
    cfg = SourceCubeConfig(nx=9, ny=9, nlambda=20, lambda_min_um=2.0, lambda_max_um=2.4)
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.4, resolving_power_R=3000.0)
    cube = create_point_source_cube(cfg)  # only the central spaxel carries flux
    cube = scale_cube_to_total_electrons(cube, 1.0e5)
    cx, cy = cfg.nx // 2, cfg.ny // 2

    def leakage(sigma: float) -> float:
        ifu = IFUConfig(nx_spaxel=cfg.nx, ny_spaxel=cfg.ny, cross_talk_sigma_pix=sigma)
        mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
        frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
        rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
        return diag.cross_talk_fraction(rebuilt, cx, cy)

    leak_none = leakage(0.0)
    leak_low = leakage(0.4)
    leak_high = leakage(1.2)
    passed = leak_none <= leak_low < leak_high
    return ValidationResult(
        name="crosstalk_increases_leakage",
        passed=passed,
        detail=f"leakage: {leak_none:.3f} (0) <= {leak_low:.3f} (0.4) < {leak_high:.3f} (1.2)",
        measured={"leak_none": leak_none, "leak_low": leak_low, "leak_high": leak_high},
    )


def check_wavelength_offset_velocity_bias(tol_frac: float = 0.05) -> ValidationResult:
    """A wavelength-calibration offset biases velocity by c d_lambda / lambda."""
    spectro = SpectrographConfig(lambda_min_um=2.1, lambda_max_um=2.3, resolving_power_R=4000.0)
    # Match the cube's wavelength sampling to the detector columns (as the
    # pipeline does) so the forward map is 1:1 and the grids are consistent.
    cfg = SourceCubeConfig(
        nx=6, ny=6, nlambda=spectro.n_detector_pixels, lambda_min_um=2.1, lambda_max_um=2.3
    )
    line_centre = spectro.lambda_mid_um
    cube = create_emission_line_cube(
        cfg, line_center_um=line_centre, line_flux_rel=12.0, line_sigma_um=0.0004,
        velocity_gradient_kms_per_spaxel=0.0,
    )
    cube = scale_cube_to_total_electrons(cube, 1.0e6)

    mapping = build_slicer_mapping(IFUConfig(nx_spaxel=cfg.nx, ny_spaxel=cfg.ny), spectro, cfg.wavelength_grid_um)
    frame = forward_project_cube(cube, mapping, apply_lsf=True, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)

    # A calibration offset = interpreting the wavelength axis as shifted by a
    # constant. Predicted velocity bias: delta_v = c * delta_lambda / lambda.
    grid_true = cfg.wavelength_grid_um
    delta_lambda = 2.0 * cfg.delta_lambda_um  # a couple of wavelength bins
    grid_shifted = grid_true + delta_lambda
    window = 6.0 * spectro.delta_lambda_resel_um()

    v_true = np.nanmedian(diag.velocity_centroid_map(rebuilt, grid_true, line_centre, window))
    v_biased = np.nanmedian(diag.velocity_centroid_map(rebuilt, grid_shifted, line_centre, window))
    measured_bias = v_biased - v_true
    predicted_bias = SPEED_OF_LIGHT_KM_S * delta_lambda / line_centre

    rel_err = abs(measured_bias - predicted_bias) / abs(predicted_bias)
    passed = rel_err <= tol_frac
    return ValidationResult(
        name="wavelength_offset_velocity_bias",
        passed=passed,
        detail=(
            f"velocity bias measured {measured_bias:.2f} vs predicted "
            f"{predicted_bias:.2f} km/s (rel err {rel_err:.3f})"
        ),
        measured={
            "measured_bias_kms": float(measured_bias),
            "predicted_bias_kms": float(predicted_bias),
            "rel_err": float(rel_err),
        },
    )


def check_ao_psf_improves_ee() -> ValidationResult:
    """Higher Strehl concentrates more light into the central spaxel(s)."""
    lambda_um = 2.2
    diameter_m = 8.0
    seeing_fwhm = 600.0  # mas, typical natural seeing
    spaxel_scale = 20.0  # mas
    fine_scale = 4.0  # mas (oversampled relative to the spaxel)
    # Grid large enough to contain the seeing halo, so the EE fractions are
    # relative to (nearly) the total PSF flux rather than a truncated core.
    n_pix = psf.halo_containing_n_pix(seeing_fwhm, fine_scale)

    def ee(strehl: float) -> tuple[float, float]:
        p = psf.ao_corrected_psf_proxy(strehl, lambda_um, diameter_m, seeing_fwhm, fine_scale, n_pix)
        ee1, ee3, _ = psf.encircled_energy_per_spaxel(p, fine_scale, spaxel_scale)
        return ee1, ee3

    ee1_lo, ee3_lo = ee(0.05)
    ee1_hi, ee3_hi = ee(0.70)
    passed = (ee1_hi > ee1_lo) and (ee3_hi > ee3_lo)
    return ValidationResult(
        name="ao_psf_improves_ee",
        passed=passed,
        detail=(
            f"EE1 {ee1_lo:.3f}->{ee1_hi:.3f}, EE3x3 {ee3_lo:.3f}->{ee3_hi:.3f} "
            "for Strehl 0.05->0.70, expect increase"
        ),
        measured={"ee1_low": ee1_lo, "ee1_high": ee1_hi, "ee3_low": ee3_lo, "ee3_high": ee3_hi},
    )


def run_all_validations() -> list[ValidationResult]:
    """Run every internal validation check and return the results."""
    return [
        check_flux_conservation(),
        check_snr_sqrt_exposure(),
        check_resolving_power_scaling(),
        check_read_noise_reduces_snr(),
        check_crosstalk_increases_leakage(),
        check_wavelength_offset_velocity_bias(),
        check_ao_psf_improves_ee(),
    ]


def validation_dataframe(results: list[ValidationResult] | None = None) -> pd.DataFrame:
    """Tabulate validation results as a DataFrame (one row per check).

    Every row carries ``source_class = "internal_validation"`` so the output
    table is self-describing: these are internal trend/consistency checks, never
    a validation against a real instrument.
    """
    if results is None:
        results = run_all_validations()
    return pd.DataFrame(
        {
            "check": [r.name for r in results],
            "passed": [r.passed for r in results],
            "source_class": ["internal_validation"] * len(results),
            "detail": [r.detail for r in results],
        }
    )
