"""Benchmark *trend* validation suite: order-of-magnitude internal sanity checks.

This module aggregates a handful of cross-cutting **internal trend checks** over
the physics/engineering extensions and the core pipeline.  Each benchmark turns a
single knob (or compares two regimes) and asserts that a quantity moves the way
first-principles physics requires, or sits in a sensible order-of-magnitude range.

These are **internal trend checks** against the model's own physics -- like
:mod:`ao_fed_ifu.validation`, just framed as a compact pass/fail "benchmark" table.
They are **trend / order-of-magnitude checks only**:

* NOT a calibrated validation against any real instrument;
* NOT an exposure-time calculator or an observatory-performance prediction;
* NOT a reproduction of HARMONI / MUSE / NIRSpec / MOSFIRE / JWST.

Each row's ``source_class`` reflects the dominant provenance of the physics it
exercises -- ``internal_validation`` for a pure noise-model / inverse-mapping
identity, ``engineering_estimate`` for the throughput / detector proxies, and
``synthetic_proxy`` for the AO-PSF proxy.  None is ``direct_public_data``: these
are synthetic trend checks, not measured data.

The seven benchmarks:

    1. snr_sqrt_exposure        photon-limited SNR scales ~ sqrt(t)
    2. resolving_power_trend    recovered R rises with configured R, and stays
                                <~ configured R for an intrinsically resolved line
    3. sky_residual_tracks_error  residual sky fraction ~ injected sky-scale error
    4. throughput_monotonic     combined throughput in [0, 1]; a lossy stage never
                                raises it
    5. ao_psf_trends            diffraction FWHM grows with wavelength; EE rises
                                with Strehl at fixed wavelength
    6. detector_regimes         more reads lower read noise; faint=read-limited,
                                bright=photon-limited
    7. reconstruction_flux      noiseless/cross-talk-free map+reconstruct conserves
                                flux; a trace offset changes the recovered flux

:func:`run_all_benchmarks` runs them all; :func:`benchmark_dataframe` tabulates the
outcome for a CSV artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cube_reconstruction as recon
from . import diagnostics as diag
from . import sky_telluric as st
from .data_sources import load_sky_spectrum
from .detector_model import DetectorConfig
from .detector_systematics import effective_read_noise
from .ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
from .psf_coupling import ao_corrected_psf_proxy, diffraction_fwhm_mas, encircled_energy_per_spaxel
from .source_cube import (
    SourceCubeConfig,
    create_emission_line_cube,
    scale_cube_to_total_electrons,
)
from .spectrograph_model import SpectrographConfig
from .throughput_budget import ThroughputBudget, ThroughputComponent, build_default_budget

BENCHMARK_COLUMNS = ["benchmark", "passed", "source_class", "measured", "expected", "criterion", "notes"]


@dataclass
class BenchmarkResult:
    """Outcome of a single benchmark trend check.

    Attributes
    ----------
    name
        Short identifier of the benchmark.
    passed
        Whether the expected trend / range held.
    measured
        The number(s) behind the verdict -- a scalar or a small JSON-serialisable
        dict (for the CSV / report).
    expected
        Human-readable statement of what was expected (the trend or range).
    criterion
        The explicit pass test that was evaluated (e.g. ``"0.45 <= slope <= 0.55"``),
        kept separate from the prose ``expected`` so the verdict is auditable.
    source_class
        Provenance class of the physics the check exercises -- one of
        ``internal_validation`` / ``engineering_estimate`` / ``synthetic_proxy``.
    notes
        Optional one-line caveat (e.g. "internal trend, not an ETC").
    """

    name: str
    passed: bool
    measured: float | dict
    expected: str
    criterion: str = ""
    source_class: str = "internal_validation"
    notes: str = ""


def _propagated_snr(mean_signal_e: float, cfg: DetectorConfig, exposure_s: float) -> float:
    """SNR of one resolution element from the detector variance model.

    Variance = shot (signal + dark) + read^2; SNR = signal / sqrt(variance) -- the
    same noise model the pipeline reports.
    """
    var = mean_signal_e + cfg.dark_current_e_s_pix * exposure_s + cfg.read_noise_e**2
    return mean_signal_e / np.sqrt(var)


def benchmark_snr_scaling() -> BenchmarkResult:
    """Photon-limited SNR scales ~ sqrt(exposure time)."""
    cfg = DetectorConfig(read_noise_e=3.0, dark_current_e_s_pix=0.0)
    rate_e_per_s = 200.0
    exposures = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    snr = np.array([_propagated_snr(rate_e_per_s * t, cfg, t) for t in exposures])
    slope = float(np.polyfit(np.log(exposures), np.log(snr), 1)[0])
    return BenchmarkResult(
        name="snr_sqrt_exposure",
        passed=bool(0.45 <= slope <= 0.55),
        measured={"slope": slope},
        expected="d log(SNR)/d log(t) in [0.45, 0.55] (photon-limited regime)",
        criterion="0.45 <= dlog(SNR)/dlog(t) <= 0.55",
        notes="Internal trend (SNR ~ sqrt(t)); not an exposure-time calculator.",
    )


def benchmark_resolving_power() -> BenchmarkResult:
    """Recovered R rises with configured R, and stays <~ configured R when resolved."""

    def recovered_R(R_config: float, intrinsic_sigma_um: float = 0.0) -> float:
        spectro = SpectrographConfig(
            lambda_min_um=2.0, lambda_max_um=2.4, resolving_power_R=R_config,
            pixels_per_resolution_element=2.5,
        )
        wl = spectro.wavelength_grid_um()
        centre = spectro.lambda_mid_um
        if intrinsic_sigma_um > 0:
            spectrum = np.exp(-0.5 * ((wl - centre) / intrinsic_sigma_um) ** 2)
        else:
            spectrum = np.full(wl.size, 0.01)
            spectrum[np.argmin(np.abs(wl - centre))] += 1.0
        spectrum = spectro.apply_lsf(spectrum)
        window = 6.0 * spectro.delta_lambda_resel_um()
        return diag.resolving_power_proxy(spectrum, wl, centre, window)

    r_low = recovered_R(2000.0)
    r_high = recovered_R(6000.0)
    # An intrinsically resolved line (sigma ~ one resolution element at R=6000):
    sigma_resel = (2.2 / 6000.0)
    r_resolved = recovered_R(6000.0, intrinsic_sigma_um=sigma_resel)
    rises = bool(np.isfinite(r_low) and np.isfinite(r_high) and r_high > r_low)
    bounded = bool(np.isfinite(r_resolved) and r_resolved <= 1.2 * 6000.0)
    return BenchmarkResult(
        name="resolving_power_trend",
        passed=rises and bounded,
        measured={"R_recovered_lowR": r_low, "R_recovered_highR": r_high,
                  "R_recovered_resolved": r_resolved},
        expected="recovered R increases with configured R; stays <~ configured R for a resolved line",
        criterion="R_recovered(6000) > R_recovered(2000) and R_recovered_resolved <= 1.2*6000",
        notes="Internal trend, not a measured instrument resolving power.",
    )


def benchmark_sky_residual() -> BenchmarkResult:
    """Residual sky fraction tracks the injected sky-subtraction scale error."""
    wl = np.linspace(2.0, 2.4, 200)
    sky, _ = load_sky_spectrum(wl)
    sky_injected = np.clip(np.asarray(sky["radiance"], dtype=float), 0.0, None)
    transmission = np.asarray(sky["transmission"], dtype=float)
    fracs = {}
    for err in (0.05, 0.10):
        # data = injected sky only -> a known sky frame; subtract a mis-scaled
        # estimate so the leftover is exactly -err * sky.
        residual = st.subtract_sky(sky_injected.copy(), sky_injected, scale_error=err)
        fracs[err] = float(st.sky_residual_metrics(residual, sky_injected, transmission)["residual_sky_fraction"])
    passed = abs(fracs[0.05] - 0.05) < 0.01 and abs(fracs[0.10] - 0.10) < 0.01
    return BenchmarkResult(
        name="sky_residual_tracks_error",
        passed=bool(passed),
        measured={"residual_at_err_0.05": fracs[0.05], "residual_at_err_0.10": fracs[0.10]},
        expected="residual_sky_fraction ~ injected scale error (0.05 -> ~0.05, 0.10 -> ~0.10)",
        criterion="|residual_sky_fraction - err| < 0.01 for err in {0.05, 0.10}",
        notes="Controlled inject-then-mis-subtract experiment, not real telluric reduction.",
    )


def benchmark_throughput_monotonicity() -> BenchmarkResult:
    """Combined throughput is in [0, 1] and an extra lossy stage never raises it."""
    grid = np.linspace(1.95, 2.45, 200)
    budget = build_default_budget(grid)
    combined = budget.combined(grid)
    eff = budget.effective_throughput(grid)
    in_unit = bool(np.all(combined >= 0.0) and np.all(combined <= 1.0) and 0.0 <= eff <= 1.0)
    # Append a lossy stage (50% transmission) -> effective must not increase.
    lossy = ThroughputComponent(name="extra_loss", kind="test", transmission=0.5,
                                source_class="synthetic_proxy", notes="test lossy stage")
    budget2 = ThroughputBudget(components=list(budget.components) + [lossy])
    eff2 = budget2.effective_throughput(grid)
    return BenchmarkResult(
        name="throughput_monotonic",
        passed=in_unit and bool(eff2 <= eff + 1e-12),
        measured={"effective": eff, "with_extra_loss": eff2,
                  "combined_min": float(combined.min()), "combined_max": float(combined.max())},
        expected="combined throughput in [0, 1]; adding a lossy stage does not increase it",
        criterion="0 <= combined <= 1 and effective_with_extra_loss <= effective",
        source_class="engineering_estimate",
        notes="Decomposed engineering-estimate budget, not a calibrated throughput.",
    )


def benchmark_ao_psf() -> BenchmarkResult:
    """Diffraction FWHM grows with wavelength; EE rises with Strehl at fixed wavelength."""
    diameter_m = 8.0
    fwhm_blue = diffraction_fwhm_mas(1.2, diameter_m)
    fwhm_red = diffraction_fwhm_mas(2.2, diameter_m)

    lambda_um, fine_scale, spaxel_scale, seeing, n_pix = 2.2, 4.0, 20.0, 600.0, 121

    def ee1(strehl: float) -> float:
        p = ao_corrected_psf_proxy(strehl, lambda_um, diameter_m, seeing, fine_scale, n_pix)
        e1, _e3, _ = encircled_energy_per_spaxel(p, fine_scale, spaxel_scale)
        return float(e1)

    ee_low, ee_high = ee1(0.10), ee1(0.70)
    return BenchmarkResult(
        name="ao_psf_trends",
        passed=bool(fwhm_red > fwhm_blue and ee_high > ee_low),
        measured={"fwhm_blue_mas": float(fwhm_blue), "fwhm_red_mas": float(fwhm_red),
                  "ee_strehl_low": ee_low, "ee_strehl_high": ee_high},
        expected="diffraction FWHM grows ~ lambda/D; encircled energy rises with Strehl",
        criterion="fwhm(2.2um) > fwhm(1.2um) and EE(Strehl=0.7) > EE(Strehl=0.1)",
        source_class="synthetic_proxy",
        notes="Wavelength-dependent AO proxy, not AO telemetry.",
    )


def benchmark_detector_regimes() -> BenchmarkResult:
    """More reads lower read noise; faint=read-limited, bright=photon-limited."""
    rn = 12.0
    rn_1 = effective_read_noise(rn, 1)
    rn_16 = effective_read_noise(rn, 16, mode="fowler")

    exposure = 60.0
    cfg_lo = DetectorConfig(read_noise_e=5.0, dark_current_e_s_pix=0.0)
    cfg_hi = DetectorConfig(read_noise_e=25.0, dark_current_e_s_pix=0.0)
    snr_faint_lo = _propagated_snr(30.0, cfg_lo, exposure)
    snr_faint_hi = _propagated_snr(30.0, cfg_hi, exposure)
    snr_bright_lo = _propagated_snr(4000.0, cfg_lo, exposure)
    snr_bright_hi = _propagated_snr(4000.0, cfg_hi, exposure)
    faint_drop = (snr_faint_lo - snr_faint_hi) / snr_faint_lo
    bright_drop = (snr_bright_lo - snr_bright_hi) / snr_bright_lo
    passed = bool(rn_16 < rn_1 and snr_faint_hi < snr_faint_lo and bright_drop < faint_drop)
    return BenchmarkResult(
        name="detector_regimes",
        passed=passed,
        measured={"read_noise_1read": float(rn_1), "read_noise_16read": float(rn_16),
                  "faint_snr_drop_frac": float(faint_drop), "bright_snr_drop_frac": float(bright_drop)},
        expected="more reads -> lower read noise; faint source read-limited (big SNR drop), "
                 "bright source photon-limited (small drop)",
        criterion="read_noise(16 reads) < read_noise(1 read) and faint_snr_drop > bright_snr_drop",
        source_class="engineering_estimate",
        notes="H2RG-like engineering proxies, not a measured device model.",
    )


def benchmark_reconstruction() -> BenchmarkResult:
    """Noiseless/cross-talk-free map+reconstruct conserves flux; a trace offset changes it."""
    cfg = SourceCubeConfig(nx=8, ny=8, nlambda=40, lambda_min_um=2.0, lambda_max_um=2.4)
    spectro = SpectrographConfig(lambda_min_um=cfg.lambda_min_um, lambda_max_um=cfg.lambda_max_um,
                                 resolving_power_R=3000.0)
    cube = create_emission_line_cube(cfg, line_flux_rel=5.0, velocity_gradient_kms_per_spaxel=10.0)
    cube = scale_cube_to_total_electrons(cube, total_electrons=1.0e6)
    mapping = build_slicer_mapping(IFUConfig(nx_spaxel=cfg.nx, ny_spaxel=cfg.ny), spectro, cfg.wavelength_grid_um)

    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
    fc = recon.flux_conservation(cube, rebuilt)

    offset_cube = recon.reconstruct_cube_trace_offset(frame, mapping, row_offset=1)
    fc_off = recon.flux_conservation(cube, offset_cube)
    flux_change = abs(fc_off["ratio"] - 1.0)
    voxel_change = float(np.mean(np.abs(offset_cube - rebuilt)) / (np.mean(np.abs(rebuilt)) + 1e-30))

    conserves = abs(fc["ratio"] - 1.0) <= 1e-9
    biased = (flux_change > 1e-6) or (voxel_change > 1e-3)
    return BenchmarkResult(
        name="reconstruction_flux",
        passed=bool(conserves and biased),
        measured={"flux_ratio_nominal": float(fc["ratio"]),
                  "flux_ratio_trace_offset": float(fc_off["ratio"]),
                  "trace_offset_voxel_change_frac": voxel_change},
        expected="nominal flux ratio = 1 (conserved); a 1-row trace offset gives a finite flux/voxel change",
        criterion="|flux_ratio_nominal - 1| <= 1e-9 and (|flux_ratio_offset - 1| > 1e-6 or voxel_change > 1e-3)",
        notes="Controlled inverse-mapping illustration, not a real reduction pipeline.",
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run every benchmark trend check and return the results."""
    return [
        benchmark_snr_scaling(),
        benchmark_resolving_power(),
        benchmark_sky_residual(),
        benchmark_throughput_monotonicity(),
        benchmark_ao_psf(),
        benchmark_detector_regimes(),
        benchmark_reconstruction(),
    ]


def _measured_to_cell(measured: float | dict) -> str:
    """Render a measured scalar/dict compactly for a CSV cell (rounded JSON)."""
    if isinstance(measured, dict):
        rounded = {k: (round(v, 6) if isinstance(v, (int, float)) else v) for k, v in measured.items()}
        return json.dumps(rounded)
    return json.dumps(round(float(measured), 6))


def benchmark_dataframe(results: list[BenchmarkResult] | None = None) -> pd.DataFrame:
    """Tabulate benchmark results as a DataFrame (one row per benchmark).

    Columns: ``benchmark``, ``passed``, ``source_class``, ``measured``,
    ``expected``, ``criterion``, ``notes``.  Every row carries a valid provenance
    class (``internal_validation`` / ``engineering_estimate`` / ``synthetic_proxy``)
    so the table is self-describing: these are internal trend / order-of-magnitude
    checks, not a validation against a real instrument.
    """
    if results is None:
        results = run_all_benchmarks()
    rows = [
        {
            "benchmark": r.name,
            "passed": r.passed,
            "source_class": r.source_class,
            "measured": _measured_to_cell(r.measured),
            "expected": r.expected,
            "criterion": r.criterion,
            "notes": r.notes,
        }
        for r in results
    ]
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)
