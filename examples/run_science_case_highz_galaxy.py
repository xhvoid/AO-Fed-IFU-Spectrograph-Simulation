#!/usr/bin/env python3
"""Science case: recover a high-z emission-line galaxy from the simulated cube.

A complete, reproducible "truth -> instrument -> reduction -> recovery" story
with **explicit truth levels**, compared like-for-like:

    intrinsic truth        synthetic galaxy cube after the photon budget,
                           before public conditioning;
    conditioned truth      after CALSPEC continuum shape x SkyCalc telluric x
                           SVO Ks filter (pre-PSF);
    observed truth         reduction-matched: the noiseless, sky-free source
                           pushed through the SAME PSF + LSF + cross-talk +
                           cube reconstruction as the science cube -- post-PSF,
                           NOT deconvolved, and reduction-specific rather than
                           an intrinsic truth;
    reduced recovery       the noisy, sky-added observation after sky
                           subtraction (the pipeline's science products).

Recovery bias is quoted against the **observed truth** (the physically matching
surface -- the same reduction operator applied to noiseless source light);
the intrinsic -> conditioned -> observed panels show the degradation chain
explicitly rather than pretending the pipeline deconvolves it.

This is a **synthetic science-recovery test**, not a real galaxy observation and
not a validated redshift/flux measurement.

Run:
    python examples/run_science_case_highz_galaxy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu.integration import make_highz_galaxy_config, run_pipeline  # noqa: E402
from ao_fed_ifu.source_cube import REST_FRAME_LINES_UM, observed_wavelength_um  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_science_case_highz_galaxy.py"


def _fit_aperture_lines(result, window_um: float) -> tuple[dict, dict]:
    """Fit recovery and truth with the same aperture, covariance and channels.

    A channel missing any aperture voxel does not measure the full aperture;
    omit it from BOTH fits instead of fitting a wavelength-dependent aperture.
    Variance is conditional on the selected aperture and extraction: nearest
    voxels are independent (variances add), a deblend needs its covariance.
    """
    from ao_fed_ifu.cube_reconstruction import extract_1d_spectrum

    ap = result.aperture_mask
    voxel_var = result.variance_cube[ap]
    complete = np.all(np.isfinite(voxel_var), axis=0)
    model = result.science_extraction_model
    variance = model.aperture_variance(ap) if model is not None else voxel_var.sum(axis=0)
    variance = np.where(complete, variance, np.inf)
    kwargs = dict(window_um=window_um, variance_1d=variance)
    recovered = diag.fit_gaussian_line(
        result.extracted_spectrum, result.wavelength_grid_um, result.line_center_um, **kwargs
    )
    observed = diag.fit_gaussian_line(
        extract_1d_spectrum(result.observed_truth_cube, ap),
        result.wavelength_grid_um, result.line_center_um, **kwargs
    )
    fit_region = np.abs(result.wavelength_grid_um - result.line_center_um) <= window_um
    complete_fraction = float(np.mean(complete[fit_region])) if np.any(fit_region) else 0.0
    for fitted in (recovered, observed):
        fitted["complete_channel_fraction"] = complete_fraction
    return recovered, observed


def generate(outdir: Path, line: str = "Halpha", redshift: float = 2.30, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    cfg = make_highz_galaxy_config(line=line, redshift=redshift)
    if quick:
        cfg.nx = cfg.ny = 10
        cfg.resolving_power_R = 2000.0
    result = run_pipeline(cfg)
    grid, lc = result.wavelength_grid_um, result.line_center_um
    # The SAME line window the pipeline diagnostics use (3 resolution elements):
    # the truth-surface maps must integrate the same channels as the recovered
    # maps, or the comparison inherits a spurious window-coverage bias.
    window = 3.0 * result.spectrograph.delta_lambda_resel_um(lc)
    obs = observed_wavelength_um(REST_FRAME_LINES_UM[line], redshift)

    # --- Truth levels, compared like-for-like (see module docstring). ---
    # intrinsic / conditioned: pre-PSF input cubes; observed: the noiseless,
    # sky-free source through the SAME PSF+LSF+reconstruction path (post-PSF,
    # NOT deconvolved) -- the surface the reduced recovery should match.
    intrinsic_lf = diag.line_flux_map(result.intrinsic_source_cube, grid, lc, window)
    cond_lf = diag.line_flux_map(result.source_cube, grid, lc, window)
    cond_vel = diag.velocity_centroid_map(result.source_cube, grid, lc, window)
    obs_maps = diag.line_window_maps(result.observed_truth_cube, result.variance_cube, grid, lc, window)
    obs_lf = obs_maps["line_flux"]
    obs_vel = diag.velocity_centroid_map(result.observed_truth_cube, grid, lc, window)
    # Reduced recovery: the pipeline diagnostics (sky-subtracted when enabled).
    rec_lf, rec_vel, rec_snr = result.line_flux_map, result.velocity_map, result.snr_map
    ap = result.aperture_mask
    sky_subtracted = bool(result.metrics["sky_subtraction_applied"])

    # --- Gaussian line-fit recovery vs window-sum recovery (aperture 1-D). ---
    from ao_fed_ifu.cube_reconstruction import extract_1d_spectrum

    rec_ap_spec = result.extracted_spectrum                          # sky-subtracted aperture spectrum
    obs_ap_spec = extract_1d_spectrum(result.observed_truth_cube, ap)
    fit_window = 4.0 * window                                        # a few resolution elements
    fit_rec, fit_obs = _fit_aperture_lines(result, fit_window)
    dlam = float(grid[1] - grid[0])                                  # um per channel
    # Convert the Gaussian's wavelength integral (e-*um) to summed-channel e- so
    # it is comparable to the window-sum estimator (a sum over channels).
    fit_rec_flux_e = fit_rec["line_flux"] / dlam if fit_rec["success"] else float("nan")
    win_rec_flux_e = float(np.nansum(result.line_flux_partial_map[ap]))
    # Fit-based bias vs observed truth (a unit-independent ratio -> no dlam needed),
    # the parametric analogue of the window-sum bias reported below.
    if fit_rec["success"] and fit_obs["success"] and fit_obs["line_flux"] > 0:
        fit_lf_bias_obs = (fit_rec["line_flux"] - fit_obs["line_flux"]) / fit_obs["line_flux"]
        fit_vel_bias_obs = fit_rec["velocity_kms"] - fit_obs["velocity_kms"]
    else:
        fit_lf_bias_obs = float("nan")
        fit_vel_bias_obs = float("nan")

    # --- Covariance-aware aperture SNR (diagonal vs AR(1)-inflated). ---
    # The observed-truth cube is the SAME reduction applied to the noiseless
    # source, so (science - observed_truth) approximately isolates the
    # reconstructed noise after the known bias/dark/flat calibration. Deliberate
    # sky/flat calibration errors would remain deterministic residuals.
    cov = diag.aperture_spectrum_uncertainty(
        result.science_cube, result.variance_cube, ap, grid, lc, window,
        noise_reference_cube=result.observed_truth_cube,
        extraction_model=result.science_extraction_model,
    )

    # --- Figure 1: input truth levels (pre-PSF). ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    _img(fig, ax[0], cond_lf, "conditioned truth line flux [e-]\n(pre-PSF)")
    _img(fig, ax[1], cond_vel, "truth velocity [km/s] (pre-PSF)", symmetric=True, cmap="coolwarm")
    cy, cx = cfg.ny // 2, cfg.nx // 2
    ax[2].plot(grid, result.intrinsic_source_cube[cy, cx, :], color="C7", lw=0.9, label="intrinsic")
    ax[2].plot(grid, result.source_cube[cy, cx, :], color="C0", lw=0.9, label="conditioned")
    ax[2].axvline(obs, color="r", ls="--", lw=0.8, label=f"{line} @ z={redshift}")
    ax[2].set_title("central-spaxel truth spectrum\n(intrinsic vs public-conditioned)")
    ax[2].set_xlabel("wavelength [um]"); ax[2].legend(fontsize=8)
    fig.suptitle(f"High-z galaxy input truth levels ({line}, z={redshift}, observed {obs:.3f} um)")
    p1 = _save(fig, outdir / "highz_truth_cube_summary.png")

    # --- Figure 2: detector frame. ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    _img(fig, ax[0], result.detector_clean, "detector (noiseless)", aspect="auto", cmap="magma")
    _img(fig, ax[1], result.detector_noisy, "detector (noisy)", aspect="auto", cmap="magma")
    fig.suptitle("High-z galaxy on the image-slicer detector")
    p2 = _save(fig, outdir / "highz_detector_frame.png")

    # --- Figure 3: reduced recovered products (sky-added vs sky-subtracted). ---
    red_tag = "sky-subtracted" if sky_subtracted else "sky-added"
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    _img(fig, ax[0, 0], result.line_flux_partial_map, f"line flux [e-] ({red_tag})")
    detected = result.metrics["line_moment_detected_spaxel_fraction"]
    _img(fig, ax[0, 1], rec_vel, f"velocity [km/s] (SNR >= {cfg.moment_min_snr:g}: {detected:.0%} of spaxels)",
         symmetric=True, cmap="coolwarm")
    coverage = result.metrics["line_window_measured_spaxel_fraction"]
    _img(fig, ax[0, 2], rec_snr, f"line SNR (blank = bad pixel in window; {coverage:.0%} complete)",
         cmap="cividis")
    _img(fig, ax[1, 0], result.line_window_coverage_map, "measured fraction of line-window channels",
         cmap="cividis")
    _img(fig, ax[1, 1], result.snr_partial_map, "line SNR over measured channels", cmap="cividis")
    # Aperture spectra: sky-added vs sky-subtracted vs noiseless source-only,
    # the explicit product-level distinction, with the Gaussian fit overplotted.
    spec_sky_added = extract_1d_spectrum(result.science_cube_sky_added, ap)
    ax[1, 2].plot(grid, spec_sky_added, color="C1", lw=0.8, alpha=0.8, label="sky-added")
    if sky_subtracted:
        ax[1, 2].plot(grid, rec_ap_spec, color="C0", lw=0.9, label="sky-subtracted")
    ax[1, 2].plot(grid, obs_ap_spec, color="k", lw=0.8, ls=":", label="source-only (noiseless)")
    if fit_rec["success"]:
        fsel = np.abs(grid - lc) <= fit_window
        # Plot the FULL fitted model c0 + c1*(lambda-lambda0) + Gaussian -- the
        # same model the fit optimised, not just its flat-continuum part.
        model = (fit_rec["continuum"]
                 + fit_rec["continuum_slope"] * (grid[fsel] - lc)
                 + fit_rec["amplitude"] * np.exp(-0.5 * ((grid[fsel] - fit_rec["centroid_um"]) / fit_rec["sigma_um"]) ** 2))
        ax[1, 2].plot(grid[fsel], model, color="C3", lw=1.3, ls="--",
                      label=f"Gaussian fit (R={fit_rec['resolving_power']:.0f})")
    ax[1, 2].axvline(obs, color="r", ls="--", lw=0.8)
    ax[1, 2].set_title("aperture spectra (missing voxels omitted)")
    ax[1, 2].set_xlabel("wavelength [um]"); ax[1, 2].legend(fontsize=7)
    fig.suptitle(f"High-z galaxy: recovered products ({result.science_extraction} extraction, "
                 f"{red_tag}, post-PSF)")
    p3 = _save(fig, outdir / "highz_recovered_products.png")

    # --- Figure 4: recovery vs OBSERVED truth (the matching surface). ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].scatter(obs_lf[ap], rec_lf[ap], s=8, alpha=0.6)
    _diag_line(ax[0], obs_lf[ap], rec_lf[ap])
    ax[0].set_xlabel("observed-truth line flux [e-]"); ax[0].set_ylabel("recovered (reduced)")
    ax[0].set_title("line flux vs observed truth")
    good = np.isfinite(obs_vel[ap]) & np.isfinite(rec_vel[ap])
    ax[1].scatter(obs_vel[ap][good], rec_vel[ap][good], s=8, alpha=0.6)
    _diag_line(ax[1], obs_vel[ap][good], rec_vel[ap][good])
    ax[1].set_xlabel("observed-truth velocity [km/s]"); ax[1].set_ylabel("recovered (reduced)")
    ax[1].set_title("velocity vs observed truth")
    resid = rec_vel - obs_vel
    _img(fig, ax[2], resid, "velocity residual [km/s]\n(recovered - observed truth)", symmetric=True, cmap="coolwarm")
    fig.suptitle("High-z recovery vs observed truth (post-PSF+LSF+reconstruction; NOT a deconvolved intrinsic recovery)")
    p4 = _save(fig, outdir / "highz_recovery_bias.png")

    # --- Metrics + biases (each quoted against its explicit truth surface). ---
    eps = 1e-30
    lf_bias_obs = float(np.nanmedian((rec_lf[ap] - obs_lf[ap]) / (obs_lf[ap] + eps)))
    vel_bias_obs = float(np.nanmedian((rec_vel - obs_vel)[ap]))
    # Degradation chain (aperture-summed line flux ratios between truth levels).
    ap_cond = float(np.nansum(cond_lf[ap]))
    ap_obs = float(np.nansum(obs_maps["partial_line_flux"][ap]))
    ap_intr = float(np.nansum(intrinsic_lf[ap]))
    conditioning_ratio = ap_cond / ap_intr if ap_intr > 0 else float("nan")
    observed_over_cond = ap_obs / ap_cond if ap_cond > 0 else float("nan")
    m = result.metrics
    rows = [
        rep.metric_row("line", line, "literature_derived", "rest-frame line", "nebular line id"),
        rep.metric_row("rest_wavelength_um", REST_FRAME_LINES_UM[line], "literature_derived", "atomic constant"),
        rep.metric_row("redshift", redshift, "engineering_estimate", "scenario", "chosen to place line in band"),
        rep.metric_row("observed_wavelength_um", obs, "literature_derived", "redshifted line"),
        rep.metric_row("median_snr", m["median_snr"], "synthetic_proxy", "reduced recovery",
                       "full-window median over measured spaxels; unavailable windows excluded, not zeroed"),
        rep.metric_row("median_snr_partial_window", m["median_snr_partial_window"], "synthetic_proxy",
                       "partial-window recovery", "measured channels only; NOT full-line SNR"),
        rep.metric_row("mean_line_window_channel_coverage", m["mean_line_window_channel_coverage"],
                       "internal_validation", "extraction coverage"),
        rep.metric_row("median_snr_measured_spaxels", m["median_snr_measured_spaxels"], "synthetic_proxy",
                       "reduced recovery", "median over spaxels with every line-window voxel measured"),
        rep.metric_row("line_window_measured_spaxel_fraction", m["line_window_measured_spaxel_fraction"],
                       "internal_validation", "extraction coverage", "incomplete windows are not physical non-detections"),
        rep.metric_row("unmeasured_voxel_fraction", m["unmeasured_voxel_fraction"], "internal_validation",
                       "extraction coverage",
                       f"{result.science_extraction} extraction; masked, not interpolated"),
        rep.metric_row("science_extraction", m["science_extraction"], "internal_validation", "reduction",
                       "estimator behind the science products"),
        rep.metric_row("median_snr_nearest", m["median_snr_nearest"], "synthetic_proxy", "reduced recovery",
                       "nearest flux with nearest variance"),
        rep.metric_row("median_snr_deblend", m["median_snr_deblend"], "synthetic_proxy", "reduced recovery",
                       "joint least-squares deblend flux with its own variance (trade-study baseline)"),
        rep.metric_row("recovered_R_bright_spaxel", m["recovered_R_bright_spaxel"], "synthetic_proxy", "central spaxel"),
        rep.metric_row("sky_to_source_ratio", m["sky_to_source_ratio"], "engineering_estimate", "public sky budget"),
        rep.metric_row("sky_subtraction_applied", float(m["sky_subtraction_applied"]), "internal_validation",
                       "reduction", "diagnostics computed on the sky-subtracted cube"),
        rep.metric_row("sky_line_masked_in_window_fraction", m["sky_line_masked_in_window_fraction"],
                       "engineering_estimate", "OH mask report", "reported, not applied"),
        rep.metric_row("filter_weighted_throughput", m["filter_weighted_throughput"], "engineering_estimate", "SVO Ks"),
        rep.metric_row("transmission_at_line", m["transmission_at_line"], "engineering_estimate", "filter x atmosphere"),
        rep.metric_row("line_weighted_transmission", m["line_weighted_transmission"], "engineering_estimate", "line-photon-weighted filter x atmosphere"),
        rep.metric_row("line_transmission_status", m["line_transmission_status"], "internal_validation", "band usability"),
        rep.metric_row("detector_calibration", m["detector_calibration"], "internal_validation", "reduction"),
        rep.metric_row("psf_field_throughput", m["psf_field_throughput"], "synthetic_proxy", "PSF coupling"),
        rep.metric_row("line_flux_bias_vs_observed_truth_frac", lf_bias_obs, "internal_validation",
                       "reduced recovery vs observed truth", "matched surface; NOT vs intrinsic"),
        rep.metric_row("velocity_bias_vs_observed_truth_kms", vel_bias_obs, "internal_validation",
                       "reduced recovery vs observed truth", "matched surface; NOT vs intrinsic"),
        rep.metric_row("line_flux_conditioning_ratio", conditioning_ratio, "engineering_estimate",
                       "conditioned/intrinsic truth", "telluric x filter x continuum reshaping at the line"),
        rep.metric_row("line_flux_observed_over_conditioned", observed_over_cond, "synthetic_proxy",
                       "observed/conditioned truth",
                       "PSF field loss + LSF window spill + unmeasured extraction voxels"),
        # --- Gaussian line-fit recovery (parametric estimator, cf. window-sum). ---
        rep.metric_row("fit_success", float(fit_rec["success"]), "internal_validation",
                       "aperture Gaussian fit",
                       "1 = converged line detection; does NOT certify fit quality (see fit_quality_ok)"),
        rep.metric_row("fit_quality_ok", float(fit_rec["fit_quality_ok"]), "internal_validation",
                       "aperture Gaussian fit",
                       f"1 = reduced chi2 in {diag.FIT_QUALITY_CHI2_RANGE}; 0 flags an imperfect line/error model"),
        rep.metric_row("fit_complete_channel_fraction", fit_rec["complete_channel_fraction"],
                       "internal_validation", "aperture Gaussian fit",
                       "fraction of fit-region channels measuring the entire aperture; same mask in truth fit"),
        rep.metric_row("fit_line_flux_recovered_e", fit_rec_flux_e, "synthetic_proxy",
                       "aperture Gaussian fit", "amp*sigma*sqrt(2pi)/dlambda -> summed-channel e-"),
        rep.metric_row("window_line_flux_recovered_e", win_rec_flux_e, "synthetic_proxy",
                       "aperture partial-window sum", "measured voxels only; missing flux not estimated"),
        rep.metric_row("fit_velocity_recovered_kms", fit_rec["velocity_kms"] if fit_rec["success"] else float("nan"), "synthetic_proxy",
                       "aperture Gaussian fit", "centroid mu vs rest wavelength"),
        rep.metric_row("fit_resolving_power", fit_rec["resolving_power"] if fit_rec["success"] else float("nan"), "synthetic_proxy",
                       "aperture Gaussian fit", "lambda0 / fitted FWHM"),
        rep.metric_row("fit_reduced_chi2", fit_rec["reduced_chi2"], "internal_validation",
                       "aperture Gaussian fit",
                       "propagated-variance weighted; ~1 = good error model, >>1 = formal errors underestimated"),
        rep.metric_row("fit_snr_formal", fit_rec["snr_fit"], "synthetic_proxy",
                       "aperture Gaussian fit",
                       "fitted line flux / FORMAL flux error; optimistic when reduced chi2 >> 1"),
        rep.metric_row("fit_snr_chi2_scaled", fit_rec["snr_fit_scaled"], "synthetic_proxy",
                       "aperture Gaussian fit",
                       "formal SNR / sqrt(max(1, reduced chi2)); the conservative quotable SNR"),
        rep.metric_row("fit_line_flux_bias_vs_observed_truth_frac", fit_lf_bias_obs, "internal_validation",
                       "fit recovery vs fit observed truth", "parametric analogue of the window-sum bias"),
        rep.metric_row("fit_velocity_bias_vs_observed_truth_kms", fit_vel_bias_obs, "internal_validation",
                       "fit recovery vs fit observed truth", "parametric analogue of the window-sum bias"),
        # --- Covariance-aware aperture SNR (task: spectral covariance). ---
        rep.metric_row("aperture_snr_diagonal", cov["snr_diagonal"], "synthetic_proxy",
                       "aperture uncertainty", "independent-spaxel, independent-channel (diagonal) budget"),
        rep.metric_row("aperture_snr_covariance_aware", cov["snr_covariance_aware"], "synthetic_proxy",
                       "aperture uncertainty",
                       "exact spatial covariance of the extraction x AR(1) spectral inflation (approximate)"),
        rep.metric_row("aperture_spatial_covariance_factor", cov["spatial_covariance_factor"], "engineering_estimate",
                       "aperture uncertainty",
                       f"Var(aperture sum) / sum of spaxel variances ({cov['spatial_covariance_method']}); "
                       "1 for independent nearest voxels, < 1 for anti-correlated deblended neighbours"),
        rep.metric_row("aperture_covariance_inflation_factor", cov["covariance_inflation_factor"], "engineering_estimate",
                       "aperture uncertainty",
                       "Var(sum) inflation from the lag-1 spectral autocorrelation; conservative (rho<0 clamped, factor >= 1)"),
        rep.metric_row("aperture_spectral_lag1_autocorr", cov["lag1_autocorr"], "synthetic_proxy",
                       "aperture uncertainty",
                       f"reconstructed-noise lag-1 autocorr ({cov['lag1_autocorr_method']}); ~0 => channels ~independent"),
        rep.metric_row("public_data_components_used", m["public_data_components_used"], "internal_validation", "sky+source+filter"),
    ]
    csv = rep.write_metrics_csv(outdir / "highz_science_metrics.csv", rows)
    limitations = [
        "Synthetic high-z galaxy, not a real observation.",
        "Sky background level is an engineering_estimate, not a calibrated ETC; sky subtraction "
        "assumes a perfectly known forward-modelled sky (scale error configurable).",
        "Recovery bias is quoted against the observed truth (post-PSF+LSF+reconstruction); "
        "no deconvolved intrinsic recovery is claimed.",
        f"Science products use the {result.science_extraction} extraction. Voxels it cannot "
        "measure (bad pixels) are masked, not interpolated; Gaussian aperture fits omit "
        "channels missing any aperture voxel. Fit errors remain conditional on the selected "
        "aperture and Gaussian model.",
        "Explicit partial-window sums use surviving measured voxels, whereas the Gaussian fit estimates a "
        "full-aperture line from complete channels. Their fluxes can differ through missing "
        "coverage. Full-window flux/SNR are unavailable for incomplete windows; velocity moments "
        "also require positive SNR >= the reporting threshold. Negative-intensity clipping biases weak-line centroids.",
        "median_snr excludes unavailable line windows. Read it together with "
        "line_window_measured_spaxel_fraction and unmeasured_voxel_fraction.",
        "Continuum-median uncertainty uses a constant-continuum, independent-Gaussian approximation; "
        "the aperture is data-selected and uncertainties are conditional on that selection.",
        "Cache wavelength coverage is not a detectability guarantee: inspect transmission_at_line, "
        "line_weighted_transmission, line_transmission_status and the detection flags.",
        "The covariance-aware SNR uses a single lag-1 AR(1) proxy, not a full spectral "
        "covariance matrix; it is an approximate, conservative inflation (negative measured "
        "autocorrelation is clamped, factor >= 1), not a calibrated error budget.",
    ]
    if not fit_rec["fit_quality_ok"]:
        limitations.append(
            f"The aperture Gaussian fit is a poor statistical description of the spectrum "
            f"(reduced_chi2 = {fit_rec['reduced_chi2']:.1f}): the Gaussian + linear-continuum "
            "model with the propagated variance does not fully describe the data, so formal fit "
            "errors are underestimated and fit_snr_formal is optimistic -- quote fit_snr_chi2_scaled."
        )
    summ = rep.write_summary_json(outdir / "highz_science_summary.json", {
        "configuration": cfg,
        "input_caches": rep.input_cache_identity(result.provenance.to_records()),
        "mode": cfg.mode, "line": line, "rest_wavelength_um": REST_FRAME_LINES_UM[line],
        "redshift": redshift, "observed_wavelength_um": obs,
        "band_um": [cfg.lambda_min_um, cfg.lambda_max_um],
        "truth_levels": {
            "intrinsic": "synthetic galaxy cube after the photon budget, before public conditioning",
            "conditioned": "after CALSPEC continuum shape x SkyCalc telluric x SVO Ks filter (pre-PSF)",
            "observed": "reduction-matched: noiseless sky-free source through the same "
                        "PSF+LSF+cross-talk+cube reconstruction as the science cube, post-PSF "
                        "and not deconvolved; reduction-specific, not an intrinsic truth",
            "reduced_recovery": "noisy sky-added observation after forward-modelled sky subtraction",
        },
        "bias_reference_surface": "observed",
        "recovery_estimators": {
            "window_sum": "fixed-window continuum-subtracted integral over +/-3 resolution elements",
            "gaussian_fit": "Gaussian+continuum least-squares fit of the aperture spectrum "
                            "(flux, velocity, width, fitted R, formal errors)",
        },
        "gaussian_fit_quality": {
            "converged": bool(fit_rec["fit_converged"]),
            "detected": bool(fit_rec["fit_detected"]),
            "quality_ok": bool(fit_rec["fit_quality_ok"]),
            "complete_channel_fraction": fit_rec["complete_channel_fraction"],
            "spatial_covariance_method": cov["spatial_covariance_method"],
            "reduced_chi2": fit_rec["reduced_chi2"],
            "error_scale": fit_rec["error_scale"],
            "snr_formal": fit_rec["snr_fit"],
            "snr_chi2_scaled": fit_rec["snr_fit_scaled"],
            "note": "reduced_chi2 >> 1 means the Gaussian + linear-continuum model with the "
                    "propagated variance does not fully describe the aperture spectrum "
                    "(non-Gaussian LSF wings, telluric continuum structure), so formal fit "
                    "errors are underestimated by ~sqrt(reduced_chi2); quote fit_snr_chi2_scaled.",
        },
        "aperture_uncertainty": {
            "diagonal": "independent-spaxel, independent-channel SNR (snr_map assumption)",
            "covariance_aware": "spatial covariance of the extraction propagated exactly (independent "
                                "voxels for nearest; u^T (A^T W A)^-1 u per channel for the deblend), "
                                "then AR(1)-inflated for the lag-1 spectral autocorrelation "
                                "(approximate, NOT a full spectral covariance matrix; negative "
                                "measured autocorrelation clamped to zero)",
            "spatial_covariance_method": cov["spatial_covariance_method"],
            "spatial_covariance_factor": cov["spatial_covariance_factor"],
            "lag1_autocorr_method": cov["lag1_autocorr_method"],
            "lag1_autocorr": cov["lag1_autocorr"],
            "inflation_factor": cov["covariance_inflation_factor"],
            "note": "reconstructed noise is added per-pixel after the LSF and each channel reads a "
                    "distinct detector column, so the isolated-noise autocorrelation is ~0 and the "
                    "spectral independence is a reasonable approximation here. Spatial "
                    "independence holds only for the nearest read-out; a deblend's covariance "
                    "is propagated explicitly.",
        },
        "metrics": m, "provenance": result.provenance.to_records(),
        "provenance_counts": result.provenance.summary_counts(),
        "public_data_components": result.conditioning_info["components"],
        "limitations": limitations,
    })

    entries = [
        rep.manifest_entry(p.name, "figure", GEN, "synthetic_proxy",
                           ["direct_public_data", "engineering_estimate", "synthetic_proxy", "internal_validation"],
                           "high-z galaxy public-conditioned run", note)
        for p, note in [(p1, "Input truth levels (intrinsic vs conditioned, pre-PSF)."),
                        (p2, "Detector frame."),
                        (p3, "Reduced products (sky-added vs sky-subtracted vs source-only)."),
                        (p4, "Recovery vs observed truth (matched surface, not deconvolved).")]
    ]
    entries.append(rep.manifest_entry(csv.name, "table", GEN, "internal_validation",
                   ["literature_derived", "engineering_estimate", "synthetic_proxy", "internal_validation"],
                   "high-z science run", "Science metrics with provenance; biases vs observed truth."))
    entries.append(rep.manifest_entry(summ.name, "table", GEN, "internal_validation",
                   ["direct_public_data", "literature_derived", "engineering_estimate", "synthetic_proxy"],
                   "high-z science run", "Config, truth levels, provenance, limitations."))
    rep.update_manifest(outdir, entries)
    return {"truth": p1, "detector": p2, "recovered": p3, "bias": p4, "metrics_csv": csv, "summary_json": summ}


def _img(fig, ax, data, title, cmap="viridis", symmetric=False, aspect=None):
    kw = {"origin": "lower", "cmap": cmap}
    if symmetric:
        vmax = np.nanmax(np.abs(data)) or 1.0
        kw.update(vmin=-vmax, vmax=vmax)
    if aspect:
        kw["aspect"] = aspect
    im = ax.imshow(data, **kw); ax.set_title(title); fig.colorbar(im, ax=ax)


def _diag_line(ax, x, y):
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)])); hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote high-z science artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
