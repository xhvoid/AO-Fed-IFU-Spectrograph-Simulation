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
                           weighted reconstruction (with the same variance
                           weights as the science cube) -- post-PSF, NOT
                           deconvolved, and reduction-specific rather than an
                           intrinsic/deterministic truth;
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
OUTDIR = REPO / "figures" / "generated"
GEN = "examples/run_science_case_highz_galaxy.py"


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
    obs_lf = diag.line_flux_map(result.observed_truth_cube, grid, lc, window)
    obs_vel = diag.velocity_centroid_map(result.observed_truth_cube, grid, lc, window)
    # Reduced recovery: the pipeline diagnostics (sky-subtracted when enabled).
    rec_lf, rec_vel, rec_snr = result.line_flux_map, result.velocity_map, result.snr_map
    ap = result.aperture_mask
    sky_subtracted = bool(result.metrics["sky_subtraction_applied"])

    # --- Gaussian line-fit recovery vs window-sum recovery (aperture 1-D). ---
    from ao_fed_ifu.cube_reconstruction import extract_1d_spectrum

    science_weighted = result.rebuilt_weighted_skysub if sky_subtracted else result.rebuilt_weighted
    rec_ap_spec = result.extracted_spectrum                          # sky-subtracted aperture spectrum
    obs_ap_spec = extract_1d_spectrum(result.observed_truth_cube, ap)
    var_vox = np.where(np.isfinite(result.variance_cube_weighted), result.variance_cube_weighted, 0.0)
    ap_var_spec = var_vox[ap, :].sum(axis=0)                         # variances add over spaxels
    fit_window = 4.0 * window                                        # a few resolution elements
    fit_rec = diag.fit_gaussian_line(rec_ap_spec, grid, lc, window_um=fit_window, variance_1d=ap_var_spec)
    fit_obs = diag.fit_gaussian_line(obs_ap_spec, grid, lc, window_um=fit_window)
    dlam = float(grid[1] - grid[0])                                  # um per channel
    # Convert the Gaussian's wavelength integral (e-*um) to summed-channel e- so
    # it is comparable to the window-sum estimator (a sum over channels).
    fit_rec_flux_e = fit_rec["line_flux"] / dlam if fit_rec["success"] else float("nan")
    win_rec_flux_e = float(np.nansum(rec_lf[ap]))
    win_obs_flux_e = float(np.nansum(obs_lf[ap]))
    # Fit-based bias vs observed truth (a unit-independent ratio -> no dlam needed),
    # the parametric analogue of the window-sum bias reported below.
    if fit_rec["success"] and fit_obs["success"] and fit_obs["line_flux"] > 0:
        fit_lf_bias_obs = (fit_rec["line_flux"] - fit_obs["line_flux"]) / fit_obs["line_flux"]
        fit_vel_bias_obs = fit_rec["velocity_kms"] - fit_obs["velocity_kms"]
    else:
        fit_lf_bias_obs = float("nan")
        fit_vel_bias_obs = float("nan")

    # --- Covariance-aware aperture SNR (diagonal vs AR(1)-inflated). ---
    # The observed-truth cube is the SAME weighted reduction applied to the
    # noiseless source, so (science - observed_truth) approximately isolates the
    # reconstructed noise (exact for a linear chain; the small deterministic
    # dark/flat residual is sub-dominant and drops out of the autocorrelation).
    cov = diag.aperture_spectrum_uncertainty(
        science_weighted, result.variance_cube_weighted, ap, grid, lc, window,
        noise_reference_cube=result.observed_truth_cube,
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
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    _img(fig, ax[0, 0], rec_lf, f"recovered line flux [e-]\n({red_tag}, not deconvolved)")
    _img(fig, ax[0, 1], rec_vel, f"recovered velocity [km/s]\n({red_tag})", symmetric=True, cmap="coolwarm")
    _img(fig, ax[1, 0], rec_snr, "recovered SNR (matched variance)", cmap="cividis")
    # Aperture spectra: sky-added vs sky-subtracted vs noiseless source-only,
    # the explicit product-level distinction, with the Gaussian fit overplotted.
    spec_sky_added = extract_1d_spectrum(result.rebuilt_weighted, ap)
    ax[1, 1].plot(grid, spec_sky_added, color="C1", lw=0.8, alpha=0.8, label="sky-added")
    if sky_subtracted:
        ax[1, 1].plot(grid, rec_ap_spec, color="C0", lw=0.9, label="sky-subtracted")
    ax[1, 1].plot(grid, obs_ap_spec, color="k", lw=0.8, ls=":", label="source-only (noiseless)")
    if fit_rec["success"]:
        fsel = np.abs(grid - lc) <= fit_window
        # Plot the FULL fitted model c0 + c1*(lambda-lambda0) + Gaussian -- the
        # same model the fit optimised, not just its flat-continuum part.
        model = (fit_rec["continuum"]
                 + fit_rec["continuum_slope"] * (grid[fsel] - lc)
                 + fit_rec["amplitude"] * np.exp(-0.5 * ((grid[fsel] - fit_rec["centroid_um"]) / fit_rec["sigma_um"]) ** 2))
        ax[1, 1].plot(grid[fsel], model, color="C3", lw=1.3, ls="--",
                      label=f"Gaussian fit (R={fit_rec['resolving_power']:.0f})")
    ax[1, 1].axvline(obs, color="r", ls="--", lw=0.8)
    ax[1, 1].set_title("aperture spectrum by reduction level + Gaussian fit")
    ax[1, 1].set_xlabel("wavelength [um]"); ax[1, 1].legend(fontsize=7)
    fig.suptitle(f"High-z galaxy reduced products ({red_tag}; post-PSF, not deconvolved)")
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
    ap_obs = float(np.nansum(obs_lf[ap]))
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
                       "weighted flux with matching (Horne) variance"),
        rep.metric_row("median_snr_nearest", m["median_snr_nearest"], "synthetic_proxy", "reduced recovery",
                       "nearest flux with nearest variance (baseline)"),
        rep.metric_row("recovered_R_bright_spaxel", m["recovered_R_bright_spaxel"], "synthetic_proxy", "central spaxel"),
        rep.metric_row("sky_to_source_ratio", m["sky_to_source_ratio"], "engineering_estimate", "public sky budget"),
        rep.metric_row("sky_subtraction_applied", float(m["sky_subtraction_applied"]), "internal_validation",
                       "reduction", "diagnostics computed on the sky-subtracted cube"),
        rep.metric_row("sky_line_masked_in_window_fraction", m["sky_line_masked_in_window_fraction"],
                       "engineering_estimate", "OH mask report", "reported, not applied"),
        rep.metric_row("filter_weighted_throughput", m["filter_weighted_throughput"], "engineering_estimate", "SVO Ks"),
        rep.metric_row("psf_field_throughput", m["psf_field_throughput"], "synthetic_proxy", "PSF coupling"),
        rep.metric_row("line_flux_bias_vs_observed_truth_frac", lf_bias_obs, "internal_validation",
                       "reduced recovery vs observed truth", "matched surface; NOT vs intrinsic"),
        rep.metric_row("velocity_bias_vs_observed_truth_kms", vel_bias_obs, "internal_validation",
                       "reduced recovery vs observed truth", "matched surface; NOT vs intrinsic"),
        rep.metric_row("line_flux_conditioning_ratio", conditioning_ratio, "engineering_estimate",
                       "conditioned/intrinsic truth", "telluric x filter x continuum reshaping at the line"),
        rep.metric_row("line_flux_observed_over_conditioned", observed_over_cond, "synthetic_proxy",
                       "observed/conditioned truth",
                       "PSF field loss + LSF window spill + weighted-extraction over-count"),
        # --- Gaussian line-fit recovery (parametric estimator, cf. window-sum). ---
        rep.metric_row("fit_success", float(fit_rec["success"]), "internal_validation",
                       "aperture Gaussian fit",
                       "1 = converged line detection; does NOT certify fit quality (see fit_quality_ok)"),
        rep.metric_row("fit_quality_ok", float(fit_rec["fit_quality_ok"]), "internal_validation",
                       "aperture Gaussian fit",
                       f"1 = reduced chi2 in {diag.FIT_QUALITY_CHI2_RANGE}; 0 flags an imperfect line/error model"),
        rep.metric_row("fit_line_flux_recovered_e", fit_rec_flux_e, "synthetic_proxy",
                       "aperture Gaussian fit", "amp*sigma*sqrt(2pi)/dlambda -> summed-channel e-"),
        rep.metric_row("window_line_flux_recovered_e", win_rec_flux_e, "synthetic_proxy",
                       "aperture window-sum", "sum of continuum-subtracted flux over the line window"),
        rep.metric_row("fit_velocity_recovered_kms", fit_rec["velocity_kms"], "synthetic_proxy",
                       "aperture Gaussian fit", "centroid mu vs rest wavelength"),
        rep.metric_row("fit_resolving_power", fit_rec["resolving_power"], "synthetic_proxy",
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
                       "aperture uncertainty", "independent-channel (diagonal) budget"),
        rep.metric_row("aperture_snr_covariance_aware", cov["snr_covariance_aware"], "synthetic_proxy",
                       "aperture uncertainty", "AR(1) covariance-inflated; approximate, not a full matrix"),
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
        "Weighted reconstruction can over-count flux from closely packed traces; the over-count "
        "cancels in the matched-surface comparison but is visible against pre-PSF truth "
        "(see line_flux_observed_over_conditioned).",
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
        "mode": cfg.mode, "line": line, "rest_wavelength_um": REST_FRAME_LINES_UM[line],
        "redshift": redshift, "observed_wavelength_um": obs,
        "band_um": [cfg.lambda_min_um, cfg.lambda_max_um],
        "truth_levels": {
            "intrinsic": "synthetic galaxy cube after the photon budget, before public conditioning",
            "conditioned": "after CALSPEC continuum shape x SkyCalc telluric x SVO Ks filter (pre-PSF)",
            "observed": "reduction-matched: noiseless sky-free source through the same "
                        "PSF+LSF+cross-talk+weighted reconstruction (same variance weights as the "
                        "science cube), post-PSF and not deconvolved; reduction-specific, not an "
                        "intrinsic or strictly deterministic truth",
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
            "diagonal": "independent-channel SNR (snr_map assumption)",
            "covariance_aware": "AR(1) covariance-inflated SNR from the lag-1 spectral "
                                "autocorrelation; approximate, NOT a full covariance matrix; "
                                "conservative: negative measured autocorrelation is clamped to "
                                "zero, so the factor is >= 1 and the covariance-aware SNR never "
                                "exceeds the diagonal SNR",
            "lag1_autocorr_method": cov["lag1_autocorr_method"],
            "lag1_autocorr": cov["lag1_autocorr"],
            "inflation_factor": cov["covariance_inflation_factor"],
            "note": "reconstructed noise is added per-pixel after the LSF and each channel reads a "
                    "distinct detector column, so the isolated-noise autocorrelation is ~0 and the "
                    "diagonal SNR is ~valid here; the diagnostic would catch correlation if present.",
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
