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
    # the explicit product-level distinction.
    from ao_fed_ifu.cube_reconstruction import extract_1d_spectrum

    spec_sky_added = extract_1d_spectrum(result.rebuilt_weighted, ap)
    ax[1, 1].plot(grid, spec_sky_added, color="C1", lw=0.8, alpha=0.8, label="sky-added")
    if sky_subtracted:
        ax[1, 1].plot(grid, result.extracted_spectrum, color="C0", lw=0.9, label="sky-subtracted")
    spec_truth = extract_1d_spectrum(result.observed_truth_cube, ap)
    ax[1, 1].plot(grid, spec_truth, color="k", lw=0.8, ls=":", label="source-only (noiseless)")
    ax[1, 1].axvline(obs, color="r", ls="--", lw=0.8)
    ax[1, 1].set_title("aperture spectrum by reduction level")
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
        rep.metric_row("public_data_components_used", m["public_data_components_used"], "internal_validation", "sky+source+filter"),
    ]
    csv = rep.write_metrics_csv(outdir / "highz_science_metrics.csv", rows)
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
        "metrics": m, "provenance": result.provenance.to_records(),
        "provenance_counts": result.provenance.summary_counts(),
        "public_data_components": result.conditioning_info["components"],
        "limitations": [
            "Synthetic high-z galaxy, not a real observation.",
            "Sky background level is an engineering_estimate, not a calibrated ETC; sky subtraction "
            "assumes a perfectly known forward-modelled sky (scale error configurable).",
            "Recovery bias is quoted against the observed truth (post-PSF+LSF+reconstruction); "
            "no deconvolved intrinsic recovery is claimed.",
            "Weighted reconstruction can over-count flux from closely packed traces; the over-count "
            "cancels in the matched-surface comparison but is visible against pre-PSF truth "
            "(see line_flux_observed_over_conditioned).",
        ],
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
