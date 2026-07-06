#!/usr/bin/env python3
"""Sky-subtraction and telluric-correction residuals (controlled experiment).

Makes the ESO SkyCalc public data a technical highlight: inject a known sky
background and telluric absorption, then remove them *imperfectly* and measure
the residual's effect on line flux / velocity / SNR.

This is a controlled synthetic experiment, **not** a real reduction pipeline.
SkyCalc radiance/transmission are ``direct_public_data`` when cached; the
radiance->electrons scaling is an ``engineering_estimate``.

Run:
    python examples/run_sky_telluric_residuals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S  # noqa: E402
from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu import sky_telluric as skt  # noqa: E402
from ao_fed_ifu.data_sources import load_filter_curve, load_sky_spectrum  # noqa: E402
from ao_fed_ifu.source_cube import (  # noqa: E402
    SourceCubeConfig, create_emission_line_cube, scale_cube_to_total_electrons,
)

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_sky_telluric_residuals.py"


def generate(outdir: Path, quick: bool = False, cache_dir=None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    n = 6 if quick else 10
    cfg = SourceCubeConfig(nx=n, ny=n, nlambda=300 if quick else 500,
                           lambda_min_um=2.00, lambda_max_um=2.40,
                           exposure_s=600.0, telescope_diameter_m=8.0, throughput_scalar=0.3)
    grid = cfg.wavelength_grid_um
    line_center = 2.20
    window = 0.004
    read_noise = 10.0

    # Public (or synthetic) sky + filter.
    sky, sky_prov = load_sky_spectrum(grid, cache_dir=cache_dir)
    filt, filt_prov = load_filter_curve("K", cache_dir=cache_dir)
    filt_t = np.clip(np.interp(grid, filt["wavelength_um"], filt["transmission"], left=0, right=0), 0, 1)
    transmission = np.clip(sky["transmission"], 1e-3, 1.0)

    # Truth source cube (flat continuum + line), then atmospheric absorption.
    source = scale_cube_to_total_electrons(
        create_emission_line_cube(cfg, line_center_um=line_center, line_flux_rel=12.0, line_sigma_um=0.0008), 4.0e6)
    source_absorbed = skt.apply_telluric_absorption(source, transmission)

    st_cfg = skt.SkyTelluricConfig(
        collecting_area_m2=cfg.collecting_area_m2, exposure_s=cfg.exposure_s,
        throughput_scalar=cfg.throughput_scalar, spaxel_scale_mas=cfg.spaxel_scale_mas,
        delta_lambda_um=cfg.delta_lambda_um, sky_scaling=1.0)
    sky_e = skt.build_sky_background_spectrum(grid, sky, st_cfg, filter_transmission=filt_t)
    observed = source_absorbed + sky_e[None, None, :]
    variance = np.clip(observed, 0, None) + read_noise**2

    scale_errors = [0.0, 0.05, 0.10] if quick else [0.0, 0.02, 0.05, 0.10]
    full = np.ones((cfg.ny, cfg.nx), dtype=bool)

    # --- Figure 1: inputs. ---
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(grid, sky_e); ax[0, 0].set_title(f"sky background [e-/spaxel/bin] ({sky_prov.source_class})")
    ax[0, 1].plot(grid, transmission); ax[0, 1].set_title(f"telluric transmission ({sky_prov.source_class})")
    ax[1, 0].plot(grid, filt_t); ax[1, 0].set_title(f"SVO Ks bandpass ({filt_prov.source_class})")
    ax[1, 1].plot(grid, source[cfg.ny // 2, cfg.nx // 2, :], label="truth source")
    ax[1, 1].plot(grid, observed[cfg.ny // 2, cfg.nx // 2, :], label="observed (src*tell+sky)", alpha=0.8)
    ax[1, 1].axvline(line_center, color="r", ls="--", lw=0.7); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("central spaxel")
    for a in ax.ravel():
        a.set_xlabel("wavelength [um]")
    fig.suptitle("Sky / telluric inputs (public data) and the observed spectrum")
    p1 = _save(fig, outdir / "sky_telluric_inputs.png")

    # --- Figure 2: sky-subtraction residuals. ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    resid_fracs = []
    for se in scale_errors:
        subtracted = skt.subtract_sky(observed, sky_e, scale_error=se)
        residual = subtracted - source_absorbed  # leftover sky (truth source removed)
        ax[0].plot(grid, residual[cfg.ny // 2, cfg.nx // 2, :], lw=0.8, label=f"scale err {se:.0%}")
        resid_fracs.append(skt.sky_residual_metrics(residual, sky_e, transmission)["residual_sky_fraction"])
    ax[0].set_title("sky-subtraction residual spectrum"); ax[0].set_xlabel("wavelength [um]"); ax[0].legend(fontsize=8)
    ax[1].plot([s * 100 for s in scale_errors], resid_fracs, "o-")
    ax[1].set_xlabel("sky scale error [%]"); ax[1].set_ylabel("residual sky fraction")
    ax[1].set_title("residual sky vs scale error"); ax[1].grid(alpha=0.3)
    p2 = _save(fig, outdir / "sky_subtraction_residuals.png")

    # --- Figure 3: telluric-correction effect. ---
    subtracted0 = skt.subtract_sky(observed, sky_e, scale_error=0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(grid, source[cfg.ny // 2, cfg.nx // 2, :], label="truth source", lw=1.0)
    ax.plot(grid, subtracted0[cfg.ny // 2, cfg.nx // 2, :], label="sky-subtracted (still absorbed)", lw=0.9, alpha=0.8)
    corrected0 = skt.correct_telluric(subtracted0, transmission, transmission_error=0.0)
    ax.plot(grid, corrected0[cfg.ny // 2, cfg.nx // 2, :], label="telluric-corrected", lw=0.9)
    ax.axvline(line_center, color="r", ls="--", lw=0.7); ax.legend(fontsize=8)
    ax.set_title(f"Telluric correction (min transmission {np.min(transmission):.2f})")
    ax.set_xlabel("wavelength [um]"); ax.set_ylabel("flux [e-]")
    p3 = _save(fig, outdir / "telluric_correction_effect.png")

    # --- Figure 4: science impact vs residual. ---
    truth_lf = diag.line_flux_map(source, grid, line_center, window)
    truth_v = diag.velocity_centroid_map(source, grid, line_center, window)
    truth_snr = diag.snr_map(source, variance, grid, line_center, window)
    lf_bias, v_bias, snr_loss = [], [], []
    for se in scale_errors:
        subtracted = skt.subtract_sky(observed, sky_e, scale_error=se)
        corrected = skt.correct_telluric(subtracted, transmission, transmission_error=se)
        lf = diag.line_flux_map(corrected, grid, line_center, window)
        v = diag.velocity_centroid_map(corrected, grid, line_center, window)
        snr = diag.snr_map(corrected, variance, grid, line_center, window)
        lf_bias.append(float(np.nanmedian((lf[full] - truth_lf[full]) / (truth_lf[full] + 1e-30))))
        v_bias.append(float(np.nanmedian((v - truth_v)[full])))
        snr_loss.append(float(np.nanmedian((truth_snr[full] - snr[full]) / (truth_snr[full] + 1e-30))))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xe = [s * 100 for s in scale_errors]
    ax.plot(xe, np.array(lf_bias) * 100, "o-", label="line-flux bias [%]")
    ax.plot(xe, v_bias, "s-", label="velocity bias [km/s]")
    ax.plot(xe, np.array(snr_loss) * 100, "^-", label="SNR loss [%]")
    ax.set_xlabel("sky + telluric residual error [%]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Science impact of sky/telluric residuals (controlled, not a real pipeline)")
    p4 = _save(fig, outdir / "sky_residual_science_impact.png")

    # --- Tables. ---
    rows = [
        rep.metric_row("sky_provenance", sky_prov.source_class, sky_prov.source_class, "ESO SkyCalc", "cache present?"),
        rep.metric_row("telluric_min_transmission", float(np.min(transmission)), sky_prov.source_class, "ESO SkyCalc"),
        rep.metric_row("filter_provenance", filt_prov.source_class, filt_prov.source_class, "SVO Ks"),
        rep.metric_row("sky_scale_error_max", float(max(scale_errors)), "engineering_estimate", "assumed residual"),
        rep.metric_row("residual_sky_fraction_max", float(np.max(resid_fracs)), "internal_validation", "leftover sky"),
        rep.metric_row("line_flux_bias_max_frac", float(np.max(np.abs(lf_bias))), "internal_validation", "science impact"),
        rep.metric_row("velocity_bias_max_kms", float(np.max(np.abs(v_bias))), "internal_validation", "science impact"),
        rep.metric_row("snr_loss_max_frac", float(np.max(snr_loss)), "internal_validation", "science impact"),
    ]
    csv = rep.write_metrics_csv(outdir / "sky_telluric_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "sky_telluric_summary.json", {
        "sky_provenance": sky_prov.source_class, "filter_provenance": filt_prov.source_class,
        "scale_errors": scale_errors, "residual_sky_fraction": resid_fracs,
        "line_flux_bias_frac": lf_bias, "velocity_bias_kms": v_bias, "snr_loss_frac": snr_loss,
        "limitations": [
            "Controlled inject-then-remove experiment, not a real reduction pipeline.",
            "SkyCalc radiance->electrons scaling is an engineering_estimate.",
            "Velocity in km/s uses c = %.3f km/s." % SPEED_OF_LIGHT_KM_S,
        ],
    })

    # Provenance classes that actually feed each artifact: the SkyCalc sky +
    # telluric and the SVO filter (public or proxy), the engineering_estimate sky
    # scaling, the synthetic source line, and the internal_validation biases.
    sky_class = sky_prov.source_class
    filt_class = filt_prov.source_class
    data_classes = sorted({sky_class, filt_class})
    full = ["internal_validation", "engineering_estimate", "synthetic_proxy", *data_classes]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, sky_class, [*data_classes, "engineering_estimate"], "SkyCalc + SVO cache", "Sky/telluric inputs."),
        rep.manifest_entry(p2.name, "figure", GEN, "internal_validation", ["internal_validation", "engineering_estimate", *data_classes], "controlled sky subtraction", "Subtraction residuals."),
        rep.manifest_entry(p3.name, "figure", GEN, "internal_validation", ["internal_validation", "synthetic_proxy", *data_classes], "controlled telluric correction", "Telluric correction effect."),
        rep.manifest_entry(p4.name, "figure", GEN, "internal_validation", full, "controlled residual scan", "Science impact (sky+telluric+filter+source)."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", full, "sky/telluric experiment", "Residual + science metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", full, "sky/telluric experiment", "Residuals + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"inputs": p1, "subtraction": p2, "telluric": p3, "impact": p4, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote sky/telluric artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
