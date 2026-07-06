#!/usr/bin/env python3
"""High-z galaxy source report: morphology, kinematics, multi-line, recovery.

Builds a configurable synthetic high-z galaxy (Sersic light, inclined rotating
disc, velocity dispersion, redshifted H-alpha + [N II] triplet, optional dust),
forward-models it through a small IFU + detector path, reconstructs it, and
compares the recovered kinematics / line flux to the truth.

A synthetic source-recovery experiment, **not** measured galaxy physics.

Run:
    python examples/run_galaxy_source_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import cube_reconstruction as recon  # noqa: E402
from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu import galaxy_kinematics as gk  # noqa: E402
from ao_fed_ifu.detector_model import DetectorConfig, expose  # noqa: E402
from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube  # noqa: E402
from ao_fed_ifu.source_cube import REST_FRAME_LINES_UM, observed_wavelength_um, scale_cube_to_total_electrons  # noqa: E402
from ao_fed_ifu.spectrograph_model import SpectrographConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_galaxy_source_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    redshift = 2.30
    n = 14 if quick else 20
    cfg = gk.GalaxySourceConfig(nx=n, ny=n, sersic_index=1.0, half_light_radius_spaxel=4.0,
                                inclination_deg=60.0, rotation_amplitude_kms=180.0,
                                velocity_dispersion_kms=60.0, dust_attenuation_av_mag=0.5,
                                dust_wavelength_law=True)
    lines = ["Halpha", "NII6548", "NII6583"]
    obs_ha = observed_wavelength_um(REST_FRAME_LINES_UM["Halpha"], redshift)

    sp = SpectrographConfig(lambda_min_um=obs_ha - 0.12, lambda_max_um=obs_ha + 0.12,
                            resolving_power_R=4000.0 if quick else 6000.0)
    grid = sp.wavelength_grid_um()
    truth_cube = gk.build_galaxy_source_cube(cfg, grid, redshift, lines=lines)
    truth_cube = scale_cube_to_total_electrons(truth_cube, 4.0e6)
    in_band = gk.lines_in_band(lines, grid, redshift)

    # Velocity window: wide enough to hold the full rotation + dispersion of the
    # H-alpha line, but narrower than the offset to the nearest [N II] line so it
    # stays isolated. [N II] 6548/6583 sit ~0.0049/0.0068 um from H-alpha here.
    win = 0.0025
    truth_v = gk.projected_velocity_field(cfg)
    truth_lf = diag.line_flux_map(truth_cube, grid, obs_ha, win)
    sersic = gk.sersic_light_profile(cfg)

    # Forward-model: IFU dispersion + LSF + detector noise -> reconstruct.
    ifu = IFUConfig(nx_spaxel=cfg.nx, ny_spaxel=cfg.ny, cross_talk_sigma_pix=0.5)
    mapping = build_slicer_mapping(ifu, sp, grid)
    frame = forward_project_cube(truth_cube, mapping, apply_lsf=True, apply_cross_talk=True)
    readout = expose(frame, DetectorConfig(read_noise_e=8.0, dark_current_e_s_pix=0.005), 900.0, rng=0)
    rec_cube = recon.reconstruct_cube_nearest(readout.image_e, mapping)
    var_cube = recon.estimate_variance_cube(readout.variance_e2, mapping)
    rec_v = diag.velocity_centroid_map(rec_cube, grid, obs_ha, win)
    rec_lf = diag.line_flux_map(rec_cube, grid, obs_ha, win)
    rec_snr = diag.snr_map(rec_cube, var_cube, grid, obs_ha, win)
    ap = sersic > 0.2

    # --- Figure 1: truth maps. ---
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    _img(fig, ax[0, 0], sersic, "Sersic light (truth)")
    _img(fig, ax[0, 1], truth_v, "rotation velocity [km/s] (truth)", symmetric=True, cmap="coolwarm")
    _img(fig, ax[1, 0], truth_lf, "H-alpha line flux (truth) [e-]")
    _img(fig, ax[1, 1], np.full_like(truth_v, cfg.velocity_dispersion_kms), "velocity dispersion [km/s]", cmap="magma")
    fig.suptitle(f"High-z galaxy truth (z={redshift}, i={cfg.inclination_deg:.0f}deg, Vrot={cfg.rotation_amplitude_kms:.0f} km/s)")
    p1 = _save(fig, outdir / "galaxy_source_truth_maps.png")

    # --- Figure 2: integrated multi-line spectrum. ---
    spec = rec_cube[ap, :].sum(axis=0)
    fig, axx = plt.subplots(figsize=(9, 4.2))
    axx.plot(grid, spec, lw=1.0)
    for ln in in_band:
        lo = observed_wavelength_um(REST_FRAME_LINES_UM[ln], redshift)
        axx.axvline(lo, color="r", ls="--", lw=0.7)
        axx.annotate(ln, (lo, np.nanmax(spec)), fontsize=7, rotation=90, va="top")
    axx.set_xlabel("wavelength [um]"); axx.set_ylabel("flux [e-]")
    axx.set_title(f"Recovered multi-line spectrum (H-alpha + [N II], z={redshift})")
    p2 = _save(fig, outdir / "galaxy_multiline_spectrum.png")

    # --- Figure 3: recovery vs truth. ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    good = np.isfinite(truth_v[ap]) & np.isfinite(rec_v[ap])
    ax[0].scatter(truth_v[ap][good], rec_v[ap][good], s=8, alpha=0.6)
    lim = [np.nanmin(truth_v[ap]), np.nanmax(truth_v[ap])]
    ax[0].plot(lim, lim, "k--", lw=0.7); ax[0].set_xlabel("truth velocity"); ax[0].set_ylabel("recovered"); ax[0].set_title("velocity")
    ax[1].scatter(truth_lf[ap], rec_lf[ap], s=8, alpha=0.6, color="C1")
    ax[1].set_xlabel("truth line flux"); ax[1].set_ylabel("recovered"); ax[1].set_title("line flux")
    _img(fig, ax[2], rec_v - truth_v, "velocity residual [km/s]", symmetric=True, cmap="coolwarm")
    fig.suptitle("High-z galaxy recovery vs truth")
    p3 = _save(fig, outdir / "galaxy_recovery_vs_truth.png")

    # --- Metrics. ---
    eps = 1e-30
    vspan_truth = float(np.nanpercentile(truth_v[ap], 95) - np.nanpercentile(truth_v[ap], 5))
    vspan_rec = float(np.nanpercentile(rec_v[ap], 95) - np.nanpercentile(rec_v[ap], 5))
    v_bias = float(np.nanmedian((rec_v - truth_v)[ap]))
    lf_bias = float(np.nanmedian((rec_lf[ap] - truth_lf[ap]) / (truth_lf[ap] + eps)))
    rows = [
        rep.metric_row("redshift", redshift, "engineering_estimate", "scenario"),
        rep.metric_row("lines_in_band", ";".join(in_band), "literature_derived", "rest wavelengths"),
        rep.metric_row("inclination_deg", cfg.inclination_deg, "synthetic_proxy", "morphology"),
        rep.metric_row("rotation_amplitude_kms", cfg.rotation_amplitude_kms, "engineering_estimate", "kinematics"),
        rep.metric_row("velocity_dispersion_kms", cfg.velocity_dispersion_kms, "engineering_estimate", "kinematics"),
        rep.metric_row("sersic_index", cfg.sersic_index, "synthetic_proxy", "morphology"),
        rep.metric_row("velocity_span_truth_kms", vspan_truth, "synthetic_proxy", "truth"),
        rep.metric_row("velocity_span_recovered_kms", vspan_rec, "synthetic_proxy", "recovered"),
        rep.metric_row("velocity_recovery_bias_kms", v_bias, "internal_validation", "recovered vs truth"),
        rep.metric_row("line_flux_recovery_bias_frac", lf_bias, "internal_validation", "recovered vs truth"),
        rep.metric_row("median_snr", float(np.nanmedian(rec_snr[ap])), "synthetic_proxy", "recovered"),
        rep.metric_row("dust_av_mag", cfg.dust_attenuation_av_mag, "engineering_estimate", "dust"),
    ]
    csv = rep.write_metrics_csv(outdir / "galaxy_source_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "galaxy_source_summary.json", {
        "redshift": redshift, "lines_requested": lines, "lines_in_band": in_band,
        "observed_halpha_um": obs_ha,
        "config": {"inclination_deg": cfg.inclination_deg, "rotation_amplitude_kms": cfg.rotation_amplitude_kms,
                   "velocity_dispersion_kms": cfg.velocity_dispersion_kms, "sersic_index": cfg.sersic_index,
                   "dust_av_mag": cfg.dust_attenuation_av_mag, "dust_wavelength_law": cfg.dust_wavelength_law},
        "velocity_span_truth_kms": vspan_truth, "velocity_span_recovered_kms": vspan_rec,
        "limitations": [
            "Synthetic source-recovery experiment, NOT measured galaxy physics.",
            "Rest wavelengths are literature_derived; line ratios and kinematics are "
            "engineering estimates; morphology/dust are synthetic proxies.",
        ],
    })

    classes = ["literature_derived", "engineering_estimate", "synthetic_proxy", "internal_validation"]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "synthetic_proxy", classes, "galaxy truth", "Sersic + rotation + dispersion."),
        rep.manifest_entry(p2.name, "figure", GEN, "literature_derived", classes, "multi-line", "Recovered Halpha+[NII] spectrum."),
        rep.manifest_entry(p3.name, "figure", GEN, "internal_validation", classes, "recovery", "Recovery vs truth."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes, "galaxy source", "Kinematics + recovery metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes, "galaxy source", "Config + lines + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"truth_maps": p1, "multiline_spectrum": p2, "recovery": p3, "metrics_csv": csv, "summary_json": summ}


def _img(fig, ax, data, title, cmap="viridis", symmetric=False):
    kw = {"origin": "lower", "cmap": cmap}
    if symmetric:
        vmax = np.nanmax(np.abs(data)) or 1.0
        kw.update(vmin=-vmax, vmax=vmax)
    im = ax.imshow(data, **kw); ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote galaxy-source artifacts:")
    for k, p in paths.items():
        print(f"  {k:18s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
