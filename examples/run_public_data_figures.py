#!/usr/bin/env python3
"""Public-data demonstration figures and tables (offline).

Reads the cached public data under ``data/public_cache/`` (or the documented
synthetic fallback when it is absent) and writes, under ``figures/generated/``:

    public_inputs_overview.png          SVO J/H/K filters, SkyCalc trans+radiance,
                                        CALSPEC source spectrum
    public_vs_synthetic_comparison.png  same pipeline with synthetic fallback vs
                                        public_conditioned (spectrum / SNR / line flux)
    psf_sampling_strehl_grid.png        Strehl x spaxel-scale -> EE1, EE3x3, FWHM,
                                        undersampling flag
    calibration_sensitivity_matrix.png  wavelength offset / cross-talk / read noise /
                                        flat-field -> velocity, line-flux, SNR response
    public_data_metrics.csv             metric, value, source_class, component/notes
    public_data_summary.json            provenance log + public/synthetic counts
    artifact_manifest.csv               one row per generated artifact

Everything runs offline and CPU-only; the network is only needed to (re)populate
the cache via ``examples/fetch_public_data.py``.

This is a learning/portfolio demonstrator -- NOT a calibrated instrument
simulator or ETC.  The sky background and filter weighting are physically
*motivated* but their absolute normalisation is an engineering_estimate.

Run:
    python examples/run_public_data_figures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import SPEED_OF_LIGHT_KM_S  # noqa: E402
from ao_fed_ifu import diagnostics as diag  # noqa: E402
from ao_fed_ifu import psf_coupling as psf  # noqa: E402
from ao_fed_ifu.data_sources import (  # noqa: E402
    VALID_SOURCE_CLASSES,
    ProvenanceLog,
    load_filter_curve,
    load_sky_spectrum,
    load_source_spectrum,
)
from ao_fed_ifu.integration import (  # noqa: E402
    make_public_conditioned_config,
    metric_provenance,
    run_pipeline,
)

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures" / "generated"


# ---------------------------------------------------------------------------
# Public inputs (figure a) -- load the curves and record provenance.
# ---------------------------------------------------------------------------
def load_public_inputs(cache_dir=None) -> tuple[dict, ProvenanceLog]:
    """Load SVO filters, SkyCalc sky and a CALSPEC source, with provenance."""
    log = ProvenanceLog()
    filt_grid = np.linspace(0.9, 2.5, 600)
    sky_grid = np.linspace(1.90, 2.50, 600)
    src_grid = np.linspace(0.4, 2.5, 600)

    filters = {}
    for band in ("J", "H", "K"):
        curve, prov = load_filter_curve(band, cache_dir=cache_dir)
        log.records.append(prov)
        # Interpolate onto the common display grid (0 outside the curve).
        filters[band] = np.interp(
            filt_grid, curve["wavelength_um"], curve["transmission"], left=0.0, right=0.0
        )
    sky, sky_prov = load_sky_spectrum(sky_grid, cache_dir=cache_dir)
    log.records.append(sky_prov)
    source, src_prov = load_source_spectrum(src_grid, cache_dir=cache_dir)
    log.records.append(src_prov)

    data = {
        "filt_grid": filt_grid, "filters": filters,
        "sky_grid": sky_grid, "sky": sky,
        "src_grid": src_grid, "source": source,
        "provenance": {"sky": sky_prov.source_class, "source": src_prov.source_class,
                       "filter": log.records[0].source_class},
    }
    return data, log


def fig_public_inputs_overview(outdir: Path, cache_dir=None) -> tuple[Path, list[dict]]:
    data, log = load_public_inputs(cache_dir)
    prov = data["provenance"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    for band, color in zip("JHK", ("C0", "C1", "C2")):
        ax[0, 0].plot(data["filt_grid"], data["filters"][band], color=color, label=f"2MASS {band}")
    ax[0, 0].set_title(f"SVO filter transmission ({prov['filter']})")
    ax[0, 0].set_xlabel("wavelength [um]"); ax[0, 0].set_ylabel("transmission"); ax[0, 0].legend()

    ax[0, 1].plot(data["sky_grid"], data["sky"]["transmission"], color="C3")
    ax[0, 1].set_title(f"ESO SkyCalc telluric transmission ({prov['sky']})")
    ax[0, 1].set_xlabel("wavelength [um]"); ax[0, 1].set_ylabel("transmission")

    ax[1, 0].plot(data["sky_grid"], data["sky"]["radiance"], color="C4", lw=0.8)
    ax[1, 0].set_title(f"ESO SkyCalc sky radiance ({prov['sky']})")
    ax[1, 0].set_xlabel("wavelength [um]"); ax[1, 0].set_ylabel("radiance [arb / SkyCalc units]")
    ax[1, 0].set_yscale("log")

    ax[1, 1].plot(data["src_grid"], data["source"]["flux"], color="C5")
    ax[1, 1].set_title(f"CALSPEC source spectrum ({prov['source']})")
    ax[1, 1].set_xlabel("wavelength [um]"); ax[1, 1].set_ylabel("flux [peak-normalised]")

    fig.suptitle("Public data inputs (cached, read offline)", fontsize=13)
    fig.tight_layout()
    path = outdir / "public_inputs_overview.png"
    fig.savefig(path, dpi=110); plt.close(fig)

    # Metric rows: in-band mean filter transmission etc.
    kband = data["filters"]["K"]
    rows = [
        {"metric": "filter_K_mean_transmission", "value": float(np.mean(kband[kband > 0.01])),
         "source_class": prov["filter"], "component": "SVO 2MASS Ks"},
        {"metric": "sky_transmission_min", "value": float(np.min(data["sky"]["transmission"])),
         "source_class": prov["sky"], "component": "ESO SkyCalc telluric"},
        {"metric": "sky_radiance_peak_over_median",
         "value": float(np.max(data["sky"]["radiance"]) / (np.median(data["sky"]["radiance"]) + 1e-30)),
         "source_class": prov["sky"], "component": "ESO SkyCalc radiance (OH contrast)"},
    ]
    return path, rows


# ---------------------------------------------------------------------------
# Public vs synthetic comparison (figure b).
# ---------------------------------------------------------------------------
def fig_public_vs_synthetic(outdir: Path, base_config=None) -> tuple[Path, list[dict]]:
    base_config = base_config or make_public_conditioned_config(nx=16, ny=16, resolving_power_R=2500.0)
    public = run_pipeline(base_config)
    with tempfile.TemporaryDirectory() as empty:
        synth_cfg = make_public_conditioned_config(
            nx=base_config.nx, ny=base_config.ny, resolving_power_R=base_config.resolving_power_R,
            public_cache_dir=empty,
        )
        synth = run_pipeline(synth_cfg)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    wl = public.wavelength_grid_um
    ax[0, 0].plot(wl, public.extracted_spectrum, label="public", color="C0", lw=1.0)
    ax[0, 0].plot(synth.wavelength_grid_um, synth.extracted_spectrum, label="synthetic", color="C1", lw=1.0, alpha=0.8)
    ax[0, 0].axvline(public.line_center_um, color="r", ls="--", lw=0.7)
    ax[0, 0].set_title("Extracted spectrum"); ax[0, 0].set_xlabel("wavelength [um]"); ax[0, 0].legend(fontsize=8)

    for col, (res, name) in enumerate([(public, "public"), (synth, "synthetic")], start=1):
        im = ax[0, col].imshow(res.line_flux_map, origin="lower", cmap="viridis")
        ax[0, col].set_title(f"line flux [e-] ({name})"); fig.colorbar(im, ax=ax[0, col])
        im2 = ax[1, col].imshow(res.snr_map, origin="lower", cmap="cividis")
        ax[1, col].set_title(f"SNR ({name})"); fig.colorbar(im2, ax=ax[1, col])

    # Key-metrics bar comparison.
    keys = ["filter_weighted_throughput", "sky_to_source_ratio", "median_snr",
            "recovered_R_bright_spaxel", "psf_field_throughput"]
    pub_v = [public.metrics[k] for k in keys]
    syn_v = [synth.metrics[k] for k in keys]
    # Normalise each metric to the public value for a readable grouped bar.
    norm = [p if p else 1.0 for p in pub_v]
    x = np.arange(len(keys))
    ax[1, 0].bar(x - 0.18, [p / n for p, n in zip(pub_v, norm)], 0.36, label="public", color="C0")
    ax[1, 0].bar(x + 0.18, [s / n for s, n in zip(syn_v, norm)], 0.36, label="synthetic", color="C1")
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=6)
    ax[1, 0].set_title("Key metrics (normalised to public)"); ax[1, 0].legend(fontsize=8)

    fig.suptitle("Public-conditioned vs synthetic fallback (same pipeline)", fontsize=13)
    fig.tight_layout()
    path = outdir / "public_vs_synthetic_comparison.png"
    fig.savefig(path, dpi=110); plt.close(fig)

    rows = []
    for k in keys:
        rows.append({"metric": f"public_{k}", "value": float(public.metrics[k]),
                     "source_class": metric_provenance(k), "component": "public_conditioned run"})
        rows.append({"metric": f"synthetic_{k}", "value": float(synth.metrics[k]),
                     "source_class": metric_provenance(k), "component": "synthetic fallback run"})
    return path, rows


# ---------------------------------------------------------------------------
# PSF sampling grid (figure c).
# ---------------------------------------------------------------------------
def fig_psf_sampling_grid(
    outdir: Path,
    strehls=(0.05, 0.2, 0.4, 0.6, 0.8),
    spaxel_scales_mas=(5.0, 10.0, 20.0, 40.0),
    lambda_um=2.2,
    diameter_m=8.0,
    seeing_fwhm_mas=600.0,
) -> tuple[Path, list[dict]]:
    strehls = np.asarray(strehls, dtype=float)
    scales = np.asarray(spaxel_scales_mas, dtype=float)
    ee1 = np.zeros((strehls.size, scales.size))
    ee3 = np.zeros_like(ee1)
    under = np.zeros_like(ee1)
    fine = 2.0
    for i, s in enumerate(strehls):
        p = psf.ao_corrected_psf_proxy(s, lambda_um, diameter_m, seeing_fwhm_mas, fine, n_pix=161)
        for j, sc in enumerate(scales):
            m = psf.psf_metrics(p, fine, float(sc))
            ee1[i, j], ee3[i, j], under[i, j] = m.ee1_spaxel, m.ee3x3_spaxel, float(m.undersampled)

    # FWHM is core-dominated (Strehl drives *encircled energy*, not core width),
    # so show its diffraction lambda/D scaling vs wavelength for two Strehls.
    wl_sweep = np.linspace(1.0, 2.4, 12)
    fwhm_lo = [psf.psf_fwhm_mas(
        psf.ao_corrected_psf_proxy(0.2, w, diameter_m, seeing_fwhm_mas, fine, 161), fine) for w in wl_sweep]
    fwhm_hi = [psf.psf_fwhm_mas(
        psf.ao_corrected_psf_proxy(0.7, w, diameter_m, seeing_fwhm_mas, fine, 161), fine) for w in wl_sweep]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    def _heat(a, M, title, cmap="viridis"):
        im = a.imshow(M, origin="lower", cmap=cmap, aspect="auto")
        a.set_xticks(range(scales.size)); a.set_xticklabels([f"{s:g}" for s in scales])
        a.set_yticks(range(strehls.size)); a.set_yticklabels([f"{s:g}" for s in strehls])
        a.set_xlabel("spaxel scale [mas]"); a.set_ylabel("Strehl"); a.set_title(title)
        fig.colorbar(im, ax=a)

    _heat(ax[0, 0], ee1, "EE1 (central spaxel)")
    _heat(ax[0, 1], ee3, "EE3x3")
    _heat(ax[1, 1], under, "undersampling flag (1=undersampled)", cmap="inferno")
    ax[1, 0].plot(wl_sweep, fwhm_lo, "o-", label="Strehl 0.2")
    ax[1, 0].plot(wl_sweep, fwhm_hi, "s-", label="Strehl 0.7")
    ax[1, 0].set_xlabel("wavelength [um]"); ax[1, 0].set_ylabel("PSF FWHM [mas]")
    ax[1, 0].set_title(f"PSF FWHM vs wavelength (D={diameter_m}m)"); ax[1, 0].legend(fontsize=8)
    fig.suptitle("AO PSF coupling: encircled energy & sampling", fontsize=13)
    fig.tight_layout()
    path = outdir / "psf_sampling_strehl_grid.png"
    fig.savefig(path, dpi=110); plt.close(fig)

    rows = [
        {"metric": "ee1_strehl0.6_spaxel10mas",
         "value": float(ee1[np.argmin(abs(strehls - 0.6)), np.argmin(abs(scales - 10))]),
         "source_class": "synthetic_proxy", "component": "AO PSF proxy"},
        {"metric": "ee3x3_strehl0.6_spaxel10mas",
         "value": float(ee3[np.argmin(abs(strehls - 0.6)), np.argmin(abs(scales - 10))]),
         "source_class": "synthetic_proxy", "component": "AO PSF proxy"},
        {"metric": "fwhm_strehl0.6_lambda2.2um_mas",
         "value": float(psf.psf_fwhm_mas(
             psf.ao_corrected_psf_proxy(0.6, lambda_um, diameter_m, seeing_fwhm_mas, fine, 161), fine)),
         "source_class": "synthetic_proxy", "component": "AO PSF proxy"},
    ]
    return path, rows


# ---------------------------------------------------------------------------
# Calibration sensitivity matrix (figure d).
# ---------------------------------------------------------------------------
def compute_calibration_sensitivity(base_config, n_levels=3):
    """Return {perturbation: {'x':levels, 'velocity':..,'lineflux':..,'snr':..}}."""
    from ao_fed_ifu.integration import run_pipeline as _run

    ref = _run(base_config)
    ap = ref.aperture_mask
    wl, lc = ref.wavelength_grid_um, ref.line_center_um

    def biases(pert):
        v = np.nanmedian(pert.velocity_map[ap]) - np.nanmedian(ref.velocity_map[ap])
        lf = np.nanmedian((pert.line_flux_map[ap] - ref.line_flux_map[ap])
                          / (ref.line_flux_map[ap] + 1e-30))
        sl = np.nanmedian((ref.snr_map[ap] - pert.snr_map[ap]) / (ref.snr_map[ap] + 1e-30))
        return float(v), float(lf), float(sl)

    out = {}

    # 1. Wavelength offset -> velocity bias = c * dlam / lambda (analytic, on ref cube).
    dlam_levels = np.linspace(0, 3, n_levels + 1)[1:] * base_config_delta_lambda(ref)
    vbias = [SPEED_OF_LIGHT_KM_S * d / lc for d in dlam_levels]
    out["wavelength_offset"] = {"x": dlam_levels, "xlabel": "offset [um]",
                                "velocity": vbias, "lineflux": [0.0] * len(dlam_levels),
                                "snr": [0.0] * len(dlam_levels)}

    # 2. Cross-talk sigma.
    levels = np.linspace(0.6, 1.6, n_levels)
    v, lf, sl = [], [], []
    for x in levels:
        p = _run(_clone(base_config, cross_talk_sigma_pix=float(x)))
        bb = biases(p); v.append(bb[0]); lf.append(bb[1]); sl.append(bb[2])
    out["cross_talk_sigma"] = {"x": levels, "xlabel": "cross-talk sigma [pix]",
                               "velocity": v, "lineflux": lf, "snr": sl}

    # 3. Read noise.
    levels = np.linspace(8, 40, n_levels)
    v, lf, sl = [], [], []
    for x in levels:
        p = _run(_clone(base_config, read_noise_e=float(x)))
        bb = biases(p); v.append(bb[0]); lf.append(bb[1]); sl.append(bb[2])
    out["read_noise_e"] = {"x": levels, "xlabel": "read noise [e-]",
                           "velocity": v, "lineflux": lf, "snr": sl}

    # 4. Flat-field sigma.
    levels = np.linspace(0.01, 0.08, n_levels)
    v, lf, sl = [], [], []
    for x in levels:
        p = _run(_clone(base_config, flat_field_sigma=float(x)))
        bb = biases(p); v.append(bb[0]); lf.append(bb[1]); sl.append(bb[2])
    out["flat_field_sigma"] = {"x": levels, "xlabel": "flat-field sigma",
                               "velocity": v, "lineflux": lf, "snr": sl}
    return out


def base_config_delta_lambda(result) -> float:
    """Wavelength bin width from a pipeline result."""
    wl = result.wavelength_grid_um
    return float(wl[1] - wl[0])


def _clone(cfg, **changes):
    """Shallow clone of a PipelineConfig with detector/field overrides applied."""
    import copy

    from ao_fed_ifu.detector_model import DetectorConfig

    new = copy.deepcopy(cfg)
    det_fields = {k: changes.pop(k) for k in list(changes)
                  if k in DetectorConfig.__dataclass_fields__}
    if det_fields:
        new.detector = DetectorConfig(**{**new.detector.__dict__, **det_fields})
    for k, v in changes.items():
        setattr(new, k, v)
    return new


def fig_calibration_matrix(outdir: Path, base_config=None, n_levels=3) -> tuple[Path, list[dict]]:
    from ao_fed_ifu.integration import make_toy_config

    base_config = base_config or make_toy_config(nx=10, ny=10)
    sens = compute_calibration_sensitivity(base_config, n_levels=n_levels)

    perts = list(sens)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for a, key in zip(axes.ravel(), perts):
        s = sens[key]
        a.plot(s["x"], s["velocity"], "o-", label="velocity bias [km/s]", color="C0")
        a.plot(s["x"], np.array(s["lineflux"]) * 100, "s-", label="line-flux bias [%]", color="C1")
        a.plot(s["x"], np.array(s["snr"]) * 100, "^-", label="SNR loss [%]", color="C2")
        a.set_xlabel(s["xlabel"]); a.set_title(key); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("Calibration sensitivity (internal trend checks, not an ETC)", fontsize=13)
    fig.tight_layout()
    path = outdir / "calibration_sensitivity_matrix.png"
    fig.savefig(path, dpi=110); plt.close(fig)

    rows = []
    for key, s in sens.items():
        rows.append({"metric": f"{key}_velocity_bias_max_kms", "value": float(np.max(np.abs(s["velocity"]))),
                     "source_class": "internal_validation", "component": f"perturbation={key}"})
        rows.append({"metric": f"{key}_snr_loss_max_frac", "value": float(np.max(s["snr"])),
                     "source_class": "internal_validation", "component": f"perturbation={key}"})
    return path, rows


# ---------------------------------------------------------------------------
# Tables + manifest.
# ---------------------------------------------------------------------------
def write_tables(outdir: Path, metric_rows: list[dict], cache_dir=None) -> dict:
    """Write public_data_metrics.csv and public_data_summary.json."""
    # Validate every row carries a known source_class.
    for r in metric_rows:
        r.setdefault("component", "")
        assert r["source_class"] in VALID_SOURCE_CLASSES, r
    csv_path = outdir / "public_data_metrics.csv"
    pd.DataFrame(metric_rows, columns=["metric", "value", "source_class", "component"]).to_csv(
        csv_path, index=False
    )

    # Provenance log from a public_conditioned run + public/synthetic counts.
    result = run_pipeline(make_public_conditioned_config(nx=10, ny=10, resolving_power_R=2000.0,
                                                         public_cache_dir=cache_dir))
    counts = result.provenance.summary_counts()
    components = result.conditioning_info["components"]
    summary = {
        "author": "Xu Han",
        "note": "Learning/portfolio demonstrator; sky background & filter weighting are "
                "engineering_estimate scalings, not a calibrated ETC.",
        "provenance": result.provenance.to_records(),
        "provenance_counts": counts,
        "public_data_components": components,
        "public_data_components_used": int(result.conditioning_info["public_data_components_used"]),
        "n_public": sum(1 for v in components.values() if v == "direct_public_data"),
        "n_synthetic": sum(1 for v in components.values() if v == "synthetic_proxy"),
    }
    json_path = outdir / "public_data_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"metrics_csv": csv_path, "summary_json": json_path}


def write_manifest(outdir: Path, entries: list[dict]) -> Path:
    """Write artifact_manifest.csv (one row per generated artifact)."""
    path = outdir / "artifact_manifest.csv"
    cols = ["artifact_path", "artifact_type", "generated_by", "source_class", "inputs_used", "notes"]
    pd.DataFrame(entries, columns=cols).to_csv(path, index=False)
    return path


def generate_all(outdir: Path, *, quick: bool = False, cache_dir=None) -> dict:
    """Generate every public-data figure + tables + manifest. Returns paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    gen = "examples/run_public_data_figures.py"
    metric_rows: list[dict] = []
    manifest: list[dict] = []

    # Tunables shrink for the test path.
    strehls = (0.1, 0.5) if quick else (0.05, 0.2, 0.4, 0.6, 0.8)
    scales = (10.0, 20.0) if quick else (5.0, 10.0, 20.0, 40.0)
    comp_cfg = make_public_conditioned_config(
        nx=8 if quick else 16, ny=8 if quick else 16,
        resolving_power_R=1500.0 if quick else 2500.0, public_cache_dir=cache_dir,
    )
    from ao_fed_ifu.integration import make_toy_config
    calib_cfg = make_toy_config(nx=8, ny=8, resolving_power_R=1500.0)
    n_levels = 2 if quick else 3

    pa, ra = fig_public_inputs_overview(outdir, cache_dir=cache_dir)
    metric_rows += ra
    manifest.append(dict(artifact_path=pa.name, artifact_type="figure", generated_by=gen,
                         source_class=ra[0]["source_class"], inputs_used="SVO,SkyCalc,CALSPEC cache",
                         notes="Public input curves (filters, sky, source)."))

    pb, rb = fig_public_vs_synthetic(outdir, base_config=comp_cfg)
    metric_rows += rb
    manifest.append(dict(artifact_path=pb.name, artifact_type="figure", generated_by=gen,
                         source_class="internal_validation", inputs_used="public_conditioned + synthetic fallback",
                         notes="Same pipeline, public vs synthetic inputs."))

    pc, rc = fig_psf_sampling_grid(outdir, strehls=strehls, spaxel_scales_mas=scales)
    metric_rows += rc
    manifest.append(dict(artifact_path=pc.name, artifact_type="figure", generated_by=gen,
                         source_class="synthetic_proxy", inputs_used="AO PSF proxy",
                         notes="EE1/EE3x3/FWHM/undersampling vs Strehl & spaxel scale."))

    pd_, rd = fig_calibration_matrix(outdir, base_config=calib_cfg, n_levels=n_levels)
    metric_rows += rd
    manifest.append(dict(artifact_path=pd_.name, artifact_type="figure", generated_by=gen,
                         source_class="internal_validation", inputs_used="toy pipeline perturbations",
                         notes="Velocity/line-flux/SNR response to calibration perturbations."))

    tables = write_tables(outdir, metric_rows, cache_dir=cache_dir)
    manifest.append(dict(artifact_path=tables["metrics_csv"].name, artifact_type="table", generated_by=gen,
                         source_class="internal_validation", inputs_used="all figures",
                         notes="metric,value,source_class,component."))
    manifest.append(dict(artifact_path=tables["summary_json"].name, artifact_type="table", generated_by=gen,
                         source_class="internal_validation", inputs_used="public_conditioned run",
                         notes="Provenance log + public/synthetic counts."))

    # Reference the artifacts produced by run_fast_integration too, if present.
    for name, kind, by in [("fast_detector_frame.png", "figure", "examples/run_fast_integration.py"),
                           ("fast_reconstructed_diagnostics.png", "figure", "examples/run_fast_integration.py"),
                           ("fast_metrics.csv", "table", "examples/run_fast_integration.py"),
                           ("fast_summary.json", "table", "examples/run_fast_integration.py")]:
        if (outdir / name).exists():
            manifest.append(dict(artifact_path=name, artifact_type=kind, generated_by=by,
                                 source_class="synthetic_proxy", inputs_used="toy pipeline",
                                 notes="Toy end-to-end run artifacts."))

    manifest_path = write_manifest(outdir, manifest)
    paths = {"public_inputs": pa, "comparison": pb, "psf_grid": pc, "calibration": pd_,
             **tables, "manifest": manifest_path}
    return paths


def main() -> int:
    paths = generate_all(OUTDIR)
    print("Wrote public-data artifacts:")
    for name, p in paths.items():
        print(f"  {name:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
