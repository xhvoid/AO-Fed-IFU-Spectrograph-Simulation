#!/usr/bin/env python3
"""Sky / telluric time-variability and residual-realism report (controlled).

Builds on the cached ESO SkyCalc public data to show night-to-night sky
variability (continuum + OH-line scaling + a smooth residual), telluric mismatch
from a different airmass / water-vapour column, and masking of the bright
sky-line regions.

This is a controlled residual experiment, **not** a real telluric reduction.
SkyCalc radiance/transmission stay `direct_public_data`; the variability /
mismatch scalings are `engineering_estimate`.

Run:
    python examples/run_sky_variability_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ao_fed_ifu import sky_telluric as skt  # noqa: E402
from ao_fed_ifu.data_sources import load_sky_spectrum  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_sky_variability_report.py"


def generate(outdir: Path, quick: bool = False, cache_dir=None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    wl = np.linspace(1.95, 2.45, 300 if quick else 600)
    sky, sky_prov = load_sky_spectrum(wl, cache_dir=cache_dir)
    rad, trans = sky["radiance"], np.clip(sky["transmission"], 1e-3, 1.0)
    sky_class = sky_prov.source_class
    line_center, window = 2.20, 0.004

    # --- Figure 1: sky variability examples. ---
    variants = {
        "nominal": skt.vary_sky_radiance(rad),
        "continuum x1.3": skt.vary_sky_radiance(rad, continuum_scale=1.3),
        "OH x1.8": skt.vary_sky_radiance(rad, oh_line_scale=1.8),
        "residual 10% (seed)": skt.vary_sky_radiance(rad, residual_frac=0.1, seed=1),
    }
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, v in variants.items():
        ax.plot(wl, v, lw=0.8, label=name)
    ax.set_yscale("log"); ax.set_xlabel("wavelength [um]"); ax.set_ylabel("sky radiance")
    ax.set_title(f"Sky variability examples ({sky_class})"); ax.legend(fontsize=8)
    p1 = _save(fig, outdir / "sky_variability_examples.png")

    # --- Figure 2: telluric mismatch effect. ---
    airmass_ratio = 1.3
    t_actual = skt.telluric_with_airmass(trans, airmass_ratio=airmass_ratio)
    src = np.full_like(wl, 100.0)
    corrected = (src * t_actual) / trans            # observed at higher airmass, corrected with reference
    resid = corrected - src
    low_t = trans < 0.6
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(wl, trans, label="reference"); ax[0].plot(wl, t_actual, label=f"airmass x{airmass_ratio}")
    ax[0].set_xlabel("wavelength [um]"); ax[0].set_ylabel("transmission"); ax[0].legend(fontsize=8)
    ax[0].set_title("Telluric: reference vs higher airmass")
    ax[1].plot(wl, resid, color="C3", lw=0.8)
    ax[1].fill_between(wl, resid.min(), resid.max(), where=low_t, color="C1", alpha=0.2, label="low transmission")
    ax[1].set_xlabel("wavelength [um]"); ax[1].set_ylabel("corrected-flux residual")
    ax[1].set_title("Mismatch residual (worst in low-T regions)"); ax[1].legend(fontsize=8)
    p2 = _save(fig, outdir / "telluric_mismatch_effect.png")

    # --- Figure 3: sky-line mask. ---
    threshold = 3.0
    mask = skt.sky_line_mask(rad, threshold)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(wl, rad, lw=0.7, color="C4")
    ax.fill_between(wl, rad.min() + 1e-9, rad.max(), where=mask, color="C3", alpha=0.25, label="masked sky lines")
    ax.axvspan(line_center - window, line_center + window, color="C2", alpha=0.2, label="science window")
    ax.set_yscale("log"); ax.set_xlabel("wavelength [um]"); ax.set_ylabel("sky radiance")
    mfrac = skt.masked_fraction(mask)
    swfrac = skt.science_window_masked_fraction(wl, mask, line_center, window)
    ax.set_title(f"Sky-line mask (threshold x{threshold:g}: {mfrac:.0%} masked, "
                 f"{swfrac:.0%} of the science window)"); ax.legend(fontsize=8)
    p3 = _save(fig, outdir / "sky_line_mask.png")

    # --- Metrics. ---
    cont, lines = skt.split_sky_continuum_lines(rad)
    line_px = lines > 3 * np.median(lines)
    oh_resid = float(np.mean((skt.vary_sky_radiance(rad, oh_line_scale=2.0)
                              - skt.vary_sky_radiance(rad, oh_line_scale=1.0))[line_px]))
    perfect = float(np.max(np.abs((src * skt.telluric_with_airmass(trans, 1.0)) / trans - src)))
    rows = [
        rep.metric_row("sky_provenance", sky_class, sky_class, "ESO SkyCalc", "cache present?"),
        rep.metric_row("oh_scale2_residual_near_lines", oh_resid, "engineering_estimate", "OH variability"),
        rep.metric_row("telluric_mismatch_resid_lowT", float(np.mean(np.abs(resid)[low_t])), "internal_validation", "airmass mismatch"),
        rep.metric_row("telluric_mismatch_resid_hiT", float(np.mean(np.abs(resid)[trans > 0.95])), "internal_validation", "airmass mismatch"),
        rep.metric_row("telluric_perfect_match_residual", perfect, "internal_validation", "airmass=1", "~0"),
        rep.metric_row("mask_fraction_thr3", mfrac, "internal_validation", "sky-line mask"),
        rep.metric_row("mask_fraction_thr2", skt.masked_fraction(skt.sky_line_mask(rad, 2.0)), "internal_validation", "sky-line mask"),
        rep.metric_row("science_window_masked_fraction", swfrac, "internal_validation", "science window"),
        rep.metric_row("telluric_min_transmission", float(np.min(trans)), sky_class, "ESO SkyCalc"),
    ]
    csv = rep.write_metrics_csv(outdir / "sky_variability_metrics.csv", rows)
    summ = rep.write_summary_json(outdir / "sky_variability_summary.json", {
        "sky_provenance": sky_class, "airmass_ratio": airmass_ratio, "mask_threshold": threshold,
        "mask_fraction": mfrac, "science_window_masked_fraction": swfrac,
        "telluric_perfect_match_residual": perfect,
        "limitations": [
            "Controlled residual experiment, NOT a real telluric reduction.",
            "SkyCalc radiance/transmission are public data; the variability/airmass "
            "scalings are engineering estimates.",
        ],
    })

    classes = ["internal_validation", "engineering_estimate", sky_class]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, sky_class, [sky_class, "engineering_estimate"], "SkyCalc cache", "Sky variability examples."),
        rep.manifest_entry(p2.name, "figure", GEN, "internal_validation", classes, "airmass mismatch", "Telluric mismatch effect."),
        rep.manifest_entry(p3.name, "figure", GEN, sky_class, [sky_class, "internal_validation"], "SkyCalc cache", "Sky-line mask."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", classes, "sky variability", "Variability / mismatch / mask metrics."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", classes, "sky variability", "Sky/telluric residual + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"variability": p1, "telluric_mismatch": p2, "sky_mask": p3, "metrics_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote sky-variability artifacts:")
    for k, p in paths.items():
        print(f"  {k:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
