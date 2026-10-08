#!/usr/bin/env python3
"""Wavelength-dependent AO PSF report: Strehl(lambda), core size, EE & sampling.

Sweeps the AO PSF proxy across wavelength and writes PSF thumbnails plus
encircled-energy / sampling diagnostics vs wavelength.  The Strehl rises toward
the red (fixed residual wavefront error), while the diffraction core grows as
lambda/D -- so the encircled energy in a fixed spaxel trades off between the two.

This is a wavelength-dependent AO *proxy* (synthetic_proxy / engineering_estimate
via the Marechal Strehl law), **not** an AO telemetry or phase-screen simulator.

Run:
    python examples/run_ao_psf_wavelength_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ao_fed_ifu.ao_psf_physics import AOPSFConfig, ao_psf_wavelength_sweep  # noqa: E402
from ao_fed_ifu.psf_coupling import ao_corrected_psf_proxy  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
GEN = "examples/run_ao_psf_wavelength_report.py"


def generate(outdir: Path, quick: bool = False) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    cfg = AOPSFConfig(telescope_diameter_m=8.0, wavefront_error_nm=150.0,
                      seeing_fwhm_mas=600.0, spaxel_scale_mas=10.0,
                      # An odd number of fine pixels tiles each centred spaxel,
                      # keeping fine-image EE identical to direct spaxel integrals.
                      fine_scale_mas=10.0 / 3.0 if quick else 2.0,
                      n_pix=121 if quick else 161)
    wavelengths = np.linspace(1.0, 2.4, 6 if quick else 12)
    rows = ao_psf_wavelength_sweep(wavelengths, cfg)
    wl = np.array([r["wavelength_um"] for r in rows])

    # --- Figure: PSF thumbnails (top) + diagnostics vs wavelength (bottom). ---
    thumb_wl = [wl[0], wl[len(wl) // 2], wl[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for a, w in zip(axes[0], thumb_wl):
        p = ao_corrected_psf_proxy(cfg.strehl_at(w), w, cfg.telescope_diameter_m,
                                   cfg.seeing_fwhm_mas, cfg.fine_scale_mas, cfg.n_pix)
        a.imshow(np.log10(p + p.max() * 1e-4), origin="lower", cmap="magma")
        a.set_title(f"lambda={w:.2f} um, Strehl={cfg.strehl_at(w):.2f}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])

    def _get(key):
        return np.array([r[key] for r in rows], dtype=float)

    ax = axes[1, 0]
    ax.plot(wl, _get("strehl"), "o-", color="C0", label="Strehl")
    ax2 = ax.twinx(); ax2.plot(wl, _get("diffraction_fwhm_mas"), "s--", color="C1", label="diffraction FWHM")
    ax.set_xlabel("wavelength [um]"); ax.set_ylabel("Strehl", color="C0"); ax2.set_ylabel("diffraction FWHM [mas]", color="C1")
    ax.set_title("Strehl & core size vs lambda")

    ax = axes[1, 1]
    ax.plot(wl, _get("ee1_spaxel"), "o-", label="EE1 (central spaxel)")
    ax.plot(wl, _get("ee3x3_spaxel"), "s-", label="EE3x3")
    ax.set_xlabel("wavelength [um]"); ax.set_ylabel("encircled energy (of total PSF)")
    ax.set_title("Encircled energy vs lambda"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(wl, _get("ee_field"), "o-", color="C2", label="field throughput (7x7)")
    ax.plot(wl, _get("psf_fwhm_mas") / (2 * cfg.spaxel_scale_mas), "d-", color="C3",
            label="FWHM / (2*spaxel)  (>1 = sampled)")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("wavelength [um]"); ax.set_title("Field throughput & sampling")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"Wavelength-dependent AO PSF proxy (D={cfg.telescope_diameter_m} m, "
                 f"WFE={cfg.wavefront_error_nm} nm) -- synthetic proxy, not an AO simulator", fontsize=12)
    p1 = _save(fig, outdir / "ao_psf_wavelength_grid.png")

    # --- CSV: one row per wavelength, with a per-row source_class. ---
    table = [{**r, "undersampled": int(r["undersampled"]), "source_class": "synthetic_proxy"} for r in rows]
    cols = ["wavelength_um", "strehl", "diffraction_fwhm_mas", "psf_fwhm_mas",
            "ee1_spaxel", "ee3x3_spaxel", "ee_field", "undersampled", "source_class"]
    csv = outdir / "ao_psf_ee_vs_wavelength.csv"
    pd.DataFrame(table, columns=cols).to_csv(csv, index=False)

    # --- JSON summary. ---
    summ = rep.write_summary_json(outdir / "ao_psf_sampling_summary.json", {
        "config": {"telescope_diameter_m": cfg.telescope_diameter_m, "wavefront_error_nm": cfg.wavefront_error_nm,
                   "seeing_fwhm_mas": cfg.seeing_fwhm_mas, "spaxel_scale_mas": cfg.spaxel_scale_mas},
        "wavelength_um": wl.tolist(),
        "strehl_range": [float(_get("strehl").min()), float(_get("strehl").max())],
        "diffraction_fwhm_mas_range": [float(_get("diffraction_fwhm_mas").min()), float(_get("diffraction_fwhm_mas").max())],
        "ee1_spaxel_range": [float(_get("ee1_spaxel").min()), float(_get("ee1_spaxel").max())],
        "any_undersampled": bool(any(r["undersampled"] for r in rows)),
        "provenance_counts": {"synthetic_proxy": len(rows)},
        "limitations": [
            "Wavelength-dependent AO PROXY (Marechal Strehl law), not AO telemetry or a phase screen.",
            "Strehl-vs-wavelength is an engineering estimate; core+halo shape is a synthetic proxy.",
        ],
    })

    classes = ["synthetic_proxy", "engineering_estimate"]
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "synthetic_proxy", classes, "AO PSF proxy sweep", "PSF thumbnails + EE/sampling vs wavelength."),
        rep.manifest_entry(csv.name, "table", GEN, "synthetic_proxy", classes, "AO PSF proxy sweep", "Per-wavelength Strehl/FWHM/EE/sampling."),
        rep.manifest_entry(summ.name, "table", GEN, "synthetic_proxy", classes, "AO PSF proxy sweep", "Ranges + limitations."),
    ]
    rep.update_manifest(outdir, entries)
    return {"wavelength_grid": p1, "ee_csv": csv, "summary_json": summ}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); return path


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote AO PSF wavelength artifacts:")
    for k, p in paths.items():
        print(f"  {k:16s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
