#!/usr/bin/env python3
"""One-page comparison of the major pipeline modes.

Runs toy / instrument_inspired / public_conditioned / highz_galaxy with small
configs and compares SNR maps, extracted spectra and key metrics side by side.

Run:
    python examples/run_mode_comparison_dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu import reporting as rep  # noqa: E402

rep.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ao_fed_ifu.integration import (  # noqa: E402
    make_highz_galaxy_config, make_instrument_inspired_config,
    make_public_conditioned_config, make_toy_config, run_pipeline,
)

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures" / "generated"
GEN = "examples/run_mode_comparison_dashboard.py"

# Metric -> provenance class for the comparison table.
_PROV = {
    "detector_rows": "synthetic_proxy", "detector_cols": "synthetic_proxy",
    "runtime_s": "internal_validation", "median_snr": "synthetic_proxy",
    "recovered_R_bright_spaxel": "synthetic_proxy", "psf_ee1_spaxel": "synthetic_proxy",
    "psf_ee3x3_spaxel": "synthetic_proxy", "sky_to_source_ratio": "engineering_estimate",
    "filter_weighted_throughput": "engineering_estimate", "public_data_components_used": "internal_validation",
}


def _small(cfg):
    cfg.nx = cfg.ny = 10
    cfg.resolving_power_R = min(cfg.resolving_power_R, 2000.0)
    return cfg


def generate(outdir: Path, quick: bool = False, cache_dir=None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)  # create --out before any savefig
    builders = {
        "toy": make_toy_config(),
        "instrument_inspired": make_instrument_inspired_config(),
        "public_conditioned": make_public_conditioned_config(public_cache_dir=cache_dir),
        "highz_galaxy": make_highz_galaxy_config(public_cache_dir=cache_dir),
    }
    results, records = {}, []
    for name, cfg in builders.items():
        _small(cfg)
        t0 = time.perf_counter()
        res = run_pipeline(cfg)
        dt = time.perf_counter() - t0
        results[name] = res
        m = res.metrics
        row = {
            "detector_rows": m["detector_shape"][0], "detector_cols": m["detector_shape"][1],
            "runtime_s": round(dt, 3), "median_snr": m["median_snr"],
            "recovered_R_bright_spaxel": m["recovered_R_bright_spaxel"],
            "psf_ee1_spaxel": m.get("psf_ee1_spaxel", float("nan")),
            "psf_ee3x3_spaxel": m.get("psf_ee3x3_spaxel", float("nan")),
            "sky_to_source_ratio": m["sky_to_source_ratio"],
            "filter_weighted_throughput": m["filter_weighted_throughput"],
            "public_data_components_used": m["public_data_components_used"],
        }
        for metric, value in row.items():
            records.append({"mode": name, "metric": metric, "value": value,
                            "source_class": _PROV[metric], "notes": ""})
        counts = res.provenance.summary_counts()
        for cls, cnt in counts.items():
            records.append({"mode": name, "metric": f"provenance_count.{cls}", "value": cnt,
                            "source_class": "internal_validation", "notes": "provenance class count"})

    # --- Dashboard figure: rows = modes; cols = SNR map, spectrum, metrics text. ---
    modes = list(results)
    fig, axes = plt.subplots(len(modes), 3, figsize=(13, 3.0 * len(modes)))
    for i, name in enumerate(modes):
        res = results[name]
        im = axes[i, 0].imshow(res.snr_map, origin="lower", cmap="cividis")
        axes[i, 0].set_title(f"{name}: SNR map", fontsize=9); fig.colorbar(im, ax=axes[i, 0])
        axes[i, 1].plot(res.wavelength_grid_um, res.extracted_spectrum, lw=0.8)
        axes[i, 1].axvline(res.line_center_um, color="r", ls="--", lw=0.7)
        axes[i, 1].set_title(f"{name}: extracted spectrum", fontsize=9); axes[i, 1].set_xlabel("um")
        m = results[name].metrics
        txt = (f"det {m['detector_shape'][0]}x{m['detector_shape'][1]}\n"
               f"median SNR {m['median_snr']:.1f}\n"
               f"recovered R {m['recovered_R_bright_spaxel']:.0f}\n"
               f"EE1 {m.get('psf_ee1_spaxel', float('nan')):.3f} / EE3x3 {m.get('psf_ee3x3_spaxel', float('nan')):.3f}\n"
               f"sky/source {m['sky_to_source_ratio']:.2f}\n"
               f"filter thr {m['filter_weighted_throughput']:.2f}\n"
               f"public comps {m['public_data_components_used']}")
        axes[i, 2].axis("off")
        axes[i, 2].text(0.02, 0.95, txt, va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("Mode comparison dashboard (small configs; learning/portfolio demonstrator)", fontsize=13)
    fig.tight_layout()
    p1 = outdir / "mode_comparison_dashboard.png"
    fig.savefig(p1, dpi=110); plt.close(fig)

    # --- Tables (mode,metric,value,source_class,notes). ---
    for r in records:
        rep.validate_class(r["source_class"])
    csv = outdir / "mode_comparison_metrics.csv"
    pd.DataFrame(records, columns=["mode", "metric", "value", "source_class", "notes"]).to_csv(csv, index=False)
    summ = rep.write_summary_json(outdir / "mode_comparison_summary.json", {
        "modes": modes,
        "provenance_counts": {name: results[name].provenance.summary_counts() for name in modes},
        "limitations": ["Small configs for a fast one-page comparison; not performance predictions."],
    })

    # Provenance classes actually represented by the compared runs (public_conditioned
    # and highz add direct_public_data / literature_derived), plus the metric labels.
    run_classes = sorted({cls for name in modes for cls, n in results[name].provenance.summary_counts().items() if n})
    metric_classes = sorted({r["source_class"] for r in records})
    full = sorted(set(run_classes) | set(metric_classes))
    entries = [
        rep.manifest_entry(p1.name, "figure", GEN, "synthetic_proxy", full,
                           "all four modes", "Mode comparison dashboard."),
        rep.manifest_entry(csv.name, "table", GEN, "internal_validation", full,
                           "all four modes", "mode,metric,value,source_class,notes."),
        rep.manifest_entry(summ.name, "table", GEN, "internal_validation", full,
                           "all four modes", "Modes + provenance counts."),
    ]
    rep.update_manifest(outdir, entries)
    return {"dashboard": p1, "metrics_csv": csv, "summary_json": summ}


def main() -> int:
    paths = generate(OUTDIR)
    print("Wrote mode-comparison artifacts:")
    for k, p in paths.items():
        print(f"  {k:14s} -> {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
