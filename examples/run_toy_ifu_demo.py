#!/usr/bin/env python3
"""Run the toy IFU pipeline end to end and print a metrics summary.

This is the fastest way to see the whole data path work:

    source cube -> AO PSF -> image-slicer remap -> dispersion -> detector noise
    -> cube reconstruction -> diagnostics

It uses the fully-synthetic ``toy`` preset (no public data required) and is meant
to complete in well under 30 seconds on a laptop CPU.

Run:
    python examples/run_toy_ifu_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_fed_ifu.integration import make_toy_config, run_pipeline  # noqa: E402


def main() -> None:
    t0 = time.time()
    result = run_pipeline(make_toy_config())
    elapsed = time.time() - t0

    m = result.metrics
    print("=" * 64)
    print(" AO-fed IFU spectrograph -- toy pipeline demo")
    print("=" * 64)
    print(f" mode                       : {m['mode']}")
    print(f" field (spaxels)            : {result.config.nx} x {result.config.ny}")
    print(f" wavelength channels        : {m['n_wavelength']}")
    print(f" detector frame (rows x cols): {m['detector_shape'][0]} x {m['detector_shape'][1]}")
    print(f" input source electrons      : {m['input_source_electrons']:.3e}  (AB-mag photon budget)")
    print(f" PSF field throughput        : {m['psf_field_throughput']:.4f}  (post-coupling / input)")
    print("-" * 64)
    print(f" flux conservation ratio     : {m['flux_conservation_ratio']:.6f}   (target 1.0)")
    print(f" detected fraction (LSF+xt)  : {m['detected_fraction_with_lsf_crosstalk']:.4f}")
    print(f" median / max line SNR       : {m['median_snr']:.1f} / {m['max_snr']:.1f}")
    print(f" recovered R (central spaxel): {m['recovered_R_bright_spaxel']:.0f}  "
          f"(configured {m['configured_R']:.0f})")
    print(f" velocity span (p5-p95)      : {m['velocity_span_p5_p95_kms']:.1f} km/s")
    print(f" pixels per resolution elem. : {m['pixels_per_resel']:.2f}  "
          f"(well sampled: {m['well_sampled_spectrum']})")
    if "psf_ee1_spaxel" in m:
        print(f" PSF EE1 / EE3x3 / FWHM      : {m['psf_ee1_spaxel']:.3f} / "
              f"{m['psf_ee3x3_spaxel']:.3f} / {m['psf_fwhm_mas']:.1f} mas")
    print("-" * 64)
    print(f" provenance class counts     : {result.provenance.summary_counts()}")
    print(f" wall time                   : {elapsed:.2f} s")
    print("=" * 64)

    if elapsed > 30:
        print("WARNING: demo exceeded the 30 s budget.", file=sys.stderr)


if __name__ == "__main__":
    main()
