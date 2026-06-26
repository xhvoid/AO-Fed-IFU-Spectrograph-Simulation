#!/usr/bin/env python3
"""Report which inputs are public data vs synthetic proxies.

This script does not run the full pipeline.  Instead it exercises the optional
public-data loaders and prints, for each, whether a local cache was found
(``direct_public_data``) or a documented synthetic proxy was used
(``synthetic_proxy``).  It makes the project's "public anchors vs synthetic
internals" boundary explicit and reproducible.

Drop cached files into ``data/public_cache/`` (see that folder's README) to flip
entries from synthetic to public.

Run:
    python examples/run_public_anchor_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ao_fed_ifu.data_sources import (  # noqa: E402
    ProvenanceLog,
    load_filter_curve,
    load_sky_spectrum,
    load_source_spectrum,
)


def main() -> None:
    wl = np.linspace(1.95, 2.45, 200)  # a K-band-ish probe grid
    log = ProvenanceLog()

    for band in ("J", "H", "K"):
        _, prov = load_filter_curve(band)
        log.records.append(prov)
    _, sky_prov = load_sky_spectrum(wl)
    log.records.append(sky_prov)
    _, src_prov = load_source_spectrum(wl)
    log.records.append(src_prov)

    print("=" * 78)
    print(" Public-data anchors vs synthetic proxies")
    print("=" * 78)
    print(f" {'component':22s} {'provenance':20s} reference / note")
    print("-" * 78)
    for r in log.records:
        print(f" {r.name:22s} {r.source_class:20s} {r.url_or_reference}")
    print("-" * 78)
    print(" class counts:", log.summary_counts())
    print("=" * 78)
    print(" Rows marked 'direct_public_data' read real cached samples under")
    print(" data/public_cache/ (SVO / ESO SkyCalc / CALSPEC). To refresh them:")
    print("   pip install -e .[data] && python examples/fetch_public_data.py")
    print(" Remove the cached files and these rows fall back to 'synthetic_proxy'.")


if __name__ == "__main__":
    main()
