#!/usr/bin/env python3
"""Optional interactive Streamlit dashboard (portfolio demo interface).

Launch with:
    pip install -e .[dashboard]
    streamlit run examples/run_interactive_dashboard.py

It lets you pick a mode and move sliders (exposure, Strehl, R, spaxel scale,
sky scaling) and shows the line-flux / velocity / SNR maps, the extracted
spectrum, the provenance counts and key metrics.  Learning/portfolio
demonstrator -- **not** a calibrated simulator.

Importing this module never requires Streamlit; the formatting helpers below are
plain functions, and ``main()`` prints an install hint if Streamlit is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DASHBOARD_NOTE = (
    "Learning/portfolio demonstrator -- NOT a calibrated observatory simulator, "
    "NOT an ETC, NOT a reproduction of any real instrument."
)


def format_metric(name: str, value) -> str:
    """Format a metric value for display (compact, type-aware)."""
    if isinstance(value, bool):
        return f"{name}: {'yes' if value else 'no'}"
    if isinstance(value, float):
        if value == 0 or 1e-3 <= abs(value) < 1e5:
            return f"{name}: {value:.3f}"
        return f"{name}: {value:.3e}"
    return f"{name}: {value}"


def provenance_summary(counts: dict) -> str:
    """Render provenance class counts as a compact one-line summary."""
    parts = [f"{cls}={n}" for cls, n in counts.items() if n]
    return " | ".join(parts) if parts else "(none)"


def _build_config(mode: str, exposure_s: float, strehl: float, R: float,
                  spaxel_scale_mas: float, sky_scaling: float):
    """Build a small PipelineConfig for the chosen mode + slider values."""
    from ao_fed_ifu.integration import (
        make_highz_galaxy_config, make_public_conditioned_config, make_toy_config,
    )

    if mode == "highz":
        cfg = make_highz_galaxy_config()
    elif mode == "public_conditioned":
        cfg = make_public_conditioned_config()
    else:
        cfg = make_toy_config()
    cfg.nx = cfg.ny = 12
    cfg.exposure_s = exposure_s
    cfg.strehl = strehl
    cfg.resolving_power_R = R
    cfg.spaxel_scale_mas = spaxel_scale_mas
    cfg.sky_scaling = sky_scaling
    return cfg


def render(st) -> None:  # pragma: no cover - requires a Streamlit runtime
    """Render the dashboard using a Streamlit module object."""
    import numpy as np

    from ao_fed_ifu.integration import run_pipeline

    st.title("AO-fed IFU spectrograph -- interactive demonstrator")
    st.warning(DASHBOARD_NOTE)

    mode = st.sidebar.selectbox("Mode", ["toy", "public_conditioned", "highz"])
    exposure_s = st.sidebar.slider("Exposure [s]", 60.0, 3600.0, 600.0, 60.0)
    strehl = st.sidebar.slider("Strehl", 0.05, 0.9, 0.3, 0.05)
    R = st.sidebar.slider("Resolving power R", 1000.0, 6000.0, 2500.0, 250.0)
    spaxel = st.sidebar.slider("Spaxel scale [mas]", 5.0, 50.0, 10.0, 1.0)
    sky_scaling = st.sidebar.slider("Sky scaling (engineering_estimate)", 0.0, 3.0, 1.0, 0.1)

    cfg = _build_config(mode, exposure_s, strehl, R, spaxel, sky_scaling)
    result = run_pipeline(cfg)
    m = result.metrics

    c1, c2, c3 = st.columns(3)
    c1.metric("median SNR", f"{m['median_snr']:.1f}")
    c2.metric("recovered R", f"{m['recovered_R_bright_spaxel']:.0f}")
    c3.metric("sky/source", f"{m['sky_to_source_ratio']:.2f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].imshow(result.line_flux_map, origin="lower", cmap="viridis"); ax[0].set_title("line flux")
    vmax = np.nanmax(np.abs(result.velocity_map)) or 1.0
    ax[1].imshow(result.velocity_map, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax); ax[1].set_title("velocity")
    ax[2].imshow(result.snr_map, origin="lower", cmap="cividis"); ax[2].set_title("SNR")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(result.wavelength_grid_um, result.extracted_spectrum)
    ax2.axvline(result.line_center_um, color="r", ls="--", lw=0.8)
    ax2.set_xlabel("wavelength [um]"); ax2.set_ylabel("flux [e-]")
    st.pyplot(fig2)

    st.text("Provenance: " + provenance_summary(result.provenance.summary_counts()))
    st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


def main() -> int:
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Install the optional extra and launch with:")
        print("  pip install -e .[dashboard]")
        print("  streamlit run examples/run_interactive_dashboard.py")
        return 0
    render(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
