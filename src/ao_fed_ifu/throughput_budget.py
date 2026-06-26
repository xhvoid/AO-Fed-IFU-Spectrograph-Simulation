"""End-to-end throughput budget: where the photons are lost.

A real instrument's "throughput" is a product of many stages -- the atmosphere,
the telescope mirrors, the AO relay, the IFU slicer, the spectrograph optics, the
filter bandpass and the detector quantum efficiency.  This module makes that
decomposition explicit and inspectable, so the photon budget reads like an
engineering estimate instead of one opaque scalar.

Provenance is honest per component:
    * atmosphere (telluric) and the filter come from cached public data
      (ESO SkyCalc / SVO) when present -> ``direct_public_data``; otherwise the
      documented synthetic fallback -> ``synthetic_proxy``;
    * telescope / AO relay / spectrograph / detector-QE curves are smooth
      ``engineering_estimate`` proxies (plausible levels, not a measured curve);
    * the IFU slicer throughput is a ``synthetic_proxy``.

This is NOT a calibrated throughput model or an exposure-time calculator; it is a
decomposed, labelled engineering estimate for visualisation and for an optional
richer photon-budget path in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data_sources import load_filter_curve, load_sky_spectrum

# Order in which the stages are applied (atmosphere first, detector last).  Used
# for the waterfall plot and the cumulative effective-throughput cascade.
STAGE_ORDER = ("atmosphere", "telescope", "ao_relay", "ifu", "spectrograph", "filter", "qe")

# Instrument-only stages (exclude atmosphere + filter, which the public-conditioned
# pipeline already applies spectrally) -- used when threading into the pipeline.
INSTRUMENT_STAGES = ("telescope", "ao_relay", "ifu", "spectrograph", "qe")


@dataclass
class ThroughputComponent:
    """One multiplicative throughput stage (scalar or wavelength-dependent).

    ``transmission`` is either a scalar or an array matching ``wavelength_um``;
    ``on_grid`` resamples it onto an arbitrary wavelength grid.
    """

    name: str
    kind: str
    transmission: float | np.ndarray
    source_class: str
    reference: str = ""
    notes: str = ""
    wavelength_um: np.ndarray | None = None

    def on_grid(self, grid_um: np.ndarray) -> np.ndarray:
        """Transmission resampled onto ``grid_um`` and clipped to [0, 1]."""
        grid_um = np.asarray(grid_um, dtype=float)
        if self.wavelength_um is None:
            values = np.full(grid_um.shape, float(self.transmission))
        else:
            values = np.interp(grid_um, self.wavelength_um, np.asarray(self.transmission, dtype=float),
                               left=0.0, right=0.0)
        return np.clip(values, 0.0, 1.0)

    def effective(self, grid_um: np.ndarray) -> float:
        """Band-averaged effective transmission of this stage on ``grid_um``."""
        return float(np.mean(self.on_grid(grid_um)))


@dataclass
class ThroughputBudget:
    """An ordered set of :class:`ThroughputComponent` stages."""

    components: list[ThroughputComponent] = field(default_factory=list)

    def combined(self, grid_um: np.ndarray) -> np.ndarray:
        """Product of every stage's transmission on ``grid_um`` (in [0, 1])."""
        grid_um = np.asarray(grid_um, dtype=float)
        out = np.ones_like(grid_um)
        for c in self.components:
            out = out * c.on_grid(grid_um)
        return out

    def effective_throughput(self, grid_um: np.ndarray, source_weight: np.ndarray | None = None) -> float:
        """Band-averaged combined throughput (optionally source-weighted)."""
        comb = self.combined(grid_um)
        if source_weight is None:
            return float(np.mean(comb))
        w = np.clip(np.asarray(source_weight, dtype=float), 0.0, None)
        return float(np.sum(comb * w) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(comb))

    def cumulative_cascade(self, grid_um: np.ndarray) -> list[tuple[str, float]]:
        """Effective throughput *remaining* after each stage is applied in order.

        Returns ``[(stage_name, cumulative_effective_throughput), ...]`` -- a
        monotonically non-increasing cascade, the data behind the waterfall plot.
        """
        grid_um = np.asarray(grid_um, dtype=float)
        running = np.ones_like(grid_um)
        cascade = []
        for c in self.components:
            running = running * c.on_grid(grid_um)
            cascade.append((c.name, float(np.mean(running))))
        return cascade

    def component_table(self, grid_um: np.ndarray) -> list[dict]:
        """One row per component: name, kind, effective transmission, provenance."""
        return [
            {
                "component": c.name,
                "kind": c.kind,
                "value_or_effective_throughput": c.effective(grid_um),
                "source_class": c.source_class,
                "reference": c.reference,
                "notes": c.notes,
            }
            for c in self.components
        ]

    def provenance_counts(self) -> dict[str, int]:
        """Count components by provenance class."""
        counts: dict[str, int] = {}
        for c in self.components:
            counts[c.source_class] = counts.get(c.source_class, 0) + 1
        return counts


# Engineering-estimate / synthetic-proxy levels for the non-public stages.
# Plausible values, NOT measured curves.
_INSTRUMENT_DEFAULTS = {
    "telescope": (0.90, "engineering_estimate", "VLT/ELT-like reflective surfaces (HARMONI overview arXiv:2103.11215)",
                  "A few protected-metal reflections; engineering estimate, not a measured curve."),
    "ao_relay": (0.80, "engineering_estimate", "AO relay optics", "Engineering estimate for AO relay transmission."),
    "ifu": (0.70, "synthetic_proxy", "image-slicer proxy (HARMONI arXiv:1807.06313)",
            "Synthetic slicer throughput; not a real slicer measurement."),
    "spectrograph": (0.75, "engineering_estimate", "spectrograph optics + grating",
                     "Engineering estimate for collimator/camera/grating throughput."),
}


def build_default_budget(
    grid_um: np.ndarray,
    cache_dir=None,
    include=STAGE_ORDER,
) -> ThroughputBudget:
    """Build a default decomposed throughput budget on ``grid_um``.

    Public stages (atmosphere, filter) use the cached SkyCalc / SVO data when
    present (``direct_public_data``) and the documented synthetic fallback
    otherwise.  Instrument stages are smooth engineering-estimate / synthetic
    proxies.  ``include`` selects which stages to build (e.g. instrument-only).
    """
    grid_um = np.asarray(grid_um, dtype=float)
    components: list[ThroughputComponent] = []

    for stage in STAGE_ORDER:
        if stage not in include:
            continue
        if stage == "atmosphere":
            sky, prov = load_sky_spectrum(grid_um, cache_dir=cache_dir)
            components.append(ThroughputComponent(
                name="atmosphere", kind="atmosphere", transmission=np.clip(sky["transmission"], 0.0, 1.0),
                wavelength_um=grid_um, source_class=prov.source_class,
                reference="ESO SkyCalc / Cerro Paranal Sky Model",
                notes="Telluric transmission; " + ("cached public data." if prov.source_class == "direct_public_data" else "synthetic proxy.")))
        elif stage == "filter":
            filt, prov = load_filter_curve("K", cache_dir=cache_dir)
            components.append(ThroughputComponent(
                name="filter_Ks", kind="filter", transmission=filt["transmission"],
                wavelength_um=filt["wavelength_um"], source_class=prov.source_class,
                reference="SVO Filter Profile Service 2MASS/Ks",
                notes="Filter bandpass; " + ("cached public data." if prov.source_class == "direct_public_data" else "synthetic top-hat proxy.")))
        elif stage == "qe":
            # Gentle, wavelength-dependent QE proxy (declines slightly to the red).
            span = max(float(np.ptp(grid_um)), 1e-9)
            qe = 0.85 - 0.07 * (grid_um - grid_um.min()) / span
            components.append(ThroughputComponent(
                name="detector_qe", kind="qe", transmission=qe, wavelength_um=grid_um,
                source_class="engineering_estimate", reference="H2RG/H4RG-like QE (MOSFIRE arXiv:1208.0314)",
                notes="Smooth NIR QE proxy ~0.78-0.85; engineering estimate, not a measured QE curve."))
        else:
            level, cls, ref, notes = _INSTRUMENT_DEFAULTS[stage]
            components.append(ThroughputComponent(
                name=stage, kind=stage, transmission=level, source_class=cls, reference=ref, notes=notes))

    return ThroughputBudget(components)
