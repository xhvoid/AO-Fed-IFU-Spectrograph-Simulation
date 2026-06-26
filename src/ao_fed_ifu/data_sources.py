"""Provenance tracking and optional public-data loaders.

Why this module exists
----------------------
A portfolio simulator is only honest if every number can be traced to its
origin.  This module provides a tiny provenance system so that each table,
figure or metric the pipeline emits can be labelled as one of:

    direct_public_data   a value/array taken verbatim from a public dataset
                         that the user cached locally (e.g. an SVO filter curve)
    literature_derived   a parameter chosen to resemble values quoted in the
                         literature (e.g. "ELT-like 39 m aperture")
    engineering_estimate a plausible engineering number with no single public
                         source (e.g. an H2RG-like read noise of 10 e-)
    synthetic_proxy      a fully invented stand-in for an internal instrument
                         detail that is not public (e.g. our slicer geometry)
    internal_validation  a number produced by our own sanity checks

The loaders below all follow the same rule: **use a local cache if present,
otherwise fall back to a clearly-labelled synthetic proxy.**  Nothing here
touches the network, so tests and examples run fully offline.

Reference anchors (recommended cache sources; see docs/references.md):
    * SVO Filter Profile Service (https://svo2.cab.inta-csic.es/theory/fps/)
      -- filter transmission curves (e.g. ID 2MASS/2MASS.J).
    * ESO Cerro Paranal Sky Model / SkyCalc
      (https://www.eso.org/sci/software/pipelines/skytools/skymodel) -- sky
      emission radiance + telluric transmission; physics from Noll et al. 2012
      (arXiv:1205.2003); telluric/sky-line context: Molecfit (arXiv:1501.07239),
      Skycorr (arXiv:1405.3679).
    * CALSPEC (STScI) / PHOENIX Goettingen library (arXiv:1303.5632) -- stellar /
      flux-standard source spectra.

The repository bundles small curated ASCII samples of these under
``data/public_cache/``.  The loaders read a cache file if present (provenance
``direct_public_data``); if a cache file is absent they fall back to a
clearly-labelled ``synthetic_proxy``.  The loaders never touch the network, so
everything runs offline either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np

# The five provenance classes, as a typing Literal so static checkers and the
# reader both see the closed vocabulary.
SourceClass = Literal[
    "direct_public_data",
    "literature_derived",
    "engineering_estimate",
    "synthetic_proxy",
    "internal_validation",
]

VALID_SOURCE_CLASSES: tuple[str, ...] = (
    "direct_public_data",
    "literature_derived",
    "engineering_estimate",
    "synthetic_proxy",
    "internal_validation",
)


@dataclass
class ProvenanceRecord:
    """One row of provenance for a value, array, table or figure.

    Parameters
    ----------
    name
        Short identifier of *what* is being described (e.g. ``"telescope_diameter"``).
    source_class
        One of :data:`VALID_SOURCE_CLASSES`.
    url_or_reference
        Where the value came from: a URL, a paper reference, or a short note
        such as ``"synthetic top-hat fallback"``.
    retrieved_or_assumed_date
        ISO date the public data was retrieved, or the date the assumption was
        made.  Free text so historical records stay readable.
    notes
        Human-readable context: assumptions, caveats, non-overclaiming notes.
    """

    name: str
    source_class: SourceClass
    url_or_reference: str
    retrieved_or_assumed_date: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source_class not in VALID_SOURCE_CLASSES:
            raise ValueError(
                f"source_class={self.source_class!r} is not one of "
                f"{VALID_SOURCE_CLASSES}"
            )

    def as_dict(self) -> dict:
        """Return a plain ``dict`` (handy for JSON / pandas)."""
        return asdict(self)


@dataclass
class ProvenanceLog:
    """An ordered collection of :class:`ProvenanceRecord` objects.

    The pipeline threads one of these through a run so that the final JSON
    artifact carries the full audit trail of where every choice came from.
    """

    records: list[ProvenanceRecord] = field(default_factory=list)

    def add(
        self,
        name: str,
        source_class: SourceClass,
        url_or_reference: str,
        retrieved_or_assumed_date: str = "",
        notes: str = "",
    ) -> ProvenanceRecord:
        """Create, store and return a new :class:`ProvenanceRecord`."""
        record = ProvenanceRecord(
            name=name,
            source_class=source_class,
            url_or_reference=url_or_reference,
            retrieved_or_assumed_date=retrieved_or_assumed_date,
            notes=notes,
        )
        self.records.append(record)
        return record

    def to_records(self) -> list[dict]:
        """Return the log as a list of plain dicts (JSON / pandas friendly)."""
        return [r.as_dict() for r in self.records]

    def write_json(self, path: str | Path) -> Path:
        """Serialise the log to ``path`` as indented JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_records(), indent=2), encoding="utf-8")
        return path

    def summary_counts(self) -> dict[str, int]:
        """Count how many records fall in each provenance class."""
        counts = {cls: 0 for cls in VALID_SOURCE_CLASSES}
        for r in self.records:
            counts[r.source_class] += 1
        return counts


# ----------------------------------------------------------------------------
# Optional public-data loaders.
#
# Each loader returns a tuple ``(data_dict, ProvenanceRecord)`` so the caller
# always knows whether it received cached public data or a synthetic fallback.
# ----------------------------------------------------------------------------

# Default location of the small, optional local cache of public data.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "public_cache"


def load_filter_curve(
    band: str = "J",
    cache_dir: str | Path | None = None,
) -> tuple[dict, ProvenanceRecord]:
    """Load a photometric filter transmission curve.

    Looks for a cached two-column ASCII file ``<band>.dat`` (wavelength in
    micrometres, transmission 0..1) under ``cache_dir``.  If found it is read
    verbatim (``direct_public_data``); otherwise a documented top-hat profile
    is synthesised for the named near-infrared band (``synthetic_proxy``).

    Parameters
    ----------
    band
        One of ``"J"``, ``"H"``, ``"K"`` (case-insensitive).
    cache_dir
        Directory to search for cached curves; defaults to ``data/public_cache``.

    Returns
    -------
    (curve, provenance)
        ``curve`` has keys ``"wavelength_um"`` and ``"transmission"`` (arrays),
        plus ``"band"``.
    """
    band = band.upper()
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cached = cache_dir / f"{band}.dat"

    if cached.exists():
        arr = np.loadtxt(cached)
        curve = {
            "band": band,
            "wavelength_um": np.atleast_1d(arr[:, 0]),
            "transmission": np.atleast_1d(arr[:, 1]),
        }
        prov = ProvenanceRecord(
            name=f"filter_{band}",
            source_class="direct_public_data",
            url_or_reference=f"local cache {cached}",
            notes="Cached filter curve; original e.g. SVO Filter Profile Service.",
        )
        return curve, prov

    # Synthetic top-hat fallback.  Centres/widths are nominal NIR band edges
    # (micrometres) -- representative, not authoritative.
    nir_bands = {
        "J": (1.17, 1.33),
        "H": (1.49, 1.78),
        "K": (1.95, 2.40),
    }
    if band not in nir_bands:
        raise ValueError(f"Unknown band {band!r}; expected one of {list(nir_bands)}")
    lam_lo, lam_hi = nir_bands[band]
    wavelength_um = np.linspace(lam_lo - 0.05, lam_hi + 0.05, 200)
    transmission = ((wavelength_um >= lam_lo) & (wavelength_um <= lam_hi)).astype(float)
    curve = {"band": band, "wavelength_um": wavelength_um, "transmission": transmission}
    prov = ProvenanceRecord(
        name=f"filter_{band}",
        source_class="synthetic_proxy",
        url_or_reference="synthetic top-hat fallback",
        notes=(
            f"No cached SVO curve found for {band}; using top-hat between "
            f"{lam_lo}-{lam_hi} um.  Replace with SVO data for direct_public_data."
        ),
    )
    return curve, prov


def load_sky_spectrum(
    wavelength_um: np.ndarray,
    cache_dir: str | Path | None = None,
) -> tuple[dict, ProvenanceRecord]:
    """Load (or synthesise) a sky emission + telluric transmission spectrum.

    If a cached SkyCalc-style file ``skycalc.dat`` exists (columns:
    wavelength_um, radiance, transmission) it is interpolated onto
    ``wavelength_um`` (``direct_public_data``).  Otherwise a smooth continuum
    plus a comb of Gaussian OH-airglow lines is synthesised, together with a
    mild telluric absorption proxy (``synthetic_proxy``).

    Returns
    -------
    (sky, provenance)
        ``sky`` has keys ``"wavelength_um"``, ``"radiance"`` (relative
        sky-emission units) and ``"transmission"`` (0..1).
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cached = cache_dir / "skycalc.dat"

    if cached.exists():
        arr = np.loadtxt(cached)
        radiance = np.interp(wavelength_um, arr[:, 0], arr[:, 1])
        transmission = np.interp(wavelength_um, arr[:, 0], arr[:, 2])
        sky = {
            "wavelength_um": wavelength_um,
            "radiance": radiance,
            "transmission": transmission,
        }
        prov = ProvenanceRecord(
            name="sky_spectrum",
            source_class="direct_public_data",
            url_or_reference=f"local cache {cached}",
            notes="Cached SkyCalc-style sky model (radiance + telluric transmission).",
        )
        return sky, prov

    # Synthetic proxy: low continuum + OH airglow comb + shallow telluric dips.
    rng = np.random.default_rng(0)
    continuum = np.full_like(wavelength_um, 0.05)
    # Place a deterministic comb of OH-like emission lines across the range.
    n_lines = 30
    line_centres = np.linspace(wavelength_um.min(), wavelength_um.max(), n_lines + 2)[1:-1]
    line_sigma = 0.0005 * (wavelength_um.mean())  # narrow relative to the band
    radiance = continuum.copy()
    for lc in line_centres:
        amp = 0.5 + rng.uniform(0.0, 1.0)
        radiance += amp * np.exp(-0.5 * ((wavelength_um - lc) / line_sigma) ** 2)
    # Telluric proxy: a few broad shallow absorption troughs.
    transmission = np.ones_like(wavelength_um)
    span = float(np.ptp(wavelength_um))
    for tc, depth, width in [
        (wavelength_um.min() + 0.30 * span, 0.3, 0.02),
        (wavelength_um.min() + 0.65 * span, 0.2, 0.03),
    ]:
        transmission -= depth * np.exp(-0.5 * ((wavelength_um - tc) / width) ** 2)
    transmission = np.clip(transmission, 0.0, 1.0)
    sky = {
        "wavelength_um": wavelength_um,
        "radiance": radiance,
        "transmission": transmission,
    }
    prov = ProvenanceRecord(
        name="sky_spectrum",
        source_class="synthetic_proxy",
        url_or_reference="synthetic sky continuum + OH-line proxy",
        notes=(
            "No cached SkyCalc spectrum found; using synthetic continuum + OH "
            "airglow comb + shallow telluric proxy.  Replace with ESO SkyCalc "
            "output for direct_public_data."
        ),
    )
    return sky, prov


def load_source_spectrum(
    wavelength_um: np.ndarray,
    temperature_k: float = 5000.0,
    cache_dir: str | Path | None = None,
) -> tuple[dict, ProvenanceRecord]:
    """Load (or synthesise) a 1-D source spectrum.

    If a cached file ``source.dat`` exists (columns: wavelength_um, flux) it is
    interpolated onto ``wavelength_um`` (``direct_public_data``).  Otherwise a
    blackbody continuum at ``temperature_k`` is synthesised (``synthetic_proxy``).

    The returned flux is in *relative* units (peak-normalised); absolute scaling
    is handled separately by the photon-budget function in ``source_cube``.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cached = cache_dir / "source.dat"

    if cached.exists():
        arr = np.loadtxt(cached)
        flux = np.interp(wavelength_um, arr[:, 0], arr[:, 1])
        flux = flux / np.max(flux) if np.max(flux) > 0 else flux
        spec = {"wavelength_um": wavelength_um, "flux": flux}
        prov = ProvenanceRecord(
            name="source_spectrum",
            source_class="direct_public_data",
            url_or_reference=f"local cache {cached}",
            notes="Cached source spectrum, e.g. CALSPEC/PHOENIX, peak-normalised.",
        )
        return spec, prov

    flux = blackbody_relative(wavelength_um, temperature_k)
    spec = {"wavelength_um": wavelength_um, "flux": flux}
    prov = ProvenanceRecord(
        name="source_spectrum",
        source_class="synthetic_proxy",
        url_or_reference="analytic blackbody fallback",
        notes=(
            f"No cached source spectrum found; using a {temperature_k:.0f} K "
            "blackbody continuum.  Replace with CALSPEC/PHOENIX for direct_public_data."
        ),
    )
    return spec, prov


def blackbody_relative(wavelength_um: np.ndarray, temperature_k: float) -> np.ndarray:
    """Planck spectral radiance B_lambda(T), peak-normalised to 1.

    Used only as a smooth continuum shape, so absolute units do not matter --
    we return the shape normalised to a unit peak.  The formula is the standard

        B_lambda(T) ~ 1 / lambda^5 / (exp(hc / (lambda kB T)) - 1)

    Parameters
    ----------
    wavelength_um
        Wavelength grid in micrometres.
    temperature_k
        Blackbody temperature in kelvin.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    # Physical constants in SI.
    h = 6.626_070_15e-34  # J s
    c = 2.997_924_58e8  # m / s
    kB = 1.380_649e-23  # J / K
    lam_m = wavelength_um * 1e-6
    # Guard the exponential against overflow at short wavelengths / low T.
    x = h * c / (lam_m * kB * temperature_k)
    x = np.clip(x, 1e-6, 700.0)
    b = 1.0 / (lam_m**5) / (np.expm1(x))
    return b / np.max(b)
