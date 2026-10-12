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

import hashlib
import io
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np

from .spectrograph_model import bin_average_spectral_density, wavelength_bin_edges

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
    content_sha256
        SHA-256 of the exact bytes a cached-data loader parsed (empty for
        records that do not come from a file).  It identifies the input that
        was actually used, even if the file changes later.
    """

    name: str
    source_class: SourceClass
    url_or_reference: str
    retrieved_or_assumed_date: str = ""
    notes: str = ""
    content_sha256: str = ""

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
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "public_cache"
LOCAL_CACHE_PREFIX = "local cache "


def _cache_reference(path: Path) -> str:
    """Provenance reference for a cache file: repo-relative when inside the
    repository, so saved artifacts carry no machine-specific absolute path."""
    resolved = Path(path).resolve()
    try:
        label = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Outside the repository: absolute, so a relative label always means
        # repo-relative and resolve_cache_reference cannot pick another file.
        label = str(resolved)
    return LOCAL_CACHE_PREFIX + label


def resolve_cache_reference(reference: str) -> Path | None:
    """Inverse of :func:`_cache_reference`; ``None`` if not a local-cache reference."""
    if not reference.startswith(LOCAL_CACHE_PREFIX):
        return None
    path = Path(reference[len(LOCAL_CACHE_PREFIX):])
    return path if path.is_absolute() else REPO_ROOT / path

# Numerical slack (um) for the cache-coverage check below: exact-boundary
# requests (grid endpoint == cache endpoint) must count as covered.
_COVERAGE_TOL_UM = 1e-9


def _cache_coverage_note(
    requested_um: np.ndarray, cache_wavelength_um: np.ndarray, cached_path: Path
) -> str | None:
    """Explain why a cached spectrum cannot serve ``requested_um``, or ``None``.

    ``np.interp`` silently reuses endpoint values outside the tabulated range,
    which would present cache-edge data as if it covered any requested band.
    Loaders call this first: when the requested grid extends beyond the cache
    coverage (beyond a tiny numerical tolerance) they fall back to the labelled
    synthetic proxy instead, and put the returned note into the provenance.
    """
    lo, hi = float(np.min(cache_wavelength_um)), float(np.max(cache_wavelength_um))
    r_lo, r_hi = float(np.min(requested_um)), float(np.max(requested_um))
    if r_lo >= lo - _COVERAGE_TOL_UM and r_hi <= hi + _COVERAGE_TOL_UM:
        return None
    return (
        f"Cached file {cached_path.name} covers {lo:.4f}-{hi:.4f} um but "
        f"{r_lo:.4f}-{r_hi:.4f} um was requested; refusing to extrapolate "
        "cached public data beyond its tabulated range."
    )


def _read_cache_table(
    path: Path, columns: tuple[str, ...], ranges: dict[int, tuple[float, float]]
) -> tuple[np.ndarray, bool, str]:
    """Load a cached ASCII table with validated, strictly increasing wavelengths.

    Column 0 is wavelength in um.  Paired rows are sorted by wavelength (a
    descending export describes the same curve, but ``np.interp`` silently
    returns garbage for unsorted abscissae).  Duplicate wavelengths, non-finite
    values, non-positive wavelengths and physical columns outside ``ranges``
    (``{column_index: (lo, hi)}``) raise ``ValueError`` naming the file: a
    present-but-invalid cache must never be labelled direct public data.
    Returns ``(table, resorted, sha256)``, the digest being of the exact bytes
    parsed (read once, so the table and its identity cannot disagree).
    """
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    arr = np.loadtxt(io.BytesIO(content), ndmin=2)
    if arr.shape[0] < 2 or arr.shape[1] < len(columns):
        raise ValueError(
            f"Cached file {path.name} must have >= 2 rows and columns {list(columns)}; "
            f"found shape {arr.shape}."
        )
    arr = arr[:, : len(columns)]
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Cached file {path.name} contains non-finite values.")
    if np.any(arr[:, 0] <= 0):
        raise ValueError(f"Cached file {path.name} has non-positive wavelengths.")
    resorted = bool(np.any(np.diff(arr[:, 0]) < 0))
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    if np.any(np.diff(arr[:, 0]) == 0):
        raise ValueError(
            f"Cached file {path.name} has duplicate wavelengths; refusing to guess "
            "which tabulated value applies."
        )
    for index, (lo, hi) in ranges.items():
        values = arr[:, index]
        if np.any(values < lo) or np.any(values > hi):
            raise ValueError(
                f"Cached file {path.name} column {columns[index]!r} lies outside its "
                f"physical range [{lo}, {hi}] (found {values.min():.6g}..{values.max():.6g})."
            )
    return arr, resorted, digest


_RESORTED_NOTE = "  Cache rows were re-sorted into ascending wavelength order."


def _sampling_check(
    cache_wavelength_um: np.ndarray, requested_um: np.ndarray
) -> tuple[float, float, bool]:
    """Median cache step, median requested step, and whether the cache is
    coarser than the request (``np.interp`` cannot invent sub-sample
    structure, so an under-sampled sky cache silently smooths the OH comb)."""
    cache_step = float(np.median(np.diff(np.sort(np.asarray(cache_wavelength_um, dtype=float)))))
    req = np.sort(np.asarray(requested_um, dtype=float))
    grid_step = float(np.median(np.diff(req))) if req.size > 1 else float("inf")
    return cache_step, grid_step, bool(cache_step > grid_step * (1.0 + 1e-9))


def _padded_fine_grid(requested_um: np.ndarray) -> np.ndarray:
    """Small, LSF-friendly native grid for the synthetic sky fallback.

    Four samples per requested interval preserve narrow synthetic structure, and
    eight requested-pixel widths of padding on either side comfortably cover the
    current roughly one-pixel-sigma LSF without making broad-band fallbacks huge.
    """
    requested = np.asarray(requested_um, dtype=float)
    if requested.ndim != 1 or requested.size < 2:
        return requested.copy()
    span = float(requested[-1] - requested[0])
    native_step = span / (4 * (requested.size - 1))
    requested_step = float(np.median(np.diff(requested)))
    padding = 8.0 * requested_step
    n_intervals = int(np.ceil((span + 2 * padding) / native_step))
    return requested[0] - padding + np.arange(n_intervals + 1) * native_step


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
        arr, resorted, digest = _read_cache_table(
            cached, ("wavelength_um", "transmission"), {1: (0.0, 1.0)})
        curve = {
            "band": band,
            "wavelength_um": np.atleast_1d(arr[:, 0]),
            "transmission": np.atleast_1d(arr[:, 1]),
        }
        prov = ProvenanceRecord(
            name=f"filter_{band}",
            source_class="direct_public_data",
            url_or_reference=_cache_reference(cached),
            content_sha256=digest,
            notes="Cached filter curve; original e.g. SVO Filter Profile Service."
            + (_RESORTED_NOTE if resorted else ""),
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
    wavelength_um, radiance, transmission) **and covers the requested range**,
    its radiance is flux-conservingly averaged into bins centred on
    ``wavelength_um`` (``direct_public_data``); transmission remains point
    sampled because multiplying a coarse source cube by a native transmission
    requires a separate high-resolution source path.  Native arrays are retained
    under ``native_*`` keys so the main pipeline can multiply the sky by its
    filter, apply the instrumental LSF, and only then integrate detector bins.
    The cache's sampling step is compared with the requested grid's and reported
    (``sky["cache_step_um"]``, ``sky["grid_step_um"]``,
    ``sky["cache_undersampled"]`` plus a provenance warning): linear
    interpolation cannot recover line-scale structure from a cache coarser than
    the science grid, so such a cache is flagged rather than trusted.
    Otherwise a smooth continuum plus a comb of Gaussian OH-airglow lines is
    synthesised, together with a mild telluric absorption proxy
    (``synthetic_proxy``).  A cache that exists but does not cover the request
    is never silently extrapolated: the loader falls back to the synthetic
    proxy and records why in the provenance notes.

    Returns
    -------
    (sky, provenance)
        ``sky`` has legacy science-grid keys ``"wavelength_um"``, ``"radiance"``
        (bin-mean sky-emission density) and ``"transmission"`` (point-sampled,
        0..1), plus ``"native_wavelength_um"``, ``"native_radiance"`` and
        ``"native_transmission"`` for LSF-aware processing.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    if (
        wavelength_um.ndim != 1
        or wavelength_um.size == 0
        or not np.all(np.isfinite(wavelength_um))
        or np.any(np.diff(wavelength_um) <= 0)
    ):
        raise ValueError("requested sky wavelengths must be finite and strictly increasing")
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cached = cache_dir / "skycalc.dat"

    coverage_note = None
    if cached.exists():
        arr, resorted, digest = _read_cache_table(
            cached, ("wavelength_um", "radiance", "transmission"),
            {1: (0.0, np.inf), 2: (0.0, 1.0)})
        coverage_note = _cache_coverage_note(wavelength_um, arr[:, 0], cached)
        if coverage_note is None:
            requested_edges = (
                wavelength_bin_edges(wavelength_um)
                if wavelength_um.size >= 2
                else wavelength_um
            )
            native_edge_clipped = bool(
                requested_edges[0] < arr[0, 0] - _COVERAGE_TOL_UM
                or requested_edges[-1] > arr[-1, 0] + _COVERAGE_TOL_UM
            )
            radiance = (
                bin_average_spectral_density(arr[:, 0], arr[:, 1], wavelength_um)
                if wavelength_um.size >= 2
                else np.interp(wavelength_um, arr[:, 0], arr[:, 1])
            )
            transmission = np.interp(wavelength_um, arr[:, 0], arr[:, 2])
            cache_step, grid_step, undersampled = _sampling_check(arr[:, 0], wavelength_um)
            sky = {
                "wavelength_um": wavelength_um,
                "radiance": radiance,
                "transmission": transmission,
                "native_wavelength_um": arr[:, 0].copy(),
                "native_radiance": arr[:, 1].copy(),
                "native_transmission": arr[:, 2].copy(),
                "cache_step_um": cache_step,
                "grid_step_um": grid_step,
                "cache_undersampled": undersampled,
                "native_edge_clipped": native_edge_clipped,
            }
            note = (
                "Cached SkyCalc-style sky model (radiance + telluric transmission); "
                f"cache step {cache_step * 1e3:.3f} nm vs requested grid step "
                f"{grid_step * 1e3:.3f} nm. Radiance uses flux-conserving "
                "detector-bin means; native samples are retained for single-LSF "
                "pipeline processing. Transmission remains point-sampled on the "
                "requested grid; this is not a high-resolution source/telluric model."
            )
            if undersampled:
                note += (
                    "  WARNING: the cache is sampled more coarsely than the requested "
                    "grid, so line-scale sky structure (OH lines, narrow telluric "
                    "features) is under-resolved by linear interpolation; sky-line "
                    "mask fractions and line-window sky residuals are not reliable "
                    "at this resolution -- re-fetch with a finer wdelta."
                )
            if resorted:
                note += _RESORTED_NOTE
            if native_edge_clipped:
                note += (
                    "  WARNING: outer detector-bin edges extend beyond the native "
                    "cache; unsupported fractions are zero-filled rather than "
                    "invented by endpoint extrapolation."
                )
            prov = ProvenanceRecord(
                name="sky_spectrum",
                source_class="direct_public_data",
                url_or_reference=_cache_reference(cached),
                content_sha256=digest,
                notes=note,
            )
            return sky, prov

    # Synthetic proxy: low continuum + OH airglow comb + shallow telluric dips.
    # Retain an inexpensive fine, padded representation so the pipeline can
    # apply its LSF before detector-bin integration just as it does for a cache.
    native_wavelength_um = _padded_fine_grid(wavelength_um)
    rng = np.random.default_rng(0)
    continuum = np.full_like(native_wavelength_um, 0.05)
    # Place a deterministic comb of OH-like emission lines across the range.
    n_lines = 30
    line_centres = np.linspace(wavelength_um.min(), wavelength_um.max(), n_lines + 2)[1:-1]
    line_sigma = 0.0005 * (wavelength_um.mean())  # narrow relative to the band
    native_radiance = continuum.copy()
    for lc in line_centres:
        amp = 0.5 + rng.uniform(0.0, 1.0)
        native_radiance += amp * np.exp(
            -0.5 * ((native_wavelength_um - lc) / line_sigma) ** 2
        )
    # Telluric proxy: a few broad shallow absorption troughs.
    native_transmission = np.ones_like(native_wavelength_um)
    span = float(np.ptp(wavelength_um))
    for tc, depth, width in [
        (wavelength_um.min() + 0.30 * span, 0.3, 0.02),
        (wavelength_um.min() + 0.65 * span, 0.2, 0.03),
    ]:
        native_transmission -= depth * np.exp(
            -0.5 * ((native_wavelength_um - tc) / width) ** 2
        )
    native_transmission = np.clip(native_transmission, 0.0, 1.0)
    radiance = (
        bin_average_spectral_density(
            native_wavelength_um, native_radiance, wavelength_um
        )
        if wavelength_um.size >= 2
        else native_radiance.copy()
    )
    transmission = np.interp(
        wavelength_um, native_wavelength_um, native_transmission
    )
    sky = {
        "wavelength_um": wavelength_um,
        "radiance": radiance,
        "transmission": transmission,
        "native_wavelength_um": native_wavelength_um,
        "native_radiance": native_radiance,
        "native_transmission": native_transmission,
        "native_edge_clipped": False,
    }
    base_note = (
        "No cached SkyCalc spectrum found; using synthetic continuum + OH "
        "airglow comb + shallow telluric proxy.  Replace with ESO SkyCalc "
        "output for direct_public_data."
        if coverage_note is None
        else coverage_note + "  Using the synthetic sky proxy for the whole range instead."
    )
    prov = ProvenanceRecord(
        name="sky_spectrum",
        source_class="synthetic_proxy",
        url_or_reference="synthetic sky continuum + OH-line proxy",
        notes=(
            base_note
            + " Radiance uses flux-conserving detector-bin means and retains a "
            "fine padded native proxy for single-LSF processing. Transmission "
            "remains point-sampled on the requested grid."
        ),
    )
    return sky, prov


def load_source_spectrum(
    wavelength_um: np.ndarray,
    temperature_k: float = 5000.0,
    cache_dir: str | Path | None = None,
) -> tuple[dict, ProvenanceRecord]:
    """Load (or synthesise) a 1-D source spectrum.

    If a cached file ``source.dat`` exists (columns: wavelength_um, flux) **and
    covers the requested range**, it is interpolated onto ``wavelength_um``
    (``direct_public_data``).  Otherwise a blackbody continuum at
    ``temperature_k`` is synthesised (``synthetic_proxy``).  A cache that
    exists but does not cover the request is never silently extrapolated: the
    loader falls back to the blackbody proxy and records why in the provenance
    notes.

    The returned ``"flux"`` is the **photon-count spectral shape** in relative
    units (peak-normalised): the simulation counts electrons per wavelength
    bin, and a CALSPEC file tabulates the ENERGY flux density ``F_lambda``
    (erg/s/cm^2/A), so the shape that multiplies an electron cube is
    ``F_lambda * lambda / (h c)`` -- the photon factor ``lambda / h c`` tilts
    the continuum redward (across 1.99-2.34 um the energy shape falls to
    0.326 of its blue end, the photon shape to 0.642).  The energy shape is
    returned alongside as ``"energy_flux"`` for reference.  Absolute scaling
    is handled separately by the photon-budget function in ``source_cube``.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cached = cache_dir / "source.dat"

    coverage_note = None
    if cached.exists():
        arr, resorted, digest = _read_cache_table(cached, ("wavelength_um", "flux"), {1: (0.0, np.inf)})
        coverage_note = _cache_coverage_note(wavelength_um, arr[:, 0], cached)
        if coverage_note is None:
            energy = np.interp(wavelength_um, arr[:, 0], arr[:, 1])
            energy = energy / np.max(energy) if np.max(energy) > 0 else energy
            photon = energy * wavelength_um            # F_lambda * lambda / hc, relative
            photon = photon / np.max(photon) if np.max(photon) > 0 else photon
            spec = {"wavelength_um": wavelength_um, "flux": photon, "energy_flux": energy}
            prov = ProvenanceRecord(
                name="source_spectrum",
                source_class="direct_public_data",
                url_or_reference=_cache_reference(cached),
                content_sha256=digest,
                notes=(
                    "Cached source spectrum, e.g. CALSPEC/PHOENIX; energy flux density "
                    "converted to a photon-count shape (x lambda), peak-normalised."
                    + (_RESORTED_NOTE if resorted else "")
                ),
            )
            return spec, prov

    energy = blackbody_relative(wavelength_um, temperature_k)
    photon = energy * wavelength_um
    photon = photon / np.max(photon)
    spec = {"wavelength_um": wavelength_um, "flux": photon, "energy_flux": energy}
    base_note = (
        f"No cached source spectrum found; using a {temperature_k:.0f} K "
        "blackbody continuum (photon-count shape).  Replace with CALSPEC/PHOENIX "
        "for direct_public_data."
        if coverage_note is None
        else coverage_note + f"  Using the {temperature_k:.0f} K blackbody proxy for the whole range instead."
    )
    prov = ProvenanceRecord(
        name="source_spectrum",
        source_class="synthetic_proxy",
        url_or_reference="analytic blackbody fallback",
        notes=base_note,
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
