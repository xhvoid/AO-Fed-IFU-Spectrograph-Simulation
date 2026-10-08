"""Shared reporting helpers: provenance-tagged tables and an artifact manifest.

The example scripts and the CLI all emit the same kinds of artifacts -- a CSV of
metrics, a JSON run summary, and an entry in a shared artifact manifest -- and
the project requires every one of them to carry provenance labels.  Centralising
that here keeps the rules in one place:

* every metrics-CSV row has ``source_class`` (a single VALID_SOURCE_CLASSES value)
  plus ``component`` and ``notes``;
* every JSON summary carries the author and a non-overclaiming note;
* the manifest's ``source_class`` is one primary class and ``source_classes_used``
  is a ';'-separated list of *valid* classes (never the placeholder "mixed").

``update_manifest`` merges new rows into ``artifact_manifest.csv`` keyed by
``artifact_path``, so running any generator keeps a single comprehensive manifest
up to date without clobbering rows written by the others.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data_sources import VALID_SOURCE_CLASSES

# One-line boundary statement attached to every JSON summary we write.
NON_OVERCLAIM_NOTE = (
    "Learning/portfolio demonstrator. NOT a calibrated observatory simulator, "
    "NOT an exposure-time calculator, NOT a reproduction of HARMONI/MUSE/NIRSpec, "
    "and NOT a validated performance prediction."
)

MANIFEST_COLUMNS = [
    "artifact_path", "artifact_type", "generated_by", "source_class",
    "source_classes_used", "inputs_used", "notes",
]


def use_agg() -> None:
    """Select the headless matplotlib backend (call before importing pyplot)."""
    import matplotlib

    matplotlib.use("Agg")


def validate_class(source_class: str) -> str:
    """Return ``source_class`` if it is a valid provenance class, else raise."""
    if source_class not in VALID_SOURCE_CLASSES:
        raise ValueError(f"unknown source_class {source_class!r}; expected one of {VALID_SOURCE_CLASSES}")
    return source_class


def join_source_classes(classes) -> str:
    """Dedup + validate a list of class tokens into a ';'-separated string."""
    ordered: list[str] = []
    for c in classes:
        if c and c not in ordered:
            validate_class(c)
            ordered.append(c)
    return ";".join(ordered)


def metric_row(metric: str, value, source_class: str, component: str = "", notes: str = "") -> dict:
    """Build one validated metrics-table row."""
    return {
        "metric": metric,
        "value": value,
        "source_class": validate_class(source_class),
        "component": component,
        "notes": notes,
    }


def write_metrics_csv(path: str | Path, rows: list[dict]) -> Path:
    """Write a metrics CSV with columns metric, value, source_class, component, notes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        validate_class(r["source_class"])
        r.setdefault("component", "")
        r.setdefault("notes", "")
    pd.DataFrame(rows, columns=["metric", "value", "source_class", "component", "notes"]).to_csv(
        path, index=False
    )
    return path


def write_summary_json(path: str | Path, payload: dict) -> Path:
    """Write a strict-JSON summary with author, note and software identity.

    Unavailable numbers (NaN, +/-inf) are written as ``null`` -- strict JSON has
    no NaN/Infinity -- and each one's JSON path and original marker is listed
    under ``unavailable_values`` so "not measured" is never confused with zero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {"author": "Xu Han", "package_version": __version__, "note": NON_OVERCLAIM_NOTE,
            "software": software_identity()}
    full.update(payload)
    unavailable: dict[str, str] = {}
    clean = to_strict_json(full, unavailable)
    if unavailable:
        clean["unavailable_values"] = unavailable
    path.write_text(json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8")
    return path


def to_strict_json(obj, unavailable: dict[str, str] | None = None, _path: str = ""):
    """Convert ``obj`` to strict-JSON types, replacing non-finite floats by ``None``.

    numpy scalars/arrays, tuples, dataclasses and paths are converted; the JSON
    path of every replaced value is recorded in ``unavailable`` as ``"nan"``,
    ``"inf"`` or ``"-inf"``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_strict_json(v, unavailable, f"{_path}.{k}" if _path else str(k))
                for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_strict_json(v, unavailable, f"{_path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        if unavailable is not None:
            unavailable[_path] = "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def file_sha256(path: str | Path) -> str:
    """SHA-256 of one file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha256() -> str:
    """Content hash of the code that generates artifacts.

    Covers the package (``src/ao_fed_ifu``) and, in a source checkout, the
    ``examples/run_*.py`` generators that compute and write most reports.
    Comparing it with a saved summary's ``software.source_sha256`` shows whether
    that artifact predates the current implementation.
    """
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    files = [(f"ao_fed_ifu/{p.relative_to(package).as_posix()}", p) for p in sorted(package.rglob("*.py"))]
    examples = package.parents[1] / "examples"
    if examples.is_dir():
        files += [(f"examples/{p.name}", p) for p in sorted(examples.glob("run_*.py"))]
    for label, path in files:
        digest.update(label.encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def software_identity() -> dict:
    """Package/source identity and the versions of the numerical dependencies."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"package_version": __version__, "source_sha256": source_sha256(), "versions": versions}


def input_cache_identity(provenance_records: list[dict]) -> dict:
    """The public-data cache files that actually conditioned a run, by content hash.

    Derived from the run's provenance: only components served as
    ``direct_public_data`` from a local cache file count (a cache rejected for
    coverage fell back to a synthetic proxy and is not an input).
    """
    from .data_sources import resolve_cache_reference

    files: dict[str, str | None] = {}
    for record in provenance_records:
        if record.get("source_class") != "direct_public_data":
            continue
        reference = str(record.get("url_or_reference", ""))
        path = resolve_cache_reference(reference)
        if path is not None:
            label = reference.split(" ", 2)[-1]
            files[label] = file_sha256(path) if path.exists() else None
    return {"used": bool(files), "files_sha256": dict(sorted(files.items()))}


def manifest_entry(
    artifact_path: str,
    artifact_type: str,
    generated_by: str,
    source_class: str,
    source_classes_used,
    inputs_used: str,
    notes: str,
) -> dict:
    """Build one validated artifact-manifest row.

    ``source_class`` is a single primary class; ``source_classes_used`` may be a
    list or a ';'-string -- it is normalised to a validated ';'-string.
    """
    if isinstance(source_classes_used, str):
        classes = source_classes_used.split(";")
    else:
        classes = list(source_classes_used)
    return {
        "artifact_path": artifact_path,
        "artifact_type": artifact_type,
        "generated_by": generated_by,
        "source_class": validate_class(source_class),
        "source_classes_used": join_source_classes(classes),
        "inputs_used": inputs_used,
        "notes": notes,
    }


def update_manifest(outdir: str | Path, entries: list[dict]) -> Path:
    """Merge ``entries`` into ``outdir/artifact_manifest.csv`` keyed by path.

    Existing rows for the same ``artifact_path`` are replaced; other rows are
    preserved, so each generator can keep its own rows current without disturbing
    the others.  The result is sorted by artifact_path for a stable diff.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "artifact_manifest.csv"

    existing: dict[str, dict] = {}
    if path.exists():
        for _, row in pd.read_csv(path).iterrows():
            existing[str(row["artifact_path"])] = {c: row.get(c, "") for c in MANIFEST_COLUMNS}
    for e in entries:
        existing[e["artifact_path"]] = {c: e.get(c, "") for c in MANIFEST_COLUMNS}

    rows = [existing[k] for k in sorted(existing)]
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(path, index=False)
    return path
