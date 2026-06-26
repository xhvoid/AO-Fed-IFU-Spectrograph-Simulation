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

import json
from pathlib import Path

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
    """Write a JSON summary, injecting the author and non-overclaiming note."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {"author": "Xu Han", "package_version": __version__, "note": NON_OVERCLAIM_NOTE}
    full.update(payload)
    path.write_text(json.dumps(full, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(obj):
    """Fallback JSON encoder for numpy scalars / arrays."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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
