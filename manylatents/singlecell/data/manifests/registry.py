"""Dataset manifest registry — the single source of truth for available datasets.

The manifest is a CSV (``datasets_10x.csv``, in this directory) derived from the
Geomancer "Datasets for Geomancer - 10x Genomics" spreadsheet. It is the registry
of *what data exists*; loaders and tests read it instead of hardcoding paths.

**The spreadsheet is proprietary and is git-ignored — it is not committed.**
Obtain it out of band and drop it in as ``datasets_10x.csv``, or point
``GEOMANCER_DATASETS_CSV`` at your copy (it takes precedence over the in-repo
default). When the manifest is absent, ``load_manifest`` raises and dataset
tests skip; that is expected on a fresh clone.

The spreadsheet is expected to change as datasets are added or removed — just
replace the CSV with a fresh export (same columns), no code change needed.

Expected columns (only two are required): ``Dataset_Name`` and ``wget_commands``
(which must contain a downloadable ``.h5`` URL). Every other column is carried
through on each entry's ``meta`` dict, so the spreadsheet can grow freely.
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

# Default in-repo manifest. Override with the GEOMANCER_DATASETS_CSV env var.
_DEFAULT_MANIFEST = Path(__file__).with_name("datasets_10x.csv")

# Matches a wget URL ending in .h5 inside the spreadsheet's wget_commands column.
_H5_URL_RE = re.compile(r"https?://\S+?\.h5\b")


@dataclass
class DatasetEntry:
    """One manifest row: name, downloadable ``.h5`` URL, and the remaining columns.

    ``meta`` carries every other spreadsheet column verbatim (``Species``,
    ``Tissue``, ``Number_of_Cells``, ``DOI``, ...), so new columns flow through
    without a code change. It is a plain ``dict`` (hence this dataclass is not
    frozen/hashable).
    """

    name: str
    url: str
    meta: dict[str, str] = field(default_factory=dict)


def manifest_path() -> Path:
    """Path to the active manifest CSV (env override, else the in-repo default)."""
    env = os.environ.get("GEOMANCER_DATASETS_CSV")
    return Path(env) if env else _DEFAULT_MANIFEST


def load_manifest(path: Path | None = None) -> list[DatasetEntry]:
    """Parse the manifest into entries that carry a usable ``.h5`` URL.

    Rows whose ``wget_commands`` contains no ``.h5`` link are skipped. Raises
    ``FileNotFoundError`` if the manifest is absent, so a missing registry fails
    loudly rather than silently yielding an empty list.
    """
    path = path or manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {path}. "
            f"Set GEOMANCER_DATASETS_CSV to point at a copy, or restore "
            f"datasets_10x.csv in the manifests directory."
        )

    entries: list[DatasetEntry] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            match = _H5_URL_RE.search(row.get("wget_commands") or "")
            if match:
                entries.append(
                    DatasetEntry(
                        name=row.get("Dataset_Name") or match.group(0),
                        url=match.group(0),
                        meta={
                            k: v
                            for k, v in row.items()
                            if k not in ("Dataset_Name", "wget_commands")
                        },
                    )
                )
    return entries


def select_random(
    n: int = 5, *, seed: int | None = None, path: Path | None = None
) -> list[DatasetEntry]:
    """Pick ``n`` random entries from the manifest (deterministic when ``seed`` set)."""
    entries = load_manifest(path)
    rng = random.Random(seed)
    return rng.sample(entries, min(n, len(entries)))
