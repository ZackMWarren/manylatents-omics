"""10x Genomics dataset registry — the single source of truth for 10x datasets.

This registry is **specific to 10x Genomics data.** The manifest is a CSV
(``datasets_10x.csv``, in this directory) derived from the Geomancer "Datasets
for Geomancer - 10x Genomics" spreadsheet, and every entry points at a 10x
``.h5`` (feature-barcode matrix) download. It is the registry of *what 10x data
exists*; loaders and tests read it instead of hardcoding paths. Other data
sources, if added later, get their own registry rather than sharing this one.

**The spreadsheet is proprietary and is git-ignored — it is not committed.**
Obtain it out of band and drop it in as ``datasets_10x.csv``, or point
``GEOMANCER_DATASETS_CSV`` at your copy (it takes precedence over the in-repo
default). When the manifest is absent, ``load_tenx_manifest`` raises and dataset
tests skip; that is expected on a fresh clone.

The spreadsheet is expected to change as datasets are added or removed — just
replace the CSV with a fresh export (same columns), no code change needed.

Expected columns (only two are required): ``Dataset_Name`` and ``wget_commands``
(which must contain a downloadable 10x ``.h5`` URL). Every other column is
carried through on each entry's ``meta`` dict, so the spreadsheet can grow freely.
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

# Default in-repo 10x manifest. Override with the GEOMANCER_DATASETS_CSV env var.
_DEFAULT_10X_MANIFEST = Path(__file__).with_name("datasets_10x.csv")

# Matches a wget URL ending in .h5 inside the spreadsheet's wget_commands column.
_H5_URL_RE = re.compile(r"https?://\S+?\.h5\b")


@dataclass
class TenxDatasetEntry:
    """One 10x manifest row: name, downloadable ``.h5`` URL, and remaining columns.

    ``meta`` carries every other spreadsheet column verbatim (``Species``,
    ``Tissue``, ``Number_of_Cells``, ``DOI``, ...), so new columns flow through
    without a code change. It is a plain ``dict`` (hence this dataclass is not
    frozen/hashable).
    """

    name: str
    url: str
    meta: dict[str, str] = field(default_factory=dict)


def tenx_manifest_path() -> Path:
    """Path to the active 10x manifest CSV (env override, else in-repo default)."""
    env = os.environ.get("GEOMANCER_DATASETS_CSV")
    return Path(env) if env else _DEFAULT_10X_MANIFEST


def load_tenx_manifest(path: Path | None = None) -> list[TenxDatasetEntry]:
    """Parse the 10x manifest into entries that carry a usable ``.h5`` URL.

    Rows whose ``wget_commands`` contains no ``.h5`` link are skipped. Raises
    ``FileNotFoundError`` if the manifest is absent, so a missing registry fails
    loudly rather than silently yielding an empty list.
    """
    path = path or tenx_manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"10x dataset manifest not found at {path}. "
            f"Set GEOMANCER_DATASETS_CSV to point at a copy, or restore "
            f"datasets_10x.csv in the manifests directory."
        )

    entries: list[TenxDatasetEntry] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            match = _H5_URL_RE.search(row.get("wget_commands") or "")
            if match:
                entries.append(
                    TenxDatasetEntry(
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

# used for loader tests
def select_random_tenx(
    n: int = 5, *, seed: int | None = None, path: Path | None = None
) -> list[TenxDatasetEntry]:
    """Pick ``n`` random 10x entries from the manifest (deterministic when ``seed`` set)."""
    entries = load_tenx_manifest(path)
    rng = random.Random(seed)
    return rng.sample(entries, min(n, len(entries)))
