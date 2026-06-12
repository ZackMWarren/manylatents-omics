"""
Tests for typed kinds: construction, round-trip, validation, and op contracts.

Demonstrates that:
1. A kind survives the construct → serialize → load → validate cycle.
2. Malformed kinds are rejected by ``validate`` (the wrapped object must be a
   ``DataArray``).
3. Required dimensions/coordinates are enforced structurally by ``require`` —
   the contract an op declares for the kind it consumes.
4. Real 10x datasets (sampled from the Geomancer CSV) load, validate, and have
   their dimension requirements enforced by ``require``.

The real-data tests are marked ``network``/``slow`` and skip gracefully when the
dataset CSV is absent or the host is offline. Knobs (all optional):
  - ``GEOMANCER_DATASETS_CSV``  path to the datasets CSV
  - ``GEOMANCER_10X_CACHE``     directory to cache downloaded ``.h5`` files
  - ``GEOMANCER_MAX_MB``        skip datasets larger than this (default 1500)
  - ``GEOMANCER_TEST_SEED``     seed the random dataset pick for reproducibility
"""

import csv
import os
import random
import re
import shutil
import socket
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from manylatents.singlecell.data.kinds.kinds import Kind, LabeledArray


# ==============================================================================
# Base Kind
# ==============================================================================


def test_kind_is_abstract():
    """The abstract base cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Kind()  # type: ignore[abstract]


# ==============================================================================
# LabeledArray: construct → serialize → load round-trip
# ==============================================================================


class TestLabeledArrayRoundTrip:
    """Test construct → serialize → load → validate cycle.

    ``load`` uses ``open_dataarray``, so the wrapped array round-trips whatever
    its name (including the unnamed arrays the adapter produces).
    """

    def test_round_trip_basic(self):
        """Round-trip with minimal data preserves values and structure."""
        data = np.random.rand(10, 5)
        da = xr.DataArray(
            data,
            dims=["cell", "gene"],
            coords={"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            loaded.validate()
            assert (loaded.da == kind.da).all()
            assert list(loaded.da.dims) == list(kind.da.dims)
            assert list(loaded.da.cell) == list(kind.da.cell)
            assert list(loaded.da.gene) == list(kind.da.gene)

    def test_round_trip_with_attrs(self):
        """Round-trip preserves attributes."""
        da = xr.DataArray(
            np.ones((5, 3)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3", "c4", "c5"], "gene": ["g1", "g2", "g3"]},
            attrs={"genome": "GRCh38", "source": "10x"},
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            assert loaded.da.attrs["genome"] == "GRCh38"
            assert loaded.da.attrs["source"] == "10x"

    def test_round_trip_with_time_dim(self):
        """Round-trip with an optional time dimension."""
        da = xr.DataArray(
            np.random.rand(10, 5, 3),  # cells × genes × time
            dims=["cell", "gene", "time"],
            coords={
                "cell": [f"c{i}" for i in range(10)],
                "gene": [f"g{i}" for i in range(5)],
                "time": [0, 1, 2],
            },
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            assert "time" in loaded.da.dims
            assert len(loaded.da.time) == 3


# ==============================================================================
# LabeledArray: validation (the wrapped object must be a DataArray)
# ==============================================================================


class TestLabeledArrayValidation:
    """``validate`` is a structural type check and returns the kind for chaining."""

    def test_validate_passes_on_dataarray(self):
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        kind = LabeledArray(da)
        assert kind.validate() is kind  # returns self, no raise

    def test_validate_fails_on_non_dataarray(self):
        kind = LabeledArray([1, 2, 3])  # not an xr.DataArray
        with pytest.raises(ValueError, match="must wrap a DataArray"):
            kind.validate()


# ==============================================================================
# Op contracts: requiring dimensions and coordinates via ``require``
# ==============================================================================


class TestRequireContract:
    """Ops declare and enforce the dims/coords they consume via ``kind.require``."""

    def _array(self, dims, coords):
        return xr.DataArray(np.ones(tuple(len(coords[d]) for d in dims)), dims=dims, coords=coords)

    def test_require_passes_with_present_dims(self):
        da = self._array(
            ["cell", "gene"],
            {"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        kind = LabeledArray(da)
        assert kind.require("cell", "gene") is kind  # returns self for chaining

    def test_require_rejects_missing_dim(self):
        da = xr.DataArray(np.ones((5,)), dims=["gene"], coords={"gene": [f"g{i}" for i in range(5)]})
        with pytest.raises(ValueError, match="requires dims"):
            LabeledArray(da).require("cell", "gene")

    def test_require_rejects_wrong_dims(self):
        da = self._array(
            ["samples", "features"],  # wrong names
            {"samples": list(range(10)), "features": list(range(5))},
        )
        with pytest.raises(ValueError, match="requires dims"):
            LabeledArray(da).require("cell", "gene")

    def test_require_rejects_missing_coord(self):
        da = self._array(
            ["cell", "gene"],
            {"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        with pytest.raises(ValueError, match="requires coords"):
            LabeledArray(da).require("cell", "gene", coords=("gene_ids",))

    def test_temporal_op_requires_time_dim(self):
        """An op that needs a ``time`` dim accepts a kind that carries one."""
        da = self._array(
            ["cell", "gene", "time"],
            {
                "cell": [f"c{i}" for i in range(10)],
                "gene": [f"g{i}" for i in range(5)],
                "time": [0, 1, 2],
            },
        )
        kind = LabeledArray(da)
        assert kind.require("cell", "gene", "time") is kind

    def test_temporal_op_rejects_missing_time_dim(self):
        """The same op fails cleanly when ``time`` is absent."""
        da = self._array(
            ["cell", "gene"],
            {"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        with pytest.raises(ValueError, match="requires dims"):
            LabeledArray(da).require("cell", "gene", "time")


# ==============================================================================
# Real 10x datasets sampled from the Geomancer CSV
# ==============================================================================


def _datasets_csv() -> Path | None:
    """Locate the Geomancer datasets CSV (env override, else repo root)."""
    env = os.environ.get("GEOMANCER_DATASETS_CSV")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # tests/singlecell/test_kinds.py -> tests/singlecell -> tests -> manylatents-omics
    candidate = Path(__file__).resolve().parents[2] / "Datasets for Geomancer - 10x Genomics.csv"
    return candidate if candidate.exists() else None


_H5_URL_RE = re.compile(r"https?://\S+?\.h5\b")


def _select_random_datasets(n: int = 5) -> list[tuple[str, str]]:
    """Pick ``n`` random (name, .h5 URL) pairs from the datasets CSV."""
    csv_path = _datasets_csv()
    if csv_path is None:
        return []

    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            match = _H5_URL_RE.search(row.get("wget_commands") or "")
            if match:
                rows.append((row.get("Dataset_Name") or match.group(0), match.group(0)))
    if not rows:
        return []

    seed = os.environ.get("GEOMANCER_TEST_SEED")
    rng = random.Random(int(seed)) if (seed and seed.isdigit()) else random.Random()
    return rng.sample(rows, min(n, len(rows)))


# Resolved once at collection time so each dataset is a separate test case.
_SELECTED_DATASETS = _select_random_datasets(5)


def _short_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40]


def _cache_dir() -> Path:
    d = Path(os.environ.get("GEOMANCER_10X_CACHE") or (Path(tempfile.gettempdir()) / "geomancer_10x_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_h5(url: str) -> Path:
    """Download (and cache) a 10x ``.h5``; skip the test on network failure or size cap."""
    dest = _cache_dir() / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    max_mb = int(os.environ.get("GEOMANCER_MAX_MB", "1500"))
    req = urllib.request.Request(url, headers={"User-Agent": "geomancer-tests"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            length = resp.headers.get("Content-Length")
            if length and int(length) > max_mb * 1024 * 1024:
                pytest.skip(
                    f"{dest.name} is {int(length) // (1024 * 1024)} MB > cap {max_mb} MB "
                    f"(raise GEOMANCER_MAX_MB to include it)"
                )
            part = dest.with_suffix(dest.suffix + ".part")
            with open(part, "wb") as out:
                shutil.copyfileobj(resp, out)
        part.rename(dest)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        if dest.exists():
            dest.unlink()
        pytest.skip(f"could not download {url}: {e}")
    return dest


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.skipif(not _SELECTED_DATASETS, reason="Geomancer 10x datasets CSV not found")
@pytest.mark.parametrize(
    "ds_name,url",
    _SELECTED_DATASETS,
    ids=[_short_id(name) for name, _ in _SELECTED_DATASETS],
)
def test_random_10x_dataset_loads_validates_and_enforces_dims(ds_name, url):
    """A randomly chosen 10x dataset loads, validates, and respects op dim contracts."""
    from manylatents.singlecell.data.adapters.sources.tenx import make_data

    h5 = _download_h5(url)

    # 1. Loading: real 10x .h5 -> typed LabeledArray via the production adapter.
    try:
        kind = make_data(str(h5))
    except ValueError as e:
        # The adapter rejects non-scRNA-seq (antibody/CRISPR) modalities at the
        # edge — that's correct behavior, just not testable as scRNA-seq here.
        if "not scRNA-seq" in str(e):
            pytest.skip(f"{ds_name} is multimodal, not pure scRNA-seq: {e}")
        raise
    assert isinstance(kind, LabeledArray)

    # 2. Validation: the loaded kind wraps a well-formed DataArray and satisfies
    #    the cell/gene contract every downstream op assumes.
    kind.validate()
    kind.require("cell", "gene", coords=("cell", "gene"))
    assert {"cell", "gene"} <= set(kind.da.dims)
    assert {"cell", "gene"} <= set(kind.da.coords)

    # 3. Requiring dimensions for specific ops.
    #    A temporal op needs a 'time' dim, which raw 10x data lacks, so require
    #    must reject this kind cleanly rather than fail deep in a computation.
    assert "time" not in kind.da.dims
    with pytest.raises(ValueError, match="requires dims"):
        kind.require("cell", "gene", "time")

    #    A cell/gene-only op is satisfied. Exercise it on a small dense slice so
    #    we don't materialize the full (often >10k×30k) sparse matrix.
    n_cells = min(64, kind.da.sizes["cell"])
    n_genes = min(128, kind.da.sizes["gene"])
    small = kind.da.isel(cell=slice(0, n_cells), gene=slice(0, n_genes))
    small = small.copy(data=np.asarray(small.data.todense()))
    small_kind = LabeledArray(small)

    small_kind.validate()
    small_kind.require("cell", "gene")
    assert {"cell", "gene"} <= set(small_kind.da.dims)
