"""
Tests for typed kinds: construction, round-trip, validation, and op contracts.

Demonstrates that:
1. A kind survives the construct → serialize → load → validate cycle.
2. Malformed kinds are rejected by ``validate`` (the wrapped object must be a
   non-empty ``DataArray``).
3. Required dimensions/coordinates are enforced structurally by ``require`` —
   the contract an op declares for the kind it consumes.

Dataset-loading tests (malformed-dataset rejection, real 10x loads) live in
``test_tenxdatasets.py``.
"""

import tempfile
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

    def test_validate_fails_on_empty(self):
        da = xr.DataArray(np.empty((0, 0)), dims=["cell", "gene"])  # size-0 array
        kind = LabeledArray(da)
        with pytest.raises(ValueError, match="empty"):
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
# Example ops: the shipped demonstration ops stay in sync with the kind API
# ==============================================================================


class TestExampleOps:
    """Smoke tests so ``tests/singlecell/test_op/example_ops.py`` can't silently rot."""

    def _array(self, dims, coords):
        return xr.DataArray(np.ones(tuple(len(coords[d]) for d in dims)), dims=dims, coords=coords)

    def test_temporal_analysis_requires_time(self):
        from tests.singlecell.test_op.example_ops import temporal_analysis

        no_time = LabeledArray(self._array(
            ["cell", "gene"],
            {"cell": ["c0", "c1"], "gene": ["g0", "g1", "g2"]},
        ))
        with pytest.raises(ValueError, match="requires dims"):
            temporal_analysis(no_time)

    def test_temporal_analysis_runs_with_time(self):
        from tests.singlecell.test_op.example_ops import temporal_analysis

        with_time = LabeledArray(self._array(
            ["cell", "gene", "time"],
            {"cell": ["c0", "c1"], "gene": ["g0", "g1", "g2"], "time": [0, 1]},
        ))
        out = temporal_analysis(with_time)
        assert isinstance(out, LabeledArray)
        assert "time" in out.da.dims

    def test_basic_filter_drops_low_expression(self):
        from tests.singlecell.test_op.example_ops import basic_filter

        da = xr.DataArray(
            np.array([[0.0, 5.0], [0.0, 5.0]]),  # gene g0 mean 0, gene g1 mean 5
            dims=["cell", "gene"],
            coords={"cell": ["c0", "c1"], "gene": ["g0", "g1"]},
        )
        out = basic_filter(LabeledArray(da), min_expression=1.0)
        assert isinstance(out, LabeledArray)
        assert list(out.da.gene.values) == ["g1"]  # low-expression g0 removed
