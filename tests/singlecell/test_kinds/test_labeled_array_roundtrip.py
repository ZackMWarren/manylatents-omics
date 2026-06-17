import tempfile
from pathlib import Path
import pytest
import numpy as np
import xarray as xr
from manylatents.singlecell.data.kinds.kinds import LabeledArray

class TestLabeledArrayRoundTrip:
    def test_round_trip_basic(self):
        """Round-trip with minimal data preserves values and structure."""
        rng = np.random.default_rng()
        da = xr.DataArray(
            rng.random((10, 5)),
            dims=["cell", "gene"],
            coords={"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.zarr")
            kind.serialize(path)
            loaded = LabeledArray.load(path)

            xr.testing.assert_equal(loaded.da, kind.da)

    def test_round_trip_preserves_attrs(self):
        """Round-trip preserves attributes (assert_equal ignores attrs, so check explicitly)."""
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
        """Round-trip with an optional time dimension preserves it."""
        rng = np.random.default_rng()
        da = xr.DataArray(
            rng.random((10, 5, 3)),  # cells × genes × time
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

            xr.testing.assert_equal(loaded.da, kind.da)
            assert "time" in loaded.da.dims
            assert len(loaded.da.time) == 3

    def test_assert_wrong_filetype(self):
        """Round-trip with minimal data preserves values and structure."""
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        kind = LabeledArray(da)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")
            with pytest.raises(ValueError, match=r"\.zarr"):
                kind.serialize(path)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            LabeledArray.load("nonexistent.zarr")  # passes _normalize, fails at open

    def test_load_rejects_wrong_extension(self):
        with pytest.raises(ValueError, match=r"\.zarr"):
            LabeledArray.load("nonexistent.csv")
