import pytest
import numpy as np
import sparse
import xarray as xr
from manylatents.kinds import LabeledArray
from scipy.sparse import random_array


@pytest.fixture(params=["dense", "sparse"])
def roundtrip(request, tmp_path):
    """Factory: build a labeled array, serialize it, and reload it.

    Call it to run one round-trip and get back the ``(original, loaded)``
    DataArrays to compare. Pass ``with_time=True`` to add an optional non-
    dimensional ``time`` coordinate (defined along ``cell``).

    The sparse param builds a pydata ``sparse.COO`` duck array; serialize stores
    its bare COO components so it round-trips without densifying.
    """

    def _run(with_time: bool = False):
        rng = np.random.default_rng()
        n_cells, n_genes = 1000, 1000
        cell_ids = [f"c{i}" for i in range(n_cells)]
        gene_ids = [f"g{i}" for i in range(n_genes)]

        if request.param == "dense":
            data = rng.random((n_cells, n_genes))
        else:
            # xarray needs a pydata `sparse` array (COO); scipy sparse isn't
            # recognized as an ndarray-like backing store.
            scipy_arr = random_array(
                (n_cells, n_genes), density=0.01, random_state=rng, format="csr"
            )
            data = sparse.COO.from_scipy_sparse(scipy_arr)

        coords = {"cell": cell_ids, "gene": gene_ids}
        if with_time:
            coords["time"] = ("cell", [f"t{(i % 2) + 1}" for i in range(n_cells)])

        original = xr.DataArray(
            data,
            dims=["cell", "gene"],
            coords=coords,
            attrs={"genome": "GRCh38"},
        )
        kind = LabeledArray(original)

        path = str(tmp_path / "test.zarr")
        kind.serialize(path)
        loaded = LabeledArray.load(path)
        return original, loaded.da

    return _run


class TestLabeledArrayRoundTrip:
    def test_round_trip_identical(self, roundtrip):
        """Round-trip preserves values, structure, and attributes."""
        original, loaded = roundtrip()
        assert loaded.identical(original)

    def test_round_trip_no_drift(self, roundtrip):
        """Round-trip preserves dtype, dims, storage type, and coord dtypes."""
        original, loaded = roundtrip()
        assert loaded.dtype == original.dtype
        assert loaded.dims == original.dims
        assert type(loaded.data) == type(original.data)
        for k in original.coords:
            assert loaded.coords[k].dtype == original.coords[k].dtype

    def test_round_trip_with_time_dim(self, roundtrip):
        """Round-trip preserves an optional non-dimensional coordinate."""
        original, loaded = roundtrip(with_time=True)
        assert "time" in loaded.coords
        xr.testing.assert_identical(loaded, original)

    def test_assert_wrong_filetype(self, tmp_path):
        """serialize rejects non-.zarr paths."""
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        kind = LabeledArray(da)

        path = str(tmp_path / "test.csv")
        with pytest.raises(ValueError, match=r"\.zarr"):
            kind.serialize(path)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            LabeledArray.load("nonexistent.zarr")  # passes _normalize, fails at open

    def test_load_rejects_wrong_extension(self):
        with pytest.raises(ValueError, match=r"\.zarr"):
            LabeledArray.load("nonexistent.csv")
