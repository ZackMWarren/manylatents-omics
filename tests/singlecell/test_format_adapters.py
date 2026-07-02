"""Tests for the AnnData -> LabeledArray format adapter (``from_anndata``)."""
import numpy as np
import pytest
import scipy.sparse as sp
import sparse

ad = pytest.importorskip("anndata")
import pandas as pd

from manylatents.kinds import LabeledArray
from manylatents.singlecell.data.adapters.formats.adapters import from_anndata


N_CELLS, N_GENES = 6, 4


def _make_adata(X):
    """Wrap an expression matrix in an AnnData with simple cell/gene names."""
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )


def _coords(adata):
    return {"cell": list(adata.obs_names), "gene": list(adata.var_names)}


@pytest.fixture
def dense_X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_CELLS, N_GENES)).astype(np.float32)


# --- basic conversion -------------------------------------------------------


def test_returns_labeled_array(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata))
    assert isinstance(kind, LabeledArray)


def test_dims_and_shape(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata))
    assert kind.da.dims == ("cell", "gene")
    assert kind.da.shape == (N_CELLS, N_GENES)


def test_coords_preserved(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata))
    assert list(kind.da.coords["cell"].values) == [f"c{i}" for i in range(N_CELLS)]
    assert list(kind.da.coords["gene"].values) == [f"g{i}" for i in range(N_GENES)]


def test_values_preserved(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata))
    assert np.allclose(kind.da.data.todense(), dense_X)


def test_backed_by_coo_duck_array(dense_X):
    """Adapter always yields a pydata sparse.COO backing store (xarray-safe)."""
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata))
    assert isinstance(kind.da.data, sparse.COO)


# --- sparse input -----------------------------------------------------------


def test_sparse_input_preserved():
    rng = np.random.default_rng(1)
    dense = (rng.random((N_CELLS, N_GENES)) < 0.3).astype(np.float32)
    X = sp.csr_matrix(dense)
    adata = _make_adata(X)
    kind = from_anndata(adata, _coords(adata))
    assert isinstance(kind.da.data, sparse.COO)
    assert np.allclose(kind.da.data.todense(), dense)


# --- metadata ---------------------------------------------------------------


def test_metadata_becomes_attrs(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata), metadata={"genome": "GRCh38"})
    assert kind.da.attrs["genome"] == "GRCh38"


def test_no_metadata_yields_empty_attrs(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata))
    assert kind.da.attrs == {}


# --- matrix selection precedence -------------------------------------------


def test_layer_selected_when_present(dense_X):
    adata = _make_adata(dense_X)
    adata.layers["counts"] = dense_X * 2
    kind = from_anndata(adata, _coords(adata), layer="counts")
    assert np.allclose(kind.da.data.todense(), dense_X * 2)


def test_missing_layer_falls_back_to_X(dense_X):
    adata = _make_adata(dense_X)
    kind = from_anndata(adata, _coords(adata), layer="nonexistent")
    assert np.allclose(kind.da.data.todense(), dense_X)


def test_use_raw_selects_raw(dense_X):
    adata = _make_adata(dense_X)
    raw_X = dense_X * 3
    raw = _make_adata(raw_X)
    adata.raw = raw
    kind = from_anndata(adata, _coords(adata), use_raw=True)
    assert np.allclose(kind.da.data.todense(), raw_X)


def test_use_raw_falls_back_to_X_when_no_raw(dense_X):
    adata = _make_adata(dense_X)
    assert adata.raw is None
    kind = from_anndata(adata, _coords(adata), use_raw=True)
    assert np.allclose(kind.da.data.todense(), dense_X)


def test_raw_takes_precedence_over_layer(dense_X):
    """use_raw wins over a provided layer (raw is checked first)."""
    adata = _make_adata(dense_X)
    adata.layers["counts"] = dense_X * 2
    adata.raw = _make_adata(dense_X * 3)
    kind = from_anndata(adata, _coords(adata), use_raw=True, layer="counts")
    assert np.allclose(kind.da.data.todense(), dense_X * 3)
