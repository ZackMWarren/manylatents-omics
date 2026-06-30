import logging
from typing import Optional
import numpy as np
import xarray as xr
import scipy.sparse as sp
import sparse
from manylatents.kinds import LabeledArray
import pandas as pd

logger = logging.getLogger(__name__)

def from_anndata(
    adata,
    coords: dict,
    metadata: Optional[dict] = None,
    use_raw: bool = False,
    layer: Optional[str] = None,
) -> LabeledArray:
    """
    Convert AnnData object to typed LabeledArray kind.

    The expression matrix is selected with the same precedence as AnnDataset:
    ``adata.raw.X`` (if ``use_raw``) → ``adata.layers[layer]`` (if ``layer``
    is given and present) → ``adata.X``.
    """
        
    if use_raw and adata.raw is not None:
        X = adata.raw.X
    elif layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # Convert to ``sparse.COO`` duck array (needed for xarray)
    if sp.issparse(X):
        data = sparse.COO.from_scipy_sparse(X.tocsr())
    else:
        data = sparse.COO.from_numpy(np.asarray(X))
    
    # Transposed to gene x samples
    data_T = data.T
        
    da = xr.DataArray(
        data_T,
        dims=["gene", "samples"],
        coords=coords,
        attrs=metadata or {},
    )

    kind = LabeledArray(da)

    logger.info(
        f"Successfully converted to LabeledArray: "
        f"shape={kind.da.shape}, dims={list(kind.da.dims)}"
    )

    return kind

def from_df(
    df: pd.DataFrame,
    metadata: Optional[dict] = None,
) -> LabeledArray:
    """
    Convert a bulk expression matrix to a typed LabeledArray kind.

    ``df`` is genes × samples (rows indexed by gene id, columns by sample
    id). ``metadata`` is attached as DataArray attributes, the same as
    :func:`from_anndata`.

    The frame must declare its orientation via axis names
    (``index.name == "gene"``, ``columns.name == "sample"``) and carry real
    labels — this requirement is strict. Any df MUST convert their index to "gene"
    and column names to "sample"
    """
    if df.index.name != "gene" or df.columns.name != "sample":
        raise ValueError(
            "from_bulk expects a genes × samples frame with "
            "index.name='gene' and columns.name='sample'; got "
            f"index.name={df.index.name!r}, columns.name={df.columns.name!r}"
        )
    if isinstance(df.index, pd.RangeIndex):
        raise ValueError("gene ids missing: df.index is a default RangeIndex")
    if isinstance(df.columns, pd.RangeIndex):
        raise ValueError("sample ids missing: df.columns is a default RangeIndex")

    gene_ids = df.index.to_list()
    sample_ids = df.columns.to_list()

    # Convert to ``sparse.COO`` duck array (needed for xarray)
    data = sparse.COO.from_numpy(np.asarray(df.to_numpy()))

    da = xr.DataArray(
        data,
        dims=["gene", "sample"],
        coords={"gene": gene_ids, "sample": sample_ids},
        attrs=metadata or {},
    )

    kind = LabeledArray(da)
    logger.info(
        f"Successfully converted to LabeledArray: "
        f"shape={kind.da.shape}, dims={list(kind.da.dims)}"
    )

    return kind

    