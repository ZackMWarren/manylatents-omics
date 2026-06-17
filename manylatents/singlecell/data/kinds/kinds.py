"""
Typed internal data representations (kinds).

Each kind carries its own structural semantics (dims, labels, coords).
This ensures ops can read and validate structure instead of guessing.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger(__name__)


class Kind(ABC):
    """Base class for all data kinds.

    Each kind owns its own structural semantics *and* its own persistence:
    subclasses must implement ``validate``, ``serialize``, and ``load`` for the
    storage format appropriate to that kind. The base declares only the contract.
    """

    @abstractmethod
    def validate(self) -> "Kind":
        """Validate the kind's structure. Raise on failure."""
        ...

    @abstractmethod
    def serialize(self, path: str) -> None:
        """Write the kind to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Kind":
        """Load the kind from disk, validating on read."""
        ...


class LabeledArray(Kind):
    """xarray DataArray with named dimensions."""
    
    def __init__(self, da: xr.DataArray): 
        self._da = da
        self.validate()
        
    def validate(self) -> "LabeledArray":
        if not isinstance(self._da, xr.DataArray):
            raise ValueError("LabeledArray must wrap a DataArray")
        if self._da.size == 0:
            raise ValueError(f"LabeledArray is empty (shape {self._da.shape})")
        return self
    
    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "LabeledArray":
        missing_dims = [d for d in dims if d not in self._da.dims]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {tuple(self._da.dims)}")
        
        # Code can be removed if time is decided to be a dim rather than a coord
        missing_coords = [c for c in coords if c not in self._da.coords]
        if missing_coords:
            raise ValueError(f"requires coords {missing_coords}; got {tuple(self._da.coords)}")
        return self


    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".zarr"):
            raise ValueError(f"path must end in .zarr, got {path!r}")
        return str(path)
    
    def serialize(self, path: str) -> None:
        path = self._normalize(path)
        logger.info(f"Serializing {type(self).__name__} to {path}")
        self._da.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        path = cls._normalize(path)
        da = xr.open_dataarray(path, engine="zarr")
        return cls(da) # validate called from class creation

    @property
    def da(self) -> xr.DataArray:
        return self._da
    
    def __repr__(self) -> str:
        return f"LabeledArray(dims={list(self._da.dims)}, shape={self._da.shape})"


class SparseGraph(Kind):
    """2 np arrays: edge list and node ids"""
    def __init__(self, edges: np.ndarray, node_ids: np.ndarray):
        self._edges = np.asarray(edges)        
        self._node_ids = np.asarray(node_ids)
        self.validate()

    def validate(self) -> "SparseGraph":
        if self._edges.ndim != 2 or self._edges.shape[1] != 2:
            raise ValueError(f"edges must be E×2, got shape {self._edges.shape}")
        if self._node_ids.ndim != 1:
            raise ValueError(f"node_ids must be 1-D, got shape {self._node_ids.shape}")
        if not np.issubdtype(self._edges.dtype, np.integer):
            raise ValueError(f"edges must be integer dtype, got {self._edges.dtype}")
        return self
    
    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".npz"):
            raise ValueError(f"path must end in .npz, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        logger.info(f"Serializing {type(self).__name__} to {path}")
        np.savez_compressed(self._normalize(path), edges=self._edges, node_ids=self._node_ids)

    @classmethod
    def load(cls, path):
        
        with np.load(cls._normalize(path)) as d:
            return cls(d['edges'], d['node_ids']) # validate called from class creation

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._edges, self._node_ids

    def __repr__(self) -> str:
        return f"SparseGraph(num_nodes={self._node_ids.shape[0]}, num_edges={self._edges.shape[0]})"