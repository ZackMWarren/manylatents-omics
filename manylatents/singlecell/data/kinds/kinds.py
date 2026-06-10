"""
Typed internal data representations (kinds).

Each kind carries its own structural semantics (dims, labels, coords).
This ensures ops can read and validate structure instead of guessing.
"""

import logging
from abc import ABC, abstractmethod

import xarray as xr

logger = logging.getLogger(__name__)


class Kind(ABC):
    """Base class for all data kinds.

    Each kind owns its own structural semantics *and* its own persistence:
    subclasses must implement ``validate``, ``serialize``, and ``load`` for the
    storage format appropriate to that kind. The base declares only the contract.
    """

    @abstractmethod
    def validate(self) -> None:
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
    """xarray DataArray with named dimensions.

    Required dims are declared by the adapter at ingestion and passed in; the
    kind only enforces whatever it was handed.
    """

    def __init__(self, da: xr.DataArray, required_dims: set[str] | None = None, required_coords: set[str] | None = None):
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"Expected xr.DataArray, got {type(da)}")
        self._da = da
        self.required_dims = required_dims or set()
        self.required_coords = required_coords or set()
        
    def validate(self) -> None:
        missing = self.required_dims - set(self._da.dims)
        if missing:
            raise ValueError(
                f"LabeledArray missing required dims: {missing}. "
                f"Present dims: {list(self._da.dims)}"
            )

        missing_coords = self.required_coords - set(self._da.coords)
        if missing_coords:
            raise ValueError(
                f"LabeledArray missing coordinates for dims: {missing_coords}. "
                f"Present coords: {list(self._da.coords)}"
            )

        logger.debug(
            f"LabeledArray validated: dims={list(self._da.dims)}, "
            f"coords={list(self._da.coords)}"
        )
        
        # Check for nulls; indicates metadata loss
        for coord in self.required_coords:
            if coord in self._da.coords:
                null_count = self._da.coords[coord].isnull().sum().item()
                if null_count:
                    raise ValueError(
                        f"Coord '{coord}' has {null_count} null values across cells."
                    )

    def serialize(self, path: str) -> None:
        """Write the underlying DataArray to disk as zarr."""
        logger.info(f"Serializing {type(self).__name__} to {path}")
        self._da.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path: str) -> "LabeledArray":
        """Load a zarr DataArray from disk and validate on read."""
        logger.info(f"Loading {cls.__name__} from {path}")
        da = xr.open_dataarray(path, engine="zarr")
        obj = cls(da)
        obj.validate()
        return obj

    @property
    def data(self) -> xr.DataArray:
        """Access the underlying DataArray."""
        return self._da

    def __repr__(self) -> str:
        return f"LabeledArray(dims={list(self._da.dims)}, shape={self._da.shape})"

class test:
    pass