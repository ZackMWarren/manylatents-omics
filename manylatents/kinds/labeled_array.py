"""
LabeledArray: an xarray DataArray with named dimensions.
"""

import logging
from dataclasses import dataclass

import xarray as xr

from .base import Kind

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class LabeledArray(Kind):
    """xarray DataArray with named dimensions."""

    da: xr.DataArray

    def __post_init__(self):
        self.validate()

    def validate(self) -> "LabeledArray":
        if not isinstance(self.da, xr.DataArray):
            raise ValueError("LabeledArray must wrap a DataArray")
        if self.da.size == 0:
            raise ValueError(f"LabeledArray is empty (shape {self.da.shape})")
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "LabeledArray":
        missing_dims = [d for d in dims if d not in self.da.dims]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {tuple(self.da.dims)}")

        # Code can be removed if time is decided to be a dim rather than a coord
        missing_coords = [c for c in coords if c not in self.da.coords]
        if missing_coords:
            raise ValueError(f"requires coords {missing_coords}; got {tuple(self.da.coords)}")
        return self


    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".zarr"):
            raise ValueError(f"path must end in .zarr, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        path = self._normalize(path)
        logger.info(f"Serializing {type(self).__name__} to {path}")
        self.da.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        path = cls._normalize(path)
        da = xr.open_dataarray(path, engine="zarr")
        return cls(da) # validate called from __post_init__

    def __repr__(self) -> str:
        return f"LabeledArray(dims={list(self.da.dims)}, shape={self.da.shape})"
