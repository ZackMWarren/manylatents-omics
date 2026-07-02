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
        import sparse

        da = self.da
        # Dense arrays serialize natively; only sparse needs the 
        # COO-component format below (serialization doesn't work cleanly with sparse)
        if not isinstance(da.data, sparse.COO):
            da.to_zarr(path, mode="w")
            return

        coo = da.data
        ds = xr.Dataset(
            {
                "coo_coords": (("ndim", "nnz"), coo.coords),  
                "coo_data": (("nnz",), coo.data),             
            },
            attrs={
                "shape": list(coo.shape),
                "dims": list(da.dims),
                "fill_value": coo.fill_value.item(), 
                "name": da.name or "",
                "da_attrs": dict(da.attrs),
                # in conjunction with the for loop, ensures additional coords
                # stay alligned with their dim (i.e cell + time)
                "coord_dims": {name: list(c.dims) for name, c in da.coords.items()},
            },
        )
    
        for name, c in da.coords.items():
            ds = ds.assign_coords(
                {f"coord_{name}": (tuple(f"len_{d}" for d in c.dims), c.values)}
            )
        ds.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        path = cls._normalize(path)
        ds = xr.open_zarr(path)
        if "coo_data" not in ds:
            # Dense: reload as a DataArray to preserve its numpy backing.
            return cls(xr.open_dataarray(path, engine="zarr"))  # validate via __post_init__

        import sparse

        coo = sparse.COO(
            coords=ds["coo_coords"].values,
            data=ds["coo_data"].values,
            shape=tuple(ds.attrs["shape"]),
            fill_value=ds.attrs["fill_value"],
        )
        coord_dims = ds.attrs["coord_dims"]
        coords = {
            name: (tuple(coord_dims[name]), ds[f"coord_{name}"].values)
            for name in coord_dims
        }
        da = xr.DataArray(
            coo, dims=ds.attrs["dims"], coords=coords, name=ds.attrs["name"] or None
        )
        da.attrs = dict(ds.attrs.get("da_attrs", {}))
        return cls(da)  # validate called from __post_init__

    def __repr__(self) -> str:
        return f"LabeledArray(dims={list(self.da.dims)}, shape={self.da.shape})"
