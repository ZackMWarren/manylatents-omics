"""Op contracts: requiring dimensions and coordinates via ``require``.

Required dimensions/coordinates are enforced structurally by ``require`` — the
contract an op declares for the kind it consumes.
"""

import numpy as np
import pytest
import xarray as xr

from manylatents.singlecell.data.kinds.kinds import LabeledArray


def labeled_da(dims, coords):
    """Build a ``DataArray`` of ones sized to ``coords`` along ``dims``."""
    return xr.DataArray(
        np.ones(tuple(len(coords[d]) for d in dims)),
        dims=dims,
        coords=coords,
    )


class TestRequireContract:
    """Ops declare and enforce the dims/coords they consume via ``kind.require``."""

    def test_require_passes_with_present_dims(self):
        da = labeled_da(
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
        da = labeled_da(
            ["samples", "features"],  # wrong names
            {"samples": list(range(10)), "features": list(range(5))},
        )
        with pytest.raises(ValueError, match="requires dims"):
            LabeledArray(da).require("cell", "gene")

    def test_require_rejects_missing_coord(self):
        da = labeled_da(
            ["cell", "gene"],
            {"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
        with pytest.raises(ValueError, match="requires coords"):
            LabeledArray(da).require("cell", "gene", coords=("gene_ids",))

    def test_temporal_op_requires_time_dim(self):
        """An op that needs a ``time`` dim accepts a kind that carries one."""
        da = labeled_da(
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
        da = labeled_da(
            ["cell", "gene"],
            {"cell": [f"c{i}" for i in range(10)], "gene": [f"g{i}" for i in range(5)]},
        )
        with pytest.raises(ValueError, match="requires dims"):
            LabeledArray(da).require("cell", "gene", "time")
