"""Example ops: the shipped demonstration ops stay in sync with the kind API."""

import numpy as np
import pytest
import xarray as xr

from manylatents.kinds import LabeledArray


def labeled_da(dims, coords):
    """Build a ``DataArray`` of ones sized to ``coords`` along ``dims``."""
    return xr.DataArray(
        np.ones(tuple(len(coords[d]) for d in dims)),
        dims=dims,
        coords=coords,
    )


class TestExampleOps:

    def test_temporal_analysis_requires_time(self):
        from tests.singlecell.test_op.example_ops import temporal_analysis

        no_time = LabeledArray(labeled_da(
            ["cell", "gene"],
            {"cell": ["c0", "c1"], "gene": ["g0", "g1", "g2"]},
        ))
        with pytest.raises(ValueError, match="requires dims"):
            temporal_analysis(no_time)

    def test_temporal_analysis_runs_with_time(self):
        from tests.singlecell.test_op.example_ops import temporal_analysis

        with_time = LabeledArray(labeled_da(
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
