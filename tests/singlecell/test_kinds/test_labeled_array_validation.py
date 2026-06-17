import numpy as np
import pytest
import xarray as xr
from manylatents.singlecell.data.kinds.kinds import LabeledArray

@pytest.fixture
def da():
    return xr.DataArray(
        np.ones((3, 2)),
        dims=["cell", "gene"],
        coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
    )

class TestLabeledArrayValidation:
    def test_validate_passes_and_chains(self, da):
        kind = LabeledArray(da)
        assert kind.validate() is kind

    def test_fails_on_non_dataarray(self):
        with pytest.raises(ValueError, match="must wrap a DataArray"):
            LabeledArray([1, 2, 3])

    def test_fails_on_empty(self):
        empty = xr.DataArray(np.empty((0, 0)), dims=["cell", "gene"])
        with pytest.raises(ValueError, match="empty"):
            LabeledArray(empty)
