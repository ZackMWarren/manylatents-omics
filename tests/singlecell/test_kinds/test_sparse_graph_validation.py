import numpy as np
import pytest
from manylatents.singlecell.data.kinds.kinds import SparseGraph

@pytest.fixture
def rng():
    return np.random.default_rng(0)  

class TestSparseGraphValidation:
    def test_validate_passes_and_chains(self, rng):
        edges = rng.integers(size=(10, 2), low=0, high=11)
        node_ids = rng.integers(size=11, low=0, high=11)
        g = SparseGraph(edges, node_ids)
        assert g.validate() is g            

    def test_fails_on_edge_transpose(self, rng):
        edges = rng.integers(size=(2, 10), low=0, high=11)
        node_ids = rng.integers(size=11, low=0, high=11)
        with pytest.raises(ValueError, match="E×2"):
            SparseGraph(edges, node_ids)

    def test_fails_on_edge_three_cols(self, rng):
        edges = rng.integers(size=(10, 3), low=0, high=11)
        node_ids = rng.integers(size=11, low=0, high=11)
        with pytest.raises(ValueError, match="E×2"):
            SparseGraph(edges, node_ids)

    def test_fails_on_float_edges(self, rng):
        edges = rng.random((10, 2))         
        node_ids = rng.integers(size=11, low=0, high=11)
        with pytest.raises(ValueError, match="integer"):
            SparseGraph(edges, node_ids)

    def test_fails_on_2d_node_ids(self, rng):
        edges = rng.integers(size=(10, 2), low=0, high=11)   
        node_ids = rng.integers(size=(11, 2), low=0, high=11)
        with pytest.raises(ValueError, match="1-D"):
            SparseGraph(edges, node_ids)