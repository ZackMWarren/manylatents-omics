import tempfile
from pathlib import Path
import pytest
import numpy as np
from manylatents.singlecell.data.kinds.kinds import SparseGraph

class TestSparseGraphRoundTrip:
    def test_round_trip_basic(self):
        """Round-trip with minimal data preserves values and structure."""
        rng = np.random.default_rng()
        edge_list = rng.integers(size = (10,2), low = 0, high = 11)
        node_ids = rng.random(size = 11)
        kind = SparseGraph(edge_list, node_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.npz")
            kind.serialize(path)
            loaded = SparseGraph.load(path)
            
            loaded_edges, loaded_ids = loaded.data
            np.testing.assert_array_equal(loaded_edges, edge_list)
            np.testing.assert_array_equal(loaded_ids, node_ids)
    
    def test_assert_wrong_filetype(self):
        """Round-trip with minimal data preserves values and structure."""
        rng = np.random.default_rng()
        edge_list = rng.integers(size = (10,2), low = 0, high = 11)
        node_ids = rng.random(size = 11)
        kind = SparseGraph(edge_list, node_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.csv")
            with pytest.raises(ValueError, match=r"\.npz"):
                kind.serialize(path)
                
    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            SparseGraph.load("nonexistent.npz")   # passes _normalize, fails at np.load
            
    def test_load_rejects_wrong_extension(self):
        with pytest.raises(ValueError, match=r"\.npz"):
            SparseGraph.load("nonexistent.csv")
            