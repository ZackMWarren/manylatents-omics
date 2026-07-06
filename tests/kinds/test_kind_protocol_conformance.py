"""LabeledArray satisfies geomancy's Kind protocol (provenance + require + tagged).

Asserts the surface + behavior *structurally*, with no ``geomancy`` import, so
this repo's CI catches drift on its own (keeps the dependency direction clean).
The matching contract test on the geomancy side lives in geomancy#26.
"""
import tempfile
from pathlib import Path

import numpy as np
import sparse
import xarray as xr

from manylatents.kinds import LabeledArray


def _toy() -> LabeledArray:
    return LabeledArray(
        xr.DataArray(
            np.ones((3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
    )


def _toy_sparse() -> LabeledArray:
    """Same as ``_toy`` but backed by a pydata ``sparse.COO`` (production path)."""
    return LabeledArray(
        xr.DataArray(
            sparse.COO.from_numpy(np.eye(3, 2)),
            dims=["cell", "gene"],
            coords={"cell": ["c1", "c2", "c3"], "gene": ["g1", "g2"]},
        )
    )


class TestKindProtocolConformance:
    def test_offers_all_three_protocol_members(self):
        la = _toy()
        assert isinstance(la.provenance, tuple)
        assert callable(la.require) and callable(la.tagged)

    def test_default_provenance_is_empty(self):
        assert _toy().provenance == ()

    def test_provenance_normalized_to_tuple(self):
        la = LabeledArray(_toy().da, provenance=["a", "b"])
        assert la.provenance == ("a", "b")

    def test_tagged_appends_immutably(self):
        la = _toy()
        out = la.tagged("mean_over_time")
        assert out.provenance == ("mean_over_time",)
        assert la.provenance == ()  # original untouched
        assert isinstance(out, LabeledArray)

    def test_tagged_accumulates(self):
        assert _toy().tagged("a").tagged("b").provenance == ("a", "b")

    def test_require_still_returns_self_for_chaining(self):
        la = _toy()
        assert la.require("cell", "gene") is la

    def test_provenance_survives_zarr_round_trip(self):
        tagged = _toy().tagged("mean_over_time").tagged("velocity")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "k.zarr")
            tagged.serialize(path)
            loaded = LabeledArray.load(path)
            assert loaded.provenance == ("mean_over_time", "velocity")
            assert "provenance" not in loaded.da.attrs

    def test_provenance_survives_sparse_round_trip(self):
        tagged = _toy_sparse().tagged("mean_over_time").tagged("velocity")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "k.zarr")
            tagged.serialize(path)
            loaded = LabeledArray.load(path)
            assert loaded.provenance == ("mean_over_time", "velocity")
            assert isinstance(loaded.da.data, sparse.COO)
            assert "provenance" not in loaded.da.attrs

    def test_empty_provenance_adds_no_attr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "k.zarr")
            _toy().serialize(path)
            loaded = LabeledArray.load(path)
            assert loaded.provenance == () and "provenance" not in loaded.da.attrs
