"""Typed internal data representations (kinds).

Each kind owns its structural semantics and persistence. The base class lives in
``base.py``; each concrete kind has its own module (``labeled_array.py``,
``sparse_graph.py``).
"""

from .base import Kind
from .labeled_array import LabeledArray
from .sparse_graph import SparseGraph

__all__ = ["Kind", "LabeledArray", "SparseGraph"]
