"""Typed internal data representations (kinds).

Each kind owns its structural semantics and persistence. See ``kinds.py``.
"""

from .kinds import Kind, LabeledArray, SparseGraph

__all__ = ["Kind", "LabeledArray", "SparseGraph"]
