"""
SparseGraph: a graph as two plain numpy arrays (edge list + node ids).
"""

import logging
from dataclasses import dataclass

import numpy as np

from .base import Kind

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class SparseGraph(Kind):
    """2 np arrays: edge list and node ids"""

    edges: np.ndarray
    node_ids: np.ndarray

    def __post_init__(self):
        # frozen: bypass the immutability guard to normalize inputs in place.
        object.__setattr__(self, "edges", np.asarray(self.edges))
        object.__setattr__(self, "node_ids", np.asarray(self.node_ids))
        self.validate()

    def validate(self) -> "SparseGraph":
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be E×2, got shape {self.edges.shape}")
        if self.node_ids.ndim != 1:
            raise ValueError(f"node_ids must be 1-D, got shape {self.node_ids.shape}")
        if not np.issubdtype(self.edges.dtype, np.integer):
            raise ValueError(f"edges must be integer dtype, got {self.edges.dtype}")
        return self

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".npz"):
            raise ValueError(f"path must end in .npz, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        logger.info(f"Serializing {type(self).__name__} to {path}")
        np.savez_compressed(self._normalize(path), edges=self.edges, node_ids=self.node_ids)

    @classmethod
    def load(cls, path):

        with np.load(cls._normalize(path)) as d:
            return cls(d['edges'], d['node_ids']) # validate called from __post_init__

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.edges, self.node_ids

    def __repr__(self) -> str:
        return f"SparseGraph(num_nodes={self.node_ids.shape[0]}, num_edges={self.edges.shape[0]})"
