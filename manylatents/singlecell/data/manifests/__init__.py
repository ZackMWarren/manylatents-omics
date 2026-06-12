"""Dataset manifest registry: the single source of truth for available datasets."""

from .registry import DatasetEntry, load_manifest, manifest_path, select_random

__all__ = ["DatasetEntry", "load_manifest", "manifest_path", "select_random"]
