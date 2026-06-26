"""Tests for the abstract ``Kind`` base class."""

import pytest

from manylatents.kinds import Kind


def test_kind_is_abstract():
    """The abstract base cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Kind()  # type: ignore[abstract]
