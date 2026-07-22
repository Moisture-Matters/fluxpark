"""Tests for validate_grid, including the None = 'skip this day' contract."""

import numpy as np
import pytest

import fluxpark as flp

SHAPE = (3, 4)


def _grid(fill=1.0):
    return np.full(SHAPE, fill, dtype="float32")


def test_none_always_skips_regardless_of_policy():
    # None means "no data for this day": it must skip under every policy and
    # never raise, so a provider can signal a skip while keeping nan_policy
    # strict on genuine NaN grids.
    for policy in ("error", "skip", "allow"):
        assert flp.utils.validate_grid(None, SHAPE, nan_policy=policy) is True


def test_clean_grid_never_skips():
    for policy in ("error", "skip", "allow"):
        assert (
            flp.utils.validate_grid(_grid(), SHAPE, nan_policy=policy) is False
        )


def test_nan_grid_follows_policy():
    grid = _grid()
    grid[0, 0] = np.nan
    assert flp.utils.validate_grid(grid, SHAPE, nan_policy="skip") is True
    assert flp.utils.validate_grid(grid, SHAPE, nan_policy="allow") is False
    with pytest.raises(ValueError):
        flp.utils.validate_grid(grid, SHAPE, nan_policy="error")


def test_shape_mismatch_raises_even_with_allow():
    with pytest.raises(ValueError):
        flp.utils.validate_grid(_grid(), (2, 2), nan_policy="allow")


def test_inf_always_raises():
    grid = _grid()
    grid[0, 0] = np.inf
    with pytest.raises(ValueError):
        flp.utils.validate_grid(grid, SHAPE, nan_policy="allow")
