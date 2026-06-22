"""Tests for the Makkink reference evapotranspiration."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402

from fluxpark.submodels.etref import makkink  # noqa: E402


def test_zero_radiation_gives_zero_et():
    assert makkink(15.0, 0.0) == pytest.approx(0.0)


def test_increases_with_radiation():
    low = makkink(15.0, 1000.0)
    high = makkink(15.0, 2000.0)
    assert 0.0 < low < high


def test_reference_value():
    # Regression value for a typical spring day (15 degC, 1000 J/cm2/day).
    assert makkink(15.0, 1000.0) == pytest.approx(1.651, abs=1e-3)


def test_array_input_is_elementwise():
    tair = np.array([15.0, 15.0], dtype=np.float32)
    rs_in = np.array([0.0, 1000.0], dtype=np.float32)
    out = makkink(tair, rs_in)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.651, abs=1e-2)
