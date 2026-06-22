"""Tests for the Boesten-Stroosnijder soil evaporation submodel."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402

from fluxpark.submodels.soilevaporation import (  # noqa: E402
    soilevap_boestenstroosnijder,
)


def test_wet_soil_evaporates_at_potential():
    # Excess rain (throughfall > epot): actual equals potential evaporation.
    ea, _, _ = soilevap_boestenstroosnijder(
        np.array([10.0]), np.array([2.0]), np.array([0.5]),
        np.array([0.0]), np.array([0.0]),
    )
    assert ea[0] == pytest.approx(2.0)


def test_dry_soil_is_supply_limited():
    # Dry spell with a large accumulated deficit: actual drops below potential.
    ea, _, _ = soilevap_boestenstroosnijder(
        np.array([0.0]), np.array([4.0]), np.array([0.05]),
        np.array([100.0]), np.array([15.0]),
    )
    assert 0.0 <= ea[0] < 4.0


def test_evaporation_is_non_negative():
    rng = np.random.default_rng(0)
    n = 1000
    tf = rng.uniform(0.0, 10.0, n)
    ep = rng.uniform(0.0, 6.0, n)
    beta = np.full(n, 0.5)
    sum_ep = rng.uniform(0.0, 50.0, n)
    sum_ea = rng.uniform(0.0, 50.0, n)
    ea, _, _ = soilevap_boestenstroosnijder(tf, ep, beta, sum_ep, sum_ea)
    assert ea.min() >= 0.0
