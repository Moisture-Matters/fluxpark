"""Tests for the interception submodel (Voortman, 2015)."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402

from fluxpark.submodels.interception import interception_voortman  # noqa: E402


def test_water_is_conserved():
    # Over one step: rain == throughfall + change in store + evaporation.
    # This holds regardless of the storage/evaporation details, so it is a
    # robust invariant to pin.
    etp_wet = np.array([2.0, 0.0, 5.0], dtype=np.float32)
    rain = np.array([10.0, 0.0, 3.0], dtype=np.float32)
    max_int = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    cover = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    store_old = np.array([0.5, 1.0, 0.0], dtype=np.float32)

    int_evap, store, throughfall, _ = interception_voortman(
        etp_wet, rain, max_int, cover, store_old
    )
    balance = throughfall + (store - store_old) + int_evap
    assert np.allclose(balance, rain, atol=1e-5)


def test_no_cover_passes_all_rain():
    # With zero vegetation cover nothing is intercepted.
    etp_wet = np.array([3.0], dtype=np.float32)
    rain = np.array([5.0], dtype=np.float32)
    max_int = np.array([2.0], dtype=np.float32)
    cover = np.array([0.0], dtype=np.float32)
    store_old = np.array([0.0], dtype=np.float32)

    int_evap, _, throughfall, _ = interception_voortman(
        etp_wet, rain, max_int, cover, store_old
    )
    assert throughfall[0] == pytest.approx(5.0)
    assert int_evap[0] == pytest.approx(0.0)


def test_store_fills_to_capacity():
    # Full cover, no evaporation: the store fills to capacity and the rest
    # becomes throughfall.
    etp_wet = np.array([0.0], dtype=np.float32)
    rain = np.array([5.0], dtype=np.float32)
    max_int = np.array([2.0], dtype=np.float32)
    cover = np.array([1.0], dtype=np.float32)
    store_old = np.array([0.0], dtype=np.float32)

    int_evap, store, throughfall, _ = interception_voortman(
        etp_wet, rain, max_int, cover, store_old
    )
    assert store[0] == pytest.approx(2.0)        # capacity
    assert throughfall[0] == pytest.approx(3.0)  # 5 - 2
    assert int_evap[0] == pytest.approx(0.0)
