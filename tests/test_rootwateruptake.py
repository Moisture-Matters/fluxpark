"""Tests for the unsaturated-zone reservoir model bounds."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), which is provided via conda, not the
# pip requirements; skip the whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402

from fluxpark.submodels.rootwateruptake import unsat_reservoirmodel  # noqa: E402


def _arr(x):
    return np.array([x], dtype=np.float32)


def test_overfill_is_floored_to_zero():
    # high throughfall + high ET: smda would go negative -> floored to zero,
    # so soilm_root (= pwp - smda) stays within [0, pwp]. The model only
    # guarantees this non-negative deficit; masking lives in post-processing.
    scp, pwp = _arr(2.0), _arr(10.0)
    eta, smdp, smda, drain = unsat_reservoirmodel(
        _arr(15.0), _arr(20.0), _arr(0.0), scp, pwp
    )
    assert smda[0] == 0.0
    assert 0.0 <= smda[0] <= pwp[0]


def test_well_watered_matches_old_rules():
    # cond1 (smdp <= scp): eta == etp; drainage from smdp; smda unchanged.
    scp, pwp = _arr(5.0), _arr(20.0)
    eta, smdp, smda, drain = unsat_reservoirmodel(
        _arr(2.0), _arr(3.0), _arr(4.0), scp, pwp
    )
    assert eta[0] == pytest.approx(3.0)   # etp
    assert smda[0] == pytest.approx(5.0)  # 4 - 2 + 3
    assert drain[0] == pytest.approx(0.0)


def test_drainage_still_from_smdp():
    # very wet: smdp < 0 -> drainage = -smdp (unchanged old behaviour).
    scp, pwp = _arr(2.0), _arr(10.0)
    eta, smdp, smda, drain = unsat_reservoirmodel(
        _arr(8.0), _arr(1.0), _arr(0.0), scp, pwp
    )
    assert smdp[0] == pytest.approx(-7.0)
    assert drain[0] == pytest.approx(7.0)  # -smdp
    assert smda[0] == 0.0


def test_soilm_root_within_bounds_for_valid_pixels():
    rng = np.random.default_rng(0)
    n = 5000
    pwp = rng.uniform(20.0, 120.0, n).astype(np.float32)
    scp = (pwp * rng.uniform(0.1, 0.5, n)).astype(np.float32)
    smd_old = (pwp * rng.uniform(0.0, 1.0, n)).astype(np.float32)
    rain = rng.uniform(0.0, 60.0, n).astype(np.float32)
    etp = rng.uniform(0.0, 30.0, n).astype(np.float32)
    _, _, smda, _ = unsat_reservoirmodel(rain, etp, smd_old, scp, pwp)
    soilm_root = pwp - smda
    assert smda.min() >= 0.0
    assert soilm_root.min() >= -1e-4
    assert soilm_root.max() <= pwp.max() + 1e-4
