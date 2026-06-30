"""Tests for nodata / open-water masking in the daily post-processing."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402

from fluxpark.postprocessing.post_process import post_process_daily  # noqa: E402

# Soil parameters use NaN as the internal nodata value.
ND = np.nan


def _call(landuse, soilm_pwp, soilm_scp, prec_surplus, rain,
          open_water_evap, mask_open_water=True, open_water_ids=(16,)):
    n = landuse.shape[0]
    one = np.ones(n, dtype=np.float32)
    return post_process_daily(
        eta=one.copy(),
        trans_pot=one.copy(),
        soil_evap_act_est=one.copy(),
        int_evap=one.copy(),
        soil_evap_pot=one.copy(),
        open_water_evap_act=open_water_evap.copy(),
        smda=np.full(n, 5.0, np.float32),
        soilm_pwp=soilm_pwp.copy(),
        soilm_scp=soilm_scp.copy(),
        rain=rain.copy(),
        etref=one.copy(),
        landuse_map=landuse,
        prec_surplus=prec_surplus.copy(),
        open_water_ids=list(open_water_ids),
        mask_open_water=mask_open_water,
    )


def test_three_pixel_categories():
    # pixel 0 = normal land, 1 = open water (id 16), 2 = data gap (soil nodata)
    landuse = np.array([1, 16, 1], dtype=np.int32)
    pwp = np.array([50.0, ND, ND], dtype=np.float32)
    scp = np.array([10.0, ND, ND], dtype=np.float32)
    prec = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    rain = np.array([8.0, 8.0, 8.0], dtype=np.float32)
    ow_evap = np.array([0.0, 2.0, 0.0], dtype=np.float32)

    out = _call(landuse, pwp, scp, prec, rain, ow_evap, mask_open_water=True)

    sr, ps, et = out["soilm_root"], out["prec_surplus"], out["evap_total_act"]
    # normal pixel: valid
    assert sr[0] == pytest.approx(45.0)          # 50 - 5
    assert ps[0] == pytest.approx(3.0)
    # open water: soil fluxes NaN, prec_surplus NaN (masking), but evap_total
    # keeps the open-water evaporation
    assert np.isnan(sr[1]) and np.isnan(ps[1])
    assert et[1] == pytest.approx(2.0)
    # data gap: fully nodata
    assert np.isnan(sr[2]) and np.isnan(ps[2]) and np.isnan(et[2])


def test_masks_when_only_scp_is_nodata():
    # safety: a pixel where only one soil parameter is nodata is also masked
    landuse = np.array([1], dtype=np.int32)
    pwp = np.array([50.0], dtype=np.float32)
    scp = np.array([ND], dtype=np.float32)
    prec = np.array([3.0], dtype=np.float32)
    rain = np.array([8.0], dtype=np.float32)
    ow_evap = np.array([0.0], dtype=np.float32)

    out = _call(landuse, pwp, scp, prec, rain, ow_evap)
    assert np.isnan(out["soilm_root"][0])
    assert np.isnan(out["prec_surplus"][0])
