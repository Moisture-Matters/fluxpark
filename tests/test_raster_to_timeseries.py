"""Tests for the common-valid-mask aggregation (water balance eval)."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from osgeo import gdal, osr  # noqa: E402

from fluxpark.postprocessing.raster_to_timeseries import (  # noqa: E402
    rasters_to_timeseries,
)


def _write_tif(path, arr):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(28992)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(arr)
    ds = None


def _two_params(tmp_path):
    # param "a" valid everywhere, "b" nodata on 2 of 4 cells
    a = np.array([[10.0, 8.0], [8.0, 8.0]], dtype=np.float32)
    b = np.array([[3.0, -9999.0], [-9999.0, 5.0]], dtype=np.float32)
    _write_tif(tmp_path / "20180101-a.tif", a)
    _write_tif(tmp_path / "20180101-b.tif", b)
    return pd.DatetimeIndex(["2018-01-01"])


def test_common_valid_mask_shares_denominator(tmp_path):
    dates = _two_params(tmp_path)
    df = rasters_to_timeseries(
        tmp_path, ["a", "b"], dates, common_valid_mask=True
    )
    r = df.iloc[0]
    # both averaged over the 2 cells valid in BOTH params
    assert r["a"] == pytest.approx(9.0)            # (10 + 8) / 2
    assert r["b"] == pytest.approx(4.0)            # (3 + 5) / 2
    assert r["valid_fraction"] == pytest.approx(0.5)


def test_default_is_per_parameter_nanmean(tmp_path):
    dates = _two_params(tmp_path)
    df = rasters_to_timeseries(tmp_path, ["a", "b"], dates)
    r = df.iloc[0]
    assert r["a"] == pytest.approx(8.5)            # mean over all 4 cells
    assert r["b"] == pytest.approx(4.0)            # mean over its 2 valid cells
    assert "valid_fraction" not in df.columns
