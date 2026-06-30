"""Tests for nodata handling in GeoTiffReader.read_and_reproject.

The model reads float rasters with ``dst_nodata=np.nan`` so nodata comes back
as NaN, which never collides with a valid 0 (e.g. 0% impervious). The sentinel
behaviour is still supported and tested: with a destination nodata of 0 GDAL
bumps a genuine averaged 0 to 1, which is why a sentinel reader must use an
out-of-range value (the customer-prep script writes such files).
"""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402
from osgeo import gdal, osr  # noqa: E402

from fluxpark.io.raster_tools import GeoTiffReader  # noqa: E402

# Fine raster: the top-left 2x2 block is all zero, the rest is 100. Downsampled
# 2x with "average", the top-left output cell averages to exactly 0.
_FINE = np.array(
    [[0, 0, 100, 100],
     [0, 0, 100, 100],
     [100, 100, 100, 100],
     [100, 100, 100, 100]],
    dtype=np.uint8,
)
_GRID = dict(dst_epsg=28992, bounds=(0, 4, 0, 4), cellsize=2)


def _write_source(path):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), 4, 4, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, 4, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(28992)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(_FINE)
    ds = None


def test_out_of_range_nodata_keeps_valid_zero(tmp_path):
    path = tmp_path / "imperv.tif"
    _write_source(path)
    out = GeoTiffReader(str(path), dst_nodata=255).read_and_reproject(
        **_GRID, resample_alg="average", src_nodata="None"
    )
    assert out[0, 0] == 0.0          # genuine 0% stays 0


def test_colliding_nodata_bumps_valid_zero(tmp_path):
    # documents the GDAL gotcha that motivates the 255 nodata above
    path = tmp_path / "imperv.tif"
    _write_source(path)
    out = GeoTiffReader(str(path), dst_nodata=0).read_and_reproject(
        **_GRID, resample_alg="average", src_nodata="None"
    )
    assert out[0, 0] == 1.0          # valid 0 bumped to 1


def test_nan_nodata_keeps_valid_zero_and_returns_nan(tmp_path):
    # the model's path: NaN nodata keeps a genuine 0 and never collides with it
    path = tmp_path / "imperv.tif"
    _write_source(path)
    out = GeoTiffReader(str(path), dst_nodata=np.nan).read_and_reproject(
        **_GRID, resample_alg="average", src_nodata="None"
    )
    assert out.dtype == np.float32
    assert out[0, 0] == 0.0          # genuine 0% stays 0, not bumped

    # honour the source nodata (0): a fully-nodata cell comes back as NaN
    out2 = GeoTiffReader(str(path), dst_nodata=np.nan).read_and_reproject(
        **_GRID, resample_alg="average"
    )
    assert np.isnan(out2[0, 0])      # the all-zero (nodata) block -> NaN
