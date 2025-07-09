from osgeo import gdal
import numpy as np
from pathlib import Path
from typing import Union, Optional
from scipy.interpolate import Rbf
import fluxpark as flp


def interpolate_rain(
    x,
    y,
    point_values,
    dst_epsg,
    bounds,
    cellsize,
    radius=50000.0,
    max_points=3,
    coarse_cellsize=2500.0,
    intermediate_path: Optional[Union[str, Path]] = None,
):
    """
    Interpolate point rainfall values to a raster grid using gdal.Grid,
    then resample to coarser grid and smooth back to target resolution.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points.
    values : array-like
        Rainfall values.
    dst_epsg : int
        EPSG code of output projection.
    bounds : tuple
        (x_min, x_max, y_min, y_max) in output CRS.
    cellsize : float
        Final output resolution (meters).
    radius : float
        Search radius for IDW interpolation.
    max_points : int
        Max number of neighbors in interpolation.
    coarse_cellsize : float
        Intermediate resolution (default: 2500m) to speed up interpolation.
    intermediate_path : str or Path, optional
        Full filename (including directory) where intermediate outputs
        will be saved (e.g. a GeoPackage or VRT).  If the directory
        doesn’t exist, it will be created. If None (the default),
        no intermediate files are written.

    Returns
    -------
    np.ndarray
        Interpolated raster as NumPy array (np.nan = no data).
    """
    x_min, x_max, y_min, y_max = bounds

    # 1. create a point data set.
    ds_point = flp.io.write_point_layer(x, y, point_values, dst_epsg, intermediate_path)

    # 2. Interpolate to coarse grid using IDW
    ncols = int((x_max - x_min) / coarse_cellsize) + 1
    nrows = int((y_max - y_min) / coarse_cellsize) + 1
    x_max_snap = x_min + ncols * coarse_cellsize
    y_min_snap = y_max - nrows * coarse_cellsize

    ds_initial = gdal.Grid(
        "",
        ds_point,
        format="MEM",
        outputBounds=[x_min, y_max, x_max_snap, y_min_snap],
        width=ncols,
        height=nrows,
        outputType=gdal.GDT_Float32,
        algorithm=(
            f"invdistnn:power=2.0:smoothing=0.0:"
            f"radius={radius}:max_points={max_points}:"
            f"min_points=0:nodata=-9999"
        ),
        zfield="z",
    )

    # 3. Warp to coarser grid with averaging
    x_max_coarse = (
        x_min + (int((x_max - x_min) / coarse_cellsize) + 1) * coarse_cellsize
    )
    y_min_coarse = (
        y_max - (int((y_max - y_min) / coarse_cellsize) + 1) * coarse_cellsize
    )

    ds_coarse = gdal.Warp(
        "",
        ds_initial,
        dstSRS=f"EPSG:{dst_epsg}",
        format="VRT",
        resampleAlg=gdal.GRA_Average,
        xRes=coarse_cellsize,
        yRes=-coarse_cellsize,
        outputBounds=[x_min, y_min_coarse, x_max_coarse, y_max],
        outputBoundsSRS=f"EPSG:{dst_epsg}",
        dstNodata=-9999,
    )

    # 4. Warp back to fine resolution with smoothing
    ds_fine = gdal.Warp(
        "",
        ds_coarse,
        dstSRS=f"EPSG:{dst_epsg}",
        format="VRT",
        resampleAlg=gdal.GRA_CubicSpline,
        xRes=cellsize,
        yRes=-cellsize,
        outputBounds=[x_min, y_min, x_max, y_max],
        outputBoundsSRS=f"EPSG:{dst_epsg}",
        dstNodata=-9999,
    )

    # 5. Read array and clean up
    rain = ds_fine.ReadAsArray().astype(np.float32)
    rain[rain == -9999] = np.nan
    rain[rain < 0.1] = 0.0

    # Clean memory
    ds_point = None
    ds_initial = None
    ds_coarse = None
    ds_fine = None

    return rain


def interpolate_makkink(
    x: np.ndarray,
    y: np.ndarray,
    point_values: np.ndarray,
    dst_epsg: int,
    bounds: tuple[float, float, float, float],
    cellsize: float,
    coarse_cellsize: float = 2500.0,
    intermediate_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Interpolate Makkink reference ET to a raster using
    a thin-plate spline RBF on a coarse grid + GDAL Warp to
    resample back to full resolution.

    Parameters
    ----------
    x, y : array-like
        Coordinates of observation points (in output CRS).
    mak_values : array-like
        Makkink ET values at those points.
    dst_epsg : int
        EPSG code of output projection.
    bounds : (x_min, x_max, y_min, y_max)
        Target extent.
    cellsize : float
        Final output resolution (meters).
    coarse_cellsize : float, optional
        Resolution of the intermediate grid (default: 2500 m).
    intermediate_path : str or Path, optional
        If provided, writes your **input points** (with field "mak")
        to the given filename (GeoPackage, Shapefile, etc.).  No
        intermediate grids are written. Parent directory is created
        if needed. If None, no files are written.
    Returns
    -------
    np.ndarray
        Interpolated raster (dtype float32).
    """
    x_min, x_max, y_min, y_max = bounds

    # 1. Build the RBF interpolator
    rbf = Rbf(x, y, point_values, function="thin_plate", smooth=0)

    # 2. Define & populate coarse grid
    nx = int(np.ceil((x_max - x_min) / coarse_cellsize)) + 1
    ny = int(np.ceil((y_max - y_min) / coarse_cellsize)) + 1
    x_max_snap = x_min + (nx - 1) * coarse_cellsize
    y_min_snap = y_max - (ny - 1) * coarse_cellsize

    xs = np.linspace(x_min, x_max_snap, nx)
    ys = np.linspace(y_max, y_min_snap, ny)  # descending
    xi, yi = np.meshgrid(xs, ys)
    mak_course = rbf(xi, yi).astype(np.float32)

    # 3. Create an in-memory GeoTIFF from that coarse array
    mak_course_ds = flp.io.write_geotiff(
        "", "", mak_course, x_min, y_max, coarse_cellsize, dst_epsg
    )

    # 4. Warp back to full resolution with cubic-spline resampling
    ds_fine = gdal.Warp(
        "",
        mak_course_ds,
        dstSRS=f"EPSG:{dst_epsg}",
        format="VRT",
        resampleAlg=gdal.GRA_CubicSpline,
        xRes=cellsize,
        yRes=-cellsize,
        outputBounds=[x_min, y_min, x_max, y_max],
        outputBoundsSRS=f"EPSG:{dst_epsg}",
        dstNodata=-9999,
    )

    # 5. Read and post-process
    mak = ds_fine.ReadAsArray().astype(np.float32)
    mak[mak < 0.0] = 0.05
    mn, mx = point_values.min(), point_values.max()
    mak = np.clip(mak, mn * 0.99, mx * 1.01)

    # 6. create a point data set for analysis.
    if intermediate_path:
        flp.io.write_point_layer(x, y, point_values, dst_epsg, intermediate_path)

    # 7. Clean up
    mak_course_ds = None
    ds_fine = None

    return mak
