import fluxpark as flp
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import os
from osgeo import gdal


def write_output_tif(
    variable,
    filename,
    landuse_map,
    write_nan_for_landuse_ids,
    replace_nan_with_zero,
    outdir,
    x_min,
    y_max,
    cellsize,
    epsg_code,
    dtype=gdal.GDT_Float32,
):
    """
    Write a single variable array as GeoTIFF, applying masks and scaling.

    Parameters
    ----------
    variable : np.ndarray
        2D array to write.
    filename : str
        Output filename (e.g. '20250601-rain.tif').
    landuse_map : np.ndarray
        2D landuse class map for masking sea/zero.
    write_nan_for_landuse_ids : list of int
        the landuse ids will get nan. e.g. sea or the no data value in landuse map
    replace_nan_with_zero : bool
        if true all nan get zero
    outdir : str or Path
        Directory to write output file.
    x_min : float
        Upper-left X coordinate (corner of top-left pixel).
    y_max : float
        Upper-left Y coordinate (corner of top-left pixel).
    cellsize : float
        Pixel resolution (assumed square).
    epsg_code : int
        EPSG code for projection (e.g. 28992).
    dtype : int, optional
        GDAL data type constant (default: GDT_Float32).
    """
    # convert soil_cov fraction to percent
    if "soil_cov" in filename:
        variable = variable * 100.0

    # mask nan (e.g. for sea or you no data value)
    if write_nan_for_landuse_ids:
        mask = np.isin(landuse_map, write_nan_for_landuse_ids)
        variable[mask] = np.nan
    # replace nan with zero.
    if replace_nan_with_zero:
        variable = np.where(np.isnan(variable), 0, variable)

    # replace NaNs with nodata flag
    variable = np.where(np.isnan(variable), -9999, variable)

    # round to 3 decimals
    variable = np.around(variable, 3)

    flp.io.write_geotiff(
        outdir,
        filename,
        variable.astype(np.float32),
        x_min,
        y_max,
        cellsize,
        epsg_code,
        nodata_value=-9999,
        dtype=dtype,
    )


def write_all_tiffs(
    date,
    out_par_list,
    conv_output,
    daily_output,
    cum_output,
    landuse_map,
    write_nan_for_landuse_ids,
    replace_nan_with_zero,
    outdir,
    x_min,
    y_max,
    cellsize,
    epsg_code,
    only_yearly_output=False,
    parallel=False,
    max_workers=None,
):
    """
    Write all requested parameters as GeoTIFFs for a given date.

    Parameters
    ----------
    date : datetime
        Current simulation date.
    out_par_list : list of str
        FluxPark parameter names to write.
    conv_output : dict
        Mapping from fluxpark param to variable name.
    daily_output : dict
        Daily variable arrays.
    cum_output : dict
        Cumulative variable values.
    landuse_map : np.ndarray
        Land use mask array.
    write_nan_for_landuse_ids : list of int
        the landuse ids will get nan. e.g. sea or the no data value in landuse map
    replace_nan_with_zero : bool
        if true all nan get zero
    outdir : str or Path
        Directory for output files.
    x_min, y_max : float
        Upper-left corner coordinates.
    cellsize : float
        Resolution of output grid.
    epsg_code : int
        EPSG projection code.
    only_yearly_output : bool, optional
        Skip writing except year-end (default: False).
    parallel : bool, optional
        Use threaded output (default: False).
    max_workers : int, optional
        the amount of parallel workers, by default it will be derived from your machine
    """
    date_str = date.strftime("%Y%m%d")
    dtype = gdal.GDT_Float32

    # skip until year end if requested
    if only_yearly_output:
        if not (date.month == 12 and date.day == 31):
            return

    # prepare tasks
    tasks = []
    for par in out_par_list:
        filename = f"{date_str}-{par}.tif"
        var = conv_output[par]
        if var not in daily_output.keys() and var not in cum_output.keys():
            logging.warning(f"{var} not in the output dicts")
        array = cum_output[var] if var.endswith("_c") else daily_output[var]
        tasks.append((array, filename))

    # wrapper
    def _worker(args):
        arr, fname = args
        write_output_tif(
            arr,
            fname,
            landuse_map,
            write_nan_for_landuse_ids,
            replace_nan_with_zero,
            outdir,
            x_min,
            y_max,
            cellsize,
            epsg_code,
            dtype,
        )

    if parallel:
        if not max_workers:
            max_workers = min((os.cpu_count() or 4), 64)
        with ThreadPoolExecutor(max_workers=max_workers) as exec:
            list(exec.map(_worker, tasks))
    else:
        for t in tasks:
            _worker(t)
