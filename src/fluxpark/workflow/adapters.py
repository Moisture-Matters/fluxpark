from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import fluxpark as flp
from osgeo import gdal


DEFAULT_KNMI_NC_PATTERN = r".*_{date}T\d{{6}}_\d{{8}}T\d{{6}}_.*\.nc$"


def find_knmi_netcdf_file(
    date,
    source_dir: Path,
    pattern_template: str = DEFAULT_KNMI_NC_PATTERN,
) -> Path:
    """
    Find the NetCDF file matching a given date and filename pattern.

    Parameters
    ----------
    date : datetime-like
        Date for which the file should be found.
    source_dir : Path
        Directory containing NetCDF files.
    pattern_template : str, default=DEFAULT_KNMI_NC_PATTERN
        Regular expression template used to match the filename.
        The template must contain the placeholder ``{date}``, which will be
        replaced by ``date.strftime("%Y%m%d")``.

    Returns
    -------
    Path
        Path to the matching NetCDF file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    date_str = date.strftime("%Y%m%d")
    pattern = re.compile(pattern_template.format(date=date_str))

    files = os.listdir(source_dir)
    matches = [fname for fname in files if pattern.match(fname)]

    if not matches:
        raise FileNotFoundError(
            f"No file found in {source_dir} matching pattern "
            f"{pattern.pattern!r} for date {date.strftime('%Y-%m-%d')}."
        )

    return source_dir / matches[0]


def _read_netcdf_to_grid(
    date,
    source_dir: Path,
    grid_params: dict,
    pattern_template: str,
    resample_alg: int = gdal.GRA_Bilinear,
):
    """
    Read and reproject a NetCDF raster to the FluxPark grid.

    Parameters
    ----------
    date : datetime-like
        Date for which the raster should be read.
    source_dir : Path
        Directory containing NetCDF files.
    grid_params : dict
        FluxPark grid parameters.
    pattern_template : str
        Regular expression template used to match the filename.
    resample_alg : int, default=gdal.GRA_Bilinear
        GDAL resampling algorithm.

    Returns
    -------
    np.ndarray
        Reprojected raster array.
    """
    source_path = find_knmi_netcdf_file(
        date=date,
        source_dir=source_dir,
        pattern_template=pattern_template,
    )
    ncfile = flp.io.NetCDFReader(source_path)

    read_grid_params = dict(grid_params)
    read_grid_params["cutline_path"] = None

    return ncfile.read_and_reproject(
        **read_grid_params,
        resample_alg=resample_alg,
    )


def make_knmi_netcdf_rain_provider(
    source_dir: Path | str,
    pattern_template: str = DEFAULT_KNMI_NC_PATTERN,
    resample_alg: int = gdal.GRA_Bilinear,
) -> Callable:
    """
    Create a rain provider based on local NetCDF files.

    Parameters
    ----------
    source_dir : Path or str
        Directory containing precipitation NetCDF files.
    pattern_template : str, default=DEFAULT_KNMI_NC_PATTERN
        Regular expression template used to match the filename.
    resample_alg : int, default=gdal.GRA_Bilinear
        GDAL resampling algorithm.

    Returns
    -------
    Callable
        Rain provider function compatible with RainProviderPort.
    """
    source_dir = Path(source_dir)

    def rain_provider(runner):
        return _read_netcdf_to_grid(
            date=runner.date,
            source_dir=source_dir,
            grid_params=runner.grid_params,
            pattern_template=pattern_template,
            resample_alg=resample_alg,
        )

    return rain_provider


def make_knmi_netcdf_etref_provider(
    source_dir: Path | str,
    pattern_template: str = DEFAULT_KNMI_NC_PATTERN,
    resample_alg: int = gdal.GRA_Bilinear,
) -> Callable:
    """
    Create an ETref provider based on local NetCDF files.

    Parameters
    ----------
    source_dir : Path or str
        Directory containing ETref NetCDF files.
    pattern_template : str, default=DEFAULT_KNMI_NC_PATTERN
        Regular expression template used to match the filename.
    resample_alg : int, default=gdal.GRA_Bilinear
        GDAL resampling algorithm.

    Returns
    -------
    Callable
        ETref provider function compatible with EtrefProviderPort.
    """
    source_dir = Path(source_dir)

    def etref_provider(runner):
        return _read_netcdf_to_grid(
            date=runner.date,
            source_dir=source_dir,
            grid_params=runner.grid_params,
            pattern_template=pattern_template,
            resample_alg=resample_alg,
        )

    return etref_provider
