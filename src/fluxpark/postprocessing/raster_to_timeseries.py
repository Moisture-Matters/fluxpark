"""Convert FluxPark output rasters to spatial-mean timeseries.

The main entry point is ``rasters_to_timeseries``.  It reads one tif per
(date, parameter) combination, computes the spatial mean ignoring nodata
(-9999), and optionally breaks the result down by land-use class.

The function is used internally by ``eval_waterbalance`` but can also be
called standalone to turn any set of output rasters into a tidy DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from osgeo import gdal

logger = logging.getLogger(__name__)

NODATA = -9999.0


def _read_tif_as_array(path: Path) -> np.ndarray:
    """Return a float32 array from a GeoTIFF; nodata → NaN."""
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    arr[arr == NODATA] = np.nan
    return arr


def _spatial_mean(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Nanmean over arr, optionally restricted to cells where mask is True."""
    if mask is not None:
        values = arr[mask]
    else:
        values = arr.ravel()
    valid = values[~np.isnan(values)]
    return float(np.mean(valid)) if valid.size > 0 else np.nan


def check_required_files(
    outdir: Path,
    dates: pd.DatetimeIndex,
    parameters: Sequence[str],
) -> list[str]:
    """Return a list of missing tif paths (empty when all present)."""
    missing = []
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        for par in parameters:
            p = outdir / f"{date_str}-{par}.tif"
            if not p.exists():
                missing.append(str(p))
    return missing


def rasters_to_timeseries(
    outdir: Union[str, Path],
    parameters: Sequence[str],
    dates: pd.DatetimeIndex,
    luse_map: Optional[np.ndarray] = None,
    luse_ids: Optional[np.ndarray] = None,
    luse_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute spatial-mean timeseries from FluxPark output rasters.

    Parameters
    ----------
    outdir:
        Directory containing the output tif files.
    parameters:
        Parameter names to read, e.g. ``["prec_cum_ytd_mm", "runoff_cum_ytd_mm"]``.
    dates:
        Dates for which to read rasters.
    luse_map:
        2-D integer array of land-use classes.  When provided the result
        contains one row per (date, luse_class) combination in addition to
        an ``"all"`` aggregate row.  Must match the spatial extent of the
        rasters.
    luse_ids:
        1-D array of land-use class integers to include in the breakdown.
        If *None* all unique values in *luse_map* are used.
    luse_labels:
        Human-readable labels for *luse_ids* (same length).  When provided
        a ``"luse_label"`` column is added to the result.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``luse_class`` (``"all"`` or integer),
        optionally ``luse_label``, then one column per parameter.
        Missing files produce NaN for the affected parameter/date.
    """
    outdir = Path(outdir)

    do_per_class = luse_map is not None
    if do_per_class and luse_ids is None:
        luse_ids = np.unique(luse_map[~np.isnan(luse_map.astype(float))])

    label_map: dict[int, str] = {}
    if do_per_class and luse_labels is not None and luse_ids is not None:
        label_map = dict(zip(luse_ids.tolist(), luse_labels.tolist()))

    rows = []
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        arrays: dict[str, np.ndarray] = {}
        for par in parameters:
            path = outdir / f"{date_str}-{par}.tif"
            if path.exists():
                try:
                    arrays[par] = _read_tif_as_array(path)
                except Exception as exc:
                    logger.warning("Could not read %s: %s", path, exc)
                    arrays[par] = None
            else:
                arrays[par] = None

        # whole-area row
        row: dict = {"date": date.date(), "luse_class": "all"}
        if label_map:
            row["luse_label"] = "all"
        for par in parameters:
            arr = arrays[par]
            row[par] = _spatial_mean(arr) if arr is not None else np.nan
        rows.append(row)

        # per land-use class
        if do_per_class and luse_ids is not None:
            for cls in luse_ids:
                cls_mask = luse_map == cls
                if not cls_mask.any():
                    continue
                row_cls: dict = {"date": date.date(), "luse_class": int(cls)}
                if label_map:
                    row_cls["luse_label"] = label_map.get(int(cls), str(cls))
                for par in parameters:
                    arr = arrays[par]
                    row_cls[par] = (
                        _spatial_mean(arr, cls_mask) if arr is not None else np.nan
                    )
                rows.append(row_cls)

    return pd.DataFrame(rows)
