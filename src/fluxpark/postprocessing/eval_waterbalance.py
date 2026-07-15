"""Water balance evaluation for FluxPark output.

The water balance identity that is verified:

    prec_cum_ytd + ΔSMD = int_act_cum_ytd + trans_act_cum_ytd + soil_evap_act_cum_ytd
                        + runoff_cum_ytd + recharge_cum_ytd

where ΔSMD = soilm_def_act_mm(today) − soilm_def_act_mm(at cumulative reset).
A positive ΔSMD means the soil dried out; that water was already counted in the
output fluxes and therefore acts as an additional input alongside precipitation.

Usage
-----
After a run with ``eval_waterbalance=True``::

    flp.postprocessing.eval_waterbalance(outdir)

Or completely standalone (requires ``fluxpark_cfg.json`` in outdir)::

    import fluxpark as flp
    flp.postprocessing.eval_waterbalance("/path/to/outdir")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import fluxpark as flp
from fluxpark.setup.core_initialization import WATERBALANCE_REQUIRED_PARAMS
from .raster_to_timeseries import check_required_files, rasters_to_timeseries

logger = logging.getLogger(__name__)

# Column names used in the balance computation
_PREC = "prec_cum_ytd_mm"
_INT = "int_act_cum_ytd_mm"
_TRANS = "trans_act_cum_ytd_mm"
_SOILEVAP = "soil_evap_act_cum_ytd_mm"
_RUNOFF = "runoff_cum_ytd_mm"
_RECHARGE = "recharge_cum_ytd_mm"
_SMD = "soilm_def_act_mm"

_BALANCE_ERROR_THRESHOLD_MM = 10.0


def _compute_balance(df: pd.DataFrame, smd_at_reset: pd.Series) -> pd.Series:
    """Return the residual (mm) of the water balance for each row in df.

    Parameters
    ----------
    df:
        DataFrame with one row per (date, luse_class) containing the
        cumulative flux columns and soilm_def_act_mm.
    smd_at_reset:
        Series indexed by luse_class with the SMD value at the last
        cumulative reset for each class.
    """
    delta_smd = df[_SMD] - df["luse_class"].map(smd_at_reset).fillna(0.0)
    rhs = (
        df[_INT] + df[_TRANS] + df[_SOILEVAP] + df[_RUNOFF] + df[_RECHARGE] + delta_smd
    )
    return df[_PREC] - rhs


def eval_waterbalance(
    outdir: Union[str, Path],
    error_threshold_mm: float = _BALANCE_ERROR_THRESHOLD_MM,
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Evaluate the water balance over all available output dates.

    The function loads the FluxPark configuration from ``outdir``, rebuilds
    the runner setup (for grid and land-use information), and reads the
    required cumulative rasters date by date.  Land use is refreshed at the
    start of each calendar year, matching the behaviour of the main run.

    Parameters
    ----------
    outdir:
        Directory that contains FluxPark output tifs and ``fluxpark_cfg.json``.
    error_threshold_mm:
        Absolute water balance error (mm, cumulative YTD) above which a
        warning is logged.  Default is 10 mm.
    output_csv:
        Path for the output CSV.  Defaults to ``outdir/waterbalance_eval.csv``.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: ``date``, ``luse_class``,
        ``luse_label``, all required parameters, and ``balance_error_mm``.
        Written to *output_csv* as well.
    """
    outdir = Path(outdir)

    cfg = flp.config.load_cfg(outdir)
    runner = flp.FluxParkRunner(cfg)
    # silence the setup chatter without touching the host's root logger
    flp_logger = logging.getLogger("fluxpark")
    prev_level = flp_logger.level
    flp_logger.setLevel(logging.WARNING)
    try:
        runner.setup()
    finally:
        flp_logger.setLevel(prev_level)

    all_dates = runner.dates
    dates = all_dates
    if cfg.only_yearly_output:
        dates = pd.DatetimeIndex(
            [d for d in all_dates if d.month == 12 and d.day == 31]
        )
    luse_ids_conv = runner.luse_ids
    luse_labels_conv = runner.luse_label

    # read initial SMD tif (written by runner on day before run start)
    init_date = all_dates[0] - pd.Timedelta(days=1)
    init_smd_path = outdir / f"{init_date.strftime('%Y%m%d')}-{_SMD}.tif"
    if init_smd_path.exists():
        init_smd_dates = pd.DatetimeIndex([init_date])
        init_ts = rasters_to_timeseries(
            outdir=outdir,
            parameters=[_SMD],
            dates=init_smd_dates,
            luse_map=_load_luse_map_for_year(runner, dates[0].year),
            luse_ids=luse_ids_conv,
            luse_labels=luse_labels_conv,
        )
        init_smd_by_class = init_ts.set_index("luse_class")[_SMD].to_dict()
    else:
        logger.warning(
            "Initial SMD file '%s' not found — assuming SMD=0 at run start. "
            "Re-run with eval_waterbalance=True to store the initial state.",
            init_smd_path,
        )
        init_smd_by_class = {}

    # check that all required tifs are present
    missing = check_required_files(outdir, dates, WATERBALANCE_REQUIRED_PARAMS)
    if missing:
        n = len(missing)
        examples = missing[:5]
        msg = (
            f"{n} required raster file(s) are missing for water balance evaluation. "
            "First examples:\n  " + "\n  ".join(examples) + "\n"
            "Tip: re-run the model with eval_waterbalance=True, or call "
            "flp.postprocessing.rasters_to_timeseries() manually after verifying "
            "the required parameters were part of the model output."
        )
        raise FileNotFoundError(msg)

    # group dates by year so we can load the correct landuse map per year
    years = sorted({d.year for d in dates})
    all_frames: list[pd.DataFrame] = []

    for year in years:
        year_dates = pd.DatetimeIndex([d for d in dates if d.year == year])

        # load landuse for this year (mirrors runner logic)
        luse_map = _load_luse_map_for_year(runner, year)

        ts = rasters_to_timeseries(
            outdir=outdir,
            parameters=WATERBALANCE_REQUIRED_PARAMS,
            dates=year_dates,
            luse_map=luse_map,
            luse_ids=luse_ids_conv,
            luse_labels=luse_labels_conv,
            common_valid_mask=True,
        )
        ts["year"] = year
        all_frames.append(ts)

    df = pd.concat(all_frames, ignore_index=True)

    # --- compute ΔSMD per year and per luse_class ---
    # The cumulative resets on (reset_cum_month, reset_cum_day).
    # SMD at reset = the SMD value on the last date BEFORE the reset within the
    # data; for the first date of a run we assume SMD_start = 0 (initialised to 0).
    reset_day = cfg.reset_cum_day
    reset_month = cfg.reset_cum_month

    def _is_reset_date(date: pd.Timestamp) -> bool:
        return date.month == reset_month and date.day == reset_day

    # Build a dict: (year, luse_class) → smd at the moment the cum was reset
    # For year Y, that is the SMD on the last available date of year Y-1
    # (or 0 when Y is the first simulation year).
    smd_at_reset_lookup: dict[tuple[int, object], float] = {}

    for year in years:
        prev_year = year - 1
        prev_year_data = df[df["year"] == prev_year]
        for cls in df["luse_class"].unique():
            if prev_year_data.empty:
                # first simulation year: use initial state from stored tif
                smd_at_reset_lookup[(year, cls)] = float(
                    init_smd_by_class.get(cls, 0.0)
                )
            else:
                cls_rows = prev_year_data[prev_year_data["luse_class"] == cls]
                if cls_rows.empty:
                    smd_at_reset_lookup[(year, cls)] = 0.0
                else:
                    smd_at_reset_lookup[(year, cls)] = float(
                        cls_rows.sort_values("date").iloc[-1][_SMD]
                    )

    def _get_smd_at_reset(row: pd.Series) -> float:
        return smd_at_reset_lookup.get((row["year"], row["luse_class"]), 0.0)

    df["smd_at_reset_mm"] = df.apply(_get_smd_at_reset, axis=1)
    df["delta_smd_mm"] = df[_SMD] - df["smd_at_reset_mm"]
    df["balance_error_mm"] = (
        df[_PREC]
        + df["delta_smd_mm"]
        - df[_INT]
        - df[_TRANS]
        - df[_SOILEVAP]
        - df[_RUNOFF]
        - df[_RECHARGE]
    )

    # --- log warnings ---
    exceeded = df[np.abs(df["balance_error_mm"]) > error_threshold_mm]
    if exceeded.empty:
        logger.info(
            "Water balance evaluation passed: all errors within %.1f mm.",
            error_threshold_mm,
        )
    else:
        logger.warning(
            "Water balance errors > %.1f mm detected on %d date/class combinations:",
            error_threshold_mm,
            len(exceeded),
        )
        for _, row in exceeded.iterrows():
            label = row.get("luse_label", row["luse_class"])
            logger.warning(
                "  date=%s  luse_class=%s (%s)  error=%.2f mm",
                row["date"],
                row["luse_class"],
                label,
                row["balance_error_mm"],
            )

    # --- write CSV ---
    if output_csv is None:
        output_csv = outdir / "waterbalance_eval.csv"
    output_csv = Path(output_csv)
    df.drop(columns=["year"], inplace=True)

    # keep valid_fraction and balance_error_mm as the last two columns
    tail = [c for c in ("valid_fraction", "balance_error_mm") if c in df.columns]
    df = df[[c for c in df.columns if c not in tail] + tail]

    df.to_csv(output_csv, index=False, float_format="%.3f")
    logger.info("Water balance evaluation written to %s", output_csv)

    # this function built its own runner (setup only); release its resources
    runner.close()
    return df


def _load_luse_map_for_year(runner: "flp.FluxParkRunner", year: int) -> np.ndarray:
    """Load the land-use raster appropriate for *year*, using runner state."""
    cfg = runner.cfg

    available_years = runner.input_raster_years.astype(int)
    # pick the most recent available year not exceeding the requested year
    valid = available_years[available_years <= year]
    if valid.size == 0:
        map_year = int(available_years[0])
    else:
        map_year = int(valid[-1])

    luse_filename = cfg.landuse_rastername.replace("{year}", str(map_year))
    # resolve through the release (honouring extends) and join URL-safely;
    # indir_rasters is a plain string for remote input, not a Path.
    luse_path = flp.setup.resolve_raster(
        runner.input_sources, runner.indir_rasters, luse_filename
    )
    reader = flp.io.GeoTiffReader(luse_path, dst_nodata=0)
    return reader.read_and_reproject(**runner.grid_params).astype(np.int32)
