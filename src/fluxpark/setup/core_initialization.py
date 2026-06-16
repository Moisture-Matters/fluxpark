from pathlib import Path
import math
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union, cast
import fluxpark as flp
import logging
import os


WATERBALANCE_REQUIRED_PARAMS = [
    "prec_cum_ytd_mm",
    "int_act_cum_ytd_mm",
    "trans_act_cum_ytd_mm",
    "soil_evap_act_cum_ytd_mm",
    "runoff_cum_ytd_mm",
    "recharge_cum_ytd_mm",
    "soilm_def_act_mm",
]


def check_output_files(
    output_files: str,
    conv_output_df: pd.DataFrame,
    mod_flags: dict,
    store_states: bool,
    eval_waterbalance: bool = False,
):
    """
    Determine which parameters should be written, calculated, and aggregated.

    Parameters
    ----------
    output_files : str or list of int
        Desired output set. Can be:
        - "all" to include all available parameters.
        - "flagship" to include key parameters.
        - "selection" for your own preference.
        - A list of row indices from the conversion table.
    conv_output_df : pandas.DataFrame
        Conversion table with parameter names as index and
        module flags (e.g., 'mod_crop_prod') as columns.
    mod_flags : dict
        Dictionary mapping module names to bool flags,
        e.g. {'mod_dynamic_grass': True, 'mod_seen': False}
    store_states: bool
        If true, calculation parameters are added to the output list.
    eval_waterbalance : bool, default False
        If true, parameters required for water balance evaluation are added
        to the output list.

    Returns
    -------
    output_par_list : list of str
        List of Nexus parameters to write as output.
    calc_par_list : list of str
        Parameters required for internal calculations.
    cum_par_list : list of str
        Subset of output_par_list requiring aggregation.
    """
    # Core model selection
    conv_output_df_sel = conv_output_df[conv_output_df["mod"] == "core"]
    calc_par_list = conv_output_df[conv_output_df["calc_core"] == 1].index.tolist()

    # Extend based on active modules
    for mod_key, enabled in mod_flags.items():
        if not enabled:
            continue

        calc_col = f"calc_{mod_key}"

        conv_output_df_sel = pd.concat(
            [
                conv_output_df_sel,
                conv_output_df[conv_output_df["mod"] == mod_key],
            ]
        )

        if calc_col in conv_output_df.columns:
            calc_par_list += conv_output_df.loc[
                conv_output_df[calc_col] == 1
            ].index.tolist()

    # Determine output parameters
    if output_files == "all":
        output_par_list = conv_output_df_sel.index.tolist()
    elif output_files == "flagship":
        output_par_list = conv_output_df_sel[
            conv_output_df_sel["flagship"] == 1
        ].index.tolist()
    elif output_files == "selection":
        output_par_list = conv_output_df_sel[
            conv_output_df_sel["selection"] == 1
        ].index.tolist()
    elif isinstance(output_files, str):
        raise ValueError(
            f"output_files can be 'all', 'flagship' or 'selection'. Or it can be a list"
            f" with parameters names or id's. Your value '{output_files}' does not meet"
            " the requirements."
        )
    else:
        # It should be a list and we Convert integer IDs to names or keep the names
        converted = []
        index = conv_output_df.index
        for val in output_files:
            if isinstance(val, int):
                if 0 <= val < len(index):
                    converted.append(index[val])
                else:
                    raise IndexError(f"Index {val} is out of range for output files.")
            elif isinstance(val, str):
                if val not in index:
                    raise ValueError(
                        f"Parameter name '{val}' not found in conv_output_df index."
                    )
                converted.append(val)
            else:
                raise TypeError(f"Output file spec '{val}' must be int or str.")

        output_par_list = [
            name for name in converted if name in conv_output_df_sel.index
        ]

        if len(output_par_list) != len(converted):
            logging.warning(
                "Some output_files do not match the active modules and were ignored."
            )

    # Handle interdependencies
    if (
        "trans_def_cum_past10d_mm" in output_par_list
        and "trans_def_pot_mm_d" not in output_par_list
    ):
        output_par_list.append("trans_def_pot_mm_d")
    if (
        "drought_stress_index_pct" in output_par_list
        and "trans_rel_pct" not in output_par_list
    ):
        output_par_list.append("trans_rel_pct")

    # Force waterbalance parameters into output list before building cum_par_list
    if eval_waterbalance:
        for par in WATERBALANCE_REQUIRED_PARAMS:
            if par not in output_par_list:
                if par not in conv_output_df.index:
                    logging.warning(
                        "Water balance parameter '%s' not found in output "
                        "mapping — skipped.",
                        par,
                    )
                    continue
                output_par_list.append(par)

    # Cumulative parameters
    cum_par_list_all = conv_output_df_sel[
        conv_output_df_sel["cumulative"] == 1
    ].index.tolist()
    cum_par_list = [x for x in cum_par_list_all if x in output_par_list]

    # Force calc parameters into output list
    if store_states:
        missing = [x for x in calc_par_list if x not in output_par_list]
        output_par_list += missing

    return output_par_list, calc_par_list, cum_par_list


def resolve_indir(
    indir: Union[str, Path],
    input_version: Optional[str] = None,
) -> Tuple[Union[str, Path], Optional[str]]:
    """Fill the "{input_version}" placeholder and derive the line root.

    Parameters
    ----------
    indir
        Base input directory: local path or HTTPS URL, optionally containing
        an "{input_version}" placeholder.
    input_version
        Value to fill the placeholder. Required when the placeholder is
        present; ignored otherwise.

    Returns
    -------
    resolved_indir, line_root
        The placeholder-filled `indir`, and the line root (the part before the
        placeholder) or None when no placeholder was used.
    """
    indir_str = str(indir)
    if "{input_version}" not in indir_str:
        return indir, None
    if not input_version:
        raise RuntimeError(
            "indir contains the '{input_version}' placeholder but "
            "input_version was not provided in the configuration."
        )
    line_root = indir_str.split("{input_version}")[0].rstrip("/\\")
    return indir_str.format(input_version=input_version), line_root


def detect_dynamic_landuse_and_years(
    landuse_filename,
    root_soilm_scp_filename,
    root_soilm_pwp_filename,
    indir_rasters,
    input_sources=None,
):
    """
    Determine if input maps are dynamic and list available years.

    Parameters
    ----------
    landuse_filename : str
        Filename pattern for land use maps, may include '{year}'.
    root_soilm_scp_filename : str
        Filename pattern for soil moisture SCP maps.
    root_soilm_pwp_filename : str
        Filename pattern for soil moisture PWP maps.
    indir_rasters : Path or str
        Directory containing input raster files. Used to discover years only
        when `input_sources` is None (legacy behavior).
    input_sources : InputSources, optional
        Resolved input sources. When given, the available years come from the
        release (its ``release.yml`` chain) instead of listing `indir_rasters`,
        which is required for remote inputs where directory listing is not
        possible.

    Returns
    -------
    dynamic : bool
        True if all patterns include '{year}', False otherwise.
    input_raster_years : ndarray
        Sorted unique list of years for which land-use maps exist.
    """
    is_luse = flp.utils.has_placeholders(landuse_filename)
    is_scp = flp.utils.has_placeholders(root_soilm_scp_filename)
    is_pwp = flp.utils.has_placeholders(root_soilm_pwp_filename)

    if is_luse and is_scp and is_pwp:
        logging.info(
            "Dynamic land use enabled: yearly map reload if available"
        )
        dynamic = True
    elif is_luse or is_scp or is_pwp:
        raise RuntimeError(
            f"Not all input maps have '{{year}}' placeholder: "
            f"{landuse_filename}, {root_soilm_scp_filename}, "
            f"{root_soilm_pwp_filename}. Either all are static or all "
            "are dynamic"
        )
    else:
        logging.info("Using static land use map")
        dynamic = False

    if input_sources is not None:
        input_raster_years = np.array(sorted(input_sources.years))
    else:
        files = os.listdir(indir_rasters)
        mask = ["luse_ids.tif" in f for f in files]
        years = np.array([f.split("_")[0] for f in np.array(files)[mask]])
        input_raster_years = np.sort(np.unique(years))

    return dynamic, input_raster_years


def parse_dates(start: str, end: str) -> pd.DatetimeIndex:
    """
    Parse start and end dates into a daily pandas DateTimeIndex.

    Parameters
    ----------
    start : str
        Start date string (e.g. '2025-06-31').
    end : str
        End date string.

    Returns
    -------
    pd.DatetimeIndex
        Sequence of dates from start to end inclusive.
    """
    s = pd.to_datetime(start, format="%Y-%m-%d", errors="raise")
    e = pd.to_datetime(end, format="%Y-%m-%d", errors="raise")
    return pd.date_range(s, e, freq="D")


def resolve_dirs(
    outdir: Union[str, Path],
    indir: Union[str, Path],
    indir_tables: Optional[Union[str, Path]] = None,
    indir_rasters: Optional[Union[str, Path]] = None,
    indir_masks: Optional[Union[str, Path]] = None,
    intermediate_dir: Optional[Union[str, Path]] = None,
    input_version: Optional[str] = None,
) -> Tuple[
    Path,
    Union[str, Path],
    Union[str, Path],
    Union[str, Path],
    Optional[Path],
]:
    """
    Normalize output/input directories and derive rasters/masks subdirs.

    Fills the "{input_version}" placeholder in `indir` when present, and
    supports both local paths and remote HTTPS URLs. For a local `indir` the
    derived subdirectories are returned as :class:`pathlib.Path`; for a remote
    `indir` they are returned as forward-slash joined URL strings (so they can
    be opened by GDAL via ``/vsicurl/`` downstream).

    Parameters
    ----------
    outdir
        Base output directory (always local).
    indir
        Base input directory: a local path or an HTTPS URL. May contain an
        "{input_version}" placeholder.
    indir_tables
        Optional override for the 'tables' subdirectory.
    indir_rasters
        Optional override for the 'rasters' subdirectory.
    indir_masks
        Optional override for the 'masks' subdirectory.
    intermediate_dir
        Optional the dir to store intermediate output.
    input_version
        Value to fill the "{input_version}" placeholder in `indir`. Required
        when the placeholder is present; ignored otherwise.

    Returns
    -------
    out_p, table_p, rasters_p, masks_p, intermediate_dir
        Output, tables, rasters and masks locations (Path for local, str for
        remote URLs), plus the optional intermediate dir.

    Notes
    -----
    Masks are shared across versions and live at the line level. When `indir`
    uses the "{input_version}" placeholder, the masks default therefore points
    at the line root (the part of `indir` before the placeholder) + "masks",
    e.g. ".../releases/nweu/masks", rather than inside the version folder. An
    explicit `indir_masks` always takes precedence, and an `indir` without the
    placeholder keeps the legacy "indir/masks" default.
    """
    # Determine the line root (part of indir before the placeholder) before we
    # format it; masks live there, shared across versions.
    indir_str = str(indir)
    has_placeholder = "{input_version}" in indir_str
    line_root = None

    # Fill the {input_version} placeholder in indir if present.
    if has_placeholder:
        if not input_version:
            raise RuntimeError(
                "indir contains the '{input_version}' placeholder but "
                "input_version was not provided in the configuration."
            )
        line_root = indir_str.split("{input_version}")[0].rstrip("/\\")
        indir = indir_str.format(input_version=input_version)

    out_p = Path(outdir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Local base stays a Path; a remote base stays a URL string.
    in_base = indir if flp.utils.is_url(indir) else Path(indir)

    def _resolve(override, subdir):
        if override:
            return override if flp.utils.is_url(override) else Path(override)
        return flp.utils.join_path_or_url(in_base, subdir)

    table_p = _resolve(indir_tables, "tables")
    rasters_p = _resolve(indir_rasters, "rasters")

    # Masks default to the line root when a placeholder is used (shared across
    # versions); otherwise to the legacy in_base/masks.
    if indir_masks:
        masks_p: Union[str, Path] = (
            indir_masks if flp.utils.is_url(indir_masks) else Path(indir_masks)
        )
    elif line_root is not None:
        masks_p = flp.utils.join_path_or_url(
            line_root if flp.utils.is_url(line_root) else Path(line_root),
            "masks",
        )
    else:
        masks_p = flp.utils.join_path_or_url(in_base, "masks")

    if intermediate_dir:
        # Normalize to a Path
        intermediate_dir = Path(intermediate_dir)
    else:
        # be excplicit prevents typing problems
        intermediate_dir = None

    return out_p, table_p, rasters_p, masks_p, intermediate_dir


def compute_grid_params(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cellsize: float,
    epsg_code: int,
    indir_masks: Path,
    mask: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute grid dimensions and package geospatial settings.

    Parameters
    ----------
    x_min, x_max, y_min, y_max
        Bounding box in target CRS.
    cellsize
        Grid resolution (assumed square).
    epsg_code
        Output projection EPSG.
    indir_masks
        Base masks directory.
    mask
        Optional mask filename.

    Returns
    -------
    grid_params
        {
          "dst_epsg": int,
          "bounds": (x_min, x_max, y_min, y_max),
          "cellsize": float,
          "cutline_path": Optional[Path],
          "ncols": int,
          "nrows": int
        }
    """
    # # 1) Align bounds on rastergrid (we use target aligned pixels)
    # x_min_aligned = math.floor(x_min / cellsize) * cellsize
    # x_max_aligned = math.ceil(x_max / cellsize) * cellsize
    # y_min_aligned = math.floor(y_min / cellsize) * cellsize
    # y_max_aligned = math.ceil(y_max / cellsize) * cellsize

    # ncols = int((x_max_aligned - x_min_aligned) / cellsize)
    # nrows = int((y_max_aligned - y_min_aligned) / cellsize)

    x_range = x_max - x_min
    y_range = y_max - y_min
    tol = 1e-9  # tolerance for floating point rounding errors

    # Ensure xmin is aligned with the grid
    assert math.isclose(
        x_min % cellsize, 0, abs_tol=tol
    ), f"x_min={x_min} is not aligned with cellsize={cellsize}"

    # Ensure ymin is aligned with the grid
    assert math.isclose(
        y_min % cellsize, 0, abs_tol=tol
    ), f"y_min={y_min} is not aligned with cellsize={cellsize}"

    # Ensure the x-range is divisible by the cellsize
    assert math.isclose(
        x_range % cellsize, 0, abs_tol=tol
    ), f"x_range={x_range} is not divisible by cellsize={cellsize}"

    # Ensure the y-range is divisible by the cellsize
    assert math.isclose(
        y_range % cellsize, 0, abs_tol=tol
    ), f"y_range={y_range} is not divisible by cellsize={cellsize}"

    # Compute number of columns and rows
    ncols = int(round(x_range / cellsize))
    nrows = int(round(y_range / cellsize))

    cutline = Path(indir_masks) / mask if mask else None

    return {
        "dst_epsg": epsg_code,
        "bounds": (x_min, x_max, y_min, y_max),
        "cellsize": cellsize,
        "cutline_path": cutline,
        "ncols": ncols,
        "nrows": nrows,
    }


def load_evap_params(indir: Path, evap_param_table: str) -> dict[str, np.ndarray]:
    """
    Load evaporation parameters per land use and day-of-year.

    Parameters
    ----------
    indir : Path
        Directory containing the Excel file.
    evap_param_table : str
        Filename of the evap params workbook.

    Returns
    -------
    params : dict
        Keys are column names, values are NumPy arrays.
    """
    path = indir / evap_param_table

    # 1) get all sheet names
    xls = pd.ExcelFile(path)
    all_sheets = xls.sheet_names  # bijv. ['Daily', 'Monthly', 'LICENSE']

    # 2) remove 'LICENSE' sheet (case-insensitive)
    data_sheets = [s for s in all_sheets if s.lower() != "license"]

    # 3) read only filtered sheets
    sheets_dict = pd.read_excel(
        path,
        sheet_name=data_sheets,
        skiprows=range(12),
        usecols="A:I",
        na_values=str(-9999),
    )

    # 4) Concat en return een enkele DataFrame
    df = pd.concat(sheets_dict.values(), ignore_index=True)
    data = df.to_dict("list")
    raw = {k: np.array(v) for k, v in data.items()}
    return cast(dict[str, np.ndarray], raw)


def load_luse_evap_conv(
    indir: Path,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.str_]]:
    """
    Read landuse→evap ID conversion table.

    Parameters
    ----------
    indir : Path
        Directory containing 'conv_luse_evap_ids.csv'.

    Returns
    -------
    luse_ids, evap_ids, labels : arrays
    """
    df = pd.read_csv(
        indir / "conv_luse_evap_ids.csv",
        dtype={
            "luse_id": np.int64,
            "evap_id": np.int64,
            "label": str,
        },
    )
    return df["luse_id"].to_numpy(), df["evap_id"].to_numpy(), df["label"].to_numpy()


def load_conv_output(
    indir: Path,
    output_mapping: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """
    Read conversion of model vars ↔ output filenames.

    Parameters
    ----------
    indir : Path
        Directory containing 'fluxpark_output_mapping.csv'.

    Returns
    -------
    conv_output, conv_var : dict
        param→var and var→param maps.
    """
    df = pd.read_csv(
        indir / output_mapping,
        skiprows=list(np.arange(0, 10, 1)),
        index_col="parameter",
    )
    out_map = dict(zip(df.index, df["variable"]))
    var_map = dict(zip(df["variable"], df.index))
    return df, out_map, var_map


def prepare_output_and_rerun_lists(
    mods: Dict[str, bool],
    output_files: Any,
    conv_output_table: "pd.DataFrame",
    conv_output: Dict[str, str],
    store_states: bool,
    eval_waterbalance: bool = False,
) -> Tuple[
    List[str],  # out_par_list
    List[str],  # calc_par_list
    List[str],  # cum_par_list
    List[str],  # out_var_list
    List[str],  # rerun_par_list
    List[str],  # rerun_var_list
]:
    """
    Determine which parameters to output and which to carry forward.

    Parameters
    ----------
    mods
        All mod flags (e.g. {'mod_...': True, …}).
    output_files
        Specification of desired outputs ('all', 'flagship', list, etc.).
    conv_output_table
        DataFrame mapping 'parameter' → output flags & module settings.
    conv_output
        Dict mapping parameter names → Python variable names.
    store_states : bool
        If true, calculation parameters are added to the output list.
    eval_waterbalance : bool, default False
        If true, parameters required for water balance evaluation are added
        to the output list.

    Returns
    -------
    out_par_list
        List of requested output parameters.
    calc_par_list
        Parameters needed for rerun initialization.
    cum_par_list
        Cumulative parameters (also must be output).
    out_var_list
        Python variable names corresponding to out_par_list.
    rerun_par_list
        Combined list of calc_par_list and cum_par_list (unique).
    rerun_var_list
        Python variable names corresponding to rerun_par_list.
    """
    # get the three lists from the conversion table
    out_par_list, calc_par_list, cum_par_list = check_output_files(
        output_files, conv_output_table, mods, store_states, eval_waterbalance
    )

    # convert output parameters to variable names
    out_var_list = [conv_output[par] for par in out_par_list]

    # rerun parameters = calculation + cumulative
    rerun_par_list = list(set(calc_par_list) | set(cum_par_list))
    rerun_var_list = [conv_output[par] for par in rerun_par_list]

    return (
        out_par_list,
        calc_par_list,
        cum_par_list,
        out_var_list,
        rerun_par_list,
        rerun_var_list,
    )


def init_old(
    rerun_var_list: list[str], nrows: int, ncols: int
) -> dict[str, np.ndarray]:
    """
    Initialize the 'old' state dict for rerun variables.

    Parameters
    ----------
    rerun_var_list : list of str
        Python var names needing initial arrays.
    nrows, ncols : int
        Grid dimensions.

    Returns
    -------
    old : dict
        Each key→array of zeros or 0.68 for 'soil_cov'.
    """
    old = {}
    shape = (nrows, ncols)
    for key in rerun_var_list:
        if key == "soil_cov":
            old[key] = np.full(shape, 0.68, dtype="float32")
        else:
            old[key] = np.zeros(shape, dtype="float32")
    return old


def read_static_maps(
    indir_rasters: Path,
    grid_params: dict,
    mods: dict[str, bool],
    decid_filename: str,
    conif_filename: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load static rasters: forest cover maps.

    Parameters
    ----------
    indir_rasters
        Directory where raster files live.
    grid_params
        Passed as **kwargs to GeoTiffReader.read_and_reproject.
    mods
        Module flags, e.g. {'mod_vegcover': True, …}.
    conif_filename
        The coniferous forest soil cover file name
    decid_filename
        The decideous forest soil cover file name

    Returns
    -------
    soil_cov_decid
        Deciduous forest cover fraction or None.
    soil_cov_conif
        Coniferous forest cover fraction or None.
    """
    soil_cov_decid = None
    soil_cov_conif = None
    if mods.get("mod_vegcover", False):
        # Deciduous cover
        reader = flp.io.GeoTiffReader(indir_rasters / decid_filename, nodata_value=0)
        soil_cov_decid = (
            reader.read_and_reproject(**grid_params).astype(np.float32) / 100.0
        )
        soil_cov_decid[soil_cov_decid == 0] = np.nan

        # Coniferous cover
        reader = flp.io.GeoTiffReader(indir_rasters / conif_filename, nodata_value=0)
        soil_cov_conif = (
            reader.read_and_reproject(**grid_params).astype(np.float32) / 100.0
        )
        soil_cov_conif[soil_cov_conif == 0] = np.nan

    return soil_cov_decid, soil_cov_conif
