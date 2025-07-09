from pathlib import Path
import math
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union, cast
import fluxpark as flp
import logging
import os


def check_output_files(
    output_files: str, conv_output_df: pd.DataFrame, mod_flags: dict, store_states: bool
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
        mod_col = mod_key
        calc_col = f"calc_{mod_key}"
        if mod_col in conv_output_df.columns:
            conv_output_df_sel = pd.concat(
                [conv_output_df_sel, conv_output_df[conv_output_df[mod_col] == 1]]
            )
        if calc_col in conv_output_df.columns:
            calc_par_list += conv_output_df[
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


def detect_dynamic_landuse_and_years(
    landuse_filename, root_soilm_scp_filename, root_soilm_pwp_filename, indir_rasters
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
        Directory containing input raster files.

    Returns
    -------
    dynamic : bool
        True if all patterns include '{year}', False otherwise.
    input_raster_years : ndarray of str
        Sorted unique list of years for which 'luse_ids.tif' files
        exist.
    """
    is_luse = flp.utils.has_placeholders(landuse_filename)
    is_scp = flp.utils.has_placeholders(root_soilm_scp_filename)
    is_pwp = flp.utils.has_placeholders(root_soilm_pwp_filename)

    if is_luse and is_scp and is_pwp:
        logging.info(
            "Run default dynamic landuse simulations that could "
            "change yearly if available"
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
        logging.info("Run with static landuse map")
        dynamic = False

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
        Start date string (e.g. '2025-06-01').
    end : str
        End date string.

    Returns
    -------
    pd.DatetimeIndex
        Sequence of dates from start to end inclusive.
    """
    s = pd.to_datetime(start, dayfirst=True, errors="raise")
    e = pd.to_datetime(end, dayfirst=True, errors="raise")
    return pd.date_range(s, e, freq="D")


def resolve_dirs(
    outdir: Union[str, Path],
    indir: Union[str, Path],
    indir_rasters: Optional[Union[str, Path]] = None,
    indir_masks: Optional[Union[str, Path]] = None,
    intermediate_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Path, Path, Optional[Path]]:
    """
    Normalize output/input directories and derive rasters/masks subdirs.

    Parameters
    ----------
    outdir
        Base output directory.
    indir
        Base input directory.
    indir_rasters
        Optional override for the 'rasters' subdirectory.
    indir_masks
        Optional override for the 'masks' subdirectory.
    intermediate_dir
        Optional the dir to store intermediate output

    Returns
    -------
    out_p, in_p, rasters_p, masks_p
        Paths for output, input, rasters, and masks.
    """
    out_p = Path(outdir)
    in_p = Path(indir)
    rasters_p = Path(indir_rasters) if indir_rasters else in_p / "rasters"
    masks_p = Path(indir_masks) if indir_masks else in_p / "masks"
    if intermediate_dir:
        # Normalize to a Path
        intermediate_dir = Path(intermediate_dir)
    else:
        # be excplicit prevents typing problems
        intermediate_dir = None

    return out_p, in_p, rasters_p, masks_p, intermediate_dir


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
    ncols = math.ceil((x_max - x_min) / cellsize)
    nrows = math.ceil((y_max - y_min) / cellsize)
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
    sheets = pd.read_excel(  # type: ignore[call]
        indir / evap_param_table,
        sheet_name=None,
        skiprows=range(0, 12),
        usecols="A:I",
        na_values=str(-9999),
    )
    df = pd.concat(sheets, ignore_index=True)
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
        output_files, conv_output_table, mods, store_states
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
    imperv_filename: str,
    decid_filename: str,
    conif_filename: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load static rasters: imperviousness and (optionally) forest cover maps.

    Parameters
    ----------
    indir_rasters
        Directory where raster files live.
    grid_params
        Passed as **kwargs to GeoTiffReader.read_and_reproject.
    mods
        Module flags, e.g. {'mod_vegcover': True, …}.
    imperv_filename
        The impervious density file name
    conif_filename
        The coniferous forest soil cover file name
    decid_filename
        The decideous forest soil cover file name

    Returns
    -------
    imperv
        Impervious fraction (0 to 1).
    soil_cov_decid
        Deciduous forest cover fraction or None.
    soil_cov_conif
        Coniferous forest cover fraction or None.
    """

    # Imperviousness (0→1)
    reader = flp.io.GeoTiffReader(indir_rasters / imperv_filename, nodata_value=0)
    imperv = reader.read_and_reproject(**grid_params).astype(np.float32) / 100.0

    soil_cov_decid = None
    soil_cov_conif = None
    if mods.get("mod_vegcover", False):
        # Deciduous cover
        reader = flp.io.GeoTiffReader(indir_rasters / decid_filename, nodata_value=0)
        soil_cov_decid = reader.read_and_reproject(**grid_params).astype(np.float32)
        soil_cov_decid[soil_cov_decid == 0] = np.nan

        # Coniferous cover
        reader = flp.io.GeoTiffReader(indir_rasters / conif_filename, nodata_value=0)
        soil_cov_conif = reader.read_and_reproject(**grid_params).astype(np.float32)
        soil_cov_conif[soil_cov_conif == 0] = np.nan

    return imperv, soil_cov_decid, soil_cov_conif
