from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class FluxParkConfig:
    """
    Configuration settings for the FluxPark core simulation model.

    This configuration object contains all general and spatial parameters
    required to run the core model. It does not include any Nexus-specific logic.

    Parameters
    ----------
    date_start : str
        Start date of the simulation period (format: 'DD-MM-YYYY').
    date_end : str
        End date of the simulation period (format: 'DD-MM-YYYY').
    mask : Optional[str]
        Filename of the cutline raster mask used to clip input layers.
    calc_epsg_code : int
        EPSG code for the coordinate reference system used in calculations.
    x_min : float
        Minimum x-coordinate of the simulation extent.
    x_max : float
        Maximum x-coordinate of the simulation extent.
    y_min : float
        Minimum y-coordinate of the simulation extent.
    y_max : float
        Maximum y-coordinate of the simulation extent.
    cellsize : float
        Grid cell size (in map units) used for resampling input rasters.
    evap_param_table : str
        Filename of the Excel file containing evaporation parameters.
    output_files : list[str] or str, default = "flagship"
        List of output parameters to write to file or a string describing differnt
        combinations of output (all, flagship).

    indir : Union[str, Path], default='./input_data'
        Root directory containing all input files.
    indir_rasters : Optional[Union[str, Path]], default=None
        Directory containing raster input files. If None, defaults to `indir/rasters`.
    indir_masks : Optional[Union[str, Path]], default=None
        Directory containing mask rasters. If None, defaults to `indir/masks`.
    landuse_rastername : str, default="{year}_luse_ids.tif"
        Filename pattern for the yearly land use raster.
    root_soilm_scp_rastername : str, default="{year}_root_soilm_fc_scp_mm.tif"
        Filename pattern for the soil moisture field capacity raster (saturated).
    root_soilm_pwp_rastername : str, default="{year}_root_soilm_fc_pwp_mm.tif"
        Filename pattern for the soil moisture wilting point raster.
    impervdens_rastername : str, default="2018_impervdens.tif"
        Filename of the imperviousness raster.
    soil_cov_decid_rastername : str, default="forest_decid_soilcov_100m_3035.tif"
        Filename of the deciduous forest cover raster.
    soil_cov_conif_rastername : str, default="forest_conif_soilcov_100m_3035.tif"
        Filename of the coniferous forest cover raster.
    output_mapping : str, default="fluxpark_output_mapping.csv"
        Filename of the mapping table from variables to parameters.
    store_states : bool, default=False
        If true, calculation parameters are added to the output list.
    reset_cum_day : int, default=1
        Day of the month when cumulative variables are reset.
    reset_cum_month : int, default=1
        Month when cumulative variables are reset.
    mod_vegcover : bool, default=False
        Whether to include dynamic vegetation cover in simulations.
    only_yearly_output : bool, default=False
        If True, only writes output at the end of the year (31 Dec).
    parallel : bool, default=True
        Whether to parallelize output writing.
    max_workers : Optional[int], default=None
        Maximum number of parallel workers (threads) to use.
    outdir : Union[str, Path], default="./output_data"
        Output directory where model results are stored.
    intermediate_dir : Optional[Union[str, Path]], default=None
        Optional intermediate directory for temporary files.
    """

    # Positional (non-default) arguments
    date_start: str
    date_end: str
    mask: Optional[str]
    calc_epsg_code: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cellsize: float
    evap_param_table: str

    # Defaulted arguments
    output_files: Union[str, list[str], list[int]] = "flagship"
    indir: Union[str, Path] = "./input_data"
    indir_rasters: Optional[Union[str, Path]] = None
    indir_masks: Optional[Union[str, Path]] = None

    landuse_rastername: str = "{year}_luse_ids.tif"
    root_soilm_scp_rastername: str = "{year}_root_soilm_fc_scp_mm.tif"
    root_soilm_pwp_rastername: str = "{year}_root_soilm_fc_pwp_mm.tif"
    impervdens_rastername: str = "2018_impervdens.tif"
    soil_cov_decid_rastername: str = "forest_decid_soilcov_100m_3035.tif"
    soil_cov_conif_rastername: str = "forest_conif_soilcov_100m_3035.tif"
    output_mapping: str = "fluxpark_output_mapping.csv"
    store_states: bool = False

    reset_cum_day: int = 1
    reset_cum_month: int = 1

    mod_vegcover: bool = False

    only_yearly_output: bool = False
    parallel: bool = True
    max_workers: Optional[int] = None

    outdir: Union[str, Path] = "./output_data"
    intermediate_dir: Optional[Union[str, Path]] = None
