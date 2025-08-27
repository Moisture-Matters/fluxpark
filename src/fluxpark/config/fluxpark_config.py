from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class FluxParkConfig:
    """
     Configuration settings for the FluxPark core simulation model.

     This configuration object contains all general and spatial parameters
     required to run the core model.

     Parameters
     ----------
     date_start : str
         Start date of the simulation period (format: 'DD-MM-YYYY').
     date_end : str
         End date of the simulation period (format: 'DD-MM-YYYY').
     mask : Optional[str]
         Filename of the cutline .shp file used to clip input and output rasters.
     calc_epsg_code : int
         EPSG code for the coordinate system used in calculations, Dutch RD = 28992
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
    output_files : str or list[str], default="flagship"
         Identifies the required output files, could be "all", "flagship" or a list of
         output parameters, e.g. ["prec_mm_d", "evap_total_act_mm_d"] or a list of
         output IDs, e.g. [1, 4, 8, 10]. Output IDs can be found in
         fluxpark_output_mapping.csv.
     indir : Union[str, Path], default='./input_data'
         Root directory containing all input files.
     indir_rasters : Optional[Union[str, Path]], default=None
         Directory containing raster input files. If None, defaults to `indir/rasters`.
     indir_masks : Optional[Union[str, Path]], default=None
         Directory containing mask rasters. If None, defaults to `indir/masks`.
     landuse_rastername : str, default="{year}_luse_ids.tif"
         Filename of the land-use raster (.tif). If land use changes over time, provide
         a filename pattern with placeholder "{year}" (e.g. "landuse_{year}.tif"). If a
         year is missing, FluxPark uses the most recent previous map until a newer one
         is found in 'indir'. It will search for a new map every 1st of january. At the
         first timestep, the earliest available map not exceeding the simulation start
         year is used.
     root_soilm_scp_rastername : str, default="{year}_root_soilm_fc_scp_mm.tif"
         Filename or filename pattern for the root zone soil moisture content (mm)
         between field capacity and the stomatal closure point (scp). If landuse changes
         yearly this file should also be available yearly.
     root_soilm_pwp_rastername : str, default="{year}_root_soilm_fc_pwp_mm.tif"
         Filename or filename pattern for the root zone soil moisture content (mm)
         between field capacity and the permanent wilting point (pwp). If landuse
         changes yearly this file should also be available yearly.
     impervdens_rastername : str, default="2018_impervdens.tif"
         Filename of the imperviousness raster.
     soil_cov_decid_rastername : str, default="forest_decid_soilcov_100m_3035.tif"
         Filename of the deciduous forest cover raster.
     soil_cov_conif_rastername : str, default="forest_conif_soilcov_100m_3035.tif"
         Filename of the coniferous forest cover raster.
     output_mapping : str, default="fluxpark_output_mapping.csv"
         Filename of the mapping table from variables to parameters.
     bare_soil_ids : list[int], default = [15]
         Land use map ids that should be treated as bare soil giving it specific beta
         parameter value in the soil evaporation calculations.
     open_water_ids : list[int], default = [16]
         To make output raster maps more nice, the output from the usaturated zone model
         will get nan for landuse ids that have open water. Make it None if you don't
         want anny masking.
     urban_ids : list[int], default = [18]
         Land use map ids that should be treated as urban area allowing the impervious
         density to have effect on calculations.
     write_nan_for_landuse_ids: list[int], default = [0, 17]
         When writing the output these land use map ids get nan. By default we do this
         for sea (17) and for the no data value in the landuse map (0).
     replace_nan_with_zero: bool, default = False
         If true, all nan values in output maps are replaced with zero.
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
         Maximum number of parallel workers (threads) to use. If None it will be derived
         from your machine (cpu count).
     outdir : Union[str, Path], default="./output_data"
         Output directory where model results are stored.
     intermediate_dir : Optional[Union[str, Path]], default=None
         Optional intermediate directory for temporary files like point information if
         using interpolation.
    """

    # Positional (non-default) arguments
    date_start: str
    date_end: str
    calc_epsg_code: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cellsize: float
    evap_param_table: str

    # Defaulted arguments
    mask: Optional[str] = None
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

    bare_soil_ids: list[int] = field(default_factory=lambda: [15])
    open_water_ids: list[int] = field(default_factory=lambda: [16])
    urban_ids: list[int] = field(default_factory=lambda: [18])
    write_nan_for_landuse_ids: list[int] = field(default_factory=lambda: [0, 17])
    replace_nan_with_zero: bool = False

    store_states: bool = False

    reset_cum_day: int = 1
    reset_cum_month: int = 1

    mod_vegcover: bool = False

    only_yearly_output: bool = False
    parallel: bool = True
    max_workers: Optional[int] = None

    outdir: Union[str, Path] = "./output_data"
    intermediate_dir: Optional[Union[str, Path]] = None
