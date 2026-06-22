import fluxpark as flp
import numpy as np
from numpy.typing import NDArray
from typing import Optional, TypedDict
import logging
import bisect

logger = logging.getLogger(__name__)

_EXT2DRIVER = {
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".csv": "CSV",
    ".dxf": "DXF",
    ".gml": "GML",
    ".kml": "KML",  # or "LIBKML" for KMZ
    ".gpx": "GPX",
    ".fgb": "FlatGeobuf",
    ".sqlite": "SQLite",
}


def load_fluxpark_raster_inputs(
    date,
    indir_rasters,
    grid_params,
    dynamic_landuse,
    landuse_filename,
    root_soilm_scp_filename,
    root_soilm_pwp_filename,
    impervdens_filename,
    input_raster_years,
    luse_ids,
    bare_soil_ids,
    urban_ids,
    input_sources=None,
):
    """
    Load basic raster input files for a given date for the FluxPark model.

    Parameters
    ----------
    date : datetime
        Date for which input maps are needed.
    indir_rasters : Path
        Directory containing input raster files.
    grid_params : dict
        Dictionary with projection and extent settings.
    dynamic_landuse : bool
        If True, input maps are year-dependent.
    landuse_filename : str
        Template filename for land use maps with {year} placeholder.
    root_soilm_scp_filename : str
        Template filename for soil moisture at SCP with {year} placeholder.
    root_soilm_pwp_filename : str
        Template filename for soil moisture at PWP with {year} placeholder.
    impervdens_filename : str
        Template filename for impervious fractions with {year} placeholder,
        (also used for beta).
    input_raster_years : list of str
        List of years with available input maps.
    luse_ids : list of int
        List of valid land use IDs.
    bare_soil_ids : list of int
        List of land use IDs that are bare and should get a lower beta param.
    urban_ids : list of int
        List of land use IDs that are urban and should be scaled using the imperv.
    input_sources : InputSources, optional
        Resolved input sources. When given, each raster is located through it
        (honouring ``extends``); otherwise files are read from `indir_rasters`.

    Returns
    -------
    tuple
        landuse_map : ndarray
            Land use class IDs.
        soilm_scp : ndarray
            Soil moisture at SCP.
        soilm_pwp : ndarray
            Soil moisture at PWP.
        beta : ndarray
            Soil evaporation beta parameter map.
    """
    if dynamic_landuse:
        year = date.year
        raster_years = [int(y) for y in input_raster_years]

        # bisect_right gives the index above, therefore -1.
        idx = bisect.bisect_right(raster_years, year) - 1
        if idx < 0:
            year = raster_years[0]
            logger.info(f"Dyn. luse, year is earlier than the inputfiles. Use: {year}")
        else:
            year = raster_years[idx]
            logger.info(f"Dynamic land use, select file with year: {year}")

        landuse_file = landuse_filename.format(year=year)
        soilm_scp_file = root_soilm_scp_filename.format(year=year)
        soilm_pwp_file = root_soilm_pwp_filename.format(year=year)
        imperv_file = impervdens_filename.format(year=year)
    else:
        landuse_file = landuse_filename
        soilm_scp_file = root_soilm_scp_filename
        soilm_pwp_file = root_soilm_pwp_filename
        imperv_file = impervdens_filename

    landuse_path = flp.setup.resolve_raster(input_sources, indir_rasters, landuse_file)
    reader = flp.io.GeoTiffReader(landuse_path, nodata_value=0)
    landuse_map = reader.read_and_reproject(**grid_params)

    if "x10" in soilm_scp_file.lower():
        conv = 0.1
    else:
        conv = 1.0
    scp_path = flp.setup.resolve_raster(input_sources, indir_rasters, soilm_scp_file)
    reader = flp.io.GeoTiffReader(scp_path, nodata_value=-9999)
    soilm_scp_raw = reader.read_and_reproject(**grid_params).astype(np.float32)
    soilm_scp = soilm_scp_raw * conv
    soilm_scp[soilm_scp_raw == -9999] = -9999

    if "x10" in soilm_pwp_file.lower():
        conv = 0.1
    else:
        conv = 1.0
    pwp_path = flp.setup.resolve_raster(input_sources, indir_rasters, soilm_pwp_file)
    reader = flp.io.GeoTiffReader(pwp_path, nodata_value=-9999)
    soilm_pwp_raw = reader.read_and_reproject(**grid_params).astype(np.float32)
    soilm_pwp = soilm_pwp_raw * conv
    soilm_pwp[soilm_pwp_raw == -9999] = -9999

    # 0 should be treated as 0, not as no data. Therefore dummy nodata_value 255.
    imperv_path = flp.setup.resolve_raster(input_sources, indir_rasters, imperv_file)
    reader = flp.io.GeoTiffReader(imperv_path, nodata_value=0)
    imperv = reader.read_and_reproject(**grid_params).astype(np.float32) / 100.0

    # # Mask open water and sea
    # mask = (landuse_map == 16) | (landuse_map == 17)
    # soilm_scp[mask] = float("nan")
    # soilm_pwp[mask] = float("nan")

    # Compute beta parameter map for soil evaporation
    beta = np.full(np.shape(landuse_map), 0.038, dtype=np.float32)

    # specify a lower beta param for bare soil
    bare_mask = np.isin(landuse_map, bare_soil_ids)
    beta[bare_mask] = 0.02

    # scale for the urban area.
    urban_mask = np.isin(landuse_map, urban_ids)
    beta[urban_mask] = (0.038 - 0.01) * (1 - imperv[urban_mask]) + 0.01

    # Warn for unexpected land use codes
    for code in np.unique(landuse_map):
        if code not in luse_ids and code != 0:
            logger.warning(f"Land use code {code} not in luse-evap conversion table.")

    logger.info("Read basic FluxPark input maps")

    return landuse_map, soilm_scp, soilm_pwp, imperv, beta


class EvapParamDict(TypedDict):
    trans_fact: NDArray[np.floating]
    soil_evap_fact: NDArray[np.floating]
    int_cap: NDArray[np.floating]
    soil_cov: NDArray[np.floating]
    openwater_fact: NDArray[np.floating]


def apply_evaporation_parameters(
    luse_ids: NDArray[np.integer],
    evap_ids: NDArray[np.integer],
    evap_params: dict[str, np.ndarray],
    doy: int,
    landuse_map: NDArray[np.integer],
    imperv: NDArray[np.floating],
    urban_ids: list[int],
    *,
    mod_vegcover: bool = False,
    soil_cov_decid: Optional[NDArray[np.floating]] = None,
    soil_cov_conif: Optional[NDArray[np.floating]] = None,
) -> EvapParamDict:
    """
    Apply evaporation parameters based on land use and day of year.

    Parameters
    ----------
    luse_ids : ndarray
        Array of land use IDs.
    evap_ids : ndarray
        Array of evaporation parameter IDs.
    evap_params : dict[str, ndarray]
        Dictionary with evaporation parameters per ``evap_id`` and day of
        year.
    doy : int
        Day-of-year for the current timestep.
    landuse_map : ndarray
        Map with land use class IDs.
    imperv : ndarray
        Map with impervious fractions.
    urban_ids : list[int]
        Array of land use IDs considered urban.
    mod_vegcover : bool, optional
        If True, apply vegetation cover corrections.
    soil_cov_decid : ndarray, optional
        Map with spatial vegetation cover for deciduous forests.
    soil_cov_conif : ndarray, optional
        Map with spatial vegetation cover for coniferous forests.

    Returns
    -------
    EvapParamDict
        Dictionary with evaporation parameter arrays for the current timestep.
    """
    # 0) allocate outputs
    shape = landuse_map.shape
    trans_fact = np.zeros(shape, dtype="float32")
    soil_evap_fact = np.zeros(shape, dtype="float32")
    int_cap = np.zeros(shape, dtype="float32")
    soil_cov = np.zeros(shape, dtype="float32")
    openwater_fact = np.zeros(shape, dtype="float32")

    # 1) pull out only the rows for this doy
    mask_doy = evap_params["doy"] == doy
    ep = {k: v[mask_doy] for k, v in evap_params.items()}

    # 2) build direct lookup tables indexed by evap_id
    max_id = int(landuse_map.max()) + 1
    tf_map = np.zeros(max_id, dtype="float32")
    se_map = np.zeros(max_id, dtype="float32")
    ic_map = np.zeros(max_id, dtype="float32")
    sc_map = np.zeros(max_id, dtype="float32")
    ow_map = np.zeros(max_id, dtype="float32")

    # fill those tables in one go
    for lid, eid in zip(luse_ids, evap_ids):
        # find the row index in ep where evap_id==eid
        i = np.nonzero(ep["evap_id"] == eid)[0][0]
        tf_map[lid] = ep["trans_fact"][i]
        se_map[lid] = ep["soil_evap_fact"][i]
        ic_map[lid] = ep["int_cap"][i]
        sc_map[lid] = ep["soil_cov"][i]
        ow_map[lid] = ep["openwater_fact"][i]

    # cast to integer indices
    luse_idx = landuse_map.astype(np.int32)

    # 3) vectorized assignment
    trans_fact = tf_map[luse_idx]
    soil_evap_fact = se_map[luse_idx]
    int_cap = ic_map[luse_idx]
    soil_cov = sc_map[luse_idx]
    openwater_fact = ow_map[luse_idx]

    # 4) special impervious correction for cfg.urban_ids
    urban_mask = np.isin(luse_idx, urban_ids)
    if urban_mask.any():
        tf = trans_fact[urban_mask] * (1 - imperv[urban_mask])
        trans_fact[urban_mask] = tf

        sef = soil_evap_fact[urban_mask]
        scf = soil_cov[urban_mask]
        corr = imperv[urban_mask] * (1 / (1 - scf) * sef - sef)
        soil_evap_fact[urban_mask] = sef + corr

        ic = int_cap[urban_mask] * (1 - imperv[urban_mask])
        ic[ic < 0.2] = 0.2
        int_cap[urban_mask] = ic

    if mod_vegcover and soil_cov_conif is not None and soil_cov_decid is not None:
        for luse_id in (11, 12, 19):
            if luse_id == 11:
                cover_map = soil_cov_decid
            else:
                cover_map = soil_cov_conif

            mask = (landuse_map == luse_id) & (~np.isnan(cover_map))
            max_table_cov = np.max(
                evap_params["soil_cov"][evap_params["evap_id"] == luse_id]
            )
            conv_fac = cover_map[mask] / max_table_cov
            soil_cov[mask] = soil_cov[mask] * conv_fac

    result: EvapParamDict = {
        "trans_fact": trans_fact,
        "soil_evap_fact": soil_evap_fact,
        "int_cap": int_cap,
        "soil_cov": soil_cov,
        "openwater_fact": openwater_fact,
    }

    return result
