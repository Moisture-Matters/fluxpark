import fluxpark as flp
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import logging
import bisect

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
    input_raster_years,
    imperv,
    luse_ids,
    bare_soil_ids,
    urban_ids,
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
    input_raster_years : list of str
        List of years with available input maps.
    imperv : ndarray
        Map with impervious fractions (used for beta).
    luse_ids : list of int
        List of valid land use IDs.
    bare_soil_ids : list of int
        List of land use IDs that are bare and should get a lower beta param.
    urban_ids : list of int
        List of land use IDs that are urban and should be scaled using the imperv.

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
            logging.info(f"Dyn. luse, year is earlier than the inputfiles. Use: {year}")
        else:
            year = raster_years[idx]
            logging.info(f"Dynamic land use, select file with year: {year}")

        landuse_file = landuse_filename.format(year=year)
        soilm_scp_file = root_soilm_scp_filename.format(year=year)
        soilm_pwp_file = root_soilm_pwp_filename.format(year=year)
    else:
        landuse_file = landuse_filename
        soilm_scp_file = root_soilm_scp_filename
        soilm_pwp_file = root_soilm_pwp_filename

    reader = flp.io.GeoTiffReader(indir_rasters / landuse_file, nodata_value=0)
    landuse_map = reader.read_and_reproject(**grid_params)

    reader = flp.io.GeoTiffReader(indir_rasters / soilm_scp_file, nodata_value=-9999)
    soilm_scp = reader.read_and_reproject(**grid_params).astype(np.float32)

    reader = flp.io.GeoTiffReader(indir_rasters / soilm_pwp_file, nodata_value=-9999)
    soilm_pwp = reader.read_and_reproject(**grid_params).astype(np.float32)

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
    beta[urban_mask] = (0.038 - 0.02) * (1 - imperv[urban_mask]) + 0.02

    # Warn for unexpected land use codes
    for code in np.unique(landuse_map):
        if code not in luse_ids and code != 0:
            logging.warning(f"Land use code {code} not in luse-evap conversion table.")

    logging.info("Read basic FluxPark input maps")

    return landuse_map, soilm_scp, soilm_pwp, beta


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
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Apply evaporation parameters based on land use and day of year.

    Parameters
    ----------
    luse_ids : ndarray
        Array of land use IDs.
    evap_ids : ndarray
        Array of evaporation parameter IDs.
    evap_params : DataFrame
        Table with evaporation parameters per evap_id and day.
    doy : ndarray
        Day-of-year for the current timestep.
    landuse_map : ndarray
        Map with land use class IDs.
    imperv : ndarray
        Map with impervious fractions.
    trans_fact : ndarray
        Output array to be filled with transpiration factors.
    soil_evap_fact : ndarray
        Output array to be filled with soil evaporation factors.
    int_cap : ndarray
        Output array to be filled with interception capacities.
    soil_cov : ndarray
        Output array to be filled with soil cover fractions.
    mod_vegcover : bool, optional
        If True, apply vegetation cover corrections.
    soil_cov_decid : ndarray, optional
        Map with spatial vegetation cover for deciduous forests.
    soil_cov_conif : ndarray, optional
        Map with spatial vegetation cover for coniferous forests.
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

    # 4) special impervious correction for landuse 18
    urban_mask = np.isin(luse_idx, urban_ids)
    if urban_mask.any():
        tf = trans_fact[urban_mask] * (1 - imperv[urban_mask])
        trans_fact[urban_mask] = tf

        sef = soil_evap_fact[urban_mask]
        scf = soil_cov[urban_mask]
        corr = imperv[urban_mask] * (1 / (1 - scf) - sef)
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

    return trans_fact, soil_evap_fact, int_cap, soil_cov, openwater_fact
