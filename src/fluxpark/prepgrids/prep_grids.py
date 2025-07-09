import fluxpark as flp
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import logging


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
        if str(year) not in input_raster_years:
            year = input_raster_years[-1]
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

    # Mask open water and sea
    mask = (landuse_map == 16) | (landuse_map == 17)
    soilm_scp[mask] = float("nan")
    soilm_pwp[mask] = float("nan")

    # Compute beta parameter map for soil evaporation
    beta = np.full(
        (grid_params["nrows"], grid_params["ncols"]), 0.038, dtype=np.float32
    )
    beta[landuse_map == 15] = 0.02
    beta[landuse_map == 18] = (0.038 - 0.02) * (1 - imperv[landuse_map == 18]) + 0.02

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
    # initiate array's
    trans_fact = np.zeros(landuse_map.shape, dtype="float32")
    soil_evap_fact = np.zeros(landuse_map.shape, dtype="float32")
    int_cap = np.zeros(landuse_map.shape, dtype="float32")
    soil_cov = np.zeros(landuse_map.shape, dtype="float32")
    openwater_fact = np.zeros(landuse_map.shape, dtype="float32")
    for luse_id in luse_ids:
        evap_id = evap_ids[luse_ids == luse_id].item()
        is_id_and_doy = (evap_params["evap_id"] == evap_id) & (
            evap_params["doy"] == doy
        )

        trans_fact[landuse_map == luse_id] = evap_params["trans_fact"][is_id_and_doy]
        soil_evap_fact[landuse_map == luse_id] = evap_params["soil_evap_fact"][
            is_id_and_doy
        ]
        int_cap[landuse_map == luse_id] = evap_params["int_cap"][is_id_and_doy]
        soil_cov[landuse_map == luse_id] = evap_params["soil_cov"][is_id_and_doy]
        openwater_fact[landuse_map == luse_id] = evap_params["openwater_fact"][
            is_id_and_doy
        ]

        if luse_id == 18:
            mask = landuse_map == luse_id
            tf = trans_fact[mask] * (1 - imperv[mask])
            trans_fact[mask] = tf

            scf = soil_cov[mask]
            sef = soil_evap_fact[mask]
            correction = imperv[mask] * (1 / (1 - scf) - sef)
            soil_evap_fact[mask] = sef + correction

            ic = int_cap[mask] * (1 - imperv[mask])
            ic[ic < 0.2] = 0.2
            int_cap[mask] = ic

        if mod_vegcover and soil_cov_conif is not None and soil_cov_decid is not None:
            if luse_id in [11, 12, 19]:
                if luse_id == 11:
                    cover_map = soil_cov_decid
                else:
                    cover_map = soil_cov_conif

                mask = (landuse_map == luse_id) & (~np.isnan(cover_map))
                max_table_cov = np.max(
                    evap_params["soil_cov"][evap_params["evap_id"] == evap_id]
                )
                conv_fac = cover_map[mask] / max_table_cov
                soil_cov[mask] = evap_params["soil_cov"][is_id_and_doy] * conv_fac
    return trans_fact, soil_evap_fact, int_cap, soil_cov, openwater_fact
