import numpy as np
import datetime as dt
from typing import Tuple


def post_process_daily(
    eta,
    trans_pot,
    soil_evap_act_est,
    int_evap,
    soil_evap_pot,
    open_water_evap_act,
    smda,
    soilm_pwp,
    rain,
    etref,
    landuse_map,
    prec_surplus,
    open_water_ids,
):
    """
    Compute actual fluxes, mask invalid areas, and derive water balances.

    Parameters
    ----------
    eta : ndarray
        Total evapotranspiration (mm).
    trans_pot : ndarray
        Potential transpiration (mm).
    soil_evap_act_est : ndarray
        Estimated potential soil evaporation (mm).
    int_evap : ndarray
        Interception evaporation (mm).
    soil_evap_pot : ndarray
        Potential soil evaporation factor array (mm).
    open_water_evap_act : ndarray
        Open‐water actual evaporation (mm).
    smda : ndarray
        Soil moisture deficit actual (mm).
    soilm_pwp : ndarray
        Permanent wilting point moisture (mm).
    rain : ndarray
        Precipitation (mm).
    etref : ndarray
        Reference evapotranspiration (mm).
    landuse_map : ndarray of int
        Land‐use codes.
    prec_surplus : ndarray
        Precipitation surplus (mm).
    open_water_ids : list[int]
        Reservoir output get nan for these landuse ids, if None no masking

    Returns
    -------
    dict of ndarray
        Keys:
        - trans_act: actual transpiration (mm)
        - soil_evap_act: actual soil evaporation (mm)
        - trans_def: transpiration deficit (mm)
        - evap_total_act: total actual evaporation (mm)
        - evap_total_pot: total potential evaporation (mm)
        - soilm_root: root‐zone moisture content (mm)
        - prec_def_knmi: KNMI precipitation deficit (mm)
        - eta: masked total evapotranspiration
        - int_evap: masked interception evaporation
        - soil_evap_pot: masked potential soil evaporation
        - soil_evap_act_est: masked estimated soil evap.
        - trans_pot: masked potential transpiration
        - prec_surplus: masked precipitation surplus
        - smda: non‐negative soil moisture deficit
    """
    # 1. Compute transpiration fraction, trans_act and soil_evap_act
    num = trans_pot + soil_evap_act_est
    frac = np.zeros_like(trans_pot, dtype="float32")
    np.divide(trans_pot, num, out=frac, where=(num != 0))
    trans_act = eta * frac
    soil_evap_act = eta - trans_act

    # 3. Masks for open water
    if open_water_ids:
        mask_open = np.isin(landuse_map, open_water_ids)

        # 4. all output from the reservoir model is masked for open water and sea.
        # if you don't mask the prec_surplus will be equal with precipitation for open
        # water and other variables will have a 0 (not nan)
        for arr in (
            eta,
            int_evap,
            trans_pot,
            trans_act,
            soil_evap_pot,
            soil_evap_act_est,
            soil_evap_act,
            prec_surplus,
            smda,
        ):
            arr[mask_open] = np.nan

    # calculate transpiration deficit
    trans_def = trans_pot - trans_act

    # 5. non-negative soil moisture deficit
    smda = np.where((smda < 0) & (smda != -9999), 0.0, smda)

    # 6. Total evaporation
    evap_total_pot = np.nansum(
        [soil_evap_act_est, trans_pot, int_evap, open_water_evap_act], axis=0
    )
    evap_total_act = np.nansum(
        [soil_evap_act, trans_act, int_evap, open_water_evap_act], axis=0
    )

    # 7. Rootzone moisture and precipitation deficit
    soilm_root = soilm_pwp - smda
    prec_def_knmi = (rain - etref) * -1.0

    return {
        "eta": eta,
        "int_evap": int_evap,
        "trans_pot": trans_pot,
        "trans_act": trans_act,
        "soil_evap_pot": soil_evap_pot,
        "soil_evap_act_est": soil_evap_act_est,
        "soil_evap_act": soil_evap_act,
        "prec_surplus": prec_surplus,
        "smda": smda,
        "trans_def": trans_def,
        "evap_total_act": evap_total_act,
        "evap_total_pot": evap_total_pot,
        "soilm_root": soilm_root,
        "prec_def_knmi": prec_def_knmi,
    }


def update_cumulative_fluxes(
    daily_output: dict[str, np.ndarray],
    old: dict[str, np.ndarray],
    current_date: dt.date,
    reset_cum_day: int,
    reset_cum_month: int,
    cum_par_list: list[str],
    conv_output: dict[str, str],
) -> Tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Reset and accumulate daily fluxes into yearly and KNMI‐defined sums.

    Parameters
    ----------
    daily_output
        Dict of daily arrays, keys are Python names without '_c' suffix.
    old
        Dict of 2D arrays holding previous cumulative values (keys end with '_c').
    current_date
        Date of current simulation step.
    reset_cum_day, reset_cum_month
        Day and month to reset yearly cumulative sums.
    cum_par_list
        List of output‐keys for cumulative variables, e.g.
        ['prec_def_knmi_cum_ytd_mm', 'trans_act_c', ...].
    conv_output
        Mapping from output‐keys in cum_par_list to Python keys in `old`.

    Returns
    -------
    cum
        Dict mapping Python '_c' keys to updated cumulative arrays.
        Also updates `old` in place.
    """
    # 1. Yearly reset (except KNMI deficit)
    if current_date.day == reset_cum_day and current_date.month == reset_cum_month:
        for output_key in cum_par_list:
            if output_key == "prec_def_knmi_cum_ytd_mm":
                continue
            py_key = conv_output[output_key]
            old[py_key] = np.zeros_like(old[py_key])

    # 2. KNMI precip deficit reset on April 1
    if (
        "rain_def_pot_etref_c" in old
        and current_date.day == 1
        and current_date.month == 4
    ):
        old["rain_def_pot_etref_c"] = np.zeros_like(old["rain_def_pot_etref_c"])

    # 3. Accumulate
    cum: dict[str, np.ndarray] = {}
    for output_key in cum_par_list:
        py_key = conv_output[output_key]
        # drop '_c' suffix to get daily key
        daily_key = py_key[:-2]
        cum_val = old[py_key] + daily_output[daily_key]
        old[py_key] = cum_val
        cum[py_key] = cum_val

    return cum, old
