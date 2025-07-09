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
    # 1. Compute transpiration fraction and actual transpiration
    frac = np.zeros_like(trans_pot)
    num = trans_pot + soil_evap_act_est
    nz = num != 0
    frac[nz] = trans_pot[nz] / num[nz]
    trans_act = eta * frac

    # 2. Actual soil evaporation and transpiration deficit
    soil_evap_act = eta - trans_act
    trans_def = trans_pot - trans_act

    # 3. Masks for open water / greenhouse and cities
    mask_open = (landuse_map == 16) | (landuse_map == 8)
    mask_city = landuse_map == 18

    # 4. Copy and mask arrays
    eta_out = eta.copy()
    int_out = int_evap.copy()
    pot_soil_out = soil_evap_pot.copy()
    est_soil_out = soil_evap_act_est.copy()
    pot_trans_out = trans_pot.copy()
    act_trans_out = trans_act.copy()
    def_trans_out = trans_def.copy()
    prec_sur_out = prec_surplus.copy()
    smda_out = smda.copy()

    for arr in (
        eta_out,
        int_out,
        pot_soil_out,
        soil_evap_act,
        est_soil_out,
        pot_trans_out,
        act_trans_out,
        prec_sur_out,
    ):
        arr[mask_open] = np.nan

    def_trans_out[mask_city] = np.nan

    # 5. Soil moisture deficit floor
    bad = (smda_out < 0) & (smda_out != -9999)
    smda_out[bad] = 0

    # 6. Total evaporation (nan‐safe sum over axis=2)
    evap_total_act = np.nansum(
        np.dstack((soil_evap_act, act_trans_out, int_out, open_water_evap_act)),
        axis=2,
    )
    evap_total_pot = np.nansum(
        np.dstack((est_soil_out, pot_trans_out, int_out, open_water_evap_act)),
        axis=2,
    )

    # 7. Rootzone moisture and precipitation deficit
    soilm_root = soilm_pwp - smda_out
    prec_def_knmi = (rain - etref) * -1

    return {
        "trans_act": act_trans_out,
        "soil_evap_act": soil_evap_act,
        "trans_def": def_trans_out,
        "evap_total_act": evap_total_act,
        "evap_total_pot": evap_total_pot,
        "soilm_root": soilm_root,
        "prec_def_knmi": prec_def_knmi,
        "eta": eta_out,
        "int_evap": int_out,
        "soil_evap_pot": pot_soil_out,
        "soil_evap_act_est": est_soil_out,
        "trans_pot": pot_trans_out,
        "prec_surplus": prec_sur_out,
        "smda": smda_out,
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
    if current_date.day == 1 and current_date.month == 4:
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
