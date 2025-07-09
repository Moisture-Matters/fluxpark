import fluxpark as flp
from fluxpark.submodels.interception import interception_voortman
from fluxpark.submodels.soilevaporation import soilevap_boestenstroosnijder
from fluxpark.submodels.rootwateruptake import unsat_reservoirmodel
import numpy as np

def compute_fluxes(
    mak,
    rain,
    int_cap,
    soil_cov,
    old_int_store,
    beta,
    old_sum_ep,
    old_sum_ea,
    landuse_map,
    soil_evap_fact,
    trans_fact,
    eta_trans_frac,
    soilm_scp,
    soilm_pwp,
):
    """
    Compute all of the FluxPark water‐flux components for one timestep.

    Parameters
    ----------
    mak : ndarray
        Makkink reference evapotranspiration [L/day].
    rain : ndarray
        Precipitation [L/day].
    int_cap : ndarray
        Interception capacity [L].
    soil_cov : ndarray
        Soil cover fraction [-].
    old_int_store : ndarray
        Previous interception store [L].
    beta : ndarray
        Soil evaporation shape factor [-].
    old_sum_ep : ndarray
        Previous cumulative soil evaporation [L].
    old_sum_ea : ndarray
        Previous cumulative actual evapotranspiration [L].
    landuse_map : ndarray
        Land‐use classification map.
    soil_evap_fact : ndarray
        Soil evaporation potential factor [-].
    trans_fac : ndarray
        transpiration potential crop factor [-].
    eta_trans_frac : ndarray
        Scratch array for eta→trans split.
    soilm_scp, soilm_pwp : ndarray
        Soil moisture at SCP and PWP [L].

    Returns
    -------
    tuple
        (int_evap,
         int_store,
         throughfall,
         int_timefrac,
         soil_evap_act_est,
         sum_ep,
         sum_ea,
         eta,
         smdp,
         smda,
         prec_surplus,
         eta_trans_frac,
         trans_pot,
         trans_act,
         soil_evap_act,
         open_water_evap_act,
         trans_def,
         evap_total_act,
         evap_total_pot,
         soilm_root,
         prec_def_knmi)
    """
    shape = landuse_map.shape
    # 1) interception
    int_evap, int_store, throughfall, int_timefrac = (
        interception_voortman(
            mak * 1.25 + int_cap,
            rain,
            int_cap,
            soil_cov,
            old_int_store,
        )
    )

    # 2) soil evaporation
    soil_evap_fact_act = (1.0 - soil_cov) * soil_evap_fact
    soil_evap_pot = mak*soil_evap_fact_act
    soil_evap_act_est, sum_ep, sum_ea = (
        soilevap_boestenstroosnijder(
            throughfall, soil_evap_pot,
            beta, old_sum_ep, old_sum_ea
        )
    )

    # 3) unsat reservoir
    trans_pot = mak * trans_fact * soil_cov * (1-int_timefrac)
    eta, smdp, smda, prec_surplus = (
        unsat_reservoirmodel(
            throughfall,
            soil_evap_act_est + mak * soil_cov,  # use trans_pot placeholder
            smda := old_sum_ep * 0,               # unused here
            soilm_scp,
            soilm_pwp,
        )
    )

    # 4) split η into transpiration vs soil evaporation
    numerator = mak * soil_cov + soil_evap_act_est  # T_pot + E_pot
    np.place(eta_trans_frac, numerator == 0, 0.0)
    np.place(eta_trans_frac, numerator != 0,
             (mak * soil_cov)[numerator != 0] / numerator[numerator != 0])
    trans_act = eta * eta_trans_frac
    soil_evap_act = eta - trans_act

    # 5) open‐water evaporation
    open_water_evap_act = np.zeros(shape, dtype="float32")
    open_water_evap_act[landuse_map == 16] = mak[landuse_map == 16] * 1.25
    open_water_evap_act[landuse_map ==  8] = mak[landuse_map ==  8] * 1.10

    # 6) transpiration deficit
    trans_def = mak * soil_cov - trans_act

    # 7) totals
    stacked_pot = np.dstack((soil_evap_act_est, mak * soil_cov,
                             int_evap, open_water_evap_act))
    stacked_act = np.dstack((soil_evap_act, trans_act,
                             int_evap, open_water_evap_act))
    evap_total_pot = np.nansum(stacked_pot, axis=2)
    evap_total_act = np.nansum(stacked_act, axis=2)

    # 8) root‐zone moisture
    soilm_root = soilm_pwp - smda

    # 9) KNMI precipitation deficit
    prec_def_knmi = (rain - mak) * -1

    return (
        int_evap,
        int_store,
        throughfall,
        int_timefrac,
        soil_evap_act_est,
        sum_ep,
        sum_ea,
        eta,
        smdp,
        smda,
        prec_surplus,
        eta_trans_frac,
        mak * soil_cov,  # trans_pot
        trans_act,
        soil_evap_act,
        open_water_evap_act,
        trans_def,
        evap_total_act,
        evap_total_pot,
        soilm_root,
        prec_def_knmi,
    )
