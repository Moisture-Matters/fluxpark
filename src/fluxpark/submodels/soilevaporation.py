"""
Computation of soil evaporation.

This module provides functions for estimating soil evaporation based on rain or
throughfall and potential evaporation.

Functions
---------
soilevap_boestenstroosnijder(throughfall, epot, beta_m, sum_ep_old, sum_ea_old)
    Calculates actual soil evaporaiton.
"""

import numpy as np


def soilevap_boestenstroosnijder(throughfall, epot, beta_m, sum_ep_old,
                                 sum_ea_old):
    """
    Calculate actual soil evaporation.

    Based on the method published by Boesten and Stroosnijder (1986).
    Supports scalar, 1D, and 2D array inputs but at least all in the same shape

    Parameters
    ----------
    throughfall : float or ndarray
        Throughfall [mm].
    epot : float or ndarray
        Potential soil evaporation [mm].
    beta_m : float or ndarray
        Boesten parameter in [m^0.5]; e.g., 0.5 in SWAP.
    sum_ep_old : float or ndarray
        Previous cumulative potential evaporation [mm].
    sum_ea_old : float or ndarray
        Previous cumulative actual evaporation [mm].

    Returns
    -------
    ea : float or ndarray
        Actual soil evaporation [mm].
    sum_ep : float or ndarray
        Updated cumulative potential evaporation [mm].
    sum_ea : float or ndarray
        Updated cumulative actual evaporation [mm].
    """
    # Ensure all inputs are arrays (at least 1D)
    tf = np.atleast_1d(throughfall)
    ep = np.atleast_1d(epot)
    beta = np.atleast_1d(beta_m)
    sum_ep_old = np.atleast_1d(sum_ep_old)
    sum_ea_old = np.atleast_1d(sum_ea_old)

    # Convert beta to mm^0.5
    beta_mm05 = np.sqrt((beta ** 2) * 1000.0)

    # Initialize arrays
    ea = np.zeros_like(tf, dtype="float32")
    sum_ep = sum_ep_old.copy()
    sum_ea = sum_ea_old.copy()

    # Determine condition for reset
    reset = (tf - ep) > sum_ep_old
    sum_ep[reset] = 0.0
    sum_ea[reset] = 0.0

    # Condition 1: no excess rain
    cond1 = tf < ep
    delta = ep - tf
    sum_ep[cond1] += delta[cond1]
    evap_limit = beta_mm05 * np.sqrt(sum_ep)
    over_limit = cond1 & (sum_ep > evap_limit)
    sum_ea[cond1] = sum_ep[cond1]
    sum_ea[over_limit] = evap_limit[over_limit]
    ea[cond1] = tf[cond1] + sum_ea[cond1] - sum_ea_old[cond1]
    ea[ea < 0.0] = 0.0

    # Update cumulative values for cond1
    sum_ep_old[cond1] = sum_ep[cond1]
    sum_ea_old[cond1] = sum_ea[cond1]

    # Condition 2: excess rain
    cond2 = ~cond1
    ea[cond2] = ep[cond2]
    sum_ea[cond2] = sum_ea_old[cond2] - (tf[cond2] - ea[cond2])
    sum_ea[cond2] = np.maximum(sum_ea[cond2], 0.0)
    sum_ep_calc = (sum_ea[cond2] ** 2) / (beta_mm05[cond2] ** 2)
    sum_ep[cond2] = np.maximum(sum_ep_calc, sum_ea[cond2])

    # Update cumulative values for cond2
    sum_ep_old[cond2] = sum_ep[cond2]
    sum_ea_old[cond2] = sum_ea[cond2]

    # Return same shape as input
    if throughfall.ndim == 1:
        return ea.flatten(), sum_ep.flatten(), sum_ea.flatten()
    else:
        return ea, sum_ep, sum_ea
