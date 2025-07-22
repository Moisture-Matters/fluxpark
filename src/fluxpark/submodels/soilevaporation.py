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

    # Determine condition for reset
    reset_mask = (tf - ep) > sum_ep_old
    sum_ep = np.where(reset_mask, 0.0, sum_ep_old)
    sum_ea = np.where(reset_mask, 0.0, sum_ea_old)

    # Condition 1: no excess rain
    cond1 = tf < ep
    sum_ep = np.where(cond1, sum_ep + (ep - tf), sum_ep)
    evap_limit = beta_mm05 * np.sqrt(sum_ep, dtype="float32")
    sum_ea = np.where(cond1, np.minimum(sum_ep, evap_limit), sum_ea)

    ea1 = tf + (sum_ea - sum_ea_old)
    ea = np.where(cond1, np.clip(ea1, 0.0, None), ep)

    # Condition 2: excess rain
    # For cond2, update sum_ea = max(old_sum_ea - (tf-ep),0)
    sum_ea2 = np.maximum(sum_ea_old - (tf - ep), 0.0)
    sum_ea = np.where(~cond1, sum_ea2, sum_ea)

    # For cond2, update sum_ep = max(sum_ea^2/(beta_mm05^2), sum_ea)
    sum_ep2 = np.maximum(sum_ea * sum_ea / (beta_mm05 * beta_mm05), sum_ea)
    sum_ep = np.where(~cond1, sum_ep2, sum_ep)

    # Return
    if np.ndim(throughfall) == 1:
        return ea.flatten(), sum_ep.flatten(), sum_ea.flatten()
    return ea, sum_ep, sum_ea
