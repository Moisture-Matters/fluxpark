"""
Computation of root water uptake.

This module provides functions to estimate soil moisture deficit and actual
transpiration due to root water uptake.

Functions
---------
unsat_reservoirmodel(rain, etp, smd_old, rc, pwp)
    Calculates actual transpiration.
"""

import numpy as np


def unsat_reservoirmodel(rain, etp, smd_old, soilm_scp, soilm_pwp):
    """
    Unsaturated zone model to estimate actual transpiration.

    Simulates actual evapotranspiration (eta), potential soil moisture deficit
    (smdp), updated soil moisture deficit (smda), and drainage using a simple
    unsaturated zone reservoir model. Based on: Ireson, A.M., Butler, A.P.,
    2013. A critical assessment of simple recharge models. Hydrol. Earth Syst.
    Sci., 17(6): 2083–2096.

    Parameters
    ----------
    rain : ndarray
        Precipitation or throughfall input [L per day]. 1D or 2D array.
    etp : ndarray
        Potential evapotranspiration from the unsaturated zone [L per day].
    smd_old : ndarray
        Previous soil moisture deficit [L].
    soilm_scp : ndarray
        Stomatal closure point [L]. The root zone soil moisture deficit at
        which ET starts to reduce.
    soilm_pwp : ndarray
        Permanent wilting point [L]. The root zone soil moisture deficit at
        which ET ceases entirely.

    Returns
    -------
    eta : ndarray
        Actual evapotranspiration [L per day], same shape as input.
    smdp : ndarray
        Potential soil moisture deficit [L].
    smda : ndarray
        Updated soil moisture deficit [L].
    drainage : ndarray
        Drainage out of the unsaturated zone [L per day].
    """
    # Input validation
    if not (
        rain.shape == etp.shape
        and etp.shape == smd_old.shape
        and smd_old.shape == soilm_scp.shape
        and soilm_scp.shape == soilm_pwp.shape
    ):
        raise ValueError("All input arrays must have the same shape.")

    # Get input shape
    shape = rain.shape

    # Initialize output arrays
    eta = np.zeros(shape, dtype=np.float32)
    drainage = np.zeros(shape, dtype=np.float32)
    smda = np.zeros(shape, dtype=np.float32)

    # Compute potential soil moisture deficit
    smdp = smd_old - rain + etp

    # Evaporation conditions
    cond1 = smdp <= soilm_scp
    cond2 = (smdp > soilm_scp) & (smdp < soilm_pwp)
    # cond3 = ~(cond1 | cond2)

    # Calculate actual evapotranspiration
    eta[cond1] = etp[cond1]
    eta[cond2] = etp[cond2] * (
        (soilm_pwp[cond2] - smdp[cond2]) / (soilm_pwp[cond2] - soilm_scp[cond2])
    )

    # eta[cond3] remains zero

    # Drainage and updated SMD
    cond_drain = smdp < 0
    cond_no_drain = ~cond_drain

    drainage[cond_drain] = -smdp[cond_drain]
    # drainage[cond_no_drain] remains zero

    smda[cond_drain] = 0.0
    smda[cond_no_drain] = (
        smd_old[cond_no_drain] - rain[cond_no_drain] + eta[cond_no_drain]
    )

    return eta, smdp, smda, drainage
