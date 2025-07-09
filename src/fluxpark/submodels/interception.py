"""
Computation of interception evaporation.

This module provides functions for estimating interception evaporation based on
staandard meteorological variables and surface/plant properties.

Functions
---------
interception_voortman(etp_wet, rain, max_int, cover, int_store_old)
    Calculates actual interception evaporation and throughfall based on
    Voortman (2015).
"""

import numpy as np


def interception_voortman(etp_wet, rain, max_int, cover, int_store_old):
    """
    Actual interception evaporation and throughfall based on Voortman (2015).

    Calculates actual interception evaporation, updated interception store,
    throughfall and fraction of the timestep with evaporation.

    Parameters
    ----------
    etp_wet : ndarray
        Potential interception evaporation assuming 100% vegetation cover
        (e.g. in mm or m).
    rain : ndarray
        Precipitation input (same unit as etp_wet).
    max_int : ndarray or float
        Maximum interception storage capacity for 100% cover (same unit).
    cover : ndarray or float
        Vegetation cover fraction [m²/m²].
    int_store_old : ndarray
        Interception store at the start of the timestep.

    Returns
    -------
    int_evap : ndarray
        Actual interception evaporation.
    int_store : ndarray
        Updated interception storage at the end of the timestep.
    throughfall : ndarray
        Precipitation not intercepted by vegetation.
    int_timefrac : ndarray
        Fraction of the timestep where interception evaporation occurs.

    Notes
    -----
    All input and output arrays are assumed to be 2D arrays with shape [x, y].
    """
    # Intercepted rain and capacity (adjusted for cover)
    rain_int = rain * cover
    int_cap = max_int * cover
    etp_int = etp_wet * cover

    # Addition to interception storage
    capacity = int_cap - int_store_old
    int_add = np.minimum(rain_int, capacity)
    int_store = int_store_old + int_add

    # Interception evaporation logic (cannot exceed int_store)
    int_evap = np.minimum(etp_int, int_store)

    # Update interception store after evaporation
    int_store = int_store - int_evap

    # Throughfall (rain not intercepted)
    throughfall = rain - int_add

    # Calculate fraction of timestep with interception evaporation
    with np.errstate(divide="ignore", invalid="ignore"):
        int_timefrac = np.divide(
            int_evap, etp_int, out=np.zeros_like(etp_int), where=etp_int != 0
        )

    return int_evap, int_store, throughfall, int_timefrac
