"""
Computation of reference crop evapotranspiration (ETref).

This module provides functions for estimating daily evapotranspiration based on
staandard meteorological variables. The formulas are suited for temperate
climates like those in Western Europe.

Functions
---------
makkink(tair, rs_in)
    Calculates reference evapotranspiration using the Makkink equation.
"""

import numpy as np


def makkink(tair, rs_in):
    """
    Calculate Makkink reference evapotranspiration.

    Estimates the daily reference evapotranspiration based on air temperature
    and incoming solar radiation, using the Makkink equation. Commonly used in
    hydrological and agricultural models in temperate regions.

    Parameters
    ----------
    tair : float or np.ndarray
        Daily mean air temperature in degrees Celsius (°C).
    rs_in : float or np.ndarray
        Daily incoming global solar radiation in joules per square centimeter
        (J/cm²).

    Returns
    -------
    et_makkink : float or np.ndarray
        Reference evapotranspiration in millimeters per day (mm/day).

    where Δ is the slope of the saturation vapor pressure curve,
    γ is the psychrometric constant, λ is the latent heat of vaporization,
    and ρw is the density of water.
    """
    # Saturation vapour pressure (hPa)
    es = 6.107 * 10 ** (7.5 * tair / (237.3 + tair))
    # Slope of saturation vapour pressure curve (hPa/°C)
    d = 7.5 * 237.3 / ((237.3 + tair) ** 2) * np.log(10) * es
    # Psychrometric constant (hPa/°C)
    gamma = 0.646 + 0.0006 * tair
    # Latent heat of vaporization (J/kg)
    Lv = 1000.0 * (2501 - 2.38 * tair)
    # Convert incoming radiation to J/m²
    rs_in_j = rs_in * 10_000.0
    # Reference evapotranspiration (mm/day)
    et_makkink = (1000.0 * 0.65 * d) / ((d + gamma) * 1000.0 * Lv) * rs_in_j

    return et_makkink
