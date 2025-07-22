"""
example_run_rain_etref_constant.py

A minimal example demonstrating how to configure and run FluxPark
with custom rain and ETref input hooks.
"""

import fluxpark as flp
import numpy as np

# configuration
cfg = flp.config.FluxParkConfig(
    date_start="2021-01-01",
    date_end="2021-01-10",
    calc_epsg_code=28992,
    x_min=81000.0,
    x_max=152000.0,
    y_min=454000.0,
    y_max=580000.0,
    cellsize=100,
    evap_param_table="20250708_evap_parameters.xlsx",
    output_files=["prec_surplus_mm_d", "evap_total_act_mm_d"],
    indir="./input_data",
    outdir="./output_data")

# define rain as constant (3.0 mm/d)
def rain_grid(date, grid_params):
    rain = np.full((grid_params['nrows'], grid_params['ncols']), 3.0)
    return rain

# define etref as constant (1.0 mm/d)
def etref_grid(date, grid_params):
    etref = np.full((grid_params['nrows'], grid_params['ncols']), 1.0)
    return etref

# run the model
runner = flp.FluxParkRunner(
    cfg, input_hooks={"get_rain": rain_grid, "get_etref": etref_grid}
)
runner.run()