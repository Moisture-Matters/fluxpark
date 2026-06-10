"""
example_run_rain_etref_constant.py

A minimal example demonstrating how to configure and run FluxPark
with custom rain and ETref adapters supplying constant values.
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
def my_rain_provider(runner):
    shape = (runner.grid_params["nrows"], runner.grid_params["ncols"])
    return np.full(shape, 3.0, dtype="float32")

# define etref as constant (1.0 mm/d)
def my_etref_provider(runner):
    shape = (runner.grid_params["nrows"], runner.grid_params["ncols"])
    return np.full(shape, 1.0, dtype="float32")

# run the model
runner_ports = flp.RunnerPorts(
    rain_provider=my_rain_provider,
    etref_provider=my_etref_provider,
)

runner = flp.FluxParkRunner(cfg, runner_ports=runner_ports)
runner.run()
