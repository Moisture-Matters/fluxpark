# FluxPark  
**A spatially explicit hydrological model for simulating evaporation fluxes & groundwater recharge**

### Overview  
FluxPark is an open‑source Python library that transforms daily meteorological data and land‑use maps into spatially distributed evaporation and recharge estimates. It's built for:

- **Speed & simplicity**: Few parameters and efficient computation make it ideal for rapid calibration and custom adaptation.  
- **Process separation**: By modelling interception, transpiration and soil evaporation separately, FluxPark reacts realistically to changing weather and soil moisture conditions.  
- **Modularity**: A core simulation engine can be extended with optional modules—perfect for groundwater modellers looking for a lightweight, transparent tool.

FluxPark is developed and maintained by **[Moisture Matters](https://www.moisture-matters.nl)** 🌍. While the core model is freely available on GitHub and PyPI, regional inputs (maps, parameter tables) are delivered as part of our support contract. Your subscription helps fund ongoing maintenance and future enhancements.

### Enterprise & SaaS  
For operational water management, FluxPark is integrated with satellite imagery, weather forecasts and field sensors in our digital‑twin platform **[StellaSpark Nexus](https://www.stellaspark.com/drought-early-warning-system)** 🌍—offered as a SaaS solution for real‑time monitoring and scenario analysis.

<img src="docs/FluxPark_fluxes.png" width="600" alt="FluxPark evaporation & recharge workflow"/>

## Installation
Install FluxPark using pip

```bash
pip install fluxpark
```
### Dependencies
> [!IMPORTANT]
> In addition to installing FluxPark with `pip`, you **must** have the GDAL library installed on your system.  
The simplest approach is to run:
 
```bash
conda install -c conda-forge gdal
```

- GDAL >= 3.10.2
- Python >= 3.9

If you have difficulty getting GDAL and FluxPark set up, we provide a ready‑made conda environment file:
1. Download `flp_environment.yml` from the `docs/` folder.  
2. In your terminal, navigate to its directory and run:
```bash
conda env create -f flp_environment.yml
```
3. Activate it
```bash
conda activate fluxpark_env
```
4. Open your editor (e.g. VS Code or Spyder) and select this environment as your interpreter.
This will ensure all dependencies —including GDAL and Python— are installed correctly.

## Usage
FluxPark requires spatial raster maps for:

- **Land‑use IDs**  
- **Plant available soil moisture between field capacity** and **permanent wilting point**  
- **Plant available soil moisture between field capacity** and **stomatal closure point** 

It also needs three tables:

1. **`fluxpark_output_mapping.csv`**  
2. **`conv_luse_evap_ids.csv`**  
3. An Excel workbook containing daily evaporation parameters (soil coverage, interception capacity, and crop factors for transpiration, soil evaporation, and interception evaporation).

A unique feature of FluxPark is that all evaporation parameters can be *scaled* by soil cover, enabling higher‑resolution, spatially explicit simulations.

The core simulation is orchestrated by the **FluxParkRunner**, which handles setup and time‑stepping. Users configure the model via the **FluxParkConfig** class. Inspect its docstring to explore all available options.

FluxPark uses a **ports and adapters** pattern to decouple data sources from the model core. The `RunnerPorts` dataclass defines all input and output connections; built‑in adapters are available under `flp.adapters`. You can also write your own adapters to supply e.g. meteorological data.

### Input data - versioned releases

FluxPark input data is organised as **versioned releases**: a release folder described by a `release.yml`, grouped under a *line*. Point `indir` at a line with an `{input_version}` placeholder and set `input_version` to the release:

```python
indir = "./releases/nweu/{input_version}"
input_version = "2025.06.0__full"
```

The evaporation parameter table and the other inputs are then taken from the release automatically. See **[docs/input_data_releases.md](docs/input_data_releases.md)** for the full folder structure and conventions.

A plain `indir` folder without the placeholder also works (the legacy method); in that case pass the evaporation parameters explicitly via `evap_param_table="evap_parameters.xlsx"`.

Each input is resampled by type (land use by mode, soil moisture by median, imperviousness and forest cover by average) and nodata is normalised to `NaN` internally. See **[docs/nodata_and_resampling.md](docs/nodata_and_resampling.md)** for the per‑parameter resampling and nodata rules, and how the output is masked (`mask_open_water`, `write_nan_for_landuse_ids`, `replace_nan_with_zero`).

### Example 1 - KNMI NetCDF files (built‑in adapter)

Note that these NetCDF files can contain NaN values for open water; therefore `nan_policy` is set to `"allow"` to prevent the raster validation from raising an error.

```python
import fluxpark as flp

cfg = flp.config.FluxParkConfig(
    date_start="2021-01-01",
    date_end="2021-01-10",
    calc_epsg_code=28992,
    x_min=81000.0,
    x_max=152000.0,
    y_min=454000.0,
    y_max=580000.0,
    cellsize=100,
    output_files=["prec_surplus_mm_d", "evap_total_act_mm_d"],
    indir="./releases/nweu/{input_version}",
    input_version="2025.06.0__full",
    outdir="./output_data",
    nan_policy="allow",
)

runner_ports = flp.RunnerPorts(
    rain_provider=flp.adapters.make_knmi_netcdf_rain_provider(
        r"./input_data/knmi/prec"
    ),
    etref_provider=flp.adapters.make_knmi_netcdf_etref_provider(
        r"./input_data/knmi/etref"
    ),
)

runner = flp.FluxParkRunner(cfg, runner_ports=runner_ports)
runner.run()
```

### Example 2 - custom adapter

You can supply any data source by writing a provider function that accepts the runner object and returns a 2D NumPy array. The runner object exposes `runner.date` (current date) and `runner.grid_params` (grid dimensions and projection).

```python
import fluxpark as flp
import numpy as np

cfg = flp.config.FluxParkConfig(
    date_start="2021-01-01",
    date_end="2021-01-10",
    calc_epsg_code=28992,
    x_min=81000.0,
    x_max=152000.0,
    y_min=454000.0,
    y_max=580000.0,
    cellsize=100,
    output_files=["prec_surplus_mm_d", "evap_total_act_mm_d"],
    indir="./releases/nweu/{input_version}",
    input_version="2025.06.0__full",
    outdir="./output_data",
)

def my_rain_provider(runner):
    # Return a constant rain grid of 3.0 mm/d
    shape = (runner.grid_params["nrows"], runner.grid_params["ncols"])
    return np.full(shape, 3.0, dtype="float32")

def my_etref_provider(runner):
    # Return a constant ETref grid of 1.0 mm/d
    shape = (runner.grid_params["nrows"], runner.grid_params["ncols"])
    return np.full(shape, 1.0, dtype="float32")

runner_ports = flp.RunnerPorts(
    rain_provider=my_rain_provider,
    etref_provider=my_etref_provider,
)

runner = flp.FluxParkRunner(cfg, runner_ports=runner_ports)
runner.run()
```

### Example 3 - water balance evaluation

Setting `eval_waterbalance=True` in the configuration ensures that all required
output parameters are automatically added to the output list. The model also
writes the initial soil moisture state and serializes the configuration to the
output directory at the start of every run.

```python
import fluxpark as flp

cfg = flp.config.FluxParkConfig(
    date_start="2021-01-01",
    date_end="2021-12-31",
    calc_epsg_code=28992,
    x_min=81000.0,
    x_max=152000.0,
    y_min=454000.0,
    y_max=580000.0,
    cellsize=100,
    indir="./releases/nweu/{input_version}",
    input_version="2025.06.0__full",
    outdir="./output_data",
    eval_waterbalance=True,
)

runner = flp.FluxParkRunner(cfg)
runner.run()
```

The evaluation runs automatically at the end of the simulation and writes
`waterbalance_eval.csv` to the output directory. The CSV contains the
cumulative water balance components and the residual error per date and
land-use class.

**Running the evaluation standalone** on an existing output directory:

```python
import fluxpark as flp

flp.postprocessing.eval_waterbalance("./output_data")
```

This requires `fluxpark_cfg.json` to be present in the output directory, which
is written automatically to the output directory at the start of every run.

**Converting output rasters to timeseries** without running the full evaluation:

```python
import fluxpark as flp
import pandas as pd

dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")

df = flp.postprocessing.rasters_to_timeseries(
    outdir="./output_data",
    parameters=["prec_cum_ytd_mm", "evap_total_act_cum_ytd_mm", "recharge_cum_ytd_mm"],
    dates=dates,
)
```

To break down results by land-use class, supply a land-use map. Use
`GeoTiffReader` with `read_and_reproject` to ensure the land-use raster aligns
exactly with the model grid:

```python
import fluxpark as flp
import numpy as np
import pandas as pd

grid_params = {
    "dst_epsg": 28992,
    "bounds": (81000.0, 152000.0, 454000.0, 580000.0),
    "cellsize": 100,
}

reader = flp.io.GeoTiffReader("./input_data/rasters/2021_luse_ids.tif", nodata_value=0)
luse_map = reader.read_and_reproject(**grid_params).astype(np.int32)

dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")

df = flp.postprocessing.rasters_to_timeseries(
    outdir="./output_data",
    parameters=["prec_cum_ytd_mm", "recharge_cum_ytd_mm"],
    dates=dates,
    luse_map=luse_map,
)
```

## License
FluxPark is released under the **FluxPark Custom License v1.0**.  
See [LICENSE.txt](LICENSE.txt) for the full terms and conditions.
