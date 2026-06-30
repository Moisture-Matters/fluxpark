# Resampling and nodata handling

How FluxPark reads each input, how nodata is represented through the run, and
how the output is masked. The aim is one clear rule per stage.

## 1. Reading and resampling

Every raster input is read with `GeoTiffReader.read_and_reproject`, which warps
the source onto the run grid (target CRS, cell size, extent, cutline). The
resampling method is chosen per parameter type:

| Input | Resampling | Why |
|---|---|---|
| Land use (`luse_ids`) | `mode` | categorical: take the dominant class in the cell |
| Soil moisture FC–SCP / FC–PWP (`*_x10`) | `med` (median) | robust central value of a continuous field |
| Imperviousness (`impervdens`) | `average` | mean impervious density; 0% counts as data |
| Forest soil cover (`*_soilcov_pct`) | `average` | mean cover fraction over the cell |
| Rain, ETref (meteo, via adapters) | `bilinear` (default) | smooth continuous meteo fields |

The aggregating methods (`mode`, `med`, `average`) ignore the **source** nodata,
so masked source pixels never bias the result. Reads always use the base
resolution (`overviewLevel="NONE"`); the source overviews are never used, as
averaged overviews would blend nodata into valid cells near edges.

## 2. Normalising nodata on read

`GeoTiffReader(..., dst_nodata=N)` sets the nodata of the **returned array**
(the warp destination), not the source's — it pairs with the `src_nodata`
argument of `read_and_reproject`, which overrides the source band's nodata. The
warp honours the source nodata separately (it may be `0` on legacy data or `255`
on cleaned data — see [input_data_releases.md](input_data_releases.md)). So
whatever the source uses is normalised to a single internal convention per
parameter:

| Input | Internal value after read | Internal nodata |
|---|---|---|
| Land use | integer class id | `0` (sentinel; an int array cannot hold NaN) |
| Soil moisture | mm (`raw × 0.1` for `_x10`) | `NaN` |
| Imperviousness | fraction `0–1` (`÷100`) | none — outside coverage set to `0` (0%) |
| Forest soil cover | fraction `0–1` (`÷100`) | `NaN` ("no cover here") |
| Rain, ETref | mm | `NaN`, handled by `nan_policy` |

The float rasters are read with `dst_nodata=np.nan`: `read_and_reproject`
then warps to Float32 and returns `NaN` where nodata, which never collides with
a valid `0` (e.g. 0% impervious). This is the same NaN convention the meteo
reader already uses, so nodata is uniform across the float inputs.

Two parameters do something extra after the read:

* **Land use** stays an integer array, so it keeps the sentinel `0` (this is why
  `write_nan_for_landuse_ids` defaults to `[0, 17]`).
* **Imperviousness** sets `NaN → 0` (0% impervious), because 0% applies
  everywhere, including outside the data coverage.

`nan_policy` (`"error"` | `"allow"` | `"skip"`, default `"error"`) controls what
happens when a rain or ETref raster contains `NaN`: raise, keep the NaNs, or
skip that simulation day.

## 3. Nodata in the calculation

The unsaturated-zone reservoir model runs on **all** pixels, but where the soil
parameters are `NaN` it has no reservoir, so it **returns `NaN`** for those
pixels (`eta`, `smdp`, `smda`, `drainage`). The nodata therefore propagates
cleanly through the carried soil-moisture state instead of becoming a
meaningless value. (On a year boundary, where the inputs reload, a pixel that
was NaN may become valid again; the runner resets a carried-over NaN deficit to
`0` so the now-valid pixel starts accumulating afresh.)

`post_process_daily` then defines the cells that have no soil reservoir and
masks the remaining fluxes:

```
open_water  = land use in open_water_ids        (config open_water_ids, default [16])
soil_nodata = np.isnan(soilm_pwp) | np.isnan(soilm_scp)
no_reservoir = open_water | soil_nodata
```

and then:

* all soil/land fluxes (`eta`, transpiration, soil evaporation, `smda`,
  `soilm_root`, `trans_def`, `prec_surplus`, ...) are set to `NaN` where
  `no_reservoir`;
* total evaporation is built with `nansum`, so **open-water evaporation is kept**
  over open water even though the soil fluxes there are NaN;
* a real land pixel without soil parameters (`data_gap = soil_nodata &
  ~open_water`) has its total evaporation set to `NaN` as well.

So a nodata pixel is `NaN` from the reservoir onward; post-processing masks the
remaining cases (open water, data gaps). Nodata is never a sentinel value that
silently participates in the fluxes.

## 4. Masking the output

Three configuration options shape what ends up in the written rasters. They act
at different stages:

| Option | Default | Stage | Effect |
|---|---|---|---|
| `mask_open_water` | `True` | post-processing | `True`: reservoir outputs over open water are `NaN`. `False`: keep the precipitation surplus over open water (`prec_surplus = rain`); open-water evaporation is kept either way. |
| `write_nan_for_landuse_ids` | `[0, 17]` | writing | Set the output to `NaN` for these land-use ids — `0` (land-use nodata) and `17` (sea). |
| `replace_nan_with_zero` | `False` | writing | `True`: turn any remaining `NaN` into `0` instead of nodata. Use when a downstream consumer wants 0 rather than nodata. |

The write step then applies, in order:

1. forest cover fractions are scaled back to percent (`soil_cov × 100`);
2. `write_nan_for_landuse_ids` → `NaN` for the listed classes;
3. `replace_nan_with_zero` → `NaN` becomes `0` when enabled;
4. any `NaN` still left becomes **`-9999`**, the nodata value of the written
   GeoTIFF;
5. values are rounded to 3 decimals.

So in the delivered output a pixel is either a value, `0` (only when
`replace_nan_with_zero=True`), or `-9999` (the GeoTIFF nodata). Internally the
model speaks `NaN`; `-9999` only appears at the very end, on disk.
