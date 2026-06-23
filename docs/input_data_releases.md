# FluxPark input data — versioned releases

FluxPark input data is organised as **versioned releases**. A release is a
folder described by a `release.yml` file. The same layout works for a local
folder and for an HTTPS server (for example a WebDAV server): the structure is
identical, you simply upload it.

## Structure

```
releases/
└── <line>/                          e.g. nweu
    ├── latest                       text file: name of the newest release
    ├── <version>[__<label>]/        a release
    │   ├── release.yml              describes this release
    │   ├── rasters/                 only the files this release provides
    │   └── tables/
    └── ...
```

## Two axes: line and version

- **Line** (`<line>/`): a dataset identity with its own timeline and its own
  `latest`. Create a separate line for a conceptually different product
  (different coverage, customer or purpose) — **not** for a different CRS,
  because FluxPark warps every input to the run's target grid anyway.
- **Version** (`<version>`): `YEAR.MONTH.SEQ`, purely numeric. Only this part is
  used for sorting and for selecting the most recent release. Examples:
  `2025.06.0`, `2025.06.1`, `2025.07.0`.

## Label

Optional, after a double underscore: `2025.06.0__full`,
`2025.07.1__landuse_scenario_x`. It is free, human-readable and ignored when
sorting. Handy to recognise a full revision (`__full`) or a scenario at a
glance.

## Full vs. partial releases (`extends`)

- **Full**: declares everything and has no `extends`.
- **Partial**: declares only what it changes and points to the release it
  builds on with `extends: "<release>"`. `extends` is transitive: a release
  building on a partial release also inherits everything below it. Resolution:
  for each file the most derived release in the chain that declares it wins;
  everything else is inherited. No copies are made.

## Selecting the newest release (`latest`)

The newest release is chosen via the `latest` pointer file, not via "the
highest number lying around". A scenario can therefore be inserted safely
without it being picked as the newest release automatically.

## Coupling with FluxParkConfig

Point `indir` at a line using an `{input_version}` placeholder and set
`input_version` to the release (including any label):

```python
indir = "./releases/nweu/{input_version}"   # the line is part of indir
input_version = "2025.06.0__full"           # the release (incl. label)
```

The evaporation parameter table and the other inputs are then taken from the
release automatically. An `indir` without the placeholder also keeps working
(the legacy method); in that case the evaporation parameters are supplied
explicitly via `evap_param_table`.

## Masks

Masks are **not** versioned. They live in a plain folder (`cfg.indir_masks`),
are customer- or project-specific, and are not listed in `release.yml`.

## Example: a full release

```yaml
version: "2025.06.0__full"
line: "nweu"
description: "Full set, years 2018-2024."
crs: "EPSG:3035"          # informational; a run's target CRS lives in cfg.

rasters:
  # Yearly rasters: each 'type' exists for every year in `years`.
  yearly:
    years: [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    types:
      - name: luse_ids
        pattern: "{year}_luse_ids.tif"
      - name: impervdens
        pattern: "{year}_impervdens.tif"
      - name: root_soilm_fc_pwp_mm_x10
        pattern: "{year}_root_soilm_fc_pwp_mm_x10.tif"
      - name: root_soilm_fc_scp_mm_x10
        pattern: "{year}_root_soilm_fc_scp_mm_x10.tif"

  # Static rasters (no year).
  static:
    types:
      - name: forest_conif_soilcov_pct
        file: "forest_conif_soilcov_pct.tif"
      - name: forest_decid_soilcov_pct
        file: "forest_decid_soilcov_pct.tif"

tables:
  - name: evap_parameters
    file: "evap_parameters.xlsx"
  - name: conv_luse_evap_ids
    file: "conv_luse_evap_ids.csv"
  - name: fluxpark_output_mapping
    file: "fluxpark_output_mapping.csv"
```

## Example: a partial release (update one table)

Replaces only the evaporation parameters; everything else is inherited from
`2025.06.0__full`. Physically this folder contains just the new Excel file
under `tables/`.

```yaml
version: "2025.07.0"
line: "nweu"
extends: "2025.06.0__full"
description: "New evaporation parameters."

tables:
  - name: evap_parameters
    file: "evap_parameters.xlsx"
```

## Example: a scenario release (replace one map)

Replaces only the 2024 land-use map. By declaring solely `luse_ids` with
`years: [2024]`, this release provides just `(luse_ids, 2024)`; everything else
comes from `2025.06.0__full` via `extends`.

```yaml
version: "2025.07.1__landuse_scenario_x"
line: "nweu"
extends: "2025.06.0__full"
description: "Scenario: modified 2024 land-use map."

rasters:
  yearly:
    years: [2024]
    types:
      - name: luse_ids
        pattern: "{year}_luse_ids.tif"
```
