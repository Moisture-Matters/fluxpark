"""Tests for the input-sources loader and extends resolution."""

import json
import textwrap
from pathlib import Path

import pytest

# These tests exercise the FluxPark package, which imports GDAL (osgeo) on
# import. GDAL is provided via conda, not via the pip requirements, so skip the
# whole module when it is unavailable (e.g. the pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402
from osgeo import gdal, osr  # noqa: E402

import fluxpark as flp  # noqa: E402


def _make_tif(path: Path, value: int) -> None:
    """Write a tiny constant Int16 GeoTIFF in EPSG:3035."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, 1, gdal.GDT_Int16)
    ds.SetGeoTransform((3900000, 100, 0, 3100000, 0, -100))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3035)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(np.full((4, 4), value, dtype=np.int16))
    ds = None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def _make_line(root: Path) -> Path:
    """Create a small 'nweu' line with a full and two partial releases."""
    line = root / "releases" / "nweu"

    _write(
        line / "2025.06.0__full" / "release.yml",
        """
        version: "2025.06.0__full"
        line: "nweu"
        rasters:
          yearly:
            years: [2018, 2019, 2020]
            types:
              - name: luse_ids
                pattern: "{year}_luse_ids.tif"
          static:
            types:
              - name: forest_conif_soilcov_pct
                file: "forest_conif_soilcov_pct.tif"
        tables:
          - name: evap_parameters
            file: "20251128_evap_parameters.xlsx"
        """,
    )

    _write(
        line / "2025.07.0" / "release.yml",
        """
        version: "2025.07.0"
        line: "nweu"
        extends: "2025.06.0__full"
        tables:
          - name: evap_parameters
            file: "20260701_evap_parameters.xlsx"
        """,
    )

    _write(
        line / "2025.07.1__landuse_scenario_x" / "release.yml",
        """
        version: "2025.07.1__landuse_scenario_x"
        line: "nweu"
        extends: "2025.07.0"
        rasters:
          yearly:
            years: [2020]
            types:
              - name: luse_ids
                pattern: "{year}_luse_ids.tif"
        """,
    )
    return line


def test_full_release(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.06.0__full")

    assert src.version == "2025.06.0__full"
    assert src.line == "nweu"
    assert src.years == [2018, 2019, 2020]
    assert Path(src.raster_path("2020_luse_ids.tif")).name == "2020_luse_ids.tif"
    assert "2025.06.0__full" in str(src.raster_path("2020_luse_ids.tif"))


def test_partial_table_replace(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.07.0")

    # table from this release, raster inherited from the base
    assert "2025.07.0" in str(src.table_path("20260701_evap_parameters.xlsx"))
    assert "2025.06.0__full" in str(src.raster_path("2019_luse_ids.tif"))
    assert src.years == [2018, 2019, 2020]


def test_transitive_extends_single_raster(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.07.1__landuse_scenario_x")

    # 2020 from the scenario; 2019 transitively from the full base
    assert "2025.07.1__landuse_scenario_x" in str(
        src.raster_path("2020_luse_ids.tif")
    )
    assert "2025.06.0__full" in str(src.raster_path("2019_luse_ids.tif"))
    # evap_parameters comes from the middle release (2025.07.0)
    assert "2025.07.0" in str(src.table_path("20260701_evap_parameters.xlsx"))


def test_sources_snapshot(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.07.1__landuse_scenario_x")
    src.write_sources_snapshot(tmp_path)

    snap = json.loads(
        (tmp_path / "fluxpark_input_sources.json").read_text(encoding="utf-8")
    )
    assert snap["input_version"] == "2025.07.1__landuse_scenario_x"
    assert snap["line"] == "nweu"
    rasters = snap["resolved"]["rasters"]
    assert rasters["2020_luse_ids.tif"] == "2025.07.1__landuse_scenario_x"
    assert rasters["2019_luse_ids.tif"] == "2025.06.0__full"
    assert snap["resolved"]["tables"]["evap_parameters"] == "2025.07.0"


def test_no_release_yml_returns_none(tmp_path):
    assert flp.setup.load_input_sources(tmp_path) is None


def test_resolve_indir_placeholder():
    resolved, line_root = flp.setup.resolve_indir(
        r"C:/data/releases/nweu/{input_version}", "2026.06.0__full"
    )
    assert resolved == r"C:/data/releases/nweu/2026.06.0__full"
    assert line_root == "C:/data/releases/nweu"


def test_resolve_indir_direct_consistent():
    # direct path + matching input_version -> allowed, no line root
    resolved, line_root = flp.setup.resolve_indir(
        r"C:/data/releases/nweu/2026.06.0__full", "2026.06.0__full"
    )
    assert str(resolved).endswith("2026.06.0__full")
    assert line_root is None


def test_resolve_indir_mismatch_raises():
    try:
        flp.setup.resolve_indir(
            r"C:/data/releases/nweu/2026.06.0__full", "2026.10.1"
        )
    except RuntimeError as exc:
        assert "input_version" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for indir/version mismatch")


def test_resolve_indir_placeholder_without_version_raises():
    try:
        flp.setup.resolve_indir(r"C:/data/releases/nweu/{input_version}", None)
    except RuntimeError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing input_version")


def test_resolve_indir_latest_reads_pointer(tmp_path):
    line = tmp_path / "nweu"
    line.mkdir()
    (line / "latest").write_text("2026.06.0__full\n", encoding="utf-8")
    resolved, line_root = flp.setup.resolve_indir(
        str(line / "{input_version}"), "latest"
    )
    assert str(resolved).endswith("2026.06.0__full")
    assert Path(line_root) == line


def test_resolve_indir_latest_without_placeholder_raises():
    try:
        flp.setup.resolve_indir(r"C:/data/releases/nweu", "latest")
    except RuntimeError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for 'latest' without placeholder")


def test_prepare_inputs_with_version_but_unreadable_release_raises(tmp_path):
    # an explicit input_version must not silently fall back to the legacy
    # folder layout when release.yml cannot be read (e.g. missing credentials
    # on remote input, or a wrong path).
    version_dir = tmp_path / "2026.06.0__full"
    version_dir.mkdir()  # exists, but contains no release.yml
    cfg = flp.config.FluxParkConfig(
        date_start="01-01-2021",
        date_end="02-01-2021",
        calc_epsg_code=28992,
        x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0,
        cellsize=25.0,
        indir=str(tmp_path / "{input_version}"),
        input_version="2026.06.0__full",
        outdir=str(tmp_path / "out"),
    )
    try:
        flp.setup.prepare_inputs(cfg)
    except RuntimeError as exc:
        assert "release.yml" in str(exc)
        assert "input_version" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unreadable release.yml")


def test_detect_dynamic_includes_impervdens(tmp_path):
    # a forgotten static impervdens map must be caught by the consistency
    # check, not later as a confusing "raster not declared" error.
    (tmp_path / "2024_luse_ids.tif").touch()
    static = dict(
        landuse_filename="2024_luse_ids.tif",
        root_soilm_scp_filename="2024_root_soilm_fc_scp_mm_x10.tif",
        root_soilm_pwp_filename="2024_root_soilm_fc_pwp_mm_x10.tif",
        indir_rasters=tmp_path,
    )

    try:
        flp.setup.detect_dynamic_landuse_and_years(
            **static, impervdens_filename="{year}_impervdens.tif"
        )
    except RuntimeError as exc:
        assert "impervdens" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for mixed static/dynamic")

    dynamic, years = flp.setup.detect_dynamic_landuse_and_years(
        **static, impervdens_filename="2024_impervdens.tif"
    )
    assert dynamic is False
    assert list(years) == ["2024"]


def test_input_context_close_removes_temp_dir(tmp_path):
    # close() must remove the download dir explicitly and be idempotent, so
    # cleanup never relies on garbage collection (ResourceWarning in e.g.
    # long-lived processes or containers).
    import tempfile

    tmp = tempfile.TemporaryDirectory(prefix="fluxpark_input_")
    ctx = flp.setup.InputContext(
        outdir=tmp_path, tables=tmp_path, rasters=tmp_path, masks=tmp_path,
        intermediate=None, input_sources=None,
        download_dir=tmp.name, _tmp=tmp,
    )
    download_dir = Path(tmp.name)
    assert download_dir.exists()
    ctx.close()
    assert not download_dir.exists()
    assert ctx.download_dir is None
    ctx.close()  # second call is a no-op

    # a context without a temp dir (local inputs) is also fine
    flp.setup.InputContext(
        outdir=tmp_path, tables=tmp_path, rasters=tmp_path, masks=tmp_path,
        intermediate=None, input_sources=None, download_dir=None,
    ).close()


def test_resolve_indir_latest_missing_pointer_raises(tmp_path):
    line = tmp_path / "nweu"
    line.mkdir()  # no 'latest' file present
    try:
        flp.setup.resolve_indir(str(line / "{input_version}"), "latest")
    except RuntimeError as exc:
        assert "latest" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a missing 'latest' file")


def test_table_path_by_name_alias(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.07.0")
    # the evap_parameters alias resolves to the overriding release's filename
    p = src.table_path_by_name("evap_parameters")
    assert Path(p).name == "20260701_evap_parameters.xlsx"
    assert "2025.07.0" in str(p)


def test_table_path_by_name_unknown_raises(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.06.0__full")
    try:
        src.table_path_by_name("does_not_exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown alias")


def test_load_conv_output_unknown_mapping_raises(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.06.0__full")
    # the release declares no 'nexus_output_mapping.csv' table
    try:
        flp.setup.load_conv_output(
            line, "nexus_output_mapping.csv", src
        )
    except RuntimeError as exc:
        assert "output_mapping" in str(exc)
        assert "release" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for unknown output_mapping")


def test_load_evap_params_requires_a_source(tmp_path):
    # no release and no filename -> clear error
    try:
        flp.setup.load_evap_params(tmp_path, None, None)
    except RuntimeError as exc:
        assert "evap_param_table" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when no source is given")


def test_resolve_helpers_legacy_and_sources(tmp_path):
    line = _make_line(tmp_path)
    src = flp.setup.load_input_sources(line / "2025.07.0")

    # legacy fallback: no input_sources -> joins onto the given dir
    legacy = flp.setup.resolve_raster(None, Path(r"C:/data/rasters"), "x.tif")
    assert legacy == Path(r"C:/data/rasters/x.tif")

    # with input_sources: resolves through the chain (base raster)
    p = flp.setup.resolve_raster(src, Path("ignored"), "2019_luse_ids.tif")
    assert "2025.06.0__full" in str(p)
    t = flp.setup.resolve_table(src, Path("ignored"), "20260701_evap_parameters.xlsx")
    assert "2025.07.0" in str(t)


def test_extends_override_is_physically_read(tmp_path):
    line = tmp_path / "releases" / "nweu"
    _make_tif(line / "2025.06.0__full" / "rasters" / "2019_luse_ids.tif", 10)
    _make_tif(line / "2025.06.0__full" / "rasters" / "2020_luse_ids.tif", 10)
    _write(
        line / "2025.06.0__full" / "release.yml",
        """
        version: "2025.06.0__full"
        line: "nweu"
        rasters:
          yearly:
            years: [2019, 2020]
            types:
              - name: luse_ids
                pattern: "{year}_luse_ids.tif"
        """,
    )
    _make_tif(line / "2025.07.1__scn" / "rasters" / "2020_luse_ids.tif", 20)
    _write(
        line / "2025.07.1__scn" / "release.yml",
        """
        version: "2025.07.1__scn"
        line: "nweu"
        extends: "2025.06.0__full"
        rasters:
          yearly:
            years: [2020]
            types:
              - name: luse_ids
                pattern: "{year}_luse_ids.tif"
        """,
    )

    src = flp.setup.load_input_sources(line / "2025.07.1__scn")
    grid = dict(
        dst_epsg=3035,
        bounds=(3900000, 3900400, 3099600, 3100000),
        cellsize=100,
    )

    # 2020 overridden by the scenario -> value 20
    p2020 = flp.setup.resolve_raster(src, Path("ignored"), "2020_luse_ids.tif")
    arr2020 = flp.io.GeoTiffReader(p2020, dst_nodata=0).read_and_reproject(**grid)
    assert int(np.round(arr2020.mean())) == 20

    # 2019 inherited from the base -> value 10
    p2019 = flp.setup.resolve_raster(src, Path("ignored"), "2019_luse_ids.tif")
    arr2019 = flp.io.GeoTiffReader(p2019, dst_nodata=0).read_and_reproject(**grid)
    assert int(np.round(arr2019.mean())) == 10


def test_masks_at_line_level_for_release_dir(tmp_path):
    line = _make_line(tmp_path)
    version_dir = line / "2025.06.0__full"  # direct path, no placeholder
    out, tab, ras, msk, _ = flp.setup.resolve_dirs(
        tmp_path / "_out", version_dir
    )
    # masks resolve to the line root (parent), not inside the version folder
    assert Path(msk) == line / "masks"
    assert Path(ras) == version_dir / "rasters"


def test_masks_legacy_folder_without_release(tmp_path):
    plain = tmp_path / "input_data"
    (plain / "rasters").mkdir(parents=True)
    out, tab, ras, msk, _ = flp.setup.resolve_dirs(tmp_path / "_out", plain)
    # no release.yml -> legacy: masks inside the folder
    assert Path(msk) == plain / "masks"


def test_local_shapefile_mask_is_allowed(tmp_path):
    # a local .shp mask passes through unchanged (no download)
    gp = flp.setup.compute_grid_params(
        x_min=0.0, x_max=1000.0, y_min=0.0, y_max=1000.0,
        cellsize=100, epsg_code=28992,
        indir_masks=tmp_path / "masks", mask="NL.shp",
    )
    assert Path(gp["cutline_path"]) == tmp_path / "masks" / "NL.shp"


def test_remote_shapefile_mask_raises():
    try:
        flp.setup.compute_grid_params(
            x_min=0.0, x_max=1000.0, y_min=0.0, y_max=1000.0,
            cellsize=100, epsg_code=28992,
            indir_masks="https://host/releases/nweu/masks", mask="NL.shp",
        )
    except RuntimeError as exc:
        assert "single-file" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a remote .shp mask")


def test_cross_line_extends_raises(tmp_path):
    line = _make_line(tmp_path)
    _write(
        line / "2025.08.0__bad" / "release.yml",
        """
        version: "2025.08.0__bad"
        line: "greece"
        extends: "2025.06.0__full"
        """,
    )
    try:
        flp.setup.load_input_sources(line / "2025.08.0__bad")
    except RuntimeError as exc:
        assert "line" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for cross-line extends")
