"""Tests for the input-sources loader and extends resolution."""

import json
import textwrap
from pathlib import Path

import fluxpark as flp


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
