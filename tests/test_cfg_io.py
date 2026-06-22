"""Tests for FluxParkConfig serialization round-trip."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

from fluxpark.config import FluxParkConfig, save_cfg, load_cfg  # noqa: E402


def _make_cfg():
    return FluxParkConfig(
        date_start="01-01-2020",
        date_end="31-12-2020",
        calc_epsg_code=28992,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
        cellsize=25.0,
        input_version="2025.01.0",
        output_files=["prec_mm_d", "evap_total_act_mm_d"],
    )


def test_save_load_round_trip(tmp_path):
    cfg = _make_cfg()
    save_cfg(cfg, tmp_path)
    loaded = load_cfg(tmp_path)
    assert loaded == cfg


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cfg(tmp_path)
