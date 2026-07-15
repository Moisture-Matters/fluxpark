"""Tests for eval_waterbalance helpers (remote-safe land-use loading)."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import types  # noqa: E402

import numpy as np  # noqa: E402

import importlib  # noqa: E402

import fluxpark as flp  # noqa: E402

# the package re-exports the same-named FUNCTION, shadowing the module
# attribute; import the module itself explicitly.
ewb = importlib.import_module("fluxpark.postprocessing.eval_waterbalance")


class _FakeReader:
    """Captures the path handed to GeoTiffReader without touching disk."""

    last_path = None

    def __init__(self, path, dst_nodata=0):
        _FakeReader.last_path = path

    def read_and_reproject(self, **kwargs):
        return np.zeros((2, 2), dtype=np.float32)


def _stub_runner(indir_rasters, input_sources=None):
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(landuse_rastername="{year}_luse_ids.tif"),
        input_raster_years=np.array(["2023", "2024"]),
        input_sources=input_sources,
        indir_rasters=indir_rasters,
        grid_params={},
    )


def test_luse_map_path_is_url_safe(monkeypatch):
    # remote input: indir_rasters is a URL string; the path must be joined
    # URL-safely (a Path '/' join raises TypeError on str and mangles '://').
    monkeypatch.setattr(flp.io, "GeoTiffReader", _FakeReader)
    runner = _stub_runner("https://example.com/releases/v1/rasters")
    out = ewb._load_luse_map_for_year(runner, 2024)
    assert _FakeReader.last_path == (
        "https://example.com/releases/v1/rasters/2024_luse_ids.tif"
    )
    assert out.dtype == np.int32


def test_luse_map_picks_most_recent_available_year(monkeypatch):
    monkeypatch.setattr(flp.io, "GeoTiffReader", _FakeReader)
    runner = _stub_runner("https://example.com/rasters")
    ewb._load_luse_map_for_year(runner, 2030)   # beyond newest -> use 2024
    assert _FakeReader.last_path.endswith("/2024_luse_ids.tif")
