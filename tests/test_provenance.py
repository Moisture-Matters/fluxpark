"""Tests for provenance: input-version resolution and GeoTIFF metadata."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402
from osgeo import gdal  # noqa: E402

from fluxpark.io.raster_tools import write_geotiff  # noqa: E402
from fluxpark.setup.input_sources import (  # noqa: E402
    InputSources,
    build_provenance,
    resolve_input_version,
)


def _sources(version):
    return InputSources(version=version, line="demo", years=[2020])


def test_release_version_wins():
    assert resolve_input_version(_sources("2025.06.0"), "ignored") == "2025.06.0"


def test_config_version_used_without_release():
    assert resolve_input_version(None, "2025.01.0") == "2025.01.0"


def test_unknown_when_nothing_specified():
    assert resolve_input_version(None, None) == "unknown"


def test_build_provenance_has_expected_keys():
    prov = build_provenance(_sources("2025.06.0"), None, "1.2.3")
    assert prov["FLUXPARK_VERSION"] == "1.2.3"
    assert prov["FLUXPARK_INPUT_VERSION"] == "2025.06.0"
    assert prov["FLUXPARK_CREATED"]  # non-empty timestamp


def test_metadata_is_written_into_geotiff(tmp_path):
    meta = {"FLUXPARK_VERSION": "1.2.3", "FLUXPARK_INPUT_VERSION": "2025.06.0"}
    write_geotiff(
        str(tmp_path),
        "out.tif",
        np.zeros((2, 2), dtype=np.float32),
        0.0,
        0.0,
        1.0,
        28992,
        metadata=meta,
    )
    ds = gdal.Open(str(tmp_path / "out.tif"))
    read_back = ds.GetMetadata()
    ds = None
    assert read_back["FLUXPARK_VERSION"] == "1.2.3"
    assert read_back["FLUXPARK_INPUT_VERSION"] == "2025.06.0"
