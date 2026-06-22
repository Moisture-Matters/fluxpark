"""Tests for path/URL helpers used to support local and remote inputs."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

from pathlib import Path  # noqa: E402

from fluxpark.utils.common import (  # noqa: E402
    has_placeholders,
    is_url,
    join_path_or_url,
    to_gdal_path,
)


def test_is_url_recognises_remote_and_vsi():
    assert is_url("https://example.com/a.tif")
    assert is_url("http://example.com/a.tif")
    assert is_url("ftp://example.com/a.tif")
    assert is_url("/vsicurl/https://example.com/a.tif")


def test_is_url_rejects_local_paths():
    assert not is_url("C:\\data\\a.tif")
    assert not is_url("/home/user/a.tif")
    assert not is_url("./a.tif")


def test_to_gdal_path_wraps_urls():
    assert (
        to_gdal_path("https://example.com/a.tif")
        == "/vsicurl/https://example.com/a.tif"
    )


def test_to_gdal_path_passes_through_vsi():
    vsi = "/vsicurl/https://example.com/a.tif"
    assert to_gdal_path(vsi) == vsi


def test_to_gdal_path_normalises_backslashes():
    assert to_gdal_path("C:\\data\\a.tif") == "C:/data/a.tif"


def test_join_path_or_url_keeps_url_a_string():
    joined = join_path_or_url("https://example.com/data/", "sub", "a.tif")
    assert joined == "https://example.com/data/sub/a.tif"


def test_join_path_or_url_returns_path_for_local():
    joined = join_path_or_url("data", "sub", "a.tif")
    assert isinstance(joined, Path)
    assert joined == Path("data", "sub", "a.tif")


def test_has_placeholders():
    assert has_placeholders("{year}_luse.tif")
    assert not has_placeholders("plain.tif")
