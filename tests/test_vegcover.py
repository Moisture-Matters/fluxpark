"""Tests for the mod_vegcover scaling of the soil cover per land use."""

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import numpy as np  # noqa: E402

import fluxpark as flp  # noqa: E402
from fluxpark.prepgrids import apply_evaporation_parameters  # noqa: E402
from fluxpark.setup.core_initialization import read_static_maps  # noqa: E402
from fluxpark.setup.input_sources import InputSources  # noqa: E402

FOREST_FILES = ("decid.tif", "conif.tif")
DRY_FILE = "dry.tif"

# land use ids in the test grid; every id is its own evap id
LUSE_IDS = np.array([1, 11, 12, 14, 19, 23])
DOYS = (1, 2)
# table soil_cov per land use on doy 1 and doy 2 (the max is on doy 2)
TABLE_COV = {1: (0.9, 0.9), 11: (0.2, 0.6), 12: (0.7, 0.7), 14: (0.5, 0.6),
             19: (0.9, 0.9), 23: (0.3, 0.8)}


def _evap_params():
    rows = [(lid, doy, TABLE_COV[lid][i])
            for lid in LUSE_IDS for i, doy in enumerate(DOYS)]
    evap_id, doy, soil_cov = (np.array(c) for c in zip(*rows))
    one = np.ones(len(rows), dtype="float32")
    return {"evap_id": evap_id, "doy": doy, "soil_cov": soil_cov,
            "trans_fact": one, "soil_evap_fact": one, "int_cap": one,
            "openwater_fact": one}


def _run(doy, landuse_map, cover=None, mod_vegcover=True):
    shape = landuse_map.shape
    maps = cover or {}
    return apply_evaporation_parameters(
        LUSE_IDS, LUSE_IDS, _evap_params(), doy, landuse_map,
        np.zeros(shape, dtype="float32"), urban_ids=[18],
        mod_vegcover=mod_vegcover,
        soil_cov_decid=maps.get("decid"),
        soil_cov_conif=maps.get("conif"),
        soil_cov_drynature=maps.get("drynature"),
    )["soil_cov"]


def _full_cover(shape, decid=0.8, conif=0.4, drynature=0.3):
    return {"decid": np.full(shape, decid, dtype="float32"),
            "conif": np.full(shape, conif, dtype="float32"),
            "drynature": np.full(shape, drynature, dtype="float32")}


def test_dry_nature_peaks_at_map_cover_and_keeps_table_shape():
    # On the day of the table maximum the soil cover equals the map; on other
    # days the ratio to the table is the same as on the peak day.
    landuse = np.array([[14, 14]])
    cover = _full_cover(landuse.shape, drynature=0.3)
    peak = _run(2, landuse, cover)
    other = _run(1, landuse, cover)
    assert np.allclose(peak, 0.3)
    assert np.allclose(other / TABLE_COV[14][0], peak / TABLE_COV[14][1])


def test_nan_cover_keeps_table_values():
    landuse = np.array([[14, 11, 12]])
    cover = _full_cover(landuse.shape)
    for m in cover.values():
        m[:] = np.nan
    soil_cov = _run(2, landuse, cover)
    assert np.allclose(soil_cov, [[TABLE_COV[i][1] for i in (14, 11, 12)]])


def test_without_mod_vegcover_table_values_are_used():
    landuse = np.array([[14, 11, 12, 19]])
    cover = _full_cover(landuse.shape)
    soil_cov = _run(2, landuse, cover, mod_vegcover=False)
    expected = [[TABLE_COV[i][1] for i in (14, 11, 12, 19)]]
    assert np.allclose(soil_cov, expected)


def test_missing_dry_nature_map_still_scales_forests():
    # Without a dry nature map (None) land use 14 keeps its table value while
    # the forests are still scaled with their own maps.
    landuse = np.array([[14, 11, 12]])
    cover = _full_cover(landuse.shape, decid=0.8, conif=0.4)
    del cover["drynature"]
    soil_cov = _run(2, landuse, cover)
    assert np.allclose(soil_cov, [[TABLE_COV[14][1], 0.8, 0.4]])


def test_forest_uses_its_own_map_and_other_classes_untouched():
    # 11 follows the deciduous map, 12 and 19 the coniferous map; land uses
    # without a cover map (here 1 and 23) keep their table value.
    landuse = np.array([[11, 12, 19, 1, 23]])
    cover = _full_cover(landuse.shape, decid=0.8, conif=0.4, drynature=0.3)
    soil_cov = _run(2, landuse, cover)
    expected = [[0.8, 0.4, 0.4, TABLE_COV[1][1], TABLE_COV[23][1]]]
    assert np.allclose(soil_cov, expected)


class _FakeReader:
    """Stands in for GeoTiffReader: returns 50 % cover, records the path."""

    paths = []

    def __init__(self, path, dst_nodata=None):
        _FakeReader.paths.append(str(path))

    def read_and_reproject(self, **kwargs):
        return np.full((1, 2), 50.0, dtype="float32")


def _read_maps(monkeypatch, input_sources, indir="unused"):
    _FakeReader.paths = []
    monkeypatch.setattr(flp.io, "GeoTiffReader", _FakeReader)
    return read_static_maps(indir, {}, {"mod_vegcover": True}, *FOREST_FILES,
                            DRY_FILE, input_sources)


def _release(files):
    src = InputSources(version="2026.01.0", line="demo", years=[2020])
    src._raster_src = {f: {"version": src.version, "dir": "rasters"}
                       for f in files}
    return src


def test_release_without_dry_nature_raster_skips_only_that_map(monkeypatch):
    decid, conif, dry = _read_maps(monkeypatch, _release(FOREST_FILES))
    assert dry is None
    assert np.allclose(decid, 0.5) and np.allclose(conif, 0.5)
    assert not any(p.endswith(DRY_FILE) for p in _FakeReader.paths)


def test_release_with_dry_nature_raster_reads_it(monkeypatch):
    *_, dry = _read_maps(monkeypatch, _release(FOREST_FILES + (DRY_FILE,)))
    assert np.allclose(dry, 0.5)


def test_legacy_folder_uses_file_presence(monkeypatch, tmp_path):
    *_, dry = _read_maps(monkeypatch, None, indir=tmp_path)
    assert dry is None
    (tmp_path / DRY_FILE).touch()
    *_, dry = _read_maps(monkeypatch, None, indir=tmp_path)
    assert np.allclose(dry, 0.5)
