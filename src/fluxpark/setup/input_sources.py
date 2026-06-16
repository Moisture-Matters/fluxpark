"""Loading and resolving FluxPark input data across versioned releases.

Input data is organized in releases: a release is a folder described by a
``release.yml`` file. A full release declares all of its rasters and tables; a
partial release declares only the files it changes and points to the release it
builds on via ``extends``. Resolution rule: for every logical input (a raster
identified by type+year or a static raster, or a table identified by name) the
source is the most derived release in the ``extends`` chain that declares it;
everything else is inherited.

The release folders are siblings under a "line" folder (e.g.
``releases/nweu/2025.06.0__full``). Both local paths and remote HTTPS URLs are
supported; remote ``release.yml`` files are read through GDAL's ``/vsicurl/``
virtual file system, reusing any GDAL HTTP credentials already configured.

Loading and resolving a release folder yields an :class:`InputSources` object:
the combined input sources for a run. It can be serialized to
``fluxpark_input_sources.json`` as a provenance snapshot.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from osgeo import gdal

from ..utils.common import is_url, join_path_or_url, to_gdal_path

RELEASE_FILENAME = "release.yml"
SOURCES_SNAPSHOT_FILENAME = "fluxpark_input_sources.json"

# Guard against accidental infinite extends chains.
_MAX_EXTENDS_DEPTH = 50

PathLike = Union[str, Path]


def _read_text(path: PathLike) -> str:
    """Read a small text file from a local path or a remote URL."""
    if is_url(path):
        gdal_path = to_gdal_path(path)
        handle = gdal.VSIFOpenL(gdal_path, "rb")
        if handle is None:
            raise FileNotFoundError(f"Could not open '{path}'.")
        try:
            stat = gdal.VSIStatL(gdal_path)
            size = stat.size if stat is not None else 0
            raw = gdal.VSIFReadL(1, size, handle) if size else b""
        finally:
            gdal.VSIFCloseL(handle)
        return bytes(raw).decode("utf-8")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _exists(path: PathLike) -> bool:
    """Return True if a local path or remote URL exists."""
    if is_url(path):
        return gdal.VSIStatL(to_gdal_path(path)) is not None
    return Path(path).exists()


def _parent(path: PathLike) -> PathLike:
    """Return the parent of a local path or a remote URL."""
    if is_url(path):
        return str(path).rstrip("/").rsplit("/", 1)[0]
    return Path(path).parent


@dataclass
class InputSources:
    """Resolved FluxPark input sources for a run (the full ``extends`` chain).

    Attributes
    ----------
    version
        Folder name of the most derived release (including any ``__label``).
    line
        Line identifier (e.g. "nweu"), or None when not specified.
    years
        Sorted list of years available across the chain (union of all yearly
        ``years``), used to drive dynamic land-use selection.
    """

    version: str
    line: Optional[str]
    years: List[int]
    # filename -> {"version": str, "dir": rasters-dir of that release}
    _raster_src: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # table alias name -> {"version": str, "dir": tables-dir, "file": filename}
    _table_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # filename -> tables-dir of the providing release
    _table_dir_by_file: Dict[str, PathLike] = field(default_factory=dict)

    def raster_path(self, filename: str) -> PathLike:
        """Full path/URL of a raster file, resolved through the chain."""
        info = self._raster_src.get(filename)
        if info is None:
            raise KeyError(f"Raster '{filename}' is not declared in the release.")
        return join_path_or_url(info["dir"], filename)

    def table_path(self, filename: str) -> PathLike:
        """Full path/URL of a table file, resolved through the chain."""
        directory = self._table_dir_by_file.get(filename)
        if directory is None:
            raise KeyError(f"Table '{filename}' is not declared in the release.")
        return join_path_or_url(directory, filename)

    def write_sources_snapshot(self, outdir: PathLike) -> Path:
        """Write a provenance snapshot of which release supplied each file.

        This is a write-only audit artifact next to ``fluxpark_cfg.json``; it
        is never read back as input (the ``release.yml`` chain stays the single
        source of truth).
        """
        out_path = Path(outdir) / SOURCES_SNAPSHOT_FILENAME
        data = {
            "input_version": self.version,
            "line": self.line,
            "resolved": {
                "rasters": {
                    fn: info["version"]
                    for fn, info in sorted(self._raster_src.items())
                },
                "tables": {
                    name: info["version"]
                    for name, info in sorted(self._table_by_name.items())
                },
            },
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return out_path


def _load_chain(indir: PathLike) -> List[Dict[str, Any]]:
    """Load the release at `indir` and its `extends` ancestors.

    Returns a list ordered most-derived first, each entry holding the parsed
    yaml plus the release folder it came from.
    """
    chain: List[Dict[str, Any]] = []
    seen: set = set()
    current: Optional[PathLike] = indir

    while current is not None:
        release_path = join_path_or_url(current, RELEASE_FILENAME)
        parsed = yaml.safe_load(_read_text(release_path)) or {}
        version = parsed.get("version")
        if version is None:
            raise RuntimeError(f"'{release_path}' has no 'version' field.")
        if version in seen:
            raise RuntimeError(
                f"Cyclic 'extends' chain detected at release '{version}'."
            )
        seen.add(version)
        if len(chain) >= _MAX_EXTENDS_DEPTH:
            raise RuntimeError(
                f"'extends' chain exceeds the maximum depth of "
                f"{_MAX_EXTENDS_DEPTH}."
            )
        chain.append({"version": version, "dir": current, "data": parsed})

        extends = parsed.get("extends")
        if not extends:
            break
        current = join_path_or_url(_parent(current), extends)

    return chain


def load_input_sources(indir: PathLike) -> Optional[InputSources]:
    """Load and resolve the release in `indir`, following ``extends``.

    Parameters
    ----------
    indir
        The resolved release (version) folder: a local path or an HTTPS URL.

    Returns
    -------
    InputSources or None
        Resolved input sources, or None when `indir` contains no
        ``release.yml`` (legacy folders without versioning).
    """
    if not _exists(join_path_or_url(indir, RELEASE_FILENAME)):
        return None

    chain = _load_chain(indir)

    line = chain[0]["data"].get("line")
    for entry in chain[1:]:
        other = entry["data"].get("line")
        if line is not None and other is not None and other != line:
            raise RuntimeError(
                f"'extends' crosses lines: '{line}' extends '{other}'. A "
                "release must build on the same line."
            )

    raster_src: Dict[str, Dict[str, Any]] = {}
    table_by_name: Dict[str, Dict[str, Any]] = {}
    table_dir_by_file: Dict[str, PathLike] = {}
    years: set = set()

    # Most-derived first: setdefault keeps the first (winning) source.
    for entry in chain:
        version = entry["version"]
        rasters_dir = join_path_or_url(entry["dir"], "rasters")
        tables_dir = join_path_or_url(entry["dir"], "tables")
        data = entry["data"]

        rasters = data.get("rasters") or {}
        yearly = rasters.get("yearly") or {}
        yearly_years = yearly.get("years") or []
        years.update(int(y) for y in yearly_years)
        for type_def in yearly.get("types") or []:
            pattern = type_def["pattern"]
            for year in yearly_years:
                filename = pattern.format(year=year)
                raster_src.setdefault(
                    filename, {"version": version, "dir": rasters_dir}
                )

        static = rasters.get("static") or {}
        for type_def in static.get("types") or []:
            filename = type_def["file"]
            raster_src.setdefault(
                filename, {"version": version, "dir": rasters_dir}
            )

        for table_def in data.get("tables") or []:
            name = table_def["name"]
            filename = table_def["file"]
            if name not in table_by_name:
                table_by_name[name] = {
                    "version": version,
                    "dir": tables_dir,
                    "file": filename,
                }
            table_dir_by_file.setdefault(filename, tables_dir)

    return InputSources(
        version=chain[0]["version"],
        line=line,
        years=sorted(years),
        _raster_src=raster_src,
        _table_by_name=table_by_name,
        _table_dir_by_file=table_dir_by_file,
    )
