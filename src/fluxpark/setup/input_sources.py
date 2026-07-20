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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from osgeo import gdal

from ..utils.common import is_url, join_path_or_url, to_gdal_path

RELEASE_FILENAME = "release.yml"
SOURCES_SNAPSHOT_FILENAME = "fluxpark_input_sources.json"

# Plain-text pointer file (no extension) at the line root naming the newest
# release; used when input_version is "latest".
LATEST_FILENAME = "latest"
LATEST_VERSION_ALIAS = "latest"

# Stable release-alias for the evaporation parameter table.
EVAP_PARAMS_TABLE_NAME = "evap_parameters"

# Guard against accidental infinite extends chains.
_MAX_EXTENDS_DEPTH = 50

PathLike = Union[str, Path]


def _read_bytes(path: PathLike) -> bytes:
    """Read raw bytes from a local path or a remote URL."""
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
        return bytes(raw)
    with open(path, "rb") as handle:
        return handle.read()


def _read_text(path: PathLike) -> str:
    """Read a small text file (UTF-8) from a local path or a remote URL."""
    return _read_bytes(path).decode("utf-8")


def localize_file(path: PathLike, dest_dir: Optional[PathLike]) -> Path:
    """Return a local path for `path`, downloading it when it is a remote URL.

    A local path is returned unchanged. A remote URL is downloaded into
    `dest_dir` (required for URLs) and the resulting local path is returned.
    Used for files that local tools cannot open over an authenticated HTTP
    connection (pandas tables, an OGR cutline).
    """
    if not is_url(path):
        return Path(path)
    if dest_dir is None:
        raise RuntimeError(
            f"A download directory is required to read the remote file "
            f"'{path}'."
        )
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    filename = str(path).rstrip("/").rsplit("/", 1)[-1]
    target = dest / filename
    target.write_bytes(_read_bytes(path))
    return target


def _exists(path: PathLike) -> bool:
    """Return True if a local path or remote URL exists."""
    if is_url(path):
        return gdal.VSIStatL(to_gdal_path(path)) is not None
    return Path(path).exists()


def parent_dir(path: PathLike) -> PathLike:
    """Return the parent of a local path or a remote URL."""
    if is_url(path):
        return str(path).rstrip("/").rsplit("/", 1)[0]
    return Path(path).parent


def is_release_dir(indir: PathLike) -> bool:
    """Return True if `indir` contains a ``release.yml`` (is a release folder)."""
    return _exists(join_path_or_url(indir, RELEASE_FILENAME))


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
    # directory to download remote tables into before they are read locally
    download_dir: Optional[PathLike] = None
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
        """Local path of a table file, downloading it when it is remote."""
        directory = self._table_dir_by_file.get(filename)
        if directory is None:
            raise KeyError(f"Table '{filename}' is not declared in the release.")
        return localize_file(join_path_or_url(directory, filename), self.download_dir)

    def table_filenames(self) -> List[str]:
        """Sorted list of table filenames declared across the chain."""
        return sorted(self._table_dir_by_file.keys())

    def table_path_by_name(self, name: str) -> PathLike:
        """Local path of a table by its release alias (its ``name`` key).

        Unlike :meth:`table_path` (which takes the on-disk filename), this
        resolves a table by the stable alias declared in ``release.yml``, so
        the actual filename may differ per release. A remote table is
        downloaded into ``download_dir`` and the local path is returned.
        """
        info = self._table_by_name.get(name)
        if info is None:
            raise KeyError(
                f"The release does not declare a '{name}' table."
            )
        return localize_file(
            join_path_or_url(info["dir"], info["file"]), self.download_dir
        )

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


def read_latest_version(line_root: PathLike) -> str:
    """Return the version named in the line's ``latest`` pointer file.

    The ``latest`` file (plain text, no extension) sits at the line root and
    contains the folder name of the newest release, e.g. ``2026.06.0__full``.
    Works for a local path and a remote URL.
    """
    pointer = join_path_or_url(line_root, LATEST_FILENAME)
    try:
        text = _read_text(pointer)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            f"input_version='latest', but no '{LATEST_FILENAME}' pointer file "
            f"was found at '{pointer}'. Create it containing the newest release "
            f"folder name, or set input_version to a specific release."
        ) from exc
    version = text.strip()
    if not version:
        raise RuntimeError(
            f"The '{LATEST_FILENAME}' pointer at '{pointer}' is empty."
        )
    return version


def resolve_raster(
    input_sources: Optional["InputSources"],
    indir_rasters: PathLike,
    filename: str,
) -> PathLike:
    """Resolve a raster filename to a path/URL.

    Uses the resolved `input_sources` (honouring ``extends``) when available,
    otherwise falls back to joining onto `indir_rasters` (legacy folders).
    """
    if input_sources is not None:
        return input_sources.raster_path(filename)
    return join_path_or_url(indir_rasters, filename)


def resolve_table(
    input_sources: Optional["InputSources"],
    indir_tables: PathLike,
    filename: str,
) -> PathLike:
    """Resolve a table filename to a path/URL.

    Uses the resolved `input_sources` (honouring ``extends``) when available,
    otherwise falls back to joining onto `indir_tables` (legacy folders).
    """
    if input_sources is not None:
        return input_sources.table_path(filename)
    return join_path_or_url(indir_tables, filename)


def resolve_input_version(
    input_sources: Optional["InputSources"],
    input_version: Optional[str],
) -> str:
    """Return the effective input-data version label for provenance.

    A release's ``release.yml`` is authoritative; otherwise the explicitly
    configured ``input_version`` is used; if neither is known the result is
    ``"unknown"``.
    """
    if input_sources is not None and input_sources.version:
        return input_sources.version
    if input_version:
        return input_version
    return "unknown"


def build_provenance(
    input_sources: Optional["InputSources"],
    input_version: Optional[str],
    package_version: str,
) -> Dict[str, str]:
    """Build the provenance tags stamped into output GeoTIFFs.

    Returns the ``FLUXPARK_*`` key/value pairs (package version, resolved
    input-data version and a UTC creation timestamp) that travel inside every
    output file for traceability.
    """
    return {
        "FLUXPARK_VERSION": package_version,
        "FLUXPARK_INPUT_VERSION": resolve_input_version(
            input_sources, input_version
        ),
        "FLUXPARK_CREATED": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def merge_extra_provenance(
    provenance: Dict[str, str], context: Any
) -> Dict[str, str]:
    """Merge runtime provenance tags supplied by the execution context.

    The execution_context port may return an ``extra_provenance`` mapping in
    its context dict (e.g. the version of a private orchestration layer built
    on top of FluxPark). Those tags are stamped into the output GeoTIFFs
    alongside the ``FLUXPARK_*`` tags. Keys and values are coerced to str so
    they are always writable as GeoTIFF metadata.
    """
    if not isinstance(context, dict):
        return provenance
    extra = context.get("extra_provenance") or {}
    provenance.update({str(k): str(v) for k, v in extra.items()})
    return provenance


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
        current = join_path_or_url(parent_dir(current), extends)

    return chain


def load_input_sources(
    indir: PathLike, download_dir: Optional[PathLike] = None
) -> Optional[InputSources]:
    """Load and resolve the release in `indir`, following ``extends``.

    Parameters
    ----------
    indir
        The resolved release (version) folder: a local path or an HTTPS URL.
    download_dir
        Directory used to download remote tables into before reading them; not
        needed for local releases.

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
        download_dir=download_dir,
        _raster_src=raster_src,
        _table_by_name=table_by_name,
        _table_dir_by_file=table_dir_by_file,
    )
