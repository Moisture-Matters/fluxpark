from pathlib import Path
from string import Formatter
from typing import Union


_URL_PREFIXES = ("http://", "https://", "ftp://", "ftps://")
_VSI_PREFIX = "/vsi"


def has_placeholders(pattern: str) -> bool:
    """Return True als pattern één of meer {veld}-placeholders bevat."""
    for _, field_name, _, _ in Formatter().parse(pattern):
        if field_name:
            return True
    return False


def is_url(path) -> bool:
    """Return True if `path` is a remote URL or a GDAL /vsi* virtual path.

    Local filesystem paths (including Windows drive paths like ``C:\\...``)
    return False.
    """
    s = str(path)
    return s.startswith(_URL_PREFIXES) or s.startswith(_VSI_PREFIX)


def to_gdal_path(source) -> str:
    """Normalize a source location to a string GDAL can open.

    Local paths are normalized via :class:`pathlib.Path` (Windows backslashes
    become forward slashes). Remote URLs (http/https/ftp) are wrapped in
    ``/vsicurl/`` so GDAL streams them with HTTP range requests. Paths that are
    already GDAL virtual file system paths (``/vsicurl/...``, ``/vsis3/...``,
    etc.) are passed through unchanged.

    Parameters
    ----------
    source : str or Path
        A local filesystem path or a remote URL.

    Returns
    -------
    str
        A string ready to hand to ``gdal.Open``.
    """
    s = str(source)
    if s.startswith(_VSI_PREFIX):
        return s
    if s.startswith(_URL_PREFIXES):
        return f"/vsicurl/{s}"
    return str(Path(source)).replace("\\", "/")


def join_path_or_url(base, *parts) -> Union[str, Path]:
    """Join path segments for either a local path or a remote URL.

    For local bases a :class:`pathlib.Path` is returned, so existing
    Path-based code keeps working unchanged. For URL/vsi bases a
    forward-slash joined string is returned, avoiding :class:`pathlib.Path`
    which would corrupt the ``://`` in a URL on Windows.

    Parameters
    ----------
    base : str or Path
        Base directory, local path or remote URL.
    *parts : str
        Path segments to append.

    Returns
    -------
    str or Path
        A URL string when `base` is a URL, otherwise a Path.
    """
    if is_url(base):
        joined = str(base).rstrip("/")
        for part in parts:
            joined = f"{joined}/{str(part).strip('/')}"
        return joined
    return Path(base, *parts)
