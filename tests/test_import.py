import importlib.metadata

def test_package_has_version():
    """Smoke‐test: fluxpark is installed and has a version string in metadata."""
    # This reads the version from the installed distribution metadata,
    # without importing any package code that might require GDAL.
    version = importlib.metadata.version("fluxpark")
    assert isinstance(version, str)
    assert version != ""

