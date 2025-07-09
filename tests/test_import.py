import pytest
import fluxpark

def test_can_import_fluxpark():
    """Smoke‐test: fluxpark-module is importeerbaar en heeft een __version__."""
    assert hasattr(fluxpark, "__version__")
    # of, als je versie in __init__ exposeert:
    from fluxpark import __version__
    assert isinstance(__version__, str)
