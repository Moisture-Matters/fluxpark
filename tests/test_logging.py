"""Tests for setup_logging: console output that survives IDE root config."""

import io
import logging

import pytest

# Importing fluxpark pulls in GDAL (osgeo), provided via conda not pip; skip the
# whole module when it is unavailable (pip-only CI).
pytest.importorskip("osgeo")

import fluxpark as flp  # noqa: E402
from fluxpark._logging import _CONSOLE_HANDLER_NAME, LOGGER_NAME  # noqa: E402


def _capture_buffer():
    """Point the FluxPark console handler at an in-memory buffer."""
    buf = io.StringIO()
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        if handler.get_name() == _CONSOLE_HANDLER_NAME:
            handler.stream = buf
    return buf


def test_child_logger_output_is_captured():
    flp.setup_logging(logging.INFO)
    buf = _capture_buffer()
    logging.getLogger("fluxpark.submodels.demo").info("hello-info")
    assert "hello-info" in buf.getvalue()


def test_repeated_calls_do_not_stack_handlers():
    flp.setup_logging()
    flp.setup_logging(logging.DEBUG)
    logger = logging.getLogger(LOGGER_NAME)
    names = [h.get_name() for h in logger.handlers]
    assert names.count(_CONSOLE_HANDLER_NAME) == 1
    # propagate disabled so IDE root handlers cannot filter or duplicate output
    assert logger.propagate is False
