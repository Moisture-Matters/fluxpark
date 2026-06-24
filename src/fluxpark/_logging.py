"""Logging helpers for FluxPark.

Following the standard library guidance for libraries, importing FluxPark
attaches only a :class:`logging.NullHandler` (done in ``__init__``) and
configures nothing else at import time.

So a run still produces visible output by default, :class:`FluxParkRunner`
calls :func:`ensure_logging` at the start of ``setup``: it sets up a console
handler only when logging has not been configured yet. Users who want to
control output call :func:`setup_logging` (e.g. to change the level); an
embedding application that configures the ``fluxpark`` logger itself is left
untouched.

This is handled on the ``fluxpark`` logger directly because IDEs such as
Spyder and VS Code pre-configure the root logger, which would otherwise filter
out FluxPark's ``INFO`` output.
"""

import logging

LOGGER_NAME = "fluxpark"
_CONSOLE_HANDLER_NAME = "fluxpark_console"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Send FluxPark log messages to the console.

    Call this once at the start of a script or interactive session::

        import fluxpark as flp
        flp.setup_logging()                 # INFO and above
        flp.setup_logging(logging.DEBUG)    # more detail

    A console handler is attached directly to the ``fluxpark`` logger and
    ``propagate`` is disabled, so output is shown regardless of how the host
    (Spyder, VS Code, ...) configured the root logger, and without duplicate
    lines.  Calling the function again only updates the level; it never stacks
    handlers.

    Parameters
    ----------
    level:
        Logging level, e.g. ``logging.DEBUG`` or ``logging.WARNING``.
        Defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
        The configured ``fluxpark`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # reuse our own handler so repeated calls don't stack output
    for handler in logger.handlers:
        if handler.get_name() == _CONSOLE_HANDLER_NAME:
            handler.setLevel(level)
            return logger

    handler = logging.StreamHandler()
    handler.set_name(_CONSOLE_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def _is_configured(logger: logging.Logger) -> bool:
    """True when the logger has a real (non-Null) handler of its own."""
    return any(
        not isinstance(h, logging.NullHandler) for h in logger.handlers
    )


def ensure_logging(level: int = logging.INFO) -> None:
    """Set up default console logging unless it is already configured.

    Called by :class:`FluxParkRunner` so a run produces visible output by
    default, even when the user did not call :func:`setup_logging`. It is a
    no-op when the ``fluxpark`` logger already has a non-Null handler — i.e.
    the user called :func:`setup_logging` or an embedding application
    configured the logger — so an explicit configuration is never overridden.

    Only the ``fluxpark`` logger is inspected, not the root logger: IDEs such
    as Spyder pre-configure the root logger at ``WARNING`` level, which must
    not count as "already configured" or FluxPark's ``INFO`` output would stay
    hidden.

    Parameters
    ----------
    level:
        Logging level used when no configuration is present yet. Defaults to
        ``logging.INFO``.
    """
    if _is_configured(logging.getLogger(LOGGER_NAME)):
        return
    setup_logging(level)
