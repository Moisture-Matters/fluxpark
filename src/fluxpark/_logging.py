"""Logging helpers for FluxPark.

Following the standard library guidance for libraries, importing FluxPark
attaches only a :class:`logging.NullHandler` (done in ``__init__``) and
configures nothing else.  Applications that embed FluxPark should configure
logging themselves.

Interactive users running FluxPark from a script or an IDE such as Spyder or
VS Code can call :func:`setup_logging` once to route FluxPark's messages to
the console.  This is needed because those IDEs pre-configure the root logger,
which otherwise filters out FluxPark's ``INFO`` output.
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
