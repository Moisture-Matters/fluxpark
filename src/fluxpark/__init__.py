import importlib
import logging
import pkgutil
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"  # for development

def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging for FluxPark.

    Called automatically on import.  Call again to change the log level::

        flp.setup_logging(logging.DEBUG)

    Parameters
    ----------
    level:
        Logging level, e.g. ``logging.DEBUG`` or ``logging.WARNING``.
        Defaults to ``logging.INFO``.
    """
    pkg_logger = logging.getLogger(__name__)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False
    # replace existing StreamHandler if present (e.g. level change); keep
    # non-StreamHandler handlers (file handlers added by the caller).
    pkg_logger.handlers = [
        h for h in pkg_logger.handlers
        if not isinstance(h, logging.StreamHandler)
    ]
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    pkg_logger.addHandler(handler)


setup_logging()


# import all modules dynamically
from . import config
from . import setup
from . import io
from . import postprocessing
from . import prepgrids
from . import submodels
from . import utils
from . import workflow
from .workflow import adapters
from .workflow import FluxParkRunner, RunnerPorts

__all__ = [
    "config",
    "setup",
    "io",
    "postprocessing",
    "prepgrids",
    "utils",
    "workflow",
    "adapters",
    "FluxParkRunner",
    "RunnerPorts",
]

for loader, module_name, is_pkg in pkgutil.iter_modules(submodels.__path__):
    if module_name == "__init__":
        continue  # skip __init__.py

    full_module_name = f"{submodels.__name__}.{module_name}"
    module = importlib.import_module(full_module_name)

    for attr in dir(module):
        if not attr.startswith("_"):
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
