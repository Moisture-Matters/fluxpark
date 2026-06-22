import importlib
import logging
import pkgutil
from importlib.metadata import version, PackageNotFoundError

from ._logging import setup_logging
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

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"  # for development

# Standard library guidance: a library attaches a NullHandler and configures
# nothing else.  Interactive users call flp.setup_logging() for console output;
# embedding applications configure logging themselves.
logging.getLogger(__name__).addHandler(logging.NullHandler())

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
    "setup_logging",
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
