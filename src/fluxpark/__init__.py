import importlib
import pkgutil

# import all modules dinamically
from . import config
from . import setup
from . import io
from . import postprocessing
from . import prepgrids
from . import submodels
from . import utils
from . import workflow
from .workflow.runner import FluxParkRunner

__all__ = []

for loader, module_name, is_pkg in pkgutil.iter_modules(submodels.__path__):
    if module_name == "__init__":
        continue  # skip __init__.py

    full_module_name = f"{submodels.__name__}.{module_name}"
    module = importlib.import_module(full_module_name)

    for attr in dir(module):
        if not attr.startswith("_"):
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
