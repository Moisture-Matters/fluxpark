from typing import Any, Callable, Dict, TypedDict
import numpy as np


ContextDict = Dict[str, Any]
ExecutionContextPort = Callable[[Any], ContextDict]


class InitialDataDict(TypedDict):
    old: Dict[str, Any]


InitalDataPort = Callable[[Any], InitialDataDict]


AncillaryRasterDict = Dict[str, Any]
AncillaryRasterPort = Callable[[Any], AncillaryRasterDict]

RainProviderArr = np.ndarray
RainProviderPort = Callable[[Any], RainProviderArr]

EtrefProviderArr = np.ndarray
EtrefProviderPort = Callable[[Any], EtrefProviderArr]

DailyInputModifierDict = Dict[str, Any]
DailyInputModifierPort = Callable[[Any], DailyInputModifierDict]

DailyOutputModifierDict = Dict[str, Any]
DailyOutputModifierPort = Callable[[Any], DailyInputModifierDict]


def default_execution_context(cfg) -> ContextDict:
    """
    Default adapter for open-source usage.
    """
    return {}


def default_inital_data(runner) -> InitialDataDict:
    """
    Default adapter for open-source usage.
    """
    return {"old": {}}


def default_ancillary_raster(runner) -> AncillaryRasterDict:
    """
    Default adapter for open-source usage.
    """
    return {}


def default_rain_provider(runner) -> RainProviderArr:
    """
    Default adapter for open-source usage.
    """
    shape = (
        runner.grid_params["nrows"],
        runner.grid_params["ncols"],
    )
    return np.full(shape, 1.0, dtype="float32")


def default_etref_provider(runner) -> EtrefProviderArr:
    """
    Default adapter for open-source usage.
    """
    shape = (
        runner.grid_params["nrows"],
        runner.grid_params["ncols"],
    )
    return np.full(shape, 1.0, dtype="float32")


def default_daily_input_modifier(cfg) -> DailyInputModifierDict:
    """
    Default adapter for open-source usage.
    """
    return {"daily_output": {}, "states": {}}


def default_daily_output_modifier(cfg) -> DailyOutputModifierDict:
    """
    Default adapter for open-source usage.
    """
    return {"daily_output": {}, "states": {}, "current_evap_params": {}}


OutputPort = Callable[[Any], None]


def default_output_port(runner) -> None:
    """
    Default adapter for open-source usage.

    Does nothing (no upload, no cleanup).
    """
    return None


# ContextDict = Dict[str, Any]


# class ExecutionContextPort(Protocol):
#     """
#     Port for runtime/environment-specific concerns.

#     The runner calls this once during setup to obtain a runtime context dict
#     (e.g., db handles, caches, worker settings, feature flags).
#     """

#     def initialize(self, cfg: Any) -> ContextDict:
#         """Create and return a runtime context dictionary."""
#         ...


# @dataclass
# class DefaultExecutionContext:
#     """
#     Default adapter for open-source usage.

#     If no other adapter is specified for the ExecutionContextPort this is the default.
#     """

#     def initialize(self, cfg: Any) -> ContextDict:
#         return {}
