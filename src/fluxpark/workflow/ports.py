from dataclasses import dataclass
from typing import Any, Callable, Dict, TypedDict

import numpy as np
from numpy.typing import NDArray


ContextDict = Dict[str, Any]
GridDict = Dict[str, Any]
StateDict = Dict[str, Any]
OutputDict = Dict[str, Any]
EvapParamDict = Dict[str, Any]
GridArray = NDArray[Any]


ExecutionContextPort = Callable[[Any], ContextDict]


class InitialDataDict(TypedDict):
    """Extra initial data returned during setup."""

    old: StateDict


InitialDataPort = Callable[[Any], InitialDataDict]


AncillaryRasterDict = GridDict
AncillaryRasterPort = Callable[[Any], AncillaryRasterDict]

RainProviderArr = GridArray
RainProviderPort = Callable[[Any], RainProviderArr]

EtrefProviderArr = GridArray
EtrefProviderPort = Callable[[Any], EtrefProviderArr]


class DailyInputModifierDict(TypedDict):
    """Return type for pre-core daily modifiers."""

    daily_output: OutputDict
    states: StateDict
    current_evap_params: EvapParamDict


DailyInputModifierPort = Callable[[Any, StateDict], DailyInputModifierDict]


class DailyOutputModifierDict(TypedDict):
    """Return type for post-core daily modifiers."""

    daily_output: OutputDict
    states: StateDict


DailyOutputModifierPort = Callable[
    [OutputDict, StateDict, Any],
    DailyOutputModifierDict,
]


OutputPort = Callable[[Any], None]


def default_execution_context(cfg: Any) -> ContextDict:
    """
    Default port for open-source usage.

    Returns an empty runtime context.
    """
    return {}


def default_initial_data(runner: Any) -> InitialDataDict:
    """
    Default port for open-source usage.

    Returns no extra initial state.
    """
    return {"old": {}}


def default_ancillary_raster(runner: Any) -> AncillaryRasterDict:
    """
    Default port for open-source usage.

    Returns no ancillary rasters.
    """
    return {}


def default_rain_provider(runner: Any) -> RainProviderArr:
    """
    Default port for open-source usage.

    Returns a constant rain grid of 1.0.
    """
    shape = (
        runner.grid_params["nrows"],
        runner.grid_params["ncols"],
    )
    return np.full(shape, 1.0, dtype="float32")


def default_etref_provider(runner: Any) -> EtrefProviderArr:
    """
    Default port for open-source usage.

    Returns a constant etref grid of 1.0.
    """
    shape = (
        runner.grid_params["nrows"],
        runner.grid_params["ncols"],
    )
    return np.full(shape, 1.0, dtype="float32")


def default_daily_input_modifier(
    runner: Any,
    old: StateDict,
) -> DailyInputModifierDict:
    """
    Default port for open-source usage.

    Returns no changes before the core calculation.
    """
    return {
        "daily_output": {},
        "states": {},
        "current_evap_params": {},
    }


def default_daily_output_modifier(
    daily_output: OutputDict,
    old: StateDict,
    runner: Any,
) -> DailyOutputModifierDict:
    """
    Default port for open-source usage.

    Returns no changes after the core calculation.
    """
    return {
        "daily_output": {},
        "states": {},
    }


def default_output_port(runner: Any) -> None:
    """
    Default port for open-source usage.

    Does nothing (no upload, no cleanup).
    """
    return None


@dataclass
class RunnerPorts:
    """
    Collection of all workflow ports used by FluxParkRunner.

    The defaults are suitable for open-source/local usage.
    """

    execution_context: ExecutionContextPort = default_execution_context
    initial_data: InitialDataPort = default_initial_data
    ancillary_raster: AncillaryRasterPort = default_ancillary_raster
    rain_provider: RainProviderPort = default_rain_provider
    etref_provider: EtrefProviderPort = default_etref_provider
    daily_input_modifier: DailyInputModifierPort = default_daily_input_modifier
    daily_output_modifier: DailyOutputModifierPort = default_daily_output_modifier
    output: OutputPort = default_output_port


# Backward-compatible aliases for older imports / temporary transition.
InitalDataPort = InitialDataPort
default_inital_data = default_initial_data
