import numpy as np
from typing import Dict, List


def update_loop_state(
    old: Dict[str, np.ndarray],
    rerun_par_list: List[str],
    conv_output: Dict[str, str],
    daily_output: Dict[str, np.ndarray],
    cum_output: Dict[str, np.ndarray],
) -> None:
    """
    Prepare state dict for the next timestep by copying relevant arrays.

    Parameters
    ----------
    old : dict[str, ndarray]
        Dictionary holding previous state arrays (will be updated in place).
    rerun_par_list : list of str
        output‐keys indicating which variables to carry over each loop.
    conv_output : dict[str, str]
        Mapping from output keys to Python variable names in `old`.
    daily_output : dict[str, ndarray]
        Daily flux outputs; keys are Python names without '_c' suffix.
    cum_output : dict[str, ndarray]
        Cumulative flux outputs; keys are Python names ending with '_c'.

    Notes
    -----
    - If the Python name ends with '_c', copy from `cum_output`.
    - Otherwise, copy from `daily_output`.
    """
    for output_key in rerun_par_list:
        py_key = conv_output[output_key]
        if py_key.endswith("_c"):
            # Carry over cumulative value
            old[py_key] = cum_output[py_key].copy()
        else:
            # Carry over today's value
            old[py_key] = daily_output[py_key].copy()
