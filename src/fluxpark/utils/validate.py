import numpy as np


def validate_grid(
    value,
    expected_shape,
    name="grid",
    nan_policy="error",
):
    """
    Validate a 2D numeric grid-like input.

    Parameters
    ----------
    value : Any
        Grid-like input returned by a provider port.
    expected_shape : tuple[int, int]
        Expected shape as (nrows, ncols).
    name : str, default="grid"
        Name used in error messages.
    nan_policy : {"error", "skip", "allow"}, default="error"
        Policy for handling NaN values *inside* a returned grid. A ``None``
        return is handled separately: it always skips the day (see Returns),
        so a provider can signal "no data for this day" while genuine NaNs
        in a grid still error under the default policy.

    Returns
    -------
    bool
        True if the current day should be skipped, otherwise False. ``None``
        (no data for the day) always returns True, independent of nan_policy.

    Raises
    ------
    TypeError
        If the input type is invalid or cannot be converted to a numeric
        NumPy array.
    ValueError
        If the input is invalid and should not be skipped.
    """
    if nan_policy not in {"error", "skip", "allow"}:
        raise ValueError("nan_policy must be 'error', 'skip', or 'allow'.")

    if (
        not isinstance(expected_shape, tuple)
        or len(expected_shape) != 2
        or not all(isinstance(i, int) for i in expected_shape)
    ):
        raise TypeError("expected_shape must be a tuple of two integers.")

    if value is None:
        # None means "no data for this day" and always skips it, regardless
        # of nan_policy. That policy governs NaN *inside* a returned grid (a
        # different concern): a genuinely corrupt grid still errors under
        # "error". A provider signals "skip this day" by returning None.
        return True

    if isinstance(value, str):
        raise TypeError(f"{name} must be a 2D numeric array, not a string.")

    try:
        arr = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} could not be converted to a NumPy array.") from exc

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.ndim}D with shape {arr.shape}.")

    if arr.shape != expected_shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")

    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values.")

    has_nan = np.isnan(arr).any()

    if has_nan:
        if nan_policy == "allow":
            return False
        if nan_policy == "skip":
            return True
        raise ValueError(f"{name} contains NaN values.")

    return False
