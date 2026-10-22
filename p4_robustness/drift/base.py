"""Shared input checks for the drift detectors.

CLAUDE.md: detectors are univariate and monitor a NAMED scalar, never a raw
high-dimensional vector. These helpers enforce that for every detector.
"""
import math
import numbers

import numpy as np


def as_scalar(x):
    """Return x as a finite float. Rejects arrays/lists of any shape (even length 1)."""
    if isinstance(x, (bool, np.bool_)):
        raise TypeError("Drift detectors take a numeric scalar, not a bool")
    if isinstance(x, np.ndarray):
        if x.ndim != 0:
            raise TypeError(f"Drift detectors take one scalar per step, got an array of shape {x.shape}")
        x = x.item()
    if not isinstance(x, numbers.Real):
        raise TypeError(f"Drift detectors take one scalar per step, got {type(x).__name__}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"Scalar must be finite, got {x}")
    return x


def as_scalar_series(values):
    """Return values as a 1-D float array of finite numbers. Rejects 2-D or higher input."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Expected a 1-D sequence of numbers, got a string")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise TypeError(f"Expected a 1-D series of scalars, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Series contains NaN or infinite values")
    return arr


def check_name(monitored):
    if not isinstance(monitored, str) or not monitored.strip():
        raise ValueError("`monitored` must name the scalar stream, e.g. 'fused_windowed_anomaly_score'")
    return monitored
