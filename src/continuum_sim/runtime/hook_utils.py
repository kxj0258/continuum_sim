"""Small shared helpers for runtime hooks."""

from __future__ import annotations

from typing import Any

import numpy as np

from continuum_sim.system.types import RobotSystemState


def metadata_vector_or_nan(
    metadata: dict[str, Any],
    key: str,
) -> np.ndarray:
    """Return a finite 3-vector metadata value or a NaN placeholder."""

    value = metadata.get(key)
    if value is None:
        return np.full(3, np.nan, dtype=float)
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return np.full(3, np.nan, dtype=float)
    return vector.copy()


def finite_metadata_float(
    metadata: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
) -> float:
    """Return a finite scalar metadata value, optionally falling back to another key."""

    value = float(metadata.get(key, np.nan))
    if np.isfinite(value) or fallback_key is None:
        return value
    return float(metadata.get(fallback_key, np.nan))


def executor_arm(state: RobotSystemState):
    """Return the executor arm state when the current robot has one."""

    return next((arm for arm in state.arms.values() if arm.role == "executor"), None)
