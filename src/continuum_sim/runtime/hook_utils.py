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


def metadata_point(
    metadata: dict[str, object],
    key: str,
) -> np.ndarray | None:
    """Return a finite 3D metadata point, or None when unavailable."""

    value = metadata.get(key)
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point.copy()


def metadata_path(
    metadata: dict[str, object],
    key: str,
) -> np.ndarray | None:
    """Return a finite Nx3 metadata path, or None when unavailable."""

    value = metadata.get(key)
    if value is None:
        return None
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) == 0
        or not np.all(np.isfinite(points))
    ):
        return None
    return points.copy()


def metadata_paths(
    metadata: dict[str, object],
    key: str,
) -> tuple[np.ndarray, ...]:
    """Return finite Nx3 paths from metadata list/tuple entries."""

    value = metadata.get(key)
    if not isinstance(value, list | tuple):
        return ()
    result: list[np.ndarray] = []
    for item in value:
        path = metadata_path({"path": item}, "path")
        if path is not None:
            result.append(path)
    return tuple(result)


def sample_overlay_points(points: np.ndarray, stride: int) -> np.ndarray:
    """Sample overlay path points while preserving the final point."""

    sampled = points[::stride]
    if (len(points) - 1) % stride != 0:
        sampled = np.vstack((sampled, points[-1]))
    return sampled


def split_target_history(
    points: list[np.ndarray],
    kinds: list[str],
    stride: int,
) -> list[list[np.ndarray]]:
    """Split target trail whenever the target kind changes."""

    segments: list[list[np.ndarray]] = []
    previous_kind: str | None = None
    for point, kind in zip(points, kinds, strict=True):
        if not segments or kind != previous_kind:
            segments.append([point])
        else:
            segments[-1].append(point)
        previous_kind = kind
    return [
        list(sample_overlay_points(np.asarray(segment), stride))
        for segment in segments
    ]


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


def metadata_norm(value: object) -> float:
    """Return the norm of a finite metadata array, or NaN when invalid."""

    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        return float("nan")
    return float(np.linalg.norm(array))


def metadata_max_abs(value: object) -> float:
    """Return max absolute finite metadata value, or NaN when unavailable."""

    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float("nan")
    return float(np.max(np.abs(finite)))


def tip_target_error_vector(
    state: RobotSystemState,
    metadata: dict[str, object],
) -> np.ndarray | None:
    """Return executor TCP-to-target error vector from command metadata."""

    del state
    target = metadata_point(metadata, "executor_target_world")
    actual = metadata_point(metadata, "executor_actual_world")
    if target is None or actual is None:
        return None
    return target - actual


def executor_arm(state: RobotSystemState):
    """Return the executor arm state when the current robot has one."""

    return next((arm for arm in state.arms.values() if arm.role == "executor"), None)
