"""Value parsing helpers for scenario configuration loading."""

from __future__ import annotations

import numpy as np

from continuum_sim.kinematics.pcc import PCC_KINEMATICS_MODES, PCCKinematicsMode


def mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def kinematics_mode(value: object, name: str) -> PCCKinematicsMode:
    mode = str(value)
    if mode not in PCC_KINEMATICS_MODES:
        raise ValueError(f"{name} must be one of {PCC_KINEMATICS_MODES}.")
    return mode  # type: ignore[return-value]


def required(values: dict, name: str, section: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required field {section}.{name}.")
    return values[name]


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def waypoint_orientations(value: object) -> np.ndarray:
    orientations = np.asarray(value, dtype=float)
    if orientations.size == 0:
        return np.zeros((0, 4), dtype=float)
    if orientations.ndim != 2 or orientations.shape[1] != 4:
        raise ValueError(
            "scenario.task.waypoint_orientations_world_wxyz must have shape (N, 4)."
        )
    norms = np.linalg.norm(orientations, axis=1)
    if np.any((~np.isfinite(norms)) | (norms <= 1.0e-12)):
        raise ValueError(
            "scenario.task.waypoint_orientations_world_wxyz rows must be finite "
            "nonzero quaternions."
        )
    return orientations / norms[:, None]


def waypoint_directions(value: object) -> np.ndarray:
    directions = np.asarray(value, dtype=float)
    if directions.size == 0:
        return np.zeros((0, 3), dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(
            "scenario.task.pose_servo.waypoint_directions_world must have shape (N, 3)."
        )
    norms = np.linalg.norm(directions, axis=1)
    if np.any((~np.isfinite(norms)) | (norms <= 1.0e-12)):
        raise ValueError(
            "scenario.task.pose_servo.waypoint_directions_world rows must be finite "
            "nonzero 3-vectors."
        )
    return directions / norms[:, None]
