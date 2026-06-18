"""SDF-gradient and null-space helpers."""

from __future__ import annotations

import numpy as np


def damped_pseudoinverse(matrix: np.ndarray, damping: float = 1.0e-9) -> np.ndarray:
    """Return a small damped Moore-Penrose pseudoinverse."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {array.shape}.")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if damping == 0.0:
        return np.linalg.pinv(array)
    return array.T @ np.linalg.inv(array @ array.T + damping**2 * np.eye(array.shape[0]))


def nullspace_projector(jacobian: np.ndarray, damping: float = 1.0e-9) -> np.ndarray:
    """Return ``N = I - J+ J`` for a task Jacobian."""

    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"jacobian must be 2D, got {J.shape}.")
    return np.eye(J.shape[1], dtype=float) - damped_pseudoinverse(J, damping) @ J


def sdf_repulsive_velocity(
    *,
    distance_m: float,
    gradient: np.ndarray,
    safe_distance_m: float,
    influence_distance_m: float,
    gain: float,
) -> np.ndarray:
    """Return a Cartesian repulsive velocity along the SDF gradient."""

    if safe_distance_m < 0.0:
        raise ValueError("safe_distance_m must be non-negative.")
    if influence_distance_m <= safe_distance_m:
        raise ValueError("influence_distance_m must be greater than safe_distance_m.")
    if gain < 0.0:
        raise ValueError("gain must be non-negative.")
    if not np.isfinite(distance_m) or distance_m >= influence_distance_m:
        return np.zeros(3, dtype=float)
    grad = np.asarray(gradient, dtype=float)
    if grad.shape != (3,):
        raise ValueError(f"gradient must have shape (3,), got {grad.shape}.")
    norm = float(np.linalg.norm(grad))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float)
    strength = (influence_distance_m - distance_m) / (
        influence_distance_m - safe_distance_m
    )
    return gain * max(0.0, strength) * grad / norm


def fuse_task_and_nullspace_velocity(
    jacobian: np.ndarray,
    task_velocity: np.ndarray,
    repulsive_qdot: np.ndarray,
    *,
    damping: float = 1.0e-9,
) -> np.ndarray:
    """Fuse a primary task velocity and a projected null-space velocity."""

    J = np.asarray(jacobian, dtype=float)
    velocity = np.asarray(task_velocity, dtype=float)
    qdot_rep = np.asarray(repulsive_qdot, dtype=float)
    if velocity.shape != (J.shape[0],):
        raise ValueError(f"task_velocity shape must be ({J.shape[0]},).")
    if qdot_rep.shape != (J.shape[1],):
        raise ValueError(f"repulsive_qdot shape must be ({J.shape[1]},).")
    return damped_pseudoinverse(J, damping) @ velocity + nullspace_projector(J, damping) @ qdot_rep
