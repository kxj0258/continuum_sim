"""Minimal CBF-QP velocity projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CBFQPConfig:
    """Numerical settings for CBF velocity projection."""

    max_projection_iterations: int = 8
    tolerance: float = 1.0e-9


def solve_cbf_qp_velocity(
    reference_velocity: np.ndarray,
    *,
    barrier_jacobian: np.ndarray | None = None,
    barrier_lower_bound: np.ndarray | None = None,
    equality_matrix: np.ndarray | None = None,
    equality_target: np.ndarray | None = None,
    config: CBFQPConfig = CBFQPConfig(),
) -> np.ndarray:
    """Minimize distance to a reference velocity subject to linear constraints.

    This is the tiny active-set projection needed by the project tests. If the
    problem grows, this function is the only place that needs an OSQP swap.
    """

    velocity = _as_vector(reference_velocity, "reference_velocity").copy()
    if equality_matrix is not None:
        Aeq = _as_matrix(equality_matrix, "equality_matrix", columns=velocity.size)
        beq = _as_vector(equality_target, "equality_target", expected_size=Aeq.shape[0])
        velocity = velocity + np.linalg.pinv(Aeq) @ (beq - Aeq @ velocity)

    if barrier_jacobian is None:
        return velocity
    A = _as_matrix(barrier_jacobian, "barrier_jacobian", columns=velocity.size)
    b = _as_vector(barrier_lower_bound, "barrier_lower_bound", expected_size=A.shape[0])
    if config.max_projection_iterations <= 0:
        raise ValueError("max_projection_iterations must be positive.")
    for _ in range(config.max_projection_iterations):
        changed = False
        for row, lower in zip(A, b, strict=True):
            value = float(row @ velocity)
            violation = float(lower - value)
            denom = float(row @ row)
            if violation > config.tolerance and denom > 1.0e-18:
                velocity = velocity + (violation / denom) * row
                changed = True
        if not changed:
            break
    if equality_matrix is not None:
        velocity = velocity + np.linalg.pinv(Aeq) @ (beq - Aeq @ velocity)
    return velocity


def cbf_lower_bound(distance_m: float, safe_distance_m: float, gamma: float) -> float:
    """Return ``-gamma * h`` for ``h = distance - safe_distance``."""

    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    return -gamma * (float(distance_m) - float(safe_distance_m))


def _as_vector(values: np.ndarray | None, name: str, *, expected_size: int | None = None) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required.")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {array.shape}.")
    if expected_size is not None and array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}.")
    return array


def _as_matrix(values: np.ndarray, name: str, *, columns: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}.")
    return array
