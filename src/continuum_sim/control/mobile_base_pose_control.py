"""Bounded world-frame pose control for the prescribed mobile base."""

from __future__ import annotations

import numpy as np

from continuum_sim.model.base_pose import Pose6D


class MobileBasePoseController:
    """Convert an SE(3) pose error into a bounded world-frame twist."""

    def __init__(self, *, position_gain: float, orientation_gain: float) -> None:
        if position_gain <= 0.0 or orientation_gain <= 0.0:
            raise ValueError("Mobile-base pose gains must be positive.")
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)

    def compute_twist(
        self,
        current: Pose6D,
        target: Pose6D,
        *,
        max_linear_speed: float,
        max_angular_speed: float,
    ) -> tuple[np.ndarray, float, float]:
        """Return bounded twist plus position and orientation error norms."""

        position_delta = target.position - current.position
        current_rotation = current.as_matrix()[:3, :3]
        target_rotation = target.as_matrix()[:3, :3]
        rotation_vector = _rotation_matrix_to_vector(
            target_rotation @ current_rotation.T
        )
        linear = _limit_norm(
            self.position_gain * position_delta,
            max_linear_speed,
        )
        angular = _limit_norm(
            self.orientation_gain * rotation_vector,
            max_angular_speed,
        )
        return (
            np.concatenate((linear, angular)),
            float(np.linalg.norm(position_delta)),
            float(np.linalg.norm(rotation_vector)),
        )


def _limit_norm(values: np.ndarray, limit: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if limit <= 0.0:
        raise ValueError("Velocity limits must be positive.")
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm <= 1.0e-15:
        return vector.copy()
    return vector * (float(limit) / norm)


def _rotation_matrix_to_vector(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3).")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )
    if angle <= 1.0e-9:
        return 0.5 * skew
    sine = float(np.sin(angle))
    if abs(sine) > 1.0e-7:
        return (0.5 * angle / sine) * skew

    diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
    axis = np.sqrt(diagonal)
    axis[1] = np.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
    axis[2] = np.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis /= norm
    return angle * axis
