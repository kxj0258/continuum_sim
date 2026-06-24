"""6D pose helpers for world/base/mount composition.

Quaternion order is `[w, x, y, z]`.
This module only provides rigid pose composition and point transforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose6D:
    """Rigid pose with position and quaternion in `[w, x, y, z]` order."""

    position: np.ndarray
    quat: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        quat = np.asarray(self.quat, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"Expected position with shape (3,), got {position.shape}.")
        if quat.shape != (4,):
            raise ValueError(f"Expected quat with shape (4,), got {quat.shape}.")
        norm = float(np.linalg.norm(quat))
        if norm <= 1.0e-12:
            raise ValueError("quat must have non-zero length.")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "quat", (quat / norm).copy())

    @classmethod
    def identity(cls) -> "Pose6D":
        """Return the identity pose."""

        return cls(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "Pose6D":
        """Build a pose from `{"position": [...], "quat": [...]}`."""

        if "position" not in values:
            raise ValueError("Missing required config field 'position'.")
        if "quat" not in values:
            raise ValueError("Missing required config field 'quat'.")
        return cls(position=np.asarray(values["position"], dtype=float), quat=np.asarray(values["quat"], dtype=float))

    @classmethod
    def from_transform(cls, transform: np.ndarray) -> "Pose6D":
        """Build a pose from a 4x4 homogeneous transform."""

        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected transform with shape (4, 4), got {matrix.shape}.")
        return cls(
            position=matrix[:3, 3],
            quat=_rotation_matrix_to_quaternion_wxyz(matrix[:3, :3]),
        )

    def to_transform(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform."""

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = _quaternion_wxyz_to_rotation_matrix(self.quat)
        transform[:3, 3] = self.position
        return transform

    def inverse(self) -> "Pose6D":
        """Return the inverse rigid pose."""

        rotation = _quaternion_wxyz_to_rotation_matrix(self.quat)
        inverse_rotation = rotation.T
        inverse_position = -inverse_rotation @ self.position
        return Pose6D(
            position=inverse_position,
            quat=_rotation_matrix_to_quaternion_wxyz(inverse_rotation),
        )

    def compose(self, other: "Pose6D") -> "Pose6D":
        """Return `self * other` in SE(3) composition order."""

        composed = self.to_transform() @ other.to_transform()
        return Pose6D.from_transform(composed)

    def apply_to_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a single 3D point."""

        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (3,):
            raise ValueError(f"Expected point with shape (3,), got {point_array.shape}.")
        rotation = _quaternion_wxyz_to_rotation_matrix(self.quat)
        return rotation @ point_array + self.position

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an `N x 3` point cloud."""

        points_array = np.asarray(points, dtype=float)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points_array.shape}.")
        rotation = _quaternion_wxyz_to_rotation_matrix(self.quat)
        return (rotation @ points_array.T).T + self.position

    def apply_to_pose(self, pose: "Pose6D") -> "Pose6D":
        """Apply this pose to another pose."""

        return self.compose(pose)


def _quaternion_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion with shape (4,), got {q.shape}.")
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation with shape (3, 3), got {matrix.shape}.")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Rotation matrix produced a zero-length quaternion.")
    return quat / norm
