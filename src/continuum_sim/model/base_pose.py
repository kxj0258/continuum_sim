"""6D pose helpers for world/base/mount composition.

Quaternion order is always ``[w, x, y, z]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose6D:
    """Rigid pose with position and quaternion in ``[w, x, y, z]`` order."""

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

    @property
    def position_m(self) -> np.ndarray:
        return self.position.copy()

    @property
    def quat_wxyz(self) -> np.ndarray:
        return self.quat.copy()

    @classmethod
    def identity(cls) -> "Pose6D":
        """Return the identity pose."""

        return cls(
            position=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "Pose6D":
        """Build a pose from a YAML-friendly dictionary."""

        if not isinstance(values, dict):
            raise ValueError("Pose config must be a mapping.")

        position = _position_field(values)
        quat = _quat_field(values)
        if quat is None:
            quat = _rpy_field_to_quaternion(values)
        if quat is None:
            raise ValueError("Pose config must define quat/quat_wxyz or rpy/rpy_deg/rpy_rad.")
        return cls(position=position, quat=quat)

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

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Pose6D":
        """Alias for :meth:`from_transform`."""

        return cls.from_transform(matrix)

    @classmethod
    def from_rpy_rad(
        cls,
        position: object = (0.0, 0.0, 0.0),
        rpy_rad: object = (0.0, 0.0, 0.0),
    ) -> "Pose6D":
        """Build a pose from position and roll/pitch/yaw in radians."""

        return cls(
            position=np.asarray(position, dtype=float),
            quat=_rpy_to_quaternion_wxyz(np.asarray(rpy_rad, dtype=float)),
        )

    @classmethod
    def from_rpy_deg(
        cls,
        position: object = (0.0, 0.0, 0.0),
        rpy_deg: object = (0.0, 0.0, 0.0),
    ) -> "Pose6D":
        """Build a pose from position and roll/pitch/yaw in degrees."""

        return cls.from_rpy_rad(position=position, rpy_rad=np.deg2rad(np.asarray(rpy_deg, dtype=float)))

    def as_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform."""

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = _quaternion_wxyz_to_rotation_matrix(self.quat)
        transform[:3, 3] = self.position
        return transform

    def to_transform(self) -> np.ndarray:
        """Backward-compatible alias for :meth:`as_matrix`."""

        return self.as_matrix()

    def to_dict(self) -> dict[str, list[float]]:
        """Export the pose as a YAML-friendly dictionary."""

        position = [float(value) for value in self.position]
        quat = [float(value) for value in self.quat]
        return {
            "position": position,
            "quat": quat,
            "position_m": position,
            "quat_wxyz": quat,
        }

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
        """Return ``self * other`` in SE(3) composition order."""

        composed = self.as_matrix() @ other.as_matrix()
        return Pose6D.from_matrix(composed)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a single 3D point."""

        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (3,):
            raise ValueError(f"Expected point with shape (3,), got {point_array.shape}.")
        rotation = _quaternion_wxyz_to_rotation_matrix(self.quat)
        return rotation @ point_array + self.position

    def apply_to_point(self, point: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for :meth:`transform_point`."""

        return self.transform_point(point)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an ``N x 3`` point cloud."""

        points_array = np.asarray(points, dtype=float)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points_array.shape}.")
        rotation = _quaternion_wxyz_to_rotation_matrix(self.quat)
        return (rotation @ points_array.T).T + self.position

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for :meth:`transform_points`."""

        return self.transform_points(points)

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        """Rotate a direction vector without translating it."""

        vector_array = np.asarray(vector, dtype=float)
        if vector_array.shape != (3,):
            raise ValueError(f"Expected vector with shape (3,), got {vector_array.shape}.")
        rotation = _quaternion_wxyz_to_rotation_matrix(self.quat)
        return rotation @ vector_array

    def transform_pose(self, pose: "Pose6D") -> "Pose6D":
        """Apply this pose to another pose."""

        return self.compose(pose)

    def apply_to_pose(self, pose: "Pose6D") -> "Pose6D":
        """Backward-compatible alias for :meth:`transform_pose`."""

        return self.transform_pose(pose)


def pose_from_yaml_dict(values: dict[str, object]) -> Pose6D:
    """Small semantic alias used by YAML loaders."""

    return Pose6D.from_dict(values)


def quaternion_wxyz_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return ``left * right`` in Hamilton product order."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != (4,) or right_array.shape != (4,):
        raise ValueError("Quaternion multiply expects two `(4,)` arrays.")
    lw, lx, ly, lz = left_array
    rw, rx, ry, rz = right_array
    quat = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion multiply produced a near-zero quaternion.")
    return quat / norm


def rotation_vector_to_quaternion_wxyz(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to a unit quaternion."""

    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Expected rotation_vector with shape (3,), got {vector.shape}.")
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = vector / angle
    half_angle = 0.5 * angle
    sin_half = float(np.sin(half_angle))
    return np.array(
        [
            float(np.cos(half_angle)),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        ],
        dtype=float,
    )


def _position_field(values: dict[str, object]) -> np.ndarray:
    if "position" in values:
        return np.asarray(values["position"], dtype=float)
    if "position_m" in values:
        return np.asarray(values["position_m"], dtype=float)
    raise ValueError("Missing required config field 'position' or 'position_m'.")


def _quat_field(values: dict[str, object]) -> np.ndarray | None:
    if "quat" in values:
        return np.asarray(values["quat"], dtype=float)
    if "quat_wxyz" in values:
        return np.asarray(values["quat_wxyz"], dtype=float)
    return None


def _rpy_field_to_quaternion(values: dict[str, object]) -> np.ndarray | None:
    if "rpy" in values:
        return _rpy_to_quaternion_wxyz(np.asarray(values["rpy"], dtype=float))
    if "rpy_rad" in values:
        return _rpy_to_quaternion_wxyz(np.asarray(values["rpy_rad"], dtype=float))
    if "rpy_deg" in values:
        return _rpy_to_quaternion_wxyz(np.deg2rad(np.asarray(values["rpy_deg"], dtype=float)))
    return None


def _rpy_to_quaternion_wxyz(rpy_rad: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy_rad, dtype=float)
    if rpy.shape != (3,):
        raise ValueError(f"Expected rpy with shape (3,), got {rpy.shape}.")
    roll, pitch, yaw = rpy
    cr = float(np.cos(0.5 * roll))
    sr = float(np.sin(0.5 * roll))
    cp = float(np.cos(0.5 * pitch))
    sp = float(np.sin(0.5 * pitch))
    cy = float(np.cos(0.5 * yaw))
    sy = float(np.sin(0.5 * yaw))
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("RPY conversion produced a near-zero quaternion.")
    return quat / norm


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
