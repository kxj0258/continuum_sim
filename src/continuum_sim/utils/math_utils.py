"""Small SE(3) and angle helpers used by continuum kinematics."""

from __future__ import annotations

import numpy as np


def deg_to_rad(degrees: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert angles in degrees to a NumPy array in radians."""
    return np.deg2rad(np.asarray(degrees, dtype=float))


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix for a 3-vector."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a homogeneous 4x4 transform from rotation and translation."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform
