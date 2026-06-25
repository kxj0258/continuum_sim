"""Pinhole camera model scaffold.

This module provides only camera intrinsics math for future observer-arm work.
It does not implement a MuJoCo renderer or visual recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from continuum_sim.model.base_pose import Pose6D


@dataclass(frozen=True)
class CameraIntrinsicsConfig:
    """Minimal pinhole camera intrinsics."""

    width: int
    height: int
    fovy_deg: float
    near: float
    far: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}.")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}.")
        if self.fovy_deg <= 1.0 or self.fovy_deg >= 179.0:
            raise ValueError(f"fovy_deg must be in (1, 179), got {self.fovy_deg}.")
        if self.near <= 0.0:
            raise ValueError(f"near must be positive, got {self.near}.")
        if self.far <= self.near:
            raise ValueError(f"far must be greater than near, got {self.far} <= {self.near}.")


@dataclass(frozen=True)
class CameraConfig:
    """Named camera frame attached to a continuum tip."""

    name: str
    intrinsics: CameraIntrinsicsConfig
    tip_to_camera: Pose6D


def vertical_fov_rad(intrinsics: CameraIntrinsicsConfig) -> float:
    """Return vertical field of view in radians."""

    return math.radians(intrinsics.fovy_deg)


def horizontal_fov_rad(intrinsics: CameraIntrinsicsConfig) -> float:
    """Return horizontal field of view in radians."""

    aspect = float(intrinsics.width) / float(intrinsics.height)
    return 2.0 * math.atan(aspect * math.tan(vertical_fov_rad(intrinsics) / 2.0))


def focal_lengths_px(intrinsics: CameraIntrinsicsConfig) -> tuple[float, float]:
    """Return `(fx, fy)` in pixels for square pixels."""

    fy = 0.5 * float(intrinsics.height) / math.tan(vertical_fov_rad(intrinsics) / 2.0)
    fx = fy
    return fx, fy


def principal_point_px(intrinsics: CameraIntrinsicsConfig) -> tuple[float, float]:
    """Return image-center principal point `(cx, cy)` in pixels."""

    return (float(intrinsics.width) - 1.0) / 2.0, (float(intrinsics.height) - 1.0) / 2.0


def camera_matrix(intrinsics: CameraIntrinsicsConfig) -> np.ndarray:
    """Return the 3x3 pinhole camera matrix."""

    fx, fy = focal_lengths_px(intrinsics)
    cx, cy = principal_point_px(intrinsics)
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
