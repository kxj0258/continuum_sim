"""Structured visual-servo feedback for observer-mounted cameras."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.sensing.camera_model import (
    CameraIntrinsicsConfig,
    focal_lengths_px,
    principal_point_px,
)


@dataclass(frozen=True)
class VisualServoFeedback:
    """Compact ROI observation used by control and artifacts."""

    target_visible: bool
    pixel_error_px: np.ndarray
    normalized_error: np.ndarray
    depth_m: float
    target_camera_m: np.ndarray
    target_world_m: np.ndarray
    camera_position_world_m: np.ndarray
    camera_quat_world_wxyz: np.ndarray
    timestamp_s: float
    source: str = "engine_roi"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pixel_error_px",
            _finite_vector(self.pixel_error_px, 2, "pixel_error_px", allow_nan=True),
        )
        object.__setattr__(
            self,
            "normalized_error",
            _finite_vector(
                self.normalized_error,
                2,
                "normalized_error",
                allow_nan=True,
            ),
        )
        object.__setattr__(
            self,
            "target_camera_m",
            _finite_vector(self.target_camera_m, 3, "target_camera_m"),
        )
        object.__setattr__(
            self,
            "target_world_m",
            _finite_vector(self.target_world_m, 3, "target_world_m"),
        )
        object.__setattr__(
            self,
            "camera_position_world_m",
            _finite_vector(
                self.camera_position_world_m,
                3,
                "camera_position_world_m",
            ),
        )
        quat = np.asarray(self.camera_quat_world_wxyz, dtype=float)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError("camera_quat_world_wxyz must be a finite 4-vector.")
        norm = float(np.linalg.norm(quat))
        if norm <= 1.0e-12:
            raise ValueError("camera_quat_world_wxyz must have non-zero length.")
        object.__setattr__(self, "camera_quat_world_wxyz", (quat / norm).copy())
        if not np.isfinite(self.depth_m):
            raise ValueError("depth_m must be finite.")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite.")

    def as_metadata(self) -> dict[str, object]:
        """Return numeric metadata safe to attach to state/command histories."""

        return {
            "visual_servo_source": self.source,
            "visual_servo_target_visible": bool(self.target_visible),
            "visual_servo_pixel_error_px": self.pixel_error_px.copy(),
            "visual_servo_normalized_error": self.normalized_error.copy(),
            "visual_servo_depth_m": float(self.depth_m),
            "visual_servo_target_camera_m": self.target_camera_m.copy(),
            "visual_servo_roi_world": self.target_world_m.copy(),
            "visual_servo_camera_position_world": (
                self.camera_position_world_m.copy()
            ),
            "visual_servo_camera_quat_world_wxyz": (
                self.camera_quat_world_wxyz.copy()
            ),
            "visual_servo_timestamp_s": float(self.timestamp_s),
        }


def project_roi_to_camera_feedback(
    roi_world_m: np.ndarray,
    camera_pose_world: Pose6D,
    intrinsics: CameraIntrinsicsConfig,
    *,
    timestamp_s: float = 0.0,
    source: str = "engine_roi",
) -> VisualServoFeedback:
    """Project one world-frame ROI point into a +Z-forward pinhole camera."""

    target_world = _finite_vector(roi_world_m, 3, "roi_world_m")
    camera_to_world = camera_pose_world.as_matrix()
    world_to_camera = np.linalg.inv(camera_to_world)
    target_camera = (world_to_camera @ np.append(target_world, 1.0))[:3]
    fx, fy = focal_lengths_px(intrinsics)
    cx, cy = principal_point_px(intrinsics)
    depth = float(target_camera[2])
    if abs(depth) <= 1.0e-12:
        pixel = np.array([np.nan, np.nan], dtype=float)
    else:
        pixel = np.array(
            [
                fx * target_camera[0] / depth + cx,
                fy * target_camera[1] / depth + cy,
            ],
            dtype=float,
        )
    center = np.array([cx, cy], dtype=float)
    pixel_error = pixel - center
    normalized = np.array(
        [
            target_camera[0] / depth if abs(depth) > 1.0e-12 else np.nan,
            target_camera[1] / depth if abs(depth) > 1.0e-12 else np.nan,
        ],
        dtype=float,
    )
    visible = bool(
        np.all(np.isfinite(pixel))
        and intrinsics.near <= depth <= intrinsics.far
        and 0.0 <= pixel[0] < intrinsics.width
        and 0.0 <= pixel[1] < intrinsics.height
    )
    return VisualServoFeedback(
        target_visible=visible,
        pixel_error_px=pixel_error,
        normalized_error=normalized,
        depth_m=depth,
        target_camera_m=target_camera,
        target_world_m=target_world,
        camera_position_world_m=camera_pose_world.position,
        camera_quat_world_wxyz=camera_pose_world.quat,
        timestamp_s=timestamp_s,
        source=source,
    )


def _finite_vector(
    values: np.ndarray,
    size: int,
    name: str,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}.")
    finite = np.isfinite(result) if not allow_nan else ~np.isinf(result)
    if not np.all(finite):
        raise ValueError(f"{name} must be finite.")
    return result.copy()
