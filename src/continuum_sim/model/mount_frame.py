"""Mount-frame and mobile-base configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    bool_field as _bool_field,
    required as _required,
    section as _section,
)
from continuum_sim.model.base_pose import Pose6D


@dataclass(frozen=True)
class MobileBaseLimitsConfig:
    """Translation and Euler-angle limits for a configured mobile base."""

    position_min_m: np.ndarray
    position_max_m: np.ndarray
    rpy_min_deg: np.ndarray
    rpy_max_deg: np.ndarray
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.3


@dataclass(frozen=True)
class MobileBaseManualControlConfig:
    """Suggested manual-control step sizes for viewer/UI integrations."""

    translation_step_m: float
    rotation_step_deg: float
    fine_translation_step_m: float
    fine_rotation_step_deg: float


@dataclass(frozen=True)
class MobileBaseVisualizationConfig:
    """Optional visualization settings for a simple MuJoCo base box."""

    enabled: bool
    type: str
    size_m: np.ndarray
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class MobileBaseConfig:
    """Configured 6D mobile-base pose and optional limits."""

    name: str
    frame: str
    pose: Pose6D
    limits: MobileBaseLimitsConfig
    manual_control: MobileBaseManualControlConfig
    visualization: MobileBaseVisualizationConfig
    type: str = "prescribed_pose"


@dataclass(frozen=True)
class MountFrameConfig:
    """Rigid mount transform between the mobile base and the continuum arm root."""

    name: str
    parent_frame: str
    child_frame: str
    pose: Pose6D

    @property
    def position_m(self) -> np.ndarray:
        return self.pose.position_m

    @property
    def quat_wxyz(self) -> np.ndarray:
        return self.pose.quat_wxyz

    def as_pose(self) -> Pose6D:
        return self.pose

    def as_matrix(self) -> np.ndarray:
        return self.pose.as_matrix()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "position_m": [float(value) for value in self.pose.position],
            "quat_wxyz": [float(value) for value in self.pose.quat],
        }


@dataclass(frozen=True)
class MobileBaseMountConfig:
    """Combined mobile-base and mount-frame configuration."""

    path: Path
    schema_version: int
    name: str
    mobile_base: MobileBaseConfig
    mounts: dict[str, MountFrameConfig]

    @property
    def mount(self) -> MountFrameConfig:
        """Backward-compatible access to the primary arm mount."""

        if "arm_mount" in self.mounts:
            return self.mounts["arm_mount"]
        return next(iter(self.mounts.values()))


def load_mobile_base_mount_config(path: str | Path) -> MobileBaseMountConfig:
    """Load a mobile-base + mount YAML with legacy compatibility."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    mobile_base_raw = _section(raw, "mobile_base")

    mounts_raw = raw.get("mounts")
    if isinstance(mounts_raw, dict) and mounts_raw:
        mounts = {
            str(key): _load_mount(str(key), value)
            for key, value in mounts_raw.items()
        }
    else:
        mount_raw = _section(raw, "mount")
        mount_name = str(mount_raw.get("name", "arm_mount"))
        mounts = {mount_name: _load_mount(mount_name, mount_raw)}

    return MobileBaseMountConfig(
        path=config_path,
        schema_version=int(raw.get("schema_version", 1)),
        name=str(raw.get("name", "mobile_base_pose")),
        mobile_base=_load_mobile_base(mobile_base_raw),
        mounts=mounts,
    )


def _load_mobile_base(values: dict[str, object]) -> MobileBaseConfig:
    pose_raw = _section(values, "pose")
    limits_raw = _section(values, "limits")
    manual_control_raw = values.get("manual_control", {})
    if not isinstance(manual_control_raw, dict):
        raise ValueError("mobile_base.manual_control must be a mapping when provided.")
    visualization_raw = values.get("visualization", {})
    if not isinstance(visualization_raw, dict):
        raise ValueError("mobile_base.visualization must be a mapping when provided.")

    return MobileBaseConfig(
        name=str(values.get("name", "mobile_base_pose")),
        frame=str(values.get("frame", "world")),
        type=str(values.get("type", "prescribed_pose")),
        pose=Pose6D.from_dict(pose_raw),
        limits=MobileBaseLimitsConfig(
            position_min_m=_vector3(
                limits_raw.get("position_min_m", limits_raw.get("xyz_min", (-1.0, -1.0, -1.0))),
                "mobile_base.limits.position_min_m",
            ),
            position_max_m=_vector3(
                limits_raw.get("position_max_m", limits_raw.get("xyz_max", (1.0, 1.0, 1.0))),
                "mobile_base.limits.position_max_m",
            ),
            rpy_min_deg=_vector3(
                limits_raw.get("rpy_min_deg", (-180.0, -180.0, -180.0)),
                "mobile_base.limits.rpy_min_deg",
            ),
            rpy_max_deg=_vector3(
                limits_raw.get("rpy_max_deg", (180.0, 180.0, 180.0)),
                "mobile_base.limits.rpy_max_deg",
            ),
            max_linear_speed=float(limits_raw.get("max_linear_speed", 0.05)),
            max_angular_speed=float(limits_raw.get("max_angular_speed", 0.3)),
        ),
        manual_control=MobileBaseManualControlConfig(
            translation_step_m=float(manual_control_raw.get("translation_step_m", 0.01)),
            rotation_step_deg=float(manual_control_raw.get("rotation_step_deg", 2.0)),
            fine_translation_step_m=float(manual_control_raw.get("fine_translation_step_m", 0.002)),
            fine_rotation_step_deg=float(manual_control_raw.get("fine_rotation_step_deg", 0.5)),
        ),
        visualization=MobileBaseVisualizationConfig(
            enabled=_bool_field(visualization_raw, "enabled", default=True),
            type=str(visualization_raw.get("type", "box")),
            size_m=_vector3(
                visualization_raw.get("size_m", (0.10, 0.06, 0.04)),
                "mobile_base.visualization.size_m",
            ),
            rgba=_rgba4(
                visualization_raw.get("rgba", (0.35, 0.35, 0.35, 1.0)),
                "mobile_base.visualization.rgba",
            ),
        ),
    )


def _load_mount(default_name: str, raw_value: object) -> MountFrameConfig:
    if not isinstance(raw_value, dict):
        raise ValueError(f"Mount config {default_name!r} must be a mapping.")
    if "pose" in raw_value:
        pose_raw = _section(raw_value, "pose")
    else:
        pose_raw = raw_value
    return MountFrameConfig(
        name=str(raw_value.get("name", default_name)),
        parent_frame=str(_required(raw_value, "parent_frame")),
        child_frame=str(_required(raw_value, "child_frame")),
        pose=Pose6D.from_dict(pose_raw),
    )


def _vector3(raw_value: object, name: str) -> np.ndarray:
    values = np.asarray(raw_value, dtype=float)
    if values.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {values.shape}.")
    return values.copy()


def _rgba4(raw_value: object, name: str) -> tuple[float, float, float, float]:
    values = np.asarray(raw_value, dtype=float)
    if values.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {values.shape}.")
    return tuple(float(value) for value in values)
