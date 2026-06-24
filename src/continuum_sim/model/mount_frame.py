"""Mount-frame abstractions for future mobile-base continuum integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    position_vector as _position_vector,
    positive_float_value as _positive_float_value,
    required as _required,
    section as _section,
)
from continuum_sim.model.base_pose import Pose6D


@dataclass(frozen=True)
class MobileBaseLimitsConfig:
    """Translation and angular limits for a prescribed mobile base."""

    xyz_min: np.ndarray
    xyz_max: np.ndarray
    max_linear_speed: float
    max_angular_speed: float


@dataclass(frozen=True)
class MobileBaseConfig:
    """Configured 6D mobile-base pose and limits."""

    type: str
    pose: Pose6D
    limits: MobileBaseLimitsConfig


@dataclass(frozen=True)
class MountFrameConfig:
    """Rigid mount transform between the mobile base and the continuum base."""

    name: str
    parent_frame: str
    child_frame: str
    pose: Pose6D


@dataclass(frozen=True)
class MobileBaseMountConfig:
    """Combined mobile-base and mount-frame configuration."""

    path: Path
    mobile_base: MobileBaseConfig
    mount: MountFrameConfig


def load_mobile_base_mount_config(path: str | Path) -> MobileBaseMountConfig:
    """Load a minimal mobile-base + mount YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    mobile_base_raw = _section(raw, "mobile_base")
    mount_raw = _section(raw, "mount")
    pose_raw = _section(mobile_base_raw, "pose")
    limits_raw = _section(mobile_base_raw, "limits")
    mount_pose_raw = _section(mount_raw, "pose")

    return MobileBaseMountConfig(
        path=config_path,
        mobile_base=MobileBaseConfig(
            type=str(_required(mobile_base_raw, "type")),
            pose=Pose6D.from_dict(
                {
                    "position": _position_vector(_required(pose_raw, "position"), "mobile_base.pose.position"),
                    "quat": _quat_field(pose_raw, "mobile_base.pose"),
                }
            ),
            limits=MobileBaseLimitsConfig(
                xyz_min=_position_vector(_required(limits_raw, "xyz_min"), "mobile_base.limits.xyz_min"),
                xyz_max=_position_vector(_required(limits_raw, "xyz_max"), "mobile_base.limits.xyz_max"),
                max_linear_speed=_positive_float_value(
                    _required(limits_raw, "max_linear_speed"),
                    "mobile_base.limits.max_linear_speed",
                ),
                max_angular_speed=_positive_float_value(
                    _required(limits_raw, "max_angular_speed"),
                    "mobile_base.limits.max_angular_speed",
                ),
            ),
        ),
        mount=MountFrameConfig(
            name=str(_required(mount_raw, "name")),
            parent_frame=str(_required(mount_raw, "parent_frame")),
            child_frame=str(_required(mount_raw, "child_frame")),
            pose=Pose6D.from_dict(
                {
                    "position": _position_vector(_required(mount_pose_raw, "position"), "mount.pose.position"),
                    "quat": _quat_field(mount_pose_raw, "mount.pose"),
                }
            ),
        ),
    )


def _quat_field(values: dict[str, object], name: str) -> np.ndarray:
    if "quat" in values:
        return np.asarray(values["quat"], dtype=float)
    if "quat_wxyz" in values:
        return np.asarray(values["quat_wxyz"], dtype=float)
    raise ValueError(f"Missing required config field '{name}.quat'.")
