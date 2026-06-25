"""Dual-continuum arm configuration scaffold.

This module only describes multi-arm configuration and mount transforms for
future engine-cleaning work. It does not implement controllers, visual servo,
collision avoidance, or snake-arm dynamics. Later milestones M4, M7, M8, and
M9 can build attachment, perception, avoidance, and task-state layers on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    bool_field as _bool_field,
    position_vector as _position_vector,
    required as _required,
    resolve_path as _resolve_path,
    section as _section,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.mount_frame import MountFrameConfig


ALLOWED_ARM_ROLES: tuple[str, ...] = ("observer", "executor")


@dataclass(frozen=True)
class ArmConfig:
    """Configuration for one continuum arm mounted on the mobile base."""

    name: str
    role: str
    robot_config_path: str | None
    mount: MountFrameConfig
    attachment: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class MultiArmConfig:
    """Configuration for a mobile-base-mounted set of continuum arms."""

    base_frame: str
    arms: dict[str, ArmConfig]
    default_arm: str | None = None
    path: Path | None = None
    allow_single_arm_mode: bool = False


def load_multi_arm_config(path: str | Path, *, strict_paths: bool = False) -> MultiArmConfig:
    """Load and validate a dual-continuum-arm YAML configuration."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    multi_arm_raw = _section(raw, "multi_arm")
    arms_raw = _section(multi_arm_raw, "arms")

    config = MultiArmConfig(
        path=config_path,
        base_frame=str(_required(multi_arm_raw, "base_frame")),
        default_arm=_optional_string(multi_arm_raw.get("default_arm")),
        allow_single_arm_mode=_bool_field(multi_arm_raw, "allow_single_arm_mode", default=False),
        arms={
            str(key): _load_arm_config(str(key), value)
            for key, value in arms_raw.items()
        },
    )
    validate_multi_arm_config(config, strict_paths=strict_paths)
    return config


def validate_multi_arm_config(config: MultiArmConfig, *, strict_paths: bool = False) -> None:
    """Validate role, naming, mount, and optional robot path constraints."""

    if not config.base_frame:
        raise ValueError("multi_arm.base_frame must be non-empty.")
    if not isinstance(config.arms, dict) or not config.arms:
        raise ValueError("multi_arm.arms must be a non-empty mapping.")

    seen_names: set[str] = set()
    roles: set[str] = set()
    for key, arm in config.arms.items():
        if not isinstance(arm, ArmConfig):
            raise ValueError(f"multi_arm.arms.{key} must be an ArmConfig.")
        if key != arm.name:
            raise ValueError(f"Arm key {key!r} must match ArmConfig.name {arm.name!r}.")
        if arm.name in seen_names:
            raise ValueError(f"Duplicate arm name {arm.name!r}.")
        seen_names.add(arm.name)
        if arm.role not in ALLOWED_ARM_ROLES:
            raise ValueError(f"arm {arm.name!r} role must be one of {ALLOWED_ARM_ROLES}, got {arm.role!r}.")
        roles.add(arm.role)
        if arm.mount.parent_frame != config.base_frame:
            raise ValueError(
                f"arm {arm.name!r} mount parent_frame must match base_frame "
                f"{config.base_frame!r}, got {arm.mount.parent_frame!r}."
            )
        if strict_paths and arm.robot_config_path is not None:
            robot_path = _resolve_robot_config_path(config, arm.robot_config_path)
            if not robot_path.is_file():
                raise FileNotFoundError(f"Robot config file does not exist for arm {arm.name!r}: {robot_path}")

    if config.default_arm is not None and config.default_arm not in config.arms:
        raise ValueError(f"multi_arm.default_arm {config.default_arm!r} is not listed in multi_arm.arms.")

    if not config.allow_single_arm_mode:
        missing_roles = [role for role in ALLOWED_ARM_ROLES if role not in roles]
        if missing_roles:
            missing = ", ".join(missing_roles)
            raise ValueError(f"multi_arm must include at least one arm for role(s): {missing}.")


def iter_enabled_arms(config: MultiArmConfig) -> Iterator[ArmConfig]:
    """Yield enabled arms in YAML/dict insertion order."""

    return (arm for arm in config.arms.values() if arm.enabled)


def get_arm(config: MultiArmConfig, name: str) -> ArmConfig:
    """Return one arm by name."""

    try:
        return config.arms[name]
    except KeyError as exc:
        raise KeyError(f"Unknown arm {name!r}.") from exc


def get_arms_by_role(config: MultiArmConfig, role: str) -> list[ArmConfig]:
    """Return all arms with the requested role."""

    if role not in ALLOWED_ARM_ROLES:
        raise ValueError(f"role must be one of {ALLOWED_ARM_ROLES}, got {role!r}.")
    return [arm for arm in config.arms.values() if arm.role == role]


def _load_arm_config(key: str, raw_value: object) -> ArmConfig:
    if not isinstance(raw_value, dict):
        raise ValueError(f"multi_arm.arms.{key} must be a mapping.")
    mount_raw = _section(raw_value, "mount")
    mount_pose_raw = _section(mount_raw, "pose")
    return ArmConfig(
        name=str(_required(raw_value, "name")),
        role=str(_required(raw_value, "role")),
        robot_config_path=_optional_string(raw_value.get("robot_config_path")),
        attachment=_optional_string(raw_value.get("attachment")),
        enabled=_bool_field(raw_value, "enabled", default=True),
        mount=MountFrameConfig(
            name=str(_required(mount_raw, "name")),
            parent_frame=str(_required(mount_raw, "parent_frame")),
            child_frame=str(_required(mount_raw, "child_frame")),
            pose=Pose6D.from_dict(
                {
                    "position": _position_vector(
                        _required(mount_pose_raw, "position"),
                        f"multi_arm.arms.{key}.mount.pose.position",
                    ),
                    "quat": _quat_field(mount_pose_raw, f"multi_arm.arms.{key}.mount.pose"),
                }
            ),
        ),
    )


def _resolve_robot_config_path(config: MultiArmConfig, raw_path: str) -> Path:
    if config.path is not None:
        return _resolve_path(config.path, raw_path)
    return Path(raw_path).resolve()


def _optional_string(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    return str(raw_value)


def _quat_field(values: dict[str, object], name: str) -> np.ndarray:
    if "quat" in values:
        return np.asarray(values["quat"], dtype=float)
    if "quat_wxyz" in values:
        return np.asarray(values["quat_wxyz"], dtype=float)
    raise ValueError(f"Missing required config field '{name}.quat'.")
