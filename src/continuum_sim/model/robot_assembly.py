"""Composable base-plus-spatial-arms robot configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import resolve_path
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_params import SegmentParams, ThreeSegmentRobotParams


BASE_CONTROL_MODES = ("fixed", "prescribed_twist")
ARM_ROLES = ("executor", "observer")


@dataclass(frozen=True)
class SpatialArmLimits:
    """Direct tendon actuator limits for one spatial arm."""

    tendon_displacement_min_m: np.ndarray
    tendon_displacement_max_m: np.ndarray
    max_tendon_rate_mps: np.ndarray
    target_lead_m: np.ndarray


@dataclass(frozen=True)
class SpatialArmConfig:
    """Reusable single-arm definition with arm-local tendon indices."""

    path: Path
    name: str
    tendon_count: int
    limits: SpatialArmLimits
    params: ThreeSegmentRobotParams
    tendons: tuple["SpatialTendonPath", ...]


@dataclass(frozen=True)
class SpatialTendonPath:
    """Arm-local spatial tendon routing without motor transmission fields."""

    id: str
    global_index: int
    anchor_segment_index: int
    angle_deg: float
    radial_offset: float
    path_segment_indices: tuple[int, ...]


@dataclass(frozen=True)
class AssemblyBaseConfig:
    """World-frame 6D base command policy."""

    control_mode: str
    command_frame: str
    initial_pose: Pose6D
    position_min_m: np.ndarray
    position_max_m: np.ndarray
    max_linear_speed_mps: float
    max_angular_speed_rad_s: float
    limits_calibrated: bool


@dataclass(frozen=True)
class AssemblyArmConfig:
    """One named spatial arm mounted on the assembly base."""

    name: str
    role: str
    spatial_arm: SpatialArmConfig
    mount_pose: Pose6D
    attachment: str | None
    enabled: bool


@dataclass(frozen=True)
class RobotAssemblyConfig:
    """Complete composable robot assembly."""

    path: Path
    name: str
    base: AssemblyBaseConfig
    arms: dict[str, AssemblyArmConfig]

    @property
    def enabled_arms(self) -> tuple[AssemblyArmConfig, ...]:
        return tuple(arm for arm in self.arms.values() if arm.enabled)


def load_spatial_arm_config(path: str | Path) -> SpatialArmConfig:
    """Load one reusable direct-tendon spatial arm configuration."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    values = _mapping(raw.get("spatial_arm"), "spatial_arm")
    tendon_count = int(_required(values, "tendon_count", "spatial_arm"))
    if tendon_count <= 0:
        raise ValueError("spatial_arm.tendon_count must be positive.")
    limits = _mapping(values.get("limits"), "spatial_arm.limits")
    lower = _vector_or_scalar(
        _required(limits, "min_tendon_displacement_m", "spatial_arm.limits"),
        tendon_count,
        "spatial_arm.limits.min_tendon_displacement_m",
    )
    upper = _vector_or_scalar(
        _required(limits, "max_tendon_displacement_m", "spatial_arm.limits"),
        tendon_count,
        "spatial_arm.limits.max_tendon_displacement_m",
    )
    max_rate = _vector_or_scalar(
        _required(limits, "max_tendon_rate_mps", "spatial_arm.limits"),
        tendon_count,
        "spatial_arm.limits.max_tendon_rate_mps",
    )
    target_lead = _vector_or_scalar(
        limits.get("target_lead_m", 0.0005),
        tendon_count,
        "spatial_arm.limits.target_lead_m",
    )
    if np.any(lower >= upper):
        raise ValueError("Spatial-arm tendon displacement lower limits must be below upper limits.")
    if np.any(max_rate <= 0.0):
        raise ValueError("Spatial-arm max tendon rates must be positive.")
    if np.any(target_lead <= 0.0):
        raise ValueError("Spatial-arm tendon target lead limits must be positive.")
    segments_raw = values.get("segments")
    if not isinstance(segments_raw, list) or len(segments_raw) != 3:
        raise ValueError("spatial_arm.segments must contain exactly three segments.")
    segments = tuple(
        SegmentParams(
            length=float(_required(segment, "length_m", f"spatial_arm.segments[{index}]")),
            tendon_radius=float(
                _required(segment, "tendon_radius_m", f"spatial_arm.segments[{index}]")
            ),
            tendon_angles_deg=tuple(
                float(angle)
                for angle in _required(
                    segment,
                    "tendon_angles_deg",
                    f"spatial_arm.segments[{index}]",
                )
            ),
            flexure_length=(
                None
                if segment.get("flexure_length_m") is None
                else float(segment["flexure_length_m"])
            ),
            distal_straight_length=float(
                segment.get("distal_straight_length_m", 0.0)
            ),
            flexure_joint_axes=tuple(
                str(axis).lower()
                for axis in segment.get(
                    "flexure_joint_axes",
                    ("y", "x", "y", "x"),
                )
            ),
            collision_radius=(
                None
                if segment.get("collision_radius_m") is None
                else float(segment["collision_radius_m"])
            ),
            mass=(
                None if segment.get("mass_kg") is None else float(segment["mass_kg"])
            ),
            bending_stiffness=(
                None
                if segment.get("bending_stiffness_n_m2") is None
                else float(segment["bending_stiffness_n_m2"])
            ),
        )
        for index, segment in enumerate(segments_raw)
        if isinstance(segment, dict)
    )
    if len(segments) != 3:
        raise ValueError("All spatial_arm.segments entries must be mappings.")
    tendons_raw = values.get("tendons")
    if not isinstance(tendons_raw, list) or len(tendons_raw) != tendon_count:
        raise ValueError(f"spatial_arm.tendons must contain {tendon_count} entries.")
    tendons = tuple(
        SpatialTendonPath(
            id=str(tendon.get("id", f"tendon_{index + 1}")),
            global_index=int(_required(tendon, "local_index", f"spatial_arm.tendons[{index}]")),
            anchor_segment_index=int(
                _required(
                    tendon,
                    "anchor_segment_index",
                    f"spatial_arm.tendons[{index}]",
                )
            ),
            angle_deg=float(
                _required(tendon, "angle_deg", f"spatial_arm.tendons[{index}]")
            ),
            radial_offset=float(
                tendon.get(
                    "radial_offset_m",
                    segments[int(tendon["anchor_segment_index"])].tendon_radius,
                )
            ),
            path_segment_indices=tuple(
                int(value)
                for value in _required(
                    tendon,
                    "path_segment_indices",
                    f"spatial_arm.tendons[{index}]",
                )
            ),
        )
        for index, tendon in enumerate(tendons_raw)
        if isinstance(tendon, dict)
    )
    if len(tendons) != tendon_count:
        raise ValueError("All spatial_arm.tendons entries must be mappings.")
    if sorted(tendon.global_index for tendon in tendons) != list(range(tendon_count)):
        raise ValueError("spatial_arm tendon local_index values must cover 0..tendon_count-1.")
    return SpatialArmConfig(
        path=config_path,
        name=str(values.get("name", config_path.stem)),
        tendon_count=tendon_count,
        limits=SpatialArmLimits(
            tendon_displacement_min_m=lower,
            tendon_displacement_max_m=upper,
            max_tendon_rate_mps=max_rate,
            target_lead_m=target_lead,
        ),
        params=ThreeSegmentRobotParams(segments=segments),  # type: ignore[arg-type]
        tendons=tuple(sorted(tendons, key=lambda tendon: tendon.global_index)),
    )


def load_robot_assembly_config(path: str | Path) -> RobotAssemblyConfig:
    """Load a single- or multi-arm assembly without global actuator indices."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    values = _mapping(raw.get("robot_assembly"), "robot_assembly")
    base_values = _mapping(values.get("base"), "robot_assembly.base")
    mode = str(base_values.get("control_mode", "prescribed_twist"))
    if mode not in BASE_CONTROL_MODES:
        raise ValueError(f"robot_assembly.base.control_mode must be one of {BASE_CONTROL_MODES}.")
    command_frame = str(base_values.get("command_frame", "world"))
    if command_frame != "world":
        raise ValueError("Only world-frame base twist commands are supported.")
    pose = Pose6D.from_dict(
        _mapping(base_values.get("initial_pose"), "robot_assembly.base.initial_pose")
    )
    limits = _mapping(base_values.get("limits"), "robot_assembly.base.limits")
    base = AssemblyBaseConfig(
        control_mode=mode,
        command_frame=command_frame,
        initial_pose=pose,
        position_min_m=_vector3(
            limits.get("position_min_m", (-1.0, -1.0, -1.0)),
            "robot_assembly.base.limits.position_min_m",
        ),
        position_max_m=_vector3(
            limits.get("position_max_m", (1.0, 1.0, 1.0)),
            "robot_assembly.base.limits.position_max_m",
        ),
        max_linear_speed_mps=_positive_float(
            limits.get("max_linear_speed_mps", 0.05),
            "robot_assembly.base.limits.max_linear_speed_mps",
        ),
        max_angular_speed_rad_s=_positive_float(
            limits.get("max_angular_speed_rad_s", 0.3),
            "robot_assembly.base.limits.max_angular_speed_rad_s",
        ),
        limits_calibrated=bool(limits.get("calibrated", False)),
    )
    if np.any(base.position_min_m >= base.position_max_m):
        raise ValueError("Base position lower limits must be below upper limits.")

    arms_values = _mapping(values.get("arms"), "robot_assembly.arms")
    if not arms_values:
        raise ValueError("robot_assembly.arms must contain at least one arm.")
    arms: dict[str, AssemblyArmConfig] = {}
    for key, raw_arm in arms_values.items():
        arm_values = _mapping(raw_arm, f"robot_assembly.arms.{key}")
        name = str(arm_values.get("name", key))
        if name != str(key):
            raise ValueError(f"Arm key {key!r} must match arm name {name!r}.")
        role = str(_required(arm_values, "role", f"robot_assembly.arms.{key}"))
        if role not in ARM_ROLES:
            raise ValueError(f"Arm {name!r} role must be one of {ARM_ROLES}.")
        arm_path = resolve_path(
            config_path,
            _required(
                arm_values,
                "spatial_arm_config_path",
                f"robot_assembly.arms.{key}",
            ),
        )
        arms[name] = AssemblyArmConfig(
            name=name,
            role=role,
            spatial_arm=load_spatial_arm_config(arm_path),
            mount_pose=Pose6D.from_dict(
                _mapping(arm_values.get("mount_pose"), f"robot_assembly.arms.{key}.mount_pose")
            ),
            attachment=(
                None
                if arm_values.get("attachment") is None
                else str(arm_values["attachment"])
            ),
            enabled=bool(arm_values.get("enabled", True)),
        )
    return RobotAssemblyConfig(
        path=config_path,
        name=str(values.get("name", config_path.stem)),
        base=base,
        arms=arms,
    )


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _required(values: dict, name: str, section: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required config field {section}.{name}.")
    return values[name]


def _vector3(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {result.shape}.")
    return result.copy()


def _vector_or_scalar(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        return np.full((size,), float(result), dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have shape ({size},), got {result.shape}.")
    return result.copy()


def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result
