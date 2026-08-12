"""Tool and camera attachment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    bool_field as _bool_field,
    choice_value as _choice_value,
    optional_section as _optional_section,
    position_vector as _position_vector,
    positive_float_value as _positive_float_value,
    required as _required,
    resolve_path as _resolve_path,
    section as _section,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.sensing.camera_model import CameraConfig, CameraIntrinsicsConfig


ATTACHMENT_TYPES: tuple[str, ...] = ("contact_sphere_tool", "camera_airgun")
COLLISION_TYPES: tuple[str, ...] = ("sphere",)


@dataclass(frozen=True)
class CollisionGeometryConfig:
    """Simple collision geometry for a tip attachment."""

    type: str
    radius_m: float | None = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rgba: tuple[float, float, float, float] = (0.2, 0.22, 0.24, 1.0)
    friction: tuple[float, float, float] = (0.8, 0.02, 0.005)
    contype: int = 1
    conaffinity: int = 1


@dataclass(frozen=True)
class ForceTorqueSensorConfig:
    """Six-axis force/torque package mounted between the arm and tool."""

    size_m: np.ndarray
    mass_kg: float
    rgba: tuple[float, float, float, float]
    force_limit_n: float
    torque_limit_nm: float
    filter_cutoff_hz: float
    tare_on_reset: bool = True
    gravity_compensation: bool = True
    output_sign: float = -1.0

    def __post_init__(self) -> None:
        size = np.asarray(self.size_m, dtype=float)
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError("force_torque_sensor.size_m must be a positive 3-vector.")
        object.__setattr__(self, "size_m", size.copy())


@dataclass(frozen=True)
class CameraVisualConfig:
    """Visible dome and lens mounted around an observer camera."""

    shape: str
    radius_m: float
    rgba: tuple[float, float, float, float]
    lens_radius_m: float
    lens_depth_m: float
    lens_rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class ContactToolConfig:
    """Contact-force limits for a placeholder contact tool."""

    target_normal_force_n: float
    max_normal_force_n: float
    standoff_distance_m: float | None = None


@dataclass(frozen=True)
class AirgunConfig:
    """Airgun standoff placeholder for the observer attachment."""

    standoff_distance_m: float


@dataclass(frozen=True)
class AttachmentConfig:
    """Attachment mounted at a continuum arm tip."""

    name: str
    type: str
    enabled: bool
    tip_to_attachment: Pose6D
    path: Path | None = None
    visual_mesh_path: Path | None = None
    collision: CollisionGeometryConfig | None = None
    mass_kg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tcp_pose: Pose6D | None = None
    contact: ContactToolConfig | None = None
    camera: CameraConfig | None = None
    camera_visual: CameraVisualConfig | None = None
    nozzle_pose: Pose6D | None = None
    airgun: AirgunConfig | None = None
    force_torque_sensor: ForceTorqueSensorConfig | None = None


def load_attachment_config(path: str | Path, *, strict_assets: bool = False) -> AttachmentConfig:
    """Load and validate a standalone attachment YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    tool_raw = _section(raw, "tool")
    attachment_type = _choice_value(_required(tool_raw, "type"), "tool.type", ATTACHMENT_TYPES)
    config = AttachmentConfig(
        path=config_path,
        name=str(_required(tool_raw, "name")),
        type=attachment_type,
        enabled=_bool_field(tool_raw, "enabled", default=True),
        tip_to_attachment=_pose_from_section(tool_raw, "tip_to_attachment", "tool.tip_to_attachment"),
        visual_mesh_path=_optional_resolved_path(config_path, tool_raw.get("visual_mesh_path")),
        collision=_load_collision(tool_raw.get("collision")),
        mass_kg=_optional_positive_float(tool_raw.get("mass_kg"), "tool.mass_kg"),
        metadata=_optional_section(tool_raw, "metadata"),
        tcp_pose=_load_optional_pose(tool_raw, "tcp_pose", "tool.tcp_pose"),
        contact=_load_contact(tool_raw.get("contact")),
        camera=_load_camera(tool_raw.get("camera")),
        camera_visual=_load_camera_visual(tool_raw.get("camera")),
        nozzle_pose=_load_optional_pose(tool_raw, "nozzle_pose", "tool.nozzle_pose"),
        airgun=_load_airgun(tool_raw.get("airgun")),
        force_torque_sensor=_load_force_torque_sensor(
            tool_raw.get("force_torque_sensor")
        ),
    )
    _validate_attachment_fields(config)
    if strict_assets:
        _validate_attachment_asset(config, strict_assets=True)
    return config


def validate_attachment_config(
    config: AttachmentConfig,
    *,
    strict_assets: bool = False,
) -> None:
    """Validate attachment type-specific fields and optional mesh existence."""

    _validate_attachment_fields(config)
    _validate_attachment_asset(config, strict_assets=strict_assets)


def _validate_attachment_fields(config: AttachmentConfig) -> None:
    if not config.name:
        raise ValueError("tool.name must be non-empty.")
    if config.type not in ATTACHMENT_TYPES:
        raise ValueError(f"tool.type must be one of {ATTACHMENT_TYPES}, got {config.type!r}.")

    if config.type == "contact_sphere_tool":
        if config.tcp_pose is None:
            raise ValueError("contact_sphere_tool requires tool.tcp_pose.")
        if config.collision is None:
            raise ValueError("contact_sphere_tool requires tool.collision.")
        if config.collision.type != "sphere":
            raise ValueError("contact_sphere_tool collision.type must be 'sphere'.")
        if config.collision.radius_m is None or config.collision.radius_m <= 0.0:
            raise ValueError("contact_sphere_tool collision.radius_m must be positive.")
        if config.contact is None:
            raise ValueError("contact_sphere_tool requires tool.contact.")
    elif config.type == "camera_airgun":
        if config.camera is None:
            raise ValueError("camera_airgun requires tool.camera.")
        if config.nozzle_pose is None:
            raise ValueError("camera_airgun requires tool.nozzle_pose.")
        if config.airgun is None:
            raise ValueError("camera_airgun requires tool.airgun.")


def _validate_attachment_asset(config: AttachmentConfig, *, strict_assets: bool) -> None:
    if config.visual_mesh_path is not None and not config.visual_mesh_path.exists():
        message = (
            f"Attachment visual mesh does not exist: {config.visual_mesh_path}. "
            "Set strict_assets=False to allow placeholders."
        )
        if strict_assets:
            raise FileNotFoundError(message)
        warnings.warn(message, UserWarning, stacklevel=2)


def load_attachment_registry(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    strict_assets: bool = False,
) -> dict[str, AttachmentConfig]:
    """Load attachment configs into a name-keyed registry."""

    registry: dict[str, AttachmentConfig] = {}
    for path in paths:
        config = load_attachment_config(path, strict_assets=strict_assets)
        if config.name in registry:
            raise ValueError(f"Duplicate attachment name {config.name!r}.")
        registry[config.name] = config
    return registry


def get_attachment(
    registry: dict[str, AttachmentConfig],
    name: str,
) -> AttachmentConfig:
    """Return an attachment by name from a registry."""

    try:
        return registry[name]
    except KeyError as exc:
        raise KeyError(f"Unknown attachment {name!r}.") from exc


def attachment_config_path(
    assembly_path: str | Path,
    attachment_name: str,
) -> Path | None:
    """Resolve one named tool config relative to an assembly configuration."""

    resolved = Path(assembly_path).resolve()
    for parent in (resolved.parent, *resolved.parents):
        candidate = parent / "tools" / f"{attachment_name}.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_assembly_attachment_configs(assembly) -> dict[str, AttachmentConfig]:
    """Load enabled arm attachments keyed by arm name."""

    result: dict[str, AttachmentConfig] = {}
    for arm in assembly.enabled_arms:
        if arm.attachment is None:
            continue
        path = attachment_config_path(assembly.path, arm.attachment)
        if path is None:
            raise FileNotFoundError(
                f"Attachment config {arm.attachment!r} for arm {arm.name!r} was not found."
            )
        result[arm.name] = load_attachment_config(path)
    return result


def _load_collision(raw_value: object) -> CollisionGeometryConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.collision must be a mapping.")
    collision_type = _choice_value(_required(raw_value, "type"), "tool.collision.type", COLLISION_TYPES)
    radius_m = _optional_positive_float(raw_value.get("radius_m"), "tool.collision.radius_m")
    position = np.asarray(raw_value.get("position", (0.0, 0.0, 0.0)), dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("tool.collision.position must be a finite 3-vector.")
    rgba = _rgba_tuple(raw_value.get("rgba", (0.2, 0.22, 0.24, 1.0)), "tool.collision.rgba")
    friction_values = np.asarray(
        raw_value.get("friction", (0.8, 0.02, 0.005)), dtype=float
    )
    if (
        friction_values.shape != (3,)
        or not np.all(np.isfinite(friction_values))
        or np.any(friction_values < 0.0)
    ):
        raise ValueError("tool.collision.friction must be a nonnegative 3-vector.")
    return CollisionGeometryConfig(
        type=collision_type,
        radius_m=radius_m,
        position=position,
        rgba=rgba,
        friction=tuple(float(value) for value in friction_values),
        contype=_nonnegative_int(raw_value.get("contype", 1), "tool.collision.contype"),
        conaffinity=_nonnegative_int(
            raw_value.get("conaffinity", 1), "tool.collision.conaffinity"
        ),
    )


def _load_contact(raw_value: object) -> ContactToolConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.contact must be a mapping.")
    return ContactToolConfig(
        target_normal_force_n=_positive_float_value(
            _required(raw_value, "target_normal_force_n"),
            "tool.contact.target_normal_force_n",
        ),
        max_normal_force_n=_positive_float_value(
            _required(raw_value, "max_normal_force_n"),
            "tool.contact.max_normal_force_n",
        ),
        standoff_distance_m=_optional_positive_float(
            raw_value.get("standoff_distance_m"),
            "tool.contact.standoff_distance_m",
        ),
    )


def _load_camera(raw_value: object) -> CameraConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.camera must be a mapping.")
    intrinsics_raw = _section(raw_value, "intrinsics")
    return CameraConfig(
        name=str(_required(raw_value, "name")),
        intrinsics=CameraIntrinsicsConfig(
            width=int(_required(intrinsics_raw, "width")),
            height=int(_required(intrinsics_raw, "height")),
            fovy_deg=float(_required(intrinsics_raw, "fovy_deg")),
            near=float(_required(intrinsics_raw, "near")),
            far=float(_required(intrinsics_raw, "far")),
        ),
        tip_to_camera=_pose_from_section(raw_value, "tip_to_camera", "tool.camera.tip_to_camera"),
    )


def _load_camera_visual(raw_value: object) -> CameraVisualConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.camera must be a mapping.")
    visual = raw_value.get("visual")
    if visual is None:
        return None
    if not isinstance(visual, dict):
        raise ValueError("tool.camera.visual must be a mapping.")
    shape = str(visual.get("shape", "hemisphere"))
    if shape != "hemisphere":
        raise ValueError("tool.camera.visual.shape must be 'hemisphere'.")
    return CameraVisualConfig(
        shape=shape,
        radius_m=_positive_float_value(
            visual.get("radius_m", 0.00375),
            "tool.camera.visual.radius_m",
        ),
        rgba=_rgba_tuple(
            visual.get("rgba", (0.08, 0.09, 0.11, 1.0)),
            "tool.camera.visual.rgba",
        ),
        lens_radius_m=_positive_float_value(
            visual.get("lens_radius_m", 0.0015),
            "tool.camera.visual.lens_radius_m",
        ),
        lens_depth_m=_positive_float_value(
            visual.get("lens_depth_m", 0.0008),
            "tool.camera.visual.lens_depth_m",
        ),
        lens_rgba=_rgba_tuple(
            visual.get("lens_rgba", (0.08, 0.25, 0.50, 1.0)),
            "tool.camera.visual.lens_rgba",
        ),
    )


def _load_airgun(raw_value: object) -> AirgunConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.airgun must be a mapping.")
    return AirgunConfig(
        standoff_distance_m=_positive_float_value(
            _required(raw_value, "standoff_distance_m"),
            "tool.airgun.standoff_distance_m",
        )
    )


def _load_force_torque_sensor(raw_value: object) -> ForceTorqueSensorConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.force_torque_sensor must be a mapping.")
    return ForceTorqueSensorConfig(
        size_m=np.asarray(
            _required(raw_value, "size_m"),
            dtype=float,
        ),
        mass_kg=_positive_float_value(
            _required(raw_value, "mass_kg"),
            "tool.force_torque_sensor.mass_kg",
        ),
        rgba=_rgba_tuple(
            raw_value.get("rgba", (0.16, 0.17, 0.18, 1.0)),
            "tool.force_torque_sensor.rgba",
        ),
        force_limit_n=_positive_float_value(
            raw_value.get("force_limit_n", 10.0),
            "tool.force_torque_sensor.force_limit_n",
        ),
        torque_limit_nm=_positive_float_value(
            raw_value.get("torque_limit_nm", 0.25),
            "tool.force_torque_sensor.torque_limit_nm",
        ),
        filter_cutoff_hz=_positive_float_value(
            raw_value.get("filter_cutoff_hz", 15.0),
            "tool.force_torque_sensor.filter_cutoff_hz",
        ),
        tare_on_reset=_bool_field(raw_value, "tare_on_reset", default=True),
        gravity_compensation=_bool_field(
            raw_value, "gravity_compensation", default=True
        ),
        output_sign=_wrench_output_sign(raw_value.get("output_sign", -1.0)),
    )


def _load_optional_pose(values: dict[str, object], key: str, name: str) -> Pose6D | None:
    if key not in values:
        return None
    return _pose_from_section(values, key, name)


def _pose_from_section(values: dict[str, object], key: str, name: str) -> Pose6D:
    pose_raw = _section(values, key)
    return Pose6D.from_dict(
        {
            "position": _position_vector(_required(pose_raw, "position"), f"{name}.position"),
            "quat": _quat_field(pose_raw, name),
        }
    )


def _quat_field(values: dict[str, object], name: str) -> np.ndarray:
    if "quat" in values:
        return np.asarray(values["quat"], dtype=float)
    if "quat_wxyz" in values:
        return np.asarray(values["quat_wxyz"], dtype=float)
    raise ValueError(f"Missing required config field '{name}.quat'.")


def _optional_resolved_path(config_path: Path, raw_value: object) -> Path | None:
    if raw_value is None:
        return None
    return _resolve_path(config_path, raw_value)


def _optional_positive_float(raw_value: object, name: str) -> float | None:
    if raw_value is None:
        return None
    return _positive_float_value(raw_value, name)


def _rgba_tuple(raw_value: object, name: str) -> tuple[float, float, float, float]:
    values = np.asarray(raw_value, dtype=float)
    if (
        values.shape != (4,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"{name} must contain four values in [0, 1].")
    return tuple(float(value) for value in values)


def _nonnegative_int(raw_value: object, name: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _wrench_output_sign(raw_value: object) -> float:
    value = float(raw_value)
    if value not in (-1.0, 1.0):
        raise ValueError("tool.force_torque_sensor.output_sign must be -1 or 1.")
    return value
