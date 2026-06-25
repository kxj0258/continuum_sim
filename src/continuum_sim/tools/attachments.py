"""Tool/camera attachment configuration scaffold.

This module only loads attachment metadata and fixed coordinate transforms.
It does not implement a MuJoCo renderer, visual recognition, contact control,
or collision avoidance. Later M5/M6/M7/M8 work can build behavior on top of
these data structures.
"""

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
    """Simple collision geometry placeholder for an attachment."""

    type: str
    radius_m: float | None = None


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
    nozzle_pose: Pose6D | None = None
    airgun: AirgunConfig | None = None


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
        nozzle_pose=_load_optional_pose(tool_raw, "nozzle_pose", "tool.nozzle_pose"),
        airgun=_load_airgun(tool_raw.get("airgun")),
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


def _load_collision(raw_value: object) -> CollisionGeometryConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("tool.collision must be a mapping.")
    collision_type = _choice_value(_required(raw_value, "type"), "tool.collision.type", COLLISION_TYPES)
    radius_m = _optional_positive_float(raw_value.get("radius_m"), "tool.collision.radius_m")
    return CollisionGeometryConfig(type=collision_type, radius_m=radius_m)


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
