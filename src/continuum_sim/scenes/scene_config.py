"""YAML-backed structured scene descriptions for MuJoCo navigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    choice_value as _choice_value,
    geom_group as _geom_group,
    nonnegative_float_value as _nonnegative_float_value,
    nonnegative_int_value as _nonnegative_int_value,
    optional_section as _optional_section,
    position_vector as _position_vector,
    positive_float_value as _positive_float_value,
    positive_int_value as _positive_int_value,
    required as _required,
    resolve_path as _resolve_config_path,
    rgba_tuple as _rgba_tuple,
    section as _section,
)
from continuum_sim.scenes.contact_surfaces import (
    WipePatchConfig,
    WorkSurfaceConfig,
    make_work_surface,
)
from continuum_sim.scenes.primitives import (
    BoxObstaclePrimitive,
    ClearancePrimitive,
    CylinderObstaclePrimitive,
    InteriorShellPrimitive,
)


SCENE_PRIMITIVE_TYPES = (
    "cylindrical_shell_segment",
    "frustum_shell_segment",
    "cylinder_obstacle",
    "box_obstacle",
    "box_surface",
)
TARGET_TYPES = ("point", "wall_point")
_MISSING = object()


@dataclass(frozen=True)
class SceneBuilderConfig:
    """MuJoCo XML generation settings for a structured scene."""

    shell_approx_sides: int
    shell_axial_slices: int
    wall_thickness_m: float
    geom_group: int
    shell_rgba: tuple[float, float, float, float]
    obstacle_rgba: tuple[float, float, float, float]
    target_rgba: tuple[float, float, float, float]
    target_radius_m: float
    contype: int
    conaffinity: int


@dataclass(frozen=True)
class ScenePrimitiveConfig:
    """Raw scene primitive with validated numeric fields."""

    id: str
    type: str
    z_min_m: float | None = None
    z_max_m: float | None = None
    radius_m: float | None = None
    radius_start_m: float | None = None
    radius_end_m: float | None = None
    center_m: np.ndarray | None = None
    half_length_m: float | None = None
    axis: str = "z"
    half_size_m: np.ndarray | None = None
    rgba: tuple[float, float, float, float] | None = None

    def to_clearance_primitive(self) -> ClearancePrimitive | None:
        if self.type == "cylindrical_shell_segment":
            radius = _required_optional_float(self.radius_m, f"{self.id}.radius_m")
            return InteriorShellPrimitive(
                id=self.id,
                z_min_m=_required_optional_float(self.z_min_m, f"{self.id}.z_min_m"),
                z_max_m=_required_optional_float(self.z_max_m, f"{self.id}.z_max_m"),
                radius_start_m=radius,
                radius_end_m=radius,
            )
        if self.type == "frustum_shell_segment":
            return InteriorShellPrimitive(
                id=self.id,
                z_min_m=_required_optional_float(self.z_min_m, f"{self.id}.z_min_m"),
                z_max_m=_required_optional_float(self.z_max_m, f"{self.id}.z_max_m"),
                radius_start_m=_required_optional_float(
                    self.radius_start_m,
                    f"{self.id}.radius_start_m",
                ),
                radius_end_m=_required_optional_float(
                    self.radius_end_m,
                    f"{self.id}.radius_end_m",
                ),
            )
        if self.type == "cylinder_obstacle":
            center = _required_optional_array(self.center_m, f"{self.id}.center_m")
            return CylinderObstaclePrimitive(
                id=self.id,
                center_m=tuple(float(value) for value in center),
                radius_m=_required_optional_float(self.radius_m, f"{self.id}.radius_m"),
                half_length_m=_required_optional_float(
                    self.half_length_m,
                    f"{self.id}.half_length_m",
                ),
                axis=self.axis,
            )
        if self.type == "box_obstacle":
            center = _required_optional_array(self.center_m, f"{self.id}.center_m")
            half_size = _required_optional_array(self.half_size_m, f"{self.id}.half_size_m")
            return BoxObstaclePrimitive(
                id=self.id,
                center_m=tuple(float(value) for value in center),
                half_size_m=tuple(float(value) for value in half_size),
            )
        if self.type == "box_surface":
            return None
        raise ValueError(f"Unsupported scene primitive type {self.type!r}.")


@dataclass(frozen=True)
class InspectionTargetConfig:
    """Named target point for ordered inspection missions."""

    id: str
    type: str
    pos_m: np.ndarray
    section_id: str | None = None
    theta_deg: float | None = None
    z_m: float | None = None
    inward_offset_m: float | None = None


@dataclass(frozen=True)
class NavigationSceneConfig:
    """Complete structured scene description."""

    path: Path
    name: str
    description: str
    builder: SceneBuilderConfig
    primitives: tuple[ScenePrimitiveConfig, ...]
    inspection_targets: tuple[InspectionTargetConfig, ...]
    work_surfaces: tuple[WorkSurfaceConfig, ...] = ()
    wipe_patches: tuple[WipePatchConfig, ...] = ()

    @property
    def clearance_primitives(self) -> tuple[ClearancePrimitive, ...]:
        return tuple(
            primitive
            for item in self.primitives
            if (primitive := item.to_clearance_primitive()) is not None
        )

    def target_positions(self, target_ids: tuple[str, ...]) -> np.ndarray:
        lookup = {target.id: target.pos_m for target in self.inspection_targets}
        missing = [target_id for target_id in target_ids if target_id not in lookup]
        if missing:
            raise ValueError(f"Unknown inspection target id(s): {missing}.")
        return np.asarray([lookup[target_id] for target_id in target_ids], dtype=float)

    def work_surface(self, surface_id: str) -> WorkSurfaceConfig:
        lookup = {surface.id: surface for surface in self.work_surfaces}
        if surface_id not in lookup:
            raise ValueError(f"Unknown work surface id {surface_id!r}.")
        return lookup[surface_id]

    def wipe_patch(self, patch_id: str) -> WipePatchConfig:
        lookup = {patch.id: patch for patch in self.wipe_patches}
        if patch_id not in lookup:
            raise ValueError(f"Unknown wipe patch id {patch_id!r}.")
        return lookup[patch_id]


def load_navigation_scene_config(path: str | Path) -> NavigationSceneConfig:
    """Load and validate a structured navigation scene YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    scene = _section(raw, "scene")
    builder = _optional_section(raw, "builder")
    primitives_raw = _required(scene, "primitives")
    targets_raw = scene.get("inspection_targets", [])
    surfaces_raw = scene.get("work_surfaces", [])
    patches_raw = scene.get("wipe_patches", [])
    if not isinstance(primitives_raw, list | tuple):
        raise ValueError("scene.primitives must be a list.")
    if not isinstance(targets_raw, list | tuple):
        raise ValueError("scene.inspection_targets must be a list.")
    if not isinstance(surfaces_raw, list | tuple):
        raise ValueError("scene.work_surfaces must be a list.")
    if not isinstance(patches_raw, list | tuple):
        raise ValueError("scene.wipe_patches must be a list.")

    primitive_configs = tuple(_load_primitive_config(item) for item in primitives_raw)
    _validate_unique_ids([primitive.id for primitive in primitive_configs], "scene.primitives")
    target_configs = tuple(
        _load_target_config(item, primitive_configs)
        for item in targets_raw
    )
    _validate_unique_ids(
        [target.id for target in target_configs],
        "scene.inspection_targets",
    )
    surface_configs = tuple(
        _load_work_surface_config(item, primitive_configs)
        for item in surfaces_raw
    )
    _validate_unique_ids([surface.id for surface in surface_configs], "scene.work_surfaces")
    patch_configs = tuple(
        _load_wipe_patch_config(item, surface_configs)
        for item in patches_raw
    )
    _validate_unique_ids([patch.id for patch in patch_configs], "scene.wipe_patches")

    return NavigationSceneConfig(
        path=config_path,
        name=str(_required(raw, "name")),
        description=str(raw.get("description", "")),
        builder=_load_builder_config(builder),
        primitives=primitive_configs,
        inspection_targets=target_configs,
        work_surfaces=surface_configs,
        wipe_patches=patch_configs,
    )


def _load_builder_config(values: dict[str, Any]) -> SceneBuilderConfig:
    return SceneBuilderConfig(
        shell_approx_sides=_positive_int_value(
            values.get("shell_approx_sides", 32),
            "builder.shell_approx_sides",
        ),
        shell_axial_slices=_positive_int_value(
            values.get("shell_axial_slices", 6),
            "builder.shell_axial_slices",
        ),
        wall_thickness_m=_positive_float_value(
            values.get("wall_thickness_m", 0.004),
            "builder.wall_thickness_m",
        ),
        geom_group=_geom_group(values.get("geom_group", 0), "builder.geom_group"),
        shell_rgba=_rgba_tuple(
            values.get("shell_rgba", (0.34, 0.42, 0.46, 0.32)),
            "builder.shell_rgba",
        ),
        obstacle_rgba=_rgba_tuple(
            values.get("obstacle_rgba", (0.93, 0.49, 0.16, 1.0)),
            "builder.obstacle_rgba",
        ),
        target_rgba=_rgba_tuple(
            values.get("target_rgba", (0.1, 0.85, 0.5, 1.0)),
            "builder.target_rgba",
        ),
        target_radius_m=_positive_float_value(
            values.get("target_radius_m", 0.003),
            "builder.target_radius_m",
        ),
        contype=_nonnegative_int_value(values.get("contype", 1), "builder.contype"),
        conaffinity=_nonnegative_int_value(
            values.get("conaffinity", 1),
            "builder.conaffinity",
        ),
    )


def _load_primitive_config(values: object) -> ScenePrimitiveConfig:
    if not isinstance(values, dict):
        raise ValueError("Each scene primitive must be a mapping.")
    primitive_type = _choice_value(_required(values, "type"), "primitive.type", SCENE_PRIMITIVE_TYPES)
    rgba = (
        None
        if values.get("rgba", _MISSING) is _MISSING
        else _rgba_tuple(values["rgba"], "primitive.rgba")
    )
    primitive = ScenePrimitiveConfig(
        id=str(_required(values, "id")),
        type=primitive_type,
        z_min_m=_optional_float(values.get("z_min_m", _MISSING), "primitive.z_min_m"),
        z_max_m=_optional_float(values.get("z_max_m", _MISSING), "primitive.z_max_m"),
        radius_m=_optional_positive_float(
            values.get("radius_m", _MISSING),
            "primitive.radius_m",
        ),
        radius_start_m=_optional_positive_float(
            values.get("radius_start_m", _MISSING),
            "primitive.radius_start_m",
        ),
        radius_end_m=_optional_positive_float(
            values.get("radius_end_m", _MISSING),
            "primitive.radius_end_m",
        ),
        center_m=_optional_position(values.get("center_m", _MISSING), "primitive.center_m"),
        half_length_m=_optional_positive_float(
            values.get("half_length_m", _MISSING),
            "primitive.half_length_m",
        ),
        axis=_choice_value(values.get("axis", "z"), "primitive.axis", ("x", "y", "z")),
        half_size_m=_optional_position(
            values.get("half_size_m", _MISSING),
            "primitive.half_size_m",
        ),
        rgba=rgba,
    )
    _validate_primitive(primitive)
    return primitive


def _load_target_config(
    values: object,
    primitives: tuple[ScenePrimitiveConfig, ...],
) -> InspectionTargetConfig:
    if not isinstance(values, dict):
        raise ValueError("Each scene.inspection_targets item must be a mapping.")
    target_type = _choice_value(_required(values, "type"), "target.type", TARGET_TYPES)
    target_id = str(_required(values, "id"))
    if target_type == "point":
        pos = _position_vector(_required(values, "pos_m"), f"{target_id}.pos_m")
        return InspectionTargetConfig(id=target_id, type=target_type, pos_m=pos)
    section_id = str(_required(values, "section_id"))
    theta_deg = float(_required(values, "theta_deg"))
    z_m = float(_required(values, "z_m"))
    inward_offset_m = _nonnegative_float_value(
        values.get("inward_offset_m", 0.0),
        f"{target_id}.inward_offset_m",
    )
    pos = _wall_point(section_id, theta_deg, z_m, inward_offset_m, primitives)
    return InspectionTargetConfig(
        id=target_id,
        type=target_type,
        pos_m=pos,
        section_id=section_id,
        theta_deg=theta_deg,
        z_m=z_m,
        inward_offset_m=inward_offset_m,
    )


def _wall_point(
    section_id: str,
    theta_deg: float,
    z_m: float,
    inward_offset_m: float,
    primitives: tuple[ScenePrimitiveConfig, ...],
) -> np.ndarray:
    lookup = {primitive.id: primitive for primitive in primitives}
    if section_id not in lookup:
        raise ValueError(f"Unknown wall-point section_id {section_id!r}.")
    section = lookup[section_id]
    if section.type not in ("cylindrical_shell_segment", "frustum_shell_segment"):
        raise ValueError(f"section_id {section_id!r} does not reference a shell primitive.")
    shell = section.to_clearance_primitive()
    if not isinstance(shell, InteriorShellPrimitive):
        raise TypeError(f"{section_id!r} did not produce an interior shell primitive.")
    if z_m < shell.z_min_m or z_m > shell.z_max_m:
        raise ValueError(
            f"wall target z_m={z_m} is outside section {section_id!r} "
            f"[{shell.z_min_m}, {shell.z_max_m}]."
        )
    radius = shell.radius_at(z_m) - inward_offset_m
    if radius <= 0.0:
        raise ValueError(f"wall target {section_id!r} inward_offset_m leaves no radius.")
    theta = np.deg2rad(theta_deg)
    return np.array([radius * np.cos(theta), radius * np.sin(theta), z_m], dtype=float)


def _load_work_surface_config(
    values: object,
    primitives: tuple[ScenePrimitiveConfig, ...],
) -> WorkSurfaceConfig:
    if not isinstance(values, dict):
        raise ValueError("Each scene.work_surfaces item must be a mapping.")
    surface_id = str(_required(values, "id"))
    primitive_id = str(_required(values, "primitive_id"))
    primitive_lookup = {primitive.id: primitive for primitive in primitives}
    if primitive_id not in primitive_lookup:
        raise ValueError(f"Unknown work surface primitive_id {primitive_id!r}.")
    return make_work_surface(
        id=surface_id,
        primitive_id=primitive_id,
        center_m=_position_vector(_required(values, "center_m"), f"{surface_id}.center_m"),
        normal=_position_vector(_required(values, "normal"), f"{surface_id}.normal"),
        tangent_u=_position_vector(
            _required(values, "tangent_u"),
            f"{surface_id}.tangent_u",
        ),
        width_m=_positive_float_value(_required(values, "width_m"), f"{surface_id}.width_m"),
        height_m=_positive_float_value(
            _required(values, "height_m"),
            f"{surface_id}.height_m",
        ),
    )


def _load_wipe_patch_config(
    values: object,
    surfaces: tuple[WorkSurfaceConfig, ...],
) -> WipePatchConfig:
    if not isinstance(values, dict):
        raise ValueError("Each scene.wipe_patches item must be a mapping.")
    patch_id = str(_required(values, "id"))
    surface_id = str(_required(values, "surface_id"))
    if surface_id not in {surface.id for surface in surfaces}:
        raise ValueError(f"Unknown wipe patch surface_id {surface_id!r}.")
    return WipePatchConfig(
        id=patch_id,
        surface_id=surface_id,
        center_m=_position_vector(_required(values, "center_m"), f"{patch_id}.center_m"),
        width_m=_positive_float_value(_required(values, "width_m"), f"{patch_id}.width_m"),
        height_m=_positive_float_value(
            _required(values, "height_m"),
            f"{patch_id}.height_m",
        ),
    )


def _validate_primitive(primitive: ScenePrimitiveConfig) -> None:
    if not primitive.id:
        raise ValueError("primitive.id must be non-empty.")
    if primitive.type == "cylindrical_shell_segment":
        _required_optional_float(primitive.z_min_m, f"{primitive.id}.z_min_m")
        _required_optional_float(primitive.z_max_m, f"{primitive.id}.z_max_m")
        _required_optional_float(primitive.radius_m, f"{primitive.id}.radius_m")
    elif primitive.type == "frustum_shell_segment":
        _required_optional_float(primitive.z_min_m, f"{primitive.id}.z_min_m")
        _required_optional_float(primitive.z_max_m, f"{primitive.id}.z_max_m")
        _required_optional_float(primitive.radius_start_m, f"{primitive.id}.radius_start_m")
        _required_optional_float(primitive.radius_end_m, f"{primitive.id}.radius_end_m")
    elif primitive.type == "cylinder_obstacle":
        _required_optional_array(primitive.center_m, f"{primitive.id}.center_m")
        _required_optional_float(primitive.radius_m, f"{primitive.id}.radius_m")
        _required_optional_float(primitive.half_length_m, f"{primitive.id}.half_length_m")
    elif primitive.type in ("box_obstacle", "box_surface"):
        _required_optional_array(primitive.center_m, f"{primitive.id}.center_m")
        _required_optional_array(primitive.half_size_m, f"{primitive.id}.half_size_m")
        if np.any(primitive.half_size_m <= 0.0):
            raise ValueError(f"{primitive.id}.half_size_m values must be positive.")
    if primitive.z_min_m is not None and primitive.z_max_m is not None:
        if primitive.z_min_m >= primitive.z_max_m:
            raise ValueError(f"{primitive.id}.z_min_m must be less than z_max_m.")


def _validate_unique_ids(ids: list[str], name: str) -> None:
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"{name} contains duplicate id(s): {duplicates}.")


def _optional_positive_float(raw_value: object, name: str) -> float | None:
    if raw_value is _MISSING:
        return None
    return _positive_float_value(raw_value, name)


def _optional_float(raw_value: object, name: str) -> float | None:
    if raw_value is _MISSING:
        return None
    del name
    return float(raw_value)


def _optional_position(raw_value: object, name: str) -> np.ndarray | None:
    if raw_value is _MISSING:
        return None
    return _position_vector(raw_value, name)


def _required_optional_float(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required config field {name!r}.")
    return float(value)


def _required_optional_array(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required config field {name!r}.")
    return value
