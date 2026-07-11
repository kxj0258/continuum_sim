"""Scenario-native wiping path generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.scenes.scene_config import NavigationSceneConfig


@dataclass(frozen=True)
class WipingPathSpec:
    """Raster wiping path on a named structured-scene work surface."""

    surface_id: str
    patch_id: str | None
    center_m: np.ndarray | None = None
    width_m: float | None = None
    height_m: float | None = None
    line_count: int = 5
    samples_per_line: int = 30
    approach_offset_m: float = 0.005
    contact_offset_m: float = 0.0

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "WipingPathSpec":
        return cls(
            surface_id=str(_required(values, "surface_id", "wiping_path")),
            patch_id=None if values.get("patch_id") is None else str(values["patch_id"]),
            center_m=_optional_vector3(values.get("center_m"), "wiping_path.center_m"),
            width_m=_optional_float(values.get("width_m")),
            height_m=_optional_float(values.get("height_m")),
            line_count=int(values.get("line_count", 5)),
            samples_per_line=int(values.get("samples_per_line", 30)),
            approach_offset_m=float(values.get("approach_offset_m", 0.005)),
            contact_offset_m=float(values.get("contact_offset_m", 0.0)),
        )


@dataclass(frozen=True)
class WipingPathPlan:
    waypoints_world: np.ndarray
    target_pose: np.ndarray
    phases: tuple[str, ...]
    waypoint_indices: np.ndarray
    surface_normal_world: np.ndarray
    surface_point_world: np.ndarray


def build_wiping_plan(
    spec: WipingPathSpec,
    scene: NavigationSceneConfig,
) -> WipingPathPlan:
    """Build approach plus boustrophedon contact waypoints for scenario wiping."""

    surface = scene.work_surface(spec.surface_id)
    patch = None if spec.patch_id is None else scene.wipe_patch(spec.patch_id)
    center = surface.center_m if spec.center_m is None else spec.center_m
    width = patch.width_m if patch is not None and spec.width_m is None else spec.width_m
    height = patch.height_m if patch is not None and spec.height_m is None else spec.height_m
    if patch is not None and spec.center_m is None:
        center = patch.center_m
    if width is None or height is None:
        raise ValueError("wiping_path width_m and height_m are required without a patch.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("wiping_path width_m and height_m must be positive.")
    if spec.line_count <= 0 or spec.samples_per_line <= 0:
        raise ValueError("wiping_path line_count and samples_per_line must be positive.")

    contact_origin = center + spec.contact_offset_m * surface.normal
    positions = [contact_origin + spec.approach_offset_m * surface.normal]
    phases = ["approach"]
    waypoint_indices = [0]
    next_index = 1
    v_offsets = (
        np.array([0.0], dtype=float)
        if spec.line_count == 1
        else np.linspace(-0.5 * height, 0.5 * height, spec.line_count)
    )
    for row_index, v_offset in enumerate(v_offsets):
        u_values = np.linspace(-0.5 * width, 0.5 * width, spec.samples_per_line)
        if row_index % 2 == 1:
            u_values = u_values[::-1]
        for u_offset in u_values:
            positions.append(
                contact_origin
                + float(u_offset) * surface.tangent_u
                + float(v_offset) * surface.tangent_v
            )
            phases.append("contact")
            waypoint_indices.append(next_index)
            next_index += 1
    waypoints = np.asarray(positions, dtype=float)
    return WipingPathPlan(
        waypoints_world=waypoints,
        target_pose=np.asarray([surface.target_pose(point) for point in waypoints], dtype=float),
        phases=tuple(phases),
        waypoint_indices=np.asarray(waypoint_indices, dtype=int),
        surface_normal_world=surface.normal.copy(),
        surface_point_world=surface.center_m.copy(),
    )


def _required(values: dict[str, object], name: str, section: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required field {section}.{name}.")
    return values[name]


def _optional_vector3(value: object, name: str) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    return result.copy()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
