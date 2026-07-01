"""Scenario-native engine cleaning path generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.scenes.engine_scene import EngineRegionConfig, EngineSceneConfig
from continuum_sim.scenes.engine_surfaces import EngineSurfacePatchConfig
from continuum_sim.tasks.engine_surface_path import (
    EngineSurfacePathConfig,
    build_raster_cleaning_path,
    path_normals_array,
    path_positions_array,
)


@dataclass(frozen=True)
class EngineCleaningPathSpec:
    """Raster cleaning path on an engine surface_patch region."""

    region_name: str
    num_passes_u: int
    num_passes_v: int
    approach_distance_m: float
    retreat_distance_m: float
    target_force_n: float
    standoff_distance_m: float
    snake_pattern: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "EngineCleaningPathSpec":
        return cls(
            region_name=str(values.get("region_name", values.get("target_region", ""))),
            num_passes_u=int(values.get("num_passes_u", 5)),
            num_passes_v=int(values.get("num_passes_v", 5)),
            approach_distance_m=float(values.get("approach_distance_m", 0.02)),
            retreat_distance_m=float(values.get("retreat_distance_m", 0.02)),
            target_force_n=float(values.get("target_force_n", 1.0)),
            standoff_distance_m=float(values.get("standoff_distance_m", 0.0)),
            snake_pattern=bool(values.get("snake_pattern", True)),
        )


@dataclass(frozen=True)
class EngineCleaningPlan:
    waypoints_world: np.ndarray
    normals_world: np.ndarray
    phases: tuple[str, ...]
    target_force_n: np.ndarray
    standoff_distance_m: np.ndarray


def build_engine_cleaning_plan(
    spec: EngineCleaningPathSpec,
    scene: EngineSceneConfig,
) -> EngineCleaningPlan:
    """Generate scenario waypoints from an engine surface patch region."""

    if not spec.region_name:
        raise ValueError("engine_cleaning.region_name is required.")
    if spec.region_name not in scene.regions:
        raise ValueError(f"Unknown engine region {spec.region_name!r}.")
    patch = _surface_patch_from_region(scene.regions[spec.region_name])
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=spec.num_passes_u,
        num_passes_v=spec.num_passes_v,
        approach_distance_m=spec.approach_distance_m,
        retreat_distance_m=spec.retreat_distance_m,
        target_force_n=spec.target_force_n,
        standoff_distance_m=spec.standoff_distance_m,
        snake_pattern=spec.snake_pattern,
    )
    waypoints = build_raster_cleaning_path(patch, path_config)
    return EngineCleaningPlan(
        waypoints_world=path_positions_array(waypoints),
        normals_world=path_normals_array(waypoints),
        phases=tuple(waypoint.phase for waypoint in waypoints),
        target_force_n=np.asarray([waypoint.target_force_n for waypoint in waypoints], dtype=float),
        standoff_distance_m=np.asarray(
            [waypoint.standoff_distance_m for waypoint in waypoints],
            dtype=float,
        ),
    )


def _surface_patch_from_region(region: EngineRegionConfig) -> EngineSurfacePatchConfig:
    if region.type != "surface_patch":
        raise ValueError(f"Engine region {region.name!r} must be a surface_patch.")
    if region.position_m is None or region.normal is None or region.extents_m is None:
        raise ValueError(f"Engine surface_patch region {region.name!r} is incomplete.")
    tangent = _default_tangent(region.normal)
    return EngineSurfacePatchConfig(
        name=region.name,
        type="plane_patch",
        center=region.position_m.copy(),
        normal=region.normal.copy(),
        tangent_u=tangent,
        size_u_m=float(region.extents_m[0]),
        size_v_m=float(region.extents_m[1]),
    )


def _default_tangent(normal: np.ndarray) -> np.ndarray:
    unit = np.asarray(normal, dtype=float)
    unit = unit / np.linalg.norm(unit)
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, unit))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = candidate - np.dot(candidate, unit) * unit
    return tangent / np.linalg.norm(tangent)
