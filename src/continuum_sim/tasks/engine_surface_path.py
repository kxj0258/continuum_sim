"""Engine surface raster-path scaffold.

This module builds geometric cleaning waypoints from local surface patches.
It does not implement CAD mesh planning, contact control, or MuJoCo runtime
integration. Later M6 work can consume these waypoints in a controller layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_sim.scenes.engine_surfaces import (
    EngineSurfacePatchConfig,
    sample_surface_grid,
    surface_frame_from_patch,
)


@dataclass(frozen=True)
class CleaningWaypoint:
    """One approach/contact/retreat waypoint along a local cleaning path."""

    position: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    phase: str
    target_force_n: float
    standoff_distance_m: float
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineSurfacePathConfig:
    """Raster path parameters for one surface patch."""

    patch_name: str
    num_passes_u: int
    num_passes_v: int
    approach_distance_m: float
    retreat_distance_m: float
    target_force_n: float
    standoff_distance_m: float
    snake_pattern: bool = True

    def __post_init__(self) -> None:
        if self.num_passes_u < 1:
            raise ValueError(f"num_passes_u must be >= 1, got {self.num_passes_u}.")
        if self.num_passes_v < 1:
            raise ValueError(f"num_passes_v must be >= 1, got {self.num_passes_v}.")
        if self.approach_distance_m < 0.0:
            raise ValueError(f"approach_distance_m must be >= 0, got {self.approach_distance_m}.")
        if self.retreat_distance_m < 0.0:
            raise ValueError(f"retreat_distance_m must be >= 0, got {self.retreat_distance_m}.")
        if self.target_force_n <= 0.0:
            raise ValueError(f"target_force_n must be > 0, got {self.target_force_n}.")
        if self.standoff_distance_m < 0.0:
            raise ValueError(f"standoff_distance_m must be >= 0, got {self.standoff_distance_m}.")


def build_raster_cleaning_path(
    patch: EngineSurfacePatchConfig,
    path_config: EngineSurfacePathConfig,
) -> list[CleaningWaypoint]:
    """Build approach/contact/retreat waypoints using a raster grid."""

    frame = surface_frame_from_patch(patch)
    grid = sample_surface_grid(patch, num_u=path_config.num_passes_u, num_v=path_config.num_passes_v)
    contact_waypoints: list[CleaningWaypoint] = []
    next_index = 1
    for row_index in range(path_config.num_passes_v):
        row_points = grid[row_index]
        if path_config.snake_pattern and row_index % 2 == 1:
            row_points = row_points[::-1]
        for col_index, point in enumerate(row_points):
            if patch.type == "sphere_patch" and patch.sphere_center is not None:
                normal = _normalize(point - patch.sphere_center, "sphere_normal")
                tangent_v = _normalize(np.cross(normal, frame.tangent_u), "tangent_v")
                tangent_u = _normalize(np.cross(tangent_v, normal), "tangent_u")
            else:
                normal = frame.normal
                tangent_u = frame.tangent_u
                tangent_v = frame.tangent_v
            contact_waypoints.append(
                CleaningWaypoint(
                    position=np.asarray(point, dtype=float),
                    normal=normal,
                    tangent_u=tangent_u,
                    tangent_v=tangent_v,
                    phase="contact",
                    target_force_n=path_config.target_force_n,
                    standoff_distance_m=path_config.standoff_distance_m,
                    index=next_index,
                    metadata={"row": row_index, "col": col_index},
                )
            )
            next_index += 1

    if not contact_waypoints:
        raise ValueError("Raster cleaning path requires at least one contact waypoint.")

    approach = build_approach_waypoint(contact_waypoints[0], path_config.approach_distance_m)
    retreat = build_retreat_waypoint(contact_waypoints[-1], path_config.retreat_distance_m)
    return [approach, *contact_waypoints, retreat]


def build_approach_waypoint(first_contact: CleaningWaypoint, approach_distance_m: float) -> CleaningWaypoint:
    """Offset the first contact waypoint along its outward normal."""

    return CleaningWaypoint(
        position=first_contact.position + first_contact.normal * approach_distance_m,
        normal=first_contact.normal.copy(),
        tangent_u=first_contact.tangent_u.copy(),
        tangent_v=first_contact.tangent_v.copy(),
        phase="approach",
        target_force_n=0.0,
        standoff_distance_m=approach_distance_m,
        index=0,
        metadata={"source_index": first_contact.index},
    )


def build_retreat_waypoint(last_contact: CleaningWaypoint, retreat_distance_m: float) -> CleaningWaypoint:
    """Offset the last contact waypoint along its outward normal."""

    return CleaningWaypoint(
        position=last_contact.position + last_contact.normal * retreat_distance_m,
        normal=last_contact.normal.copy(),
        tangent_u=last_contact.tangent_u.copy(),
        tangent_v=last_contact.tangent_v.copy(),
        phase="retreat",
        target_force_n=0.0,
        standoff_distance_m=retreat_distance_m,
        index=last_contact.index + 1,
        metadata={"source_index": last_contact.index},
    )


def split_waypoints_by_phase(waypoints: list[CleaningWaypoint]) -> dict[str, list[CleaningWaypoint]]:
    """Group waypoints by phase while preserving order."""

    grouped = {"approach": [], "contact": [], "retreat": []}
    for waypoint in waypoints:
        grouped.setdefault(waypoint.phase, []).append(waypoint)
    return grouped


def path_positions_array(waypoints: list[CleaningWaypoint]) -> np.ndarray:
    """Return an `N x 3` position array."""

    return np.asarray([waypoint.position for waypoint in waypoints], dtype=float)


def path_normals_array(waypoints: list[CleaningWaypoint]) -> np.ndarray:
    """Return an `N x 3` normal array."""

    return np.asarray([waypoint.normal for waypoint in waypoints], dtype=float)


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return array / norm
