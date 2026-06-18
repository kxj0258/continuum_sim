"""Planar contact surfaces used by wiping tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceDistanceQuery:
    """Signed distance from a point to a planar work surface."""

    signed_distance_m: float
    normal: np.ndarray
    projected_point: np.ndarray
    local_uv_m: np.ndarray
    surface_id: str


@dataclass(frozen=True)
class WorkSurfaceConfig:
    """Planar work-surface frame and rectangular extent metadata."""

    id: str
    primitive_id: str
    center_m: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    width_m: float
    height_m: float

    def signed_distance(self, point: np.ndarray) -> float:
        point_array = _position(point, "point")
        return float(np.dot(point_array - self.center_m, self.normal))

    def project(self, point: np.ndarray) -> SurfaceDistanceQuery:
        point_array = _position(point, "point")
        signed_distance = self.signed_distance(point_array)
        projected = point_array - signed_distance * self.normal
        local = np.array(
            [
                np.dot(projected - self.center_m, self.tangent_u),
                np.dot(projected - self.center_m, self.tangent_v),
            ],
            dtype=float,
        )
        return SurfaceDistanceQuery(
            signed_distance_m=signed_distance,
            normal=self.normal.copy(),
            projected_point=projected,
            local_uv_m=local,
            surface_id=self.id,
        )

    def target_pose(self, position: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, 0] = self.tangent_u
        pose[:3, 1] = self.tangent_v
        pose[:3, 2] = self.normal
        pose[:3, 3] = _position(position, "position")
        return pose


@dataclass(frozen=True)
class WipePatchConfig:
    """Named rectangular region on a work surface."""

    id: str
    surface_id: str
    center_m: np.ndarray
    width_m: float
    height_m: float


def make_work_surface(
    *,
    id: str,
    primitive_id: str,
    center_m: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    width_m: float,
    height_m: float,
) -> WorkSurfaceConfig:
    """Build an orthonormal work-surface frame from YAML vectors."""

    normal_unit = _unit(_position(normal, "normal"), "normal")
    tangent_u_raw = _position(tangent_u, "tangent_u")
    tangent_u_projected = tangent_u_raw - np.dot(tangent_u_raw, normal_unit) * normal_unit
    tangent_u_unit = _unit(tangent_u_projected, "tangent_u")
    tangent_v_unit = np.cross(normal_unit, tangent_u_unit)
    tangent_v_unit = _unit(tangent_v_unit, "tangent_v")
    return WorkSurfaceConfig(
        id=id,
        primitive_id=primitive_id,
        center_m=_position(center_m, "center_m"),
        normal=normal_unit,
        tangent_u=tangent_u_unit,
        tangent_v=tangent_v_unit,
        width_m=float(width_m),
        height_m=float(height_m),
    )


def _position(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected {name} with shape (3,), got {array.shape}.")
    return array.copy()


def _unit(values: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return np.asarray(values, dtype=float) / norm
