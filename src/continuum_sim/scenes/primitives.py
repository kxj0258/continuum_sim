"""Clearance primitives for structured inspection scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DistanceQuery:
    """Distance-to-constraint query result.

    ``distance_m`` is positive in the free space, zero on the constraint, and
    negative inside a solid obstacle or outside an interior shell.
    ``normal`` points in the direction that increases clearance.
    """

    distance_m: float
    normal: np.ndarray
    source_id: str
    point: np.ndarray


class ClearancePrimitive(Protocol):
    """Primitive that can report clearance for a point."""

    id: str

    def clearance(self, point: np.ndarray) -> DistanceQuery:
        """Return signed clearance from this primitive."""


@dataclass(frozen=True)
class InteriorShellPrimitive:
    """Interior wall of an axis-aligned cylindrical or frustum chamber."""

    id: str
    z_min_m: float
    z_max_m: float
    radius_start_m: float
    radius_end_m: float
    center_xy_m: tuple[float, float] = (0.0, 0.0)

    def radius_at(self, z_m: float) -> float:
        span = self.z_max_m - self.z_min_m
        if span <= 0.0:
            raise ValueError(f"{self.id} z_max_m must be greater than z_min_m.")
        alpha = np.clip((z_m - self.z_min_m) / span, 0.0, 1.0)
        return float((1.0 - alpha) * self.radius_start_m + alpha * self.radius_end_m)

    def clearance(self, point: np.ndarray) -> DistanceQuery:
        point_array = _point(point)
        if point_array[2] < self.z_min_m or point_array[2] > self.z_max_m:
            return DistanceQuery(
                distance_m=float("inf"),
                normal=np.zeros(3, dtype=float),
                source_id=self.id,
                point=point_array,
            )
        center = np.array([self.center_xy_m[0], self.center_xy_m[1], 0.0], dtype=float)
        radial_vector = point_array - center
        radial_vector[2] = 0.0
        radial_distance = float(np.linalg.norm(radial_vector))
        if radial_distance > 1.0e-12:
            radial_unit = radial_vector / radial_distance
        else:
            radial_unit = np.array([1.0, 0.0, 0.0], dtype=float)
        radius = self.radius_at(float(point_array[2]))
        return DistanceQuery(
            distance_m=radius - radial_distance,
            normal=-radial_unit,
            source_id=self.id,
            point=point_array,
        )


@dataclass(frozen=True)
class CylinderObstaclePrimitive:
    """Finite cylinder obstacle with axis aligned to x, y, or z."""

    id: str
    center_m: tuple[float, float, float]
    radius_m: float
    half_length_m: float
    axis: str = "z"

    def clearance(self, point: np.ndarray) -> DistanceQuery:
        point_array = _point(point)
        local = point_array - np.asarray(self.center_m, dtype=float)
        axis_index = _axis_index(self.axis)
        axial = float(local[axis_index])
        radial_vector = local.copy()
        radial_vector[axis_index] = 0.0
        radial_norm = float(np.linalg.norm(radial_vector))
        radial_signed = radial_norm - self.radius_m
        axial_signed = abs(axial) - self.half_length_m
        outside = np.array(
            [max(radial_signed, 0.0), max(axial_signed, 0.0)],
            dtype=float,
        )
        signed_distance = float(np.linalg.norm(outside) + min(max(radial_signed, axial_signed), 0.0))

        if radial_norm > 1.0e-12:
            radial_normal = radial_vector / radial_norm
        else:
            radial_normal = _perpendicular_axis(axis_index)
        axial_normal = np.zeros(3, dtype=float)
        axial_normal[axis_index] = 1.0 if axial >= 0.0 else -1.0
        if radial_signed >= axial_signed:
            normal = radial_normal
        else:
            normal = axial_normal
        return DistanceQuery(
            distance_m=signed_distance,
            normal=_unit(normal),
            source_id=self.id,
            point=point_array,
        )


@dataclass(frozen=True)
class BoxObstaclePrimitive:
    """Axis-aligned box obstacle."""

    id: str
    center_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]

    def clearance(self, point: np.ndarray) -> DistanceQuery:
        point_array = _point(point)
        center = np.asarray(self.center_m, dtype=float)
        half_size = np.asarray(self.half_size_m, dtype=float)
        local = point_array - center
        outside = np.abs(local) - half_size
        outside_positive = np.maximum(outside, 0.0)
        outside_distance = float(np.linalg.norm(outside_positive))
        inside_distance = min(float(np.max(outside)), 0.0)
        signed_distance = outside_distance + inside_distance

        if outside_distance > 1.0e-12:
            closest = np.clip(local, -half_size, half_size)
            normal = local - closest
        else:
            face_clearances = half_size - np.abs(local)
            axis = int(np.argmin(face_clearances))
            normal = np.zeros(3, dtype=float)
            normal[axis] = 1.0 if local[axis] >= 0.0 else -1.0
        return DistanceQuery(
            distance_m=signed_distance,
            normal=_unit(normal),
            source_id=self.id,
            point=point_array,
        )


def nearest_clearance(
    point: np.ndarray,
    primitives: tuple[ClearancePrimitive, ...],
) -> DistanceQuery:
    """Return the most restrictive clearance query for ``point``."""

    if not primitives:
        return DistanceQuery(
            distance_m=float("inf"),
            normal=np.zeros(3, dtype=float),
            source_id="free_space",
            point=_point(point),
        )
    queries = [primitive.clearance(point) for primitive in primitives]
    return min(queries, key=lambda query: query.distance_m)


def _point(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected point with shape (3,), got {array.shape}.")
    return array.copy()


def _axis_index(axis: str) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    if axis not in axes:
        raise ValueError(f"axis must be one of {tuple(axes)}, got {axis!r}.")
    return axes[axis]


def _perpendicular_axis(axis_index: int) -> np.ndarray:
    vector = np.zeros(3, dtype=float)
    vector[(axis_index + 1) % 3] = 1.0
    return vector


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return array / norm
