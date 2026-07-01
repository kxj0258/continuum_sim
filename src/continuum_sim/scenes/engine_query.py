"""Primitive-based engine clearance queries for real-time control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.scenes.engine_scene import (
    EngineSceneConfig,
    effective_engine_frame_position,
)
from continuum_sim.scenes.primitive_collision import PrimitiveCollisionGeomConfig
from continuum_sim.scenes.primitives import DistanceQuery


class EngineSceneQueryProtocol(Protocol):
    """Control-facing engine geometry query boundary."""

    def nearest_distance(self, point_world: np.ndarray) -> DistanceQuery:
        """Return signed distance to the nearest enabled control primitive."""

    def nearest_centerline_clearance(self, centerline_world: np.ndarray) -> DistanceQuery:
        """Return the most restrictive sampled centerline clearance."""


@dataclass(frozen=True)
class EnginePrimitiveSceneQuery:
    """Signed-distance queries backed by configured primitive collision geoms."""

    config: EngineSceneConfig

    @property
    def enabled_geoms(self) -> tuple[PrimitiveCollisionGeomConfig, ...]:
        return tuple(geom for geom in self.config.primitive_collision_geoms if geom.enabled)

    def nearest_distance(self, point_world: np.ndarray) -> DistanceQuery:
        point = _point(point_world)
        if not self.enabled_geoms:
            return DistanceQuery(
                distance_m=float("inf"),
                normal=np.zeros(3, dtype=float),
                source_id="engine_free_space",
                point=point,
            )
        return min(
            (self._clearance(geom, point) for geom in self.enabled_geoms),
            key=lambda query: query.distance_m,
        )

    def nearest_centerline_clearance(self, centerline_world: np.ndarray) -> DistanceQuery:
        centerline = np.asarray(centerline_world, dtype=float)
        if centerline.ndim != 2 or centerline.shape[1] != 3:
            raise ValueError("centerline_world must have shape (N, 3).")
        if centerline.shape[0] == 0:
            return self.nearest_distance(np.zeros(3, dtype=float))
        return min(
            (self.nearest_distance(point) for point in centerline),
            key=lambda query: query.distance_m,
        )

    def _clearance(
        self,
        geom: PrimitiveCollisionGeomConfig,
        point_world: np.ndarray,
    ) -> DistanceQuery:
        if geom.type == "capsule" and geom.fromto_m is not None:
            start, end, radius = _capsule_world(geom, self.config)
            closest = _closest_segment_point(point_world, start, end)
            offset = point_world - closest
            norm = float(np.linalg.norm(offset))
            return _query(geom.name, point_world, norm - radius, offset)

        pose, scale = _primitive_pose_world(geom, self.config)
        local = pose.inverse().transform_point(point_world)
        if geom.type == "sphere":
            radius = float(geom.radius_m) * scale
            distance = float(np.linalg.norm(local)) - radius
            normal_local = local
        elif geom.type == "box":
            half_size = 0.5 * np.asarray(geom.size_m, dtype=float) * scale
            distance, normal_local = _box_distance(local, half_size)
        elif geom.type == "cylinder":
            radius = float(geom.radius_m) * scale
            half_length = 0.5 * float(geom.length_m) * scale
            distance, normal_local = _cylinder_distance(local, radius, half_length)
        elif geom.type == "capsule":
            radius = float(geom.radius_m) * scale
            half_length = 0.5 * float(geom.length_m) * scale
            closest = np.array(
                [0.0, 0.0, np.clip(local[2], -half_length, half_length)],
                dtype=float,
            )
            offset = local - closest
            distance = float(np.linalg.norm(offset)) - radius
            normal_local = offset
        else:
            raise ValueError(f"Unsupported engine primitive type {geom.type!r}.")
        normal_world = pose.transform_vector(_unit(normal_local))
        return DistanceQuery(
            distance_m=float(distance),
            normal=_unit(normal_world),
            source_id=geom.name,
            point=point_world.copy(),
        )


def _primitive_pose_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[Pose6D, float]:
    local_pose = Pose6D(
        position=(
            np.zeros(3, dtype=float)
            if geom.position_m is None
            else np.asarray(geom.position_m, dtype=float)
        ),
        quat=(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if geom.quat_wxyz is None
            else np.asarray(geom.quat_wxyz, dtype=float)
        ),
    )
    if geom.frame == "world":
        return local_pose, 1.0
    engine_pose = Pose6D(
        position=effective_engine_frame_position(config),
        quat=config.engine.pose.quat_wxyz,
    )
    scaled_local = Pose6D(
        position=local_pose.position * config.engine.scale,
        quat=local_pose.quat,
    )
    return engine_pose.compose(scaled_local), float(config.engine.scale)


def _capsule_world(
    geom: PrimitiveCollisionGeomConfig,
    config: EngineSceneConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(geom.fromto_m, dtype=float).reshape(2, 3)
    if geom.frame == "world":
        return values[0], values[1], float(geom.radius_m)
    engine_pose = Pose6D(
        position=effective_engine_frame_position(config),
        quat=config.engine.pose.quat_wxyz,
    )
    points = engine_pose.transform_points(values * config.engine.scale)
    return points[0], points[1], float(geom.radius_m) * config.engine.scale


def _box_distance(point: np.ndarray, half_size: np.ndarray) -> tuple[float, np.ndarray]:
    outside = np.abs(point) - half_size
    positive = np.maximum(outside, 0.0)
    outside_norm = float(np.linalg.norm(positive))
    distance = outside_norm + min(float(np.max(outside)), 0.0)
    if outside_norm > 1.0e-12:
        closest = np.clip(point, -half_size, half_size)
        normal = point - closest
    else:
        axis = int(np.argmax(outside))
        normal = np.zeros(3, dtype=float)
        normal[axis] = 1.0 if point[axis] >= 0.0 else -1.0
    return distance, normal


def _cylinder_distance(
    point: np.ndarray,
    radius: float,
    half_length: float,
) -> tuple[float, np.ndarray]:
    radial_norm = float(np.linalg.norm(point[:2]))
    radial = radial_norm - radius
    axial = abs(float(point[2])) - half_length
    outside = np.maximum(np.array([radial, axial], dtype=float), 0.0)
    distance = float(np.linalg.norm(outside) + min(max(radial, axial), 0.0))
    if radial >= axial:
        normal = np.array(
            [1.0, 0.0, 0.0]
            if radial_norm <= 1.0e-12
            else [point[0] / radial_norm, point[1] / radial_norm, 0.0],
            dtype=float,
        )
    else:
        normal = np.array([0.0, 0.0, 1.0 if point[2] >= 0.0 else -1.0])
    return distance, normal


def _closest_segment_point(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= 1.0e-18:
        return start.copy()
    alpha = float(np.clip(((point - start) @ segment) / denominator, 0.0, 1.0))
    return start + alpha * segment


def _query(name: str, point: np.ndarray, distance: float, normal: np.ndarray) -> DistanceQuery:
    return DistanceQuery(
        distance_m=float(distance),
        normal=_unit(normal),
        source_id=name,
        point=point.copy(),
    )


def _point(values: np.ndarray) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,):
        raise ValueError("point_world must have shape (3,).")
    return point.copy()


def _unit(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm

