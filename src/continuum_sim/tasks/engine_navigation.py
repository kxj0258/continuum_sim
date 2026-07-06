"""Resolve staged dual-arm navigation targets from engine-scene annotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_scene import (
    EngineRegionConfig,
    EngineSceneConfig,
    effective_engine_frame_position,
)


LOCAL_PATH_TYPES = ("transverse_square",)


@dataclass(frozen=True)
class EngineNavigationSpec:
    """Validated staged engine-navigation parameters."""

    entry_region: str
    insertion_path: str
    pre_entry_standoff_m: float = 0.05
    insertion_waypoint_spacing_m: float = 0.02
    base_position_tolerance_m: float = 0.005
    base_orientation_tolerance_rad: float = 0.035
    base_position_gain: float = 1.5
    base_orientation_gain: float = 2.0
    local_path_type: str = "transverse_square"
    local_path_radius_m: float = 0.01
    local_path_samples: int = 40
    phase_timeout_steps: int = 5000

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "EngineNavigationSpec":
        """Load a spec from the ``task.engine_navigation`` mapping."""

        if not isinstance(values, dict):
            raise ValueError("task.engine_navigation must be a mapping.")
        local_path = values.get("local_path", {})
        if not isinstance(local_path, dict):
            raise ValueError("task.engine_navigation.local_path must be a mapping.")
        spec = cls(
            entry_region=str(_required(values, "entry_region")),
            insertion_path=str(_required(values, "insertion_path")),
            pre_entry_standoff_m=float(values.get("pre_entry_standoff_m", 0.05)),
            insertion_waypoint_spacing_m=float(
                values.get("insertion_waypoint_spacing_m", 0.02)
            ),
            base_position_tolerance_m=float(
                values.get("base_position_tolerance_m", 0.005)
            ),
            base_orientation_tolerance_rad=float(
                values.get("base_orientation_tolerance_rad", 0.035)
            ),
            base_position_gain=float(values.get("base_position_gain", 1.5)),
            base_orientation_gain=float(values.get("base_orientation_gain", 2.0)),
            local_path_type=str(local_path.get("type", "transverse_square")),
            local_path_radius_m=float(local_path.get("radius_m", 0.01)),
            local_path_samples=int(local_path.get("samples", 40)),
            phase_timeout_steps=int(values.get("phase_timeout_steps", 5000)),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        if not self.entry_region:
            raise ValueError("engine_navigation.entry_region must be non-empty.")
        if not self.insertion_path:
            raise ValueError("engine_navigation.insertion_path must be non-empty.")
        positive = {
            "pre_entry_standoff_m": self.pre_entry_standoff_m,
            "insertion_waypoint_spacing_m": self.insertion_waypoint_spacing_m,
            "base_position_tolerance_m": self.base_position_tolerance_m,
            "base_orientation_tolerance_rad": self.base_orientation_tolerance_rad,
            "base_position_gain": self.base_position_gain,
            "base_orientation_gain": self.base_orientation_gain,
            "local_path_radius_m": self.local_path_radius_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"engine_navigation.{name} must be positive and finite.")
        if self.local_path_type not in LOCAL_PATH_TYPES:
            raise ValueError(
                f"engine_navigation.local_path.type must be one of {LOCAL_PATH_TYPES}."
            )
        if self.local_path_samples < 4:
            raise ValueError("engine_navigation.local_path.samples must be at least 4.")
        if self.phase_timeout_steps <= 0:
            raise ValueError("engine_navigation.phase_timeout_steps must be positive.")


@dataclass(frozen=True)
class EngineNavigationPlan:
    """Resolved world-frame staged-navigation targets."""

    pre_entry_tip_world: np.ndarray
    insertion_direction_world: np.ndarray
    insertion_tip_waypoints_world: np.ndarray
    pre_entry_base_pose: Pose6D
    insertion_base_poses: tuple[Pose6D, ...]
    executor_waypoints_world: np.ndarray
    observer_roi_world: np.ndarray


def resolve_engine_navigation_plan(
    spec: EngineNavigationSpec,
    scene: EngineSceneConfig,
    assembly: RobotAssemblyConfig,
) -> EngineNavigationPlan:
    """Resolve named engine annotations into base and arm targets."""

    if assembly.base.control_mode == "fixed":
        raise ValueError("engine_navigation requires a non-fixed mobile base.")
    executors = [arm for arm in assembly.enabled_arms if arm.role == "executor"]
    observers = [arm for arm in assembly.enabled_arms if arm.role == "observer"]
    if len(executors) != 1 or len(observers) != 1:
        raise ValueError(
            "engine_navigation requires exactly one enabled executor and observer."
        )

    region = _named_region(scene, spec.entry_region)
    entry_point = _region_point_world(region, scene)
    path = _named_path_world(scene, spec.insertion_path)
    if np.linalg.norm(path[0] - entry_point) > 0.01:
        raise ValueError(
            f"Engine path {spec.insertion_path!r} does not start at region "
            f"{spec.entry_region!r} within 0.01 m."
        )
    insertion_waypoints = _resample_polyline(
        path,
        spec.insertion_waypoint_spacing_m,
    )
    insertion_direction = _unit(path[1] - path[0], "insertion path direction")
    pre_entry_tip = entry_point - spec.pre_entry_standoff_m * insertion_direction
    tip_orientation = _orientation_along_z(insertion_direction)
    base_to_straight_tip = _base_to_straight_tip_pose(executors[0])

    pre_entry_tip_pose = Pose6D(
        position=pre_entry_tip,
        quat=tip_orientation.quat,
    )
    pre_entry_base_pose = pre_entry_tip_pose.compose(base_to_straight_tip.inverse())
    insertion_base_poses = tuple(
        Pose6D(position=point, quat=tip_orientation.quat).compose(
            base_to_straight_tip.inverse()
        )
        for point in insertion_waypoints
    )
    executor_waypoints = _transverse_square(
        center=insertion_waypoints[-1],
        frame=tip_orientation,
        radius_m=spec.local_path_radius_m,
        samples=spec.local_path_samples,
    )
    return EngineNavigationPlan(
        pre_entry_tip_world=pre_entry_tip,
        insertion_direction_world=insertion_direction,
        insertion_tip_waypoints_world=insertion_waypoints,
        pre_entry_base_pose=pre_entry_base_pose,
        insertion_base_poses=insertion_base_poses,
        executor_waypoints_world=executor_waypoints,
        observer_roi_world=insertion_waypoints[-1].copy(),
    )


def _named_region(scene: EngineSceneConfig, name: str) -> EngineRegionConfig:
    try:
        return scene.regions[name]
    except KeyError as exc:
        raise ValueError(f"Unknown engine navigation region {name!r}.") from exc


def _region_point_world(
    region: EngineRegionConfig,
    scene: EngineSceneConfig,
) -> np.ndarray:
    point = region.center_m if region.center_m is not None else region.position_m
    if point is None:
        raise ValueError(
            f"Engine navigation region {region.name!r} has no center or position."
        )
    return _point_world(point, region.frame, scene)


def _named_path_world(scene: EngineSceneConfig, name: str) -> np.ndarray:
    matches = [
        path
        for path in scene.exploration_paths
        if path.enabled and path.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown enabled engine navigation path {name!r}.")
    path = matches[0]
    if path.frame == "world":
        return path.points_m.copy()
    frame = _engine_frame_pose(scene)
    return frame.transform_points(path.points_m)


def _point_world(
    point: np.ndarray,
    frame: str,
    scene: EngineSceneConfig,
) -> np.ndarray:
    values = np.asarray(point, dtype=float)
    if frame == "world":
        return values.copy()
    return _engine_frame_pose(scene).transform_point(values)


def _engine_frame_pose(scene: EngineSceneConfig) -> Pose6D:
    return Pose6D(
        position=effective_engine_frame_position(scene),
        quat=scene.engine.pose.quat_wxyz,
    )


def _resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    result = [values[0].copy()]
    for start, end in zip(values[:-1], values[1:], strict=True):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            raise ValueError("Engine navigation path contains duplicate adjacent points.")
        intervals = max(1, int(np.ceil(distance / spacing_m)))
        result.extend(
            start + (index / intervals) * delta
            for index in range(1, intervals + 1)
        )
    return np.asarray(result, dtype=float)


def _base_to_straight_tip_pose(executor) -> Pose6D:
    length = float(sum(segment.length for segment in executor.spatial_arm.params.segments))
    straight_tip = Pose6D(
        position=np.array([0.0, 0.0, length], dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    return executor.mount_pose.compose(straight_tip)


def _orientation_along_z(direction: np.ndarray) -> Pose6D:
    z_axis = _unit(direction, "insertion direction")
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _unit(reference - float(reference @ z_axis) * z_axis, "frame x-axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "frame y-axis")
    x_axis = _unit(np.cross(y_axis, z_axis), "frame x-axis")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    return Pose6D.from_matrix(transform)


def _transverse_square(
    *,
    center: np.ndarray,
    frame: Pose6D,
    radius_m: float,
    samples: int,
) -> np.ndarray:
    half_side = radius_m
    perimeter = 8.0 * half_side
    distances = np.linspace(0.0, perimeter, samples, endpoint=False)
    planar = np.empty((samples, 2), dtype=float)
    for index, distance in enumerate(distances):
        side = int(distance // (2.0 * half_side))
        offset = distance - side * 2.0 * half_side
        if side == 0:
            planar[index] = (-half_side + offset, -half_side)
        elif side == 1:
            planar[index] = (half_side, -half_side + offset)
        elif side == 2:
            planar[index] = (half_side - offset, half_side)
        else:
            planar[index] = (-half_side, half_side - offset)
    rotation = frame.as_matrix()[:3, :3]
    return (
        np.asarray(center, dtype=float)[None, :]
        + planar[:, :1] * rotation[:, 0][None, :]
        + planar[:, 1:] * rotation[:, 1][None, :]
    )


def _unit(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _required(values: dict[str, object], name: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required engine_navigation field {name!r}.")
    return values[name]
