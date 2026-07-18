"""Unified task-planner output shared by scenario controllers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TaskPhasePlan:
    """One named phase in a resolved task plan."""

    name: str
    start_index: int
    stop_index: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TaskPhasePlan.name must be non-empty.")
        if self.start_index < 0 or self.stop_index < self.start_index:
            raise ValueError("TaskPhasePlan indices must be ordered and non-negative.")


@dataclass(frozen=True)
class ClearanceConstraint:
    """Scene clearance policy emitted by a planner and consumed by controllers."""

    minimum_clearance_m: float
    terminate_on_violation: bool = True

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.minimum_clearance_m)
            or self.minimum_clearance_m < 0.0
        ):
            raise ValueError("minimum_clearance_m must be non-negative and finite.")


@dataclass(frozen=True)
class BaseApproachConstraint:
    """Mobile-base pre-positioning objective for staged tasks."""

    standoff_m: float = 0.0
    z_bias: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.standoff_m) or self.standoff_m < 0.0:
            raise ValueError("standoff_m must be non-negative and finite.")
        if not np.isfinite(self.z_bias):
            raise ValueError("z_bias must be finite.")


@dataclass(frozen=True)
class TaskPlan:
    """Planner-owned path, phase, contact, and safety targets.

    Controllers advance this plan and produce task intents.  Low-level control
    never needs to know whether waypoints came from a square trajectory, a
    wiping raster, or an engine-scene annotation.
    """

    waypoints_world: np.ndarray
    waypoint_orientations_world_wxyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=float)
    )
    waypoint_phases: tuple[str, ...] = ()
    phase_plan: tuple[TaskPhasePlan, ...] = ()
    target_force_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    surface_normal_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    surface_point_world: np.ndarray | None = None
    normals_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    standoff_distance_m: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    approach_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    source_waypoint_index: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=int)
    )
    clearance: ClearanceConstraint | None = None
    base_approach: BaseApproachConstraint | None = None

    def __post_init__(self) -> None:
        waypoints = _matrix(self.waypoints_world, "waypoints_world", columns=3)
        count = waypoints.shape[0]
        orientations = _optional_matrix(
            self.waypoint_orientations_world_wxyz,
            "waypoint_orientations_world_wxyz",
            columns=4,
        )
        if orientations.shape[0] not in (0, count):
            raise ValueError("waypoint_orientations_world_wxyz must be empty or match waypoint count.")
        phases = tuple(str(phase) for phase in self.waypoint_phases)
        if phases and len(phases) != count:
            raise ValueError("waypoint_phases must be empty or match waypoint count.")
        target_force = _vector(self.target_force_n, "target_force_n", count)
        normal = _unit_vector3(self.surface_normal_world, "surface_normal_world")
        normals = _optional_matrix(self.normals_world, "normals_world", columns=3)
        if normals.shape[0] == 0:
            normals = np.tile(normal, (count, 1))
        if normals.shape != (count, 3):
            raise ValueError("normals_world must be empty or have shape (N, 3).")
        standoff = _vector(self.standoff_distance_m, "standoff_distance_m", count)
        approach = _vector(self.approach_mask, "approach_mask", count, dtype=bool)
        source = _vector(self.source_waypoint_index, "source_waypoint_index", count, dtype=int)
        surface_point = (
            None
            if self.surface_point_world is None
            else _unitless_vector3(self.surface_point_world, "surface_point_world")
        )
        phase_plan = self.phase_plan or _phase_plan_from_labels(phases)
        object.__setattr__(self, "waypoints_world", waypoints)
        object.__setattr__(self, "waypoint_orientations_world_wxyz", orientations)
        object.__setattr__(self, "waypoint_phases", phases)
        object.__setattr__(self, "phase_plan", tuple(phase_plan))
        object.__setattr__(self, "target_force_n", target_force)
        object.__setattr__(self, "surface_normal_world", normal)
        object.__setattr__(self, "surface_point_world", surface_point)
        object.__setattr__(self, "normals_world", normals)
        object.__setattr__(self, "standoff_distance_m", standoff)
        object.__setattr__(self, "approach_mask", approach)
        object.__setattr__(self, "source_waypoint_index", source)


def _phase_plan_from_labels(phases: tuple[str, ...]) -> tuple[TaskPhasePlan, ...]:
    if not phases:
        return ()
    spans: list[TaskPhasePlan] = []
    start = 0
    current = phases[0]
    for index, phase in enumerate(phases[1:], start=1):
        if phase == current:
            continue
        spans.append(TaskPhasePlan(current, start, index - 1))
        start = index
        current = phase
    spans.append(TaskPhasePlan(current, start, len(phases) - 1))
    return tuple(spans)


def _matrix(values: np.ndarray, name: str, *, columns: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result.copy()


def _optional_matrix(values: np.ndarray, name: str, *, columns: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size == 0:
        return np.zeros((0, columns), dtype=float)
    return _matrix(result, name, columns=columns)


def _vector(
    values: np.ndarray,
    name: str,
    size: int,
    *,
    dtype: object = float,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.size == 0:
        return np.zeros(size, dtype=dtype)
    if result.shape != (size,):
        raise ValueError(f"{name} must be empty or have shape ({size},).")
    if dtype is not bool and not np.all(np.isfinite(result.astype(float))):
        raise ValueError(f"{name} must contain finite values.")
    return result.copy()


def _unit_vector3(values: np.ndarray, name: str) -> np.ndarray:
    vector = _unitless_vector3(values, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _unitless_vector3(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    return result.copy()
