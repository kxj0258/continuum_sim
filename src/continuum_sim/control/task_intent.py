"""Strongly typed contract between task controllers and low-level motion control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


CARTESIAN_CONTROL_MODES = ("position", "velocity")
ORIENTATION_CONTROL_MODES = ("disabled", "quaternion")
OBSERVER_CONTROL_MODES = ("tracking", "collision_avoidance", "disabled")


@dataclass(frozen=True)
class CartesianTaskIntent:
    """Executor objective expressed in world-frame Cartesian coordinates.

    ``target_position_world`` always records the semantic task target.  In
    velocity mode the low-level controller anchors its internal position servo
    at the measured TCP, so only ``feedforward_velocity_world`` is commanded.
    """

    target_position_world: np.ndarray
    feedforward_velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    target_orientation_world_wxyz: np.ndarray | None = None
    feedforward_angular_velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    control_mode: str = "position"
    orientation_control_mode: str = "disabled"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_position_world",
            _vector3(self.target_position_world, "target_position_world"),
        )
        object.__setattr__(
            self,
            "feedforward_velocity_world",
            _vector3(
                self.feedforward_velocity_world,
                "feedforward_velocity_world",
            ),
        )
        if self.target_orientation_world_wxyz is not None:
            object.__setattr__(
                self,
                "target_orientation_world_wxyz",
                _quat4(
                    self.target_orientation_world_wxyz,
                    "target_orientation_world_wxyz",
                ),
            )
        object.__setattr__(
            self,
            "feedforward_angular_velocity_world",
            _vector3(
                self.feedforward_angular_velocity_world,
                "feedforward_angular_velocity_world",
            ),
        )
        if self.control_mode not in CARTESIAN_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{CARTESIAN_CONTROL_MODES}."
            )
        if self.orientation_control_mode not in ORIENTATION_CONTROL_MODES:
            raise ValueError(
                "orientation_control_mode must be one of "
                f"{ORIENTATION_CONTROL_MODES}."
            )
        if (
            self.orientation_control_mode != "disabled"
            and self.target_orientation_world_wxyz is None
        ):
            raise ValueError(
                "target_orientation_world_wxyz is required when orientation "
                "control is enabled."
            )


@dataclass(frozen=True)
class ObserverTaskIntent:
    """Optional observer objective produced by an independent upper controller."""

    control_mode: str = "tracking"

    executor_offset_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.04, 0.02], dtype=float)
    )
    roi_position_world: np.ndarray | None = None
    roi_blend: float = 0.25

    def __post_init__(self) -> None:
        if self.control_mode not in OBSERVER_CONTROL_MODES:
            raise ValueError(
                "observer control_mode must be one of "
                f"{OBSERVER_CONTROL_MODES}."
            )
        object.__setattr__(
            self,
            "executor_offset_world",
            _vector3(self.executor_offset_world, "executor_offset_world"),
        )
        if self.roi_position_world is not None:
            object.__setattr__(
                self,
                "roi_position_world",
                _vector3(self.roi_position_world, "roi_position_world"),
            )
        if not np.isfinite(self.roi_blend) or not 0.0 <= self.roi_blend <= 1.0:
            raise ValueError("roi_blend must be finite and in [0, 1].")


@dataclass(frozen=True)
class ContactTaskIntent:
    """Task-level contact objective retained for safety and diagnostics."""

    surface_normal_world: np.ndarray
    target_normal_force_n: float = 0.0
    target_contact_distance_m: float = 0.0

    def __post_init__(self) -> None:
        normal = _vector3(self.surface_normal_world, "surface_normal_world")
        norm = float(np.linalg.norm(normal))
        if norm <= np.finfo(float).eps:
            raise ValueError("surface_normal_world must be nonzero.")
        object.__setattr__(self, "surface_normal_world", normal / norm)
        if not np.isfinite(self.target_normal_force_n):
            raise ValueError("target_normal_force_n must be finite.")
        if not np.isfinite(self.target_contact_distance_m):
            raise ValueError("target_contact_distance_m must be finite.")


@dataclass(frozen=True)
class SafetyTaskIntent:
    """Task-specific safety bounds consumed by supervisors or recorders."""

    minimum_clearance_m: float | None = None
    maximum_contact_force_n: float | None = None
    terminate_on_violation: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_clearance_m", self.minimum_clearance_m),
            ("maximum_contact_force_n", self.maximum_contact_force_n),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be non-negative and finite.")


@dataclass(frozen=True)
class SystemTaskIntent:
    """Unified upper-controller output for the robot system."""

    executor: CartesianTaskIntent
    observer: ObserverTaskIntent | None = None
    contact: ContactTaskIntent | None = None
    safety: SafetyTaskIntent | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskStatus:
    """Task lifecycle state independent of the actuator command."""

    task_type: str
    phase: str
    active_index: int = 0
    complete: bool = False
    stop_reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ValueError("task_type must be non-empty.")
        if not self.phase:
            raise ValueError("phase must be non-empty.")
        if self.active_index < 0:
            raise ValueError("active_index must be non-negative.")


@dataclass(frozen=True)
class TaskStep:
    """One upper-controller output sample."""

    intent: SystemTaskIntent
    status: TaskStatus


def _vector3(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    return result.copy()


def _quat4(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (4,).")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return (result / norm).copy()
