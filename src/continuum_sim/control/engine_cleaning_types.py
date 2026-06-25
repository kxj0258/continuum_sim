"""Task-space types for the M6 executor cleaning controller scaffold.

This module defines pure data containers for future engine-cleaning control.
It does not implement Jacobian, tendon, or motor mapping; it does not connect
to MuJoCo runtime; it does not implement visual servo or dual-arm avoidance.

Sign convention: `CleaningWaypoint.normal` is the outward surface normal.
The signed gap is `dot(tcp_position - waypoint.position, waypoint.normal)`,
so tools outside the surface have positive gap. Measured normal force is
positive in compression, in Newtons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_sim.model.base_pose import Pose6D


@dataclass(frozen=True)
class EngineCleaningFeedback:
    """Current executor TCP feedback consumed by the task-space controller."""

    tcp_pose: Pose6D
    measured_normal_force_n: float = 0.0
    contact_distance_m: float | None = None
    in_contact: bool = False
    timestamp_s: float | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tcp_pose, Pose6D):
            raise ValueError("tcp_pose must be a Pose6D.")
        force = float(self.measured_normal_force_n)
        if not np.isfinite(force):
            raise ValueError("measured_normal_force_n must be finite.")
        if force < 0.0:
            raise ValueError("measured_normal_force_n must be non-negative.")
        object.__setattr__(self, "measured_normal_force_n", force)
        if self.contact_distance_m is not None:
            distance = float(self.contact_distance_m)
            if not np.isfinite(distance):
                raise ValueError("contact_distance_m must be finite when provided.")
            object.__setattr__(self, "contact_distance_m", distance)
        if self.timestamp_s is not None:
            timestamp = float(self.timestamp_s)
            if not np.isfinite(timestamp):
                raise ValueError("timestamp_s must be finite when provided.")
            object.__setattr__(self, "timestamp_s", timestamp)


@dataclass(frozen=True)
class EngineCleaningCommand:
    """Task-space velocity intent produced by the M6 scaffold controller."""

    desired_tcp_velocity_world: np.ndarray
    active_waypoint_index: int
    phase: str
    waypoint_reached: bool
    safety_stop: bool
    stop_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        velocity = np.asarray(self.desired_tcp_velocity_world, dtype=float)
        if velocity.shape != (3,):
            raise ValueError(
                "desired_tcp_velocity_world must have shape (3,), "
                f"got {velocity.shape}."
            )
        if not np.all(np.isfinite(velocity)):
            raise ValueError("desired_tcp_velocity_world must be finite.")
        object.__setattr__(self, "desired_tcp_velocity_world", velocity.copy())
        if self.active_waypoint_index < 0:
            raise ValueError("active_waypoint_index must be non-negative.")


@dataclass(frozen=True)
class EngineCleaningControllerGains:
    """Gains and safety limits for executor task-space cleaning control."""

    tangential_position_gain: float
    normal_position_gain: float
    normal_force_gain: float
    approach_position_gain: float
    retreat_position_gain: float
    max_tcp_speed_mps: float
    max_normal_speed_mps: float
    waypoint_tolerance_m: float
    max_contact_force_n: float
    force_deadband_n: float
    min_clearance_m: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "tangential_position_gain",
            "normal_position_gain",
            "normal_force_gain",
            "approach_position_gain",
            "retreat_position_gain",
            "waypoint_tolerance_m",
            "force_deadband_n",
            "min_clearance_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
            object.__setattr__(self, name, value)
        for name in ("max_tcp_speed_mps", "max_normal_speed_mps", "max_contact_force_n"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")
            object.__setattr__(self, name, value)
