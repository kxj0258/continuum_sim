"""Lightweight 6D mobile-base control scaffolding.

This module intentionally stops at command/state containers, clipping, and a
minimal pose integrator. It does not implement full whole-body optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_sim.model.base_pose import (
    Pose6D,
    quaternion_wxyz_multiply,
    rotation_vector_to_quaternion_wxyz,
)
from continuum_sim.model.mount_frame import MobileBaseLimitsConfig


@dataclass(frozen=True)
class MobileBaseCommand:
    """Velocity command for a 6D mobile base."""

    twist: np.ndarray
    frame: str = "world"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        twist = np.asarray(self.twist, dtype=float)
        if twist.shape != (6,):
            raise ValueError(f"twist must have shape (6,), got {twist.shape}.")
        if not np.all(np.isfinite(twist)):
            raise ValueError("twist must be finite.")
        object.__setattr__(self, "twist", twist.copy())

    @property
    def linear_velocity(self) -> np.ndarray:
        return self.twist[:3].copy()

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.twist[3:].copy()


@dataclass(frozen=True)
class MobileBaseState:
    """Runtime state for a 6D mobile base."""

    pose: Pose6D
    locked: bool = False
    last_twist: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose6D):
            raise ValueError("pose must be a Pose6D.")
        twist = np.asarray(self.last_twist, dtype=float)
        if twist.shape != (6,):
            raise ValueError(f"last_twist must have shape (6,), got {twist.shape}.")
        object.__setattr__(self, "last_twist", twist.copy())


@dataclass(frozen=True)
class WholeBodyCommand:
    """Future whole-body command scaffold."""

    base_command: MobileBaseCommand
    arm_command: np.ndarray | None = None
    base_locked: bool = False
    arm_locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_command, MobileBaseCommand):
            raise ValueError("base_command must be a MobileBaseCommand.")
        if self.arm_command is not None:
            arm = np.asarray(self.arm_command, dtype=float)
            if arm.ndim != 1:
                raise ValueError("arm_command must be a 1D vector when provided.")
            object.__setattr__(self, "arm_command", arm.copy())


def zero_mobile_base_command(*, frame: str = "world") -> MobileBaseCommand:
    """Return a zero-twist base command."""

    return MobileBaseCommand(twist=np.zeros(6, dtype=float), frame=frame)


def clip_base_twist(
    twist: np.ndarray,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    """Clip linear and angular twist components independently."""

    values = np.asarray(twist, dtype=float)
    if values.shape != (6,):
        raise ValueError(f"twist must have shape (6,), got {values.shape}.")
    linear = np.clip(values[:3], -float(max_linear_speed), float(max_linear_speed))
    angular = np.clip(values[3:], -float(max_angular_speed), float(max_angular_speed))
    return np.concatenate((linear, angular))


def resolve_mobile_base_command(
    state: MobileBaseState,
    command: MobileBaseCommand,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
) -> MobileBaseCommand:
    """Apply lock state and speed clipping to a commanded base twist."""

    if state.locked:
        return zero_mobile_base_command(frame=command.frame)
    return MobileBaseCommand(
        twist=clip_base_twist(
            command.twist,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        ),
        frame=command.frame,
        metadata=dict(command.metadata),
    )


def integrate_base_pose(
    state: MobileBaseState,
    command: MobileBaseCommand,
    *,
    dt: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> MobileBaseState:
    """Integrate a world-frame twist into a new pose with lock-aware clipping."""

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    resolved = resolve_mobile_base_command(
        state,
        command,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )
    delta_position = resolved.linear_velocity * float(dt)
    delta_quat = rotation_vector_to_quaternion_wxyz(resolved.angular_velocity * float(dt))
    next_pose = Pose6D(
        position=state.pose.position + delta_position,
        quat=quaternion_wxyz_multiply(delta_quat, state.pose.quat),
    )
    return MobileBaseState(
        pose=next_pose,
        locked=state.locked,
        last_twist=resolved.twist,
    )


def reset_mobile_base_state(pose: Pose6D | None = None) -> MobileBaseState:
    """Reset base state to the provided pose or identity."""

    return MobileBaseState(pose=pose or Pose6D.identity(), locked=False)


def set_mobile_base_locked(state: MobileBaseState, locked: bool) -> MobileBaseState:
    """Return a copy of the state with an updated lock flag."""

    return MobileBaseState(
        pose=state.pose,
        locked=bool(locked),
        last_twist=state.last_twist,
    )


def clamp_pose_to_limits(pose: Pose6D, limits: MobileBaseLimitsConfig) -> Pose6D:
    """Clamp translation into configured limits while leaving orientation unchanged."""

    position = np.clip(pose.position, limits.position_min_m, limits.position_max_m)
    return Pose6D(position=position, quat=pose.quat)
