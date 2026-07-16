"""Task-space servo layer for executor Cartesian references."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


TASK_SPACE_CONTROL_MODES = ("position", "velocity")


@dataclass(frozen=True)
class TaskSpaceServoConfig:
    """Gains and speed limits for task-space velocity generation."""

    position_gain: float = 4.0
    feedforward_gain: float = 1.0
    max_speed_mps: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_gain) or self.position_gain < 0.0:
            raise ValueError("position_gain must be finite and non-negative.")
        if not np.isfinite(self.feedforward_gain) or self.feedforward_gain < 0.0:
            raise ValueError("feedforward_gain must be finite and non-negative.")
        if self.max_speed_mps is not None and (
            not np.isfinite(self.max_speed_mps) or self.max_speed_mps <= 0.0
        ):
            raise ValueError("max_speed_mps must be positive and finite when set.")


@dataclass(frozen=True)
class TaskSpaceReference:
    """Executor task-space command sampled from the task layer."""

    target_position_world: np.ndarray
    feedforward_velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    control_mode: str = "position"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_position_world",
            _vector3(self.target_position_world, "target_position_world"),
        )
        object.__setattr__(
            self,
            "feedforward_velocity_world",
            _vector3(self.feedforward_velocity_world, "feedforward_velocity_world"),
        )
        if self.control_mode not in TASK_SPACE_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{TASK_SPACE_CONTROL_MODES}."
            )


@dataclass(frozen=True)
class TaskSpaceVelocityCommand:
    """Layer-2 output consumed by the tendon command layer."""

    tcp_velocity_world: np.ndarray
    servo_anchor_position_world: np.ndarray
    semantic_target_position_world: np.ndarray
    position_error_world: np.ndarray
    raw_velocity_world: np.ndarray
    feedforward_velocity_world: np.ndarray
    scaled_feedforward_velocity_world: np.ndarray
    control_mode: str
    speed_limited: bool

    def __post_init__(self) -> None:
        for name in (
            "tcp_velocity_world",
            "servo_anchor_position_world",
            "semantic_target_position_world",
            "position_error_world",
            "raw_velocity_world",
            "feedforward_velocity_world",
            "scaled_feedforward_velocity_world",
        ):
            object.__setattr__(self, name, _vector3(getattr(self, name), name))
        if self.control_mode not in TASK_SPACE_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{TASK_SPACE_CONTROL_MODES}."
            )


class TaskSpaceServo:
    """Convert task references into Cartesian TCP velocity commands."""

    def __init__(self, config: TaskSpaceServoConfig = TaskSpaceServoConfig()) -> None:
        self.config = config

    def compute(
        self,
        measured_position_world: np.ndarray,
        reference: TaskSpaceReference,
    ) -> TaskSpaceVelocityCommand:
        measured = _vector3(measured_position_world, "measured_position_world")
        if reference.control_mode == "velocity":
            anchor = measured.copy()
            position_error = np.zeros(3, dtype=float)
            scaled_feedforward = reference.feedforward_velocity_world.copy()
        else:
            anchor = reference.target_position_world.copy()
            position_error = reference.target_position_world - measured
            scaled_feedforward = (
                self.config.feedforward_gain * reference.feedforward_velocity_world
            )
        raw_velocity = self.config.position_gain * position_error + scaled_feedforward
        velocity, limited = _limit_speed(raw_velocity, self.config.max_speed_mps)
        return TaskSpaceVelocityCommand(
            tcp_velocity_world=velocity,
            servo_anchor_position_world=anchor,
            semantic_target_position_world=reference.target_position_world,
            position_error_world=position_error,
            raw_velocity_world=raw_velocity,
            feedforward_velocity_world=reference.feedforward_velocity_world,
            scaled_feedforward_velocity_world=scaled_feedforward,
            control_mode=reference.control_mode,
            speed_limited=limited,
        )


def _limit_speed(
    velocity: np.ndarray,
    max_speed_mps: float | None,
) -> tuple[np.ndarray, bool]:
    result = np.asarray(velocity, dtype=float).copy()
    if max_speed_mps is None:
        return result, False
    norm = float(np.linalg.norm(result))
    if norm <= max_speed_mps:
        return result, False
    return result * (max_speed_mps / norm), True


def _vector3(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    return result.copy()
