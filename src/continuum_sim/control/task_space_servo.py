"""Task-space servo layer for executor Cartesian references."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.model.base_pose import quaternion_error_rotation_vector


TASK_SPACE_CONTROL_MODES = ("position", "velocity")
ORIENTATION_CONTROL_MODES = ("disabled", "quaternion")


@dataclass(frozen=True)
class TaskSpaceServoConfig:
    """Gains and speed limits for task-space velocity generation."""

    position_gain: float = 4.0
    orientation_gain: float = 2.0
    feedforward_gain: float = 1.0
    max_speed_mps: float | None = None
    max_angular_speed_rad_s: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_gain) or self.position_gain < 0.0:
            raise ValueError("position_gain must be finite and non-negative.")
        if not np.isfinite(self.orientation_gain) or self.orientation_gain < 0.0:
            raise ValueError("orientation_gain must be finite and non-negative.")
        if not np.isfinite(self.feedforward_gain) or self.feedforward_gain < 0.0:
            raise ValueError("feedforward_gain must be finite and non-negative.")
        if self.max_speed_mps is not None and (
            not np.isfinite(self.max_speed_mps) or self.max_speed_mps <= 0.0
        ):
            raise ValueError("max_speed_mps must be positive and finite when set.")
        if self.max_angular_speed_rad_s is not None and (
            not np.isfinite(self.max_angular_speed_rad_s)
            or self.max_angular_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "max_angular_speed_rad_s must be positive and finite when set."
            )


@dataclass(frozen=True)
class TaskSpaceReference:
    """Executor task-space command sampled from the task layer."""

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
            _vector3(self.feedforward_velocity_world, "feedforward_velocity_world"),
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
        if self.control_mode not in TASK_SPACE_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{TASK_SPACE_CONTROL_MODES}."
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
class TaskSpaceVelocityCommand:
    """Layer-2 output consumed by the tendon command layer."""

    tcp_velocity_world: np.ndarray
    tcp_angular_velocity_world: np.ndarray
    servo_anchor_position_world: np.ndarray
    semantic_target_position_world: np.ndarray
    semantic_target_orientation_world_wxyz: np.ndarray | None
    position_error_world: np.ndarray
    orientation_error_world: np.ndarray
    orientation_error_norm_rad: float
    raw_velocity_world: np.ndarray
    raw_angular_velocity_world: np.ndarray
    feedforward_velocity_world: np.ndarray
    feedforward_angular_velocity_world: np.ndarray
    scaled_feedforward_velocity_world: np.ndarray
    scaled_feedforward_angular_velocity_world: np.ndarray
    control_mode: str
    orientation_control_mode: str
    speed_limited: bool
    angular_speed_limited: bool

    def __post_init__(self) -> None:
        for name in (
            "tcp_velocity_world",
            "tcp_angular_velocity_world",
            "servo_anchor_position_world",
            "semantic_target_position_world",
            "position_error_world",
            "orientation_error_world",
            "raw_velocity_world",
            "raw_angular_velocity_world",
            "feedforward_velocity_world",
            "feedforward_angular_velocity_world",
            "scaled_feedforward_velocity_world",
            "scaled_feedforward_angular_velocity_world",
        ):
            object.__setattr__(self, name, _vector3(getattr(self, name), name))
        if self.semantic_target_orientation_world_wxyz is not None:
            object.__setattr__(
                self,
                "semantic_target_orientation_world_wxyz",
                _quat4(
                    self.semantic_target_orientation_world_wxyz,
                    "semantic_target_orientation_world_wxyz",
                ),
            )
        if not np.isfinite(self.orientation_error_norm_rad):
            raise ValueError("orientation_error_norm_rad must be finite.")
        if self.control_mode not in TASK_SPACE_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{TASK_SPACE_CONTROL_MODES}."
            )
        if self.orientation_control_mode not in ORIENTATION_CONTROL_MODES:
            raise ValueError(
                "orientation_control_mode must be one of "
                f"{ORIENTATION_CONTROL_MODES}."
            )


class TaskSpaceServo:
    """Convert task references into Cartesian TCP velocity commands."""

    def __init__(self, config: TaskSpaceServoConfig = TaskSpaceServoConfig()) -> None:
        self.config = config

    def compute(
        self,
        measured_position_world: np.ndarray,
        measured_orientation_world_wxyz: np.ndarray,
        reference: TaskSpaceReference,
    ) -> TaskSpaceVelocityCommand:
        measured = _vector3(measured_position_world, "measured_position_world")
        measured_orientation = _quat4(
            measured_orientation_world_wxyz,
            "measured_orientation_world_wxyz",
        )
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
        if reference.orientation_control_mode == "disabled":
            orientation_error = np.zeros(3, dtype=float)
            scaled_angular_feedforward = np.zeros(3, dtype=float)
            raw_angular_velocity = np.zeros(3, dtype=float)
            angular_velocity = np.zeros(3, dtype=float)
            angular_limited = False
            target_orientation = None
        else:
            target_orientation = reference.target_orientation_world_wxyz
            orientation_error = quaternion_error_rotation_vector(
                target_orientation,
                measured_orientation,
            )
            scaled_angular_feedforward = (
                self.config.feedforward_gain
                * reference.feedforward_angular_velocity_world
            )
            raw_angular_velocity = (
                self.config.orientation_gain * orientation_error
                + scaled_angular_feedforward
            )
            angular_velocity, angular_limited = _limit_speed(
                raw_angular_velocity,
                self.config.max_angular_speed_rad_s,
            )
        return TaskSpaceVelocityCommand(
            tcp_velocity_world=velocity,
            tcp_angular_velocity_world=angular_velocity,
            servo_anchor_position_world=anchor,
            semantic_target_position_world=reference.target_position_world,
            semantic_target_orientation_world_wxyz=target_orientation,
            position_error_world=position_error,
            orientation_error_world=orientation_error,
            orientation_error_norm_rad=float(np.linalg.norm(orientation_error)),
            raw_velocity_world=raw_velocity,
            raw_angular_velocity_world=raw_angular_velocity,
            feedforward_velocity_world=reference.feedforward_velocity_world,
            feedforward_angular_velocity_world=(
                reference.feedforward_angular_velocity_world
            ),
            scaled_feedforward_velocity_world=scaled_feedforward,
            scaled_feedforward_angular_velocity_world=scaled_angular_feedforward,
            control_mode=reference.control_mode,
            orientation_control_mode=reference.orientation_control_mode,
            speed_limited=limited,
            angular_speed_limited=angular_limited,
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


def _quat4(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (4,).")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must have non-zero length.")
    return (result / norm).copy()
