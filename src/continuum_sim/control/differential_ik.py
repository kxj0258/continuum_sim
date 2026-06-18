"""Damped-least-squares position tracking for the offline PCC chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    motor_position_to_tendon_delta,
)
from continuum_sim.kinematics.differential import (
    motor_position_jacobian,
    tip_position_from_q,
)
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import physical_tendon_delta_to_q


@dataclass(frozen=True)
class DifferentialIKConfig:
    """Configuration for offline differential IK position tracking."""

    dt: float = 0.02
    damping: float = 1.0e-3
    position_gain: float = 4.0
    max_motor_velocity_rad_s: float = 1.0
    position_tolerance_m: float = 1.0e-3
    max_steps: int = 1000


@dataclass(frozen=True)
class TrackingResult:
    """Recorded samples from an offline position-tracking rollout."""

    time: np.ndarray
    target_position: np.ndarray
    tip_position: np.ndarray
    error_norm: np.ndarray
    motor_position: np.ndarray
    motor_velocity: np.ndarray
    q_est: np.ndarray


def damped_least_squares(
    J: np.ndarray,
    velocity: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Solve x = J.T @ inv(J @ J.T + damping**2 I) @ velocity."""
    J_array = np.asarray(J, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    if J_array.ndim != 2:
        raise ValueError(f"Expected J to be 2D, got shape {J_array.shape}.")
    if velocity_array.shape != (J_array.shape[0],):
        raise ValueError(
            f"Expected velocity with shape ({J_array.shape[0]},), "
            f"got {velocity_array.shape}."
        )
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}.")

    lhs = J_array @ J_array.T + damping**2 * np.eye(J_array.shape[0], dtype=float)
    y = np.linalg.solve(lhs, velocity_array)
    return J_array.T @ y


def compute_motor_velocity_command(
    motor_position: np.ndarray,
    target_position: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: DifferentialIKConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Compute a clipped motor velocity command for one target tip position."""
    motor_position_array = _as_motor_vector(
        motor_position,
        "motor_position",
        expected_size=len(motor_params),
    )
    target_position_array = _as_position(target_position, "target_position")
    _validate_config(config)

    tendon_delta = motor_position_to_tendon_delta(motor_position_array, motor_params)
    q_est = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)
    tip_position = tip_position_from_q(q_est, params)
    return _compute_motor_velocity_command_from_state(
        tip_position=tip_position,
        q_est=q_est,
        target_position=target_position_array,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        config=config,
    )

def compute_motor_velocity_command_from_observation(
    actual_tip_position: np.ndarray,
    actual_tendon_delta: np.ndarray,
    target_position: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: DifferentialIKConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Compute motor velocity from observed MuJoCo tip position and tendon lengths."""
    actual_tip_position_array = _as_position(actual_tip_position, "actual_tip_position")
    actual_tendon_delta_array = _as_tendon_vector(
        actual_tendon_delta,
        "actual_tendon_delta",
        expected_size=len(physical_tendons),
    )
    target_position_array = _as_position(target_position, "target_position")
    _validate_config(config)

    q_est = physical_tendon_delta_to_q(
        actual_tendon_delta_array,
        params,
        physical_tendons,
    )
    return _compute_motor_velocity_command_from_state(
        tip_position=actual_tip_position_array,
        q_est=q_est,
        target_position=target_position_array,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        config=config,
    )


def _compute_motor_velocity_command_from_state(
    *,
    tip_position: np.ndarray,
    q_est: np.ndarray,
    target_position: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: DifferentialIKConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    position_error = target_position - tip_position
    desired_tip_velocity = config.position_gain * position_error
    J_motor = motor_position_jacobian(q_est, params, physical_tendons, motor_params)
    motor_velocity_cmd = damped_least_squares(
        J_motor,
        desired_tip_velocity,
        config.damping,
    )
    motor_velocity_cmd = np.clip(
        motor_velocity_cmd,
        -config.max_motor_velocity_rad_s,
        config.max_motor_velocity_rad_s,
    )

    info: dict[str, np.ndarray | float] = {
        "q_est": np.asarray(q_est, dtype=float).copy(),
        "tip_position": np.asarray(tip_position, dtype=float).copy(),
        "position_error": position_error,
        "error_norm": float(np.linalg.norm(position_error)),
        "desired_tip_velocity": desired_tip_velocity,
        "J_motor": J_motor,
    }
    return motor_velocity_cmd, info


def simulate_position_tracking(
    initial_motor_position: np.ndarray,
    target_positions: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    config: DifferentialIKConfig,
    position_limit_rad: float = 2.0,
    stop_on_completion: bool = False,
) -> TrackingResult:
    """Simulate waypoint tracking by integrating motor velocity commands."""
    motor_position = _as_motor_vector(
        initial_motor_position,
        "initial_motor_position",
        expected_size=len(motor_params),
    ).copy()
    targets = _as_target_positions(target_positions)
    _validate_config(config)
    if position_limit_rad <= 0.0:
        raise ValueError(f"position_limit_rad must be positive, got {position_limit_rad}.")

    waypoint_index = 0
    times: list[float] = []
    target_history: list[np.ndarray] = []
    tip_history: list[np.ndarray] = []
    error_history: list[float] = []
    motor_position_history: list[np.ndarray] = []
    motor_velocity_history: list[np.ndarray] = []
    q_history: list[np.ndarray] = []

    for step_index in range(config.max_steps):
        current_target = targets[waypoint_index]
        motor_velocity_cmd, info = compute_motor_velocity_command(
            motor_position,
            current_target,
            params,
            physical_tendons,
            motor_params,
            config,
        )

        times.append(step_index * config.dt)
        target_history.append(current_target.copy())
        tip_history.append(np.asarray(info["tip_position"], dtype=float).copy())
        error_norm = float(info["error_norm"])
        error_history.append(error_norm)
        motor_position_history.append(motor_position.copy())
        motor_velocity_history.append(motor_velocity_cmd.copy())
        q_history.append(np.asarray(info["q_est"], dtype=float).copy())

        if error_norm <= config.position_tolerance_m:
            if waypoint_index < len(targets) - 1:
                waypoint_index += 1
            elif stop_on_completion:
                break

        motor_position = motor_position + motor_velocity_cmd * config.dt
        motor_position = np.clip(motor_position, -position_limit_rad, position_limit_rad)

    return TrackingResult(
        time=np.asarray(times, dtype=float),
        target_position=np.asarray(target_history, dtype=float),
        tip_position=np.asarray(tip_history, dtype=float),
        error_norm=np.asarray(error_history, dtype=float),
        motor_position=np.asarray(motor_position_history, dtype=float),
        motor_velocity=np.asarray(motor_velocity_history, dtype=float),
        q_est=np.asarray(q_history, dtype=float),
    )


def _validate_config(config: DifferentialIKConfig) -> None:
    if config.dt <= 0.0:
        raise ValueError(f"dt must be positive, got {config.dt}.")
    if config.damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {config.damping}.")
    if config.position_gain < 0.0:
        raise ValueError(f"position_gain must be non-negative, got {config.position_gain}.")
    if config.max_motor_velocity_rad_s <= 0.0:
        raise ValueError(
            "max_motor_velocity_rad_s must be positive, "
            f"got {config.max_motor_velocity_rad_s}."
        )
    if config.position_tolerance_m < 0.0:
        raise ValueError(
            f"position_tolerance_m must be non-negative, got {config.position_tolerance_m}."
        )
    if config.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {config.max_steps}.")


def _as_motor_vector(values: np.ndarray, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array


def _as_tendon_vector(values: np.ndarray, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array


def _as_position(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected {name} with shape (3,), got {array.shape}.")
    return array


def _as_target_positions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected target_positions with shape (N, 3), got {array.shape}.")
    if array.shape[0] < 1:
        raise ValueError("target_positions must contain at least one row.")
    return array
