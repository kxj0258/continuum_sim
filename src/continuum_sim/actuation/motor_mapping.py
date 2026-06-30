"""Mappings between motor position/velocity and physical tendon length changes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml


@dataclass(frozen=True)
class MotorParams:
    """Motor-to-tendon transmission parameters for one actuator."""

    id: str
    motor_index: int
    tendon_global_index: int
    spool_radius: float
    gear_ratio: float
    direction_sign: float
    zero_position: float = 0.0


def load_motor_params_from_yaml(path: str | Path) -> tuple[MotorParams, ...]:
    """Load and validate motor parameters from the robot YAML file."""
    from continuum_sim.model.dual_arm_robot import is_dual_arm_robot_config, load_dual_arm_robot_config

    if is_dual_arm_robot_config(path):
        dual_config = load_dual_arm_robot_config(path)
        return tuple(
            MotorParams(
                id=motor.id,
                motor_index=index,
                tendon_global_index=index,
                spool_radius=motor.spool_radius,
                gear_ratio=motor.gear_ratio,
                direction_sign=motor.direction_sign,
                zero_position=motor.zero_position,
            )
            for index, motor in enumerate(dual_config.default_arm_motors)
        )
    config = load_yaml(path)
    robot = config.get("robot", {})
    expected_count = int(robot.get("total_tendon_count", 0))
    motor_items = config.get("motors", {}).get("items", [])
    if expected_count <= 0:
        expected_count = len(motor_items)
    if len(motor_items) != expected_count:
        raise ValueError(f"Expected {expected_count} motors, got {len(motor_items)}.")

    motor_params = tuple(_motor_params_from_dict(item) for item in motor_items)
    motor_params = tuple(sorted(motor_params, key=lambda motor: motor.motor_index))

    motor_indices = [motor.motor_index for motor in motor_params]
    expected_indices = list(range(expected_count))
    if motor_indices != expected_indices:
        raise ValueError(
            f"Expected motor_index values 0..{expected_count - 1}, got {motor_indices}."
        )
    tendon_indices = sorted(motor.tendon_global_index for motor in motor_params)
    if tendon_indices != expected_indices:
        raise ValueError(
            "Expected tendon_global_index values "
            f"0..{expected_count - 1}, got {tendon_indices}."
        )

    for motor in motor_params:
        if motor.spool_radius <= 0.0:
            raise ValueError(f"{motor.id} spool_radius must be > 0.")
        if motor.gear_ratio <= 0.0:
            raise ValueError(f"{motor.id} gear_ratio must be > 0.")
        if motor.direction_sign not in (-1.0, 1.0):
            raise ValueError(f"{motor.id} direction_sign must be +1.0 or -1.0.")

    return motor_params


def motor_position_to_tendon_delta(
    motor_position: np.ndarray,
    motor_params: tuple[MotorParams, ...],
) -> np.ndarray:
    """Map motor positions to physical tendon length deltas."""
    motor_position_array = _as_vector(
        motor_position,
        "motor_position",
        expected_size=len(motor_params),
    )
    zero_position = _zero_position(motor_params)
    scale = _tendon_per_motor_position(motor_params)
    tendon_delta = np.zeros(_tendon_count(motor_params), dtype=float)
    tendon_delta[_tendon_global_indices(motor_params)] = scale * (
        motor_position_array - zero_position
    )
    return tendon_delta


def tendon_delta_to_motor_position(
    tendon_delta: np.ndarray,
    motor_params: tuple[MotorParams, ...],
) -> np.ndarray:
    """Map physical tendon length deltas to motor positions."""
    tendon_delta_array = _as_vector(
        tendon_delta,
        "tendon_delta",
        expected_size=_tendon_count(motor_params),
    )
    zero_position = _zero_position(motor_params)
    scale = _tendon_per_motor_position(motor_params)
    return zero_position + tendon_delta_array[_tendon_global_indices(motor_params)] / scale


def motor_velocity_to_tendon_velocity(
    motor_velocity: np.ndarray,
    motor_params: tuple[MotorParams, ...],
) -> np.ndarray:
    """Map motor velocities to physical tendon length velocities."""
    motor_velocity_array = _as_vector(
        motor_velocity,
        "motor_velocity",
        expected_size=len(motor_params),
    )
    tendon_velocity = np.zeros(_tendon_count(motor_params), dtype=float)
    tendon_velocity[_tendon_global_indices(motor_params)] = (
        _tendon_per_motor_position(motor_params) * motor_velocity_array
    )
    return tendon_velocity


def tendon_velocity_to_motor_velocity(
    tendon_velocity: np.ndarray,
    motor_params: tuple[MotorParams, ...],
) -> np.ndarray:
    """Map physical tendon length velocities to motor velocities."""
    tendon_velocity_array = _as_vector(
        tendon_velocity,
        "tendon_velocity",
        expected_size=_tendon_count(motor_params),
    )
    return (
        tendon_velocity_array[_tendon_global_indices(motor_params)]
        / _tendon_per_motor_position(motor_params)
    )


def _motor_params_from_dict(item: dict[str, object]) -> MotorParams:
    return MotorParams(
        id=str(item["id"]),
        motor_index=int(item["motor_index"]),
        tendon_global_index=int(item["tendon_global_index"]),
        spool_radius=float(item["spool_radius"]),
        gear_ratio=float(item["gear_ratio"]),
        direction_sign=float(item["direction_sign"]),
        zero_position=float(item.get("zero_position", 0.0)),
    )


def _as_vector(values: np.ndarray, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array


def _validate_motor_params(motor_params: tuple[MotorParams, ...]) -> None:
    if not motor_params:
        raise ValueError("Expected at least one motor param.")
    motor_indices = [motor.motor_index for motor in motor_params]
    if motor_indices != list(range(len(motor_params))):
        raise ValueError(f"Expected contiguous motor_index values, got {motor_indices}.")
    tendon_indices = sorted(motor.tendon_global_index for motor in motor_params)
    if tendon_indices != list(range(_tendon_count(motor_params))):
        raise ValueError(
            "Expected tendon_global_index values to cover "
            f"0..{_tendon_count(motor_params) - 1}, got {tendon_indices}."
        )


def _zero_position(motor_params: tuple[MotorParams, ...]) -> np.ndarray:
    _validate_motor_params(motor_params)
    return np.array([motor.zero_position for motor in motor_params], dtype=float)


def _tendon_per_motor_position(motor_params: tuple[MotorParams, ...]) -> np.ndarray:
    _validate_motor_params(motor_params)
    return np.array(
        [
            motor.direction_sign * motor.spool_radius / motor.gear_ratio
            for motor in motor_params
        ],
        dtype=float,
    )


def _tendon_global_indices(motor_params: tuple[MotorParams, ...]) -> np.ndarray:
    _validate_motor_params(motor_params)
    return np.array([motor.tendon_global_index for motor in motor_params], dtype=int)


def _tendon_count(motor_params: tuple[MotorParams, ...]) -> int:
    if not motor_params:
        return 0
    return max(motor.tendon_global_index for motor in motor_params) + 1
