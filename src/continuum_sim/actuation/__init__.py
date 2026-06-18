"""Actuation mappings for physical tendon-driven hardware."""

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    load_motor_params_from_yaml,
    motor_position_to_tendon_delta,
    motor_velocity_to_tendon_velocity,
    tendon_delta_to_motor_position,
    tendon_velocity_to_motor_velocity,
)

__all__ = [
    "MotorParams",
    "load_motor_params_from_yaml",
    "motor_position_to_tendon_delta",
    "motor_velocity_to_tendon_velocity",
    "tendon_delta_to_motor_position",
    "tendon_velocity_to_motor_velocity",
]
