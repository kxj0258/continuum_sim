from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    load_motor_params_from_yaml,
    motor_position_to_tendon_delta,
    motor_velocity_to_tendon_velocity,
    tendon_delta_to_motor_position,
    tendon_velocity_to_motor_velocity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def _motor_params_with_direction(direction_sign: float) -> tuple[MotorParams, ...]:
    loaded = load_motor_params_from_yaml(ROBOT_CONFIG)
    return tuple(replace(motor, direction_sign=direction_sign) for motor in loaded)


def test_load_motor_params_from_yaml() -> None:
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)

    assert len(motor_params) == 9
    assert [motor.motor_index for motor in motor_params] == list(range(9))


def test_positive_direction_sign_maps_positive_position_to_positive_tendon_delta() -> None:
    motor_params = _motor_params_with_direction(1.0)
    motor_position = np.full(9, 0.25)

    tendon_delta = motor_position_to_tendon_delta(motor_position, motor_params)

    assert np.all(tendon_delta > 0.0)


def test_negative_direction_sign_maps_positive_position_to_negative_tendon_delta() -> None:
    motor_params = _motor_params_with_direction(-1.0)
    motor_position = np.full(9, 0.25)

    tendon_delta = motor_position_to_tendon_delta(motor_position, motor_params)

    assert np.all(tendon_delta < 0.0)


def test_tendon_delta_to_motor_position_round_trip() -> None:
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    tendon_delta = np.array(
        [
            -0.001,
            -0.0005,
            0.0,
            0.0005,
            0.001,
            -0.0015,
            0.002,
            -0.002,
            0.0012,
        ]
    )

    motor_position = tendon_delta_to_motor_position(tendon_delta, motor_params)
    recovered_tendon_delta = motor_position_to_tendon_delta(motor_position, motor_params)

    assert_allclose(recovered_tendon_delta, tendon_delta, atol=1.0e-14)


def test_motor_velocity_to_tendon_velocity_round_trip() -> None:
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    motor_velocity = np.array([0.0, 0.5, -0.3, 1.0, -1.2, 0.7, -0.8, 0.2, 1.4])

    tendon_velocity = motor_velocity_to_tendon_velocity(motor_velocity, motor_params)
    recovered_motor_velocity = tendon_velocity_to_motor_velocity(
        tendon_velocity, motor_params
    )

    assert_allclose(recovered_motor_velocity, motor_velocity, atol=1.0e-14)


def test_motor_mapping_rejects_wrong_shapes() -> None:
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)

    with np.testing.assert_raises(ValueError):
        motor_position_to_tendon_delta(np.zeros(8), motor_params)

    with np.testing.assert_raises(ValueError):
        tendon_delta_to_motor_position(np.zeros((9, 1)), motor_params)

    with np.testing.assert_raises(ValueError):
        motor_velocity_to_tendon_velocity(np.zeros(10), motor_params)

    with np.testing.assert_raises(ValueError):
        tendon_velocity_to_motor_velocity(np.zeros((1, 9)), motor_params)
