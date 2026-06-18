from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.kinematics.differential import (
    finite_difference_position_jacobian,
    motor_position_jacobian,
    motor_velocity_to_qdot_matrix,
    tip_position_from_q,
)
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def _robot_chain():
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    return params, physical_tendons, motor_params


def test_finite_difference_position_jacobian_shape_at_zero_q() -> None:
    params, _physical_tendons, _motor_params = _robot_chain()

    J = finite_difference_position_jacobian(np.zeros(9), params)

    assert J.shape == (3, 9)


def test_position_jacobian_predicts_small_q_step_near_zero() -> None:
    params, _physical_tendons, _motor_params = _robot_chain()
    q = np.zeros(9, dtype=float)
    q_dot = np.array([0.02, -0.01, 0.001, -0.015, 0.012, 0.0, 0.01, 0.006, -0.001])
    dt = 1.0e-3

    J = finite_difference_position_jacobian(q, params)
    actual_delta = tip_position_from_q(q + q_dot * dt, params) - tip_position_from_q(q, params)
    predicted_delta = J @ q_dot * dt

    assert_allclose(actual_delta, predicted_delta, atol=1.0e-7, rtol=1.0e-4)


def test_motor_velocity_to_qdot_matrix_has_expected_shape() -> None:
    params, physical_tendons, motor_params = _robot_chain()

    M_qm = motor_velocity_to_qdot_matrix(params, physical_tendons, motor_params)

    assert M_qm.shape == (9, 9)


def test_motor_position_jacobian_has_expected_shape() -> None:
    params, physical_tendons, motor_params = _robot_chain()

    J_motor = motor_position_jacobian(np.zeros(9), params, physical_tendons, motor_params)

    assert J_motor.shape == (3, 9)


def test_differential_kinematics_rejects_wrong_q_shape_and_step() -> None:
    params, physical_tendons, motor_params = _robot_chain()

    with pytest.raises(ValueError, match="q"):
        finite_difference_position_jacobian(np.zeros((9, 1)), params)

    with pytest.raises(ValueError, match="step"):
        finite_difference_position_jacobian(np.zeros(9), params, step=0.0)

    with pytest.raises(ValueError, match="step"):
        motor_position_jacobian(
            np.zeros(9),
            params,
            physical_tendons,
            motor_params,
            step=-1.0e-5,
        )


def test_motor_velocity_to_qdot_matrix_rejects_invalid_motor_params() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    invalid_motor_params = tuple(
        replace(motor, motor_index=0) if index == 1 else motor
        for index, motor in enumerate(motor_params)
    )

    with pytest.raises(ValueError, match="Duplicate motor_index"):
        motor_velocity_to_qdot_matrix(params, physical_tendons, invalid_motor_params)
