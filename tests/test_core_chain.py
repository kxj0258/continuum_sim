from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.actuation.motor_mapping import (
    load_motor_params_from_yaml,
    motor_position_to_tendon_delta,
)
from continuum_sim.control import DifferentialIKConfig, simulate_position_tracking
from continuum_sim.kinematics import ContinuumKinematicsChain
from continuum_sim.kinematics.differential import motor_position_jacobian, tip_position_from_q
from continuum_sim.model import (
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
    physical_tendon_delta_to_q,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def test_zero_motor_maps_to_zero_q_and_straight_tip() -> None:
    chain = ContinuumKinematicsChain.from_robot_config(ROBOT_CONFIG)
    motor_position = np.zeros(chain.motor_size, dtype=float)

    q = chain.motor_position_to_q(motor_position)
    tip = chain.tip_position_from_motor(motor_position)

    assert_allclose(q, np.zeros(chain.q_size), atol=1.0e-14)
    assert_allclose(tip, [0.0, 0.0, 0.12], atol=1.0e-12)


def test_single_motor_input_produces_finite_q_and_tip() -> None:
    chain = ContinuumKinematicsChain.from_robot_config(ROBOT_CONFIG)
    motor_position = np.zeros(chain.motor_size, dtype=float)
    motor_position[0] = 0.25

    q = chain.motor_position_to_q(motor_position)
    tip = chain.tip_position_from_motor(motor_position)

    assert q.shape == (chain.q_size,)
    assert tip.shape == (3,)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(tip))
    assert np.linalg.norm(q) > 0.0


def test_motor_position_jacobian_shape() -> None:
    chain = ContinuumKinematicsChain.from_robot_config(ROBOT_CONFIG)

    J_motor = chain.motor_position_jacobian(np.zeros(chain.motor_size, dtype=float))

    assert J_motor.shape == (3, chain.motor_size)


def test_core_chain_matches_existing_manual_chain() -> None:
    chain = ContinuumKinematicsChain.from_robot_config(ROBOT_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    motor_position = np.array(
        [0.05, -0.02, 0.03, -0.04, 0.01, -0.03, 0.02, -0.01, 0.04],
        dtype=float,
    )

    expected_tendon_delta = motor_position_to_tendon_delta(motor_position, motor_params)
    expected_q = physical_tendon_delta_to_q(expected_tendon_delta, params, physical_tendons)
    expected_tip = tip_position_from_q(expected_q, params)
    expected_jacobian = motor_position_jacobian(
        expected_q,
        params,
        physical_tendons,
        motor_params,
    )

    assert_allclose(chain.motor_position_to_tendon_delta(motor_position), expected_tendon_delta)
    assert_allclose(chain.motor_position_to_q(motor_position), expected_q)
    assert_allclose(chain.tip_position_from_motor(motor_position), expected_tip)
    assert_allclose(chain.motor_position_jacobian(motor_position), expected_jacobian)


def test_core_chain_simulate_tracking_matches_existing_tracker() -> None:
    chain = ContinuumKinematicsChain.from_robot_config(ROBOT_CONFIG)
    initial_motor_position = np.zeros(chain.motor_size, dtype=float)
    zero_tip = chain.tip_position_from_motor(initial_motor_position)
    target_positions = np.array([zero_tip + np.array([0.001, 0.0, 0.0])], dtype=float)
    config = DifferentialIKConfig(max_steps=5)

    result = chain.simulate_tracking(initial_motor_position, target_positions, config)
    expected = simulate_position_tracking(
        initial_motor_position,
        target_positions,
        chain.params,
        chain.physical_tendons,
        chain.motor_params,
        config,
    )

    assert_allclose(result.time, expected.time)
    assert_allclose(result.tip_position, expected.tip_position)
    assert_allclose(result.motor_position, expected.motor_position)
    assert_allclose(result.q_est, expected.q_est)
