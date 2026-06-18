from pathlib import Path

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.actuation.motor_mapping import motor_position_to_tendon_delta
from continuum_sim.control import (
    DifferentialIKConfig,
    compute_motor_velocity_command,
    compute_motor_velocity_command_from_observation,
    damped_least_squares,
    simulate_position_tracking,
)
from continuum_sim.kinematics.differential import tip_position_from_q
from continuum_sim.model import (
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
    physical_tendon_delta_to_q,
)
from continuum_sim.visualization.trajectory_tracking_viewer import (
    animate_tracking_result,
    make_circle_trajectory,
    plot_tracking_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def _robot_chain():
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    return params, physical_tendons, motor_params


def test_damped_least_squares_solves_simple_matrix() -> None:
    J = np.array([[2.0, 0.0], [0.0, 4.0]])
    velocity = np.array([1.0, 2.0])

    result = damped_least_squares(J, velocity, damping=0.0)

    assert_allclose(result, np.array([0.5, 0.5]), atol=1.0e-14)


def test_damped_least_squares_rejects_negative_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        damped_least_squares(np.eye(2), np.ones(2), damping=-1.0e-3)


def test_compute_motor_velocity_command_shape_and_limit() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    zero_tip = tip_position_from_q(np.zeros(9), params)
    target = zero_tip + np.array([0.002, 0.0, 0.0])
    config = DifferentialIKConfig(max_motor_velocity_rad_s=0.05)

    command, info = compute_motor_velocity_command(
        np.zeros(9),
        target,
        params,
        physical_tendons,
        motor_params,
        config,
    )

    assert command.shape == (9,)
    assert np.max(np.abs(command)) <= config.max_motor_velocity_rad_s
    assert info["q_est"].shape == (9,)
    assert info["tip_position"].shape == (3,)
    assert info["position_error"].shape == (3,)
    assert info["desired_tip_velocity"].shape == (3,)
    assert info["J_motor"].shape == (3, 9)
    assert float(info["error_norm"]) > 0.0


def test_compute_motor_velocity_command_from_observation_uses_actual_tip_and_tendon_delta() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    config = DifferentialIKConfig(max_motor_velocity_rad_s=0.05)
    actual_motor_position = np.array(
        [0.05, -0.02, 0.03, -0.04, 0.01, -0.03, 0.02, -0.01, 0.04],
        dtype=float,
    )
    actual_tendon_delta = motor_position_to_tendon_delta(actual_motor_position, motor_params)
    expected_q_est = physical_tendon_delta_to_q(actual_tendon_delta, params, physical_tendons)
    modeled_tip = tip_position_from_q(expected_q_est, params)
    actual_tip = modeled_tip + np.array([0.002, -0.001, 0.0015], dtype=float)
    target = actual_tip + np.array([0.001, 0.002, -0.001], dtype=float)

    command, info = compute_motor_velocity_command_from_observation(
        actual_tip,
        actual_tendon_delta,
        target,
        params,
        physical_tendons,
        motor_params,
        config,
    )

    assert command.shape == (9,)
    assert np.max(np.abs(command)) <= config.max_motor_velocity_rad_s
    assert_allclose(np.asarray(info["q_est"], dtype=float), expected_q_est, atol=1.0e-12)
    assert_allclose(np.asarray(info["tip_position"], dtype=float), actual_tip, atol=1.0e-12)
    assert_allclose(
        np.asarray(info["position_error"], dtype=float),
        target - actual_tip,
        atol=1.0e-12,
    )
    assert float(info["error_norm"]) == pytest.approx(np.linalg.norm(target - actual_tip))


def test_simulate_position_tracking_reduces_error_for_small_target() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    zero_tip = tip_position_from_q(np.zeros(9), params)
    target = zero_tip + np.array([0.001, 0.001, -0.001])
    config = DifferentialIKConfig(max_steps=150, position_gain=4.0)

    result = simulate_position_tracking(
        np.zeros(9),
        np.array([target]),
        params,
        physical_tendons,
        motor_params,
        config,
        position_limit_rad=2.0,
    )

    assert result.error_norm[-1] < result.error_norm[0]
    assert result.error_norm[-1] < 1.0e-4


def test_tracking_result_shapes_are_consistent() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    zero_tip = tip_position_from_q(np.zeros(9), params)
    targets = make_circle_trajectory(zero_tip[:2], radius=0.001, z=zero_tip[2], samples=5)
    config = DifferentialIKConfig(max_steps=12)

    result = simulate_position_tracking(
        np.zeros(9),
        targets,
        params,
        physical_tendons,
        motor_params,
        config,
    )

    assert result.time.shape == (config.max_steps,)
    assert result.target_position.shape == (config.max_steps, 3)
    assert result.tip_position.shape == (config.max_steps, 3)
    assert result.error_norm.shape == (config.max_steps,)
    assert result.motor_position.shape == (config.max_steps, 9)
    assert result.motor_velocity.shape == (config.max_steps, 9)
    assert result.q_est.shape == (config.max_steps, 9)


def test_simulate_position_tracking_can_stop_on_completion() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    zero_tip = tip_position_from_q(np.zeros(9), params)
    config = DifferentialIKConfig(max_steps=20)

    result = simulate_position_tracking(
        np.zeros(9),
        np.array([zero_tip]),
        params,
        physical_tendons,
        motor_params,
        config,
        stop_on_completion=True,
    )

    assert result.time.shape == (1,)
    assert_allclose(result.error_norm, np.zeros(1), atol=1.0e-14)


def test_plot_tracking_result_smoke() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    zero_tip = tip_position_from_q(np.zeros(9), params)
    target = zero_tip + np.array([0.001, 0.0, 0.0])
    config = DifferentialIKConfig(max_steps=8)
    result = simulate_position_tracking(
        np.zeros(9),
        np.array([target]),
        params,
        physical_tendons,
        motor_params,
        config,
    )

    fig = plot_tracking_result(result, params)
    try:
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)


def test_animate_tracking_result_smoke() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    zero_tip = tip_position_from_q(np.zeros(9), params)
    target = zero_tip + np.array([0.001, 0.0, 0.0])
    config = DifferentialIKConfig(max_steps=5)
    result = simulate_position_tracking(
        np.zeros(9),
        np.array([target]),
        params,
        physical_tendons,
        motor_params,
        config,
    )

    fig, anim = animate_tracking_result(
        result,
        params,
        samples_per_segment=5,
        interval_ms=10,
        stride=2,
    )
    try:
        assert len(fig.axes) == 1
        assert anim is not None
        x_span = fig.axes[0].get_xlim()[1] - fig.axes[0].get_xlim()[0]
        y_span = fig.axes[0].get_ylim()[1] - fig.axes[0].get_ylim()[0]
        z_span = fig.axes[0].get_zlim()[1] - fig.axes[0].get_zlim()[0]
        assert_allclose([x_span, y_span], [z_span, z_span], rtol=1.0e-12, atol=1.0e-12)
        fig.canvas.draw()
    finally:
        plt.close(fig)
