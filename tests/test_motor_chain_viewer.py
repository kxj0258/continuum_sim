from pathlib import Path

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

matplotlib.use("Agg")

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml
from continuum_sim.visualization.motor_chain_viewer import (
    MotorChainInteractiveViewer,
    compute_motor_chain_view_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def _robot_chain():
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    return params, physical_tendons, motor_params


def test_compute_motor_chain_view_data_zero_input() -> None:
    params, physical_tendons, motor_params = _robot_chain()

    view_data = compute_motor_chain_view_data(
        np.zeros(9),
        np.zeros(9),
        params,
        physical_tendons,
        motor_params,
        samples_per_segment=8,
    )

    assert_allclose(view_data.motor_position, np.zeros(9), atol=1.0e-14)
    assert_allclose(view_data.motor_velocity, np.zeros(9), atol=1.0e-14)
    assert_allclose(view_data.tendon_delta, np.zeros(9), atol=1.0e-14)
    assert_allclose(view_data.tendon_velocity, np.zeros(9), atol=1.0e-14)
    assert_allclose(view_data.q_est, np.zeros(9), atol=1.0e-14)
    assert_allclose(view_data.q_dot_est, np.zeros(9), atol=1.0e-14)
    assert view_data.centerline.shape == (22, 3)
    assert tuple(points.shape for points in view_data.segment_centerlines) == (
        (8, 3),
        (8, 3),
        (8, 3),
    )
    assert view_data.tip_position.shape == (3,)
    assert_allclose(view_data.tip_position[2], np.sum(params.segment_lengths), atol=1.0e-14)
    assert view_data.diagnostics["rank"] == 9
    assert view_data.diagnostics["is_full_rank"] is True


def test_compute_motor_chain_view_data_nonzero_position() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    motor_position = np.zeros(9)
    motor_position[0] = 0.25

    view_data = compute_motor_chain_view_data(
        motor_position,
        np.zeros(9),
        params,
        physical_tendons,
        motor_params,
        samples_per_segment=6,
    )

    assert np.linalg.norm(view_data.tendon_delta) > 0.0
    assert np.linalg.norm(view_data.q_est) > 0.0
    assert view_data.q_est.shape == (9,)
    assert view_data.centerline.shape == (16, 3)
    assert view_data.tip_position.shape == (3,)


def test_compute_motor_chain_view_data_nonzero_velocity() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    motor_velocity = np.zeros(9)
    motor_velocity[1] = 0.4

    view_data = compute_motor_chain_view_data(
        np.zeros(9),
        motor_velocity,
        params,
        physical_tendons,
        motor_params,
    )

    assert np.linalg.norm(view_data.tendon_velocity) > 0.0
    assert np.linalg.norm(view_data.q_dot_est) > 0.0
    assert_allclose(view_data.q_est, np.zeros(9), atol=1.0e-14)
    assert view_data.q_dot_est.shape == (9,)


def test_compute_motor_chain_view_data_rejects_wrong_shapes() -> None:
    params, physical_tendons, motor_params = _robot_chain()

    with pytest.raises(ValueError, match="motor_position"):
        compute_motor_chain_view_data(
            np.zeros(8),
            np.zeros(9),
            params,
            physical_tendons,
            motor_params,
        )

    with pytest.raises(ValueError, match="motor_velocity"):
        compute_motor_chain_view_data(
            np.zeros(9),
            np.zeros((9, 1)),
            params,
            physical_tendons,
            motor_params,
        )


def test_motor_chain_interactive_viewer_update_runs_without_gui_display() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    viewer = MotorChainInteractiveViewer(
        params,
        physical_tendons,
        motor_params,
        samples_per_segment=6,
    )
    try:
        zero_data = viewer.update_plot(redraw=False)
        assert_allclose(zero_data.q_est, np.zeros(9), atol=1.0e-14)

        motor_velocity = np.zeros(9)
        motor_velocity[0] = 0.5
        velocity_data = viewer.set_motor_state(np.zeros(9), motor_velocity)
        assert_allclose(velocity_data.q_est, np.zeros(9), atol=1.0e-14)
        assert np.linalg.norm(velocity_data.q_dot_est) > 0.0

        step_data = viewer.step(redraw=False)
        assert np.linalg.norm(step_data.motor_position) > 0.0
        assert np.linalg.norm(step_data.q_est) > 0.0
    finally:
        viewer.close()


def test_motor_chain_viewer_axes_stay_fixed_across_shape_updates() -> None:
    params, physical_tendons, motor_params = _robot_chain()
    viewer = MotorChainInteractiveViewer(
        params,
        physical_tendons,
        motor_params,
        samples_per_segment=6,
    )
    try:
        viewer.update_plot(redraw=False)
        initial_limits = (
            viewer.ax.get_xlim(),
            viewer.ax.get_ylim(),
            viewer.ax.get_zlim(),
        )

        motor_velocity = np.zeros(9)
        motor_velocity[0] = 0.5
        viewer.set_motor_state(np.zeros(9), motor_velocity)
        viewer.step(redraw=False)
        updated_limits = (
            viewer.ax.get_xlim(),
            viewer.ax.get_ylim(),
            viewer.ax.get_zlim(),
        )

        assert_allclose(updated_limits, initial_limits)
    finally:
        viewer.close()
