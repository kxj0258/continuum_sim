from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.control import (
    contact_measurement_from_surface_proxy,
    compute_wiping_motor_velocity_command_from_state,
    desired_hybrid_tip_velocity,
)
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml
from continuum_sim.scenes import load_navigation_scene_config
from continuum_sim.tasks import load_mujoco_wiping_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_wiping_board.yaml"


def test_hybrid_velocity_splits_tangent_tracking_and_normal_force() -> None:
    config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(config.scene.config_path)
    surface = scene.work_surface(config.motion.surface_id)
    controller = replace(
        config.controller,
        tangent_position_gain=2.0,
        normal_position_gain=0.0,
        normal_force_gain=0.004,
        max_tangent_velocity_m_s=1.0,
        max_normal_velocity_m_s=1.0,
    )
    tip = surface.center_m.copy()
    target = tip + 0.003 * surface.tangent_u - 0.002 * surface.tangent_v

    velocity, info = desired_hybrid_tip_velocity(
        tip,
        target,
        surface,
        controller,
        measured_normal_force_n=0.5,
    )

    tangent_velocity = velocity - np.dot(velocity, surface.normal) * surface.normal
    assert_allclose(tangent_velocity, 2.0 * (target - tip))
    assert np.dot(velocity, surface.normal) < 0.0
    assert info["force_error_n"] == controller.target_normal_force_n - 0.5
    assert info["contact_source"] == "mujoco_contact_force"


def test_contact_proxy_uses_pad_surface_not_pad_center() -> None:
    config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(config.scene.config_path)
    surface = scene.work_surface(config.motion.surface_id)
    pad_center = surface.center_m + 0.003 * surface.normal

    contact = contact_measurement_from_surface_proxy(
        pad_center,
        surface,
        config.controller,
        contact_radius_m=0.005,
    )

    assert_allclose(contact.signed_distance_m, -0.002)
    assert_allclose(contact.normal_force_n, 1.2)
    assert contact.in_contact is True


def test_hybrid_controller_returns_nine_motor_velocities() -> None:
    config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(config.scene.config_path)
    surface = scene.work_surface(config.motion.surface_id)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(config.robot_config_path)
    motor_params = load_motor_params_from_yaml(config.robot_config_path)
    controller = replace(config.controller, normal_position_gain=0.0)
    q_est = np.zeros(params.q_size, dtype=float)
    tip = np.array([0.0, 0.0, 0.12], dtype=float)
    target = tip + 0.001 * surface.tangent_u

    motor_velocity, info = compute_wiping_motor_velocity_command_from_state(
        tip,
        q_est,
        target,
        surface,
        params,
        physical_tendons,
        motor_params,
        controller,
        measured_normal_force_n=controller.target_normal_force_n,
    )

    assert motor_velocity.shape == (9,)
    assert np.all(np.isfinite(motor_velocity))
    assert np.max(np.abs(motor_velocity)) <= controller.max_motor_velocity_rad_s
    assert_allclose(info["normal_velocity"], np.zeros(3, dtype=float))
