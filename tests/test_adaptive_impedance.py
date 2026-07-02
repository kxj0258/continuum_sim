from pathlib import Path

import numpy as np

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.actuation.motor_mapping import motor_velocity_to_tendon_velocity
from continuum_sim.control.adaptive_impedance import (
    AdaptiveImpedanceConfig,
    compute_dynamic_wiping_motor_velocity_command_from_state,
)
from continuum_sim.model import (
    BendingSpaceModel,
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
)
from continuum_sim.scenes import load_navigation_scene_config
from continuum_sim.tasks import build_raster_wiping_path, load_mujoco_wiping_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_wiping_board.yaml"
DYNAMIC_TASK_CONFIG = (
    PROJECT_ROOT / "configs" / "tasks" / "mujoco_wiping_board_dynamic.yaml"
)


def test_dynamic_impedance_controller_returns_motor_command_and_prediction() -> None:
    task_config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(task_config.scene.config_path)
    surface = scene.work_surface(task_config.motion.surface_id)
    params = ThreeSegmentRobotParams.from_yaml(task_config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(task_config.robot_config_path)
    motor_params = load_motor_params_from_yaml(task_config.robot_config_path)
    q = np.zeros(params.q_size, dtype=float)
    qdot = np.zeros(params.q_size, dtype=float)
    tip = np.array([0.0, 0.0, 0.12], dtype=float)

    command, info = compute_dynamic_wiping_motor_velocity_command_from_state(
        tip,
        q,
        qdot,
        target_position=tip + 0.002 * surface.tangent_u,
        surface=surface,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        wiping_config=task_config.controller,
        adaptive_config=AdaptiveImpedanceConfig.default(params),
        measured_normal_force_n=task_config.controller.target_normal_force_n,
        dt=task_config.simulation.dt,
    )

    assert command.shape == (len(motor_params),)
    assert np.all(np.isfinite(command))
    assert info["predicted_q"].shape == (params.q_size,)
    assert info["stiffness_diag"].shape == (params.q_size,)
    assert BendingSpaceModel.from_arm(params, physical_tendons).is_compatible(
        motor_velocity_to_tendon_velocity(command, motor_params)
    )


def test_dynamic_impedance_controller_removes_axial_strain_dofs() -> None:
    task_config = load_mujoco_wiping_config(DYNAMIC_TASK_CONFIG)
    scene = load_navigation_scene_config(task_config.scene.config_path)
    surface = scene.work_surface(task_config.motion.surface_id)
    wipe_path = build_raster_wiping_path(
        task_config.motion,
        surface,
        contact_radius_m=task_config.tool.radius_m,
    )
    params = ThreeSegmentRobotParams.from_yaml(task_config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(task_config.robot_config_path)
    motor_params = load_motor_params_from_yaml(task_config.robot_config_path)
    q = np.zeros(params.q_size, dtype=float)
    qdot = np.zeros(params.q_size, dtype=float)
    tip = np.array([0.0, 0.0, 0.12], dtype=float)

    command, info = compute_dynamic_wiping_motor_velocity_command_from_state(
        tip,
        q,
        qdot,
        target_position=wipe_path.target_position[0],
        surface=surface,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        wiping_config=task_config.controller,
        adaptive_config=AdaptiveImpedanceConfig.default(params),
        measured_normal_force_n=0.0,
        dt=task_config.simulation.dt,
        contact_radius_m=task_config.tool.radius_m,
        force_control_enabled=False,
    )

    assert np.all(np.isfinite(command))
    assert np.allclose(info["predicted_q"][2::3], 0.0, atol=1.0e-12)
    assert np.allclose(info["predicted_qdot"][2::3], 0.0, atol=1.0e-12)
    assert np.allclose(info["predicted_qddot"][2::3], 0.0, atol=1.0e-12)
