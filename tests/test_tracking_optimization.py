from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.backends.analytic_system_backend import AnalyticSystemBackend
from continuum_sim.control.scenario_controllers import WaypointTrackingController
from continuum_sim.control.task_intent import ObserverTaskIntent
from continuum_sim.control.whole_body_controller import (
    WholeBodyController,
    WholeBodyControllerConfig,
)
from continuum_sim.kinematics.whole_body import SingularityConfig
from continuum_sim.model.robot_assembly import load_robot_assembly_config


SINGLE_ASSEMBLY = "configs/robots/assemblies/single_spatial.yaml"
DUAL_ASSEMBLY = "configs/robots/assemblies/dual_spatial.yaml"


def test_waypoint_transition_reports_achieved_and_new_target_errors() -> None:
    assembly = load_robot_assembly_config(SINGLE_ASSEMBLY)
    state = AnalyticSystemBackend(assembly).reset_system()
    initial_tip = state.arms["executor"].tip_pose_world.position
    next_target = initial_tip + np.array([0.004, 0.0, 0.0])
    controller = WaypointTrackingController(
        assembly,
        np.vstack((initial_tip, next_target)),
        waypoint_tolerance_m=0.001,
        feedforward_speed_mps=0.002,
        max_target_speed_mps=0.01,
    )

    command = controller.compute_command(state)

    assert command.metadata["waypoint_advanced"] is True
    assert command.metadata["achieved_waypoint_index"] == 0
    assert command.metadata["achieved_waypoint_error_m"] == 0.0
    assert command.metadata["waypoint_index"] == 1
    assert_allclose(command.metadata["executor_error_m"], 0.004)
    assert_allclose(
        command.metadata["executor_feedforward_velocity_world"],
        np.zeros(3),
    )


def test_completion_command_retains_final_tracking_metadata() -> None:
    assembly = load_robot_assembly_config(SINGLE_ASSEMBLY)
    state = AnalyticSystemBackend(assembly).reset_system()
    initial_tip = state.arms["executor"].tip_pose_world.position
    controller = WaypointTrackingController(
        assembly,
        initial_tip[None, :],
        waypoint_tolerance_m=0.001,
    )

    command = controller.compute_command(state)

    assert controller.done is True
    assert command.metadata["tracking_complete"] is True
    assert command.metadata["achieved_waypoint_error_m"] == 0.0
    assert command.metadata["executor_error_m"] == 0.0
    assert_allclose(command.arms["executor"].tendon_rate_mps, 0.0)


def test_feedforward_velocity_is_clipped_with_feedback_target_speed() -> None:
    assembly = load_robot_assembly_config(SINGLE_ASSEMBLY)
    state = AnalyticSystemBackend(assembly).reset_system()
    initial_tip = state.arms["executor"].tip_pose_world.position
    controller = WaypointTrackingController(
        assembly,
        np.vstack(
            (
                initial_tip + np.array([0.01, 0.0, 0.0]),
                initial_tip + np.array([0.02, 0.0, 0.0]),
            )
        ),
        waypoint_tolerance_m=0.001,
        executor_position_gain=4.0,
        feedforward_speed_mps=0.02,
        max_target_speed_mps=0.01,
    )

    command = controller.compute_command(state)

    assert_allclose(
        command.metadata["executor_target_velocity_world"],
        [0.01, 0.0, 0.0],
    )


def test_fixed_base_singularity_protection_is_decoupled_per_arm() -> None:
    assembly = load_robot_assembly_config(DUAL_ASSEMBLY)
    controller = WholeBodyController(
        assembly,
        WholeBodyControllerConfig(
            singularity=SingularityConfig(minimum_singular_value=1.0e-5),
            decouple_arm_singularity=True,
        ),
    )
    jacobian = np.zeros((6, controller.layout.size), dtype=float)
    executor = controller.layout.arms["executor"]
    observer = controller.layout.arms["observer"]
    jacobian[:3, executor.start : executor.start + 3] = 1.0e-3 * np.eye(3)
    jacobian[3:, observer.start : observer.start + 3] = 1.0e-8 * np.eye(3)

    protection = controller._singularity_protection(jacobian)

    assert protection.arm_reports["executor"].velocity_scale == 1.0
    assert protection.arm_reports["observer"].velocity_scale < 0.1
    assert_allclose(protection.velocity_scale[executor], 1.0)
    assert np.max(protection.velocity_scale[observer]) < 0.1


def test_dual_executor_command_matches_single_with_collision_only_observer() -> None:
    single_assembly = load_robot_assembly_config(SINGLE_ASSEMBLY)
    dual_assembly = load_robot_assembly_config(DUAL_ASSEMBLY)
    single_state = AnalyticSystemBackend(single_assembly).reset_system()
    dual_state = AnalyticSystemBackend(dual_assembly).reset_system()
    target = single_state.arms["executor"].tip_pose_world.position + np.array(
        [0.006, -0.004, 0.003],
        dtype=float,
    )
    solver = WholeBodyControllerConfig(
        tendon_regularization_weight=0.5,
        enforce_tendon_rate_limits=True,
    )
    single = WaypointTrackingController(
        single_assembly,
        target[None, :],
        executor_position_gain=1.5,
        observer_position_gain=1.5,
        max_target_speed_mps=0.015,
        solver_config=solver,
        enforce_backend_tendon_limits=True,
    )
    dual = WaypointTrackingController(
        dual_assembly,
        target[None, :],
        observer_control_mode="collision_avoidance",
        executor_position_gain=1.5,
        observer_position_gain=1.5,
        max_target_speed_mps=0.015,
        solver_config=solver,
        enforce_backend_tendon_limits=True,
    )

    single_command = single.compute_command(single_state)
    dual_command = dual.compute_command(dual_state)

    assert_allclose(
        dual_command.arms["executor"].tendon_rate_mps,
        single_command.arms["executor"].tendon_rate_mps,
    )
    assert dual_command.metadata["inter_arm_executor_frozen"] is False
    assert dual_command.metadata["inter_arm_hard_stop"] is False
    assert dual_command.metadata["observer_control_mode"] == "collision_avoidance"
    assert np.max(np.abs(dual_command.arms["executor"].tendon_rate_mps)) <= 0.005
    assert np.max(np.abs(dual_command.arms["observer"].tendon_rate_mps)) <= 0.005


def test_observer_intent_accepts_collision_avoidance_mode() -> None:
    intent = ObserverTaskIntent(control_mode="collision_avoidance")

    assert intent.control_mode == "collision_avoidance"
