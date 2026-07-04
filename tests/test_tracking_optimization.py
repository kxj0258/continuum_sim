from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.backends.analytic_system_backend import AnalyticSystemBackend
from continuum_sim.control.scenario_controllers import WaypointTrackingController
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
