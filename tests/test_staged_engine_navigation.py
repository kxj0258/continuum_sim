from __future__ import annotations

from dataclasses import replace

import numpy as np

from continuum_sim.control.mobile_base_pose_control import MobileBasePoseController
from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
)
from continuum_sim.control.staged_engine_navigation import (
    StagedEngineNavigationController,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.system.types import (
    ArmSystemState,
    BaseSystemState,
    RobotSystemState,
)
from continuum_sim.tasks.engine_navigation import (
    EngineNavigationLocalPathPlan,
    EngineNavigationPlan,
    EngineNavigationSpec,
)


def test_mobile_base_pose_controller_limits_linear_and_angular_norms() -> None:
    controller = MobileBasePoseController(
        position_gain=2.0,
        orientation_gain=2.0,
    )

    twist, position_error, orientation_error = controller.compute_twist(
        Pose6D.identity(),
        Pose6D.from_rpy_rad(
            position=(1.0, 0.0, 0.0),
            rpy_rad=(0.0, 0.0, 1.0),
        ),
        max_linear_speed=0.05,
        max_angular_speed=0.30,
    )

    assert np.linalg.norm(twist[:3]) <= 0.05
    assert np.linalg.norm(twist[3:]) <= 0.30
    assert position_error == 1.0
    assert orientation_error > 0.9


def test_staged_controller_holds_arms_during_base_motion() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/dual_spatial_mobile.yaml"
    )
    target = Pose6D.from_rpy_rad(position=(0.1, 0.0, 0.0))
    plan = _minimal_plan(target)
    controller = StagedEngineNavigationController(
        assembly,
        plan,
        EngineNavigationSpec.from_mapping(
            {
                "entry_region": "entry_port",
                "insertion_path": "nozzle_axis_entry",
            }
        ),
        scene_query=None,
        waypoint_tolerance_m=0.003,
        min_clearance_m=0.01,
        terminate_on_clearance_violation=True,
    )

    command = controller.compute_command(_state(assembly, Pose6D.identity()))

    assert controller.phase == "base_approach"
    assert np.linalg.norm(command.base_twist_world[:3]) > 0.0
    assert all(
        np.allclose(arm.tendon_rate_mps, 0.0)
        for arm in command.arms.values()
    )
    assert command.metadata["engine_navigation_active_target_kind"] == "base"
    np.testing.assert_allclose(
        command.metadata["engine_navigation_active_target_m"],
        target.position,
    )
    np.testing.assert_allclose(
        command.metadata["engine_navigation_pre_entry_target_m"],
        plan.pre_entry_tip_world,
    )
    np.testing.assert_allclose(
        command.metadata["engine_navigation_base_path_m"],
        np.asarray(
            [
                np.zeros(3),
                plan.pre_entry_base_pose.position,
                *(pose.position for pose in plan.insertion_base_poses),
            ]
        ),
    )
    np.testing.assert_allclose(
        command.metadata["engine_navigation_insertion_path_m"],
        plan.insertion_tip_waypoints_world,
    )
    np.testing.assert_allclose(
        command.metadata["engine_navigation_executor_path_m"],
        plan.executor_waypoints_world,
    )
    np.testing.assert_allclose(
        command.metadata["engine_navigation_observer_roi_m"],
        plan.observer_roi_world,
    )


def test_staged_controller_holds_base_during_executor_navigation() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/dual_spatial_mobile.yaml"
    )
    target = Pose6D.from_rpy_rad(position=(0.1, 0.0, 0.0))
    plan = _minimal_plan(target)
    controller = StagedEngineNavigationController(
        assembly,
        plan,
        EngineNavigationSpec.from_mapping(
            {
                "entry_region": "entry_port",
                "insertion_path": "nozzle_axis_entry",
            }
        ),
        scene_query=None,
        waypoint_tolerance_m=0.003,
        min_clearance_m=0.01,
        terminate_on_clearance_violation=True,
    )
    state = _state(assembly, target)

    controller.compute_command(state)
    command = controller.compute_command(state)

    assert controller.phase == "executor_navigation"
    np.testing.assert_allclose(command.base_twist_world, 0.0)
    assert command.metadata["engine_navigation_active_target_kind"] == "executor"
    np.testing.assert_allclose(
        command.metadata["engine_navigation_active_target_m"],
        command.metadata["executor_target_world"],
    )


def test_staged_controller_rejoins_before_resuming_base_insertion() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/dual_spatial_mobile.yaml"
    )
    base_pose = Pose6D.identity()
    straight_state = _state(assembly, base_pose)
    straight_tip = straight_state.arms["executor"].tip_pose_world.position
    local_target = straight_tip + np.array([0.001, 0.0, -0.001])
    next_base_pose = Pose6D.from_rpy_rad(position=(0.01, 0.0, 0.0))
    intermediate = EngineNavigationLocalPathPlan(
        name="one_third_circle",
        path_type="transverse_circle",
        at_fraction=1.0 / 3.0,
        insertion_index=0,
        insertion_target_world=straight_tip,
        center_world=local_target,
        waypoints_world=local_target[None, :],
        is_terminal=False,
        transition_waypoints_world=np.asarray(
            [
                straight_tip,
                0.5 * (straight_tip + local_target),
                local_target,
            ]
        ),
    )
    terminal = EngineNavigationLocalPathPlan(
        name="endpoint_square",
        path_type="transverse_square",
        at_fraction=1.0,
        insertion_index=1,
        insertion_target_world=straight_tip,
        center_world=local_target,
        waypoints_world=local_target[None, :],
        is_terminal=True,
    )
    plan = EngineNavigationPlan(
        pre_entry_tip_world=straight_tip,
        insertion_direction_world=np.array([0.0, 0.0, 1.0]),
        insertion_tip_waypoints_world=np.asarray([straight_tip, straight_tip]),
        pre_entry_base_pose=base_pose,
        insertion_base_poses=(base_pose, next_base_pose),
        executor_waypoints_world=terminal.waypoints_world,
        observer_roi_world=terminal.center_world,
        local_path_plans=(intermediate, terminal),
    )
    controller = StagedEngineNavigationController(
        assembly,
        plan,
        EngineNavigationSpec.from_mapping(
            {
                "entry_region": "entry_port",
                "insertion_path": "nozzle_axis_entry",
            }
        ),
        scene_query=None,
        waypoint_tolerance_m=0.0001,
        min_clearance_m=0.01,
        terminate_on_clearance_violation=True,
    )

    controller.compute_command(straight_state)
    path_command = controller.compute_command(straight_state)

    assert controller.phase == "executor_navigation"
    assert path_command.metadata["engine_navigation_local_path_name"] == (
        "one_third_circle"
    )
    assert path_command.metadata["engine_navigation_executor_subphase"] == "path"
    np.testing.assert_allclose(path_command.base_twist_world, 0.0)
    assert controller._tracking is not None
    np.testing.assert_allclose(
        controller._tracking.waypoints_world,
        intermediate.transition_waypoints_world,
    )

    local_state = _state(
        assembly,
        base_pose,
        tip_position=local_target,
    )
    rejoin_command = controller.compute_command(local_state)

    assert controller.phase == "executor_navigation"
    assert rejoin_command.metadata["engine_navigation_executor_subphase"] == (
        "rejoin"
    )
    np.testing.assert_allclose(
        rejoin_command.metadata["executor_target_world"],
        straight_tip,
    )

    controller.compute_command(straight_state)

    assert controller.phase == "base_insertion"
    assert controller.insertion_index == 1


def test_engine_local_tracking_mode_does_not_apply_to_rejoin() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/dual_spatial_mobile.yaml"
    )
    target = Pose6D.identity()
    controller = StagedEngineNavigationController(
        assembly,
        _minimal_plan(target),
        EngineNavigationSpec.from_mapping(
            {
                "entry_region": "entry_port",
                "insertion_path": "nozzle_axis_entry",
                "local_tracking": {
                    "advance_mode": "steps",
                    "advance_steps": 5,
                },
            }
        ),
        scene_query=None,
        waypoint_tolerance_m=0.003,
        min_clearance_m=0.01,
        terminate_on_clearance_violation=True,
    )
    point = controller.plan.executor_waypoints_world[:1]

    local_tracker = controller._make_tracker(
        point,
        observer_roi_world=point[0],
        use_local_tracking=True,
    )
    rejoin_tracker = controller._make_tracker(
        point,
        observer_roi_world=point[0],
        use_local_tracking=False,
    )

    assert local_tracker.scheduler.mode == "time"
    assert local_tracker.scheduler.step_interval == 5
    assert rejoin_tracker.scheduler.mode == "tolerance"


def test_observer_critical_avoidance_does_not_change_executor_command() -> None:
    assembly = load_robot_assembly_config(
        "configs/robots/assemblies/dual_spatial_mobile.yaml"
    )
    fixed_assembly = replace(
        assembly,
        base=replace(assembly.base, control_mode="fixed"),
    )
    state = _state(assembly, Pose6D.identity())
    target = (
        state.arms["executor"].tip_pose_world.position
        + np.array([0.002, 0.001, -0.001])
    )
    solver_config = WholeBodyControllerConfig(decouple_arm_singularity=True)
    baseline = CoordinatedTrackingController(
        fixed_assembly,
        CoordinatedTrackingTarget(executor_position_world=target),
        config=CoordinatedTrackingConfig(
            inter_arm_min_distance_m=0.001,
            observer_collision_priority=False,
        ),
        solver_config=solver_config,
    )
    avoidance = CoordinatedTrackingController(
        fixed_assembly,
        CoordinatedTrackingTarget(executor_position_world=target),
        config=CoordinatedTrackingConfig(
            observer_collision_priority=True,
            inter_arm_influence_distance_m=0.030,
            inter_arm_min_distance_m=0.025,
            inter_arm_hard_stop_distance_m=0.021,
            freeze_executor_inside_safe_distance=False,
            stop_all_on_critical_distance=False,
        ),
        solver_config=solver_config,
    )

    baseline_command = baseline.compute_command(state)
    avoidance_command = avoidance.compute_command(state)

    np.testing.assert_allclose(
        avoidance_command.arms["executor"].tendon_rate_mps,
        baseline_command.arms["executor"].tendon_rate_mps,
    )
    assert avoidance_command.metadata["inter_arm_critical_distance"] is True
    assert avoidance_command.metadata["inter_arm_hard_stop"] is False
    assert (
        avoidance_command.metadata["inter_arm_safety_mode"]
        == "critical_avoidance"
    )
    assert avoidance_command.metadata["observer_collision_active"] is True


def _minimal_plan(target: Pose6D) -> EngineNavigationPlan:
    point = target.transform_point(np.array([0.0, 0.0, 0.12]))
    return EngineNavigationPlan(
        pre_entry_tip_world=point,
        insertion_direction_world=np.array([0.0, 0.0, 1.0]),
        insertion_tip_waypoints_world=np.asarray([point]),
        pre_entry_base_pose=target,
        insertion_base_poses=(target,),
        executor_waypoints_world=np.asarray(
            [
                point,
                point + np.array([0.001, 0.0, 0.0]),
            ]
        ),
        observer_roi_world=point,
    )


def _state(
    assembly,
    base_pose: Pose6D,
    *,
    tip_position: np.ndarray | None = None,
) -> RobotSystemState:
    arms = {}
    for arm in assembly.enabled_arms:
        tendon_count = arm.spatial_arm.tendon_count
        tip = base_pose.compose(arm.mount_pose).transform_point(
            np.array(
                [
                    0.0,
                    0.0,
                    sum(segment.length for segment in arm.spatial_arm.params.segments),
                ]
            )
        )
        arms[arm.name] = ArmSystemState(
            name=arm.name,
            role=arm.role,
            tip_pose_world=Pose6D(
                position=(
                    tip
                    if tip_position is None or arm.role != "executor"
                    else tip_position
                ),
                quat=base_pose.quat,
            ),
            segment_poses_world=np.repeat(
                np.eye(4, dtype=float)[None, :, :],
                3,
                axis=0,
            ),
            tendon_displacement_m=np.zeros(tendon_count),
            tendon_velocity_mps=np.zeros(tendon_count),
        )
    return RobotSystemState(
        time_s=0.0,
        base=BaseSystemState(pose=base_pose),
        arms=arms,
    )
