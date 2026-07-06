from __future__ import annotations

import numpy as np

from continuum_sim.control.mobile_base_pose_control import MobileBasePoseController
from continuum_sim.control.staged_engine_navigation import (
    StagedEngineNavigationController,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.system.types import (
    ArmSystemState,
    BaseSystemState,
    RobotSystemState,
)
from continuum_sim.tasks.engine_navigation import (
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


def _state(assembly, base_pose: Pose6D) -> RobotSystemState:
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
            tip_pose_world=Pose6D(position=tip, quat=base_pose.quat),
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
