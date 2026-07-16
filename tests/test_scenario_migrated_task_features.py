from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.control.waypoint_scheduler import WaypointScheduler
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.scenes.engine_scene import load_engine_scene_config
from continuum_sim.scenes.scene_config import load_navigation_scene_config
from continuum_sim.tasks.engine_cleaning_path import (
    EngineCleaningPathSpec,
    build_engine_cleaning_plan,
)
from continuum_sim.tasks.navigation_mission import NavigationMissionSpec, resolve_navigation_waypoints
from continuum_sim.tasks.trajectory_generation import (
    TrajectorySpec,
    generate_trajectory_waypoints,
    prepend_tracking_approach,
)
from continuum_sim.application.scenario import load_scenario_config
from continuum_sim.control.scenario_controllers import WipingController
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.tasks.wiping_path import WipingPathSpec, build_wiping_plan


def test_scenario_trajectory_generation_supports_square_from_assembly() -> None:
    assembly = load_robot_assembly_config("configs/robots/assemblies/single_spatial.yaml")
    spec = TrajectorySpec(
        type="square",
        samples=16,
        radius_m=0.02,
        center_mode="straight_tip_xy",
        z_mode="straight_tip_minus_radius",
        plane="xy",
        side_length_m=0.04,
    )

    waypoints = generate_trajectory_waypoints(spec, assembly)

    assert waypoints.shape == (16, 3)
    assert np.all(np.isfinite(waypoints))
    assert float(np.max(waypoints[:, 0]) - np.min(waypoints[:, 0])) > 0.03


def test_tracking_approach_starts_at_straight_tip_and_preserves_path() -> None:
    assembly = load_robot_assembly_config("configs/robots/assemblies/single_spatial.yaml")
    requested = np.array(
        [
            [0.02, -0.01, 0.12],
            [0.02, 0.01, 0.12],
        ],
        dtype=float,
    )

    result = prepend_tracking_approach(requested, assembly, samples=5)

    assert result.waypoints_world.shape == (7, 3)
    assert_allclose(result.waypoints_world[-2:], requested)
    assert_allclose(result.waypoints_world[0], [0.0, 0.01, 0.14])
    assert result.approach_mask.tolist() == [True] * 5 + [False, False]
    assert result.source_waypoint_index.tolist() == [-1] * 5 + [0, 1]


def test_tracking_control_config_loads_scenario_overrides() -> None:
    config = load_scenario_config("configs/scenarios/dual_mujoco_tracking.yaml")

    assert config.task.tracking_control.approach_samples == 40
    assert config.task.tracking_control.tracking_mode == "time"
    assert config.task.tracking_control.trajectory_duration_s == 30.0
    assert config.task.observer_control_mode == "collision_avoidance"
    assert config.task.observer_control.influence_distance_m == 0.018
    assert config.task.observer_control.minimum_distance_m == 0.010
    assert config.task.observer_control.max_avoidance_speed_mps is None
    assert (
        config.task.tracking_control.executor_position_gain
        == config.task.tracking_control.observer_position_gain
        == 1.0
    )
    assembly = load_robot_assembly_config(config.assembly_config_path)
    executor_limits = assembly.arms["executor"].spatial_arm.limits
    observer_limits = assembly.arms["observer"].spatial_arm.limits
    assert_allclose(
        observer_limits.tendon_displacement_min_m,
        executor_limits.tendon_displacement_min_m,
    )
    assert_allclose(
        observer_limits.tendon_displacement_max_m,
        executor_limits.tendon_displacement_max_m,
    )
    assert_allclose(
        observer_limits.max_tendon_rate_mps,
        executor_limits.max_tendon_rate_mps,
    )
    assert_allclose(observer_limits.target_lead_m, executor_limits.target_lead_m)
    assert config.task.tracking_control.decouple_arm_singularity is True
    assert config.task.tracking_control.enforce_solver_velocity_limits is False
    assert config.task.tracking_control.enforce_backend_tendon_limits is False


def test_wiping_controller_accepts_tracking_control_parameters() -> None:
    assembly = load_robot_assembly_config("configs/robots/assemblies/single_spatial.yaml")
    solver_config = WholeBodyControllerConfig(tendon_regularization_weight=0.5)

    controller = WipingController(
        assembly,
        np.array([[0.045, 0.0, 0.095]], dtype=float),
        waypoint_tolerance_m=0.003,
        scene_query=None,
        surface_normal_world=np.array([-1.0, 0.0, 0.0], dtype=float),
        target_contact_distance_m=-0.0025,
        contact_tolerance_m=0.002,
        executor_position_gain=1.5,
        feedforward_speed_mps=0.002,
        max_target_speed_mps=0.015,
        solver_config=solver_config,
    )

    assert controller._tracking.feedforward_speed_mps == 0.002
    assert controller._tracking._controller.config.executor_position_gain == 1.5
    assert controller._tracking._controller.config.max_target_speed_mps == 0.015
    assert (
        controller._tracking._controller.solver.config.tendon_regularization_weight
        == 0.5
    )


def test_dynamic_wiping_scenario_loads_force_strategy_and_dynamics_path() -> None:
    config = load_scenario_config("configs/scenarios/single_mujoco_wiping.yaml")

    assert config.task.wiping_control_type == "dynamic_adaptive_impedance"
    assert config.task.force_strategy.type == "dynamic_adaptive_impedance"
    assert config.task.dynamics_config_path is not None
    assert config.task.dynamics_config_path.name == "pcc_reduced.yaml"
    assert config.task.tracking_control.executor_position_gain == 1.5
    assert config.task.tracking_control.max_target_speed_mps == 0.015
    assert config.task.tracking_control.tendon_regularization_weight == 0.5


def test_wiping_scenario_keeps_optional_admittance_parameters() -> None:
    config = load_scenario_config("configs/scenarios/single_mujoco_wiping.yaml")

    assert config.task.wiping_control_type == "dynamic_adaptive_impedance"
    assert config.task.force_strategy.type == "dynamic_adaptive_impedance"
    assert config.task.admittance.target_normal_force_n == 1.5
    assert config.task.admittance.stable_steps_required == 3


def test_waypoint_scheduler_supports_time_and_tolerance_modes() -> None:
    tolerance = WaypointScheduler(
        waypoint_count=3,
        mode="tolerance",
        tolerance_m=0.01,
        loop=False,
        controller_dt_s=0.02,
    )
    assert tolerance.update(error_norm_m=0.02) == 0
    assert tolerance.update(error_norm_m=0.001) == 1

    timed = WaypointScheduler(
        waypoint_count=4,
        mode="time",
        tolerance_m=0.0,
        loop=False,
        controller_dt_s=0.02,
        step_interval=2,
    )
    assert [timed.update(error_norm_m=10.0) for _ in range(5)] == [0, 1, 1, 2, 2]


def test_tolerance_scheduler_advances_after_waypoint_step_limit() -> None:
    scheduler = WaypointScheduler(
        waypoint_count=3,
        mode="tolerance",
        tolerance_m=0.001,
        loop=False,
        controller_dt_s=0.02,
        max_steps_per_waypoint=2,
    )

    assert scheduler.update(error_norm_m=0.01) == 0
    assert scheduler.update(error_norm_m=0.01) == 1
    assert scheduler.update(error_norm_m=0.01) == 1


def test_navigation_mission_resolves_scene_target_ids() -> None:
    scene = load_navigation_scene_config("configs/scenes/rocket_nozzle_entry.yaml")
    spec = NavigationMissionSpec(
        waypoint_ids=("entry_wall_30deg", "rib_gap_center", "throat_wall_210deg"),
    )

    waypoints = resolve_navigation_waypoints(spec, scene)

    assert waypoints.shape == (3, 3)
    assert_allclose(waypoints, scene.target_positions(spec.waypoint_ids))


def test_wiping_path_builds_raster_waypoints_from_structured_surface() -> None:
    scene = load_navigation_scene_config("configs/scenes/wiping_board.yaml")
    spec = WipingPathSpec(
        surface_id="board_surface",
        patch_id="center_patch",
        line_count=3,
        samples_per_line=4,
        approach_offset_m=0.005,
        contact_offset_m=-0.0025,
    )

    plan = build_wiping_plan(spec, scene)

    assert plan.waypoints_world.shape == (13, 3)
    assert plan.phases[0] == "approach"
    assert plan.phases.count("contact") == 12
    surface = scene.work_surface("board_surface")
    assert_allclose(plan.surface_point_world, surface.center_m)
    assert surface.signed_distance(plan.waypoints_world[0]) > 0.0
    assert surface.signed_distance(plan.waypoints_world[1]) < 0.0


def test_engine_cleaning_plan_uses_engine_surface_patch_region() -> None:
    scene = load_engine_scene_config("configs/scenes/engine_cleaning.yaml")
    spec = EngineCleaningPathSpec(
        region_name="cleaning_patch",
        num_passes_u=3,
        num_passes_v=2,
        approach_distance_m=0.01,
        retreat_distance_m=0.02,
        target_force_n=1.2,
        standoff_distance_m=0.005,
    )

    plan = build_engine_cleaning_plan(spec, scene)

    assert plan.waypoints_world.shape == (8, 3)
    assert plan.phases[0] == "approach"
    assert plan.target_force_n[1] == 1.2
