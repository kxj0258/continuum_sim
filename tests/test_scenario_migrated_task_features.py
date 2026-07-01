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
from continuum_sim.tasks.trajectory_generation import TrajectorySpec, generate_trajectory_waypoints
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
