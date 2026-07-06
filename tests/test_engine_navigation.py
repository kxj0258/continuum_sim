from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from continuum_sim.application.scenario import load_scenario_config
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.scenes.engine_scene import (
    effective_engine_frame_position,
    load_engine_scene_config,
)
from continuum_sim.tasks.engine_navigation import (
    EngineNavigationSpec,
    resolve_engine_navigation_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_SCENE = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning.yaml"
MOBILE_ASSEMBLY = (
    PROJECT_ROOT / "configs" / "robots" / "assemblies" / "dual_spatial_mobile.yaml"
)
SCENARIO = PROJECT_ROOT / "configs" / "scenarios" / "dual_engine_navigation.yaml"


def test_engine_navigation_scenario_loads_named_engine_plan() -> None:
    config = load_scenario_config(SCENARIO)

    assert config.task.type == "engine_navigation"
    assert config.task.engine_navigation is not None
    assert config.task.engine_navigation.entry_region == "entry_port"
    assert config.task.engine_navigation.insertion_path == "nozzle_axis_entry"


def test_engine_navigation_plan_resolves_annotations_without_mesh_scale() -> None:
    scene = load_engine_scene_config(ENGINE_SCENE)
    assembly = load_robot_assembly_config(MOBILE_ASSEMBLY)
    spec = EngineNavigationSpec.from_mapping(
        {
            "entry_region": "entry_port",
            "insertion_path": "nozzle_axis_entry",
            "pre_entry_standoff_m": 0.05,
            "insertion_waypoint_spacing_m": 0.02,
            "base_position_tolerance_m": 0.005,
            "base_orientation_tolerance_rad": 0.035,
            "base_position_gain": 1.5,
            "base_orientation_gain": 2.0,
            "phase_timeout_steps": 5000,
            "local_path": {
                "type": "transverse_square",
                "radius_m": 0.01,
                "samples": 40,
            },
        }
    )

    plan = resolve_engine_navigation_plan(spec, scene, assembly)

    entry = scene.regions["entry_port"]
    expected_entry = effective_engine_frame_position(scene) + entry.center_m
    np.testing.assert_allclose(plan.insertion_tip_waypoints_world[0], expected_entry)
    assert plan.insertion_tip_waypoints_world.shape[0] > 2
    assert len(plan.insertion_base_poses) == plan.insertion_tip_waypoints_world.shape[0]
    assert plan.executor_waypoints_world.shape == (40, 3)
    np.testing.assert_allclose(
        plan.pre_entry_tip_world,
        expected_entry - 0.05 * plan.insertion_direction_world,
    )


def test_engine_navigation_plan_maps_straight_tip_to_each_base_target() -> None:
    scene = load_engine_scene_config(ENGINE_SCENE)
    assembly = load_robot_assembly_config(MOBILE_ASSEMBLY)
    spec = EngineNavigationSpec.from_mapping(
        {
            "entry_region": "entry_port",
            "insertion_path": "nozzle_axis_entry",
            "local_path": {
                "type": "transverse_square",
                "radius_m": 0.01,
                "samples": 12,
            },
        }
    )

    plan = resolve_engine_navigation_plan(spec, scene, assembly)
    executor = next(arm for arm in assembly.enabled_arms if arm.role == "executor")
    straight_length = sum(segment.length for segment in executor.spatial_arm.params.segments)
    straight_tip_local = executor.mount_pose.transform_point(
        np.array([0.0, 0.0, straight_length], dtype=float)
    )

    for base_pose, tip_target in zip(
        plan.insertion_base_poses,
        plan.insertion_tip_waypoints_world,
        strict=True,
    ):
        np.testing.assert_allclose(
            base_pose.transform_point(straight_tip_local),
            tip_target,
            atol=1.0e-9,
        )


def test_engine_navigation_plan_rejects_unknown_path() -> None:
    scene = load_engine_scene_config(ENGINE_SCENE)
    assembly = load_robot_assembly_config(MOBILE_ASSEMBLY)
    spec = EngineNavigationSpec.from_mapping(
        {
            "entry_region": "entry_port",
            "insertion_path": "missing_path",
        }
    )

    with pytest.raises(ValueError, match="missing_path"):
        resolve_engine_navigation_plan(spec, scene, assembly)
