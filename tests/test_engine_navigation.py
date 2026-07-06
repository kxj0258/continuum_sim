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
    assert config.task.engine_navigation.local_path_axial_retraction_m == pytest.approx(
        0.045
    )
    assert [
        path.path_type
        for path in config.task.engine_navigation.intermediate_local_paths
    ] == ["transverse_circle", "ellipse"]
    assert config.task.engine_navigation.local_tracking.advance_mode == "tolerance"
    assert (
        config.task.engine_navigation.local_tracking.waypoint_tolerance_m
        == pytest.approx(0.005)
    )
    assert (
        config.task.engine_navigation.local_tracking.rejoin_tolerance_m
        == pytest.approx(0.005)
    )
    assert (
        config.task.engine_navigation.local_tracking.max_steps_per_waypoint
        == 25
    )
    assert config.task.engine_navigation.local_tracking.transition_samples == 20
    observer = config.task.engine_navigation.observer_control
    assert observer.inter_arm_influence_distance_m == pytest.approx(0.018)
    assert observer.inter_arm_safe_distance_m == pytest.approx(0.014)
    assert observer.inter_arm_critical_distance_m == pytest.approx(0.009)
    assert observer.observer_collision_weight == pytest.approx(250.0)
    assert observer.stop_all_on_critical_distance is False


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
                "axial_retraction_m": 0.01,
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
    expected_local_center = (
        plan.insertion_tip_waypoints_world[-1]
        - 0.01 * plan.insertion_direction_world
    )
    np.testing.assert_allclose(plan.observer_roi_world, expected_local_center)


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


def test_engine_navigation_plan_resolves_three_retracted_local_paths() -> None:
    config = load_scenario_config(SCENARIO)
    scene = load_engine_scene_config(ENGINE_SCENE)
    assembly = load_robot_assembly_config(MOBILE_ASSEMBLY)
    assert config.task.engine_navigation is not None

    plan = resolve_engine_navigation_plan(
        config.task.engine_navigation,
        scene,
        assembly,
    )

    assert [path.path_type for path in plan.local_path_plans] == [
        "transverse_circle",
        "ellipse",
        "transverse_square",
    ]
    assert [path.insertion_index for path in plan.local_path_plans] == sorted(
        path.insertion_index for path in plan.local_path_plans
    )
    assert [path.waypoints_world.shape[0] for path in plan.local_path_plans] == [
        60,
        75,
        60,
    ]
    for path in plan.local_path_plans:
        expected_center = (
            path.insertion_target_world
            - 0.045 * plan.insertion_direction_world
        )
        np.testing.assert_allclose(path.center_world, expected_center)
        np.testing.assert_allclose(
            path.waypoints_world[0],
            path.waypoints_world[-1],
        )
        assert path.transition_waypoints_world.shape == (20, 3)
        np.testing.assert_allclose(
            path.transition_waypoints_world[0],
            path.insertion_target_world,
        )
        np.testing.assert_allclose(
            path.transition_waypoints_world[-1],
            path.waypoints_world[0],
        )


def test_engine_navigation_plan_supports_extended_local_path_shapes() -> None:
    scene = load_engine_scene_config(ENGINE_SCENE)
    assembly = load_robot_assembly_config(MOBILE_ASSEMBLY)
    spec = EngineNavigationSpec.from_mapping(
        {
            "entry_region": "entry_port",
            "insertion_path": "nozzle_axis_entry",
            "intermediate_local_paths": [
                {
                    "name": "ellipse_probe",
                    "at_fraction": 0.25,
                    "type": "ellipse",
                    "radius_m": 0.01,
                    "shape": {
                        "radius_x_m": 0.012,
                        "radius_y_m": 0.006,
                    },
                    "samples": 16,
                },
                {
                    "name": "line_probe",
                    "at_fraction": 0.50,
                    "type": "line",
                    "shape": {"length_m": 0.018},
                    "samples": 9,
                },
                {
                    "name": "lissajous_probe",
                    "at_fraction": 0.75,
                    "type": "lissajous",
                    "radius_m": 0.008,
                    "shape": {
                        "lissajous_frequency_x": 1,
                        "lissajous_frequency_y": 2,
                        "lissajous_phase_deg": 45.0,
                    },
                    "samples": 18,
                },
            ],
            "local_path": {
                "name": "terminal_circle",
                "type": "circle",
                "radius_m": 0.01,
                "samples": 12,
            },
        }
    )

    plan = resolve_engine_navigation_plan(spec, scene, assembly)

    assert [path.path_type for path in plan.local_path_plans] == [
        "ellipse",
        "line",
        "lissajous",
        "circle",
    ]
    assert [path.waypoints_world.shape[0] for path in plan.local_path_plans] == [
        16,
        9,
        18,
        12,
    ]


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


def test_engine_navigation_rejects_negative_axial_retraction() -> None:
    with pytest.raises(ValueError, match="axial_retraction_m"):
        EngineNavigationSpec.from_mapping(
            {
                "entry_region": "entry_port",
                "insertion_path": "nozzle_axis_entry",
                "local_path": {
                    "axial_retraction_m": -0.001,
                },
            }
        )


def test_engine_navigation_time_tracking_ignores_advance_steps() -> None:
    spec = EngineNavigationSpec.from_mapping(
        {
            "entry_region": "entry_port",
            "insertion_path": "nozzle_axis_entry",
            "local_tracking": {
                "advance_mode": "time",
                "advance_time_s": 0.1,
                "advance_steps": 5,
            },
        }
    )

    assert spec.local_tracking.advance_mode == "time"
    assert spec.local_tracking.advance_time_s == pytest.approx(0.1)
    assert spec.local_tracking.advance_steps is None


def test_engine_navigation_steps_tracking_ignores_advance_time() -> None:
    spec = EngineNavigationSpec.from_mapping(
        {
            "entry_region": "entry_port",
            "insertion_path": "nozzle_axis_entry",
            "local_tracking": {
                "advance_mode": "steps",
                "advance_time_s": 0.1,
                "advance_steps": 5,
            },
        }
    )

    assert spec.local_tracking.advance_mode == "steps"
    assert spec.local_tracking.advance_time_s is None
    assert spec.local_tracking.advance_steps == 5
