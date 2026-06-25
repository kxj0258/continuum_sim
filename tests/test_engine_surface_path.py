from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.scenes.engine_surfaces import load_surface_patch_config
from continuum_sim.tasks.engine_surface_path import (
    EngineSurfacePathConfig,
    build_raster_cleaning_path,
    path_normals_array,
    path_positions_array,
    split_waypoints_by_phase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "engine_surface_path.yaml"


def test_build_raster_cleaning_path_generates_approach_contact_and_retreat() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=5,
        num_passes_v=4,
        approach_distance_m=0.04,
        retreat_distance_m=0.05,
        target_force_n=1.0,
        standoff_distance_m=0.02,
        snake_pattern=True,
    )

    waypoints = build_raster_cleaning_path(patch, path_config)
    phases = [waypoint.phase for waypoint in waypoints]

    assert phases[0] == "approach"
    assert phases[-1] == "retreat"
    assert phases.count("contact") == 20
    assert len(waypoints) == 22


def test_snake_pattern_reverses_neighboring_rows() -> None:
    patch = load_surface_patch_config(
        {
            "name": "plane_path",
            "type": "plane_patch",
            "center": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "tangent_u": [1.0, 0.0, 0.0],
            "size_u_m": 0.4,
            "size_v_m": 0.3,
        }
    )
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=3,
        num_passes_v=2,
        approach_distance_m=0.0,
        retreat_distance_m=0.0,
        target_force_n=1.0,
        standoff_distance_m=0.0,
        snake_pattern=True,
    )

    waypoints = build_raster_cleaning_path(patch, path_config)
    contacts = [waypoint for waypoint in waypoints if waypoint.phase == "contact"]
    row0 = contacts[0:3]
    row1 = contacts[3:6]

    assert row0[0].position[0] < row0[-1].position[0]
    assert row1[0].position[0] > row1[-1].position[0]


def test_approach_and_retreat_waypoints_offset_along_normal() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=2,
        num_passes_v=2,
        approach_distance_m=0.04,
        retreat_distance_m=0.05,
        target_force_n=1.0,
        standoff_distance_m=0.02,
        snake_pattern=True,
    )

    waypoints = build_raster_cleaning_path(patch, path_config)
    first_contact = waypoints[1]
    last_contact = waypoints[-2]

    assert_allclose(
        waypoints[0].position,
        first_contact.position + first_contact.normal * 0.04,
        atol=1.0e-12,
    )
    assert_allclose(
        waypoints[-1].position,
        last_contact.position + last_contact.normal * 0.05,
        atol=1.0e-12,
    )


def test_contact_waypoints_store_phase_force_and_standoff() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=3,
        num_passes_v=2,
        approach_distance_m=0.01,
        retreat_distance_m=0.02,
        target_force_n=1.4,
        standoff_distance_m=0.03,
        snake_pattern=False,
    )

    contacts = [w for w in build_raster_cleaning_path(patch, path_config) if w.phase == "contact"]

    assert all(waypoint.phase == "contact" for waypoint in contacts)
    assert all(waypoint.target_force_n == pytest.approx(1.4) for waypoint in contacts)
    assert all(waypoint.standoff_distance_m == pytest.approx(0.03) for waypoint in contacts)


def test_path_positions_and_normals_array_have_expected_shape() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=2,
        num_passes_v=2,
        approach_distance_m=0.01,
        retreat_distance_m=0.01,
        target_force_n=1.0,
        standoff_distance_m=0.02,
        snake_pattern=True,
    )
    waypoints = build_raster_cleaning_path(patch, path_config)

    positions = path_positions_array(waypoints)
    normals = path_normals_array(waypoints)

    assert positions.shape == (6, 3)
    assert normals.shape == (6, 3)


def test_split_waypoints_by_phase_groups_all_waypoints() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)
    path_config = EngineSurfacePathConfig(
        patch_name=patch.name,
        num_passes_u=2,
        num_passes_v=2,
        approach_distance_m=0.01,
        retreat_distance_m=0.01,
        target_force_n=1.0,
        standoff_distance_m=0.02,
        snake_pattern=True,
    )

    grouped = split_waypoints_by_phase(build_raster_cleaning_path(patch, path_config))

    assert [waypoint.phase for waypoint in grouped["approach"]] == ["approach"]
    assert len(grouped["contact"]) == 4
    assert [waypoint.phase for waypoint in grouped["retreat"]] == ["retreat"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "num_passes_u": 0,
                "num_passes_v": 2,
                "approach_distance_m": 0.0,
                "retreat_distance_m": 0.0,
                "target_force_n": 1.0,
                "standoff_distance_m": 0.0,
            },
            "num_passes_u",
        ),
        (
            {
                "num_passes_u": 2,
                "num_passes_v": 0,
                "approach_distance_m": 0.0,
                "retreat_distance_m": 0.0,
                "target_force_n": 1.0,
                "standoff_distance_m": 0.0,
            },
            "num_passes_v",
        ),
        (
            {
                "num_passes_u": 2,
                "num_passes_v": 2,
                "approach_distance_m": -0.1,
                "retreat_distance_m": 0.0,
                "target_force_n": 1.0,
                "standoff_distance_m": 0.0,
            },
            "approach_distance_m",
        ),
        (
            {
                "num_passes_u": 2,
                "num_passes_v": 2,
                "approach_distance_m": 0.0,
                "retreat_distance_m": -0.1,
                "target_force_n": 1.0,
                "standoff_distance_m": 0.0,
            },
            "retreat_distance_m",
        ),
        (
            {
                "num_passes_u": 2,
                "num_passes_v": 2,
                "approach_distance_m": 0.0,
                "retreat_distance_m": 0.0,
                "target_force_n": 0.0,
                "standoff_distance_m": 0.0,
            },
            "target_force_n",
        ),
        (
            {
                "num_passes_u": 2,
                "num_passes_v": 2,
                "approach_distance_m": 0.0,
                "retreat_distance_m": 0.0,
                "target_force_n": 1.0,
                "standoff_distance_m": -0.1,
            },
            "standoff_distance_m",
        ),
    ],
)
def test_invalid_path_config_values_raise_clear_error(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EngineSurfacePathConfig(
            patch_name="patch",
            snake_pattern=True,
            **kwargs,
        )
