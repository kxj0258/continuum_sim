from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.scenes.engine_surfaces import (
    EngineSurfacePatchConfig,
    load_surface_patch_config,
    sample_surface_grid,
    sample_surface_point,
    surface_frame_from_patch,
    validate_surface_patch_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "engine_surface_path.yaml"


def test_load_surface_patch_config_from_task_yaml() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)

    assert isinstance(patch, EngineSurfacePatchConfig)
    assert patch.name == "carbon_deposit_patch"
    assert patch.type == "sphere_patch"
    assert_allclose(patch.patch_center, [0.18, 0.02, 0.43])
    assert patch.radius_m == pytest.approx(0.08)


def test_plane_patch_surface_frame_is_orthonormal() -> None:
    patch = load_surface_patch_config(
        {
            "name": "plane_a",
            "type": "plane_patch",
            "center": [0.1, 0.2, 0.3],
            "normal": [0.0, 0.0, 2.0],
            "tangent_u": [3.0, 0.0, 0.0],
            "size_u_m": 0.4,
            "size_v_m": 0.2,
        }
    )

    frame = surface_frame_from_patch(patch)

    assert_allclose(frame.center, [0.1, 0.2, 0.3])
    assert_allclose(np.linalg.norm(frame.normal), 1.0)
    assert_allclose(np.linalg.norm(frame.tangent_u), 1.0)
    assert_allclose(np.linalg.norm(frame.tangent_v), 1.0)
    assert_allclose(np.dot(frame.normal, frame.tangent_u), 0.0, atol=1.0e-12)
    assert_allclose(np.dot(frame.normal, frame.tangent_v), 0.0, atol=1.0e-12)
    assert_allclose(np.dot(frame.tangent_u, frame.tangent_v), 0.0, atol=1.0e-12)


def test_sphere_patch_surface_frame_is_orthonormal() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)

    frame = surface_frame_from_patch(patch)

    assert_allclose(np.linalg.norm(frame.normal), 1.0)
    assert_allclose(np.linalg.norm(frame.tangent_u), 1.0)
    assert_allclose(np.linalg.norm(frame.tangent_v), 1.0)
    assert_allclose(np.dot(frame.normal, frame.tangent_u), 0.0, atol=1.0e-12)


def test_sphere_patch_sample_point_stays_on_sphere() -> None:
    patch = load_surface_patch_config(TASK_CONFIG)

    point = sample_surface_point(patch, u=0.25, v=-0.25)

    assert_allclose(np.linalg.norm(point - patch.sphere_center), patch.radius_m, atol=2.0e-3)


def test_sample_surface_grid_returns_expected_shape() -> None:
    patch = load_surface_patch_config(
        {
            "name": "plane_grid",
            "type": "plane_patch",
            "center": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "tangent_u": [1.0, 0.0, 0.0],
            "size_u_m": 0.2,
            "size_v_m": 0.1,
        }
    )

    grid = sample_surface_grid(patch, num_u=5, num_v=4)

    assert grid.shape == (4, 5, 3)


def test_invalid_patch_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="patch type"):
        load_surface_patch_config(
            {
                "name": "bad_patch",
                "type": "mystery_patch",
                "center": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "tangent_u": [1.0, 0.0, 0.0],
                "size_u_m": 0.2,
                "size_v_m": 0.1,
            }
        )


def test_nonpositive_size_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="size_u_m"):
        load_surface_patch_config(
            {
                "name": "bad_plane",
                "type": "plane_patch",
                "center": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "tangent_u": [1.0, 0.0, 0.0],
                "size_u_m": 0.0,
                "size_v_m": 0.1,
            }
        )


def test_sphere_patch_missing_required_fields_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="sphere_center"):
        load_surface_patch_config(
            {
                "name": "bad_sphere",
                "type": "sphere_patch",
                "patch_center": [0.0, 0.0, 0.1],
                "normal": [0.0, 0.0, 1.0],
                "tangent_u": [1.0, 0.0, 0.0],
                "size_u_m": 0.1,
                "size_v_m": 0.1,
                "radius_m": 0.2,
            }
        )


def test_validate_surface_patch_config_rejects_parallel_normal_and_tangent() -> None:
    patch = EngineSurfacePatchConfig(
        name="nearly_parallel",
        type="plane_patch",
        center=np.array([0.0, 0.0, 0.0], dtype=float),
        patch_center=None,
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        tangent_u=np.array([0.0, 0.0, 1.0], dtype=float),
        size_u_m=0.2,
        size_v_m=0.1,
        radius_m=None,
        sphere_center=None,
        metadata={},
    )

    with pytest.raises(ValueError, match="parallel"):
        validate_surface_patch_config(patch)
