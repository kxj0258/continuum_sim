from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


accuracy = _module(PROJECT_ROOT / "scripts" / "cal_accuracy.py", "cal_accuracy")
workspace = _module(
    PROJECT_ROOT / "scripts" / "cal_accuracy_workspace.py",
    "cal_accuracy_workspace",
)


def test_accuracy_model_uses_project_kinematics_limits_and_tool_tcp() -> None:
    model = accuracy.load_accuracy_model()

    assert model.kinematics_mode == "discrete_hinge"
    assert_allclose(model.encoder_angle_lower_deg, np.full(6, -30.0))
    assert_allclose(model.encoder_angle_upper_deg, np.full(6, 30.0))
    assert_allclose(model.tcp_pose_from_tip.position, [0.0, 0.0, 0.018])


def test_batch_encoder_poses_match_scalar_project_fk() -> None:
    model = accuracy.load_accuracy_model()
    theta = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, -8.0, 5.0, 16.0, -20.0, 9.0],
            [-30.0, 30.0, 20.0, -10.0, 7.0, -4.0],
        ]
    )

    position, rotation = accuracy.batch_encoder_poses(theta, model)

    for index, angles in enumerate(theta):
        scalar = accuracy.encoder_pose(angles, model)
        assert_allclose(position[index], scalar[:3, 3], atol=1.0e-12)
        assert_allclose(rotation[index], scalar[:3, :3], atol=1.0e-12)


def test_batch_encoder_poses_falls_back_to_configured_non_discrete_mode() -> None:
    model = replace(accuracy.load_accuracy_model(), kinematics_mode="constant_curvature")
    theta = np.array([[4.0, -2.0, 1.0, 3.0, -5.0, 6.0]])

    position, rotation = accuracy.batch_encoder_poses(theta, model)
    scalar = accuracy.encoder_pose(theta[0], model)

    assert_allclose(position[0], scalar[:3, 3], atol=1.0e-12)
    assert_allclose(rotation[0], scalar[:3, :3], atol=1.0e-12)


def test_workspace_sampler_obeys_six_angle_and_nine_tendon_limits() -> None:
    model = accuracy.load_accuracy_model()
    theta = workspace.sample_valid_workspace(model, 100, np.random.default_rng(7))
    bending = np.deg2rad(theta) / model.flexure_lengths_m
    tendon = bending @ model.bending_model.coupling_matrix.T

    assert theta.shape == (100, 6)
    assert np.all(theta >= model.encoder_angle_lower_deg)
    assert np.all(theta <= model.encoder_angle_upper_deg)
    assert np.all(tendon >= model.arm.limits.tendon_displacement_min_m)
    assert np.all(tendon <= model.arm.limits.tendon_displacement_max_m)


def test_workspace_analysis_returns_per_pose_corner_bounds() -> None:
    model = accuracy.load_accuracy_model()
    theta = workspace.sample_valid_workspace(model, 12, np.random.default_rng(8))

    _, _, results = workspace.analyze_workspace(theta, [0.1, 0.5], model)

    assert len(results) == 2
    assert all(result.position_error_mm.shape == (12,) for result in results)
    assert all(result.orientation_error_deg.shape == (12,) for result in results)
    assert results[0].maximum_position_error_mm < results[1].maximum_position_error_mm
    assert results[0].maximum_position_error_mm < 5.0
