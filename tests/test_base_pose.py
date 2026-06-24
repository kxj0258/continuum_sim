from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.mount_frame import load_mobile_base_mount_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "robots" / "mobile_base_pose.yaml"


def test_pose6d_identity_is_unit_pose() -> None:
    pose = Pose6D.identity()

    assert_allclose(pose.position, [0.0, 0.0, 0.0])
    assert_allclose(pose.quat, [1.0, 0.0, 0.0, 0.0])
    assert_allclose(pose.to_transform(), np.eye(4))


def test_identity_pose_leaves_point_unchanged() -> None:
    point = np.array([0.1, -0.2, 0.3], dtype=float)

    assert_allclose(Pose6D.identity().apply_to_point(point), point)


def test_translation_pose_shifts_point() -> None:
    pose = Pose6D.from_dict(
        {
            "position": [0.3, -0.1, 0.2],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    assert_allclose(
        pose.apply_to_point(np.array([0.1, 0.2, 0.3], dtype=float)),
        [0.4, 0.1, 0.5],
    )


def test_rotation_about_z_by_ninety_degrees_rotates_point() -> None:
    pose = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.0],
            "quat": [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        }
    )

    assert_allclose(
        pose.apply_to_point(np.array([1.0, 0.0, 0.0], dtype=float)),
        [0.0, 1.0, 0.0],
        atol=1.0e-12,
    )


def test_pose_compose_matches_sequential_application() -> None:
    first = Pose6D.from_dict(
        {
            "position": [0.2, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )
    second = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.0],
            "quat": [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        }
    )
    point = np.array([1.0, 0.0, 0.0], dtype=float)

    composed = first.compose(second)

    assert_allclose(
        composed.apply_to_point(point),
        first.apply_to_point(second.apply_to_point(point)),
        atol=1.0e-12,
    )


def test_inverse_round_trip_recovers_original_point_and_pose() -> None:
    pose = Pose6D.from_dict(
        {
            "position": [0.2, -0.4, 0.1],
            "quat": [0.9, 0.0, 0.3, 0.0],
        }
    )
    point = np.array([0.3, 0.1, -0.2], dtype=float)
    child_pose = Pose6D.from_dict(
        {
            "position": [0.0, 0.05, 0.02],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    point_round_trip = pose.inverse().apply_to_point(pose.apply_to_point(point))
    pose_round_trip = pose.inverse().apply_to_pose(pose.apply_to_pose(child_pose))

    assert_allclose(point_round_trip, point, atol=1.0e-12)
    assert_allclose(pose_round_trip.to_transform(), child_pose.to_transform(), atol=1.0e-12)


def test_quaternion_is_normalized_automatically() -> None:
    pose = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.0],
            "quat": [2.0, 0.0, 0.0, 0.0],
        }
    )

    assert_allclose(pose.quat, [1.0, 0.0, 0.0, 0.0])


def test_invalid_position_shape_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="position"):
        Pose6D(position=np.zeros(2, dtype=float), quat=np.array([1.0, 0.0, 0.0, 0.0]))


def test_invalid_quaternion_shape_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="quat"):
        Pose6D(position=np.zeros(3, dtype=float), quat=np.zeros(3, dtype=float))


def test_near_zero_quaternion_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="quat"):
        Pose6D(position=np.zeros(3, dtype=float), quat=np.array([0.0, 0.0, 0.0, 1.0e-16]))


def test_apply_to_points_supports_n_by_3_inputs() -> None:
    pose = Pose6D.from_dict(
        {
            "position": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )
    points = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.5]], dtype=float)

    assert_allclose(
        pose.apply_to_points(points),
        [[0.1, 0.2, 0.3], [0.3, 0.1, 0.8]],
    )


def test_mobile_base_pose_yaml_loads_base_and_mount_pose() -> None:
    config = load_mobile_base_mount_config(CONFIG_PATH)

    assert config.mobile_base.type == "prescribed_pose"
    assert_allclose(config.mobile_base.pose.position, [0.0, -0.4, 0.2])
    assert_allclose(config.mobile_base.pose.quat, [1.0, 0.0, 0.0, 0.0])
    assert config.mount.name == "continuum_mount"
    assert config.mount.parent_frame == "mobile_base"
    assert config.mount.child_frame == "continuum_base"
