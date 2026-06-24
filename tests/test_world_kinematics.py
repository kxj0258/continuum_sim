from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.kinematics.world_kinematics import (
    compose_world_tip_pose,
    compose_world_tool_pose,
    transform_centerline_to_world,
)
from continuum_sim.model.base_pose import Pose6D


def test_compose_world_tip_pose_identity_chain_returns_local_tip() -> None:
    local_tip = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.12],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    world_tip = compose_world_tip_pose(Pose6D.identity(), Pose6D.identity(), local_tip)

    assert_allclose(world_tip.to_transform(), local_tip.to_transform(), atol=1.0e-12)


def test_compose_world_tip_pose_applies_base_translation() -> None:
    base_pose = Pose6D.from_dict(
        {
            "position": [0.3, -0.2, 0.4],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )
    mount_pose = Pose6D.identity()
    local_tip = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.1],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    world_tip = compose_world_tip_pose(base_pose, mount_pose, local_tip)

    assert_allclose(world_tip.position, [0.3, -0.2, 0.5], atol=1.0e-12)


def test_compose_world_tip_pose_applies_base_rotation() -> None:
    base_pose = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.0],
            "quat": [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        }
    )
    mount_pose = Pose6D.identity()
    local_tip = Pose6D.from_dict(
        {
            "position": [1.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    world_tip = compose_world_tip_pose(base_pose, mount_pose, local_tip)

    assert_allclose(world_tip.position, [0.0, 1.0, 0.0], atol=1.0e-12)


def test_transform_centerline_to_world_transforms_n_by_3_points() -> None:
    base_pose = Pose6D.from_dict(
        {
            "position": [0.1, 0.0, 0.2],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )
    mount_pose = Pose6D.from_dict(
        {
            "position": [0.0, 0.05, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )
    local_centerline = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.10],
        ],
        dtype=float,
    )

    world_centerline = transform_centerline_to_world(base_pose, mount_pose, local_centerline)

    assert_allclose(
        world_centerline,
        [
            [0.1, 0.05, 0.2],
            [0.1, 0.05, 0.25],
            [0.1, 0.05, 0.3],
        ],
        atol=1.0e-12,
    )


def test_compose_world_tool_pose_applies_tip_to_tool_offset() -> None:
    base_pose = Pose6D.identity()
    mount_pose = Pose6D.identity()
    local_tip = Pose6D.from_dict(
        {
            "position": [0.0, 0.0, 0.12],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )
    tip_to_tool = Pose6D.from_dict(
        {
            "position": [0.0, 0.01, 0.02],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    world_tool = compose_world_tool_pose(base_pose, mount_pose, local_tip, tip_to_tool)

    assert_allclose(world_tool.position, [0.0, 0.01, 0.14], atol=1.0e-12)
