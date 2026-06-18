import numpy as np
import pytest
from numpy.testing import assert_allclose

from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model import ThreeSegmentRobotParams
from continuum_sim.model.segment_followers import (
    sample_segment_followers,
    segment_2dof_forward_kinematics,
    segment_2dof_q_to_pcc_q,
)


def test_segment_followers_straight_pose_aligns_on_z_axis() -> None:
    params = ThreeSegmentRobotParams.default()
    followers = sample_segment_followers(np.zeros(6), params, samples_per_segment=4)

    positions = np.asarray([follower.center_position for follower in followers])

    assert len(followers) == 12
    assert_allclose(positions[:, :2], 0.0, atol=1.0e-12)
    assert_allclose(
        positions[:, 2],
        [
            0.005,
            0.015,
            0.025,
            0.035,
            0.045,
            0.055,
            0.065,
            0.075,
            0.085,
            0.095,
            0.105,
            0.115,
        ],
        atol=1.0e-12,
    )
    for follower in followers:
        assert follower.name == (
            f"follower_segment_{follower.segment_index + 1}_sample_"
            f"{follower.sample_index + 1}"
        )
        assert_allclose(follower.orientation, np.eye(3), atol=1.0e-12)


def test_segment_followers_are_continuous_under_small_bending_change() -> None:
    params = ThreeSegmentRobotParams.default()
    q = np.array([0.05, -0.02, 0.03, 0.01, -0.04, 0.02], dtype=float)
    q_perturbed = q.copy()
    q_perturbed[2] += 1.0e-5

    followers = sample_segment_followers(q, params, samples_per_segment=4)
    perturbed = sample_segment_followers(q_perturbed, params, samples_per_segment=4)
    position_delta = np.asarray(
        [
            np.linalg.norm(a.center_position - b.center_position)
            for a, b in zip(followers, perturbed, strict=True)
        ],
        dtype=float,
    )

    assert np.all(np.isfinite(position_delta))
    assert float(np.max(position_delta)) < 1.0e-5


def test_segment_2dof_tip_pose_matches_existing_pcc_fk_after_conversion() -> None:
    params = ThreeSegmentRobotParams.default()
    q = np.array([0.04, -0.01, -0.03, 0.02, 0.01, 0.03], dtype=float)

    tip_pose, segment_poses = segment_2dof_forward_kinematics(q, params)
    pcc_fk = forward_kinematics(segment_2dof_q_to_pcc_q(q, params), params)

    assert_allclose(tip_pose, pcc_fk.tip_pose, atol=1.0e-12)
    assert len(segment_poses) == 3
    for actual, expected in zip(segment_poses, pcc_fk.segment_poses, strict=True):
        assert_allclose(actual, expected, atol=1.0e-12)


def test_segment_followers_reject_invalid_inputs() -> None:
    params = ThreeSegmentRobotParams.default()
    with pytest.raises(ValueError, match="shape"):
        sample_segment_followers(np.zeros(9), params)
    with pytest.raises(ValueError, match="samples_per_segment"):
        sample_segment_followers(np.zeros(6), params, samples_per_segment=0)
