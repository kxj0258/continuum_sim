import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.kinematics.pcc import (
    constant_curvature_transform,
    constant_curvature_with_offset_segment_transform,
    forward_kinematics,
    structured_segment_transform,
)
from continuum_sim.utils.math_utils import make_transform
from continuum_sim.model.robot_params import ThreeSegmentRobotParams


def test_forward_kinematics_straight_arm() -> None:
    params = ThreeSegmentRobotParams.default()
    total_length = float(np.sum(params.segment_lengths))

    result = forward_kinematics(np.zeros(9), params, samples_per_segment=5)

    assert result.centerline.shape == (13, 3)
    assert_allclose(result.tip_pose[:3, :3], np.eye(3), atol=1.0e-14)
    assert_allclose(result.tip_pose[:3, 3], [0.0, 0.0, total_length], atol=1.0e-14)
    assert_allclose(result.centerline[:, :2], 0.0, atol=1.0e-14)
    assert_allclose(result.centerline[:, 2], np.linspace(0.0, total_length, 13), atol=1.0e-14)


def test_single_segment_positive_kx_bends_toward_positive_x() -> None:
    params = ThreeSegmentRobotParams.default()
    length = params.segments[0].length
    kx = 10.0
    theta = kx * length
    q_segment = np.array([kx, 0.0, 0.0])

    transform = constant_curvature_transform(q_segment, length)

    expected_tip = np.array(
        [
            (1.0 - np.cos(theta)) / kx,
            0.0,
            np.sin(theta) / kx,
        ]
    )
    expected_tangent = np.array([np.sin(theta), 0.0, np.cos(theta)])

    assert transform[0, 3] > 0.0
    assert_allclose(transform[:3, 3], expected_tip, atol=1.0e-12)
    assert_allclose(transform[:3, :3] @ np.array([0.0, 0.0, 1.0]), expected_tangent, atol=1.0e-12)


def test_forward_kinematics_three_segment_composition() -> None:
    params = ThreeSegmentRobotParams.default()
    q = np.array(
        [
            5.0,
            0.0,
            0.0,
            0.0,
            4.0,
            0.0,
            -3.0,
            2.0,
            0.01,
        ]
    )

    result = forward_kinematics(q, params, samples_per_segment=7)
    q_segments = q.reshape(3, 3)
    expected_tip_pose = np.eye(4)
    for q_segment, segment in zip(q_segments, params.segments, strict=True):
        expected_tip_pose = expected_tip_pose @ structured_segment_transform(
            q_segment,
            segment,
        )

    assert result.centerline.shape == (19, 3)
    assert len(result.segment_centerlines) == 3
    assert_allclose(result.tip_pose, expected_tip_pose, atol=1.0e-12)
    assert_allclose(result.segment_centerlines[0][-1], result.segment_centerlines[1][0], atol=1.0e-14)
    assert_allclose(result.segment_centerlines[1][-1], result.segment_centerlines[2][0], atol=1.0e-14)
    assert np.all(np.isfinite(result.tip_pose))
    assert np.linalg.norm(result.tip_pose[:2, 3]) > 0.0


def test_forward_kinematics_supports_original_constant_curvature_mode() -> None:
    params = ThreeSegmentRobotParams.default()
    q = np.array([5.0, 0.0, 0.0, 0.0, 4.0, 0.0, -3.0, 2.0, 0.01])

    result = forward_kinematics(
        q,
        params,
        samples_per_segment=7,
        kinematics_mode="constant_curvature",
    )
    expected_tip_pose = np.eye(4)
    for q_segment, segment in zip(q.reshape(3, 3), params.segments, strict=True):
        expected_tip_pose = expected_tip_pose @ constant_curvature_transform(
            q_segment,
            segment.length,
        )

    assert_allclose(result.tip_pose, expected_tip_pose, atol=1.0e-12)


def test_forward_kinematics_supports_constant_curvature_with_offset_mode() -> None:
    params = ThreeSegmentRobotParams.default()
    q = np.array([5.0, 0.0, 0.0, 0.0, 4.0, 0.0, -3.0, 2.0, 0.01])

    result = forward_kinematics(
        q,
        params,
        samples_per_segment=7,
        kinematics_mode="constant_curvature_with_offset",
    )
    expected_tip_pose = np.eye(4)
    for q_segment, segment in zip(q.reshape(3, 3), params.segments, strict=True):
        eps = float(q_segment[2])
        expected_tip_pose = (
            expected_tip_pose
            @ constant_curvature_transform(q_segment, segment.effective_flexure_length)
            @ make_transform(
                np.eye(3),
                np.array(
                    [
                        0.0,
                        0.0,
                        segment.effective_distal_straight_length * (1.0 + eps),
                    ]
                ),
            )
        )

    assert_allclose(result.tip_pose, expected_tip_pose, atol=1.0e-12)
    assert_allclose(
        result.tip_pose,
        np.linalg.multi_dot(
            [
                constant_curvature_with_offset_segment_transform(
                    q_segment,
                    segment,
                )
                for q_segment, segment in zip(
                    q.reshape(3, 3),
                    params.segments,
                    strict=True,
                )
            ]
        ),
        atol=1.0e-12,
    )


def test_near_zero_curvature_uses_straight_rod_approximation() -> None:
    transform = constant_curvature_transform(np.array([1.0e-12, -1.0e-12, 0.02]), 0.04)

    assert_allclose(transform[:3, :3], np.eye(3), atol=1.0e-14)
    assert_allclose(transform[:3, 3], [0.0, 0.0, 0.0408], atol=1.0e-14)
