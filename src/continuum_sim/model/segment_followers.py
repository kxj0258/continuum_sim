"""Follower sample poses for the 2DOF-per-segment MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
    forward_kinematics,
    segment_transform,
)
from continuum_sim.model.robot_params import ThreeSegmentRobotParams


SEGMENT_2DOF_Q_SIZE = 6


@dataclass(frozen=True)
class SegmentFollowerPose:
    """Pose and metadata for one runtime follower body."""

    name: str
    segment_index: int
    sample_index: int
    pose: np.ndarray
    center_position: np.ndarray
    orientation: np.ndarray
    radius: float
    length: float


def segment_2dof_q_to_pcc_q(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
) -> np.ndarray:
    """Map 6D segment total bending angles to the existing 9D PCC q layout."""

    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (SEGMENT_2DOF_Q_SIZE,):
        raise ValueError(
            f"Expected segment 2DOF q with shape ({SEGMENT_2DOF_Q_SIZE},), "
            f"got {q_array.shape}."
        )
    pcc_q = np.zeros((params.segment_count, 3), dtype=float)
    segment_angles = q_array.reshape(params.segment_count, 2)
    for segment_index, segment in enumerate(params.segments):
        hinge_x, hinge_y = segment_angles[segment_index]
        pcc_q[segment_index, 0] = hinge_y / segment.effective_flexure_length
        pcc_q[segment_index, 1] = -hinge_x / segment.effective_flexure_length
    return pcc_q.reshape(-1)


def segment_2dof_forward_kinematics(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    *,
    samples_per_segment: int = 21,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return tip and segment-tip poses for a 6D segment-angle state."""

    fk = forward_kinematics(
        segment_2dof_q_to_pcc_q(q, params),
        params,
        samples_per_segment=samples_per_segment,
        kinematics_mode=kinematics_mode,
    )
    return fk.tip_pose, fk.segment_poses


def sample_segment_followers(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    samples_per_segment: int = 4,
    *,
    follower_radius: float | None = None,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> tuple[SegmentFollowerPose, ...]:
    """Sample follower body poses along each PCC segment."""

    if samples_per_segment <= 0:
        raise ValueError(
            f"samples_per_segment must be positive, got {samples_per_segment}."
        )
    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (SEGMENT_2DOF_Q_SIZE,):
        raise ValueError(
            f"Expected segment 2DOF q with shape ({SEGMENT_2DOF_Q_SIZE},), "
            f"got {q_array.shape}."
        )

    segment_angles = q_array.reshape(params.segment_count, 2)
    base_to_segment = np.eye(4, dtype=float)
    followers: list[SegmentFollowerPose] = []
    for segment_index, segment in enumerate(params.segments):
        pcc_segment_q = np.array(
            [
                segment_angles[segment_index, 1] / segment.effective_flexure_length,
                -segment_angles[segment_index, 0] / segment.effective_flexure_length,
                0.0,
            ],
            dtype=float,
        )
        sample_length = segment.length / float(samples_per_segment)
        radius = (
            float(follower_radius)
            if follower_radius is not None
            else float(segment.tendon_radius)
        )
        for sample_index in range(samples_per_segment):
            s = (sample_index + 0.5) / float(samples_per_segment)
            local_pose = segment_transform(
                pcc_segment_q,
                segment,
                kinematics_mode=kinematics_mode,
                partial_length=segment.length * s,
            )
            world_pose = base_to_segment @ local_pose
            followers.append(
                SegmentFollowerPose(
                    name=f"follower_segment_{segment_index + 1}_sample_{sample_index + 1}",
                    segment_index=segment_index,
                    sample_index=sample_index,
                    pose=world_pose,
                    center_position=world_pose[:3, 3].copy(),
                    orientation=world_pose[:3, :3].copy(),
                    radius=radius,
                    length=sample_length,
                )
            )
        base_to_segment = base_to_segment @ segment_transform(
            pcc_segment_q,
            segment,
            kinematics_mode=kinematics_mode,
        )
    return tuple(followers)
