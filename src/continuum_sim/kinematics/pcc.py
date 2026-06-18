"""Piecewise constant-curvature kinematics for the three-segment arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.robot_params import PCC_VALUES_PER_SEGMENT, ThreeSegmentRobotParams
from continuum_sim.utils.math_utils import make_transform, skew


@dataclass(frozen=True)
class PCCForwardKinematicsResult:
    """Forward-kinematics output for the whole arm."""

    tip_pose: np.ndarray
    segment_poses: tuple[np.ndarray, ...]
    segment_centerlines: tuple[np.ndarray, ...]
    centerline: np.ndarray


def _as_segment_q(q: np.ndarray, params: ThreeSegmentRobotParams) -> np.ndarray:
    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (params.q_size,):
        raise ValueError(f"Expected q with shape ({params.q_size},), got {q_array.shape}.")
    return q_array.reshape(params.segment_count, PCC_VALUES_PER_SEGMENT)


def constant_curvature_transform(
    q_segment: np.ndarray,
    length: float,
    *,
    curvature_tol: float = 1.0e-9,
) -> np.ndarray:
    """Return the SE(3) transform of one constant-curvature segment.

    The strain vector is [kx, ky, eps]. Curvature components rotate the local
    z axis along the segment; eps is axial extension strain.
    """
    kx, ky, eps = np.asarray(q_segment, dtype=float)
    arc_length = length * (1.0 + eps)
    if arc_length < 0.0:
        raise ValueError("Segment arc length must not be negative.")
    if arc_length == 0.0:
        return np.eye(4, dtype=float)

    omega = np.array([-ky, kx, 0.0], dtype=float)
    kappa = float(np.linalg.norm(omega))

    if kappa < curvature_tol:
        return make_transform(np.eye(3), np.array([0.0, 0.0, arc_length]))

    omega_hat = skew(omega)
    theta = kappa * arc_length
    rotation = (
        np.eye(3)
        + np.sin(theta) / kappa * omega_hat
        + (1.0 - np.cos(theta)) / (kappa**2) * (omega_hat @ omega_hat)
    )
    translation = (
        np.eye(3) * arc_length
        + (1.0 - np.cos(theta)) / (kappa**2) * omega_hat
        + (theta - np.sin(theta)) / (kappa**3) * (omega_hat @ omega_hat)
    ) @ np.array([0.0, 0.0, 1.0])
    return make_transform(rotation, translation)


def sample_segment_centerline(
    q_segment: np.ndarray,
    length: float,
    *,
    samples: int = 21,
    curvature_tol: float = 1.0e-9,
) -> np.ndarray:
    """Sample local-frame centerline points along one PCC segment."""
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    points = []
    for partial_length in np.linspace(0.0, length, samples):
        transform = constant_curvature_transform(
            q_segment,
            float(partial_length),
            curvature_tol=curvature_tol,
        )
        points.append(transform[:3, 3])
    return np.asarray(points, dtype=float)


def forward_kinematics(
    q: np.ndarray,
    params: ThreeSegmentRobotParams | None = None,
    *,
    samples_per_segment: int = 21,
    curvature_tol: float = 1.0e-9,
) -> PCCForwardKinematicsResult:
    """Compute centerline samples and the tip pose from a PCC state vector."""
    params = params or ThreeSegmentRobotParams.default()
    q_segments = _as_segment_q(q, params)

    base_to_current = np.eye(4, dtype=float)
    segment_poses = []
    segment_centerlines = []
    full_centerline = []

    for segment_q, segment in zip(q_segments, params.segments, strict=True):
        local_points = sample_segment_centerline(
            segment_q,
            segment.length,
            samples=samples_per_segment,
            curvature_tol=curvature_tol,
        )
        homogeneous_points = np.column_stack(
            (local_points, np.ones(local_points.shape[0], dtype=float))
        )
        world_points = (base_to_current @ homogeneous_points.T).T[:, :3]

        if full_centerline:
            world_points_for_full_line = world_points[1:]
        else:
            world_points_for_full_line = world_points
        full_centerline.extend(world_points_for_full_line)
        segment_centerlines.append(world_points)

        local_tip_transform = constant_curvature_transform(
            segment_q,
            segment.length,
            curvature_tol=curvature_tol,
        )
        base_to_current = base_to_current @ local_tip_transform
        segment_poses.append(base_to_current.copy())

    return PCCForwardKinematicsResult(
        tip_pose=base_to_current,
        segment_poses=tuple(segment_poses),  # type: ignore[arg-type]
        segment_centerlines=tuple(segment_centerlines),  # type: ignore[arg-type]
        centerline=np.asarray(full_centerline, dtype=float),
    )
