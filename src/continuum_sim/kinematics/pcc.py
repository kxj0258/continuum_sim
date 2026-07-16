"""Selectable PCC kinematics for the three-segment arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from continuum_sim.model.robot_params import (
    PCC_VALUES_PER_SEGMENT,
    SegmentParams,
    ThreeSegmentRobotParams,
)
from continuum_sim.utils.math_utils import make_transform, skew


PCCKinematicsMode = Literal[
    "constant_curvature",
    "constant_curvature_with_offset",
    "discrete_hinge",
]
PCC_KINEMATICS_MODES: tuple[PCCKinematicsMode, ...] = (
    "constant_curvature",
    "constant_curvature_with_offset",
    "discrete_hinge",
)
DEFAULT_PCC_KINEMATICS_MODE: PCCKinematicsMode = "discrete_hinge"


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


def segment_transform(
    q_segment: np.ndarray,
    segment: SegmentParams,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
    partial_length: float | None = None,
    curvature_tol: float = 1.0e-9,
) -> np.ndarray:
    """Return one segment transform using the selected PCC kinematics model."""

    length = segment.length if partial_length is None else partial_length
    if length < 0.0 or length > segment.length:
        raise ValueError("partial_length must lie within the segment length.")
    if kinematics_mode == "constant_curvature":
        return constant_curvature_transform(
            q_segment,
            length,
            curvature_tol=curvature_tol,
        )
    if kinematics_mode == "constant_curvature_with_offset":
        return constant_curvature_with_offset_segment_transform(
            q_segment,
            segment,
            partial_length=length,
            curvature_tol=curvature_tol,
        )
    if kinematics_mode == "discrete_hinge":
        return structured_segment_transform(
            q_segment,
            segment,
            partial_length=length,
        )
    raise ValueError(f"Unsupported PCC kinematics_mode {kinematics_mode!r}.")


def constant_curvature_with_offset_segment_transform(
    q_segment: np.ndarray,
    segment: SegmentParams,
    *,
    partial_length: float | None = None,
    curvature_tol: float = 1.0e-9,
) -> np.ndarray:
    """Return a 36.5 mm constant-curvature flexure plus distal straight offset.

    The segment's ``effective_flexure_length`` is modeled by one ordinary
    constant-curvature transform.  Any remaining length is a straight local-Z
    spacer, matching the physical tendon-anchor/tip offset.
    """

    return _constant_curvature_with_offset_transform_at_length(
        q_segment,
        segment,
        segment.length if partial_length is None else partial_length,
        curvature_tol=curvature_tol,
    )


def structured_segment_transform(
    q_segment: np.ndarray,
    segment: SegmentParams,
    *,
    partial_length: float | None = None,
) -> np.ndarray:
    """Return one segment transform for the physical Y/X/Y/X flexure layout.

    The PCC state remains ``[kx, ky, eps]``.  Bending curvature is converted
    into the actual alternating hinge sequence for the flexure length, then a
    straight distal spacer carries the frame to the tendon anchor/tip plane.
    """

    return _structured_segment_transform_at_length(
        q_segment,
        segment,
        segment.length if partial_length is None else partial_length,
    )


def sample_structured_segment_centerline(
    q_segment: np.ndarray,
    segment: SegmentParams,
    *,
    samples: int = 21,
) -> np.ndarray:
    """Sample one physical segment along its local centerline."""

    if samples < 2:
        raise ValueError("samples must be at least 2.")
    points = []
    for partial_length in np.linspace(0.0, segment.length, samples):
        transform = _structured_segment_transform_at_length(
            q_segment,
            segment,
            float(partial_length),
        )
        points.append(transform[:3, 3])
    return np.asarray(points, dtype=float)


def sample_segment_centerline_by_mode(
    q_segment: np.ndarray,
    segment: SegmentParams,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
    samples: int = 21,
    curvature_tol: float = 1.0e-9,
) -> np.ndarray:
    """Sample one segment centerline using the selected kinematics model."""

    if samples < 2:
        raise ValueError("samples must be at least 2.")
    points = []
    for partial_length in np.linspace(0.0, segment.length, samples):
        transform = segment_transform(
            q_segment,
            segment,
            kinematics_mode=kinematics_mode,
            partial_length=float(partial_length),
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
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> PCCForwardKinematicsResult:
    """Compute centerline samples and the tip pose from a PCC state vector."""
    params = params or ThreeSegmentRobotParams.default()
    q_segments = _as_segment_q(q, params)

    base_to_current = np.eye(4, dtype=float)
    segment_poses = []
    segment_centerlines = []
    full_centerline = []

    for segment_q, segment in zip(q_segments, params.segments, strict=True):
        local_points = sample_segment_centerline_by_mode(
            segment_q,
            segment,
            kinematics_mode=kinematics_mode,
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

        local_tip_transform = segment_transform(
            segment_q,
            segment,
            kinematics_mode=kinematics_mode,
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


def _constant_curvature_with_offset_transform_at_length(
    q_segment: np.ndarray,
    segment: SegmentParams,
    partial_length: float,
    *,
    curvature_tol: float,
) -> np.ndarray:
    q_values = np.asarray(q_segment, dtype=float)
    if q_values.shape != (PCC_VALUES_PER_SEGMENT,):
        raise ValueError(
            f"q_segment must have shape ({PCC_VALUES_PER_SEGMENT},), got {q_values.shape}."
        )
    if not np.all(np.isfinite(q_values)):
        raise ValueError("q_segment must contain only finite values.")
    if partial_length < 0.0 or partial_length > segment.length:
        raise ValueError("partial_length must lie within the segment length.")

    flexure_length = segment.effective_flexure_length
    if partial_length <= flexure_length:
        return constant_curvature_transform(
            q_values,
            partial_length,
            curvature_tol=curvature_tol,
        )

    eps = float(q_values[2])
    strain_scale = 1.0 + eps
    if strain_scale < 0.0:
        raise ValueError("Segment arc length must not be negative.")
    transform = constant_curvature_transform(
        q_values,
        flexure_length,
        curvature_tol=curvature_tol,
    )
    straight_length = min(
        partial_length - flexure_length,
        segment.effective_distal_straight_length,
    )
    return transform @ _z_translation_transform(straight_length * strain_scale)


def _structured_segment_transform_at_length(
    q_segment: np.ndarray,
    segment: SegmentParams,
    partial_length: float,
) -> np.ndarray:
    q_values = np.asarray(q_segment, dtype=float)
    if q_values.shape != (PCC_VALUES_PER_SEGMENT,):
        raise ValueError(
            f"q_segment must have shape ({PCC_VALUES_PER_SEGMENT},), got {q_values.shape}."
        )
    if not np.all(np.isfinite(q_values)):
        raise ValueError("q_segment must contain only finite values.")
    if partial_length < 0.0 or partial_length > segment.length:
        raise ValueError("partial_length must lie within the segment length.")

    kx, ky, eps = q_values
    strain_scale = 1.0 + eps
    if strain_scale < 0.0:
        raise ValueError("Segment arc length must not be negative.")

    flexure_length = segment.effective_flexure_length
    distal_length = segment.effective_distal_straight_length
    axes = segment.flexure_joint_axes
    cell_length = flexure_length / float(len(axes))
    y_count = max(1, axes.count("y"))
    x_count = max(1, axes.count("x"))
    y_angle = kx * flexure_length * strain_scale / float(y_count)
    x_angle = -ky * flexure_length * strain_scale / float(x_count)

    remaining = float(partial_length)
    transform = np.eye(4, dtype=float)
    for axis in axes:
        if remaining <= 0.0:
            return transform
        transform = transform @ _axis_rotation_transform(
            axis,
            y_angle if axis == "y" else x_angle,
        )
        step = min(remaining, cell_length)
        transform = transform @ _z_translation_transform(step * strain_scale)
        remaining -= step

    if remaining > 0.0:
        step = min(remaining, distal_length)
        transform = transform @ _z_translation_transform(step * strain_scale)
    return transform


def _axis_rotation_transform(axis: str, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == "x":
        rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=float,
        )
    elif axis == "y":
        rotation = np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported flexure joint axis {axis!r}.")
    return make_transform(rotation, np.zeros(3, dtype=float))


def _z_translation_transform(distance: float) -> np.ndarray:
    return make_transform(np.eye(3, dtype=float), np.array([0.0, 0.0, distance]))
