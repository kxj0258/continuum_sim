"""World-frame whole-body Jacobian and singularity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.kinematics.differential import finite_difference_position_jacobian
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import TendonPathLike, build_coupling_matrix
from continuum_sim.system.control_layout import ControlLayout


@dataclass(frozen=True)
class SingularityConfig:
    """Thresholds for adaptive damping and command scaling."""

    rank_tolerance: float = 1.0e-9
    minimum_singular_value: float = 1.0e-5
    nominal_damping: float = 1.0e-4
    maximum_damping: float = 5.0e-2
    minimum_velocity_scale: float = 0.05


@dataclass(frozen=True)
class SingularityReport:
    """SVD-based numerical controllability report."""

    rank: int
    full_rank: bool
    singular_values: np.ndarray
    minimum_singular_value: float
    condition_number: float
    damping: float
    velocity_scale: float


def base_point_jacobian_world(
    point_world: np.ndarray,
    base_origin_world: np.ndarray,
) -> np.ndarray:
    """Map world-frame base twist to the velocity of a world-frame point."""

    point = _vector(point_world, 3, "point_world")
    origin = _vector(base_origin_world, 3, "base_origin_world")
    radius = point - origin
    return np.hstack((np.eye(3, dtype=float), -_skew(radius)))


def tendon_rate_to_shape_rate_matrix(
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
) -> np.ndarray:
    """Return ``dq/d(delta_l)`` from the physical tendon coupling matrix."""

    return np.linalg.pinv(build_coupling_matrix(params, physical_tendons))


def tendon_position_jacobian(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    """Map direct tendon-length rates to local tip linear velocity."""

    position_jacobian = finite_difference_position_jacobian(q, params, step=step)
    return position_jacobian @ tendon_rate_to_shape_rate_matrix(params, physical_tendons)


def centerline_point_tendon_jacobian(
    q: np.ndarray,
    centerline_index: int,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
    *,
    samples_per_segment: int = 6,
    step: float = 1.0e-5,
) -> np.ndarray:
    """Map tendon rates to one sampled local centerline point velocity."""

    q_values = np.asarray(q, dtype=float)
    if q_values.shape != (params.q_size,):
        raise ValueError(f"q must have shape ({params.q_size},).")
    reference = forward_kinematics(
        q_values,
        params,
        samples_per_segment=samples_per_segment,
    ).centerline
    if centerline_index < 0 or centerline_index >= reference.shape[0]:
        raise ValueError("centerline_index is outside the sampled centerline.")
    jacobian_q = np.zeros((3, params.q_size), dtype=float)
    for index in range(params.q_size):
        offset = np.zeros(params.q_size, dtype=float)
        offset[index] = step
        plus = forward_kinematics(
            q_values + offset,
            params,
            samples_per_segment=samples_per_segment,
        ).centerline[centerline_index]
        minus = forward_kinematics(
            q_values - offset,
            params,
            samples_per_segment=samples_per_segment,
        ).centerline[centerline_index]
        jacobian_q[:, index] = (plus - minus) / (2.0 * step)
    return jacobian_q @ tendon_rate_to_shape_rate_matrix(params, physical_tendons)


def rotate_position_jacobian_to_world(
    jacobian_local: np.ndarray,
    rotation_world_from_local: np.ndarray,
) -> np.ndarray:
    """Express a local linear-velocity Jacobian in world coordinates."""

    jacobian = np.asarray(jacobian_local, dtype=float)
    rotation = np.asarray(rotation_world_from_local, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 3:
        raise ValueError("jacobian_local must have shape (3, N).")
    if rotation.shape != (3, 3):
        raise ValueError("rotation_world_from_local must have shape (3, 3).")
    return rotation @ jacobian


def assemble_whole_body_jacobian(
    layout: ControlLayout,
    arm_name: str,
    base_jacobian: np.ndarray,
    arm_jacobian: np.ndarray,
) -> np.ndarray:
    """Insert active base and one named arm Jacobian into the system layout."""

    base = np.asarray(base_jacobian, dtype=float)
    arm = np.asarray(arm_jacobian, dtype=float)
    if base.ndim != 2 or base.shape[1] != 6:
        raise ValueError("base_jacobian must have shape (M, 6).")
    arm_slice = layout.arms.get(arm_name)
    if arm_slice is None:
        raise KeyError(f"Unknown arm {arm_name!r}.")
    arm_size = arm_slice.stop - arm_slice.start
    if arm.shape != (base.shape[0], arm_size):
        raise ValueError(
            f"arm_jacobian must have shape ({base.shape[0]}, {arm_size}), "
            f"got {arm.shape}."
        )
    result = np.zeros((base.shape[0], layout.size), dtype=float)
    if layout.base_size:
        result[:, layout.base] = base
    result[:, arm_slice] = arm
    return result


def analyze_singularity(
    matrix: np.ndarray,
    config: SingularityConfig = SingularityConfig(),
) -> SingularityReport:
    """Return rank and smooth adaptive damping/velocity scaling."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D.")
    singular_values = np.linalg.svd(values, compute_uv=False)
    minimum = float(singular_values[-1]) if singular_values.size else 0.0
    maximum = float(singular_values[0]) if singular_values.size else 0.0
    rank = int(np.sum(singular_values > config.rank_tolerance))
    target_rank = min(values.shape)
    ratio = np.clip(
        minimum / max(config.minimum_singular_value, np.finfo(float).eps),
        0.0,
        1.0,
    )
    damping = (
        config.maximum_damping
        + ratio * (config.nominal_damping - config.maximum_damping)
    )
    velocity_scale = (
        config.minimum_velocity_scale
        + ratio * (1.0 - config.minimum_velocity_scale)
    )
    condition = float("inf") if minimum <= 0.0 else maximum / minimum
    return SingularityReport(
        rank=rank,
        full_rank=rank == target_rank,
        singular_values=singular_values.copy(),
        minimum_singular_value=minimum,
        condition_number=condition,
        damping=float(damping),
        velocity_scale=float(velocity_scale),
    )


def analyze_tendon_mapping(
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[TendonPathLike, ...],
    config: SingularityConfig = SingularityConfig(),
) -> SingularityReport:
    """Analyze numerical rank of the tendon-to-shape coupling matrix."""

    return analyze_singularity(
        build_coupling_matrix(params, physical_tendons),
        config,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}.")
    return result
