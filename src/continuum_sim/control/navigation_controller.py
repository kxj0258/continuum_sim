"""Differential-IK navigation controller with structured-scene clearance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    motor_position_to_tendon_delta,
)
from continuum_sim.control.cbf_qp_kinematics import cbf_lower_bound, solve_cbf_qp_velocity
from continuum_sim.control.differential_ik import damped_least_squares
from continuum_sim.kinematics.differential import (
    motor_position_jacobian,
    motor_velocity_to_qdot_matrix,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import physical_tendon_delta_to_q
from continuum_sim.scenes.primitives import ClearancePrimitive, DistanceQuery, nearest_clearance
from continuum_sim.tasks.navigation_config import NavigationControllerConfig


def compute_navigation_motor_velocity_command(
    motor_position: np.ndarray,
    target_position: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    scene_primitives: Sequence[ClearancePrimitive],
    config: NavigationControllerConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str]]:
    """Compute a navigation command from the commanded motor state."""

    motor_position_array = _as_vector(
        motor_position,
        "motor_position",
        expected_size=len(motor_params),
    )
    tendon_delta = motor_position_to_tendon_delta(motor_position_array, motor_params)
    q_est = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)
    fk = forward_kinematics(
        q_est,
        params,
        samples_per_segment=config.centerline_samples_per_segment,
    )
    return _compute_navigation_command_from_state(
        tip_position=fk.tip_pose[:3, 3],
        q_est=q_est,
        target_position=_as_position(target_position, "target_position"),
        centerline=fk.centerline,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        scene_primitives=tuple(scene_primitives),
        config=config,
    )


def compute_navigation_motor_velocity_command_from_observation(
    actual_tip_position: np.ndarray,
    actual_tendon_delta: np.ndarray,
    target_position: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    scene_primitives: Sequence[ClearancePrimitive],
    config: NavigationControllerConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str]]:
    """Compute a navigation command from MuJoCo-observed tip and tendon state."""

    tendon_delta = _as_vector(
        actual_tendon_delta,
        "actual_tendon_delta",
        expected_size=len(physical_tendons),
    )
    q_est = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)
    fk = forward_kinematics(
        q_est,
        params,
        samples_per_segment=config.centerline_samples_per_segment,
    )
    return _compute_navigation_command_from_state(
        tip_position=_as_position(actual_tip_position, "actual_tip_position"),
        q_est=q_est,
        target_position=_as_position(target_position, "target_position"),
        centerline=fk.centerline,
        params=params,
        physical_tendons=physical_tendons,
        motor_params=motor_params,
        scene_primitives=tuple(scene_primitives),
        config=config,
    )


def _compute_navigation_command_from_state(
    *,
    tip_position: np.ndarray,
    q_est: np.ndarray,
    target_position: np.ndarray,
    centerline: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    scene_primitives: tuple[ClearancePrimitive, ...],
    config: NavigationControllerConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str]]:
    _validate_config(config)
    position_error = target_position - tip_position
    desired_tip_velocity = config.position_gain * position_error
    tip_jacobian = motor_position_jacobian(
        q_est,
        params,
        physical_tendons,
        motor_params,
        step=config.finite_difference_step_rad,
    )
    track_velocity = damped_least_squares(
        tip_jacobian,
        desired_tip_velocity,
        config.damping,
    )

    clearance_query, clearance_index = _nearest_centerline_clearance(
        centerline,
        scene_primitives,
    )
    avoidance_velocity = np.zeros_like(track_velocity)
    if _avoidance_is_active(clearance_query, config):
        point_jacobian = centerline_point_motor_jacobian(
            q_est,
            clearance_index,
            params,
            physical_tendons,
            motor_params,
            samples_per_segment=config.centerline_samples_per_segment,
            step=config.finite_difference_step_rad,
        )
        if config.type == "navigation_cbf_qp":
            barrier_jacobian = clearance_query.normal[None, :] @ point_jacobian
            motor_velocity_cmd = solve_cbf_qp_velocity(
                track_velocity,
                barrier_jacobian=barrier_jacobian,
                barrier_lower_bound=np.array(
                    [
                        cbf_lower_bound(
                            clearance_query.distance_m,
                            config.clearance_min_m,
                            config.clearance_gain,
                        )
                    ],
                    dtype=float,
                ),
            )
        else:
            desired_point_velocity = _desired_clearance_velocity(clearance_query, config)
            avoidance_velocity = damped_least_squares(
                point_jacobian,
                desired_point_velocity,
                config.damping,
            )
            motor_velocity_cmd = track_velocity + avoidance_velocity
    else:
        motor_velocity_cmd = track_velocity
    motor_velocity_cmd = np.clip(
        motor_velocity_cmd,
        -config.max_motor_velocity_rad_s,
        config.max_motor_velocity_rad_s,
    )
    info: dict[str, np.ndarray | float | str] = {
        "q_est": np.asarray(q_est, dtype=float).copy(),
        "tip_position": np.asarray(tip_position, dtype=float).copy(),
        "target_position": target_position.copy(),
        "position_error": position_error,
        "error_norm": float(np.linalg.norm(position_error)),
        "desired_tip_velocity": desired_tip_velocity,
        "track_motor_velocity": track_velocity,
        "avoidance_motor_velocity": avoidance_velocity,
        "centerline": np.asarray(centerline, dtype=float).copy(),
        "min_clearance_m": float(clearance_query.distance_m),
        "clearance_normal": clearance_query.normal.copy(),
        "clearance_point": clearance_query.point.copy(),
        "clearance_source_id": clearance_query.source_id,
    }
    return motor_velocity_cmd, info


def centerline_point_motor_jacobian(
    q: np.ndarray,
    centerline_index: int,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    *,
    samples_per_segment: int,
    step: float,
) -> np.ndarray:
    """Return d(centerline point) / d(motor velocity) at one sampled centerline point."""

    q_array = _as_vector(q, "q", expected_size=params.q_size)
    if centerline_index < 0:
        raise ValueError(f"centerline_index must be non-negative, got {centerline_index}.")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}.")

    centerline_count = forward_kinematics(
        q_array,
        params,
        samples_per_segment=samples_per_segment,
    ).centerline.shape[0]
    if centerline_index >= centerline_count:
        raise ValueError(
            f"centerline_index {centerline_index} is outside centerline length "
            f"{centerline_count}."
        )

    jacobian_q = np.zeros((3, params.q_size), dtype=float)
    for index in range(params.q_size):
        offset = np.zeros(params.q_size, dtype=float)
        offset[index] = step
        point_plus = _centerline_point(
            q_array + offset,
            centerline_index,
            params,
            samples_per_segment,
        )
        point_minus = _centerline_point(
            q_array - offset,
            centerline_index,
            params,
            samples_per_segment,
        )
        jacobian_q[:, index] = (point_plus - point_minus) / (2.0 * step)

    return jacobian_q @ motor_velocity_to_qdot_matrix(
        params,
        physical_tendons,
        motor_params,
    )


def _nearest_centerline_clearance(
    centerline: np.ndarray,
    scene_primitives: tuple[ClearancePrimitive, ...],
) -> tuple[DistanceQuery, int]:
    centerline_array = np.asarray(centerline, dtype=float)
    if centerline_array.ndim != 2 or centerline_array.shape[1] != 3:
        raise ValueError(f"Expected centerline with shape (N, 3), got {centerline_array.shape}.")
    queries = [nearest_clearance(point, scene_primitives) for point in centerline_array]
    if not queries:
        return (
            DistanceQuery(
                distance_m=float("inf"),
                normal=np.zeros(3, dtype=float),
                source_id="empty_centerline",
                point=np.zeros(3, dtype=float),
            ),
            0,
        )
    index = int(np.argmin([query.distance_m for query in queries]))
    return queries[index], index


def _avoidance_is_active(
    query: DistanceQuery,
    config: NavigationControllerConfig,
) -> bool:
    return np.isfinite(query.distance_m) and query.distance_m < config.avoidance_influence_m


def _desired_clearance_velocity(
    query: DistanceQuery,
    config: NavigationControllerConfig,
) -> np.ndarray:
    span = max(
        config.avoidance_influence_m - config.clearance_min_m,
        1.0e-9,
    )
    strength = np.clip(
        (config.avoidance_influence_m - query.distance_m) / span,
        0.0,
        4.0,
    )
    return config.clearance_gain * strength * query.normal


def _centerline_point(
    q: np.ndarray,
    index: int,
    params: ThreeSegmentRobotParams,
    samples_per_segment: int,
) -> np.ndarray:
    return forward_kinematics(
        q,
        params,
        samples_per_segment=samples_per_segment,
    ).centerline[index].copy()


def _validate_config(config: NavigationControllerConfig) -> None:
    if config.damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {config.damping}.")
    if config.position_gain < 0.0:
        raise ValueError(f"position_gain must be non-negative, got {config.position_gain}.")
    if config.clearance_gain < 0.0:
        raise ValueError(f"clearance_gain must be non-negative, got {config.clearance_gain}.")
    if config.clearance_min_m < 0.0:
        raise ValueError(f"clearance_min_m must be non-negative, got {config.clearance_min_m}.")
    if config.avoidance_influence_m <= config.clearance_min_m:
        raise ValueError("avoidance_influence_m must be greater than clearance_min_m.")
    if config.max_motor_velocity_rad_s <= 0.0:
        raise ValueError("max_motor_velocity_rad_s must be positive.")
    if config.position_tolerance_m < 0.0:
        raise ValueError("position_tolerance_m must be non-negative.")
    if config.centerline_samples_per_segment < 2:
        raise ValueError("centerline_samples_per_segment must be at least 2.")
    if config.finite_difference_step_rad <= 0.0:
        raise ValueError("finite_difference_step_rad must be positive.")


def _as_vector(values: np.ndarray, name: str, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    return array


def _as_position(values: np.ndarray, name: str) -> np.ndarray:
    return _as_vector(values, name, expected_size=3)
