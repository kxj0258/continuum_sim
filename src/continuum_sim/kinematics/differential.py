"""Differential kinematics helpers for the PCC robot model."""

from __future__ import annotations

import numpy as np

from continuum_sim.actuation.motor_mapping import (
    MotorParams,
    tendon_velocity_to_motor_velocity,
)
from continuum_sim.kinematics.analytic_pcc import analytic_bending_position_jacobian
from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
    forward_kinematics,
)
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.robot_params import ThreeSegmentRobotParams
from continuum_sim.model.tendon_coupling import build_coupling_matrix


def tip_position_from_q(
    q: np.ndarray,
    params: ThreeSegmentRobotParams | None = None,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> np.ndarray:
    """Return the 3D tip position from a PCC state."""
    fk = forward_kinematics(q, params, kinematics_mode=kinematics_mode)
    return fk.tip_pose[:3, 3].copy()


def finite_difference_position_jacobian(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    step: float = 1.0e-5,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> np.ndarray:
    """Compute d tip_position / d q by centered finite differences."""
    q_array = _as_vector(q, "q", expected_size=params.q_size)
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}.")

    jacobian = np.zeros((3, params.q_size), dtype=float)
    for index in range(params.q_size):
        offset = np.zeros(params.q_size, dtype=float)
        offset[index] = step
        position_plus = tip_position_from_q(
            q_array + offset,
            params,
            kinematics_mode=kinematics_mode,
        )
        position_minus = tip_position_from_q(
            q_array - offset,
            params,
            kinematics_mode=kinematics_mode,
        )
        jacobian[:, index] = (position_plus - position_minus) / (2.0 * step)
    return jacobian


def motor_velocity_to_qdot_matrix(
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
) -> np.ndarray:
    """Return M_qm satisfying q_dot = M_qm @ motor_velocity."""
    C = build_coupling_matrix(params, physical_tendons)
    A = _motor_velocity_to_tendon_velocity_matrix(motor_params)
    return np.linalg.pinv(C) @ A


def motor_position_jacobian(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    step: float = 1.0e-5,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> np.ndarray:
    """Return the tip-position Jacobian with respect to motor velocity."""
    J_pos = finite_difference_position_jacobian(
        q,
        params,
        step=step,
        kinematics_mode=kinematics_mode,
    )
    M_qm = motor_velocity_to_qdot_matrix(params, physical_tendons, motor_params)
    return J_pos @ M_qm


def bending_position_jacobian(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    *,
    step: float = 1.0e-5,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> np.ndarray:
    """Return the tip-position Jacobian with respect to bending rate."""

    del step
    model = BendingSpaceModel.from_arm(params, physical_tendons)
    return analytic_bending_position_jacobian(
        q,
        params,
        model.selection_matrix,
        kinematics_mode=kinematics_mode,
    )


def bending_rate_to_motor_velocity(
    bending_rate: np.ndarray,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    *,
    max_motor_velocity_rad_s: float | None = None,
) -> np.ndarray:
    """Map a bending rate to compatible motor rates with common scaling."""

    model = BendingSpaceModel.from_arm(params, physical_tendons)
    tendon_rate = model.to_tendon(bending_rate)
    motor_rate = tendon_velocity_to_motor_velocity(tendon_rate, motor_params)
    if max_motor_velocity_rad_s is None:
        return motor_rate
    if max_motor_velocity_rad_s <= 0.0:
        raise ValueError("max_motor_velocity_rad_s must be positive.")
    maximum = float(np.max(np.abs(motor_rate)))
    scale = min(1.0, max_motor_velocity_rad_s / maximum) if maximum > 0.0 else 1.0
    return scale * motor_rate


def _motor_velocity_to_tendon_velocity_matrix(
    motor_params: tuple[MotorParams, ...],
) -> np.ndarray:
    motor_count = len(motor_params)
    if motor_count <= 0:
        raise ValueError("Expected at least one motor param.")

    A = np.zeros((motor_count, motor_count), dtype=float)
    seen_motor_indices: set[int] = set()
    seen_tendon_indices: set[int] = set()
    for motor in motor_params:
        if motor.motor_index < 0 or motor.motor_index >= motor_count:
            raise ValueError(f"{motor.id} has invalid motor_index {motor.motor_index}.")
        if motor.tendon_global_index < 0 or motor.tendon_global_index >= motor_count:
            raise ValueError(
                f"{motor.id} has invalid tendon_global_index {motor.tendon_global_index}."
            )
        if motor.motor_index in seen_motor_indices:
            raise ValueError(f"Duplicate motor_index {motor.motor_index}.")
        if motor.tendon_global_index in seen_tendon_indices:
            raise ValueError(f"Duplicate tendon_global_index {motor.tendon_global_index}.")
        if motor.gear_ratio <= 0.0:
            raise ValueError(f"{motor.id} gear_ratio must be > 0.")
        if motor.spool_radius <= 0.0:
            raise ValueError(f"{motor.id} spool_radius must be > 0.")

        seen_motor_indices.add(motor.motor_index)
        seen_tendon_indices.add(motor.tendon_global_index)
        A[motor.tendon_global_index, motor.motor_index] = (
            motor.direction_sign * motor.spool_radius / motor.gear_ratio
        )

    expected_indices = set(range(motor_count))
    if seen_motor_indices != expected_indices:
        raise ValueError(
            "Expected motor_index values "
            f"0..{motor_count - 1}, got {sorted(seen_motor_indices)}."
        )
    if seen_tendon_indices != expected_indices:
        raise ValueError(
            "Expected tendon_global_index values "
            f"0..{motor_count - 1}, got {sorted(seen_tendon_indices)}."
        )
    return A


def _as_vector(
    values: np.ndarray,
    name: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if expected_size is not None and array.shape != (expected_size,):
        raise ValueError(f"Expected {name} with shape ({expected_size},), got {array.shape}.")
    if expected_size is None and array.ndim != 1:
        raise ValueError(f"Expected {name} to be a 1D vector, got {array.shape}.")
    return array
