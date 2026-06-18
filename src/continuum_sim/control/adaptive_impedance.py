"""Dynamics-assisted adaptive impedance control for wiping experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.actuation.motor_mapping import MotorParams
from continuum_sim.control.differential_ik import damped_least_squares
from continuum_sim.control.hybrid_force_position import desired_hybrid_tip_velocity
from continuum_sim.dynamics import (
    PCCDynamicsConfig,
    PCCDynamicsState,
    contact_generalized_force,
    damping_matrix,
    mass_matrix,
    step_dynamics,
    stiffness_matrix,
)
from continuum_sim.kinematics.differential import (
    finite_difference_position_jacobian,
    motor_velocity_to_qdot_matrix,
)
from continuum_sim.model.physical_tendon import PhysicalTendonPath
from continuum_sim.model.robot_params import PCC_VALUES_PER_SEGMENT, ThreeSegmentRobotParams
from continuum_sim.scenes.contact_surfaces import WorkSurfaceConfig
from continuum_sim.tasks.wiping_config import WipingControllerConfig


@dataclass(frozen=True)
class AdaptiveImpedanceConfig:
    """Settings for the optional dynamics-assisted controller."""

    dynamics: PCCDynamicsConfig

    @classmethod
    def default(cls, params: ThreeSegmentRobotParams) -> "AdaptiveImpedanceConfig":
        return cls(dynamics=PCCDynamicsConfig.default(params))


def compute_dynamic_wiping_motor_velocity_command_from_state(
    tip_position: np.ndarray,
    q_est: np.ndarray,
    qdot_est: np.ndarray,
    *,
    target_position: np.ndarray,
    surface: WorkSurfaceConfig,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    motor_params: tuple[MotorParams, ...],
    wiping_config: WipingControllerConfig,
    adaptive_config: AdaptiveImpedanceConfig,
    measured_normal_force_n: float | None,
    dt: float,
    contact_radius_m: float = 0.0,
    force_control_enabled: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str | bool]]:
    """Compute a motor velocity command through a reduced dynamics step."""

    q = _remove_axial_strain_dofs(_vector(q_est, "q_est", params.q_size))
    qdot = _remove_axial_strain_dofs(_vector(qdot_est, "qdot_est", params.q_size))
    desired_velocity, info = desired_hybrid_tip_velocity(
        tip_position,
        target_position,
        surface,
        wiping_config,
        measured_normal_force_n=measured_normal_force_n,
        contact_radius_m=contact_radius_m,
        force_control_enabled=force_control_enabled,
    )
    J_tip = finite_difference_position_jacobian(
        q,
        params,
        step=wiping_config.finite_difference_step_rad,
    )
    active_dofs = _bending_dof_mask(params)
    desired_qdot = np.zeros(params.q_size, dtype=float)
    desired_qdot[active_dofs] = damped_least_squares(
        J_tip[:, active_dofs],
        desired_velocity,
        wiping_config.damping,
    )
    desired_qddot = (desired_qdot - qdot) / dt
    M = mass_matrix(q, params, adaptive_config.dynamics)
    D = damping_matrix(params, adaptive_config.dynamics)
    K = stiffness_matrix(params, adaptive_config.dynamics)
    normal_force = float(info["normal_force_n"])
    contact_tau = contact_generalized_force(q, normal_force * surface.normal, params)
    tau = M @ desired_qddot + D @ qdot + K @ q - contact_tau
    predicted, dyn_info = step_dynamics(
        PCCDynamicsState(q=q, qdot=qdot),
        applied_generalized_force=tau + contact_tau,
        params=params,
        config=adaptive_config.dynamics,
        dt=dt,
    )
    M_qm = motor_velocity_to_qdot_matrix(params, physical_tendons, motor_params)
    motor_velocity_cmd = np.linalg.pinv(M_qm) @ predicted.qdot
    motor_velocity_cmd = np.clip(
        motor_velocity_cmd,
        -wiping_config.max_motor_velocity_rad_s,
        wiping_config.max_motor_velocity_rad_s,
    )
    info.update(
        {
            "q_est": q.copy(),
            "qdot_est": qdot.copy(),
            "desired_qdot": desired_qdot,
            "desired_qddot": desired_qddot,
            "predicted_q": predicted.q.copy(),
            "predicted_qdot": predicted.qdot.copy(),
            "predicted_qddot": np.asarray(dyn_info["qddot"], dtype=float),
            "stiffness_diag": np.diag(K),
            "damping_diag": np.diag(D),
            "contact_generalized_force": contact_tau,
        }
    )
    return motor_velocity_cmd, info


def _vector(values: np.ndarray, name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    return array


def _bending_dof_mask(params: ThreeSegmentRobotParams) -> np.ndarray:
    mask = np.ones(params.q_size, dtype=bool)
    mask[PCC_VALUES_PER_SEGMENT - 1 :: PCC_VALUES_PER_SEGMENT] = False
    return mask


def _remove_axial_strain_dofs(q_vector: np.ndarray) -> np.ndarray:
    array = q_vector.copy()
    array[PCC_VALUES_PER_SEGMENT - 1 :: PCC_VALUES_PER_SEGMENT] = 0.0
    return array
