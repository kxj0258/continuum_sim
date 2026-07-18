"""PCC reduced dynamics for the three-segment continuum arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.config import load_yaml
from continuum_sim.kinematics.analytic_pcc import (
    analytic_centerline_point_jacobian,
    analytic_position_jacobian,
)
from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
    forward_kinematics,
)
from continuum_sim.model.robot_params import PCC_VALUES_PER_SEGMENT, ThreeSegmentRobotParams


@dataclass(frozen=True)
class PCCDynamicsConfig:
    """Engineering estimates for reduced PCC dynamics."""

    segment_masses_kg: np.ndarray
    bending_stiffness: np.ndarray
    axial_stiffness: np.ndarray
    damping: np.ndarray
    mass_regularization: float = 1.0e-6
    centerline_samples_per_segment: int = 7

    @classmethod
    def default(cls, params: ThreeSegmentRobotParams) -> "PCCDynamicsConfig":
        segment_count = params.segment_count
        return cls(
            segment_masses_kg=np.full(segment_count, 0.00989929, dtype=float),
            bending_stiffness=np.full(segment_count, 0.0002, dtype=float),
            axial_stiffness=np.full(segment_count, 1.0, dtype=float),
            damping=np.full(params.q_size, 0.02, dtype=float),
        )


@dataclass(frozen=True)
class PCCDynamicsState:
    """Reduced generalized state."""

    q: np.ndarray
    qdot: np.ndarray


def load_pcc_dynamics_config(
    path: str | Path,
    params: ThreeSegmentRobotParams,
) -> PCCDynamicsConfig:
    """Load dynamics estimates from YAML."""

    raw = load_yaml(path)
    dynamics = raw.get("dynamics", raw)
    return PCCDynamicsConfig(
        segment_masses_kg=_segment_vector(
            dynamics.get("segment_masses_kg", [0.00989929] * params.segment_count),
            params,
            "segment_masses_kg",
        ),
        bending_stiffness=_segment_vector(
            dynamics.get("bending_stiffness", [0.0002] * params.segment_count),
            params,
            "bending_stiffness",
        ),
        axial_stiffness=_segment_vector(
            dynamics.get("axial_stiffness", [1.0] * params.segment_count),
            params,
            "axial_stiffness",
        ),
        damping=_q_vector(
            dynamics.get("damping", [0.02] * params.q_size),
            params,
            "damping",
        ),
        mass_regularization=float(dynamics.get("mass_regularization", 1.0e-6)),
        centerline_samples_per_segment=int(
            dynamics.get("centerline_samples_per_segment", 7)
        ),
    )


def mass_matrix(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    config: PCCDynamicsConfig,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> np.ndarray:
    """Approximate generalized mass from centerline point Jacobians."""

    q_array = _q_vector(q, params, "q")
    fk = forward_kinematics(
        q_array,
        params,
        samples_per_segment=config.centerline_samples_per_segment,
        kinematics_mode=kinematics_mode,
    )
    point_count = fk.centerline.shape[0]
    if point_count <= 0:
        raise ValueError("centerline must contain at least one point.")
    point_mass = float(np.sum(config.segment_masses_kg)) / point_count
    matrix = np.zeros((params.q_size, params.q_size), dtype=float)
    for point_index in range(point_count):
        jacobian = analytic_centerline_point_jacobian(
            q_array,
            point_index,
            params,
            samples_per_segment=config.centerline_samples_per_segment,
            kinematics_mode=kinematics_mode,
        )
        matrix += point_mass * (jacobian.T @ jacobian)
    matrix += config.mass_regularization * np.eye(params.q_size, dtype=float)
    return matrix


def stiffness_matrix(
    params: ThreeSegmentRobotParams,
    config: PCCDynamicsConfig,
) -> np.ndarray:
    """Return diagonal stiffness in PCC curvature/axial-strain coordinates."""

    diagonal = np.zeros(params.q_size, dtype=float)
    for segment_index in range(params.segment_count):
        base = segment_index * PCC_VALUES_PER_SEGMENT
        bending_stiffness = (
            config.bending_stiffness[segment_index]
            * params.segments[segment_index].effective_flexure_length
        )
        diagonal[base : base + 3] = (
            bending_stiffness,
            bending_stiffness,
            config.axial_stiffness[segment_index],
        )
    return np.diag(diagonal)


def damping_matrix(
    params: ThreeSegmentRobotParams,
    config: PCCDynamicsConfig,
) -> np.ndarray:
    """Return diagonal generalized damping."""

    return np.diag(_q_vector(config.damping, params, "damping"))


def contact_generalized_force(
    q: np.ndarray,
    force_xyz_n: np.ndarray,
    params: ThreeSegmentRobotParams,
    *,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> np.ndarray:
    """Project a tip Cartesian force into generalized PCC coordinates."""

    force = np.asarray(force_xyz_n, dtype=float)
    if force.shape != (3,):
        raise ValueError(f"force_xyz_n must have shape (3,), got {force.shape}.")
    return analytic_position_jacobian(
        _q_vector(q, params, "q"),
        params,
        kinematics_mode=kinematics_mode,
    ).T @ force


def step_dynamics(
    state: PCCDynamicsState,
    *,
    applied_generalized_force: np.ndarray,
    params: ThreeSegmentRobotParams,
    config: PCCDynamicsConfig,
    dt: float,
    kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
) -> tuple[PCCDynamicsState, dict[str, np.ndarray]]:
    """Integrate one semi-implicit Euler step."""

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    q = _q_vector(state.q, params, "state.q")
    qdot = _q_vector(state.qdot, params, "state.qdot")
    tau = _q_vector(applied_generalized_force, params, "applied_generalized_force")
    mass = mass_matrix(q, params, config, kinematics_mode=kinematics_mode)
    damping = damping_matrix(params, config)
    stiffness = stiffness_matrix(params, config)
    qddot = np.linalg.solve(mass, tau - damping @ qdot - stiffness @ q)
    next_qdot = qdot + qddot * dt
    next_q = q + next_qdot * dt
    return PCCDynamicsState(q=next_q, qdot=next_qdot), {
        "qddot": qddot,
        "mass_matrix": mass,
        "damping_matrix": damping,
        "stiffness_matrix": stiffness,
    }


def _segment_vector(
    values: object,
    params: ThreeSegmentRobotParams,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (params.segment_count,):
        raise ValueError(f"{name} must have shape ({params.segment_count},), got {array.shape}.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} entries must be positive.")
    return array


def _q_vector(
    values: object,
    params: ThreeSegmentRobotParams,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (params.q_size,):
        raise ValueError(f"{name} must have shape ({params.q_size},), got {array.shape}.")
    return array
