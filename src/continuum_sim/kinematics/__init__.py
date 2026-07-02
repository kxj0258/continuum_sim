"""Kinematics models for continuum robot simulation."""

from continuum_sim.kinematics.core_chain import ContinuumKinematicsChain
from continuum_sim.kinematics.differential import (
    bending_position_jacobian as differential_bending_position_jacobian,
    bending_rate_to_motor_velocity,
    finite_difference_position_jacobian,
    motor_position_jacobian,
    motor_velocity_to_qdot_matrix,
    tip_position_from_q,
)
from continuum_sim.kinematics.pcc import PCCForwardKinematicsResult, forward_kinematics
from continuum_sim.kinematics.sdf import (
    damped_pseudoinverse,
    fuse_task_and_nullspace_velocity,
    nullspace_projector,
    sdf_repulsive_velocity,
)
from continuum_sim.kinematics.tendon_mapping import q_to_tendon_delta, tendon_delta_to_q
from continuum_sim.kinematics.whole_body import (
    SingularityConfig,
    SingularityReport,
    analyze_singularity,
    analyze_tendon_mapping,
    assemble_whole_body_jacobian,
    base_point_jacobian_world,
    bending_position_jacobian,
    centerline_point_bending_jacobian,
    centerline_point_tendon_jacobian,
    rotate_position_jacobian_to_world,
    tendon_position_jacobian,
    tendon_rate_to_shape_rate_matrix,
)

__all__ = [
    "ContinuumKinematicsChain",
    "bending_rate_to_motor_velocity",
    "differential_bending_position_jacobian",
    "PCCForwardKinematicsResult",
    "damped_pseudoinverse",
    "finite_difference_position_jacobian",
    "forward_kinematics",
    "fuse_task_and_nullspace_velocity",
    "motor_position_jacobian",
    "motor_velocity_to_qdot_matrix",
    "nullspace_projector",
    "q_to_tendon_delta",
    "sdf_repulsive_velocity",
    "tendon_delta_to_q",
    "SingularityConfig",
    "SingularityReport",
    "analyze_singularity",
    "analyze_tendon_mapping",
    "assemble_whole_body_jacobian",
    "base_point_jacobian_world",
    "bending_position_jacobian",
    "centerline_point_bending_jacobian",
    "centerline_point_tendon_jacobian",
    "rotate_position_jacobian_to_world",
    "tendon_position_jacobian",
    "tendon_rate_to_shape_rate_matrix",
    "tip_position_from_q",
]
