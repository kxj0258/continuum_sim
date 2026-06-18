"""Control algorithms for offline continuum-arm experiments."""

from continuum_sim.control.differential_ik import (
    DifferentialIKConfig,
    TrackingResult,
    compute_motor_velocity_command,
    compute_motor_velocity_command_from_observation,
    damped_least_squares,
    simulate_position_tracking,
)
from continuum_sim.control.adaptive_impedance import (
    AdaptiveImpedanceConfig,
    compute_dynamic_wiping_motor_velocity_command_from_state,
)
from continuum_sim.control.cbf_qp_kinematics import (
    CBFQPConfig,
    cbf_lower_bound,
    solve_cbf_qp_velocity,
)
from continuum_sim.control.hybrid_force_position import (
    ContactMeasurement,
    compute_wiping_motor_velocity_command_from_observation,
    compute_wiping_motor_velocity_command_from_state,
    contact_measurement_from_surface_proxy,
    desired_hybrid_tip_velocity,
)
from continuum_sim.control.navigation_controller import (
    centerline_point_motor_jacobian,
    compute_navigation_motor_velocity_command,
    compute_navigation_motor_velocity_command_from_observation,
)

__all__ = [
    "ContactMeasurement",
    "AdaptiveImpedanceConfig",
    "CBFQPConfig",
    "DifferentialIKConfig",
    "TrackingResult",
    "centerline_point_motor_jacobian",
    "compute_motor_velocity_command",
    "compute_motor_velocity_command_from_observation",
    "compute_navigation_motor_velocity_command",
    "compute_navigation_motor_velocity_command_from_observation",
    "compute_dynamic_wiping_motor_velocity_command_from_state",
    "compute_wiping_motor_velocity_command_from_observation",
    "compute_wiping_motor_velocity_command_from_state",
    "contact_measurement_from_surface_proxy",
    "cbf_lower_bound",
    "damped_least_squares",
    "desired_hybrid_tip_velocity",
    "solve_cbf_qp_velocity",
    "simulate_position_tracking",
]
