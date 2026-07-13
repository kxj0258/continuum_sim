"""Control algorithms for continuum-arm experiments and system scenarios."""

from continuum_sim.control.adaptive_impedance import (
    AdaptiveImpedanceConfig,
    compute_dynamic_wiping_motor_velocity_command_from_state,
)
from continuum_sim.control.cbf_qp_kinematics import (
    CBFQPConfig,
    cbf_lower_bound,
    solve_cbf_qp_velocity,
)
from continuum_sim.control.contact_triggered_admittance import (
    ContactTriggeredAdmittanceCommand,
    ContactTriggeredAdmittanceConfig,
    ContactTriggeredAdmittanceTracker,
)
from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
)
from continuum_sim.control.differential_ik import (
    DifferentialIKConfig,
    TrackingResult,
    compute_motor_velocity_command,
    compute_motor_velocity_command_from_observation,
    damped_least_squares,
    simulate_position_tracking,
)
from continuum_sim.control.dual_arm_adapter import DualArmCommandAdapter
from continuum_sim.control.engine_cleaning_controller import (
    EngineCleaningController,
    build_engine_cleaning_gains_from_config,
    limit_tcp_velocity,
    load_engine_cleaning_controller_config,
    validate_engine_cleaning_controller_config,
)
from continuum_sim.control.engine_cleaning_types import (
    EngineCleaningCommand,
    EngineCleaningControllerGains,
    EngineCleaningFeedback,
)
from continuum_sim.control.hybrid_force_position import (
    ContactMeasurement,
    compute_wiping_motor_velocity_command_from_observation,
    compute_wiping_motor_velocity_command_from_state,
    contact_measurement_from_surface_proxy,
    desired_hybrid_tip_velocity,
)
from continuum_sim.control.mobile_base_controller import (
    MobileBaseCommand,
    MobileBaseState,
    WholeBodyCommand,
    clamp_pose_to_limits,
    clip_base_twist,
    integrate_base_pose,
    reset_mobile_base_state,
    resolve_mobile_base_command,
    set_mobile_base_locked,
    zero_mobile_base_command,
)
from continuum_sim.control.mobile_base_pose_control import MobileBasePoseController
from continuum_sim.control.navigation_controller import (
    centerline_point_motor_jacobian,
    compute_navigation_motor_velocity_command,
    compute_navigation_motor_velocity_command_from_observation,
)
from continuum_sim.control.tendon_rate_control import (
    CompatibleTendonRateIntegrator,
    CompatibleTendonRateStep,
    TendonRateIntegrator,
    TendonRateLimits,
    TendonRateStep,
)
from continuum_sim.control.staged_engine_navigation import (
    StagedEngineNavigationController,
)
from continuum_sim.control.task_intent import (
    CartesianTaskIntent,
    ContactTaskIntent,
    ObserverTaskIntent,
    SafetyTaskIntent,
    SystemTaskIntent,
    TaskStatus,
    TaskStep,
)
from continuum_sim.control.unified_low_level import UnifiedLowLevelController
from continuum_sim.control.whole_body_controller import (
    WholeBodyController,
    WholeBodyControllerConfig,
    WholeBodySolveResult,
    WholeBodyTask,
)

__all__ = [
    "AdaptiveImpedanceConfig",
    "CBFQPConfig",
    "CartesianTaskIntent",
    "ContactMeasurement",
    "ContactTriggeredAdmittanceCommand",
    "ContactTriggeredAdmittanceConfig",
    "ContactTriggeredAdmittanceTracker",
    "ContactTaskIntent",
    "CoordinatedTrackingConfig",
    "CoordinatedTrackingController",
    "CompatibleTendonRateIntegrator",
    "CompatibleTendonRateStep",
    "CoordinatedTrackingTarget",
    "DifferentialIKConfig",
    "DualArmCommandAdapter",
    "EngineCleaningCommand",
    "EngineCleaningController",
    "EngineCleaningControllerGains",
    "EngineCleaningFeedback",
    "MobileBaseCommand",
    "MobileBasePoseController",
    "MobileBaseState",
    "ObserverTaskIntent",
    "SafetyTaskIntent",
    "StagedEngineNavigationController",
    "SystemTaskIntent",
    "TaskStatus",
    "TaskStep",
    "TendonRateIntegrator",
    "TendonRateLimits",
    "TendonRateStep",
    "TrackingResult",
    "UnifiedLowLevelController",
    "WholeBodyCommand",
    "WholeBodyController",
    "WholeBodyControllerConfig",
    "WholeBodySolveResult",
    "WholeBodyTask",
    "build_engine_cleaning_gains_from_config",
    "cbf_lower_bound",
    "centerline_point_motor_jacobian",
    "clamp_pose_to_limits",
    "clip_base_twist",
    "compute_dynamic_wiping_motor_velocity_command_from_state",
    "compute_motor_velocity_command",
    "compute_motor_velocity_command_from_observation",
    "compute_navigation_motor_velocity_command",
    "compute_navigation_motor_velocity_command_from_observation",
    "compute_wiping_motor_velocity_command_from_observation",
    "compute_wiping_motor_velocity_command_from_state",
    "contact_measurement_from_surface_proxy",
    "damped_least_squares",
    "desired_hybrid_tip_velocity",
    "integrate_base_pose",
    "limit_tcp_velocity",
    "load_engine_cleaning_controller_config",
    "reset_mobile_base_state",
    "resolve_mobile_base_command",
    "set_mobile_base_locked",
    "simulate_position_tracking",
    "solve_cbf_qp_velocity",
    "validate_engine_cleaning_controller_config",
    "zero_mobile_base_command",
]
