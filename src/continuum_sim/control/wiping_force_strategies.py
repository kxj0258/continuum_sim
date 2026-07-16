"""Optional force-control strategies for scenario-driven wiping tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.control.contact_triggered_admittance import (
    ContactTriggeredAdmittanceConfig,
    ContactTriggeredAdmittanceTracker,
)
from continuum_sim.dynamics import (
    PCCDynamicsConfig,
    PCCDynamicsState,
    contact_generalized_force,
    load_pcc_dynamics_config,
    step_dynamics,
)
from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
    forward_kinematics,
)
from continuum_sim.model.bending_space import BendingSpaceModel
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.types import ArmSystemState


@dataclass(frozen=True)
class WipingForceContext:
    """State slice needed by a wiping force strategy."""

    executor: ArmSystemState
    waypoints_world: np.ndarray
    waypoint_index: int
    phase: str
    surface_normal_world: np.ndarray
    query_normal_world: np.ndarray | None
    contact_error_m: float
    estimated_force_n: float
    target_force_n: float
    normal_force_gain: float
    force_proxy_stiffness_n_m: float
    contact_tolerance_m: float
    controller_dt_s: float

    @property
    def waypoint(self) -> np.ndarray:
        return np.asarray(self.waypoints_world[self.waypoint_index], dtype=float)

    @property
    def contact_active(self) -> bool:
        return self.phase == "contact"

    @property
    def measured_force_n(self) -> float:
        if not np.isfinite(self.estimated_force_n):
            return 0.0
        return max(0.0, float(self.estimated_force_n))

    @property
    def force_error_n(self) -> float:
        if not np.isfinite(self.estimated_force_n):
            return float("nan")
        return float(self.target_force_n - self.estimated_force_n)

    @property
    def normal_world(self) -> np.ndarray:
        if self.query_normal_world is not None:
            normal = _unit_or_none(self.query_normal_world)
            if normal is not None:
                return normal
        return _unit_or_default(self.surface_normal_world)


@dataclass(frozen=True)
class WipingForceStrategyResult:
    """Corrected target and metadata produced by a wiping force strategy."""

    corrected_waypoint: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    controls_waypoint_advance: bool = False
    waypoint_advanced: bool = False


class WipingForceStrategy:
    """Small runtime protocol implemented by scenario wiping strategies."""

    strategy_type = "contact_distance"

    def compute(self, context: WipingForceContext) -> WipingForceStrategyResult:
        raise NotImplementedError


class ContactDistanceStrategy(WipingForceStrategy):
    """Current baseline: regulate signed distance only."""

    strategy_type = "contact_distance"

    def compute(self, context: WipingForceContext) -> WipingForceStrategyResult:
        correction = _finite_or_zero(context.contact_error_m)
        corrected = context.waypoint + correction * context.normal_world
        return WipingForceStrategyResult(
            corrected_waypoint=corrected,
            metadata={
                "wiping_force_strategy": self.strategy_type,
                "wiping_force_strategy_active": context.contact_active,
                "normal_correction_m": float(correction),
            },
        )


class KinematicHybridForceStrategy(WipingForceStrategy):
    """Baseline force-position correction used by the scenario controller."""

    strategy_type = "kinematic_hybrid"

    def compute(self, context: WipingForceContext) -> WipingForceStrategyResult:
        correction = _finite_or_zero(context.contact_error_m)
        force_correction = 0.0
        if (
            context.contact_active
            and context.target_force_n > 0.0
            and np.isfinite(context.force_error_n)
        ):
            force_correction = context.normal_force_gain * (
                context.force_error_n
                / max(context.force_proxy_stiffness_n_m, 1.0e-12)
            )
            correction += force_correction
        corrected = context.waypoint + correction * context.normal_world
        return WipingForceStrategyResult(
            corrected_waypoint=corrected,
            metadata={
                "wiping_force_strategy": self.strategy_type,
                "wiping_force_strategy_active": context.contact_active,
                "normal_correction_m": float(correction),
                "force_normal_correction_m": float(force_correction),
            },
        )


class ContactTriggeredAdmittanceStrategy(WipingForceStrategy):
    """Contact-triggered admittance gate for scenario wiping."""

    strategy_type = "contact_triggered_admittance"

    def __init__(self, config: ContactTriggeredAdmittanceConfig) -> None:
        self.config = config
        self.tracker = ContactTriggeredAdmittanceTracker(config)

    def compute(self, context: WipingForceContext) -> WipingForceStrategyResult:
        if not context.contact_active:
            return WipingForceStrategyResult(
                corrected_waypoint=context.waypoint,
                metadata={
                    "wiping_force_strategy": self.strategy_type,
                    "wiping_force_strategy_active": False,
                    "force_control_active": False,
                    "measured_normal_force_n": context.measured_force_n,
                    "normal_force_source": "distance_proxy",
                },
            )
        if self.tracker.target_index != context.waypoint_index:
            self.tracker.reset(target_index=context.waypoint_index)
        command = self.tracker.step(
            tip_position=context.executor.tip_pose_world.position,
            target_positions=context.waypoints_world,
            normal=context.normal_world,
            measured_normal_force_n=context.measured_force_n,
            dt=context.controller_dt_s,
        )
        return WipingForceStrategyResult(
            corrected_waypoint=command.corrected_target_position,
            controls_waypoint_advance=True,
            waypoint_advanced=command.waypoint_advanced,
            metadata={
                "wiping_force_strategy": self.strategy_type,
                "wiping_force_strategy_active": command.contact_active,
                "force_control_active": command.contact_active,
                "measured_normal_force_n": context.measured_force_n,
                "normal_force_source": "distance_proxy",
                "filtered_normal_force_n": command.filtered_force_n,
                "admittance_position_m": command.admittance_position_m,
                "admittance_velocity_m_s": command.admittance_velocity_m_s,
                "admittance_tangent_error_m": command.tangent_error_m,
                "admittance_corrected_error_m": command.corrected_error_m,
                "waypoint_advance_reason": command.advance_reason,
            },
        )


class DynamicAdaptiveImpedanceStrategy(KinematicHybridForceStrategy):
    """Use reduced PCC dynamics as an optional predictive correction."""

    strategy_type = "dynamic_adaptive_impedance"

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        *,
        dynamics_config_path: str | None = None,
        kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
    ) -> None:
        self.executor_arm = _single_executor_arm(assembly)
        self.kinematics_mode = kinematics_mode
        self.bending_model = BendingSpaceModel.from_arm(
            self.executor_arm.spatial_arm.params,
            self.executor_arm.spatial_arm.tendons,
        )
        self.dynamics_config = (
            PCCDynamicsConfig.default(self.executor_arm.spatial_arm.params)
            if dynamics_config_path is None
            else load_pcc_dynamics_config(
                dynamics_config_path,
                self.executor_arm.spatial_arm.params,
            )
        )

    def compute(self, context: WipingForceContext) -> WipingForceStrategyResult:
        baseline = super().compute(context)
        metadata = dict(baseline.metadata)
        metadata["wiping_force_strategy"] = self.strategy_type
        metadata["wiping_dynamic_system_controller_active"] = False
        if not context.contact_active:
            return WipingForceStrategyResult(
                corrected_waypoint=baseline.corrected_waypoint,
                metadata=metadata,
            )
        try:
            q = self.bending_model.to_q(
                self.bending_model.estimate(context.executor.tendon_displacement_m)
            )
            qdot = self.bending_model.to_q(
                self.bending_model.estimate(context.executor.tendon_velocity_mps)
            )
            force_error = 0.0 if not np.isfinite(context.force_error_n) else context.force_error_n
            generalized_force = contact_generalized_force(
                q,
                force_error * context.normal_world,
                self.executor_arm.spatial_arm.params,
                kinematics_mode=self.kinematics_mode,
            )
            predicted, info = step_dynamics(
                PCCDynamicsState(q=q, qdot=qdot),
                applied_generalized_force=generalized_force,
                params=self.executor_arm.spatial_arm.params,
                config=self.dynamics_config,
                dt=context.controller_dt_s,
                kinematics_mode=self.kinematics_mode,
            )
            tip_before = forward_kinematics(
                q,
                self.executor_arm.spatial_arm.params,
                kinematics_mode=self.kinematics_mode,
            ).tip_pose[:3, 3]
            tip_after = forward_kinematics(
                predicted.q,
                self.executor_arm.spatial_arm.params,
                kinematics_mode=self.kinematics_mode,
            ).tip_pose[:3, 3]
            predicted_tip_delta = tip_after - tip_before
            normal_correction = float(predicted_tip_delta @ context.normal_world)
            normal_correction = float(
                np.clip(
                    normal_correction,
                    -context.contact_tolerance_m,
                    context.contact_tolerance_m,
                )
            )
            corrected = baseline.corrected_waypoint + normal_correction * context.normal_world
            metadata.update(
                {
                    "wiping_dynamic_system_controller_active": True,
                    "dynamic_normal_correction_m": normal_correction,
                    "dynamic_predicted_tip_delta_m": predicted_tip_delta.copy(),
                    "dynamic_predicted_q": predicted.q.copy(),
                    "dynamic_predicted_qdot": predicted.qdot.copy(),
                    "dynamic_predicted_qddot": np.asarray(info["qddot"], dtype=float).copy(),
                    "dynamic_contact_generalized_force": generalized_force.copy(),
                    "kinematics_mode": self.kinematics_mode,
                }
            )
            return WipingForceStrategyResult(
                corrected_waypoint=corrected,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - keep baseline controller alive.
            metadata.update(
                {
                    "wiping_dynamic_error": f"{type(exc).__name__}: {exc}",
                }
            )
            return WipingForceStrategyResult(
                corrected_waypoint=baseline.corrected_waypoint,
                metadata=metadata,
            )


def default_wiping_force_strategy(control_type: str) -> WipingForceStrategy:
    if control_type == "contact_distance":
        return ContactDistanceStrategy()
    if control_type == "contact_triggered_admittance":
        return ContactTriggeredAdmittanceStrategy(
            ContactTriggeredAdmittanceConfig(target_normal_force_n=0.0)
        )
    return KinematicHybridForceStrategy()


def _single_executor_arm(assembly: RobotAssemblyConfig):
    matches = [arm for arm in assembly.enabled_arms if arm.role == "executor"]
    if len(matches) != 1:
        raise ValueError("Dynamic wiping strategy requires exactly one executor arm.")
    return matches[0]


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _unit_or_default(values: np.ndarray) -> np.ndarray:
    normal = _unit_or_none(values)
    if normal is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal


def _unit_or_none(values: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return None
    return vector / norm
