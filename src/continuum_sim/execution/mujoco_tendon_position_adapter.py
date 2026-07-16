"""MuJoCo tendon-position execution adapter for tendon-rate references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.config import MujocoConfig
from continuum_sim.control.tendon_rate_control import (
    BendingRateServoConfig,
    CompatibleBendingRateServo,
    CompatibleTendonRateIntegrator,
    TendonRateLimits,
)
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import RobotSystemCommand


@dataclass(frozen=True)
class TendonPositionExecutionStep:
    """Layer-4 output for MuJoCo tendon-position actuators."""

    tendon_position_target_m: np.ndarray
    applied_rates_mps: dict[str, np.ndarray]
    arm_targets_m: dict[str, np.ndarray]
    inner_loop_modes: dict[str, str]
    saturation: dict[str, dict[str, object]]


class MujocoTendonPositionExecutionAdapter:
    """Convert tendon-rate references to MuJoCo absolute tendon position targets."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        layout: ControlLayout,
        mujoco_config: MujocoConfig,
        tendon_rate_servo_config: BendingRateServoConfig | None,
    ) -> None:
        self.assembly = assembly
        self.layout = layout
        self.config = mujoco_config
        self._integrators = {
            arm.name: CompatibleTendonRateIntegrator(
                self.layout.bending_models[arm.name],
                TendonRateLimits(
                    displacement_min_m=arm.spatial_arm.limits.tendon_displacement_min_m,
                    displacement_max_m=arm.spatial_arm.limits.tendon_displacement_max_m,
                    max_rate_mps=arm.spatial_arm.limits.max_tendon_rate_mps,
                    target_lead_m=arm.spatial_arm.limits.target_lead_m,
                ),
            )
            for arm in assembly.enabled_arms
        }
        self._rate_servos = (
            {
                arm.name: CompatibleBendingRateServo(
                    self.layout.bending_models[arm.name],
                    self._integrators[arm.name].limits,
                    tendon_rate_servo_config,
                )
                for arm in assembly.enabled_arms
            }
            if tendon_rate_servo_config is not None
            else {}
        )
        self._last_applied_rates = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in assembly.enabled_arms
        }
        self._last_tendon_targets = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in assembly.enabled_arms
        }
        self._last_inner_loop_modes = {
            arm.name: (
                "bending_rate_servo" if arm.name in self._rate_servos else "legacy"
            )
            for arm in assembly.enabled_arms
        }

    @property
    def last_applied_rates(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._last_applied_rates.items()}

    @property
    def last_tendon_targets(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._last_tendon_targets.items()}

    def reset(self, actual_tendon_displacement_m: np.ndarray) -> None:
        actual_all = np.asarray(actual_tendon_displacement_m, dtype=float)
        for arm_name in self.layout.arms:
            tendon_slice = self.layout.tendon_slice(arm_name)
            actual = actual_all[tendon_slice]
            self._integrators[arm_name].reset()
            if arm_name in self._rate_servos:
                self._rate_servos[arm_name].reset(actual)
                self._last_tendon_targets[arm_name] = (
                    self._rate_servos[arm_name].displacement_m
                )
            else:
                self._last_tendon_targets[arm_name] = (
                    self._integrators[arm_name].displacement_m
                )
            self._last_inner_loop_modes[arm_name] = (
                "bending_rate_servo" if arm_name in self._rate_servos else "legacy"
            )
        self._last_applied_rates = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in self.assembly.enabled_arms
        }

    def step(
        self,
        command: RobotSystemCommand,
        *,
        dt: float,
        actual_tendon_displacement_m: np.ndarray,
        actuator_force_n: np.ndarray,
    ) -> TendonPositionExecutionStep:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        tendon_target = np.zeros(self.layout.tendon_size, dtype=float)
        actual_all = np.asarray(actual_tendon_displacement_m, dtype=float)
        force_all = np.asarray(actuator_force_n, dtype=float)
        saturation: dict[str, dict[str, object]] = {}
        enforce_tendon_limits = not bool(
            command.metadata.get("disable_backend_tendon_limits", False)
        )
        tendon_target_mode = str(
            command.metadata.get(
                "backend_tendon_target_mode",
                "protected" if enforce_tendon_limits else "actual_anchored",
            )
        )

        for arm_name in self.layout.arms:
            tendon_slice = self.layout.tendon_slice(arm_name)
            actual = actual_all[tendon_slice]
            raw_debug = command.arms[arm_name].control_space == "raw_tendon_debug"
            selected_mode = self._select_inner_loop_mode(arm_name, raw_debug)
            self._rebase_on_mode_change(arm_name, selected_mode, actual)
            if selected_mode == "bending_rate_servo":
                arm_saturation = self._step_rate_servo(
                    arm_name,
                    command,
                    tendon_slice,
                    actual,
                    force_all,
                    dt,
                )
                tendon_target[tendon_slice] = arm_saturation["target_m"]
            else:
                arm_saturation = self._step_legacy_integrator(
                    arm_name,
                    command,
                    actual,
                    raw_debug,
                    enforce_tendon_limits,
                    tendon_target_mode,
                    dt,
                )
                tendon_target[tendon_slice] = arm_saturation["target_m"]
            saturation[arm_name] = arm_saturation
            self._last_tendon_targets[arm_name] = tendon_target[tendon_slice].copy()
            self._last_inner_loop_modes[arm_name] = selected_mode

        return TendonPositionExecutionStep(
            tendon_position_target_m=tendon_target,
            applied_rates_mps={
                name: values.copy()
                for name, values in self._last_applied_rates.items()
            },
            arm_targets_m={
                name: tendon_target[self.layout.tendon_slice(name)].copy()
                for name in self.layout.arms
            },
            inner_loop_modes=dict(self._last_inner_loop_modes),
            saturation=saturation,
        )

    def finalize_step(
        self,
        step: TendonPositionExecutionStep,
        *,
        dt: float,
        previous_actual_tendon_displacement_m: np.ndarray,
        previous_tendon_targets: dict[str, np.ndarray],
        state_arms: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        saturation = {
            name: dict(values)
            for name, values in step.saturation.items()
        }
        for arm_name in self.layout.arms:
            tendon_slice = self.layout.tendon_slice(arm_name)
            arm_state = state_arms[arm_name]
            realized_rate = (
                arm_state.tendon_displacement_m
                - previous_actual_tendon_displacement_m[tendon_slice]
            ) / float(dt)
            target_rate = (
                step.tendon_position_target_m[tendon_slice]
                - previous_tendon_targets[arm_name]
            ) / float(dt)
            realized_rate = realized_rate.astype(float)
            target_rate = target_rate.astype(float)
            saturation[arm_name]["realized_rate_mps"] = realized_rate
            saturation[arm_name]["measured_rate_fd_mps"] = realized_rate
            saturation[arm_name]["target_rate_fd_mps"] = target_rate
            saturation[arm_name]["target_lead_m"] = (
                step.tendon_position_target_m[tendon_slice]
                - arm_state.tendon_displacement_m
            )
            if "target_lead_limit_m" in saturation[arm_name]:
                saturation[arm_name]["target_lead_utilization"] = np.divide(
                    np.abs(saturation[arm_name]["target_lead_m"]),
                    saturation[arm_name]["target_lead_limit_m"],
                )
            post_force = arm_state.actuator_force_n
            force_range = self.config.actuators.tendon_position.forcerange_n
            force_denominator = np.where(
                post_force >= 0.0,
                abs(float(force_range[1])),
                abs(float(force_range[0])),
            )
            force_utilization = np.divide(
                np.abs(post_force),
                force_denominator,
                out=np.zeros_like(post_force),
                where=force_denominator > 0.0,
            )
            saturation[arm_name]["actuator_force_n"] = post_force.copy()
            saturation[arm_name]["actuator_force_utilization"] = force_utilization
            saturation[arm_name]["actuator_force_at_limit"] = (
                force_utilization >= 1.0 - 1.0e-6
            )
            saturation[arm_name]["bending_realized_rate"] = (
                self.layout.bending_models[arm_name].estimate(realized_rate)
            )
        return saturation

    def _select_inner_loop_mode(self, arm_name: str, raw_debug: bool) -> str:
        if raw_debug:
            return "raw_debug"
        if arm_name in self._rate_servos:
            return "bending_rate_servo"
        return "legacy"

    def _rebase_on_mode_change(
        self,
        arm_name: str,
        selected_mode: str,
        actual: np.ndarray,
    ) -> None:
        if selected_mode == self._last_inner_loop_modes[arm_name]:
            return
        if selected_mode == "bending_rate_servo":
            self._rate_servos[arm_name].reset(
                actual,
                retained_target_m=self._last_tendon_targets[arm_name],
            )
        elif selected_mode == "raw_debug":
            self._integrators[arm_name].reset_raw(
                self._last_tendon_targets[arm_name]
            )

    def _step_rate_servo(
        self,
        arm_name: str,
        command: RobotSystemCommand,
        tendon_slice: slice,
        actual: np.ndarray,
        force_all: np.ndarray,
        dt: float,
    ) -> dict[str, object]:
        servo_step = self._rate_servos[arm_name].step(
            command.arms[arm_name].tendon_rate_mps,
            dt,
            actual_displacement_m=actual,
            actuator_force_n=(
                force_all[tendon_slice]
                if force_all.size
                else None
            ),
        )
        self._last_applied_rates[arm_name] = servo_step.constrained_rate_mps
        model = self.layout.bending_models[arm_name]
        return {
            "rate": servo_step.rate_saturated,
            "target_rate": servo_step.target_rate_saturated,
            "displacement": servo_step.displacement_saturated,
            "lead": servo_step.lead_saturated,
            "force": servo_step.force_saturated,
            "hard_force": servo_step.hard_force_saturated,
            "common_scale": servo_step.common_scale,
            "force_scale": servo_step.force_scale,
            "requested_rate_mps": servo_step.requested_rate_mps.copy(),
            "compatible_rate_mps": servo_step.compatible_rate_mps.copy(),
            "constrained_rate_mps": servo_step.constrained_rate_mps.copy(),
            "applied_rate_mps": servo_step.constrained_rate_mps.copy(),
            "measured_rate_mps": servo_step.measured_rate_mps.copy(),
            "measured_rate_raw_mps": servo_step.measured_rate_raw_mps.copy(),
            "target_rate_mps": servo_step.target_rate_mps.copy(),
            "raw_target_m": servo_step.raw_target_m.copy(),
            "target_m": servo_step.displacement_m.copy(),
            "target_lead_m": servo_step.target_lead_m.copy(),
            "target_lead_limit_m": servo_step.target_lead_limit_m.copy(),
            "target_lead_utilization": np.divide(
                np.abs(servo_step.target_lead_m),
                servo_step.target_lead_limit_m,
            ),
            "bending_requested_rate": servo_step.bending_requested_rate.copy(),
            "bending_applied_rate": servo_step.bending_constrained_rate.copy(),
            "bending_measured_rate": servo_step.bending_measured_rate.copy(),
            "bending_rate_error": servo_step.bending_rate_error.copy(),
            "bending_integral": servo_step.bending_integral.copy(),
            "rate_error_integral_m": model.to_tendon(servo_step.bending_integral),
            "anti_windup_correction": servo_step.anti_windup_correction.copy(),
            "anti_windup_correction_m": model.to_tendon(
                servo_step.anti_windup_correction
            ),
            "anti_windup_active": np.abs(
                model.to_tendon(servo_step.anti_windup_correction)
            ) > 1.0e-12,
            "force_constraint_active": servo_step.force_saturated.copy(),
            "guard_feasible": servo_step.guard_feasible,
            "max_constraint_violation_m": servo_step.max_constraint_violation_m,
            "compatibility_bypassed_for_safety": (
                servo_step.compatibility_bypassed_for_safety
            ),
            "hold_target_retained": servo_step.hold_target_retained,
            "compatibility_residual_mps": (
                servo_step.compatibility_residual_mps.copy()
            ),
            "raw_debug": False,
            "target_mode": "bending_rate_servo",
            "inner_loop_mode": "bending_rate_servo",
            "execution_layer": "mujoco_tendon_position",
        }

    def _step_legacy_integrator(
        self,
        arm_name: str,
        command: RobotSystemCommand,
        actual: np.ndarray,
        raw_debug: bool,
        enforce_tendon_limits: bool,
        tendon_target_mode: str,
        dt: float,
    ) -> dict[str, object]:
        step = self._integrators[arm_name].step(
            command.arms[arm_name].tendon_rate_mps,
            dt,
            raw_debug=raw_debug,
            actual_displacement_m=actual,
            enforce_limits=enforce_tendon_limits,
            target_mode=tendon_target_mode,
        )
        self._last_applied_rates[arm_name] = step.applied_rate_mps
        return {
            "rate": step.rate_saturated,
            "displacement": step.displacement_saturated,
            "common_scale": step.common_scale,
            "requested_rate_mps": step.requested_rate_mps.copy(),
            "constrained_rate_mps": step.applied_rate_mps.copy(),
            "applied_rate_mps": step.applied_rate_mps.copy(),
            "target_m": step.displacement_m.copy(),
            "compatibility_residual_mps": step.compatibility_residual_mps.copy(),
            "raw_debug": step.raw_debug,
            "target_mode": "raw_debug" if raw_debug else tendon_target_mode,
            "inner_loop_mode": "raw_debug" if raw_debug else "legacy",
            "execution_layer": "mujoco_tendon_position",
        }
