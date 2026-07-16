"""Composable MuJoCo backend with bending-compatible tendon commands."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from continuum_sim.backends.mujoco_backend import MujocoBackend
from continuum_sim.config import MujocoConfig, load_mujoco_config
from continuum_sim.control.mobile_base_controller import (
    MobileBaseCommand,
    MobileBaseState,
    integrate_base_pose,
)
from continuum_sim.control.tendon_rate_control import (
    BendingRateServoConfig,
    CompatibleBendingRateServo,
    CompatibleTendonRateIntegrator,
    TendonRateLimits,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.mount_frame import MobileBaseLimitsConfig
from continuum_sim.model.robot_assembly import RobotAssemblyConfig, load_robot_assembly_config
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import (
    ArmSystemState,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


class MujocoSystemBackend:
    """MuJoCo boundary for named base and compatible tendon-rate control."""

    def __init__(
        self,
        mujoco_config: MujocoConfig,
        assembly: RobotAssemblyConfig,
        *,
        xml_path: str | Path | None = None,
        tendon_rate_servo_config: BendingRateServoConfig | None = None,
    ) -> None:
        if len(assembly.enabled_arms) == 1 and mujoco_config.tendon_model.count != 9:
            arm = assembly.enabled_arms[0]
            mujoco_config = replace(
                mujoco_config,
                model=replace(mujoco_config.model, type="distributed_links"),
                tendon_model=replace(
                    mujoco_config.tendon_model,
                    count=arm.spatial_arm.tendon_count,
                ),
                site_names=replace(
                    mujoco_config.site_names,
                    base=f"{arm.name}_base_site",
                    segments=tuple(
                        f"{arm.name}_segment_{index}_tip" for index in range(1, 4)
                    ),
                    tip=f"{arm.name}_tip",
                ),
            )
        if mujoco_config.control_mode != "tendon_position":
            raise ValueError("MujocoSystemBackend requires tendon_position control mode.")
        self.config = mujoco_config
        self.assembly = assembly
        self.layout = ControlLayout.from_assembly(assembly)
        if self.layout.tendon_size != mujoco_config.tendon_model.count:
            raise ValueError(
                "Assembly tendon count must match the MuJoCo tendon model: "
                f"{self.layout.tendon_size} and {mujoco_config.tendon_model.count}."
            )
        self.physics = MujocoBackend(
            mujoco_config,
            xml_path=xml_path,
        )
        # System commands are already complete named layouts; default-arm
        # expansion is not part of this backend contract.
        self.physics._dual_arm_command_adapter = None
        tendon_rate_servo_config = _bound_servo_config_to_actuator(
            tendon_rate_servo_config,
            mujoco_config,
        )
        self._integrators = {
            arm.name: CompatibleTendonRateIntegrator(
                self.layout.bending_models[arm.name],
                TendonRateLimits(
                    displacement_min_m=arm.spatial_arm.limits.tendon_displacement_min_m,
                    displacement_max_m=arm.spatial_arm.limits.tendon_displacement_max_m,
                    max_rate_mps=arm.spatial_arm.limits.max_tendon_rate_mps,
                    target_lead_m=arm.spatial_arm.limits.target_lead_m,
                )
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
        self._base_state = MobileBaseState(
            pose=assembly.base.initial_pose,
            locked=assembly.base.control_mode == "fixed",
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

    @classmethod
    def from_config(
        cls,
        mujoco_config: str | Path | MujocoConfig,
        assembly_config: str | Path | RobotAssemblyConfig,
        *,
        xml_path: str | Path | None = None,
        tendon_rate_servo_config: BendingRateServoConfig | None = None,
    ) -> "MujocoSystemBackend":
        resolved_mujoco = (
            mujoco_config
            if isinstance(mujoco_config, MujocoConfig)
            else load_mujoco_config(mujoco_config)
        )
        resolved_assembly = (
            assembly_config
            if isinstance(assembly_config, RobotAssemblyConfig)
            else load_robot_assembly_config(assembly_config)
        )
        return cls(
            resolved_mujoco,
            resolved_assembly,
            xml_path=xml_path,
            tendon_rate_servo_config=tendon_rate_servo_config,
        )

    def reset_system(self) -> RobotSystemState:
        self.physics.reset()
        actual_tendon_displacement = self.physics.get_tendon_length()
        for arm_name in self.layout.arms:
            tendon_slice = self.layout.tendon_slice(arm_name)
            actual = actual_tendon_displacement[tendon_slice]
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
                "bending_rate_servo"
                if arm_name in self._rate_servos
                else "legacy"
            )
        self._last_applied_rates = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in self.assembly.enabled_arms
        }
        self._base_state = MobileBaseState(
            pose=self.assembly.base.initial_pose,
            locked=self.assembly.base.control_mode == "fixed",
        )
        return self.get_system_state()

    def step_system(
        self,
        command: RobotSystemCommand,
        *,
        dt: float,
        n_substeps: int = 1,
    ) -> RobotSystemState:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if set(command.arms) != set(self.layout.arms):
            raise ValueError(
                "System command arms must exactly match the control layout: "
                f"{sorted(command.arms)} and {sorted(self.layout.arms)}."
            )
        self._base_state = integrate_base_pose(
            self._base_state,
            MobileBaseCommand(command.base_twist_world, frame="world"),
            dt=dt,
            max_linear_speed=(
                self.assembly.base.max_linear_speed_mps
                if bool(command.metadata.get("enforce_backend_base_speed_limits", False))
                else None
            ),
            max_angular_speed=(
                self.assembly.base.max_angular_speed_rad_s
                if bool(command.metadata.get("enforce_backend_base_speed_limits", False))
                else None
            ),
        )
        base_position = np.clip(
            self._base_state.pose.position,
            self.assembly.base.position_min_m,
            self.assembly.base.position_max_m,
        )
        self._base_state = MobileBaseState(
            pose=Pose6D(position=base_position, quat=self._base_state.pose.quat),
            locked=self._base_state.locked,
            last_twist=self._base_state.last_twist,
        )

        tendon_target = np.zeros(self.layout.tendon_size, dtype=float)
        actual_tendon_displacement = self.physics.get_tendon_length()
        actual_actuator_force = self.physics.get_actuator_force()
        previous_tendon_target = {
            arm_name: target.copy()
            for arm_name, target in self._last_tendon_targets.items()
        }
        saturation: dict[str, dict[str, object]] = {}
        enforce_tendon_limits = not bool(
            command.metadata.get("disable_backend_tendon_limits", False)
        )
        tendon_target_mode = command.metadata.get("backend_tendon_target_mode")
        if tendon_target_mode is None:
            tendon_target_mode = "protected" if enforce_tendon_limits else "actual_anchored"
        tendon_target_mode = str(tendon_target_mode)
        for arm_name in self.layout.arms:
            tendon_slice = self.layout.tendon_slice(arm_name)
            actual = actual_tendon_displacement[tendon_slice]
            raw_debug = command.arms[arm_name].control_space == "raw_tendon_debug"
            if arm_name in self._rate_servos and not raw_debug:
                selected_inner_loop_mode = "bending_rate_servo"
            elif raw_debug:
                selected_inner_loop_mode = "raw_debug"
            else:
                selected_inner_loop_mode = "legacy"
            if selected_inner_loop_mode != self._last_inner_loop_modes[arm_name]:
                if selected_inner_loop_mode == "bending_rate_servo":
                    self._rate_servos[arm_name].reset(
                        actual,
                        retained_target_m=self._last_tendon_targets[arm_name],
                    )
                elif selected_inner_loop_mode == "raw_debug":
                    self._integrators[arm_name].reset_raw(
                        self._last_tendon_targets[arm_name]
                    )
            if arm_name in self._rate_servos and not raw_debug:
                servo_step = self._rate_servos[arm_name].step(
                    command.arms[arm_name].tendon_rate_mps,
                    dt,
                    actual_displacement_m=actual,
                    actuator_force_n=(
                        actual_actuator_force[tendon_slice]
                        if actual_actuator_force.size
                        else None
                    ),
                )
                tendon_target[tendon_slice] = servo_step.displacement_m
                self._last_applied_rates[arm_name] = servo_step.constrained_rate_mps
                saturation[arm_name] = {
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
                    "rate_error_integral_m": (
                        self.layout.bending_models[arm_name].to_tendon(
                            servo_step.bending_integral
                        )
                    ),
                    "anti_windup_correction": servo_step.anti_windup_correction.copy(),
                    "anti_windup_correction_m": (
                        self.layout.bending_models[arm_name].to_tendon(
                            servo_step.anti_windup_correction
                        )
                    ),
                    "anti_windup_active": np.abs(
                        self.layout.bending_models[arm_name].to_tendon(
                            servo_step.anti_windup_correction
                        )
                    ) > 1.0e-12,
                    "force_constraint_active": servo_step.force_saturated.copy(),
                    "guard_feasible": servo_step.guard_feasible,
                    "max_constraint_violation_m": (
                        servo_step.max_constraint_violation_m
                    ),
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
                }
            else:
                step = self._integrators[arm_name].step(
                    command.arms[arm_name].tendon_rate_mps,
                    dt,
                    raw_debug=raw_debug,
                    actual_displacement_m=actual,
                    enforce_limits=enforce_tendon_limits,
                    target_mode=tendon_target_mode,
                )
                tendon_target[tendon_slice] = step.displacement_m
                self._last_applied_rates[arm_name] = step.applied_rate_mps
                saturation[arm_name] = {
                    "rate": step.rate_saturated,
                    "displacement": step.displacement_saturated,
                    "common_scale": step.common_scale,
                    "requested_rate_mps": step.requested_rate_mps.copy(),
                    "constrained_rate_mps": step.applied_rate_mps.copy(),
                    "applied_rate_mps": step.applied_rate_mps.copy(),
                    "target_m": step.displacement_m.copy(),
                    "compatibility_residual_mps": (
                        step.compatibility_residual_mps.copy()
                    ),
                    "raw_debug": step.raw_debug,
                    "target_mode": "raw_debug" if raw_debug else tendon_target_mode,
                    "inner_loop_mode": selected_inner_loop_mode,
                }
            self._last_tendon_targets[arm_name] = tendon_target[tendon_slice].copy()
            self._last_inner_loop_modes[arm_name] = selected_inner_loop_mode

        base_rpy = _pose_to_xyz_rpy(self._base_state.pose)
        raw_control = (
            tendon_target
            if self.assembly.base.control_mode == "fixed"
            else np.concatenate((tendon_target, base_rpy))
        )
        self.physics.step(raw_control, n_substeps=n_substeps)
        state = self.get_system_state()
        for arm_name in self.layout.arms:
            tendon_slice = self.layout.tendon_slice(arm_name)
            realized_rate = (
                state.arms[arm_name].tendon_displacement_m
                - actual_tendon_displacement[tendon_slice]
            ) / float(dt)
            target_rate = (
                tendon_target[tendon_slice] - previous_tendon_target[arm_name]
            ) / float(dt)
            post_force = state.arms[arm_name].actuator_force_n
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
            saturation[arm_name]["realized_rate_mps"] = realized_rate
            saturation[arm_name]["measured_rate_fd_mps"] = realized_rate
            saturation[arm_name]["target_rate_fd_mps"] = target_rate
            saturation[arm_name]["target_lead_m"] = (
                tendon_target[tendon_slice]
                - state.arms[arm_name].tendon_displacement_m
            )
            if "target_lead_limit_m" in saturation[arm_name]:
                saturation[arm_name]["target_lead_utilization"] = np.divide(
                    np.abs(saturation[arm_name]["target_lead_m"]),
                    saturation[arm_name]["target_lead_limit_m"],
                )
            saturation[arm_name]["actuator_force_n"] = post_force.copy()
            saturation[arm_name]["actuator_force_utilization"] = force_utilization
            saturation[arm_name]["actuator_force_at_limit"] = (
                force_utilization >= 1.0 - 1.0e-6
            )
            saturation[arm_name]["bending_realized_rate"] = (
                self.layout.bending_models[arm_name].estimate(realized_rate)
            )
        return RobotSystemState(
            time_s=state.time_s,
            base=state.base,
            arms=state.arms,
            metadata={**state.metadata, "saturation": saturation},
        )

    def get_system_state(self) -> RobotSystemState:
        tendon_displacement = self.physics.get_tendon_length()
        tendon_velocity = self.physics.get_tendon_velocity()
        actuator_force = self.physics.get_actuator_force()
        arms: dict[str, ArmSystemState] = {}
        dual = len(self.layout.arms) > 1
        for arm in self.assembly.enabled_arms:
            tendon_slice = self.layout.tendon_slice(arm.name)
            tip_name, segment_names = self._site_names(arm.name, dual=dual)
            tip_pose_matrix = self._site_pose(tip_name)
            segment_pose_matrices = np.asarray(
                [self._site_pose(name) for name in segment_names],
                dtype=float,
            )
            mount_position_world = self._base_state.pose.compose(
                arm.mount_pose
            ).position
            centerline_world = np.vstack(
                (
                    mount_position_world,
                    segment_pose_matrices[:, :3, 3],
                    tip_pose_matrix[:3, 3],
                )
            )
            arms[arm.name] = ArmSystemState(
                name=arm.name,
                role=arm.role,
                tip_pose_world=Pose6D.from_matrix(tip_pose_matrix),
                segment_poses_world=segment_pose_matrices,
                tendon_displacement_m=tendon_displacement[tendon_slice],
                tendon_velocity_mps=(
                    tendon_velocity[tendon_slice]
                    if tendon_velocity.size
                    else self._last_applied_rates[arm.name]
                ),
                tendon_target_m=self._last_tendon_targets[arm.name],
                actuator_force_n=(
                    actuator_force[tendon_slice]
                    if actuator_force.size
                    else np.zeros_like(tendon_displacement[tendon_slice])
                ),
                centerline_world=centerline_world,
                metadata={
                    "attachment": arm.attachment,
                    "bending": self.layout.bending_models[arm.name].estimate(
                        tendon_displacement[tendon_slice]
                    ),
                    "compatibility_residual_m": self.layout.bending_models[
                        arm.name
                    ].residual(tendon_displacement[tendon_slice]),
                    "compatibility_residual_norm_m": self.layout.bending_models[
                        arm.name
                    ].residual_norm(tendon_displacement[tendon_slice]),
                },
            )
        return RobotSystemState(
            time_s=float(self.physics.data.time),
            base=BaseSystemState(
                pose=self._base_state.pose,
                twist_world=self._base_state.last_twist,
            ),
            arms=arms,
            metadata={
                "backend": "mujoco",
                "control": "bending_compatible",
                "mujoco_mobile_base_pose_rpy": self.physics.get_mobile_base_pose_rpy(),
                "mujoco_mobile_base_frame_pose": self.physics.get_site_pose(
                    "mobile_base_frame"
                ),
            },
        )

    def _site_names(self, arm_name: str, *, dual: bool) -> tuple[str, tuple[str, ...]]:
        if dual:
            return (
                f"{arm_name}_tip",
                tuple(f"{arm_name}_segment_{index}_tip" for index in range(1, 4)),
            )
        return self.config.site_names.tip, tuple(self.config.site_names.segments)

    def _site_pose(self, name: str) -> np.ndarray:
        site_id = self.physics._site_id(name)
        return self.physics._site_pose(site_id)


def _pose_to_xyz_rpy(pose: Pose6D) -> np.ndarray:
    rotation = pose.as_matrix()[:3, :3]
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1.0e-8:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        yaw = 0.0
    return np.concatenate((pose.position, np.array([roll, pitch, yaw], dtype=float)))


def _bound_servo_config_to_actuator(
    config: BendingRateServoConfig | None,
    mujoco_config: MujocoConfig,
) -> BendingRateServoConfig | None:
    if config is None:
        return None
    actuator = mujoco_config.actuators.tendon_position
    if not actuator.forcelimited:
        return config
    physical_force_limit = min(abs(value) for value in actuator.forcerange_n)
    hard_force_limit = (
        physical_force_limit
        if config.hard_force_limit_n is None
        else min(config.hard_force_limit_n, physical_force_limit)
    )
    soft_force_limit = (
        0.8 * hard_force_limit
        if config.soft_force_limit_n is None
        else min(config.soft_force_limit_n, hard_force_limit)
    )
    force_limited_lead = soft_force_limit / actuator.kp
    max_target_lead = (
        force_limited_lead
        if config.max_target_lead_m is None
        else np.minimum(config.max_target_lead_m, force_limited_lead)
    )
    return replace(
        config,
        max_target_lead_m=max_target_lead,
        soft_force_limit_n=soft_force_limit,
        hard_force_limit_n=hard_force_limit,
    )
