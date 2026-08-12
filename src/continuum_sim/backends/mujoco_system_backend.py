"""Composable MuJoCo backend with bending-compatible tendon commands."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from continuum_sim.backends.mujoco_backend import MujocoBackend
from continuum_sim.config import MujocoConfig, load_mujoco_config
from continuum_sim.system.mobile_base import (
    MobileBaseCommand,
    MobileBaseState,
    integrate_base_pose,
)
from continuum_sim.execution.tendon_rate_control import (
    BendingRateServoConfig,
)
from continuum_sim.execution import ActuationCompatibilityLayer
from continuum_sim.kinematics.pcc import (
    DEFAULT_PCC_KINEMATICS_MODE,
    PCCKinematicsMode,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import RobotAssemblyConfig, load_robot_assembly_config
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import (
    ArmSystemState,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
    ToolWrenchState,
)
from continuum_sim.tools.attachments import AttachmentConfig


class MujocoSystemBackend:
    """MuJoCo boundary for named base and compatible tendon-rate control."""

    def __init__(
        self,
        mujoco_config: MujocoConfig,
        assembly: RobotAssemblyConfig,
        *,
        xml_path: str | Path | None = None,
        tendon_rate_servo_config: BendingRateServoConfig | None = None,
        kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
        attachment_configs: Mapping[str, AttachmentConfig] | None = None,
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
        self.kinematics_mode = kinematics_mode
        self.attachment_configs = dict(attachment_configs or {})
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
        self._tendon_execution = ActuationCompatibilityLayer(
            assembly,
            self.layout,
            mujoco_config,
            tendon_rate_servo_config,
        )
        self._base_state = MobileBaseState(
            pose=assembly.base.initial_pose,
            locked=assembly.base.control_mode == "fixed",
        )
        self._tool_wrench_bias: dict[str, np.ndarray] = {}
        self._tool_wrench_filtered: dict[str, np.ndarray] = {}
        self._tool_wrench_time_s: dict[str, float] = {}
        self.runtime_timing = None

    def set_runtime_timing(self, reporter) -> None:
        """Attach one optional timing reporter to system and physics stages."""

        self.runtime_timing = reporter
        self.physics.runtime_timing = reporter

    @classmethod
    def from_config(
        cls,
        mujoco_config: str | Path | MujocoConfig,
        assembly_config: str | Path | RobotAssemblyConfig,
        *,
        xml_path: str | Path | None = None,
        tendon_rate_servo_config: BendingRateServoConfig | None = None,
        kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
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
            kinematics_mode=kinematics_mode,
        )

    def reset_system(self) -> RobotSystemState:
        self.physics.reset()
        actual_tendon_displacement = self.physics.get_tendon_length()
        self._tendon_execution.reset(actual_tendon_displacement)
        self._base_state = MobileBaseState(
            pose=self.assembly.base.initial_pose,
            locked=self.assembly.base.control_mode == "fixed",
        )
        self._reset_tool_wrench_state()
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
        timing = self.runtime_timing
        with (
            nullcontext()
            if timing is None
            else timing.measure("system.prepare")
        ):
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
            actual_tendon_displacement = self.physics.get_tendon_length()
            actual_actuator_force = self.physics.get_actuator_force()
            previous_tendon_target = {
                arm_name: target.copy()
                for arm_name, target in self._tendon_execution.last_tendon_targets.items()
            }

        with (
            nullcontext()
            if timing is None
            else timing.measure("control.inner_loop")
        ):
            execution_step = self._tendon_execution.project_and_track(
                command,
                dt=dt,
                actual_tendon_displacement_m=actual_tendon_displacement,
                actuator_force_n=actual_actuator_force,
            )

        base_rpy = _pose_to_xyz_rpy(self._base_state.pose)
        raw_control = (
            execution_step.tendon_position_target_m
            if self.assembly.base.control_mode == "fixed"
            else np.concatenate((execution_step.tendon_position_target_m, base_rpy))
        )
        self.physics.advance(raw_control, n_substeps=n_substeps)
        with (
            nullcontext()
            if timing is None
            else timing.measure("state.build")
        ):
            state = self.get_system_state()
        with (
            nullcontext()
            if timing is None
            else timing.measure("control.finalize")
        ):
            saturation = self._tendon_execution.finalize_step(
                execution_step,
                dt=dt,
                previous_actual_tendon_displacement_m=actual_tendon_displacement,
                previous_tendon_targets=previous_tendon_target,
                state_arms=state.arms,
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
        last_applied_rates = self._tendon_execution.last_applied_rates
        last_tendon_targets = self._tendon_execution.last_tendon_targets
        arms: dict[str, ArmSystemState] = {}
        dual = len(self.layout.arms) > 1
        for arm in self.assembly.enabled_arms:
            tendon_slice = self.layout.tendon_slice(arm.name)
            tip_name, segment_names = self._site_names(arm.name, dual=dual)
            tip_pose_matrix = self._site_pose(tip_name)
            tool_pose_matrix = self.physics.get_site_pose(f"{arm.name}_tool_tcp")
            sensor_pose_matrix = self.physics.get_site_pose(
                f"{arm.name}_ft_sensor_site"
            )
            tool_wrench = self._tool_wrench_state(
                arm.name,
                sensor_pose_matrix=sensor_pose_matrix,
            )
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
                    else last_applied_rates[arm.name]
                ),
                tendon_target_m=last_tendon_targets[arm.name],
                actuator_force_n=(
                    actuator_force[tendon_slice]
                    if actuator_force.size
                    else np.zeros_like(tendon_displacement[tendon_slice])
                ),
                centerline_world=centerline_world,
                tool_pose_world=(
                    None
                    if tool_pose_matrix is None
                    else Pose6D.from_matrix(tool_pose_matrix)
                ),
                tool_wrench=tool_wrench,
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
                    "kinematics_mode": self.kinematics_mode,
                    "arm_tip_pose_world": tip_pose_matrix.copy(),
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
                "kinematics_mode": self.kinematics_mode,
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

    def _reset_tool_wrench_state(self) -> None:
        self._tool_wrench_bias.clear()
        self._tool_wrench_filtered.clear()
        self._tool_wrench_time_s.clear()
        for arm_name, attachment in self.attachment_configs.items():
            sensor_config = attachment.force_torque_sensor
            if sensor_config is None:
                continue
            raw = self._raw_tool_wrench(arm_name)
            sensor_pose = self.physics.get_site_pose(f"{arm_name}_ft_sensor_site")
            if raw is None or sensor_pose is None:
                continue
            gravity = self._expected_gravity_wrench_raw(
                attachment,
                sensor_pose,
            )
            self._tool_wrench_bias[arm_name] = (
                raw - gravity if sensor_config.tare_on_reset else np.zeros(6)
            )
            self._tool_wrench_filtered[arm_name] = np.zeros(6, dtype=float)
            self._tool_wrench_time_s[arm_name] = float(self.physics.data.time)

    def _tool_wrench_state(
        self,
        arm_name: str,
        *,
        sensor_pose_matrix: np.ndarray | None,
    ) -> ToolWrenchState | None:
        attachment = self.attachment_configs.get(arm_name)
        if attachment is None or attachment.force_torque_sensor is None:
            return None
        raw = self._raw_tool_wrench(arm_name)
        if raw is None or sensor_pose_matrix is None:
            return None
        sensor_config = attachment.force_torque_sensor
        gravity = self._expected_gravity_wrench_raw(
            attachment,
            sensor_pose_matrix,
        )
        bias = self._tool_wrench_bias.get(arm_name, np.zeros(6, dtype=float))
        corrected = sensor_config.output_sign * (raw - bias - gravity)
        now = float(self.physics.data.time)
        previous_time = self._tool_wrench_time_s.get(arm_name, now)
        dt = max(0.0, now - previous_time)
        previous = self._tool_wrench_filtered.get(arm_name)
        if previous is None or dt <= 0.0:
            filtered = corrected
        else:
            alpha = 1.0 - np.exp(-2.0 * np.pi * sensor_config.filter_cutoff_hz * dt)
            filtered = previous + alpha * (corrected - previous)
        self._tool_wrench_filtered[arm_name] = filtered.copy()
        self._tool_wrench_time_s[arm_name] = now

        rotation = np.asarray(sensor_pose_matrix[:3, :3], dtype=float)
        force_world = rotation @ filtered[:3]
        torque_world = rotation @ filtered[3:]
        saturated = bool(
            np.linalg.norm(filtered[:3]) > sensor_config.force_limit_n
            or np.linalg.norm(filtered[3:]) > sensor_config.torque_limit_nm
        )
        return ToolWrenchState(
            raw_force_sensor_n=raw[:3],
            raw_torque_sensor_nm=raw[3:],
            force_sensor_n=filtered[:3],
            torque_sensor_nm=filtered[3:],
            force_world_n=force_world,
            torque_world_nm=torque_world,
            sensor_pose_world=Pose6D.from_matrix(sensor_pose_matrix),
            tared=sensor_config.tare_on_reset,
            saturated=saturated,
        )

    def _raw_tool_wrench(self, arm_name: str) -> np.ndarray | None:
        force = self.physics.get_sensor_data(f"{arm_name}_ft_force")
        torque = self.physics.get_sensor_data(f"{arm_name}_ft_torque")
        if force is None or torque is None or force.shape != (3,) or torque.shape != (3,):
            return None
        return np.concatenate((force, torque))

    def _expected_gravity_wrench_raw(
        self,
        attachment: AttachmentConfig,
        sensor_pose_matrix: np.ndarray,
    ) -> np.ndarray:
        sensor_config = attachment.force_torque_sensor
        collision = attachment.collision
        if (
            sensor_config is None
            or not sensor_config.gravity_compensation
            or collision is None
        ):
            return np.zeros(6, dtype=float)
        rotation = np.asarray(sensor_pose_matrix[:3, :3], dtype=float)
        gravity_world = np.asarray(self.physics.model.opt.gravity, dtype=float)
        sensor_force_world = sensor_config.mass_kg * gravity_world
        tool_mass = float(attachment.mass_kg or 0.0)
        tool_force_world = tool_mass * gravity_world
        tool_offset_world = rotation @ np.asarray(collision.position, dtype=float)
        gravity_force_world = sensor_force_world + tool_force_world
        gravity_torque_world = np.cross(tool_offset_world, tool_force_world)
        inverse_sign = 1.0 / sensor_config.output_sign
        return inverse_sign * np.concatenate(
            (
                rotation.T @ gravity_force_world,
                rotation.T @ gravity_torque_world,
            )
        )


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
    if config.enforce_target_lead_limit:
        hard_force_limited_lead = hard_force_limit / actuator.kp
        max_target_lead = (
            hard_force_limited_lead
            if config.max_target_lead_m is None
            else np.minimum(config.max_target_lead_m, hard_force_limited_lead)
        )
    else:
        max_target_lead = config.max_target_lead_m
    return replace(
        config,
        max_target_lead_m=max_target_lead,
        soft_force_limit_n=soft_force_limit,
        hard_force_limit_n=hard_force_limit,
    )
