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
        kinematics_mode: PCCKinematicsMode = DEFAULT_PCC_KINEMATICS_MODE,
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

        actual_tendon_displacement = self.physics.get_tendon_length()
        actual_actuator_force = self.physics.get_actuator_force()
        previous_tendon_target = {
            arm_name: target.copy()
            for arm_name, target in self._tendon_execution.last_tendon_targets.items()
        }
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
        self.physics.step(raw_control, n_substeps=n_substeps)
        state = self.get_system_state()
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
