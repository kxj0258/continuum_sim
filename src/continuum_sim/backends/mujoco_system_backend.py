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
        self._base_state = MobileBaseState(
            pose=assembly.base.initial_pose,
            locked=assembly.base.control_mode == "fixed",
        )
        self._last_applied_rates = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in assembly.enabled_arms
        }

    @classmethod
    def from_config(
        cls,
        mujoco_config: str | Path | MujocoConfig,
        assembly_config: str | Path | RobotAssemblyConfig,
        *,
        xml_path: str | Path | None = None,
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
        return cls(resolved_mujoco, resolved_assembly, xml_path=xml_path)

    def reset_system(self) -> RobotSystemState:
        self.physics.reset()
        for integrator in self._integrators.values():
            integrator.reset()
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
            step = self._integrators[arm_name].step(
                command.arms[arm_name].tendon_rate_mps,
                dt,
                raw_debug=(
                    command.arms[arm_name].control_space == "raw_tendon_debug"
                ),
                actual_displacement_m=actual_tendon_displacement[tendon_slice],
                enforce_limits=enforce_tendon_limits,
                target_mode=tendon_target_mode,
            )
            tendon_target[tendon_slice] = step.displacement_m
            self._last_applied_rates[arm_name] = step.applied_rate_mps
            saturation[arm_name] = {
                "rate": step.rate_saturated,
                "displacement": step.displacement_saturated,
                "common_scale": step.common_scale,
                "compatibility_residual_mps": step.compatibility_residual_mps,
                "raw_debug": step.raw_debug,
                "target_mode": tendon_target_mode,
            }

        base_rpy = _pose_to_xyz_rpy(self._base_state.pose)
        raw_control = (
            tendon_target
            if self.assembly.base.control_mode == "fixed"
            else np.concatenate((tendon_target, base_rpy))
        )
        self.physics.step(raw_control, n_substeps=n_substeps)
        state = self.get_system_state()
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
            arms[arm.name] = ArmSystemState(
                name=arm.name,
                role=arm.role,
                tip_pose_world=Pose6D.from_matrix(self._site_pose(tip_name)),
                segment_poses_world=np.asarray(
                    [self._site_pose(name) for name in segment_names],
                    dtype=float,
                ),
                tendon_displacement_m=tendon_displacement[tendon_slice],
                tendon_velocity_mps=(
                    tendon_velocity[tendon_slice]
                    if tendon_velocity.size
                    else self._last_applied_rates[arm.name]
                ),
                tendon_target_m=self._integrators[arm.name].displacement_m,
                actuator_force_n=(
                    actuator_force[tendon_slice]
                    if actuator_force.size
                    else np.zeros_like(tendon_displacement[tendon_slice])
                ),
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
