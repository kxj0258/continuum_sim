"""Analytic PCC backend for named direct-tendon system scenarios."""

from __future__ import annotations

import numpy as np

from continuum_sim.control.mobile_base_controller import (
    MobileBaseCommand,
    MobileBaseState,
    integrate_base_pose,
)
from continuum_sim.control.tendon_rate_control import (
    CompatibleTendonRateIntegrator,
    TendonRateLimits,
    resolve_tendon_target_policy,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import (
    ArmSystemState,
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


class AnalyticSystemBackend:
    """Deterministic bending-compatible PCC system backend."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        *,
        samples_per_segment: int = 12,
    ) -> None:
        if samples_per_segment < 2:
            raise ValueError("samples_per_segment must be at least 2.")
        self.assembly = assembly
        self.layout = ControlLayout.from_assembly(assembly)
        self.samples_per_segment = samples_per_segment
        self._time_s = 0.0
        self._base_state = self._initial_base_state()
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
        self._last_rates = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in assembly.enabled_arms
        }

    def reset_system(self) -> RobotSystemState:
        self._time_s = 0.0
        self._base_state = self._initial_base_state()
        for integrator in self._integrators.values():
            integrator.reset()
        self._last_rates = {
            arm.name: np.zeros(arm.spatial_arm.tendon_count, dtype=float)
            for arm in self.assembly.enabled_arms
        }
        return self.get_system_state()

    def step_system(
        self,
        command: RobotSystemCommand,
        *,
        dt: float,
        n_substeps: int = 1,
    ) -> RobotSystemState:
        del n_substeps
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if set(command.arms) != set(self.layout.arms):
            raise ValueError("Command arms must exactly match the assembly layout.")
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
        clipped_position = np.clip(
            self._base_state.pose.position,
            self.assembly.base.position_min_m,
            self.assembly.base.position_max_m,
        )
        self._base_state = MobileBaseState(
            pose=Pose6D(clipped_position, self._base_state.pose.quat),
            locked=self._base_state.locked,
            last_twist=self._base_state.last_twist,
        )
        enforce_tendon_limits, tendon_target_mode = resolve_tendon_target_policy(
            command.metadata
        )
        saturation: dict[str, dict[str, object]] = {}
        for arm_name, arm_command in command.arms.items():
            actual_before = self._integrators[arm_name].displacement_m
            previous_target = actual_before.copy()
            step = self._integrators[arm_name].step(
                arm_command.tendon_rate_mps,
                dt,
                raw_debug=arm_command.control_space == "raw_tendon_debug",
                actual_displacement_m=actual_before,
                enforce_limits=enforce_tendon_limits,
                target_mode=tendon_target_mode,
            )
            self._last_rates[arm_name] = step.applied_rate_mps
            actual_after = step.displacement_m
            saturation[arm_name] = {
                "rate": step.rate_saturated,
                "displacement": step.displacement_saturated,
                "common_scale": step.common_scale,
                "requested_rate_mps": step.requested_rate_mps.copy(),
                "applied_rate_mps": step.applied_rate_mps.copy(),
                "applied_rate_semantics": "constrained_command_rate",
                "target_rate_fd_mps": (
                    (step.displacement_m - previous_target) / float(dt)
                ),
                "realized_rate_fd_mps": (
                    (actual_after - actual_before) / float(dt)
                ),
                "target_m": step.displacement_m.copy(),
                "target_error_m": step.displacement_m - actual_after,
                "compatibility_residual_mps": step.compatibility_residual_mps,
                "raw_debug": step.raw_debug,
                "target_mode": tendon_target_mode,
            }
        self._time_s += float(dt)
        state = self.get_system_state()
        return RobotSystemState(
            time_s=state.time_s,
            base=state.base,
            arms=state.arms,
            metadata={**state.metadata, "saturation": saturation},
        )

    def get_system_state(self) -> RobotSystemState:
        arms: dict[str, ArmSystemState] = {}
        for arm in self.assembly.enabled_arms:
            displacement = self._integrators[arm.name].displacement_m
            model = self.layout.bending_models[arm.name]
            bending = model.estimate(displacement)
            q = model.to_q(bending)
            fk = forward_kinematics(
                q,
                arm.spatial_arm.params,
                samples_per_segment=self.samples_per_segment,
            )
            world_mount = self._base_state.pose.compose(arm.mount_pose)
            tip_world = world_mount.as_matrix() @ fk.tip_pose
            segment_world = np.asarray(
                [world_mount.as_matrix() @ pose for pose in fk.segment_poses],
                dtype=float,
            )
            arms[arm.name] = ArmSystemState(
                name=arm.name,
                role=arm.role,
                tip_pose_world=Pose6D.from_matrix(tip_world),
                segment_poses_world=segment_world,
                tendon_displacement_m=displacement,
                tendon_velocity_mps=self._last_rates[arm.name],
                tendon_target_m=displacement,
                actuator_force_n=np.zeros_like(displacement),
                centerline_world=world_mount.transform_points(fk.centerline),
                metadata={
                    "q": q,
                    "bending": bending,
                    "compatibility_residual_m": model.residual(displacement),
                    "compatibility_residual_norm_m": model.residual_norm(displacement),
                    "backend": "analytic",
                },
            )
        return RobotSystemState(
            time_s=self._time_s,
            base=BaseSystemState(
                pose=self._base_state.pose,
                twist_world=self._base_state.last_twist,
            ),
            arms=arms,
            metadata={"backend": "analytic", "control": "bending_compatible"},
        )

    def _initial_base_state(self) -> MobileBaseState:
        return MobileBaseState(
            pose=self.assembly.base.initial_pose,
            locked=self.assembly.base.control_mode == "fixed",
        )
