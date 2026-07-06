"""Staged mobile-base and dual-arm navigation through an engine scene."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from continuum_sim.control.mobile_base_pose_control import (
    MobileBasePoseController,
)
from continuum_sim.control.scenario_controllers import WaypointTrackingController
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState
from continuum_sim.tasks.engine_navigation import (
    EngineNavigationPlan,
    EngineNavigationSpec,
)


ENGINE_NAVIGATION_PHASES = (
    "base_approach",
    "base_insertion",
    "executor_navigation",
    "complete",
    "failed",
)


class StagedEngineNavigationController:
    """Approach and insert with the base, then navigate with fixed-base arms."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        plan: EngineNavigationPlan,
        spec: EngineNavigationSpec,
        *,
        scene_query: EngineSceneQueryProtocol | None,
        waypoint_tolerance_m: float,
        min_clearance_m: float,
        terminate_on_clearance_violation: bool,
        controller_dt_s: float = 0.02,
    ) -> None:
        if assembly.base.control_mode == "fixed":
            raise ValueError("Staged engine navigation requires a mobile base.")
        self.assembly = assembly
        self.plan = plan
        self.spec = spec
        self.scene_query = scene_query
        self.min_clearance_m = float(min_clearance_m)
        self.terminate_on_clearance_violation = bool(
            terminate_on_clearance_violation
        )
        self.phase = "base_approach"
        self.terminal_reason = ""
        self.insertion_index = 0
        self._phase_steps = 0
        self._pose_controller = MobileBasePoseController(
            position_gain=spec.base_position_gain,
            orientation_gain=spec.base_orientation_gain,
        )
        fixed_assembly = replace(
            assembly,
            base=replace(assembly.base, control_mode="fixed"),
        )
        self._tracking = WaypointTrackingController(
            fixed_assembly,
            plan.executor_waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            observer_roi_world=plan.observer_roi_world,
            target_advance_mode="tolerance",
            controller_dt_s=controller_dt_s,
            executor_position_gain=3.0,
            observer_position_gain=3.0,
            feedforward_speed_mps=0.0,
            max_target_speed_mps=0.02,
        )
        self._tendon_counts = {
            arm.name: arm.spatial_arm.tendon_count
            for arm in assembly.enabled_arms
        }

    @property
    def done(self) -> bool:
        return self.phase in ("complete", "failed")

    @property
    def failed(self) -> bool:
        return self.phase == "failed"

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        """Compute one staged command and expose phase diagnostics."""

        clearance = self._minimum_clearance(state)
        if (
            self.terminate_on_clearance_violation
            and np.isfinite(clearance)
            and clearance < self.min_clearance_m
        ):
            self._fail("clearance_violation")
        if not self.done:
            self._phase_steps += 1
            if self._phase_steps > self.spec.phase_timeout_steps:
                self._fail("phase_timeout")
        if self.done:
            return self._zero_command(
                state,
                clearance=clearance,
                position_error=0.0,
                orientation_error=0.0,
            )
        if self.phase == "executor_navigation":
            return self._executor_command(state, clearance)
        return self._base_command(state, clearance)

    def _base_command(
        self,
        state: RobotSystemState,
        clearance: float,
    ) -> RobotSystemCommand:
        target = (
            self.plan.pre_entry_base_pose
            if self.phase == "base_approach"
            else self.plan.insertion_base_poses[self.insertion_index]
        )
        twist, position_error, orientation_error = (
            self._pose_controller.compute_twist(
                state.base.pose,
                target,
                max_linear_speed=self.assembly.base.max_linear_speed_mps,
                max_angular_speed=self.assembly.base.max_angular_speed_rad_s,
            )
        )
        reached = (
            position_error <= self.spec.base_position_tolerance_m
            and orientation_error <= self.spec.base_orientation_tolerance_rad
        )
        if reached:
            if self.phase == "base_approach":
                self._set_phase("base_insertion")
                self.insertion_index = 0
            elif self.insertion_index < len(self.plan.insertion_base_poses) - 1:
                self.insertion_index += 1
            else:
                self._set_phase("executor_navigation")
                return self._executor_command(state, clearance)
        command = RobotSystemCommand.zeros(self._tendon_counts)
        return RobotSystemCommand(
            base_twist_world=twist,
            arms=command.arms,
            metadata=self._metadata(
                target_position=target.position,
                position_error=position_error,
                orientation_error=orientation_error,
                clearance=clearance,
            ),
        )

    def _executor_command(
        self,
        state: RobotSystemState,
        clearance: float,
    ) -> RobotSystemCommand:
        tracked = self._tracking.compute_command(state)
        if self._tracking.done:
            self._set_phase("complete")
            self.terminal_reason = "completed"
            return RobotSystemCommand(
                base_twist_world=np.zeros(6, dtype=float),
                arms=tracked.arms,
                metadata={
                    **tracked.metadata,
                    **self._metadata(
                        target_position=state.base.pose.position,
                        position_error=0.0,
                        orientation_error=0.0,
                        clearance=clearance,
                    ),
                },
            )
        metadata = {
            **tracked.metadata,
            **self._metadata(
                target_position=state.base.pose.position,
                position_error=0.0,
                orientation_error=0.0,
                clearance=clearance,
            ),
        }
        return RobotSystemCommand(
            base_twist_world=np.zeros(6, dtype=float),
            arms=tracked.arms,
            metadata=metadata,
        )

    def _minimum_clearance(self, state: RobotSystemState) -> float:
        if self.scene_query is None:
            return float("inf")
        return float(
            min(
                self.scene_query.nearest_distance(
                    arm.tip_pose_world.position
                ).distance_m
                for arm in state.arms.values()
            )
        )

    def _zero_command(
        self,
        state: RobotSystemState,
        *,
        clearance: float,
        position_error: float,
        orientation_error: float,
    ) -> RobotSystemCommand:
        command = RobotSystemCommand.zeros(self._tendon_counts)
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata=self._metadata(
                target_position=state.base.pose.position,
                position_error=position_error,
                orientation_error=orientation_error,
                clearance=clearance,
            ),
        )

    def _metadata(
        self,
        *,
        target_position: np.ndarray,
        position_error: float,
        orientation_error: float,
        clearance: float,
    ) -> dict[str, object]:
        progress_denominator = max(len(self.plan.insertion_base_poses) - 1, 1)
        progress = (
            float(self.insertion_index / progress_denominator)
            if self.phase in ("base_insertion", "executor_navigation", "complete")
            else 0.0
        )
        return {
            "task_type": "engine_navigation",
            "engine_navigation_phase": self.phase,
            "engine_navigation_terminal_reason": self.terminal_reason,
            "engine_navigation_insertion_index": self.insertion_index,
            "engine_navigation_progress": progress,
            "base_target_position_m": np.asarray(target_position, dtype=float).copy(),
            "base_position_error_m": float(position_error),
            "base_orientation_error_rad": float(orientation_error),
            "min_clearance_m": float(clearance),
        }

    def _set_phase(self, phase: str) -> None:
        if phase not in ENGINE_NAVIGATION_PHASES:
            raise ValueError(f"Unknown engine navigation phase {phase!r}.")
        self.phase = phase
        self._phase_steps = 0

    def _fail(self, reason: str) -> None:
        self.phase = "failed"
        self.terminal_reason = reason
