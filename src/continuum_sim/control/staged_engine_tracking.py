"""Reaction-isolated mobile-base approach for engine trajectory tracking."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.mobile_base_pose_control import MobileBasePoseController
from continuum_sim.control.scenario_controllers import (
    TimedTrajectoryTrackingController,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class StagedEngineTrackingController:
    """Move the base once, then track with fixed-base tendon control."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        waypoints_world: np.ndarray,
        *,
        trajectory_duration_s: float,
        waypoint_tolerance_m: float = 1.0e-3,
        observer_roi_world: np.ndarray | None = None,
        observer_control_mode: str = "tracking",
        loop: bool = False,
        scene_query: EngineSceneQueryProtocol | None = None,
        approach_mask: np.ndarray | None = None,
        source_waypoint_index: np.ndarray | None = None,
        executor_position_gain: float = 4.0,
        observer_position_gain: float = 5.0,
        max_target_speed_mps: float | None = None,
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        enforce_backend_tendon_limits: bool = False,
        coordinated_config: CoordinatedTrackingConfig | None = None,
        base_position_gain: float = 1.5,
        base_orientation_gain: float = 2.0,
        base_position_tolerance_m: float = 0.005,
        base_orientation_tolerance_rad: float = 0.035,
    ) -> None:
        if assembly.base.control_mode == "fixed":
            raise ValueError("Staged engine tracking requires a mobile base.")
        waypoints = np.asarray(waypoints_world, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
            raise ValueError("waypoints_world must have shape (N, 3) with N > 0.")
        self.assembly = assembly
        self.phase = "base_approach"
        self._executor_name = _single_role_name(assembly, "executor")
        self._first_waypoint = waypoints[0].copy()
        self._base_target: Pose6D | None = None
        self._base_position_tolerance_m = float(base_position_tolerance_m)
        self._base_orientation_tolerance_rad = float(
            base_orientation_tolerance_rad
        )
        self._pose_controller = MobileBasePoseController(
            position_gain=base_position_gain,
            orientation_gain=base_orientation_gain,
        )
        self._tendon_counts = {
            arm.name: arm.spatial_arm.tendon_count
            for arm in assembly.enabled_arms
        }
        fixed_assembly = replace(
            assembly,
            base=replace(assembly.base, control_mode="fixed"),
        )
        self._tracking = TimedTrajectoryTrackingController(
            fixed_assembly,
            waypoints,
            trajectory_duration_s=trajectory_duration_s,
            waypoint_tolerance_m=waypoint_tolerance_m,
            observer_roi_world=observer_roi_world,
            observer_control_mode=observer_control_mode,
            loop=loop,
            scene_query=scene_query,
            approach_mask=approach_mask,
            source_waypoint_index=source_waypoint_index,
            executor_position_gain=executor_position_gain,
            observer_position_gain=observer_position_gain,
            max_target_speed_mps=max_target_speed_mps,
            solver_config=solver_config,
            enforce_backend_tendon_limits=enforce_backend_tendon_limits,
            coordinated_config=coordinated_config,
        )

    @property
    def done(self) -> bool:
        return self.phase == "complete"

    @property
    def terminal_reason(self) -> str:
        return "duration_elapsed" if self.done else ""

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._tracking.last_diagnostics

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        """Compute a base-only approach or fixed-base tendon command."""

        if self._base_target is None:
            tip_position = state.arms[
                self._executor_name
            ].tip_pose_world.position
            self._base_target = Pose6D(
                position=(
                    state.base.pose.position
                    + self._first_waypoint
                    - tip_position
                ),
                quat=state.base.pose.quat,
            )
        if self.phase == "base_approach":
            return self._base_approach_command(state)
        return self._tracking_command(state)

    def _base_approach_command(
        self,
        state: RobotSystemState,
    ) -> RobotSystemCommand:
        if self._base_target is None:
            raise RuntimeError("Base target has not been initialized.")
        twist, position_error, orientation_error = (
            self._pose_controller.compute_twist(
                state.base.pose,
                self._base_target,
                max_linear_speed=None,
                max_angular_speed=None,
            )
        )
        reached = (
            position_error <= self._base_position_tolerance_m
            and orientation_error <= self._base_orientation_tolerance_rad
        )
        if reached:
            self.phase = "tracking"
            return self._tracking_command(state)

        zero = RobotSystemCommand.zeros(self._tendon_counts)
        tip_position = state.arms[self._executor_name].tip_pose_world.position
        return RobotSystemCommand(
            base_twist_world=twist,
            arms=zero.arms,
            metadata={
                "task_type": "tracking",
                "tracking_mode": "time",
                "engine_tracking_phase": self.phase,
                "trajectory_time_s": 0.0,
                "trajectory_duration_s": self._tracking.trajectory_duration_s,
                "waypoint_index": 0,
                "source_waypoint_index": 0,
                "trajectory_local_fraction": 0.0,
                "executor_target_world": tip_position.copy(),
                "executor_error_m": np.nan,
                "executor_feedforward_velocity_world": np.zeros(3, dtype=float),
                "achieved_waypoint_index": -1,
                "achieved_waypoint_error_m": np.nan,
                "waypoint_advanced": False,
                "tracking_complete": False,
                "tracking_approach": True,
                "base_target_position_m": self._base_target.position.copy(),
                "base_position_error_m": position_error,
                "base_orientation_error_rad": orientation_error,
                "tendon_reaction_isolated": True,
            },
        )

    def _tracking_command(
        self,
        state: RobotSystemState,
    ) -> RobotSystemCommand:
        tracked = self._tracking.compute_command(state)
        if self._tracking.done:
            self.phase = "complete"
        if self._base_target is None:
            raise RuntimeError("Base target has not been initialized.")
        return RobotSystemCommand(
            base_twist_world=np.zeros(6, dtype=float),
            arms=tracked.arms,
            metadata={
                **tracked.metadata,
                "engine_tracking_phase": self.phase,
                "base_target_position_m": self._base_target.position.copy(),
                "base_position_error_m": float(
                    np.linalg.norm(
                        self._base_target.position - state.base.pose.position
                    )
                ),
                "base_orientation_error_rad": 0.0,
                "tendon_reaction_isolated": True,
            },
        )


def _single_role_name(assembly: RobotAssemblyConfig, role: str) -> str:
    matches = [arm.name for arm in assembly.enabled_arms if arm.role == role]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one enabled {role!r} arm, got {matches}."
        )
    return matches[0]
