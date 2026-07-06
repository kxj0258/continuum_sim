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
    EngineNavigationLocalPathPlan,
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
    """Alternate mobile-base insertion with fixed-base local arm paths."""

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
        self._fixed_assembly = replace(
            assembly,
            base=replace(assembly.base, control_mode="fixed"),
        )
        self._waypoint_tolerance_m = float(waypoint_tolerance_m)
        self._controller_dt_s = float(controller_dt_s)
        self._local_paths = (
            plan.local_path_plans
            if plan.local_path_plans
            else (
                EngineNavigationLocalPathPlan(
                    name="endpoint_path",
                    path_type=spec.local_path_type,
                    at_fraction=1.0,
                    insertion_index=len(plan.insertion_base_poses) - 1,
                    insertion_target_world=(
                        plan.insertion_tip_waypoints_world[-1].copy()
                    ),
                    center_world=plan.observer_roi_world.copy(),
                    waypoints_world=plan.executor_waypoints_world.copy(),
                    is_terminal=True,
                ),
            )
        )
        self._local_path_by_insertion_index = {
            path.insertion_index: index
            for index, path in enumerate(self._local_paths)
        }
        self._completed_local_paths: set[int] = set()
        self._active_local_path_index = -1
        self._executor_subphase = ""
        self._tracking: WaypointTrackingController | None = None
        self._tendon_counts = {
            arm.name: arm.spatial_arm.tendon_count
            for arm in assembly.enabled_arms
        }
        self._initial_base_position: np.ndarray | None = None
        self._last_active_target = plan.pre_entry_base_pose.position.copy()
        self._last_active_target_kind = "base"

    @property
    def done(self) -> bool:
        return self.phase in ("complete", "failed")

    @property
    def failed(self) -> bool:
        return self.phase == "failed"

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        """Compute one staged command and expose phase diagnostics."""

        if self._initial_base_position is None:
            self._initial_base_position = state.base.pose.position.copy()
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
            elif self._pending_local_path_index() is not None:
                self._start_local_path(self._pending_local_path_index())
                return self._executor_command(state, clearance)
            elif self.insertion_index < len(self.plan.insertion_base_poses) - 1:
                self.insertion_index += 1
            else:
                self._set_phase("complete")
                self.terminal_reason = "completed"
        command = RobotSystemCommand.zeros(self._tendon_counts)
        return RobotSystemCommand(
            base_twist_world=twist,
            arms=command.arms,
            metadata=self._metadata(
                target_position=target.position,
                active_target=target.position,
                active_target_kind="base",
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
        if self._tracking is None or self._active_local_path_index < 0:
            raise RuntimeError("Executor navigation has no active local path.")
        tracked = self._tracking.compute_command(state)
        active_target = np.asarray(
            tracked.metadata.get(
                "executor_target_world",
                self._last_active_target,
            ),
            dtype=float,
        )
        if self._tracking.done:
            active_path = self._local_paths[self._active_local_path_index]
            if self._executor_subphase == "path" and not active_path.is_terminal:
                self._start_rejoin(active_path)
                return self._executor_command(state, clearance)
            self._completed_local_paths.add(self._active_local_path_index)
            if active_path.is_terminal:
                self._set_phase("complete")
                self.terminal_reason = "completed"
                return self._executor_result(
                    state,
                    tracked,
                    active_target,
                    clearance,
                )
            self._tracking = None
            self._executor_subphase = ""
            self._active_local_path_index = -1
            self._set_phase("base_insertion")
            if self.insertion_index < len(self.plan.insertion_base_poses) - 1:
                self.insertion_index += 1
            return self._base_command(state, clearance)
        return self._executor_result(
            state,
            tracked,
            active_target,
            clearance,
        )

    def _executor_result(
        self,
        state: RobotSystemState,
        tracked: RobotSystemCommand,
        active_target: np.ndarray,
        clearance: float,
    ) -> RobotSystemCommand:
        metadata = {
            **tracked.metadata,
            **self._metadata(
                target_position=state.base.pose.position,
                active_target=active_target,
                active_target_kind="executor",
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

    def _pending_local_path_index(self) -> int | None:
        event_index = self._local_path_by_insertion_index.get(self.insertion_index)
        if event_index is None or event_index in self._completed_local_paths:
            return None
        return event_index

    def _start_local_path(self, event_index: int) -> None:
        path = self._local_paths[event_index]
        self._active_local_path_index = event_index
        self._executor_subphase = "path"
        self._tracking = self._make_tracker(
            path.waypoints_world,
            observer_roi_world=path.center_world,
        )
        self._set_phase("executor_navigation")

    def _start_rejoin(self, path: EngineNavigationLocalPathPlan) -> None:
        self._executor_subphase = "rejoin"
        self._tracking = self._make_tracker(
            path.insertion_target_world[None, :],
            observer_roi_world=path.center_world,
        )
        self._phase_steps = 0

    def _make_tracker(
        self,
        waypoints_world: np.ndarray,
        *,
        observer_roi_world: np.ndarray,
    ) -> WaypointTrackingController:
        return WaypointTrackingController(
            self._fixed_assembly,
            waypoints_world,
            waypoint_tolerance_m=self._waypoint_tolerance_m,
            observer_roi_world=observer_roi_world,
            target_advance_mode="tolerance",
            controller_dt_s=self._controller_dt_s,
            executor_position_gain=3.0,
            observer_position_gain=3.0,
            feedforward_speed_mps=0.0,
            max_target_speed_mps=0.02,
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
        active_target: np.ndarray | None = None,
        active_target_kind: str | None = None,
        position_error: float,
        orientation_error: float,
        clearance: float,
    ) -> dict[str, object]:
        if active_target is not None:
            self._last_active_target = np.asarray(active_target, dtype=float).copy()
        if active_target_kind is not None:
            self._last_active_target_kind = active_target_kind
        progress_denominator = max(len(self.plan.insertion_base_poses) - 1, 1)
        progress = (
            float(self.insertion_index / progress_denominator)
            if self.phase in ("base_insertion", "executor_navigation", "complete")
            else 0.0
        )
        active_path = (
            None
            if self._active_local_path_index < 0
            else self._local_paths[self._active_local_path_index]
        )
        displayed_executor_path = (
            self.plan.executor_waypoints_world
            if active_path is None
            else active_path.waypoints_world
        )
        displayed_observer_roi = (
            self.plan.observer_roi_world
            if active_path is None
            else active_path.center_world
        )
        return {
            "task_type": "engine_navigation",
            "engine_navigation_phase": self.phase,
            "engine_navigation_terminal_reason": self.terminal_reason,
            "engine_navigation_insertion_index": self.insertion_index,
            "engine_navigation_progress": progress,
            "engine_navigation_local_path_index": self._active_local_path_index,
            "engine_navigation_local_path_name": (
                "" if active_path is None else active_path.name
            ),
            "engine_navigation_local_path_type": (
                "" if active_path is None else active_path.path_type
            ),
            "engine_navigation_local_path_fraction": (
                np.nan if active_path is None else active_path.at_fraction
            ),
            "engine_navigation_executor_subphase": self._executor_subphase,
            "engine_navigation_pre_entry_target_m": (
                self.plan.pre_entry_tip_world.copy()
            ),
            "engine_navigation_base_path_m": np.asarray(
                [
                    (
                        self.plan.pre_entry_base_pose.position
                        if self._initial_base_position is None
                        else self._initial_base_position
                    ),
                    self.plan.pre_entry_base_pose.position,
                    *(pose.position for pose in self.plan.insertion_base_poses),
                ],
                dtype=float,
            ),
            "engine_navigation_insertion_path_m": (
                self.plan.insertion_tip_waypoints_world.copy()
            ),
            "engine_navigation_executor_path_m": (
                displayed_executor_path.copy()
            ),
            "engine_navigation_executor_paths_m": tuple(
                path.waypoints_world.copy() for path in self._local_paths
            ),
            "engine_navigation_observer_roi_m": (
                displayed_observer_roi.copy()
            ),
            "engine_navigation_active_target_m": self._last_active_target.copy(),
            "engine_navigation_active_target_kind": (
                self._last_active_target_kind
            ),
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
