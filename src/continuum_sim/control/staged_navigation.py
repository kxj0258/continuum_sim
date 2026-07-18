"""Per-waypoint mobile-base approach followed by fixed-base navigation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from continuum_sim.control.base_approach_stage import BaseApproachStage
from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.scenario_controllers import (
    NavigationController,
    OnlineReachabilityConfig,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.model.base_pose import (
    Pose6D,
    quaternion_error_rotation_vector,
)
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class StagedNavigationController:
    """Move the base near each waypoint, then servo that waypoint locally."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        waypoints_world: np.ndarray,
        *,
        scene_query: EngineSceneQueryProtocol,
        waypoint_tolerance_m: float,
        min_clearance_m: float,
        terminate_on_clearance_violation: bool,
        waypoint_orientations_world_wxyz: np.ndarray | None = None,
        waypoint_directions_world: np.ndarray | None = None,
        orientation_tolerance_rad: float = 0.08,
        observer_roi_world: np.ndarray | None = None,
        observer_control_mode: str = "tracking",
        target_advance_mode: str = "tolerance",
        controller_dt_s: float = 0.02,
        advance_time_s: float | None = None,
        advance_steps: int | None = None,
        max_steps_per_waypoint: int | None = None,
        executor_position_gain: float = 4.0,
        observer_position_gain: float = 5.0,
        feedforward_speed_mps: float = 0.0,
        max_target_speed_mps: float | None = None,
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        enforce_backend_tendon_limits: bool = False,
        coordinated_config: CoordinatedTrackingConfig | None = None,
        control_type: str = "whole_body",
        cbf_gain: float = 4.0,
        cbf_influence_distance_m: float | None = None,
        online_reachability: OnlineReachabilityConfig | None = None,
        base_position_gain: float = 1.5,
        base_orientation_gain: float = 2.0,
        base_position_tolerance_m: float = 0.005,
        base_orientation_tolerance_rad: float = 0.035,
        base_approach_standoff_m: float = 0.030,
        base_approach_z_bias: float = 1.0,
        intermediate_waypoints_per_waypoint: int = 0,
    ) -> None:
        if assembly.base.control_mode == "fixed":
            raise ValueError("Staged navigation requires a mobile base.")
        waypoints = np.asarray(waypoints_world, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
            raise ValueError("waypoints_world must have shape (N, 3) with N > 0.")
        if scene_query is None:
            raise ValueError("Staged navigation requires a scene query.")
        self.assembly = assembly
        self.scene_query = scene_query
        self.phase = "base_approach"
        self._terminal_reason = ""
        self._executor_name = _single_role_name(assembly, "executor")
        self._waypoints = waypoints.copy()
        self._orientations = _waypoint_quaternions(
            waypoint_orientations_world_wxyz,
            waypoints.shape[0],
        )
        self._directions = _waypoint_directions(
            waypoint_directions_world,
            waypoints.shape[0],
        )
        self._active_index = 0
        self._base_target: Pose6D | None = None
        self._base_target_index = -1
        self._waypoint_tolerance_m = float(waypoint_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)
        if (
            not np.isfinite(self._orientation_tolerance_rad)
            or self._orientation_tolerance_rad < 0.0
        ):
            raise ValueError("orientation_tolerance_rad must be non-negative.")
        self._min_clearance_m = float(min_clearance_m)
        self._terminate_on_clearance_violation = bool(
            terminate_on_clearance_violation
        )
        self._base_approach_standoff_m = float(base_approach_standoff_m)
        self._base_approach_z_bias = float(base_approach_z_bias)
        self._intermediate_waypoints_per_waypoint = int(
            intermediate_waypoints_per_waypoint
        )
        if (
            not np.isfinite(self._base_approach_standoff_m)
            or self._base_approach_standoff_m < 0.0
        ):
            raise ValueError("base_approach_standoff_m must be non-negative and finite.")
        if not np.isfinite(self._base_approach_z_bias):
            raise ValueError("base_approach_z_bias must be finite.")
        if self._intermediate_waypoints_per_waypoint < 0:
            raise ValueError(
                "intermediate_waypoints_per_waypoint must be non-negative."
            )
        self._base_stage = BaseApproachStage(
            position_gain=base_position_gain,
            orientation_gain=base_orientation_gain,
            position_tolerance_m=base_position_tolerance_m,
            orientation_tolerance_rad=base_orientation_tolerance_rad,
            standoff_m=base_approach_standoff_m,
            z_bias=base_approach_z_bias,
        )
        self._tendon_counts = {
            arm.name: arm.spatial_arm.tendon_count
            for arm in assembly.enabled_arms
        }
        fixed_assembly = replace(
            assembly,
            base=replace(assembly.base, control_mode="fixed"),
        )
        self._fixed_assembly = fixed_assembly
        self._tracker_kwargs = {
            "scene_query": scene_query,
            "waypoint_tolerance_m": waypoint_tolerance_m,
            "min_clearance_m": min_clearance_m,
            "terminate_on_clearance_violation": terminate_on_clearance_violation,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "observer_roi_world": observer_roi_world,
            "observer_control_mode": observer_control_mode,
            "target_advance_mode": target_advance_mode,
            "controller_dt_s": controller_dt_s,
            "advance_time_s": advance_time_s,
            "advance_steps": advance_steps,
            "max_steps_per_waypoint": max_steps_per_waypoint,
            "executor_position_gain": executor_position_gain,
            "observer_position_gain": observer_position_gain,
            "feedforward_speed_mps": feedforward_speed_mps,
            "max_target_speed_mps": max_target_speed_mps,
            "solver_config": solver_config,
            "enforce_backend_tendon_limits": enforce_backend_tendon_limits,
            "coordinated_config": coordinated_config,
            "control_type": control_type,
            "cbf_gain": cbf_gain,
            "cbf_influence_distance_m": cbf_influence_distance_m,
            "online_reachability": online_reachability,
        }
        self._tracking: NavigationController | None = None

    @property
    def done(self) -> bool:
        return self.phase in ("complete", "failed")

    @property
    def terminal_reason(self) -> str:
        if self._terminal_reason:
            return self._terminal_reason
        if self._tracking is None:
            return ""
        return self._tracking.terminal_reason

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        """Compute a base-only approach or fixed-base navigation command."""

        self._ensure_base_target(state)
        clearance = self._minimum_clearance(state)
        if (
            self._terminate_on_clearance_violation
            and np.isfinite(clearance.distance_m)
            and clearance.distance_m < self._min_clearance_m
        ):
            self.phase = "failed"
            self._terminal_reason = "clearance_violation"
            return self._zero_command(state, clearance)
        if self.phase == "base_approach":
            return self._base_approach_command(state, clearance)
        return self._tracking_command(state, clearance)

    def _ensure_base_target(self, state: RobotSystemState) -> None:
        if (
            self._base_target is not None
            and self._base_target_index == self._active_index
        ):
            return
        target = self._base_approach_tip_target(self._active_index)
        target_base_quat = state.base.pose.quat
        self._base_target = self._base_stage.base_pose_for_tip_target(
            state,
            self._executor_name,
            target,
            target_base_quat,
        )
        self._base_target_index = self._active_index

    def _base_approach_command(
        self,
        state: RobotSystemState,
        clearance,
    ) -> RobotSystemCommand:
        if self._base_target is None:
            raise RuntimeError("Base target has not been initialized.")
        approach = self._base_stage.compute_to_pose(
            state,
            self._base_target,
            max_linear_speed=self.assembly.base.max_linear_speed_mps,
            max_angular_speed=self.assembly.base.max_angular_speed_rad_s,
            ignore_orientation=True,
        )
        if approach.reached:
            self.phase = "tracking"
            self._tracking = self._make_tracker(self._active_index, state)
            return self._tracking_command(state, clearance)
        zero = RobotSystemCommand.zeros(self._tendon_counts)
        tip_position = state.arms[self._executor_name].tip_pose_world.position
        approach_tip_target = self._base_approach_tip_target(self._active_index)
        approach_direction = self._base_approach_direction(self._active_index)
        return RobotSystemCommand(
            base_twist_world=approach.twist_world,
            arms=zero.arms,
            metadata={
                "task_type": "navigation",
                "staged_navigation_phase": self.phase,
                "staged_navigation_waypoint_index": self._active_index,
                "waypoint_index": self._active_index,
                "source_waypoint_index": self._active_index,
                "target_advance_mode": "base_approach",
                "executor_target_world": tip_position.copy(),
                "executor_error_m": np.nan,
                "executor_feedforward_velocity_world": np.zeros(3, dtype=float),
                "achieved_waypoint_index": -1,
                "achieved_waypoint_error_m": np.nan,
                "waypoint_advanced": False,
                "waypoint_advance_reason": "",
                "tracking_complete": False,
                "tracking_approach": True,
                "base_target_position_m": self._base_target.position.copy(),
                "base_approach_tip_target_world": approach_tip_target.copy(),
                "base_approach_direction_world": approach_direction.copy(),
                "base_approach_standoff_m": self._base_approach_standoff_m,
                "base_position_error_m": approach.position_error_m,
                "base_orientation_error_rad": approach.orientation_error_rad,
                "tendon_reaction_isolated": True,
                "min_clearance_m": clearance.distance_m,
                **self._arm_clearance_metadata(state),
                "clearance_point": clearance.point.copy(),
                "clearance_normal": clearance.normal.copy(),
                "clearance_source_id": clearance.source_id,
                "clearance_violated": clearance.distance_m < self._min_clearance_m,
                "navigation_control_type": "staged_base_approach",
                "navigation_cbf_applied": False,
                "task_status_phase": self.phase,
                "task_status_active_index": self._active_index,
                "task_status_complete": False,
            },
        )

    def _tracking_command(
        self,
        state: RobotSystemState,
        clearance,
    ) -> RobotSystemCommand:
        if self._tracking is None:
            self._tracking = self._make_tracker(self._active_index, state)
        waypoint_index = self._active_index
        tracked = self._tracking.compute_command(state)
        metadata = self._global_tracking_metadata(
            tracked.metadata,
            waypoint_index,
        )
        if not self._tracking.done:
            metadata["waypoint_advanced"] = False
            metadata["waypoint_advance_reason"] = ""
            metadata["achieved_waypoint_index"] = -1
            metadata["achieved_waypoint_error_m"] = np.nan
            metadata["achieved_waypoint_orientation_error_rad"] = np.nan
        base_target = self._base_target
        if base_target is None:
            raise RuntimeError("Base target has not been initialized.")
        command_phase = self.phase
        if self._tracking.done:
            tracker_reason = self._tracking.terminal_reason
            tracker_advance_reason = str(
                metadata.get("waypoint_advance_reason", "")
            )
            tip_position = state.arms[self._executor_name].tip_pose_world.position
            waypoint_error = float(
                np.linalg.norm(self._waypoints[waypoint_index] - tip_position)
            )
            orientation_error = self._orientation_error(state, waypoint_index)
            waypoint_reached = (
                waypoint_error <= self._waypoint_tolerance_m
                and (
                    not np.isfinite(orientation_error)
                    or orientation_error <= self._orientation_tolerance_rad
                )
            )
            advance_reason = (
                "tolerance_reached"
                if waypoint_reached
                else (
                    "max_steps_reached"
                    if tracker_advance_reason == "max_steps_reached"
                    else ""
                )
            )
            if not advance_reason:
                self._tracking = self._make_tracker(waypoint_index, state)
                metadata["tracking_complete"] = False
                metadata["waypoint_advanced"] = False
                metadata["waypoint_advance_reason"] = ""
                metadata["achieved_waypoint_index"] = -1
                metadata["achieved_waypoint_error_m"] = np.nan
                metadata["achieved_waypoint_orientation_error_rad"] = np.nan
            elif waypoint_index >= self._waypoints.shape[0] - 1:
                self.phase = "complete"
                command_phase = self.phase
                self._terminal_reason = (
                    "max_steps_reached"
                    if advance_reason == "max_steps_reached"
                    else (tracker_reason or "completed")
                )
                metadata["tracking_complete"] = True
                metadata["waypoint_advanced"] = True
                metadata["waypoint_advance_reason"] = advance_reason
                metadata["achieved_waypoint_index"] = waypoint_index
                metadata["achieved_waypoint_error_m"] = waypoint_error
                metadata["achieved_waypoint_orientation_error_rad"] = (
                    orientation_error
                )
            else:
                self._active_index = waypoint_index + 1
                self.phase = "base_approach"
                self._tracking = None
                self._base_target = None
                self._base_target_index = -1
                metadata["tracking_complete"] = False
                metadata["waypoint_advanced"] = True
                metadata["waypoint_advance_reason"] = advance_reason
                metadata["achieved_waypoint_index"] = waypoint_index
                metadata["achieved_waypoint_error_m"] = waypoint_error
                metadata["achieved_waypoint_orientation_error_rad"] = (
                    orientation_error
                )
        return RobotSystemCommand(
            base_twist_world=np.zeros(6, dtype=float),
            arms=tracked.arms,
            metadata={
                **self._arm_clearance_metadata(state),
                **metadata,
                "staged_navigation_phase": command_phase,
                "staged_navigation_waypoint_index": waypoint_index,
                "base_target_position_m": base_target.position.copy(),
                "base_approach_tip_target_world": (
                    self._base_approach_tip_target(waypoint_index).copy()
                ),
                "base_approach_direction_world": (
                    self._base_approach_direction(waypoint_index).copy()
                ),
                "base_approach_standoff_m": self._base_approach_standoff_m,
                "base_position_error_m": float(
                    np.linalg.norm(
                        base_target.position - state.base.pose.position
                    )
                ),
                "base_orientation_error_rad": 0.0,
                "tendon_reaction_isolated": True,
                "task_status_phase": command_phase,
            },
        )

    def _make_tracker(
        self,
        waypoint_index: int,
        state: RobotSystemState,
    ) -> NavigationController:
        waypoints = self._tracking_waypoints(waypoint_index, state)
        orientation = self._target_orientation(waypoint_index)
        orientations = (
            None
            if orientation is None
            else np.repeat(orientation[None, :], waypoints.shape[0], axis=0)
        )
        return NavigationController(
            self._fixed_assembly,
            waypoints,
            waypoint_orientations_world_wxyz=orientations,
            **self._tracker_kwargs,
        )

    def _global_tracking_metadata(
        self,
        metadata: dict[str, object],
        waypoint_index: int,
    ) -> dict[str, object]:
        adjusted = dict(metadata)
        adjusted["staged_navigation_subtarget_index"] = int(
            adjusted.get("waypoint_index", 0)
        )
        adjusted["staged_navigation_subtarget_count"] = int(
            self._intermediate_waypoints_per_waypoint + 1
        )
        adjusted["staged_navigation_subtarget_advanced"] = bool(
            adjusted.get("waypoint_advanced", False)
        )
        adjusted["staged_navigation_subtarget_advance_reason"] = str(
            adjusted.get("waypoint_advance_reason", "")
        )
        adjusted["waypoint_index"] = waypoint_index
        adjusted["source_waypoint_index"] = waypoint_index
        achieved = int(adjusted.get("achieved_waypoint_index", -1))
        if achieved >= 0:
            adjusted["achieved_waypoint_index"] = waypoint_index
        adjusted["task_status_active_index"] = waypoint_index
        return adjusted

    def _tracking_waypoints(
        self,
        waypoint_index: int,
        state: RobotSystemState,
    ) -> np.ndarray:
        target = self._waypoints[waypoint_index].copy()
        count = self._intermediate_waypoints_per_waypoint
        if count <= 0:
            return target[None, :]
        start = state.arms[self._executor_name].tip_pose_world.position.copy()
        fractions = np.linspace(0.0, 1.0, count + 2, dtype=float)[1:]
        return start[None, :] + fractions[:, None] * (target - start)[None, :]

    def _zero_command(self, state: RobotSystemState, clearance) -> RobotSystemCommand:
        zero = RobotSystemCommand.zeros(self._tendon_counts)
        if self._base_target is None:
            raise RuntimeError("Base target has not been initialized.")
        return RobotSystemCommand(
            base_twist_world=np.zeros(6, dtype=float),
            arms=zero.arms,
            metadata={
                "task_type": "navigation",
                "staged_navigation_phase": self.phase,
                "staged_navigation_waypoint_index": self._active_index,
                "waypoint_index": self._active_index,
                "source_waypoint_index": self._active_index,
                "executor_target_world": self._waypoints[self._active_index].copy(),
                "executor_error_m": np.nan,
                "achieved_waypoint_index": -1,
                "achieved_waypoint_error_m": np.nan,
                "waypoint_advanced": False,
                "waypoint_advance_reason": "",
                "tracking_complete": True,
                "tracking_approach": False,
                "base_target_position_m": self._base_target.position.copy(),
                "base_position_error_m": np.nan,
                "base_orientation_error_rad": np.nan,
                "tendon_reaction_isolated": True,
                "min_clearance_m": clearance.distance_m,
                **self._arm_clearance_metadata(state),
                "clearance_point": clearance.point.copy(),
                "clearance_normal": clearance.normal.copy(),
                "clearance_source_id": clearance.source_id,
                "clearance_violated": True,
                "navigation_control_type": "staged_base_approach",
                "navigation_cbf_applied": False,
                "task_status_phase": self.phase,
                "task_status_active_index": self._active_index,
                "task_status_complete": True,
            },
        )

    def _arm_clearance_metadata(self, state: RobotSystemState) -> dict[str, object]:
        distances = {"executor": float("nan"), "observer": float("nan")}
        for arm_name, arm_state in state.arms.items():
            role = self.assembly.arms[arm_name].role
            if role not in distances:
                continue
            query = self.scene_query.nearest_centerline_clearance(
                arm_state.centerline_world
                if arm_state.centerline_world is not None
                else np.asarray([arm_state.tip_pose_world.position])
            )
            distances[role] = float(query.distance_m)
        return {
            "executor_clearance_m": distances["executor"],
            "observer_clearance_m": distances["observer"],
            "executor_scene_collision_active": False,
            "observer_scene_collision_active": False,
        }

    def _minimum_clearance(self, state: RobotSystemState):
        queries = [
            self.scene_query.nearest_centerline_clearance(
                arm.centerline_world
                if arm.centerline_world is not None
                else np.asarray([arm.tip_pose_world.position])
            )
            for arm in state.arms.values()
        ]
        return min(queries, key=lambda value: value.distance_m)

    def _target_orientation(self, waypoint_index: int) -> np.ndarray | None:
        if self._orientations.shape[0] == 0:
            return None
        return self._orientations[waypoint_index].copy()

    def _orientation_error(
        self,
        state: RobotSystemState,
        waypoint_index: int,
    ) -> float:
        target_orientation = self._target_orientation(waypoint_index)
        if target_orientation is None:
            return float("nan")
        current_orientation = state.arms[self._executor_name].tip_pose_world.quat
        return float(
            np.linalg.norm(
                quaternion_error_rotation_vector(
                    target_orientation,
                    current_orientation,
                )
            )
        )

    def _target_direction(self, waypoint_index: int) -> np.ndarray | None:
        if self._directions.shape[0] > 0:
            return self._directions[waypoint_index].copy()
        target_orientation = self._target_orientation(waypoint_index)
        if target_orientation is None:
            return None
        rotation = quaternion_wxyz_to_rotation_matrix(target_orientation)
        direction = np.asarray(rotation[:, 2], dtype=float)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm <= 1.0e-12:
            return None
        return direction / norm

    def _base_approach_direction(self, waypoint_index: int) -> np.ndarray:
        return self._base_stage.approach_direction(
            self._target_direction(waypoint_index)
        )

    def _base_approach_tip_target(self, waypoint_index: int) -> np.ndarray:
        return self._base_stage.staged_tip_target(
            self._waypoints[waypoint_index],
            self._target_direction(waypoint_index),
        )


def _single_role_name(assembly: RobotAssemblyConfig, role: str) -> str:
    matches = [arm.name for arm in assembly.enabled_arms if arm.role == role]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one enabled {role!r} arm, got {matches}."
        )
    return matches[0]


def _waypoint_quaternions(
    values: np.ndarray | None,
    size: int,
) -> np.ndarray:
    if values is None:
        return np.zeros((0, 4), dtype=float)
    result = np.asarray(values, dtype=float)
    if result.size == 0:
        return np.zeros((0, 4), dtype=float)
    if result.shape != (size, 4):
        raise ValueError(f"waypoint_orientations_world_wxyz must have shape ({size}, 4).")
    norms = np.linalg.norm(result, axis=1)
    if np.any((~np.isfinite(norms)) | (norms <= 1.0e-12)):
        raise ValueError("waypoint_orientations_world_wxyz rows must be finite nonzero quaternions.")
    return result / norms[:, None]


def _waypoint_directions(
    values: np.ndarray | None,
    size: int,
) -> np.ndarray:
    if values is None:
        return np.zeros((0, 3), dtype=float)
    result = np.asarray(values, dtype=float)
    if result.size == 0:
        return np.zeros((0, 3), dtype=float)
    if result.shape != (size, 3):
        raise ValueError(f"waypoint_directions_world must have shape ({size}, 3).")
    norms = np.linalg.norm(result, axis=1)
    if np.any((~np.isfinite(norms)) | (norms <= 1.0e-12)):
        raise ValueError(
            "waypoint_directions_world rows must be finite nonzero 3-vectors."
        )
    return result / norms[:, None]
