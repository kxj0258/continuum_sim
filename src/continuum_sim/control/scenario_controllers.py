"""Controllers used by scenario-driven baseline and engine workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from continuum_sim.control.cbf_qp_kinematics import cbf_lower_bound, solve_cbf_qp_velocity
from continuum_sim.control.waypoint_scheduler import WaypointScheduler
from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.control.task_intent import (
    CartesianTaskIntent,
    ContactTaskIntent,
    ObserverTaskIntent,
    SystemTaskIntent,
    TaskStatus,
    TaskStep,
)
from continuum_sim.control.unified_low_level import UnifiedLowLevelController
from continuum_sim.control.wiping_force_strategies import (
    WipingForceContext,
    WipingForceStrategy,
    default_wiping_force_strategy,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.kinematics.whole_body import (
    assemble_whole_body_jacobian,
    base_point_jacobian_world,
    centerline_point_bending_jacobian,
    rotate_position_jacobian_to_world,
)
from continuum_sim.model.base_pose import quaternion_error_rotation_vector
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import (
    RobotSystemCommand,
    RobotSystemState,
)


@dataclass(frozen=True)
class OnlineReachabilityConfig:
    """Online waypoint reachability scoring and optional auto-advance."""

    enabled: bool = True
    auto_advance_enabled: bool = True
    score_threshold: float = 0.3
    window_steps: int = 25
    min_steps_before_auto_advance: int = 50
    low_score_patience_steps: int = 25
    good_progress_mps: float = 0.001
    good_tendon_speed_ratio: float = 0.75
    good_alignment: float = 0.8
    bad_model_residual_mps: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("online_reachability.score_threshold must be in [0, 1].")
        if self.window_steps <= 1:
            raise ValueError("online_reachability.window_steps must be greater than 1.")
        if self.min_steps_before_auto_advance < 0:
            raise ValueError(
                "online_reachability.min_steps_before_auto_advance must be non-negative."
            )
        if self.low_score_patience_steps <= 0:
            raise ValueError(
                "online_reachability.low_score_patience_steps must be positive."
            )
        positive = {
            "good_progress_mps": self.good_progress_mps,
            "good_tendon_speed_ratio": self.good_tendon_speed_ratio,
            "good_alignment": self.good_alignment,
            "bad_model_residual_mps": self.bad_model_residual_mps,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"online_reachability.{name} must be positive and finite."
                )


@dataclass
class _ReachabilitySample:
    time_s: float
    error_m: float
    tip_position_world: np.ndarray


class _OnlineReachabilityEvaluator:
    """Score whether the active waypoint is currently making useful progress."""

    def __init__(self, config: OnlineReachabilityConfig) -> None:
        self.config = config
        self._active_index: int | None = None
        self._samples: list[_ReachabilitySample] = []
        self._steps_on_waypoint = 0
        self._low_score_steps = 0

    def update(
        self,
        *,
        active_index: int,
        state: RobotSystemState,
        executor_name: str,
        target_position_world: np.ndarray,
        error_m: float,
        tolerance_m: float,
        previous_tendon_rate_ref_mps: np.ndarray | None,
        previous_model_residual_mps: float,
    ) -> dict[str, object]:
        if self._active_index != active_index:
            self._active_index = active_index
            self._samples.clear()
            self._steps_on_waypoint = 0
            self._low_score_steps = 0

        self._steps_on_waypoint += 1
        executor = state.arms[executor_name]
        self._samples.append(
            _ReachabilitySample(
                time_s=float(state.time_s),
                error_m=float(error_m),
                tip_position_world=executor.tip_pose_world.position.copy(),
            )
        )
        if len(self._samples) > self.config.window_steps:
            del self._samples[: len(self._samples) - self.config.window_steps]

        metadata = self._empty_metadata()
        metadata["online_reachability_enabled"] = bool(self.config.enabled)
        metadata["online_reachability_steps_on_waypoint"] = int(
            self._steps_on_waypoint
        )
        metadata["online_reachability_window_steps"] = int(self.config.window_steps)
        metadata["online_reachability_score_threshold"] = float(
            self.config.score_threshold
        )
        metadata["online_reachability_min_steps_before_auto_advance"] = int(
            self.config.min_steps_before_auto_advance
        )
        metadata["online_reachability_low_score_patience_steps"] = int(
            self.config.low_score_patience_steps
        )
        metadata["online_reachability_auto_advance_enabled"] = bool(
            self.config.auto_advance_enabled
        )
        if not self.config.enabled:
            return metadata

        if len(self._samples) < 2:
            metadata["online_reachability_low_score_steps"] = int(
                self._low_score_steps
            )
            return metadata

        oldest = self._samples[0]
        newest = self._samples[-1]
        elapsed = max(newest.time_s - oldest.time_s, np.finfo(float).eps)
        progress_rate_mps = (oldest.error_m - newest.error_m) / elapsed
        progress_component = _unit_interval(
            progress_rate_mps / self.config.good_progress_mps
        )

        actual_tip_velocity_world = (
            newest.tip_position_world - oldest.tip_position_world
        ) / elapsed
        target_delta = (
            np.asarray(target_position_world, dtype=float) - newest.tip_position_world
        )
        alignment = _direction_alignment(actual_tip_velocity_world, target_delta)
        alignment_component = (
            1.0
            if not np.isfinite(alignment)
            else _unit_interval(
                (alignment + 0.2) / (self.config.good_alignment + 0.2)
            )
        )

        tendon_speed_ratio = _speed_ratio(
            executor.tendon_velocity_mps,
            previous_tendon_rate_ref_mps,
        )
        tendon_component = (
            1.0
            if not np.isfinite(tendon_speed_ratio)
            else _unit_interval(
                tendon_speed_ratio / self.config.good_tendon_speed_ratio
            )
        )

        model_residual_mps = float(previous_model_residual_mps)
        model_component = (
            1.0
            if not np.isfinite(model_residual_mps)
            else _unit_interval(
                1.0 - model_residual_mps / self.config.bad_model_residual_mps
            )
        )

        reachability_components = (
            progress_component,
            alignment_component,
            model_component,
        )
        reachability_score = float(
            np.prod(np.asarray(reachability_components, dtype=float))
        )
        execution_score = float(tendon_component)
        combined_score = float(reachability_score * execution_score)
        low_score = bool(
            self._steps_on_waypoint >= self.config.min_steps_before_auto_advance
            and error_m > tolerance_m
            and reachability_score < self.config.score_threshold
        )
        self._low_score_steps = self._low_score_steps + 1 if low_score else 0
        auto_advance_requested = bool(
            self.config.auto_advance_enabled
            and low_score
            and self._low_score_steps >= self.config.low_score_patience_steps
        )

        metadata.update(
            {
                "online_reachability_score": reachability_score,
                "online_reachability_execution_score": execution_score,
                "online_reachability_combined_score": combined_score,
                "online_reachability_progress_component": progress_component,
                "online_reachability_alignment_component": alignment_component,
                "online_reachability_tendon_component": tendon_component,
                "online_reachability_model_component": model_component,
                "online_reachability_progress_rate_mps": float(progress_rate_mps),
                "online_reachability_target_alignment": float(alignment),
                "online_reachability_tendon_speed_ratio": float(tendon_speed_ratio),
                "online_reachability_model_residual_mps": model_residual_mps,
                "online_reachability_low_score_steps": int(self._low_score_steps),
                "online_reachability_auto_advance_requested": auto_advance_requested,
            }
        )
        return metadata

    def _empty_metadata(self) -> dict[str, object]:
        return {
            "online_reachability_score": np.nan,
            "online_reachability_execution_score": np.nan,
            "online_reachability_combined_score": np.nan,
            "online_reachability_progress_component": np.nan,
            "online_reachability_alignment_component": np.nan,
            "online_reachability_tendon_component": np.nan,
            "online_reachability_model_component": np.nan,
            "online_reachability_progress_rate_mps": np.nan,
            "online_reachability_target_alignment": np.nan,
            "online_reachability_tendon_speed_ratio": np.nan,
            "online_reachability_model_residual_mps": np.nan,
            "online_reachability_low_score_steps": int(self._low_score_steps),
            "online_reachability_auto_advance_requested": False,
        }


class ZeroSystemController:
    """Hold the base and all direct tendon targets at their current values."""

    def __init__(self, assembly: RobotAssemblyConfig) -> None:
        self._tendon_counts = {
            arm.name: arm.spatial_arm.tendon_count
            for arm in assembly.enabled_arms
        }

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        del state
        return RobotSystemCommand.zeros(self._tendon_counts)


class WaypointTrackingController:
    """Advance executor waypoints while coordinating an optional observer."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        waypoints_world: np.ndarray,
        *,
        waypoint_tolerance_m: float = 1.0e-3,
        waypoint_orientations_world_wxyz: np.ndarray | None = None,
        orientation_tolerance_rad: float = 0.08,
        observer_roi_world: np.ndarray | None = None,
        observer_control_mode: str = "tracking",
        loop: bool = False,
        target_advance_mode: str = "tolerance",
        controller_dt_s: float = 0.02,
        advance_time_s: float | None = None,
        advance_steps: int | None = None,
        max_steps_per_waypoint: int | None = None,
        scene_query: EngineSceneQueryProtocol | None = None,
        approach_mask: np.ndarray | None = None,
        source_waypoint_index: np.ndarray | None = None,
        executor_position_gain: float = 4.0,
        observer_position_gain: float = 5.0,
        feedforward_speed_mps: float = 0.0,
        max_target_speed_mps: float | None = None,
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        enforce_backend_tendon_limits: bool = False,
        coordinated_config: CoordinatedTrackingConfig | None = None,
        observer_executor_offset_world: np.ndarray | None = None,
        observer_roi_blend: float = 0.25,
        advance_enabled: bool = True,
        online_reachability: OnlineReachabilityConfig | None = None,
    ) -> None:
        waypoints = np.asarray(waypoints_world, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
            raise ValueError("waypoints_world must have shape (N, 3) with N > 0.")
        if waypoint_tolerance_m < 0.0:
            raise ValueError("waypoint_tolerance_m must be non-negative.")
        self.assembly = assembly
        self.waypoints_world = waypoints.copy()
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.waypoint_orientations_world_wxyz = _waypoint_quaternions(
            waypoint_orientations_world_wxyz,
            waypoints.shape[0],
        )
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        if (
            not np.isfinite(self.orientation_tolerance_rad)
            or self.orientation_tolerance_rad < 0.0
        ):
            raise ValueError("orientation_tolerance_rad must be non-negative.")
        self.observer_roi_world = (
            None
            if observer_roi_world is None
            else np.asarray(observer_roi_world, dtype=float).copy()
        )
        if self.observer_roi_world is not None and self.observer_roi_world.shape != (3,):
            raise ValueError("observer_roi_world must have shape (3,).")
        self.observer_control_mode = str(observer_control_mode)
        self.observer_executor_offset_world = np.asarray(
            (
                [0.0, -0.04, 0.02]
                if observer_executor_offset_world is None
                else observer_executor_offset_world
            ),
            dtype=float,
        )
        if (
            self.observer_executor_offset_world.shape != (3,)
            or not np.all(np.isfinite(self.observer_executor_offset_world))
        ):
            raise ValueError("observer_executor_offset_world must be a finite 3-vector.")
        self.observer_roi_blend = float(observer_roi_blend)
        if not 0.0 <= self.observer_roi_blend <= 1.0:
            raise ValueError("observer_roi_blend must be in [0, 1].")
        self.loop = loop
        self.approach_mask = _waypoint_vector(
            approach_mask,
            waypoints.shape[0],
            default=False,
            dtype=bool,
            name="approach_mask",
        )
        self.source_waypoint_index = _waypoint_vector(
            source_waypoint_index,
            waypoints.shape[0],
            default=None,
            dtype=int,
            name="source_waypoint_index",
        )
        self.feedforward_speed_mps = float(feedforward_speed_mps)
        if not np.isfinite(self.feedforward_speed_mps) or self.feedforward_speed_mps < 0.0:
            raise ValueError("feedforward_speed_mps must be non-negative and finite.")
        self.scheduler = WaypointScheduler(
            waypoint_count=waypoints.shape[0],
            mode=target_advance_mode,
            tolerance_m=self.waypoint_tolerance_m,
            loop=loop,
            controller_dt_s=controller_dt_s,
            step_interval=advance_steps,
            time_interval_s=advance_time_s,
            max_steps_per_waypoint=max_steps_per_waypoint,
        )
        self._executor_name = _single_role_name(assembly, "executor")
        self.advance_enabled = bool(advance_enabled)
        self.executor_velocity_override_world: np.ndarray | None = None
        self.online_reachability = (
            OnlineReachabilityConfig()
            if online_reachability is None
            else online_reachability
        )
        self._reachability = _OnlineReachabilityEvaluator(self.online_reachability)
        self._previous_executor_tendon_rate_ref_mps: np.ndarray | None = None
        self._previous_model_residual_mps = float("nan")
        low_level_config = (
            CoordinatedTrackingConfig(
                kinematics_mode=solver_config.kinematics_mode,
                executor_position_gain=executor_position_gain,
                observer_position_gain=observer_position_gain,
                max_target_speed_mps=max_target_speed_mps,
                enforce_backend_tendon_limits=enforce_backend_tendon_limits,
            )
            if coordinated_config is None
            else coordinated_config
        )
        self._controller = UnifiedLowLevelController(
            assembly,
            coordinated_config=low_level_config,
            solver_config=solver_config,
            scene_query=scene_query,
        )

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._controller.last_diagnostics

    @property
    def active_index(self) -> int:
        return self.scheduler.active_index

    @property
    def done(self) -> bool:
        return self.scheduler.done

    @property
    def terminal_reason(self) -> str:
        return "completed" if self.done else ""

    def compute_command(
        self,
        state: RobotSystemState,
        *,
        advance: bool = True,
        contact: ContactTaskIntent | None = None,
    ) -> RobotSystemCommand:
        position = state.arms[self._executor_name].tip_pose_world.position
        achieved_index = self.active_index
        achieved_error = float(
            np.linalg.norm(self.waypoints_world[achieved_index] - position)
        )
        achieved_orientation_error = self._orientation_error(
            state,
            achieved_index,
        )
        pose_reached = (
            achieved_error <= self.waypoint_tolerance_m
            and (
                not np.isfinite(achieved_orientation_error)
                or achieved_orientation_error <= self.orientation_tolerance_rad
            )
        )
        scheduler_paused = not (advance and self.advance_enabled)
        reachability_metadata = self._reachability.update(
            active_index=achieved_index,
            state=state,
            executor_name=self._executor_name,
            target_position_world=self.waypoints_world[achieved_index],
            error_m=achieved_error,
            tolerance_m=self.waypoint_tolerance_m,
            previous_tendon_rate_ref_mps=self._previous_executor_tendon_rate_ref_mps,
            previous_model_residual_mps=self._previous_model_residual_mps,
        )
        auto_advance_requested = bool(
            reachability_metadata.get(
                "online_reachability_auto_advance_requested",
                False,
            )
        )
        if not scheduler_paused:
            if auto_advance_requested and not pose_reached:
                self.scheduler.advance(reason="online_reachability_low")
            else:
                self.scheduler.update(
                    error_norm_m=achieved_error if pose_reached else float("inf")
                )
        waypoint_advanced = self.done or self.active_index != achieved_index
        waypoint_advance_reason = (
            self.scheduler.last_advance_reason if waypoint_advanced else ""
        )
        step = self._task_step()
        if contact is not None:
            step = replace(
                step,
                intent=replace(step.intent, contact=contact),
            )
        command = self._controller.compute_command(state, step)
        controller_metadata = command.metadata
        self._previous_executor_tendon_rate_ref_mps = (
            command.arms[self._executor_name].tendon_rate_mps.copy()
            if self._executor_name in command.arms
            else None
        )
        self._previous_model_residual_mps = _model_residual_from_metadata(
            controller_metadata
        )
        if self.done:
            command = RobotSystemCommand.zeros(
                {
                    arm.name: arm.spatial_arm.tendon_count
                    for arm in self.assembly.enabled_arms
                }
            )
        command_error = float(
            np.linalg.norm(
                step.intent.executor.target_position_world - position
            )
        )
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **controller_metadata,
                **reachability_metadata,
                "task_type": "tracking",
                "waypoint_index": self.active_index,
                "source_waypoint_index": int(
                    self.source_waypoint_index[self.active_index]
                ),
                "target_advance_mode": self.scheduler.mode,
                "waypoint_scheduler_paused": scheduler_paused,
                "executor_target_world": (
                    step.intent.executor.target_position_world.copy()
                ),
                "executor_error_m": command_error,
                "executor_feedforward_velocity_world": (
                    step.intent.executor.feedforward_velocity_world.copy()
                ),
                "executor_target_orientation_world_wxyz": (
                    np.full(4, np.nan, dtype=float)
                    if step.intent.executor.target_orientation_world_wxyz is None
                    else step.intent.executor.target_orientation_world_wxyz.copy()
                ),
                "executor_orientation_error_rad": achieved_orientation_error,
                "achieved_waypoint_index": (
                    achieved_index if waypoint_advanced else -1
                ),
                "achieved_waypoint_error_m": (
                    achieved_error if waypoint_advanced else np.nan
                ),
                "achieved_waypoint_orientation_error_rad": (
                    achieved_orientation_error if waypoint_advanced else np.nan
                ),
                "waypoint_advanced": waypoint_advanced,
                "waypoint_advance_reason": waypoint_advance_reason,
                "tracking_complete": self.done,
                "tracking_approach": bool(
                    self.approach_mask[self.active_index]
                ),
            },
        )

    def _task_step(self) -> TaskStep:
        velocity_override = self.executor_velocity_override_world
        target_orientation = self._target_orientation(self.active_index)
        return TaskStep(
            intent=SystemTaskIntent(
                executor=CartesianTaskIntent(
                    target_position_world=self.waypoints_world[self.active_index],
                    feedforward_velocity_world=(
                        velocity_override.copy()
                        if velocity_override is not None
                        else self._feedforward_velocity()
                    ),
                    control_mode=(
                        "velocity" if velocity_override is not None else "position"
                    ),
                    target_orientation_world_wxyz=target_orientation,
                    orientation_control_mode=(
                        "quaternion"
                        if target_orientation is not None
                        else "disabled"
                    ),
                ),
                observer=ObserverTaskIntent(
                    control_mode=self.observer_control_mode,
                    roi_position_world=self.observer_roi_world,
                    executor_offset_world=self.observer_executor_offset_world,
                    roi_blend=self.observer_roi_blend,
                ),
            ),
            status=TaskStatus(
                task_type="tracking",
                phase=("complete" if self.done else "tracking"),
                active_index=self.active_index,
                complete=self.done,
                stop_reason=("completed" if self.done else ""),
            ),
        )

    def _target_orientation(self, index: int) -> np.ndarray | None:
        if self.waypoint_orientations_world_wxyz.shape[0] == 0:
            return None
        return self.waypoint_orientations_world_wxyz[index].copy()

    def _orientation_error(
        self,
        state: RobotSystemState,
        index: int,
    ) -> float:
        target_orientation = self._target_orientation(index)
        if target_orientation is None:
            return float("nan")
        current_orientation = state.arms[self._executor_name].tip_pose_world.quat
        error = quaternion_error_rotation_vector(
            target_orientation,
            current_orientation,
        )
        return float(np.linalg.norm(error))

    def _feedforward_velocity(self) -> np.ndarray:
        return self._path_feedforward_velocity()

    def _path_feedforward_velocity(self) -> np.ndarray:
        if (
            self.done
            or self.feedforward_speed_mps <= 0.0
            or self.active_index >= self.waypoints_world.shape[0] - 1
        ):
            return np.zeros(3, dtype=float)
        delta = (
            self.waypoints_world[self.active_index + 1]
            - self.waypoints_world[self.active_index]
        )
        distance = float(np.linalg.norm(delta))
        if distance <= np.finfo(float).eps:
            return np.zeros(3, dtype=float)
        return self.feedforward_speed_mps * delta / distance


class TimedTrajectoryTrackingController:
    """Track a time-parameterized Cartesian trajectory."""

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
        observer_executor_offset_world: np.ndarray | None = None,
        observer_roi_blend: float = 0.25,
    ) -> None:
        waypoints = np.asarray(waypoints_world, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
            raise ValueError("waypoints_world must have shape (N, 3) with N > 0.")
        if trajectory_duration_s <= 0.0 or not np.isfinite(trajectory_duration_s):
            raise ValueError("trajectory_duration_s must be positive and finite.")
        self.assembly = assembly
        self.waypoints_world = waypoints.copy()
        self.trajectory_duration_s = float(trajectory_duration_s)
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.loop = loop
        self.approach_mask = _waypoint_vector(
            approach_mask,
            waypoints.shape[0],
            default=False,
            dtype=bool,
            name="approach_mask",
        )
        self.source_waypoint_index = _waypoint_vector(
            source_waypoint_index,
            waypoints.shape[0],
            default=None,
            dtype=int,
            name="source_waypoint_index",
        )
        self.observer_roi_world = (
            None
            if observer_roi_world is None
            else np.asarray(observer_roi_world, dtype=float).copy()
        )
        self.observer_control_mode = str(observer_control_mode)
        self.observer_executor_offset_world = np.asarray(
            (
                [0.0, -0.04, 0.02]
                if observer_executor_offset_world is None
                else observer_executor_offset_world
            ),
            dtype=float,
        )
        self.observer_roi_blend = float(observer_roi_blend)
        self._executor_name = _single_role_name(assembly, "executor")
        self._start_time_s: float | None = None
        self._elapsed_s = 0.0
        self._active_index = 0
        self._done = False
        low_level_config = (
            CoordinatedTrackingConfig(
                kinematics_mode=solver_config.kinematics_mode,
                executor_position_gain=executor_position_gain,
                observer_position_gain=observer_position_gain,
                max_target_speed_mps=max_target_speed_mps,
                enforce_backend_tendon_limits=enforce_backend_tendon_limits,
            )
            if coordinated_config is None
            else coordinated_config
        )
        self._controller = UnifiedLowLevelController(
            assembly,
            coordinated_config=low_level_config,
            solver_config=solver_config,
            scene_query=scene_query,
        )

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._controller.last_diagnostics

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def done(self) -> bool:
        return self._done

    @property
    def terminal_reason(self) -> str:
        return "duration_elapsed" if self.done else ""

    def compute_command(
        self,
        state: RobotSystemState,
        *,
        contact: ContactTaskIntent | None = None,
    ) -> RobotSystemCommand:
        if self._start_time_s is None:
            self._start_time_s = float(state.time_s)
        elapsed = max(0.0, float(state.time_s) - self._start_time_s)
        complete = (not self.loop) and elapsed >= self.trajectory_duration_s
        step, source_index, local_fraction = self._target_at(
            elapsed,
            complete=complete,
            contact=contact,
        )
        self._elapsed_s = elapsed
        self._active_index = int(source_index)
        self._done = complete
        command = self._controller.compute_command(state, step)
        controller_metadata = command.metadata
        position = state.arms[self._executor_name].tip_pose_world.position
        error = float(
            np.linalg.norm(
                step.intent.executor.target_position_world - position
            )
        )
        if complete:
            command = RobotSystemCommand.zeros(
                {
                    arm.name: arm.spatial_arm.tendon_count
                    for arm in self.assembly.enabled_arms
                }
            )
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **controller_metadata,
                "task_type": "tracking",
                "tracking_mode": "time",
                "trajectory_time_s": min(elapsed, self.trajectory_duration_s),
                "trajectory_duration_s": self.trajectory_duration_s,
                "waypoint_index": int(source_index),
                "source_waypoint_index": int(self.source_waypoint_index[source_index]),
                "trajectory_local_fraction": float(local_fraction),
                "executor_target_world": (
                    step.intent.executor.target_position_world.copy()
                ),
                "executor_error_m": error,
                "executor_feedforward_velocity_world": (
                    step.intent.executor.feedforward_velocity_world.copy()
                ),
                "achieved_waypoint_index": -1,
                "achieved_waypoint_error_m": np.nan,
                "waypoint_advanced": False,
                "tracking_complete": self._done,
                "tracking_approach": bool(self.approach_mask[source_index]),
            },
        )

    def _target_at(
        self,
        elapsed_s: float,
        *,
        complete: bool = False,
        contact: ContactTaskIntent | None = None,
    ) -> tuple[TaskStep, int, float]:
        position, velocity, source_index, fraction = self._sample_trajectory(elapsed_s)
        return (
            TaskStep(
                intent=SystemTaskIntent(
                    executor=CartesianTaskIntent(
                        target_position_world=position,
                        feedforward_velocity_world=velocity,
                    ),
                    observer=ObserverTaskIntent(
                        control_mode=self.observer_control_mode,
                        roi_position_world=self.observer_roi_world,
                        executor_offset_world=self.observer_executor_offset_world,
                        roi_blend=self.observer_roi_blend,
                    ),
                    contact=contact,
                ),
                status=TaskStatus(
                    task_type="tracking",
                    phase=("complete" if complete else "tracking"),
                    active_index=int(source_index),
                    complete=complete,
                    stop_reason=("duration_elapsed" if complete else ""),
                ),
            ),
            source_index,
            fraction,
        )

    def _sample_trajectory(
        self,
        elapsed_s: float,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        count = self.waypoints_world.shape[0]
        if count == 1:
            return self.waypoints_world[0].copy(), np.zeros(3, dtype=float), 0, 0.0
        if self.loop:
            phase = (elapsed_s % self.trajectory_duration_s) / self.trajectory_duration_s
            scaled = phase * count
            index = int(np.floor(scaled)) % count
            next_index = (index + 1) % count
            fraction = float(scaled - np.floor(scaled))
            segment_dt = self.trajectory_duration_s / count
        else:
            clamped = min(max(elapsed_s, 0.0), self.trajectory_duration_s)
            scaled = (clamped / self.trajectory_duration_s) * (count - 1)
            index = min(int(np.floor(scaled)), count - 2)
            next_index = index + 1
            fraction = float(scaled - index)
            segment_dt = self.trajectory_duration_s / (count - 1)
        current = self.waypoints_world[index]
        following = self.waypoints_world[next_index]
        position = current + fraction * (following - current)
        velocity = (following - current) / segment_dt
        active_index = next_index if fraction > np.finfo(float).eps else index
        if (not self.loop) and elapsed_s >= self.trajectory_duration_s:
            velocity = np.zeros(3, dtype=float)
            active_index = count - 1
        return position, velocity, active_index, fraction

class NavigationController:
    """Waypoint tracking with explicit clearance reporting and termination."""

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
    ) -> None:
        if scene_query is None:
            raise ValueError("NavigationController requires a scene query.")
        self._tracking = WaypointTrackingController(
            assembly,
            waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            waypoint_orientations_world_wxyz=waypoint_orientations_world_wxyz,
            orientation_tolerance_rad=orientation_tolerance_rad,
            observer_roi_world=observer_roi_world,
            observer_control_mode=observer_control_mode,
            target_advance_mode=target_advance_mode,
            controller_dt_s=controller_dt_s,
            advance_time_s=advance_time_s,
            advance_steps=advance_steps,
            max_steps_per_waypoint=max_steps_per_waypoint,
            scene_query=scene_query,
            executor_position_gain=executor_position_gain,
            observer_position_gain=observer_position_gain,
            feedforward_speed_mps=feedforward_speed_mps,
            max_target_speed_mps=max_target_speed_mps,
            solver_config=solver_config,
            enforce_backend_tendon_limits=enforce_backend_tendon_limits,
            coordinated_config=coordinated_config,
            online_reachability=online_reachability,
        )
        self.scene_query = scene_query
        self.min_clearance_m = min_clearance_m
        self.terminate_on_clearance_violation = terminate_on_clearance_violation
        self.clearance_violated = False
        self._last_clearance_query = None
        self.control_type = control_type
        self.cbf_gain = float(cbf_gain)
        self.cbf_influence_distance_m = (
            None if cbf_influence_distance_m is None else float(cbf_influence_distance_m)
        )
        self._layout = ControlLayout.from_assembly(assembly)
        self._assembly = assembly
        self._kinematics_mode = solver_config.kinematics_mode

    @property
    def done(self) -> bool:
        return self._tracking.done or (
            self.terminate_on_clearance_violation and self.clearance_violated
        )

    @property
    def terminal_reason(self) -> str:
        if self.terminate_on_clearance_violation and self.clearance_violated:
            return "clearance_violation"
        return self._tracking.terminal_reason

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        queries = [
            self.scene_query.nearest_centerline_clearance(
                arm.centerline_world
                if arm.centerline_world is not None
                else np.asarray([arm.tip_pose_world.position])
            )
            for arm in state.arms.values()
        ]
        query = min(queries, key=lambda value: value.distance_m)
        self._last_clearance_query = query
        minimum = float(query.distance_m)
        self.clearance_violated = minimum < self.min_clearance_m
        command = self._tracking.compute_command(state)
        cbf_applied = False
        if self.control_type == "navigation_cbf_qp" and not self._tracking.done:
            command, cbf_applied = self._project_command_with_cbf(
                state,
                command,
                minimum,
            )
        if self.terminate_on_clearance_violation and self.clearance_violated:
            zero = RobotSystemCommand.zeros(
                {
                    name: values.tendon_rate_mps.size
                    for name, values in command.arms.items()
                }
            )
            command = RobotSystemCommand(
                base_twist_world=zero.base_twist_world,
                arms=zero.arms,
                metadata=command.metadata,
            )
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                "task_type": "navigation",
                "min_clearance_m": minimum,
                "clearance_point": query.point.copy(),
                "clearance_normal": query.normal.copy(),
                "clearance_source_id": query.source_id,
                "clearance_violated": self.clearance_violated,
                "navigation_control_type": self.control_type,
                "navigation_cbf_applied": cbf_applied,
            },
        )

    def _project_command_with_cbf(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        minimum_clearance_m: float,
    ) -> tuple[RobotSystemCommand, bool]:
        influence = (
            max(2.5 * self.min_clearance_m, self.min_clearance_m + 0.005)
            if self.cbf_influence_distance_m is None
            else self.cbf_influence_distance_m
        )
        if not np.isfinite(minimum_clearance_m) or minimum_clearance_m >= influence:
            return command, False
        velocity = self._layout.flatten(command)
        row = self._nearest_clearance_jacobian(state)
        if row is None:
            return command, False
        projected = solve_cbf_qp_velocity(
            velocity,
            barrier_jacobian=row[None, :],
            barrier_lower_bound=np.array(
                [cbf_lower_bound(minimum_clearance_m, self.min_clearance_m, self.cbf_gain)],
                dtype=float,
            ),
        )
        updated = self._layout.unflatten(projected)
        return (
            RobotSystemCommand(
                base_twist_world=updated.base_twist_world,
                arms=updated.arms,
                metadata=command.metadata,
            ),
            True,
        )

    def _nearest_clearance_jacobian(self, state: RobotSystemState) -> np.ndarray | None:
        best: tuple[float, np.ndarray] | None = None
        for arm in self._assembly.enabled_arms:
            arm_state = state.arms[arm.name]
            model = self._layout.bending_models[arm.name]
            q = model.to_q(model.estimate(arm_state.tendon_displacement_m))
            centerline = forward_kinematics(
                q,
                arm.spatial_arm.params,
                samples_per_segment=6,
                kinematics_mode=self._kinematics_mode,
            ).centerline
            mount = state.base.pose.compose(arm.mount_pose)
            centerline_world = mount.transform_points(centerline)
            queries = [
                self.scene_query.nearest_distance(point)
                for point in centerline_world
            ]
            if not queries:
                continue
            index = int(np.argmin([query.distance_m for query in queries]))
            query = queries[index]
            local_jacobian = centerline_point_bending_jacobian(
                q,
                index,
                arm.spatial_arm.params,
                arm.spatial_arm.tendons,
                samples_per_segment=6,
                kinematics_mode=self._kinematics_mode,
            )
            world_jacobian = rotate_position_jacobian_to_world(
                local_jacobian,
                mount.as_matrix()[:3, :3],
            )
            system_jacobian = assemble_whole_body_jacobian(
                self._layout,
                arm.name,
                base_point_jacobian_world(
                    centerline_world[index],
                    state.base.pose.position,
                ),
                world_jacobian,
            )
            row = query.normal @ system_jacobian
            if best is None or query.distance_m < best[0]:
                best = (float(query.distance_m), row)
        return None if best is None else best[1]


class WipingController:
    """Direct-tendon wiping path with normal-distance contact regulation."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        waypoints_world: np.ndarray,
        *,
        waypoint_tolerance_m: float,
        scene_query: EngineSceneQueryProtocol | None,
        surface_normal_world: np.ndarray,
        target_contact_distance_m: float,
        contact_tolerance_m: float,
        surface_point_world: np.ndarray | None = None,
        target_advance_mode: str = "tolerance",
        controller_dt_s: float = 0.02,
        advance_time_s: float | None = None,
        advance_steps: int | None = None,
        phases: tuple[str, ...] = (),
        target_force_n: np.ndarray | None = None,
        control_type: str = "contact_distance",
        normal_force_gain: float = 0.0,
        force_proxy_stiffness_n_m: float = 600.0,
        force_feedback_mode: str = "proxy_distance",
        max_normal_velocity_m_s: float = 0.03,
        force_control_weight: float = 20.0,
        max_contact_force_n: float | None = None,
        force_strategy: WipingForceStrategy | None = None,
        tracking_mode: str = "waypoint",
        trajectory_duration_s: float | None = None,
        approach_samples: int = 0,
        observer_control_mode: str = "tracking",
        executor_position_gain: float = 4.0,
        observer_position_gain: float = 5.0,
        feedforward_speed_mps: float = 0.0,
        max_target_speed_mps: float | None = None,
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        enforce_backend_tendon_limits: bool = False,
        coordinated_config: CoordinatedTrackingConfig | None = None,
        online_reachability: OnlineReachabilityConfig | None = None,
    ) -> None:
        self.control_type = control_type
        self._tracking_mode = str(tracking_mode)
        if self._tracking_mode not in ("waypoint", "time"):
            raise ValueError("Wiping tracking_mode must be 'waypoint' or 'time'.")
        if self._tracking_mode == "time" and (
            trajectory_duration_s is None
            or not np.isfinite(trajectory_duration_s)
            or trajectory_duration_s <= 0.0
        ):
            raise ValueError(
                "Wiping time tracking requires a positive trajectory_duration_s."
            )
        self.assembly = assembly
        self.scene_query = scene_query
        normal = np.asarray(surface_normal_world, dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal.shape != (3,) or normal_norm <= 1.0e-12:
            raise ValueError("surface_normal_world must be a nonzero 3-vector.")
        self.surface_normal_world = normal / normal_norm
        self.surface_point_world = (
            None
            if surface_point_world is None
            else np.asarray(surface_point_world, dtype=float)
        )
        self.target_contact_distance_m = target_contact_distance_m
        self.contact_tolerance_m = contact_tolerance_m
        self.phases = tuple(phases)
        self._original_waypoints_world = np.asarray(waypoints_world, dtype=float).copy()
        self._approach_samples = int(approach_samples)
        self._runtime_approach_initialized = False
        self._tracking_kwargs = {
            "waypoint_tolerance_m": waypoint_tolerance_m,
            "target_advance_mode": target_advance_mode,
            "controller_dt_s": controller_dt_s,
            "advance_time_s": advance_time_s,
            "advance_steps": advance_steps,
            "advance_enabled": True,
            "observer_control_mode": observer_control_mode,
            "executor_position_gain": executor_position_gain,
            "observer_position_gain": observer_position_gain,
            "feedforward_speed_mps": feedforward_speed_mps,
            "max_target_speed_mps": max_target_speed_mps,
            "solver_config": solver_config,
            "enforce_backend_tendon_limits": enforce_backend_tendon_limits,
            "coordinated_config": coordinated_config,
            "trajectory_duration_s": trajectory_duration_s,
            "online_reachability": online_reachability,
        }
        self._tracking = self._make_tracking_controller(self._original_waypoints_world)
        force = (
            np.zeros(self._tracking.waypoints_world.shape[0], dtype=float)
            if target_force_n is None
            else np.asarray(target_force_n, dtype=float)
        )
        if force.shape != (self._tracking.waypoints_world.shape[0],):
            raise ValueError("target_force_n must match waypoint count.")
        self.target_force_n = force
        self._original_target_force_n = force.copy()
        self._original_phases = self.phases
        self.normal_force_gain = float(normal_force_gain)
        self.force_proxy_stiffness_n_m = float(force_proxy_stiffness_n_m)
        self.force_feedback_mode = str(force_feedback_mode)
        self.max_normal_velocity_m_s = float(max_normal_velocity_m_s)
        if (
            not np.isfinite(self.max_normal_velocity_m_s)
            or self.max_normal_velocity_m_s <= 0.0
        ):
            raise ValueError("max_normal_velocity_m_s must be positive and finite.")
        self.force_control_weight = float(force_control_weight)
        if not np.isfinite(self.force_control_weight) or self.force_control_weight <= 0.0:
            raise ValueError("force_control_weight must be positive and finite.")
        self.max_contact_force_n = max_contact_force_n
        self.force_strategy = (
            default_wiping_force_strategy(control_type)
            if force_strategy is None
            else force_strategy
        )
        self.force_limit_exceeded = False
        self.phase = "approach"
        self.controller_dt_s = float(controller_dt_s)
        self._executor_name = _single_role_name(assembly, "executor")
        self._kinematics_mode = solver_config.kinematics_mode

    @property
    def done(self) -> bool:
        return self._tracking.done

    @property
    def terminal_reason(self) -> str:
        return self._tracking.terminal_reason

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        self._ensure_runtime_approach(state)
        executor = next(arm for arm in state.arms.values() if arm.role == "executor")
        distance = float("nan")
        query = None
        if self.surface_point_world is not None:
            distance = float(
                np.dot(
                    executor.tip_pose_world.position - self.surface_point_world,
                    self.surface_normal_world,
                )
            )
        elif self.scene_query is not None:
            query = self.scene_query.nearest_distance(executor.tip_pose_world.position)
            distance = float(query.distance_m)
        if self.phases:
            self.phase = self.phases[self._tracking.active_index]
        elif self._tracking.active_index == 0:
            self.phase = "approach"
        elif self._tracking.active_index == self._tracking.waypoints_world.shape[0] - 1:
            self.phase = "retreat"
        else:
            self.phase = "contact"
        contact_error = (
            float("nan")
            if not np.isfinite(distance)
            else self.target_contact_distance_m - distance
        )
        estimated_force = (
            float("nan")
            if not np.isfinite(distance)
            else max(0.0, -distance * self.force_proxy_stiffness_n_m)
        )
        self.force_limit_exceeded = bool(
            self.max_contact_force_n is not None
            and np.isfinite(estimated_force)
            and estimated_force > self.max_contact_force_n
        )
        waypoint_index = self._tracking.active_index
        target_force = float(self.target_force_n[waypoint_index])
        force_error = (
            float("nan")
            if not np.isfinite(estimated_force)
            else target_force - estimated_force
        )
        strategy_result = self.force_strategy.compute(
            WipingForceContext(
                executor=executor,
                waypoints_world=self._tracking.waypoints_world,
                waypoint_index=waypoint_index,
                phase=self.phase,
                surface_normal_world=self.surface_normal_world,
                query_normal_world=None if query is None else query.normal,
                contact_error_m=contact_error,
                estimated_force_n=estimated_force,
                target_force_n=target_force,
                normal_force_gain=self.normal_force_gain,
                force_proxy_stiffness_n_m=self.force_proxy_stiffness_n_m,
                contact_tolerance_m=self.contact_tolerance_m,
                controller_dt_s=self.controller_dt_s,
            )
        )
        force_control_velocity = self._force_control_velocity_mps(
            contact_error,
            force_error,
        )
        force_control_enabled = bool(
            self.control_type == "hybrid_force_position"
            and self.phase == "contact"
            and np.isfinite(force_control_velocity)
            and abs(force_control_velocity) > 1.0e-12
        )
        contact_intent = ContactTaskIntent(
            surface_normal_world=(
                self.surface_normal_world if query is None else query.normal
            ),
            target_normal_force_n=target_force,
            target_contact_distance_m=self.target_contact_distance_m,
            contact_distance_m=(distance if np.isfinite(distance) else None),
            measured_normal_force_n=(
                estimated_force if np.isfinite(estimated_force) else None
            ),
            force_feedback_mode=self.force_feedback_mode,
            force_proxy_stiffness_n_m=self.force_proxy_stiffness_n_m,
            max_normal_velocity_m_s=self.max_normal_velocity_m_s,
            force_control_enabled=force_control_enabled,
            force_control_velocity_mps=(
                force_control_velocity if force_control_enabled else 0.0
            ),
            force_control_weight=self.force_control_weight,
        )
        original_waypoint = self._tracking.waypoints_world[waypoint_index].copy()
        waypoint_correction_applied = self.phase == "contact" and not force_control_enabled
        if waypoint_correction_applied:
            self._tracking.waypoints_world[waypoint_index] = (
                strategy_result.corrected_waypoint
            )
        try:
            if self._tracking_mode == "time":
                command = self._tracking.compute_command(
                    state,
                    contact=contact_intent,
                )
            else:
                command = self._tracking.compute_command(
                    state,
                    advance=not strategy_result.controls_waypoint_advance,
                    contact=contact_intent,
                )
        finally:
            self._tracking.waypoints_world[waypoint_index] = original_waypoint
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                **strategy_result.metadata,
                "task_type": "wiping",
                "wiping_phase": self.phase,
                "wiping_control_type": self.control_type,
                "target_normal_force_n": float(self.target_force_n[waypoint_index]),
                "estimated_normal_force_n": estimated_force,
                "force_error_n": force_error,
                "normal_force_gain": self.normal_force_gain,
                "force_proxy_stiffness_n_m": self.force_proxy_stiffness_n_m,
                "force_feedback_mode": self.force_feedback_mode,
                "max_normal_velocity_m_s": self.max_normal_velocity_m_s,
                "force_control_velocity_mps": force_control_velocity,
                "force_control_enabled": force_control_enabled,
                "wiping_waypoint_correction_applied": waypoint_correction_applied,
                "max_contact_force_n": self.max_contact_force_n,
                "force_limit_exceeded": self.force_limit_exceeded,
                "contact_distance_m": distance,
                "contact_error_m": contact_error,
                "contact_established": bool(
                    np.isfinite(contact_error)
                    and abs(contact_error) <= self.contact_tolerance_m
                ),
                "waypoint_advanced": bool(
                    command.metadata.get("waypoint_advanced", False)
                    or strategy_result.waypoint_advanced
                ),
            },
        )

    def _force_control_velocity_mps(
        self,
        contact_error_m: float,
        force_error_n: float,
    ) -> float:
        if not np.isfinite(contact_error_m):
            return 0.0
        distance_term = contact_error_m / max(self.controller_dt_s, 1.0e-12)
        force_term = 0.0
        if np.isfinite(force_error_n) and self.force_proxy_stiffness_n_m > 0.0:
            force_term = -(
                self.normal_force_gain
                * force_error_n
                / self.force_proxy_stiffness_n_m
                / max(self.controller_dt_s, 1.0e-12)
            )
        return float(
            np.clip(
                distance_term + force_term,
                -self.max_normal_velocity_m_s,
                self.max_normal_velocity_m_s,
            )
        )

    def _make_tracking_controller(
        self,
        waypoints_world: np.ndarray,
    ) -> WaypointTrackingController | TimedTrajectoryTrackingController:
        kwargs = self._tracking_kwargs
        if self._tracking_mode == "time":
            return TimedTrajectoryTrackingController(
                self.assembly,
                waypoints_world,
                trajectory_duration_s=float(kwargs["trajectory_duration_s"]),
                waypoint_tolerance_m=float(kwargs["waypoint_tolerance_m"]),
                scene_query=None,
                observer_control_mode=str(kwargs["observer_control_mode"]),
                executor_position_gain=float(kwargs["executor_position_gain"]),
                observer_position_gain=float(kwargs["observer_position_gain"]),
                max_target_speed_mps=kwargs["max_target_speed_mps"],
                solver_config=kwargs["solver_config"],
                enforce_backend_tendon_limits=bool(
                    kwargs["enforce_backend_tendon_limits"]
                ),
                coordinated_config=kwargs["coordinated_config"],
            )
        return WaypointTrackingController(
            self.assembly,
            waypoints_world,
            waypoint_tolerance_m=float(kwargs["waypoint_tolerance_m"]),
            target_advance_mode=str(kwargs["target_advance_mode"]),
            controller_dt_s=float(kwargs["controller_dt_s"]),
            advance_time_s=kwargs["advance_time_s"],
            advance_steps=kwargs["advance_steps"],
            advance_enabled=bool(kwargs["advance_enabled"]),
            # Contact regulation intentionally replaces generic clearance
            # avoidance for the executor at the wiping surface.
            scene_query=None,
            observer_control_mode=str(kwargs["observer_control_mode"]),
            executor_position_gain=float(kwargs["executor_position_gain"]),
            observer_position_gain=float(kwargs["observer_position_gain"]),
            feedforward_speed_mps=float(kwargs["feedforward_speed_mps"]),
            max_target_speed_mps=kwargs["max_target_speed_mps"],
            solver_config=kwargs["solver_config"],
            enforce_backend_tendon_limits=bool(
                kwargs["enforce_backend_tendon_limits"]
            ),
            coordinated_config=kwargs["coordinated_config"],
            online_reachability=kwargs["online_reachability"],
        )

    def _ensure_runtime_approach(self, state: RobotSystemState) -> None:
        if self._runtime_approach_initialized:
            return
        self._runtime_approach_initialized = True
        if self._approach_samples < 2:
            return
        executor = next(arm for arm in state.arms.values() if arm.role == "executor")
        start = executor.tip_pose_world.position.copy()
        end = self._original_waypoints_world[0].copy()
        s = np.linspace(0.0, 1.0, self._approach_samples)
        alpha = 3.0 * s**2 - 2.0 * s**3
        approach = start[None, :] + alpha[:, None] * (end - start)[None, :]
        waypoints = np.vstack((approach, self._original_waypoints_world[1:]))
        self._tracking = self._make_tracking_controller(waypoints)
        approach_force = np.zeros(self._approach_samples, dtype=float)
        self.target_force_n = np.concatenate(
            (approach_force, self._original_target_force_n[1:])
        )
        if self._original_phases:
            self.phases = (
                *("approach" for _ in range(self._approach_samples)),
                *self._original_phases[1:],
            )
        else:
            trailing = waypoints.shape[0] - self._approach_samples
            if trailing <= 0:
                self.phases = tuple("approach" for _ in range(waypoints.shape[0]))
            else:
                self.phases = (
                    *("approach" for _ in range(self._approach_samples)),
                    *(
                        "contact"
                        for _ in range(max(0, trailing - 1))
                    ),
                    "retreat",
                )
def _unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(value, 0.0, 1.0))


def _direction_alignment(velocity: np.ndarray, target_delta: np.ndarray) -> float:
    velocity_norm = float(np.linalg.norm(velocity))
    target_norm = float(np.linalg.norm(target_delta))
    if velocity_norm <= 1.0e-12 or target_norm <= 1.0e-12:
        return float("nan")
    return float(
        np.clip(
            np.dot(velocity, target_delta) / (velocity_norm * target_norm),
            -1.0,
            1.0,
        )
    )


def _speed_ratio(actual: np.ndarray, reference: np.ndarray | None) -> float:
    if reference is None:
        return float("nan")
    actual_norm = float(np.linalg.norm(actual))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 1.0e-12:
        return float("nan")
    return actual_norm / reference_norm


def _model_residual_from_metadata(metadata: dict[str, object]) -> float:
    for key in (
        "pcc_command_mujoco_velocity_residual_norm_mps",
        "pcc_mujoco_model_velocity_residual_norm_mps",
        "residual_norm",
    ):
        value = metadata.get(key)
        if value is not None:
            scalar = _finite_scalar_norm(value)
            if np.isfinite(scalar):
                return scalar
    solver = metadata.get("whole_body_solver")
    if isinstance(solver, dict):
        for key in ("residual_norm", "target_projection_residual_norm"):
            value = solver.get(key)
            if value is not None:
                scalar = _finite_scalar_norm(value)
                if np.isfinite(scalar):
                    return scalar
    return float("nan")


def _finite_scalar_norm(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan")
    return float(np.linalg.norm(array))


def _single_role_name(assembly: RobotAssemblyConfig, role: str) -> str:
    names = [arm.name for arm in assembly.enabled_arms if arm.role == role]
    if len(names) != 1:
        raise ValueError(f"Assembly must contain exactly one enabled {role!r} arm.")
    return names[0]


def _waypoint_vector(
    values: np.ndarray | None,
    size: int,
    *,
    default: bool | None,
    dtype,
    name: str,
) -> np.ndarray:
    if values is None:
        if default is None:
            return np.arange(size, dtype=dtype)
        return np.full(size, default, dtype=dtype)
    result = np.asarray(values, dtype=dtype)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    return result.copy()


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


def _default_tangent(normal: np.ndarray) -> np.ndarray:
    unit = np.asarray(normal, dtype=float)
    unit = unit / np.linalg.norm(unit)
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, unit))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = candidate - np.dot(candidate, unit) * unit
    return tangent / np.linalg.norm(tangent)
