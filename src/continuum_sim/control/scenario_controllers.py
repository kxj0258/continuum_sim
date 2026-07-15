"""Controllers used by scenario-driven baseline and engine workflows."""

from __future__ import annotations

import numpy as np

from continuum_sim.control.cbf_qp_kinematics import cbf_lower_bound, solve_cbf_qp_velocity
from continuum_sim.control.contact_triggered_admittance import (
    ContactTriggeredAdmittanceConfig,
    ContactTriggeredAdmittanceTracker,
)
from continuum_sim.control.waypoint_scheduler import WaypointScheduler
from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
)
from continuum_sim.control.engine_cleaning_controller import EngineCleaningController
from continuum_sim.control.engine_cleaning_types import (
    EngineCleaningControllerGains,
    EngineCleaningFeedback,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.control.task_intent import (
    CartesianTaskIntent,
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
from continuum_sim.dynamics import (
    PCCDynamicsConfig,
    PCCDynamicsState,
    contact_generalized_force,
    damping_matrix,
    mass_matrix,
    stiffness_matrix,
    step_dynamics,
)
from continuum_sim.kinematics.differential import finite_difference_position_jacobian
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.model.robot_params import PCC_VALUES_PER_SEGMENT
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.control_layout import ControlLayout
from continuum_sim.system.types import (
    ArmTendonRateCommand,
    RobotSystemCommand,
    RobotSystemState,
)
from continuum_sim.tasks.engine_surface_path import CleaningWaypoint


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
    ) -> None:
        waypoints = np.asarray(waypoints_world, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] == 0:
            raise ValueError("waypoints_world must have shape (N, 3) with N > 0.")
        if waypoint_tolerance_m < 0.0:
            raise ValueError("waypoint_tolerance_m must be non-negative.")
        self.assembly = assembly
        self.waypoints_world = waypoints.copy()
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
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
        low_level_config = (
            CoordinatedTrackingConfig(
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
    ) -> RobotSystemCommand:
        position = state.arms[self._executor_name].tip_pose_world.position
        achieved_index = self.active_index
        achieved_error = float(
            np.linalg.norm(self.waypoints_world[achieved_index] - position)
        )
        scheduler_paused = not (advance and self.advance_enabled)
        if not scheduler_paused:
            self.scheduler.update(error_norm_m=achieved_error)
        waypoint_advanced = self.done or self.active_index != achieved_index
        step = self._task_step()
        command = self._controller.compute_command(state, step)
        controller_metadata = command.metadata
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
                "achieved_waypoint_index": (
                    achieved_index if waypoint_advanced else -1
                ),
                "achieved_waypoint_error_m": (
                    achieved_error if waypoint_advanced else np.nan
                ),
                "waypoint_advanced": waypoint_advanced,
                "tracking_complete": self.done,
                "tracking_approach": bool(
                    self.approach_mask[self.active_index]
                ),
            },
        )

    def _task_step(self) -> TaskStep:
        velocity_override = self.executor_velocity_override_world
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
        approach_duration_s: float = 0.0,
        time_parameterization: str = "uniform_waypoint",
        trajectory_interpolation: str = "linear",
        reference_governor_enabled: bool = False,
        reference_error_slow_m: float = 0.003,
        reference_error_stop_m: float = 0.010,
        reference_lead_slow_ratio: float = 0.60,
        reference_lead_stop_ratio: float = 0.90,
        reference_scale_recovery_per_s: float = 1.0,
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
        if approach_duration_s < 0.0 or not np.isfinite(approach_duration_s):
            raise ValueError("approach_duration_s must be non-negative and finite.")
        if loop and approach_duration_s > 0.0:
            raise ValueError("A separately timed approach cannot be looped.")
        if time_parameterization not in ("uniform_waypoint", "arc_length"):
            raise ValueError(
                "time_parameterization must be 'uniform_waypoint' or 'arc_length'."
            )
        if trajectory_interpolation not in ("linear", "corner_stop_hermite"):
            raise ValueError(
                "trajectory_interpolation must be 'linear' or "
                "'corner_stop_hermite'."
            )
        if loop and (
            time_parameterization != "uniform_waypoint"
            or trajectory_interpolation != "linear"
        ):
            raise ValueError(
                "Looped time tracking currently requires uniform linear sampling."
            )
        governor_values = {
            "reference_error_slow_m": reference_error_slow_m,
            "reference_error_stop_m": reference_error_stop_m,
            "reference_lead_slow_ratio": reference_lead_slow_ratio,
            "reference_lead_stop_ratio": reference_lead_stop_ratio,
            "reference_scale_recovery_per_s": reference_scale_recovery_per_s,
        }
        for name, value in governor_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if reference_error_stop_m <= reference_error_slow_m:
            raise ValueError(
                "reference_error_stop_m must exceed reference_error_slow_m."
            )
        if reference_lead_stop_ratio <= reference_lead_slow_ratio:
            raise ValueError(
                "reference_lead_stop_ratio must exceed reference_lead_slow_ratio."
            )
        self.assembly = assembly
        self.waypoints_world = waypoints.copy()
        self.trajectory_duration_s = float(trajectory_duration_s)
        self.approach_duration_s = float(approach_duration_s)
        self.time_parameterization = str(time_parameterization)
        self.trajectory_interpolation = str(trajectory_interpolation)
        self.reference_governor_enabled = bool(reference_governor_enabled)
        self.reference_error_slow_m = float(reference_error_slow_m)
        self.reference_error_stop_m = float(reference_error_stop_m)
        self.reference_lead_slow_ratio = float(reference_lead_slow_ratio)
        self.reference_lead_stop_ratio = float(reference_lead_stop_ratio)
        self.reference_scale_recovery_per_s = float(
            reference_scale_recovery_per_s
        )
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
        self._waypoint_times_s = self._build_waypoint_times()
        self.total_duration_s = float(self._waypoint_times_s[-1])
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
        self._last_state_time_s: float | None = None
        self._reference_time_s = 0.0
        self._reference_scale = 1.0
        self._reference_error_scale = 1.0
        self._reference_lead_scale = 1.0
        self._reference_lead_utilization = 0.0
        self._reference_hold_reason = "none"
        self._corner_hold_time_s: float | None = None
        self._elapsed_s = 0.0
        self._active_index = 0
        self._done = False
        self._runtime_approach_initialized = False
        low_level_config = (
            CoordinatedTrackingConfig(
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
        if not self.done:
            return ""
        return (
            "reference_complete"
            if self.reference_governor_enabled
            else "duration_elapsed"
        )

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        self._initialize_runtime_approach(state)
        state_time = float(state.time_s)
        if self._start_time_s is None:
            self._start_time_s = state_time
        wall_elapsed = max(0.0, state_time - self._start_time_s)
        physical_dt = (
            0.0
            if self._last_state_time_s is None
            else max(0.0, state_time - self._last_state_time_s)
        )
        self._last_state_time_s = state_time
        if self.reference_governor_enabled:
            raw_scale = self._reference_governor_scale(state)
            self._reference_scale = min(
                raw_scale,
                self._reference_scale
                + self.reference_scale_recovery_per_s * physical_dt,
            )
            advanced_reference_time = (
                self._reference_time_s + self._reference_scale * physical_dt
            )
            unclamped_reference_time = advanced_reference_time
            advanced_reference_time = self._hold_at_crossed_corner(
                self._reference_time_s,
                advanced_reference_time,
            )
            if advanced_reference_time < unclamped_reference_time:
                self._reference_hold_reason = (
                    "corner"
                    if self._reference_hold_reason == "none"
                    else f"{self._reference_hold_reason}+corner"
                )
            self._reference_time_s = (
                advanced_reference_time
                if self.loop
                else min(self.total_duration_s, advanced_reference_time)
            )
        else:
            self._reference_scale = 1.0
            self._reference_error_scale = 1.0
            self._reference_lead_scale = 1.0
            self._reference_lead_utilization = 0.0
            self._reference_hold_reason = "disabled"
            self._reference_time_s = wall_elapsed
        reference_time = self._reference_time_s
        reference_position = self._sample_trajectory(reference_time)[0]
        terminal_error = float(
            np.linalg.norm(
                reference_position
                - state.arms[self._executor_name].tip_pose_world.position
            )
        )
        reference_at_end = (
            (not self.loop) and reference_time >= self.total_duration_s
        )
        complete = reference_at_end and (
            not self.reference_governor_enabled
            or terminal_error <= self.waypoint_tolerance_m
        )
        step, source_index, local_fraction = self._target_at(
            reference_time,
            complete=complete,
        )
        self._elapsed_s = reference_time
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
                "tracking_wall_elapsed_s": wall_elapsed,
                "tracking_elapsed_s": min(reference_time, self.total_duration_s),
                "reference_time_s": min(reference_time, self.total_duration_s),
                "reference_time_scale": self._reference_scale,
                "reference_error_scale": self._reference_error_scale,
                "reference_lead_scale": self._reference_lead_scale,
                "reference_lead_utilization": (
                    self._reference_lead_utilization
                ),
                "reference_hold_reason": self._reference_hold_reason,
                "reference_governor_enabled": self.reference_governor_enabled,
                "time_parameterization": self.time_parameterization,
                "trajectory_interpolation": self.trajectory_interpolation,
                "approach_time_s": min(
                    reference_time,
                    self.approach_duration_s,
                ),
                "trajectory_time_s": min(
                    max(reference_time - self.approach_duration_s, 0.0),
                    self.trajectory_duration_s,
                ),
                "trajectory_phase": (
                    "approach"
                    if self.approach_duration_s > 0.0
                    and reference_time < self.approach_duration_s
                    else (
                        "complete"
                        if complete
                        else ("settling" if reference_at_end else "path")
                    )
                ),
                "approach_duration_s": self.approach_duration_s,
                "trajectory_duration_s": self.trajectory_duration_s,
                "tracking_total_duration_s": self.total_duration_s,
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
                "tracking_terminal_error_m": terminal_error,
                "tracking_approach": bool(self.approach_mask[source_index]),
                "approach_start_source": (
                    "measured_executor_tip"
                    if self.approach_duration_s > 0.0
                    else "configured_waypoints"
                ),
            },
        )

    def _target_at(
        self,
        elapsed_s: float,
        *,
        complete: bool = False,
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
                ),
                status=TaskStatus(
                    task_type="tracking",
                    phase=("complete" if complete else "tracking"),
                    active_index=int(source_index),
                    complete=complete,
                    stop_reason=(
                        (
                            "reference_complete"
                            if self.reference_governor_enabled
                            else "duration_elapsed"
                        )
                        if complete
                        else ""
                    ),
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
            clamped = min(max(elapsed_s, 0.0), self.total_duration_s)
            index = int(np.searchsorted(self._waypoint_times_s, clamped, side="right")) - 1
            index = int(np.clip(index, 0, count - 2))
            next_index = index + 1
            segment_start = float(self._waypoint_times_s[index])
            segment_dt = float(
                self._waypoint_times_s[next_index] - self._waypoint_times_s[index]
            )
            fraction = float(np.clip((clamped - segment_start) / segment_dt, 0.0, 1.0))
        current = self.waypoints_world[index]
        following = self.waypoints_world[next_index]
        if (
            self.trajectory_interpolation == "corner_stop_hermite"
            and not self.loop
            and not self.approach_mask[index]
        ):
            start_velocity = self._path_waypoint_velocity(index)
            end_velocity = self._path_waypoint_velocity(next_index)
            position, velocity = _sample_cubic_hermite(
                current,
                following,
                start_velocity,
                end_velocity,
                segment_dt,
                fraction,
            )
        else:
            position = current + fraction * (following - current)
            velocity = (following - current) / segment_dt
        active_index = next_index if fraction > np.finfo(float).eps else index
        if (not self.loop) and elapsed_s >= self.total_duration_s:
            velocity = np.zeros(3, dtype=float)
            active_index = count - 1
        return position, velocity, active_index, fraction

    def _build_waypoint_times(self) -> np.ndarray:
        count = self.waypoints_world.shape[0]
        if count == 1:
            return np.array([self.trajectory_duration_s], dtype=float)
        if self.approach_duration_s <= 0.0:
            return self._path_waypoint_times(0, start_time_s=0.0)
        path_indices = np.flatnonzero(~self.approach_mask)
        if path_indices.size == 0:
            raise ValueError("A separately timed approach requires path waypoints.")
        path_start = int(path_indices[0])
        if path_start < 1 or np.any(self.approach_mask[path_start:]):
            raise ValueError(
                "approach_mask must be one contiguous prefix before path waypoints."
            )
        if count - path_start < 2:
            raise ValueError(
                "A separately timed trajectory requires at least two path waypoints."
            )
        approach_times = np.linspace(
            0.0,
            self.approach_duration_s,
            path_start + 1,
        )
        path_times = self._path_waypoint_times(
            path_start,
            start_time_s=self.approach_duration_s,
        )
        return np.concatenate((approach_times[:-1], path_times))

    def _path_waypoint_times(
        self,
        path_start: int,
        *,
        start_time_s: float,
    ) -> np.ndarray:
        path = self.waypoints_world[path_start:]
        if path.shape[0] < 2:
            return np.array([start_time_s + self.trajectory_duration_s])
        if self.time_parameterization == "uniform_waypoint":
            offsets = np.linspace(0.0, self.trajectory_duration_s, path.shape[0])
            return start_time_s + offsets
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        if np.any(lengths <= np.finfo(float).eps):
            raise ValueError(
                "arc_length time parameterization requires distinct consecutive "
                "path waypoints."
            )
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        return start_time_s + self.trajectory_duration_s * (
            cumulative / cumulative[-1]
        )

    def _path_waypoint_velocity(self, index: int) -> np.ndarray:
        count = self.waypoints_world.shape[0]
        path_indices = np.flatnonzero(~self.approach_mask)
        path_start = int(path_indices[0])
        if index <= path_start or index >= count - 1:
            return np.zeros(3, dtype=float)
        previous_delta = (
            self.waypoints_world[index] - self.waypoints_world[index - 1]
        )
        next_delta = (
            self.waypoints_world[index + 1] - self.waypoints_world[index]
        )
        previous_length = float(np.linalg.norm(previous_delta))
        next_length = float(np.linalg.norm(next_delta))
        if (
            previous_length <= np.finfo(float).eps
            or next_length <= np.finfo(float).eps
        ):
            return np.zeros(3, dtype=float)
        previous_direction = previous_delta / previous_length
        next_direction = next_delta / next_length
        if float(np.dot(previous_direction, next_direction)) < np.cos(
            np.deg2rad(15.0)
        ):
            return np.zeros(3, dtype=float)
        tangent = previous_direction + next_direction
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= np.finfo(float).eps:
            return np.zeros(3, dtype=float)
        previous_dt = float(
            self._waypoint_times_s[index] - self._waypoint_times_s[index - 1]
        )
        next_dt = float(
            self._waypoint_times_s[index + 1] - self._waypoint_times_s[index]
        )
        speed = 0.5 * (
            previous_length / previous_dt + next_length / next_dt
        )
        return speed * tangent / tangent_norm

    def _hold_at_crossed_corner(
        self,
        start_time_s: float,
        proposed_time_s: float,
    ) -> float:
        if self.trajectory_interpolation != "corner_stop_hermite":
            return proposed_time_s
        tolerance = 1.0e-12
        if (
            self._corner_hold_time_s is not None
            and abs(start_time_s - self._corner_hold_time_s) <= tolerance
        ):
            self._corner_hold_time_s = None
            return proposed_time_s
        path_indices = np.flatnonzero(~self.approach_mask)
        path_start = int(path_indices[0])
        for index in range(path_start, self.waypoints_world.shape[0] - 1):
            corner_time = float(self._waypoint_times_s[index])
            if not (
                corner_time > start_time_s + tolerance
                and corner_time <= proposed_time_s + tolerance
            ):
                continue
            if np.linalg.norm(self._path_waypoint_velocity(index)) <= tolerance:
                self._corner_hold_time_s = corner_time
                return corner_time
        return proposed_time_s

    def _initialize_runtime_approach(self, state: RobotSystemState) -> None:
        if self._runtime_approach_initialized:
            return
        self._runtime_approach_initialized = True
        if self.approach_duration_s <= 0.0:
            return
        path_indices = np.flatnonzero(~self.approach_mask)
        path_start = int(path_indices[0])
        start = state.arms[self._executor_name].tip_pose_world.position
        end = self.waypoints_world[path_start]
        progress = np.linspace(0.0, 1.0, path_start, endpoint=False)
        blend = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
        self.waypoints_world[:path_start] = (
            start[None, :] + blend[:, None] * (end - start)[None, :]
        )

    def _reference_governor_scale(self, state: RobotSystemState) -> float:
        reference_position = self._sample_trajectory(self._reference_time_s)[0]
        executor = state.arms[self._executor_name]
        tracking_error = float(
            np.linalg.norm(reference_position - executor.tip_pose_world.position)
        )
        self._reference_error_scale = _slowdown_scale(
            tracking_error,
            self.reference_error_slow_m,
            self.reference_error_stop_m,
        )
        lead_limit = self.assembly.arms[
            self._executor_name
        ].spatial_arm.limits.target_lead_m
        lead_ratio = np.divide(
            np.abs(executor.tendon_target_m - executor.tendon_displacement_m),
            lead_limit,
            out=np.zeros_like(lead_limit),
            where=lead_limit > 0.0,
        )
        self._reference_lead_utilization = float(np.max(lead_ratio))
        self._reference_lead_scale = _slowdown_scale(
            self._reference_lead_utilization,
            self.reference_lead_slow_ratio,
            self.reference_lead_stop_ratio,
        )
        reasons: list[str] = []
        if self._reference_error_scale < 1.0:
            reasons.append("tracking_error")
        if self._reference_lead_scale < 1.0:
            reasons.append("tendon_lead")
        self._reference_hold_reason = "+".join(reasons) if reasons else "none"
        return min(self._reference_error_scale, self._reference_lead_scale)


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
        observer_roi_world: np.ndarray | None = None,
        observer_control_mode: str = "tracking",
        target_advance_mode: str = "tolerance",
        controller_dt_s: float = 0.02,
        advance_time_s: float | None = None,
        advance_steps: int | None = None,
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
    ) -> None:
        if scene_query is None:
            raise ValueError("NavigationController requires a scene query.")
        self._tracking = WaypointTrackingController(
            assembly,
            waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            observer_roi_world=observer_roi_world,
            observer_control_mode=observer_control_mode,
            target_advance_mode=target_advance_mode,
            controller_dt_s=controller_dt_s,
            advance_time_s=advance_time_s,
            advance_steps=advance_steps,
            scene_query=scene_query,
            executor_position_gain=executor_position_gain,
            observer_position_gain=observer_position_gain,
            feedforward_speed_mps=feedforward_speed_mps,
            max_target_speed_mps=max_target_speed_mps,
            solver_config=solver_config,
            enforce_backend_tendon_limits=enforce_backend_tendon_limits,
            coordinated_config=coordinated_config,
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
        if self.done and self.clearance_violated:
            command = RobotSystemCommand.zeros(
                {name: values.tendon_rate_mps.size for name, values in command.arms.items()}
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
        dynamics_config: PCCDynamicsConfig | None = None,
        admittance_config: ContactTriggeredAdmittanceConfig | None = None,
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
        self.surface_normal_world = np.asarray(surface_normal_world, dtype=float)
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
            "advance_enabled": (control_type != "contact_triggered_admittance"),
            "observer_control_mode": observer_control_mode,
            "executor_position_gain": executor_position_gain,
            "observer_position_gain": observer_position_gain,
            "feedforward_speed_mps": feedforward_speed_mps,
            "max_target_speed_mps": max_target_speed_mps,
            "solver_config": solver_config,
            "enforce_backend_tendon_limits": enforce_backend_tendon_limits,
            "coordinated_config": coordinated_config,
            "trajectory_duration_s": trajectory_duration_s,
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
        executor = assembly.arms[self._executor_name]
        self._executor_bending_model = self._tracking._controller.solver.layout.bending_models[
            self._executor_name
        ]
        self._dynamics_config = (
            PCCDynamicsConfig.default(executor.spatial_arm.params)
            if dynamics_config is None
            else dynamics_config
        )
        self._admittance = (
            None
            if admittance_config is None
            else ContactTriggeredAdmittanceTracker(admittance_config)
        )

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
        original_waypoint = self._tracking.waypoints_world[waypoint_index].copy()
        if self.phase == "contact":
            self._tracking.waypoints_world[waypoint_index] = (
                strategy_result.corrected_waypoint
            )
        try:
            if self._tracking_mode == "time":
                command = self._tracking.compute_command(state)
            else:
                command = self._tracking.compute_command(
                    state,
                    advance=not strategy_result.controls_waypoint_advance,
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
                "wiping_dynamic_requested": (
                    self.control_type == "dynamic_adaptive_impedance"
                ),
                "wiping_dynamic_system_controller_active": False,
                "target_normal_force_n": float(self.target_force_n[waypoint_index]),
                "estimated_normal_force_n": estimated_force,
                "force_error_n": force_error,
                "normal_force_gain": self.normal_force_gain,
                "force_proxy_stiffness_n_m": self.force_proxy_stiffness_n_m,
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
        self._executor_bending_model = self._tracking._controller.solver.layout.bending_models[
            self._executor_name
        ]

    def _with_dynamic_executor_command(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        target_position_world: np.ndarray,
        normal_force_n: float,
        normal_world: np.ndarray,
    ) -> RobotSystemCommand:
        arm_config = self.assembly.arms[self._executor_name]
        arm_state = state.arms[self._executor_name]
        model = self._executor_bending_model
        q = model.to_q(model.estimate(arm_state.tendon_displacement_m))
        qdot = model.to_q(model.estimate(arm_state.tendon_velocity_mps))
        q = _zero_axial(q)
        qdot = _zero_axial(qdot)
        desired_world = 4.0 * (
            np.asarray(target_position_world, dtype=float)
            - arm_state.tip_pose_world.position
        )
        mount = state.base.pose.compose(arm_config.mount_pose)
        rotation = mount.as_matrix()[:3, :3]
        desired_local = rotation.T @ desired_world
        tip_jacobian = finite_difference_position_jacobian(
            q,
            arm_config.spatial_arm.params,
        )
        active = _bending_dof_mask(arm_config.spatial_arm.params)
        desired_qdot = np.zeros_like(q)
        desired_qdot[active] = np.linalg.pinv(tip_jacobian[:, active]) @ desired_local
        desired_qddot = (desired_qdot - qdot) / max(self.controller_dt_s, 1.0e-12)
        M = mass_matrix(q, arm_config.spatial_arm.params, self._dynamics_config)
        D = damping_matrix(arm_config.spatial_arm.params, self._dynamics_config)
        K = stiffness_matrix(arm_config.spatial_arm.params, self._dynamics_config)
        contact_tau = contact_generalized_force(
            q,
            float(max(0.0, normal_force_n if np.isfinite(normal_force_n) else 0.0))
            * (rotation.T @ np.asarray(normal_world, dtype=float)),
            arm_config.spatial_arm.params,
        )
        tau = M @ desired_qddot + D @ qdot + K @ q - contact_tau
        predicted, _info = step_dynamics(
            PCCDynamicsState(q=q, qdot=qdot),
            applied_generalized_force=tau + contact_tau,
            params=arm_config.spatial_arm.params,
            config=self._dynamics_config,
            dt=self.controller_dt_s,
        )
        tendon_rate = model.to_tendon(predicted.qdot[active])
        arms = dict(command.arms)
        arms[self._executor_name] = ArmTendonRateCommand(tendon_rate)
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=arms,
            metadata={
                **command.metadata,
                "dynamic_predicted_q": predicted.q.copy(),
                "dynamic_predicted_qdot": predicted.qdot.copy(),
                "dynamic_stiffness_diag": np.diag(K),
                "dynamic_damping_diag": np.diag(D),
            },
        )


class EngineCleaningSystemController:
    """Scenario adapter for the task-space engine cleaning controller."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        waypoints_world: np.ndarray,
        normals_world: np.ndarray,
        phases: tuple[str, ...],
        target_force_n: np.ndarray,
        standoff_distance_m: np.ndarray,
        *,
        scene_query: EngineSceneQueryProtocol | None,
        gains: EngineCleaningControllerGains,
        controller_dt_s: float,
        observer_roi_world: np.ndarray | None = None,
        observer_control_mode: str = "tracking",
        executor_position_gain: float = 4.0,
        observer_position_gain: float = 5.0,
        max_target_speed_mps: float | None = None,
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        enforce_backend_tendon_limits: bool = False,
        coordinated_config: CoordinatedTrackingConfig | None = None,
    ) -> None:
        self.assembly = assembly
        self.scene_query = scene_query
        self._executor_name = _single_role_name(assembly, "executor")
        self._waypoints_world = np.asarray(waypoints_world, dtype=float).copy()
        self._observer_roi_world = (
            None
            if observer_roi_world is None
            else np.asarray(observer_roi_world, dtype=float).copy()
        )
        self._observer_control_mode = str(observer_control_mode)
        low_level_config = (
            CoordinatedTrackingConfig(
                executor_position_gain=executor_position_gain,
                observer_position_gain=observer_position_gain,
                max_target_speed_mps=max_target_speed_mps,
                enforce_backend_tendon_limits=enforce_backend_tendon_limits,
            )
            if coordinated_config is None
            else coordinated_config
        )
        self._low_level = UnifiedLowLevelController(
            assembly,
            coordinated_config=low_level_config,
            solver_config=solver_config,
            scene_query=None,
        )
        self._controller = EngineCleaningController(
            gains,
            _cleaning_waypoints(
                waypoints_world,
                normals_world,
                phases,
                target_force_n,
                standoff_distance_m,
            ),
        )

    @property
    def done(self) -> bool:
        return self._controller.is_done() or self._controller.safety_stop

    @property
    def terminal_reason(self) -> str:
        if self._controller.safety_stop:
            return str(self._controller.stop_reason or "safety_stop")
        return "completed" if self._controller.is_done() else ""

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        executor = state.arms[self._executor_name]
        distance = None
        force = 0.0
        in_contact = False
        if self.scene_query is not None:
            query = self.scene_query.nearest_distance(executor.tip_pose_world.position)
            distance = float(query.distance_m)
            force = max(0.0, -distance * 600.0)
            in_contact = distance <= 0.0
        cleaning_command = self._controller.step(
            EngineCleaningFeedback(
                tcp_pose=executor.tip_pose_world,
                measured_normal_force_n=force,
                contact_distance_m=distance,
                in_contact=in_contact,
                timestamp_s=state.time_s,
            )
        )
        active_index = int(
            np.clip(
                cleaning_command.active_waypoint_index,
                0,
                self._waypoints_world.shape[0] - 1,
            )
        )
        step = TaskStep(
            intent=SystemTaskIntent(
                executor=CartesianTaskIntent(
                    target_position_world=self._waypoints_world[active_index],
                    feedforward_velocity_world=(
                        cleaning_command.desired_tcp_velocity_world
                    ),
                    control_mode="velocity",
                ),
                observer=ObserverTaskIntent(
                    control_mode=self._observer_control_mode,
                    roi_position_world=self._observer_roi_world,
                ),
            ),
            status=TaskStatus(
                task_type="engine_cleaning",
                phase=cleaning_command.phase,
                active_index=active_index,
                complete=self.done,
                stop_reason=cleaning_command.stop_reason or "",
            ),
        )
        command = self._low_level.compute_command(state, step)
        controller_metadata = command.metadata
        if self.done:
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
                **cleaning_command.metadata,
                "task_type": "engine_cleaning",
                "engine_cleaning_controller": "task_space",
                "engine_cleaning_phase": cleaning_command.phase,
                "engine_cleaning_waypoint_index": cleaning_command.active_waypoint_index,
                "engine_cleaning_waypoint_reached": cleaning_command.waypoint_reached,
                "engine_cleaning_safety_stop": cleaning_command.safety_stop,
                "engine_cleaning_stop_reason": cleaning_command.stop_reason,
                "engine_cleaning_contact_distance_m": (
                    np.nan if distance is None else distance
                ),
                "engine_cleaning_estimated_force_n": force,
            },
        )


def _sample_cubic_hermite(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    duration_s: float,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = float(np.clip(fraction, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    start_tangent = duration_s * np.asarray(start_velocity, dtype=float)
    end_tangent = duration_s * np.asarray(end_velocity, dtype=float)
    position = h00 * start + h10 * start_tangent + h01 * end + h11 * end_tangent
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -dh00
    dh11 = 3.0 * u2 - 2.0 * u
    velocity = (
        dh00 * start
        + dh10 * start_tangent
        + dh01 * end
        + dh11 * end_tangent
    ) / duration_s
    return position, velocity


def _slowdown_scale(value: float, slow: float, stop: float) -> float:
    if value <= slow:
        return 1.0
    if value >= stop:
        return 0.0
    fraction = (value - slow) / (stop - slow)
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return float(1.0 - smooth)


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


def _bending_dof_mask(params) -> np.ndarray:
    mask = np.ones(params.q_size, dtype=bool)
    mask[PCC_VALUES_PER_SEGMENT - 1 :: PCC_VALUES_PER_SEGMENT] = False
    return mask


def _zero_axial(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=float).copy()
    result[PCC_VALUES_PER_SEGMENT - 1 :: PCC_VALUES_PER_SEGMENT] = 0.0
    return result


def _cleaning_waypoints(
    positions: np.ndarray,
    normals: np.ndarray,
    phases: tuple[str, ...],
    target_force_n: np.ndarray,
    standoff_distance_m: np.ndarray,
) -> tuple[CleaningWaypoint, ...]:
    position_array = np.asarray(positions, dtype=float)
    normal_array = np.asarray(normals, dtype=float)
    waypoints: list[CleaningWaypoint] = []
    for index in range(position_array.shape[0]):
        normal = normal_array[index]
        tangent_u = _default_tangent(normal)
        tangent_v = np.cross(normal / np.linalg.norm(normal), tangent_u)
        tangent_v = tangent_v / np.linalg.norm(tangent_v)
        waypoints.append(
            CleaningWaypoint(
                position=position_array[index],
                normal=normal,
                tangent_u=tangent_u,
                tangent_v=tangent_v,
                phase=phases[index],
                target_force_n=float(target_force_n[index]),
                standoff_distance_m=float(standoff_distance_m[index]),
                index=index,
            )
        )
    return tuple(waypoints)


def _default_tangent(normal: np.ndarray) -> np.ndarray:
    unit = np.asarray(normal, dtype=float)
    unit = unit / np.linalg.norm(unit)
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, unit))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = candidate - np.dot(candidate, unit) * unit
    return tangent / np.linalg.norm(tangent)
