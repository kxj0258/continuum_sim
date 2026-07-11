"""Controllers used by scenario-driven baseline and engine workflows."""

from __future__ import annotations

import numpy as np

from continuum_sim.control.waypoint_scheduler import WaypointScheduler
from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.control.wiping_force_strategies import (
    WipingForceContext,
    WipingForceStrategy,
    default_wiping_force_strategy,
)
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


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
        coordinated_config: CoordinatedTrackingConfig | None = None,
        observer_executor_offset_world: np.ndarray | None = None,
        observer_roi_blend: float = 0.25,
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
        self._controller = CoordinatedTrackingController(
            assembly,
            self._target(),
            config=(
                CoordinatedTrackingConfig(
                    executor_position_gain=executor_position_gain,
                    observer_position_gain=observer_position_gain,
                    max_target_speed_mps=max_target_speed_mps,
                )
                if coordinated_config is None
                else coordinated_config
            ),
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
        scheduler_paused = bool(
            self._controller.last_diagnostics.get(
                "inter_arm_executor_frozen",
                False,
            )
            or self._controller.last_diagnostics.get(
                "inter_arm_hard_stop",
                False,
            )
        )
        if advance and not scheduler_paused:
            self.scheduler.update(error_norm_m=achieved_error)
        waypoint_advanced = self.done or self.active_index != achieved_index
        target = self._target()
        self._controller.set_target(target)
        command = self._controller.compute_command(state)
        controller_metadata = command.metadata
        if self.done:
            command = RobotSystemCommand.zeros(
                {
                    arm.name: arm.spatial_arm.tendon_count
                    for arm in self.assembly.enabled_arms
                }
            )
        command_error = float(
            np.linalg.norm(target.executor_position_world - position)
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
                "executor_target_world": target.executor_position_world.copy(),
                "executor_error_m": command_error,
                "executor_feedforward_velocity_world": (
                    target.executor_velocity_world.copy()
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

    def _target(self) -> CoordinatedTrackingTarget:
        return CoordinatedTrackingTarget(
            executor_position_world=self.waypoints_world[self.active_index],
            executor_velocity_world=self._feedforward_velocity(),
            observer_roi_position_world=self.observer_roi_world,
            observer_executor_offset_world=self.observer_executor_offset_world,
            observer_roi_blend=self.observer_roi_blend,
        )

    def _feedforward_velocity(self) -> np.ndarray:
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
        target_advance_mode: str = "tolerance",
        controller_dt_s: float = 0.02,
        advance_time_s: float | None = None,
        advance_steps: int | None = None,
    ) -> None:
        if scene_query is None:
            raise ValueError("NavigationController requires a scene query.")
        self._tracking = WaypointTrackingController(
            assembly,
            waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            observer_roi_world=observer_roi_world,
            target_advance_mode=target_advance_mode,
            controller_dt_s=controller_dt_s,
            advance_time_s=advance_time_s,
            advance_steps=advance_steps,
            scene_query=scene_query,
        )
        self.scene_query = scene_query
        self.min_clearance_m = min_clearance_m
        self.terminate_on_clearance_violation = terminate_on_clearance_violation
        self.clearance_violated = False
        self._last_clearance_query = None

    @property
    def done(self) -> bool:
        return self._tracking.done or (
            self.terminate_on_clearance_violation and self.clearance_violated
        )

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
            },
        )


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
        executor_position_gain: float = 4.0,
        observer_position_gain: float = 5.0,
        feedforward_speed_mps: float = 0.0,
        max_target_speed_mps: float | None = None,
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
    ) -> None:
        self._tracking = WaypointTrackingController(
            assembly,
            waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            target_advance_mode=target_advance_mode,
            controller_dt_s=controller_dt_s,
            advance_time_s=advance_time_s,
            advance_steps=advance_steps,
            # Contact regulation intentionally replaces generic clearance
            # avoidance for the executor at the wiping surface.
            scene_query=None,
            executor_position_gain=executor_position_gain,
            observer_position_gain=observer_position_gain,
            feedforward_speed_mps=feedforward_speed_mps,
            max_target_speed_mps=max_target_speed_mps,
            solver_config=solver_config,
        )
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
        force = (
            np.zeros(self._tracking.waypoints_world.shape[0], dtype=float)
            if target_force_n is None
            else np.asarray(target_force_n, dtype=float)
        )
        if force.shape != (self._tracking.waypoints_world.shape[0],):
            raise ValueError("target_force_n must match waypoint count.")
        self.target_force_n = force
        self.control_type = control_type
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

    @property
    def done(self) -> bool:
        return self._tracking.done or self.force_limit_exceeded

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
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
        original_waypoint = self._tracking.waypoints_world[waypoint_index].copy()
        strategy_result = self.force_strategy.compute(
            WipingForceContext(
                executor=executor,
                waypoints_world=self._tracking.waypoints_world,
                waypoint_index=waypoint_index,
                phase=self.phase,
                surface_normal_world=self.surface_normal_world,
                query_normal_world=None,
                contact_error_m=contact_error,
                estimated_force_n=estimated_force,
                target_force_n=target_force,
                normal_force_gain=self.normal_force_gain,
                force_proxy_stiffness_n_m=self.force_proxy_stiffness_n_m,
                contact_tolerance_m=self.contact_tolerance_m,
                controller_dt_s=self._tracking.scheduler.controller_dt_s,
            )
        )
        if self.phase == "contact":
            self._tracking.waypoints_world[waypoint_index] = (
                strategy_result.corrected_waypoint
            )
        try:
            command = self._tracking.compute_command(
                state,
                advance=not strategy_result.controls_waypoint_advance,
            )
        finally:
            self._tracking.waypoints_world[waypoint_index] = original_waypoint
        if strategy_result.waypoint_advanced:
            self._tracking.scheduler.advance()
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
                "wiping_dynamic_system_controller_active": bool(
                    strategy_result.metadata.get(
                        "wiping_dynamic_system_controller_active",
                        False,
                    )
                ),
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
