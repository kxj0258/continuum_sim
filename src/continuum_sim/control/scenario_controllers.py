"""Controllers used by scenario-driven baseline and engine workflows."""

from __future__ import annotations

import numpy as np

from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
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
        scene_query: EngineSceneQueryProtocol | None = None,
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
        self.loop = loop
        self.active_index = 0
        self.done = False
        self._executor_name = _single_role_name(assembly, "executor")
        self._controller = CoordinatedTrackingController(
            assembly,
            self._target(),
            scene_query=scene_query,
        )

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._controller.last_diagnostics

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        position = state.arms[self._executor_name].tip_pose_world.position
        error = self.waypoints_world[self.active_index] - position
        if float(np.linalg.norm(error)) <= self.waypoint_tolerance_m:
            self._advance()
        if self.done:
            return RobotSystemCommand.zeros(
                {
                    arm.name: arm.spatial_arm.tendon_count
                    for arm in self.assembly.enabled_arms
                }
            )
        target = self._target()
        self._controller.set_target(target)
        command = self._controller.compute_command(state)
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                "task_type": "tracking",
                "waypoint_index": self.active_index,
                "executor_target_world": target.executor_position_world.copy(),
                "executor_error_m": float(np.linalg.norm(
                    target.executor_position_world - position
                )),
            },
        )

    def _target(self) -> CoordinatedTrackingTarget:
        return CoordinatedTrackingTarget(
            executor_position_world=self.waypoints_world[self.active_index],
            observer_roi_position_world=self.observer_roi_world,
        )

    def _advance(self) -> None:
        if self.active_index < self.waypoints_world.shape[0] - 1:
            self.active_index += 1
        elif self.loop:
            self.active_index = 0
        else:
            self.done = True


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
    ) -> None:
        if scene_query is None:
            raise ValueError("NavigationController requires a scene query.")
        self._tracking = WaypointTrackingController(
            assembly,
            waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            observer_roi_world=observer_roi_world,
            scene_query=scene_query,
        )
        self.scene_query = scene_query
        self.min_clearance_m = min_clearance_m
        self.terminate_on_clearance_violation = terminate_on_clearance_violation
        self.clearance_violated = False

    @property
    def done(self) -> bool:
        return self._tracking.done or (
            self.terminate_on_clearance_violation and self.clearance_violated
        )

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        clearances = [
            self.scene_query.nearest_centerline_clearance(
                arm.centerline_world
                if arm.centerline_world is not None
                else np.asarray([arm.tip_pose_world.position])
            ).distance_m
            for arm in state.arms.values()
        ]
        minimum = float(min(clearances))
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
    ) -> None:
        self._tracking = WaypointTrackingController(
            assembly,
            waypoints_world,
            waypoint_tolerance_m=waypoint_tolerance_m,
            # Contact regulation intentionally replaces generic clearance
            # avoidance for the executor at the wiping surface.
            scene_query=None,
        )
        self.scene_query = scene_query
        self.surface_normal_world = np.asarray(surface_normal_world, dtype=float)
        self.target_contact_distance_m = target_contact_distance_m
        self.contact_tolerance_m = contact_tolerance_m
        self.phase = "approach"

    @property
    def done(self) -> bool:
        return self._tracking.done

    def compute_command(self, state: RobotSystemState) -> RobotSystemCommand:
        executor = next(arm for arm in state.arms.values() if arm.role == "executor")
        distance = float("nan")
        if self.scene_query is not None:
            query = self.scene_query.nearest_distance(executor.tip_pose_world.position)
            distance = float(query.distance_m)
        if self._tracking.active_index == 0:
            self.phase = "approach"
        elif self._tracking.active_index == self._tracking.waypoints_world.shape[0] - 1:
            self.phase = "retract"
        else:
            self.phase = "contact"
        contact_error = (
            float("nan")
            if not np.isfinite(distance)
            else self.target_contact_distance_m - distance
        )
        waypoint_index = self._tracking.active_index
        original_waypoint = self._tracking.waypoints_world[waypoint_index].copy()
        if self.phase == "contact" and np.isfinite(contact_error):
            query = self.scene_query.nearest_distance(
                executor.tip_pose_world.position
            )
            self._tracking.waypoints_world[waypoint_index] = (
                original_waypoint + contact_error * query.normal
            )
        try:
            command = self._tracking.compute_command(state)
        finally:
            self._tracking.waypoints_world[waypoint_index] = original_waypoint
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                "task_type": "wiping",
                "wiping_phase": self.phase,
                "contact_distance_m": distance,
                "contact_error_m": contact_error,
                "contact_established": bool(
                    np.isfinite(contact_error)
                    and abs(contact_error) <= self.contact_tolerance_m
                ),
            },
        )


def _single_role_name(assembly: RobotAssemblyConfig, role: str) -> str:
    names = [arm.name for arm in assembly.enabled_arms if arm.role == role]
    if len(names) != 1:
        raise ValueError(f"Assembly must contain exactly one enabled {role!r} arm.")
    return names[0]
