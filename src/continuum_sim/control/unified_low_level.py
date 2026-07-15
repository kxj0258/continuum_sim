"""Shared low-level Cartesian-to-whole-body command pipeline."""

from __future__ import annotations

import numpy as np

from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
)
from continuum_sim.control.task_intent import TaskStep
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class UnifiedLowLevelController:
    """Convert typed task intent into the common whole-body tendon-rate command."""

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        *,
        coordinated_config: CoordinatedTrackingConfig = CoordinatedTrackingConfig(),
        solver_config: WholeBodyControllerConfig = WholeBodyControllerConfig(),
        scene_query: EngineSceneQueryProtocol | None = None,
    ) -> None:
        self.assembly = assembly
        self._executor_name = _single_role_name(assembly, "executor")
        self._controller = CoordinatedTrackingController(
            assembly,
            CoordinatedTrackingTarget(
                executor_position_world=np.zeros(3, dtype=float),
                executor_velocity_world=np.zeros(3, dtype=float),
            ),
            config=coordinated_config,
            solver_config=solver_config,
            scene_query=scene_query,
        )

    @property
    def solver(self):
        """Expose the shared solver for existing diagnostics and dynamics adapters."""

        return self._controller.solver

    @property
    def config(self) -> CoordinatedTrackingConfig:
        """Expose the active shared low-level profile for diagnostics."""

        return self._controller.config

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._controller.last_diagnostics

    def compute_command(
        self,
        state: RobotSystemState,
        step: TaskStep,
    ) -> RobotSystemCommand:
        executor = step.intent.executor
        measured_position = state.arms[
            self._executor_name
        ].tip_pose_world.position
        servo_position = (
            measured_position
            if executor.control_mode == "velocity"
            else executor.target_position_world
        )
        scaled_feedforward_velocity = (
            executor.feedforward_velocity_world * self.config.feedforward_gain
            if executor.control_mode == "position"
            else executor.feedforward_velocity_world.copy()
        )
        observer = step.intent.observer
        self._controller.set_target(
            CoordinatedTrackingTarget(
                executor_position_world=servo_position,
                executor_velocity_world=scaled_feedforward_velocity,
                observer_roi_position_world=(
                    None if observer is None else observer.roi_position_world
                ),
                observer_executor_offset_world=(
                    np.array([0.0, -0.04, 0.02], dtype=float)
                    if observer is None
                    else observer.executor_offset_world
                ),
                observer_roi_blend=(
                    0.25 if observer is None else observer.roi_blend
                ),
                observer_control_mode=(
                    "disabled" if observer is None else observer.control_mode
                ),
            )
        )
        command = self._controller.compute_command(state)
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                **step.intent.metadata,
                **step.status.metadata,
                "task_intent_control_mode": executor.control_mode,
                "task_intent_target_world": (
                    executor.target_position_world.copy()
                ),
                "task_intent_velocity_world": (
                    executor.feedforward_velocity_world.copy()
                ),
                "executor_feedforward_gain": self.config.feedforward_gain,
                "executor_scaled_feedforward_velocity_world": (
                    scaled_feedforward_velocity.copy()
                ),
                "task_status_type": step.status.task_type,
                "task_status_phase": step.status.phase,
                "task_status_active_index": step.status.active_index,
                "task_status_complete": step.status.complete,
                "task_status_stop_reason": step.status.stop_reason,
            },
        )


def _single_role_name(assembly: RobotAssemblyConfig, role: str) -> str:
    names = [arm.name for arm in assembly.enabled_arms if arm.role == role]
    if len(names) != 1:
        raise ValueError(f"Assembly must contain exactly one enabled {role!r} arm.")
    return names[0]
