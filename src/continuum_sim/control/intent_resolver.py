"""Resolve task intents into whole-body tendon-rate references."""

from __future__ import annotations

import numpy as np

from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.task_space_servo import (
    TaskSpaceReference,
    TaskSpaceServo,
    TaskSpaceServoConfig,
)
from continuum_sim.control.task_intent import TaskStep
from continuum_sim.control.tendon_command_controller import (
    ObserverCommandReference,
    TendonCommandController,
)
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.model.robot_assembly import RobotAssemblyConfig
from continuum_sim.scenes.engine_query import EngineSceneQueryProtocol
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class IntentResolver:
    """Convert controller intent bundles into raw tendon-rate references.

    This layer owns task-space servoing and whole-body priority solving.  It
    intentionally stops at tendon rates; actuator compatibility and MuJoCo
    position-target tracking are handled by the execution layer.
    """

    def __init__(
        self,
        assembly: RobotAssemblyConfig,
        *,
        coordinated_config: CoordinatedTrackingConfig,
        tendon_config: CoordinatedTrackingConfig | None = None,
        solver_config: WholeBodyControllerConfig,
        scene_query: EngineSceneQueryProtocol | None = None,
    ) -> None:
        self.assembly = assembly
        self._executor_name = _single_role_name(assembly, "executor")
        self._config = coordinated_config
        self._task_space_servo = TaskSpaceServo(
            TaskSpaceServoConfig(
                position_gain=coordinated_config.executor_position_gain,
                orientation_gain=coordinated_config.executor_orientation_gain,
                feedforward_gain=coordinated_config.feedforward_gain,
                max_speed_mps=coordinated_config.max_target_speed_mps,
                max_angular_speed_rad_s=(
                    coordinated_config.max_target_angular_speed_rad_s
                ),
            )
        )
        self._tendon_controller = TendonCommandController(
            assembly,
            coordinated_config=(
                coordinated_config if tendon_config is None else tendon_config
            ),
            solver_config=solver_config,
            scene_query=scene_query,
        )

    @property
    def solver(self):
        return self._tendon_controller.solver

    @property
    def config(self) -> CoordinatedTrackingConfig:
        return self._config

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._tendon_controller.last_diagnostics

    def resolve(
        self,
        state: RobotSystemState,
        step: TaskStep,
    ) -> RobotSystemCommand:
        executor = step.intent.executor
        measured_position = state.arms[self._executor_name].tip_pose_world.position
        measured_orientation = state.arms[self._executor_name].tip_pose_world.quat
        task_velocity = self._task_space_servo.compute(
            measured_position,
            measured_orientation,
            TaskSpaceReference(
                target_position_world=executor.target_position_world,
                feedforward_velocity_world=executor.feedforward_velocity_world,
                target_orientation_world_wxyz=(
                    executor.target_orientation_world_wxyz
                ),
                feedforward_angular_velocity_world=(
                    executor.feedforward_angular_velocity_world
                ),
                control_mode=executor.control_mode,
                orientation_control_mode=executor.orientation_control_mode,
            ),
        )
        observer = step.intent.observer
        command = self._tendon_controller.compute_command(
            state,
            task_velocity,
            ObserverCommandReference(
                control_mode="disabled" if observer is None else observer.control_mode,
                roi_position_world=None if observer is None else observer.roi_position_world,
                executor_offset_world=(
                    ObserverCommandReference().executor_offset_world
                    if observer is None
                    else observer.executor_offset_world
                ),
                roi_blend=0.25 if observer is None else observer.roi_blend,
            ),
            contact=step.intent.contact,
        )
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                **step.intent.metadata,
                **step.status.metadata,
                "intent_resolver": type(self).__name__,
                "intent_resolver_output": "raw_tendon_rate_ref",
                "task_intent_control_mode": executor.control_mode,
                "task_intent_target_world": (
                    executor.target_position_world.copy()
                ),
                "task_intent_velocity_world": (
                    executor.feedforward_velocity_world.copy()
                ),
                "task_intent_target_orientation_world_wxyz": (
                    np.full(4, np.nan, dtype=float)
                    if executor.target_orientation_world_wxyz is None
                    else executor.target_orientation_world_wxyz.copy()
                ),
                "task_intent_angular_velocity_world": (
                    executor.feedforward_angular_velocity_world.copy()
                ),
                "task_intent_priority_stack": (
                    self._config.priority_stack.as_metadata()
                ),
                "executor_feedforward_gain": self._task_space_servo.config.feedforward_gain,
                "executor_scaled_feedforward_velocity_world": (
                    task_velocity.scaled_feedforward_velocity_world.copy()
                ),
                "executor_scaled_feedforward_angular_velocity_world": (
                    task_velocity.scaled_feedforward_angular_velocity_world.copy()
                ),
                "task_space_servo": type(self._task_space_servo).__name__,
                "task_space_position_error_world": (
                    task_velocity.position_error_world.copy()
                ),
                "task_space_orientation_error_world": (
                    task_velocity.orientation_error_world.copy()
                ),
                "task_space_orientation_error_norm_rad": (
                    task_velocity.orientation_error_norm_rad
                ),
                "task_space_raw_velocity_world": (
                    task_velocity.raw_velocity_world.copy()
                ),
                "task_space_raw_angular_velocity_world": (
                    task_velocity.raw_angular_velocity_world.copy()
                ),
                "task_space_velocity_world": (
                    task_velocity.tcp_velocity_world.copy()
                ),
                "task_space_angular_velocity_world": (
                    task_velocity.tcp_angular_velocity_world.copy()
                ),
                "task_space_speed_limited": task_velocity.speed_limited,
                "task_space_angular_speed_limited": (
                    task_velocity.angular_speed_limited
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
