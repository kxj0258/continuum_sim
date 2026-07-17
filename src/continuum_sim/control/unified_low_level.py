"""Shared low-level Cartesian-to-whole-body command pipeline."""

from __future__ import annotations

from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
)
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
        self._config = coordinated_config
        self._task_space_servo = TaskSpaceServo(
            TaskSpaceServoConfig(
                position_gain=coordinated_config.executor_position_gain,
                feedforward_gain=coordinated_config.feedforward_gain,
                max_speed_mps=coordinated_config.max_target_speed_mps,
            )
        )
        tendon_config = CoordinatedTrackingConfig(
            kinematics_mode=coordinated_config.kinematics_mode,
            executor_position_gain=0.0,
            observer_position_gain=coordinated_config.observer_position_gain,
            feedforward_gain=1.0,
            max_target_speed_mps=None,
            inter_arm_min_distance_m=coordinated_config.inter_arm_min_distance_m,
            inter_arm_influence_distance_m=(
                coordinated_config.inter_arm_influence_distance_m
            ),
            inter_arm_hard_stop_distance_m=(
                coordinated_config.inter_arm_hard_stop_distance_m
            ),
            inter_arm_release_margin_m=(
                coordinated_config.inter_arm_release_margin_m
            ),
            inter_arm_avoidance_gain=coordinated_config.inter_arm_avoidance_gain,
            inter_arm_max_avoidance_speed_mps=(
                coordinated_config.inter_arm_max_avoidance_speed_mps
            ),
            inter_arm_collision_pair_count=(
                coordinated_config.inter_arm_collision_pair_count
            ),
            inter_arm_collision_pair_index_separation=(
                coordinated_config.inter_arm_collision_pair_index_separation
            ),
            observer_look_at_executor_tip=(
                coordinated_config.observer_look_at_executor_tip
            ),
            observer_look_at_gain=coordinated_config.observer_look_at_gain,
            observer_look_at_weight=coordinated_config.observer_look_at_weight,
            observer_look_at_distance_m=(
                coordinated_config.observer_look_at_distance_m
            ),
            observer_look_at_max_speed_mps=(
                coordinated_config.observer_look_at_max_speed_mps
            ),
            observer_collision_priority=coordinated_config.observer_collision_priority,
            freeze_executor_inside_safe_distance=(
                coordinated_config.freeze_executor_inside_safe_distance
            ),
            stop_all_on_critical_distance=coordinated_config.stop_all_on_critical_distance,
            centerline_samples_per_segment=(
                coordinated_config.centerline_samples_per_segment
            ),
            engine_min_clearance_m=coordinated_config.engine_min_clearance_m,
            engine_influence_distance_m=(
                coordinated_config.engine_influence_distance_m
            ),
            engine_avoidance_gain=coordinated_config.engine_avoidance_gain,
            enforce_backend_tendon_limits=(
                coordinated_config.enforce_backend_tendon_limits
            ),
        )
        self._controller = TendonCommandController(
            assembly,
            coordinated_config=tendon_config,
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

        return self._config

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._controller.last_diagnostics

    def compute_command(
        self,
        state: RobotSystemState,
        step: TaskStep,
    ) -> RobotSystemCommand:
        executor = step.intent.executor
        measured_position = state.arms[self._executor_name].tip_pose_world.position
        task_velocity = self._task_space_servo.compute(
            measured_position,
            TaskSpaceReference(
                target_position_world=executor.target_position_world,
                feedforward_velocity_world=executor.feedforward_velocity_world,
                control_mode=executor.control_mode,
            ),
        )
        observer = step.intent.observer
        command = self._controller.compute_command(
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
        )
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
                "executor_feedforward_gain": self._task_space_servo.config.feedforward_gain,
                "executor_scaled_feedforward_velocity_world": (
                    task_velocity.scaled_feedforward_velocity_world.copy()
                ),
                "task_space_servo": type(self._task_space_servo).__name__,
                "task_space_position_error_world": (
                    task_velocity.position_error_world.copy()
                ),
                "task_space_raw_velocity_world": (
                    task_velocity.raw_velocity_world.copy()
                ),
                "task_space_velocity_world": (
                    task_velocity.tcp_velocity_world.copy()
                ),
                "task_space_speed_limited": task_velocity.speed_limited,
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
