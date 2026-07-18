"""Shared low-level facade for intent resolving and tendon-rate commands."""

from __future__ import annotations

from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingConfig,
)
from continuum_sim.control.intent_resolver import IntentResolver
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
        self._config = coordinated_config
        tendon_config = CoordinatedTrackingConfig(
            kinematics_mode=coordinated_config.kinematics_mode,
            executor_position_gain=0.0,
            executor_orientation_tracking_weight=(
                coordinated_config.executor_orientation_tracking_weight
            ),
            executor_orientation_tracking_mode=(
                coordinated_config.executor_orientation_tracking_mode
            ),
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
            observer_visual_servo_center_gain=(
                coordinated_config.observer_visual_servo_center_gain
            ),
            observer_visual_servo_depth_gain=(
                coordinated_config.observer_visual_servo_depth_gain
            ),
            observer_visual_servo_depth_target_m=(
                coordinated_config.observer_visual_servo_depth_target_m
            ),
            observer_visual_servo_max_speed_mps=(
                coordinated_config.observer_visual_servo_max_speed_mps
            ),
            observer_visual_servo_max_angular_speed_rad_s=(
                coordinated_config.observer_visual_servo_max_angular_speed_rad_s
            ),
            observer_collision_priority=coordinated_config.observer_collision_priority,
            freeze_executor_inside_safe_distance=(
                coordinated_config.freeze_executor_inside_safe_distance
            ),
            stop_all_on_critical_distance=coordinated_config.stop_all_on_critical_distance,
            centerline_samples_per_segment=(
                coordinated_config.centerline_samples_per_segment
            ),
            scene_avoidance_enabled=coordinated_config.scene_avoidance_enabled,
            executor_scene_avoidance_mode=(
                coordinated_config.executor_scene_avoidance_mode
            ),
            observer_scene_avoidance_mode=(
                coordinated_config.observer_scene_avoidance_mode
            ),
            engine_min_clearance_m=coordinated_config.engine_min_clearance_m,
            engine_influence_distance_m=(
                coordinated_config.engine_influence_distance_m
            ),
            engine_avoidance_gain=coordinated_config.engine_avoidance_gain,
            enforce_backend_tendon_limits=(
                coordinated_config.enforce_backend_tendon_limits
            ),
            priority_stack=coordinated_config.priority_stack,
        )
        self._resolver = IntentResolver(
            assembly,
            coordinated_config=coordinated_config,
            tendon_config=tendon_config,
            solver_config=solver_config,
            scene_query=scene_query,
        )

    @property
    def solver(self):
        """Expose the shared solver for existing diagnostics and dynamics adapters."""

        return self._resolver.solver

    @property
    def config(self) -> CoordinatedTrackingConfig:
        """Expose the active shared low-level profile for diagnostics."""

        return self._config

    @property
    def last_diagnostics(self) -> dict[str, object]:
        return self._resolver.last_diagnostics

    def compute_command(
        self,
        state: RobotSystemState,
        step: TaskStep,
    ) -> RobotSystemCommand:
        command = self._resolver.resolve(state, step)
        return RobotSystemCommand(
            base_twist_world=command.base_twist_world,
            arms=command.arms,
            metadata={
                **command.metadata,
                "low_level_facade": type(self).__name__,
            },
        )
