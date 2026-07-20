"""Control configuration factories used by the scenario composition root."""

from __future__ import annotations

from dataclasses import replace

from continuum_sim.application.scenario import (
    ScenarioObserverControlConfig,
    ScenarioSceneAvoidanceConfig,
    ScenarioTrackingControlConfig,
)
from continuum_sim.control.coordinated_tracking import CoordinatedTrackingConfig
from continuum_sim.control.scenario_controllers import OnlineReachabilityConfig
from continuum_sim.control.whole_body_controller import WholeBodyControllerConfig
from continuum_sim.control.wiping_force_strategies import (
    ContactDistanceStrategy,
    ContactTriggeredAdmittanceStrategy,
    DynamicAdaptiveImpedanceStrategy,
    KinematicHybridForceStrategy,
)
from continuum_sim.kinematics.whole_body import SingularityConfig
from continuum_sim.tasks.engine_navigation import (
    EngineNavigationObserverControlSpec,
    EngineNavigationSpec,
)


def build_wiping_force_strategy(config, assembly):
    strategy_type = config.task.force_strategy.type
    if strategy_type == "contact_distance":
        return ContactDistanceStrategy()
    if strategy_type == "kinematic_hybrid":
        return KinematicHybridForceStrategy()
    if strategy_type == "contact_triggered_admittance":
        if config.task.contact_admittance is None:
            raise ValueError(
                "contact_triggered_admittance requires "
                "scenario.task.contact_admittance."
            )
        return ContactTriggeredAdmittanceStrategy(config.task.contact_admittance)
    if strategy_type == "dynamic_adaptive_impedance":
        return DynamicAdaptiveImpedanceStrategy(
            assembly,
            dynamics_config_path=(
                None
                if config.task.dynamics_config_path is None
                else str(config.task.dynamics_config_path)
            ),
            kinematics_mode=config.backend.kinematics_mode,
        )
    raise ValueError(f"Unsupported wiping force strategy {strategy_type!r}.")


def engine_navigation_with_unified_observer_control(
    spec: EngineNavigationSpec,
    observer: ScenarioObserverControlConfig,
    tracking: ScenarioTrackingControlConfig,
) -> EngineNavigationSpec:
    engine_observer = spec.observer_control
    return replace(
        spec,
        observer_control=EngineNavigationObserverControlSpec(
            position_gain=tracking.observer_position_gain,
            executor_offset_world_m=engine_observer.executor_offset_world_m,
            roi_blend=engine_observer.roi_blend,
            inter_arm_influence_distance_m=observer.influence_distance_m,
            inter_arm_safe_distance_m=observer.minimum_distance_m,
            inter_arm_critical_distance_m=observer.critical_distance_m,
            inter_arm_release_margin_m=observer.release_margin_m,
            inter_arm_avoidance_gain=observer.avoidance_gain,
            inter_arm_max_avoidance_speed_mps=observer.max_avoidance_speed_mps,
            centerline_samples_per_segment=(
                engine_observer.centerline_samples_per_segment
            ),
            observer_tracking_weight=tracking.observer_tracking_weight,
            observer_collision_weight=tracking.executor_collision_avoidance_weight,
            stop_all_on_critical_distance=False,
        ),
    )


def tracking_solver_config(
    tracking: ScenarioTrackingControlConfig,
) -> WholeBodyControllerConfig:
    return WholeBodyControllerConfig(
        executor_tracking_weight=tracking.executor_tracking_weight,
        observer_tracking_weight=tracking.observer_tracking_weight,
        executor_collision_avoidance_weight=(
            tracking.executor_collision_avoidance_weight
        ),
        base_regularization_weight=tracking.base_regularization_weight,
        tendon_regularization_weight=tracking.tendon_regularization_weight,
        singularity=SingularityConfig(
            rank_tolerance=tracking.rank_tolerance,
            minimum_singular_value=tracking.minimum_singular_value,
            nominal_damping=tracking.nominal_damping,
            maximum_damping=tracking.maximum_damping,
            minimum_velocity_scale=tracking.minimum_velocity_scale,
        ),
        decouple_arm_singularity=tracking.decouple_arm_singularity,
        singularity_strategy=tracking.singularity_strategy,
        enforce_base_velocity_limits=tracking.enforce_solver_velocity_limits,
        enforce_tendon_rate_limits=tracking.enforce_solver_velocity_limits,
        kinematics_mode=tracking.kinematics_mode,
    )


def tracking_coordinated_config(
    tracking: ScenarioTrackingControlConfig,
    observer: ScenarioObserverControlConfig | None = None,
    scene_avoidance: ScenarioSceneAvoidanceConfig | None = None,
) -> CoordinatedTrackingConfig:
    observer = ScenarioObserverControlConfig() if observer is None else observer
    scene_avoidance = (
        ScenarioSceneAvoidanceConfig()
        if scene_avoidance is None
        else scene_avoidance
    )
    return CoordinatedTrackingConfig(
        kinematics_mode=tracking.kinematics_mode,
        executor_position_gain=tracking.executor_position_gain,
        executor_orientation_gain=tracking.executor_orientation_gain,
        observer_position_gain=tracking.observer_position_gain,
        feedforward_gain=tracking.feedforward_gain,
        max_target_speed_mps=tracking_target_speed_limit(tracking),
        max_target_angular_speed_rad_s=tracking.max_target_angular_speed_rad_s,
        executor_orientation_tracking_weight=(
            tracking.executor_orientation_tracking_weight
        ),
        executor_orientation_tracking_mode=(
            tracking.executor_orientation_tracking_mode
        ),
        inter_arm_min_distance_m=observer.minimum_distance_m,
        inter_arm_influence_distance_m=observer.influence_distance_m,
        inter_arm_hard_stop_distance_m=observer.critical_distance_m,
        inter_arm_release_margin_m=observer.release_margin_m,
        inter_arm_avoidance_gain=observer.avoidance_gain,
        inter_arm_max_avoidance_speed_mps=observer.max_avoidance_speed_mps,
        inter_arm_collision_pair_count=observer.collision_pair_count,
        inter_arm_collision_pair_index_separation=(
            observer.collision_pair_index_separation
        ),
        observer_look_at_executor_tip=observer.look_at_executor_tip,
        observer_look_at_gain=observer.look_at_gain,
        observer_look_at_weight=observer.look_at_weight,
        observer_look_at_distance_m=observer.look_at_distance_m,
        observer_look_at_max_speed_mps=(
            observer.look_at_max_angular_speed_rad_s
            if observer.look_at_max_angular_speed_rad_s is not None
            else observer.look_at_max_speed_mps
        ),
        observer_visual_servo_center_gain=observer.visual_servo_center_gain,
        observer_visual_servo_depth_gain=observer.visual_servo_depth_gain,
        observer_visual_servo_depth_target_m=observer.visual_servo_depth_target_m,
        observer_visual_servo_max_speed_mps=observer.visual_servo_max_speed_mps,
        observer_visual_servo_max_angular_speed_rad_s=(
            observer.visual_servo_max_angular_speed_rad_s
        ),
        observer_collision_priority=True,
        freeze_executor_inside_safe_distance=False,
        stop_all_on_critical_distance=False,
        scene_avoidance_enabled=scene_avoidance.enabled,
        executor_scene_avoidance_mode=scene_avoidance.executor_mode,
        observer_scene_avoidance_mode=scene_avoidance.observer_mode,
        engine_min_clearance_m=scene_avoidance.engine_min_clearance_m,
        engine_influence_distance_m=scene_avoidance.engine_influence_distance_m,
        engine_avoidance_gain=scene_avoidance.engine_avoidance_gain,
        enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
        priority_stack=tracking.priority_stack,
    )


def online_reachability_config(
    tracking: ScenarioTrackingControlConfig,
) -> OnlineReachabilityConfig:
    config = tracking.online_reachability
    return OnlineReachabilityConfig(
        enabled=config.enabled,
        auto_advance_enabled=config.auto_advance_enabled,
        score_threshold=config.score_threshold,
        window_steps=config.window_steps,
        min_steps_before_auto_advance=config.min_steps_before_auto_advance,
        low_score_patience_steps=config.low_score_patience_steps,
        good_progress_mps=config.good_progress_mps,
        good_tendon_speed_ratio=config.good_tendon_speed_ratio,
        good_alignment=config.good_alignment,
        bad_model_residual_mps=config.bad_model_residual_mps,
    )


def tracking_target_speed_limit(
    tracking: ScenarioTrackingControlConfig,
) -> float | None:
    if not tracking.enforce_target_speed_limit:
        return None
    return tracking.max_target_speed_mps
