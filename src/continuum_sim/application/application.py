"""Scenario composition root and primary simulation application API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from continuum_sim.application.scenario import (
    ScenarioConfig,
    load_scenario_config,
)
from continuum_sim.application.control_config_factory import (
    build_wiping_force_strategy,
    engine_navigation_with_unified_observer_control,
    online_reachability_config,
    tracking_coordinated_config,
    tracking_solver_config,
    tracking_target_speed_limit,
)
from continuum_sim.application.backend_factory import build_mujoco_backend
from continuum_sim.application.hook_factory import build_runtime_hooks
from continuum_sim.application.task_plan_factory import resolve_task_plan
from continuum_sim.control.scenario_controllers import (
    NavigationController,
    TimedTrajectoryTrackingController,
    WaypointTrackingController,
    WipingController,
    ZeroSystemController,
)
from continuum_sim.control.staged_engine_navigation import (
    StagedEngineNavigationController,
)
from continuum_sim.control.staged_navigation import StagedNavigationController
from continuum_sim.model.base_pose import look_rotation_quaternion_wxyz
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.io.scenario_artifacts import save_scenario_artifacts
from continuum_sim.runtime.simulation_loop import (
    SimulationLoop,
    SimulationLoopConfig,
    SimulationLoopResult,
)
from continuum_sim.scenes.engine_query import EnginePrimitiveSceneQuery
from continuum_sim.scenes.engine_scene import load_engine_scene_config
from continuum_sim.scenes.scene_config import load_navigation_scene_config
from continuum_sim.scenes.structured_query import StructuredSceneQuery
from continuum_sim.tasks.engine_navigation import (
    resolve_engine_navigation_plan,
)


@dataclass
class SimulationApplication:
    """Fully composed backend, controller, and hook lifecycle."""

    config: ScenarioConfig
    loop: SimulationLoop
    hooks_by_name: dict[str, object]
    last_artifacts: object | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationApplication":
        return cls.from_config(load_scenario_config(path))

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> "SimulationApplication":
        assembly = load_robot_assembly_config(config.assembly_config_path)
        engine_scene = (
            None
            if config.scene.engine_config_path is None
            else load_engine_scene_config(config.scene.engine_config_path)
        )
        structured_scene = (
            None
            if config.scene.structured_config_path is None
            else load_navigation_scene_config(config.scene.structured_config_path)
        )
        if engine_scene is not None and structured_scene is not None:
            raise ValueError("A scenario cannot select engine and structured scenes together.")
        backend = build_mujoco_backend(
            config,
            assembly,
            engine_scene,
            structured_scene,
        )
        if engine_scene is not None:
            scene_query = EnginePrimitiveSceneQuery(engine_scene)
        elif structured_scene is not None:
            scene_query = StructuredSceneQuery(structured_scene)
        else:
            scene_query = None
        task_plan = resolve_task_plan(config, assembly, engine_scene, structured_scene)
        observer_camera_target_world = (
            None
            if config.task.observer_roi_world is None
            else config.task.observer_roi_world.copy()
        )
        if config.task.type == "idle":
            controller = ZeroSystemController(assembly)
        elif config.task.type == "engine_navigation":
            if engine_scene is None or config.task.engine_navigation is None:
                raise ValueError(
                    "engine_navigation requires an engine scene and task specification."
                )
            engine_navigation_spec = engine_navigation_with_unified_observer_control(
                config.task.engine_navigation,
                config.task.observer_control,
                config.task.tracking_control,
            )
            plan = resolve_engine_navigation_plan(
                engine_navigation_spec,
                engine_scene,
                assembly,
            )
            observer_camera_target_world = plan.observer_roi_world.copy()
            tracking = config.task.tracking_control
            executor_orientation_world_wxyz = (
                look_rotation_quaternion_wxyz(
                    plan.insertion_direction_world,
                    config.task.orientation_roll_reference_world,
                )
                if (
                    config.task.pose_servo_enabled
                    and config.task.waypoint_orientation_source
                    == "insertion_direction"
                )
                else None
            )
            controller = StagedEngineNavigationController(
                assembly,
                plan,
                engine_navigation_spec,
                scene_query=scene_query,
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                min_clearance_m=config.task.min_clearance_m,
                terminate_on_clearance_violation=(
                    config.task.terminate_on_clearance_violation
                ),
                observer_control_mode=config.task.observer_control_mode,
                controller_dt_s=config.runtime.controller_dt_s,
                executor_orientation_world_wxyz=(
                    executor_orientation_world_wxyz
                ),
                orientation_tolerance_rad=(
                    config.task.orientation_tolerance_rad
                ),
                low_level_coordinated_config=(
                    tracking_coordinated_config(
                        tracking,
                        config.task.observer_control,
                        config.task.scene_avoidance,
                    )
                ),
                low_level_solver_config=tracking_solver_config(tracking),
                online_reachability=online_reachability_config(tracking),
            )
        elif config.task.type == "navigation":
            tracking = config.task.tracking_control
            navigation_kwargs = {
                "waypoint_tolerance_m": config.task.waypoint_tolerance_m,
                "observer_roi_world": config.task.observer_roi_world,
                "observer_control_mode": config.task.observer_control_mode,
                "scene_query": scene_query,
                "min_clearance_m": config.task.min_clearance_m,
                "terminate_on_clearance_violation": (
                    config.task.terminate_on_clearance_violation
                ),
                "target_advance_mode": config.task.target_advance_mode,
                "controller_dt_s": config.runtime.controller_dt_s,
                "advance_time_s": config.task.advance_time_s,
                "advance_steps": config.task.advance_steps,
                "max_steps_per_waypoint": tracking.max_steps_per_waypoint,
                "executor_position_gain": tracking.executor_position_gain,
                "observer_position_gain": tracking.observer_position_gain,
                "feedforward_speed_mps": tracking.feedforward_speed_mps,
                "max_target_speed_mps": tracking_target_speed_limit(tracking),
                "waypoint_orientations_world_wxyz": (
                    task_plan.waypoint_orientations_world_wxyz
                ),
                "orientation_tolerance_rad": (
                    config.task.orientation_tolerance_rad
                ),
                "solver_config": tracking_solver_config(tracking),
                "enforce_backend_tendon_limits": (
                    tracking.enforce_backend_tendon_limits
                ),
                "coordinated_config": tracking_coordinated_config(
                    tracking,
                    config.task.observer_control,
                    config.task.scene_avoidance,
                ),
                "control_type": config.task.navigation_control_type,
                "cbf_gain": config.task.navigation_cbf_gain,
                "cbf_influence_distance_m": (
                    config.task.navigation_cbf_influence_distance_m
                ),
                "online_reachability": online_reachability_config(tracking),
            }
            if tracking.stage_mobile_base:
                controller = StagedNavigationController(
                    assembly,
                    task_plan.waypoints_world,
                    **navigation_kwargs,
                    base_position_gain=tracking.base_position_gain,
                    base_orientation_gain=tracking.base_orientation_gain,
                    waypoint_directions_world=(
                        config.task.waypoint_directions_world
                    ),
                    base_position_tolerance_m=tracking.base_position_tolerance_m,
                    base_orientation_tolerance_rad=(
                        tracking.base_orientation_tolerance_rad
                    ),
                    base_approach_standoff_m=(
                        tracking.base_approach_standoff_m
                    ),
                    base_approach_z_bias=tracking.base_approach_z_bias,
                    intermediate_waypoints_per_waypoint=(
                        tracking.intermediate_waypoints_per_waypoint
                    ),
                )
            else:
                controller = NavigationController(
                    assembly,
                    task_plan.waypoints_world,
                    **navigation_kwargs,
                )
        elif config.task.type == "wiping":
            tracking = config.task.tracking_control
            controller = WipingController(
                assembly,
                task_plan.waypoints_world,
                waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                scene_query=scene_query,
                surface_normal_world=task_plan.surface_normal_world,
                surface_point_world=task_plan.surface_point_world,
                target_contact_distance_m=config.task.target_contact_distance_m,
                contact_tolerance_m=config.task.contact_tolerance_m,
                target_advance_mode=config.task.target_advance_mode,
                controller_dt_s=config.runtime.controller_dt_s,
                advance_time_s=config.task.advance_time_s,
                advance_steps=config.task.advance_steps,
                phases=task_plan.waypoint_phases,
                target_force_n=task_plan.target_force_n,
                control_type=config.task.wiping_control_type,
                normal_force_gain=config.task.normal_force_gain,
                force_proxy_stiffness_n_m=config.task.force_proxy_stiffness_n_m,
                force_feedback_mode=config.task.force_feedback_mode,
                max_normal_velocity_m_s=config.task.max_normal_velocity_m_s,
                force_control_weight=config.task.force_control_weight,
                max_contact_force_n=config.task.max_contact_force_n,
                force_strategy=build_wiping_force_strategy(config, assembly),
                tracking_mode=tracking.tracking_mode,
                trajectory_duration_s=tracking.trajectory_duration_s,
                approach_samples=tracking.approach_samples,
                observer_control_mode=config.task.observer_control_mode,
                executor_position_gain=tracking.executor_position_gain,
                observer_position_gain=tracking.observer_position_gain,
                feedforward_speed_mps=tracking.feedforward_speed_mps,
                max_target_speed_mps=tracking_target_speed_limit(tracking),
                solver_config=tracking_solver_config(tracking),
                enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
                coordinated_config=tracking_coordinated_config(
                    tracking,
                    config.task.observer_control,
                    config.task.scene_avoidance,
                ),
                online_reachability=online_reachability_config(tracking),
            )
        else:
            tracking = config.task.tracking_control
            if tracking.stage_mobile_base:
                raise ValueError(
                    "tracking_control.stage_mobile_base is only supported by "
                    "navigation tasks in the cleaned mainline."
                )
            if tracking.tracking_mode == "time":
                controller = TimedTrajectoryTrackingController(
                    assembly,
                    task_plan.waypoints_world,
                    trajectory_duration_s=float(tracking.trajectory_duration_s),
                    waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                    observer_roi_world=config.task.observer_roi_world,
                    observer_control_mode=config.task.observer_control_mode,
                    loop=config.task.loop,
                    scene_query=scene_query,
                    approach_mask=task_plan.approach_mask,
                    source_waypoint_index=task_plan.source_waypoint_index,
                    executor_position_gain=tracking.executor_position_gain,
                    observer_position_gain=tracking.observer_position_gain,
                    max_target_speed_mps=tracking_target_speed_limit(tracking),
                    solver_config=tracking_solver_config(tracking),
                    enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
                    coordinated_config=tracking_coordinated_config(
                        tracking,
                        config.task.observer_control,
                        config.task.scene_avoidance,
                    ),
                )
            else:
                controller = WaypointTrackingController(
                    assembly,
                    task_plan.waypoints_world,
                    waypoint_tolerance_m=config.task.waypoint_tolerance_m,
                    waypoint_orientations_world_wxyz=(
                        task_plan.waypoint_orientations_world_wxyz
                    ),
                    orientation_tolerance_rad=(
                        config.task.orientation_tolerance_rad
                    ),
                    observer_roi_world=config.task.observer_roi_world,
                    observer_control_mode=config.task.observer_control_mode,
                    loop=config.task.loop,
                    target_advance_mode=config.task.target_advance_mode,
                    controller_dt_s=config.runtime.controller_dt_s,
                    advance_time_s=config.task.advance_time_s,
                    advance_steps=config.task.advance_steps,
                    max_steps_per_waypoint=tracking.max_steps_per_waypoint,
                    scene_query=scene_query,
                    approach_mask=task_plan.approach_mask,
                    source_waypoint_index=task_plan.source_waypoint_index,
                    executor_position_gain=tracking.executor_position_gain,
                    observer_position_gain=tracking.observer_position_gain,
                    feedforward_speed_mps=tracking.feedforward_speed_mps,
                    max_target_speed_mps=tracking_target_speed_limit(tracking),
                    solver_config=tracking_solver_config(tracking),
                    enforce_backend_tendon_limits=tracking.enforce_backend_tendon_limits,
                    coordinated_config=tracking_coordinated_config(
                        tracking,
                        config.task.observer_control,
                        config.task.scene_avoidance,
                    ),
                    online_reachability=online_reachability_config(tracking),
                )
        hooks_by_name = build_runtime_hooks(
            config=config,
            backend=backend,
            controller=controller,
            assembly=assembly,
            observer_camera_target_world=observer_camera_target_world,
        )
        loop = SimulationLoop(
            backend,
            controller,
            SimulationLoopConfig(
                controller_dt_s=config.runtime.controller_dt_s,
                n_substeps=config.runtime.n_substeps,
                max_steps=config.runtime.max_steps,
            ),
            hooks=tuple(hooks_by_name.values()),
        )
        return cls(config=config, loop=loop, hooks_by_name=hooks_by_name)

    def run(self) -> SimulationLoopResult:
        result = self.loop.run()
        self.last_artifacts = save_scenario_artifacts(self, result)
        return result


