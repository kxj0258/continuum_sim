"""Scenario composition root and primary simulation application API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
from continuum_sim.tasks.navigation_mission import resolve_navigation_waypoints
from continuum_sim.tasks.task_plan import (
    BaseApproachConstraint,
    ClearanceConstraint,
    TaskPlan,
)
from continuum_sim.tasks.trajectory_generation import (
    generate_trajectory_waypoints,
    prepend_tracking_approach,
)
from continuum_sim.tasks.wiping_path import build_wiping_plan


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
        task_plan = _resolve_task_plan(config, assembly, engine_scene, structured_scene)
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


def _resolve_task_plan(config, assembly, engine_scene, structured_scene) -> TaskPlan:
    task = config.task
    normals = np.zeros((0, 3), dtype=float)
    standoff_distance = np.zeros(0, dtype=float)
    if task.type in ("idle", "engine_navigation"):
        waypoints = task.waypoints_world
        phases: tuple[str, ...] = ()
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.trajectory is not None:
        waypoints = generate_trajectory_waypoints(task.trajectory, assembly)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.mission is not None:
        if structured_scene is None:
            raise ValueError("scenario.task.mission requires scenario.scene.structured_config_path.")
        waypoints = resolve_navigation_waypoints(task.mission, structured_scene)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.wiping_path is not None:
        if structured_scene is None:
            raise ValueError("scenario.task.wiping_path requires scenario.scene.structured_config_path.")
        plan = build_wiping_plan(task.wiping_path, structured_scene)
        waypoints = plan.waypoints_world
        phases = plan.phases
        target_force = task.target_force_n
        normal = plan.surface_normal_world
        surface_point = plan.surface_point_world
    else:
        waypoints = task.waypoints_world
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    use_arm_approach = task.type in ("tracking", "navigation") and not (
        task.type == "navigation" and task.tracking_control.stage_mobile_base
    )
    if use_arm_approach:
        tracking_plan = prepend_tracking_approach(
            waypoints,
            assembly,
            samples=task.tracking_control.approach_samples,
        )
        waypoints = tracking_plan.waypoints_world
        approach_mask = tracking_plan.approach_mask
        source_waypoint_index = tracking_plan.source_waypoint_index
    else:
        approach_mask = np.zeros(waypoints.shape[0], dtype=bool)
        source_waypoint_index = np.arange(waypoints.shape[0], dtype=int)
    if normals.shape[0] != waypoints.shape[0]:
        normals = np.tile(normal, (waypoints.shape[0], 1))
    if standoff_distance.shape != (waypoints.shape[0],):
        standoff_distance = np.zeros(waypoints.shape[0], dtype=float)
    if phases and len(phases) != waypoints.shape[0]:
        raise ValueError("scenario.task.waypoint_phases must match waypoint count.")
    if target_force.size == 0:
        target_force = np.zeros(waypoints.shape[0], dtype=float)
        if task.target_normal_force_n > 0.0:
            for index, phase in enumerate(phases):
                if phase == "contact":
                    target_force[index] = task.target_normal_force_n
    elif target_force.shape != (waypoints.shape[0],):
        raise ValueError("scenario.task.target_force_n must match waypoint count.")
    waypoint_orientations = _resolve_waypoint_orientations(
        task,
        waypoints,
        structured_scene,
    )
    return TaskPlan(
        waypoints_world=waypoints,
        waypoint_orientations_world_wxyz=waypoint_orientations,
        waypoint_phases=phases,
        target_force_n=target_force,
        surface_normal_world=normal,
        surface_point_world=surface_point,
        normals_world=normals,
        standoff_distance_m=standoff_distance,
        approach_mask=approach_mask,
        source_waypoint_index=source_waypoint_index,
        clearance=ClearanceConstraint(
            minimum_clearance_m=task.min_clearance_m,
            terminate_on_violation=task.terminate_on_clearance_violation,
        ),
        base_approach=BaseApproachConstraint(
            standoff_m=task.tracking_control.base_approach_standoff_m,
            z_bias=task.tracking_control.base_approach_z_bias,
        ),
    )


def _resolve_waypoint_orientations(task, waypoints, structured_scene) -> np.ndarray:
    if not task.pose_servo_enabled or task.waypoint_orientation_source == "none":
        return np.zeros((0, 4), dtype=float)
    explicit = task.waypoint_orientations_world_wxyz
    if explicit.shape[0] > 0:
        if explicit.shape[0] != waypoints.shape[0]:
            raise ValueError(
                "scenario.task.waypoint_orientations_world_wxyz must match "
                "resolved waypoint count."
            )
        return explicit.copy()
    directions = task.waypoint_directions_world
    if directions.shape[0] > 0:
        if directions.shape[0] != waypoints.shape[0]:
            raise ValueError(
                "scenario.task.pose_servo.waypoint_directions_world must match "
                "resolved waypoint count."
            )
        return np.asarray(
            [
                look_rotation_quaternion_wxyz(
                    direction,
                    task.orientation_roll_reference_world,
                )
                for direction in directions
            ],
            dtype=float,
        )
    if task.waypoint_orientation_source == "explicit_directions":
        raise ValueError(
            "scenario.task.pose_servo.orientation_source='explicit_directions' "
            "requires waypoint_directions_world."
        )
    if task.waypoint_orientation_source == "insertion_direction":
        raise ValueError(
            "scenario.task.pose_servo.orientation_source='insertion_direction' "
            "is only supported by engine_navigation tasks."
        )
    if task.waypoint_orientation_source != "nearest_clearance":
        raise ValueError(
            "Unsupported waypoint orientation source "
            f"{task.waypoint_orientation_source!r}."
        )
    if structured_scene is None:
        raise ValueError(
            "scenario.task.pose_servo.orientation_source='nearest_clearance' "
            "requires scenario.scene.structured_config_path."
        )
    query = StructuredSceneQuery(structured_scene)
    orientations = []
    for waypoint in waypoints:
        clearance = query.nearest_distance(waypoint)
        direction = -np.asarray(clearance.normal, dtype=float)
        if np.linalg.norm(direction) <= 1.0e-12:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
        orientations.append(
            look_rotation_quaternion_wxyz(
                direction,
                task.orientation_roll_reference_world,
            )
        )
    return np.asarray(orientations, dtype=float)


