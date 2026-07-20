"""Task-plan resolution for scenario applications."""

from __future__ import annotations

import numpy as np

from continuum_sim.model.base_pose import look_rotation_quaternion_wxyz
from continuum_sim.scenes.structured_query import StructuredSceneQuery
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


def resolve_task_plan(config, assembly, engine_scene, structured_scene) -> TaskPlan:
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
            raise ValueError(
                "scenario.task.mission requires scenario.scene.structured_config_path."
            )
        waypoints = resolve_navigation_waypoints(task.mission, structured_scene)
        phases = task.waypoint_phases
        target_force = task.target_force_n
        normal = task.surface_normal_world
        surface_point = None
    elif task.wiping_path is not None:
        if structured_scene is None:
            raise ValueError(
                "scenario.task.wiping_path requires "
                "scenario.scene.structured_config_path."
            )
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
    waypoint_orientations = resolve_waypoint_orientations(
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


def resolve_waypoint_orientations(task, waypoints, structured_scene) -> np.ndarray:
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
