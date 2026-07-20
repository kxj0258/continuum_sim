"""Shared metadata keys emitted by controllers and consumed by artifacts."""

from __future__ import annotations

OBSERVER_SCALAR_METADATA: tuple[tuple[str, type], ...] = (
    ("observer_control_mode", str),
    ("kinematics_mode", str),
    ("observer_collision_active", bool),
    ("observer_tracking_active", bool),
    ("observer_look_at_active", bool),
    ("observer_visual_servo_active", bool),
    ("visual_servo_target_visible", bool),
    ("executor_scene_collision_active", bool),
    ("observer_scene_collision_active", bool),
    ("executor_clearance_m", float),
    ("observer_clearance_m", float),
    ("staged_navigation_subtarget_index", int),
    ("staged_navigation_subtarget_count", int),
    ("staged_navigation_subtarget_advanced", bool),
    ("staged_navigation_subtarget_advance_reason", str),
    ("inter_arm_safety_mode", str),
    ("inter_arm_executor_frozen", bool),
    ("inter_arm_critical_distance", bool),
    ("inter_arm_hard_stop", bool),
    ("inter_arm_closest_observer_index", int),
    ("inter_arm_closest_executor_index", int),
    ("inter_arm_distance_m", float),
    ("inter_arm_min_distance_m", float),
    ("inter_arm_influence_distance_m", float),
    ("inter_arm_hard_stop_distance_m", float),
    ("inter_arm_release_margin_m", float),
    ("observer_avoidance_desired_speed_mps", float),
    ("observer_residual_norm", float),
    ("executor_feedforward_gain", float),
    ("executor_orientation_error_rad", float),
    ("observer_visual_servo_depth_error_m", float),
    ("visual_servo_depth_m", float),
    ("task_space_orientation_error_norm_rad", float),
    ("task_space_angular_speed_limited", bool),
)

OBSERVER_VECTOR3_METADATA: tuple[str, ...] = (
    "executor_target_velocity_world",
    "task_intent_velocity_world",
    "task_intent_angular_velocity_world",
    "executor_scaled_feedforward_velocity_world",
    "executor_scaled_feedforward_angular_velocity_world",
    "observer_target_position_world",
    "observer_target_error_world",
    "observer_target_velocity_world",
    "observer_look_at_error_world",
    "observer_look_at_velocity_world",
    "observer_visual_servo_angular_velocity_world",
    "observer_visual_servo_position_velocity_world",
    "executor_orientation_error_world",
    "executor_target_angular_velocity_world",
    "task_space_orientation_error_world",
    "task_space_raw_angular_velocity_world",
    "task_space_angular_velocity_world",
    "inter_arm_closest_observer_point_world",
    "inter_arm_closest_executor_point_world",
    "visual_servo_roi_world",
    "visual_servo_camera_position_world",
)

OBSERVER_VECTOR2_METADATA: tuple[str, ...] = (
    "observer_visual_servo_pixel_error_px",
    "visual_servo_normalized_error",
)

ORIENTATION_VECTOR4_METADATA: tuple[str, ...] = (
    "task_intent_target_orientation_world_wxyz",
    "executor_target_orientation_world_wxyz",
)
