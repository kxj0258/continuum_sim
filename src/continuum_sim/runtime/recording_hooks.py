"""Recording-oriented runtime hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_sim.runtime.hook_utils import (
    executor_arm as _executor_arm,
    metadata_vector_or_nan as _metadata_vector_or_nan,
)
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass
class StateRecorderHook:
    """Record compact named state samples independently from backend details."""

    time_s: list[float] = field(default_factory=list)
    base_position_m: list[np.ndarray] = field(default_factory=list)
    arm_tip_position_m: dict[str, list[np.ndarray]] = field(default_factory=dict)
    target_position_m: list[np.ndarray] = field(default_factory=list)
    target_actual_position_m: list[np.ndarray] = field(default_factory=list)
    target_engine_local_path_name: list[str] = field(default_factory=list)
    target_engine_local_path_type: list[str] = field(default_factory=list)
    target_engine_executor_subphase: list[str] = field(default_factory=list)
    target_engine_local_path_center_m: list[np.ndarray] = field(default_factory=list)
    target_engine_insertion_direction_world: list[np.ndarray] = field(
        default_factory=list
    )
    waypoint_index: list[int] = field(default_factory=list)
    tracking_error_m: list[float] = field(default_factory=list)
    achieved_waypoint_error_m: list[float] = field(default_factory=list)
    waypoint_advanced: list[bool] = field(default_factory=list)
    waypoint_advance_reason: list[str] = field(default_factory=list)
    tracking_complete: list[bool] = field(default_factory=list)
    tracking_approach: list[bool] = field(default_factory=list)
    online_reachability_score: list[float] = field(default_factory=list)
    online_reachability_execution_score: list[float] = field(default_factory=list)
    online_reachability_combined_score: list[float] = field(default_factory=list)
    online_reachability_progress_component: list[float] = field(default_factory=list)
    online_reachability_alignment_component: list[float] = field(default_factory=list)
    online_reachability_tendon_component: list[float] = field(default_factory=list)
    online_reachability_model_component: list[float] = field(default_factory=list)
    online_reachability_progress_rate_mps: list[float] = field(default_factory=list)
    online_reachability_target_alignment: list[float] = field(default_factory=list)
    online_reachability_tendon_speed_ratio: list[float] = field(default_factory=list)
    online_reachability_model_residual_mps: list[float] = field(default_factory=list)
    online_reachability_low_score_steps: list[int] = field(default_factory=list)
    online_reachability_auto_advance_requested: list[bool] = field(
        default_factory=list
    )
    arm_saturation_scale: dict[str, list[float]] = field(default_factory=dict)
    arm_tendon_target_error_norm_m: dict[str, list[float]] = field(default_factory=dict)
    arm_tendon_target_error_max_m: dict[str, list[float]] = field(default_factory=dict)
    arm_peak_actuator_force_n: dict[str, list[float]] = field(default_factory=dict)
    min_clearance_m: list[float] = field(default_factory=list)
    executor_clearance_m: list[float] = field(default_factory=list)
    observer_clearance_m: list[float] = field(default_factory=list)
    executor_scene_collision_active: list[bool] = field(default_factory=list)
    observer_scene_collision_active: list[bool] = field(default_factory=list)
    contact_distance_m: list[float] = field(default_factory=list)
    target_force_n: list[float] = field(default_factory=list)
    estimated_force_n: list[float] = field(default_factory=list)
    force_error_n: list[float] = field(default_factory=list)
    contact_error_m: list[float] = field(default_factory=list)
    measured_force_n: list[float] = field(default_factory=list)
    normal_force_source: list[str] = field(default_factory=list)
    admittance_position_m: list[float] = field(default_factory=list)
    admittance_velocity_m_s: list[float] = field(default_factory=list)
    dynamic_normal_correction_m: list[float] = field(default_factory=list)
    wiping_dynamic_active: list[bool] = field(default_factory=list)
    task_phase: list[str] = field(default_factory=list)
    engine_navigation_phase: list[str] = field(default_factory=list)
    engine_navigation_terminal_reason: list[str] = field(default_factory=list)
    engine_navigation_progress: list[float] = field(default_factory=list)
    base_target_position_m: list[np.ndarray] = field(default_factory=list)
    base_position_error_m: list[float] = field(default_factory=list)
    base_orientation_error_rad: list[float] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        self.time_s.clear()
        self.base_position_m.clear()
        self.arm_tip_position_m = {name: [] for name in state.arms}
        self.target_position_m.clear()
        self.target_actual_position_m.clear()
        self.target_engine_local_path_name.clear()
        self.target_engine_local_path_type.clear()
        self.target_engine_executor_subphase.clear()
        self.target_engine_local_path_center_m.clear()
        self.target_engine_insertion_direction_world.clear()
        self.waypoint_index.clear()
        self.tracking_error_m.clear()
        self.achieved_waypoint_error_m.clear()
        self.waypoint_advanced.clear()
        self.waypoint_advance_reason.clear()
        self.tracking_complete.clear()
        self.tracking_approach.clear()
        self.online_reachability_score.clear()
        self.online_reachability_execution_score.clear()
        self.online_reachability_combined_score.clear()
        self.online_reachability_progress_component.clear()
        self.online_reachability_alignment_component.clear()
        self.online_reachability_tendon_component.clear()
        self.online_reachability_model_component.clear()
        self.online_reachability_progress_rate_mps.clear()
        self.online_reachability_target_alignment.clear()
        self.online_reachability_tendon_speed_ratio.clear()
        self.online_reachability_model_residual_mps.clear()
        self.online_reachability_low_score_steps.clear()
        self.online_reachability_auto_advance_requested.clear()
        self.arm_saturation_scale = {name: [] for name in state.arms}
        self.arm_tendon_target_error_norm_m = {name: [] for name in state.arms}
        self.arm_tendon_target_error_max_m = {name: [] for name in state.arms}
        self.arm_peak_actuator_force_n = {name: [] for name in state.arms}
        self.min_clearance_m.clear()
        self.executor_clearance_m.clear()
        self.observer_clearance_m.clear()
        self.executor_scene_collision_active.clear()
        self.observer_scene_collision_active.clear()
        self.contact_distance_m.clear()
        self.target_force_n.clear()
        self.estimated_force_n.clear()
        self.force_error_n.clear()
        self.contact_error_m.clear()
        self.measured_force_n.clear()
        self.normal_force_source.clear()
        self.admittance_position_m.clear()
        self.admittance_velocity_m_s.clear()
        self.dynamic_normal_correction_m.clear()
        self.wiping_dynamic_active.clear()
        self.task_phase.clear()
        self.engine_navigation_phase.clear()
        self.engine_navigation_terminal_reason.clear()
        self.engine_navigation_progress.clear()
        self.base_target_position_m.clear()
        self.base_position_error_m.clear()
        self.base_orientation_error_rad.clear()
        self._append(state)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        self._append(state)
        saturation = state.metadata.get("saturation", {})
        for name, arm in state.arms.items():
            arm_saturation = saturation.get(name, {})
            self.arm_saturation_scale[name].append(
                float(arm_saturation.get("common_scale", np.nan))
            )
            target_error = arm.tendon_target_m - arm.tendon_displacement_m
            self.arm_tendon_target_error_norm_m[name].append(
                float(np.linalg.norm(target_error))
            )
            self.arm_tendon_target_error_max_m[name].append(
                float(np.max(np.abs(target_error)))
            )
            self.arm_peak_actuator_force_n[name].append(
                float(np.max(np.abs(arm.actuator_force_n)))
            )
        if command.metadata.get("task_type") == "engine_navigation":
            self.engine_navigation_phase.append(
                str(command.metadata.get("engine_navigation_phase", ""))
            )
            self.engine_navigation_terminal_reason.append(
                str(command.metadata.get("engine_navigation_terminal_reason", ""))
            )
            self.engine_navigation_progress.append(
                float(command.metadata.get("engine_navigation_progress", np.nan))
            )
            self.base_target_position_m.append(
                np.asarray(
                    command.metadata.get(
                        "base_target_position_m",
                        np.full(3, np.nan, dtype=float),
                    ),
                    dtype=float,
                ).copy()
            )
            self.base_position_error_m.append(
                float(command.metadata.get("base_position_error_m", np.nan))
            )
            self.base_orientation_error_rad.append(
                float(command.metadata.get("base_orientation_error_rad", np.nan))
            )
        target = command.metadata.get("executor_target_world")
        if target is not None:
            self.target_position_m.append(np.asarray(target, dtype=float).copy())
            actual = command.metadata.get("executor_actual_world")
            self.target_actual_position_m.append(
                np.full(3, np.nan, dtype=float)
                if actual is None
                else np.asarray(actual, dtype=float).copy()
            )
            self.target_engine_local_path_name.append(
                str(command.metadata.get("engine_navigation_local_path_name", ""))
            )
            self.target_engine_local_path_type.append(
                str(command.metadata.get("engine_navigation_local_path_type", ""))
            )
            self.target_engine_executor_subphase.append(
                str(command.metadata.get("engine_navigation_executor_subphase", ""))
            )
            self.target_engine_local_path_center_m.append(
                _metadata_vector_or_nan(
                    command.metadata,
                    "engine_navigation_observer_roi_m",
                )
            )
            self.target_engine_insertion_direction_world.append(
                _metadata_vector_or_nan(
                    command.metadata,
                    "engine_navigation_insertion_direction_world",
                )
            )
            self.waypoint_index.append(int(command.metadata.get("waypoint_index", 0)))
            self.tracking_error_m.append(
                float(command.metadata.get("executor_error_m", np.nan))
            )
            self.achieved_waypoint_error_m.append(
                float(command.metadata.get("achieved_waypoint_error_m", np.nan))
            )
            self.waypoint_advanced.append(
                bool(command.metadata.get("waypoint_advanced", False))
            )
            self.waypoint_advance_reason.append(
                str(command.metadata.get("waypoint_advance_reason", ""))
            )
            self.tracking_complete.append(
                bool(command.metadata.get("tracking_complete", False))
            )
            self.tracking_approach.append(
                bool(command.metadata.get("tracking_approach", False))
            )
            self.online_reachability_score.append(
                float(command.metadata.get("online_reachability_score", np.nan))
            )
            self.online_reachability_execution_score.append(
                float(
                    command.metadata.get(
                        "online_reachability_execution_score",
                        np.nan,
                    )
                )
            )
            self.online_reachability_combined_score.append(
                float(
                    command.metadata.get(
                        "online_reachability_combined_score",
                        np.nan,
                    )
                )
            )
            self.online_reachability_progress_component.append(
                float(
                    command.metadata.get(
                        "online_reachability_progress_component",
                        np.nan,
                    )
                )
            )
            self.online_reachability_alignment_component.append(
                float(
                    command.metadata.get(
                        "online_reachability_alignment_component",
                        np.nan,
                    )
                )
            )
            self.online_reachability_tendon_component.append(
                float(
                    command.metadata.get(
                        "online_reachability_tendon_component",
                        np.nan,
                    )
                )
            )
            self.online_reachability_model_component.append(
                float(
                    command.metadata.get(
                        "online_reachability_model_component",
                        np.nan,
                    )
                )
            )
            self.online_reachability_progress_rate_mps.append(
                float(
                    command.metadata.get(
                        "online_reachability_progress_rate_mps",
                        np.nan,
                    )
                )
            )
            self.online_reachability_target_alignment.append(
                float(
                    command.metadata.get(
                        "online_reachability_target_alignment",
                        np.nan,
                    )
                )
            )
            self.online_reachability_tendon_speed_ratio.append(
                float(
                    command.metadata.get(
                        "online_reachability_tendon_speed_ratio",
                        np.nan,
                    )
                )
            )
            self.online_reachability_model_residual_mps.append(
                float(
                    command.metadata.get(
                        "online_reachability_model_residual_mps",
                        np.nan,
                    )
                )
            )
            self.online_reachability_low_score_steps.append(
                int(command.metadata.get("online_reachability_low_score_steps", 0))
            )
            self.online_reachability_auto_advance_requested.append(
                bool(
                    command.metadata.get(
                        "online_reachability_auto_advance_requested",
                        False,
                    )
                )
            )
            self.min_clearance_m.append(
                float(command.metadata.get("min_clearance_m", np.nan))
            )
            self.executor_clearance_m.append(
                float(command.metadata.get("executor_clearance_m", np.nan))
            )
            self.observer_clearance_m.append(
                float(command.metadata.get("observer_clearance_m", np.nan))
            )
            self.executor_scene_collision_active.append(
                bool(command.metadata.get("executor_scene_collision_active", False))
            )
            self.observer_scene_collision_active.append(
                bool(command.metadata.get("observer_scene_collision_active", False))
            )
            self.contact_distance_m.append(
                float(command.metadata.get("contact_distance_m", np.nan))
            )
            self.target_force_n.append(
                float(command.metadata.get("target_normal_force_n", np.nan))
            )
            self.estimated_force_n.append(
                float(command.metadata.get("estimated_normal_force_n", np.nan))
            )
            self.force_error_n.append(
                float(command.metadata.get("force_error_n", np.nan))
            )
            self.contact_error_m.append(
                float(command.metadata.get("contact_error_m", np.nan))
            )
            self.measured_force_n.append(
                float(command.metadata.get("measured_normal_force_n", np.nan))
            )
            self.normal_force_source.append(
                str(command.metadata.get("normal_force_source", ""))
            )
            self.admittance_position_m.append(
                float(command.metadata.get("admittance_position_m", np.nan))
            )
            self.admittance_velocity_m_s.append(
                float(command.metadata.get("admittance_velocity_m_s", np.nan))
            )
            self.dynamic_normal_correction_m.append(
                float(command.metadata.get("dynamic_normal_correction_m", np.nan))
            )
            self.wiping_dynamic_active.append(
                bool(command.metadata.get("wiping_dynamic_system_controller_active", False))
            )
            self.task_phase.append(str(command.metadata.get("wiping_phase", "")))

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _append(self, state: RobotSystemState) -> None:
        self.time_s.append(float(state.time_s))
        self.base_position_m.append(state.base.pose.position.copy())
        for name, arm in state.arms.items():
            self.arm_tip_position_m.setdefault(name, []).append(
                arm.tip_pose_world.position.copy()
            )


@dataclass
class MujocoReplayRecorderHook:
    """Record generalized state needed for deterministic offscreen replay."""

    backend: object
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    mocap_pos: list[np.ndarray] = field(default_factory=list)
    mocap_quat: list[np.ndarray] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self.qpos.clear()
        self.qvel.clear()
        self.mocap_pos.clear()
        self.mocap_quat.clear()
        self._append()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state, command, step_index
        self._append()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _append(self) -> None:
        data = self.backend.physics.data
        self.qpos.append(np.asarray(data.qpos, dtype=float).copy())
        self.qvel.append(np.asarray(data.qvel, dtype=float).copy())
        self.mocap_pos.append(np.asarray(data.mocap_pos, dtype=float).copy())
        self.mocap_quat.append(np.asarray(data.mocap_quat, dtype=float).copy())

__all__ = [
    "MujocoReplayRecorderHook",
    "StateRecorderHook",
]
