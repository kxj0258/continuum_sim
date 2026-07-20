"""Optional observers for scenario-driven simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from continuum_sim.runtime.hook_utils import (
    executor_arm as _executor_arm,
    finite_metadata_float as _finite_metadata_float,
    metadata_max_abs as _metadata_max_abs,
    metadata_norm as _metadata_norm,
    metadata_path as _metadata_path,
    metadata_paths as _metadata_paths,
    metadata_point as _metadata_point,
)
from continuum_sim.runtime.metadata_schema import ENGINE_NAVIGATION_OVERLAY_METADATA
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass
class _TrackingOverlayState:
    """Shared, bounded tracking data for live and recorded MuJoCo overlays."""

    tip_trail: list[np.ndarray] = field(default_factory=list)
    target_trail: list[np.ndarray] = field(default_factory=list)
    target_trail_kinds: list[str] = field(default_factory=list)
    base_trail: list[np.ndarray] = field(default_factory=list)
    observer_roi_world: np.ndarray | None = None
    navigation_metadata: dict[str, object] = field(default_factory=dict)

    def clear(self) -> None:
        self.tip_trail.clear()
        self.target_trail.clear()
        self.target_trail_kinds.clear()
        self.base_trail.clear()
        self.observer_roi_world = None
        self.navigation_metadata.clear()

    def capture(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        *,
        max_points: int,
    ) -> None:
        is_engine_navigation = (
            command.metadata.get("task_type") == "engine_navigation"
        )
        target_key = (
            "engine_navigation_active_target_m"
            if is_engine_navigation
            else "executor_target_world"
        )
        target = _metadata_point(command.metadata, target_key)
        target_kind = str(
            command.metadata.get(
                "engine_navigation_active_target_kind",
                "executor",
            )
        )
        if target is not None and (
            not self.target_trail
            or target_kind != self.target_trail_kinds[-1]
            or not np.array_equal(target, self.target_trail[-1])
        ):
            self.target_trail.append(target)
            self.target_trail_kinds.append(target_kind)
        executor = _executor_arm(state)
        if executor is not None:
            self.tip_trail.append(executor.tip_pose_world.position.copy())
        observer_roi = _metadata_point(command.metadata, "visual_servo_roi_world")
        if observer_roi is None:
            observer_roi = _metadata_point(
                command.metadata,
                "engine_navigation_observer_roi_m",
            )
        self.observer_roi_world = None if observer_roi is None else observer_roi.copy()
        if is_engine_navigation:
            self.base_trail.append(state.base.pose.position.copy())
            self.navigation_metadata = {
                key: _copy_overlay_metadata_value(command.metadata[key])
                for key in ENGINE_NAVIGATION_OVERLAY_METADATA
                if key in command.metadata
            }
        else:
            self.navigation_metadata.clear()
            self.base_trail.clear()
        self._trim(max_points)

    def _trim(self, max_points: int) -> None:
        for trail in (self.tip_trail, self.target_trail, self.base_trail):
            if len(trail) > max_points:
                del trail[:-max_points]
        if len(self.target_trail_kinds) > max_points:
            del self.target_trail_kinds[:-max_points]


def _copy_overlay_metadata_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list | tuple):
        return tuple(_copy_overlay_metadata_value(item) for item in value)
    return value




def _update_follow_camera(
    render_camera: object | None,
    camera_config: object | None,
    state: RobotSystemState,
) -> None:
    if render_camera in (None, -1) or camera_config is None:
        return
    target = _follow_camera_target(camera_config, state)
    if target is not None:
        render_camera.lookat[:] = target


def _follow_camera_target(
    camera_config: object,
    state: RobotSystemState,
) -> np.ndarray | None:
    follow = str(getattr(camera_config, "follow", "none"))
    if follow == "base":
        return state.base.pose.position.copy()
    if follow == "executor_tip":
        executor = _executor_arm(state)
        if executor is not None:
            return executor.tip_pose_world.position.copy()
    return None


class LiveDiagnosticsPanelHook:
    """Optional compact live panel for tracking, safety, and actuator diagnostics."""

    def __init__(self, *, stride: int = 5, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveDiagnosticsPanelHook stride must be positive.")
        if history_points <= 0:
            raise ValueError("LiveDiagnosticsPanelHook history_points must be positive.")
        self.stride = int(stride)
        self.history_points = int(history_points)
        self._plt = None
        self._figure = None
        self._axes = None
        self._info_text = None
        self._time: list[float] = []
        self._tracking_error: list[float] = []
        self._tip_target_error: list[float] = []
        self._tip_error_xyz: list[np.ndarray] = []
        self._task_reference_jump: list[float] = []
        self._task_space_error: list[float] = []
        self._task_space_velocity: list[float] = []
        self._task_space_speed_limited: list[float] = []
        self._base_error: list[float] = []
        self._clearance: list[float] = []
        self._inter_arm_distance: list[float] = []
        self._contact_distance: list[float] = []
        self._force_error: list[float] = []
        self._condition: list[float] = []
        self._velocity_scale: list[float] = []
        self._ik_residual: list[float] = []
        self._ik_projection_residual: list[float] = []
        self._saturation_scale: list[float] = []
        self._tendon_error: list[float] = []
        self._observer_tendon_error: list[float] = []
        self._force_utilization: list[float] = []
        self._execution_saturation_active: list[float] = []
        self._reachability_score: list[float] = []
        self._reachability_execution_score: list[float] = []
        self._reachability_combined_score: list[float] = []
        self._reachability_progress_component: list[float] = []
        self._reachability_alignment_component: list[float] = []
        self._reachability_tendon_component: list[float] = []
        self._reachability_model_component: list[float] = []
        self._reachability_progress_rate: list[float] = []
        self._reachability_alignment: list[float] = []
        self._reachability_tendon_ratio: list[float] = []
        self._reachability_model_residual: list[float] = []
        self._reachability_low_score_steps: list[float] = []
        self._reachability_auto_advance_requested: list[float] = []
        self._reachability_threshold: list[float] = []
        self._waypoint_indices: list[int] = []
        self._tracking_approach_flags: list[float] = []
        self._waypoint_advanced_flags: list[float] = []
        self._phase = ""
        self._observer_mode = ""
        self._waypoint_index = -1
        self._reachability_low_score_patience_steps = 0
        self._ik_right_axis = None
        self._backend_right_axis = None
        self._drivers_right_axis = None
        self._last_task_target: np.ndarray | None = None

    def on_reset(self, state: RobotSystemState) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, axes = plt.subplots(3, 2, figsize=(12.0, 9.6))
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim live diagnostics")
        self._axes = axes.reshape(-1)
        self._ik_right_axis = self._axes[2].twinx()
        self._backend_right_axis = self._axes[3].twinx()
        self._drivers_right_axis = self._axes[5].twinx()
        for axis in (
            self._ik_right_axis,
            self._backend_right_axis,
            self._drivers_right_axis,
        ):
            axis.patch.set_alpha(0.0)
        self._info_text = None
        self._clear()
        self._append(state, None)
        plt.ion()
        plt.show(block=False)
        self._draw()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride != 0:
            return
        self._append(state, command)
        self._trim()
        self._draw()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._plt is None:
            return
        try:
            self._plt.ioff()
            if self._figure is not None:
                self._plt.close(self._figure)
        except Exception:
            pass
        self._figure = None
        self._axes = None
        self._ik_right_axis = None
        self._backend_right_axis = None
        self._drivers_right_axis = None

    def _clear(self) -> None:
        for values in (
            self._time,
            self._tracking_error,
            self._tip_target_error,
            self._task_reference_jump,
            self._task_space_error,
            self._task_space_velocity,
            self._task_space_speed_limited,
            self._base_error,
            self._clearance,
            self._inter_arm_distance,
            self._contact_distance,
            self._force_error,
            self._condition,
            self._velocity_scale,
            self._ik_residual,
            self._ik_projection_residual,
            self._saturation_scale,
            self._tendon_error,
            self._observer_tendon_error,
            self._force_utilization,
            self._execution_saturation_active,
            self._reachability_score,
            self._reachability_execution_score,
            self._reachability_combined_score,
            self._reachability_progress_component,
            self._reachability_alignment_component,
            self._reachability_tendon_component,
            self._reachability_model_component,
            self._reachability_progress_rate,
            self._reachability_alignment,
            self._reachability_tendon_ratio,
            self._reachability_model_residual,
            self._reachability_low_score_steps,
            self._reachability_auto_advance_requested,
            self._reachability_threshold,
            self._waypoint_indices,
            self._tracking_approach_flags,
            self._waypoint_advanced_flags,
        ):
            values.clear()
        self._tip_error_xyz.clear()
        self._phase = ""
        self._observer_mode = ""
        self._waypoint_index = -1
        self._reachability_low_score_patience_steps = 0
        self._last_task_target = None

    def _append(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand | None,
    ) -> None:
        metadata = {} if command is None else command.metadata
        self._time.append(float(state.time_s))
        tip_error_vector = _tip_target_error_vector(state, metadata)
        if tip_error_vector is None:
            tip_error_norm = np.nan
            tip_error_vector = np.full(3, np.nan, dtype=float)
        else:
            tip_error_norm = float(np.linalg.norm(tip_error_vector))
        self._tip_target_error.append(tip_error_norm)
        self._tip_error_xyz.append(tip_error_vector)
        self._tracking_error.append(
            tip_error_norm
            if np.isfinite(tip_error_norm)
            else float(metadata.get("executor_error_m", np.nan))
        )
        task_target = _metadata_point(metadata, "task_intent_target_world")
        if task_target is None:
            task_target = _metadata_point(metadata, "executor_target_world")
        if task_target is None or self._last_task_target is None:
            self._task_reference_jump.append(np.nan)
        else:
            self._task_reference_jump.append(
                float(np.linalg.norm(task_target - self._last_task_target))
            )
        if task_target is not None:
            self._last_task_target = task_target.copy()
        task_space_error = metadata.get("task_space_position_error_world")
        self._task_space_error.append(_metadata_norm(task_space_error))
        task_space_velocity = metadata.get(
            "task_space_velocity_world",
            metadata.get("executor_target_velocity_world"),
        )
        self._task_space_velocity.append(_metadata_norm(task_space_velocity))
        self._task_space_speed_limited.append(
            1.0 if bool(metadata.get("task_space_speed_limited", False)) else 0.0
        )
        self._base_error.append(float(metadata.get("base_position_error_m", np.nan)))
        self._clearance.append(float(metadata.get("min_clearance_m", np.nan)))
        self._inter_arm_distance.append(
            float(metadata.get("inter_arm_distance_m", np.nan))
        )
        self._contact_distance.append(float(metadata.get("contact_distance_m", np.nan)))
        self._force_error.append(float(metadata.get("force_error_n", np.nan)))
        singularity = metadata.get("whole_body_singularity")
        self._condition.append(
            float(getattr(singularity, "condition_number", np.nan))
        )
        self._velocity_scale.append(
            float(getattr(singularity, "velocity_scale", np.nan))
        )
        self._ik_residual.append(float(metadata.get("residual_norm", np.nan)))
        solver = metadata.get("whole_body_solver")
        projection_residual = (
            solver.get("target_projection_residual_norm", np.nan)
            if isinstance(solver, dict)
            else np.nan
        )
        self._ik_projection_residual.append(float(projection_residual))
        saturation = state.metadata.get("saturation", {})
        scales = [
            float(values.get("common_scale", np.nan))
            for values in saturation.values()
            if isinstance(values, dict)
        ]
        finite_scales = [value for value in scales if np.isfinite(value)]
        self._saturation_scale.append(
            float(min(finite_scales)) if finite_scales else np.nan
        )
        tendon_errors = []
        for arm in state.arms.values():
            tendon_errors.append(
                float(np.linalg.norm(arm.tendon_target_m - arm.tendon_displacement_m))
            )
        self._tendon_error.append(float(max(tendon_errors)) if tendon_errors else np.nan)
        observer_errors = [
            float(np.linalg.norm(arm.tendon_target_m - arm.tendon_displacement_m))
            for arm in state.arms.values()
            if arm.role == "observer"
        ]
        self._observer_tendon_error.append(
            observer_errors[0] if observer_errors else np.nan
        )
        force_utilization = []
        saturation_active = []
        for values in saturation.values():
            if not isinstance(values, dict):
                continue
            utilization = values.get("actuator_force_utilization")
            if utilization is not None:
                force_utilization.append(_metadata_max_abs(utilization))
            for key in (
                "rate",
                "target_rate",
                "displacement",
                "lead",
                "force_constraint_active",
                "anti_windup_active",
                "actuator_force_at_limit",
            ):
                raw_active = values.get(key)
                if raw_active is not None:
                    saturation_active.append(bool(np.any(raw_active)))
        finite_utilization = [
            value for value in force_utilization if np.isfinite(value)
        ]
        self._force_utilization.append(
            float(max(finite_utilization)) if finite_utilization else np.nan
        )
        self._execution_saturation_active.append(
            1.0 if any(saturation_active) else 0.0
        )
        self._reachability_score.append(
            float(metadata.get("online_reachability_score", np.nan))
        )
        self._reachability_execution_score.append(
            float(metadata.get("online_reachability_execution_score", np.nan))
        )
        self._reachability_combined_score.append(
            float(metadata.get("online_reachability_combined_score", np.nan))
        )
        self._reachability_progress_component.append(
            float(metadata.get("online_reachability_progress_component", np.nan))
        )
        self._reachability_alignment_component.append(
            float(metadata.get("online_reachability_alignment_component", np.nan))
        )
        self._reachability_tendon_component.append(
            float(metadata.get("online_reachability_tendon_component", np.nan))
        )
        self._reachability_model_component.append(
            float(metadata.get("online_reachability_model_component", np.nan))
        )
        self._reachability_progress_rate.append(
            float(metadata.get("online_reachability_progress_rate_mps", np.nan))
        )
        self._reachability_alignment.append(
            float(metadata.get("online_reachability_target_alignment", np.nan))
        )
        self._reachability_tendon_ratio.append(
            float(metadata.get("online_reachability_tendon_speed_ratio", np.nan))
        )
        self._reachability_model_residual.append(
            float(metadata.get("online_reachability_model_residual_mps", np.nan))
        )
        self._reachability_low_score_steps.append(
            float(metadata.get("online_reachability_low_score_steps", np.nan))
        )
        self._reachability_auto_advance_requested.append(
            1.0
            if bool(metadata.get("online_reachability_auto_advance_requested", False))
            else 0.0
        )
        self._reachability_threshold.append(
            float(metadata.get("online_reachability_score_threshold", 0.3))
        )
        self._waypoint_indices.append(int(metadata.get("waypoint_index", -1)))
        self._tracking_approach_flags.append(
            1.0 if bool(metadata.get("tracking_approach", False)) else 0.0
        )
        self._waypoint_advanced_flags.append(
            1.0 if bool(metadata.get("waypoint_advanced", False)) else 0.0
        )
        self._reachability_low_score_patience_steps = int(
            metadata.get(
                "online_reachability_low_score_patience_steps",
                self._reachability_low_score_patience_steps,
            )
        )
        self._phase = str(
            metadata.get(
                "engine_navigation_phase",
                metadata.get("wiping_phase", metadata.get("task_type", "")),
            )
        )
        self._observer_mode = str(metadata.get("observer_control_mode", ""))
        self._waypoint_index = int(metadata.get("waypoint_index", -1))

    def _trim(self) -> None:
        extra = len(self._time) - self.history_points
        if extra <= 0:
            return
        for values in (
            self._time,
            self._tracking_error,
            self._tip_target_error,
            self._task_reference_jump,
            self._task_space_error,
            self._task_space_velocity,
            self._task_space_speed_limited,
            self._base_error,
            self._clearance,
            self._inter_arm_distance,
            self._contact_distance,
            self._force_error,
            self._condition,
            self._velocity_scale,
            self._ik_residual,
            self._ik_projection_residual,
            self._saturation_scale,
            self._tendon_error,
            self._observer_tendon_error,
            self._force_utilization,
            self._execution_saturation_active,
            self._reachability_score,
            self._reachability_execution_score,
            self._reachability_combined_score,
            self._reachability_progress_component,
            self._reachability_alignment_component,
            self._reachability_tendon_component,
            self._reachability_model_component,
            self._reachability_progress_rate,
            self._reachability_alignment,
            self._reachability_tendon_ratio,
            self._reachability_model_residual,
            self._reachability_low_score_steps,
            self._reachability_auto_advance_requested,
            self._reachability_threshold,
            self._waypoint_indices,
            self._tracking_approach_flags,
            self._waypoint_advanced_flags,
        ):
            del values[:extra]
        del self._tip_error_xyz[:extra]

    def _draw(self) -> None:
        if self._axes is None or self._figure is None:
            return
        time_s = np.asarray(self._time, dtype=float)
        axes = self._axes
        for axis in axes:
            axis.cla()
            axis.grid(True, alpha=0.25)
        right_axes = (
            self._ik_right_axis,
            self._backend_right_axis,
            self._drivers_right_axis,
        )
        for axis in right_axes:
            if axis is not None:
                axis.cla()
                axis.grid(False)
                axis.patch.set_alpha(0.0)

        waypoint_indices = np.asarray(self._waypoint_indices, dtype=float)
        approach_flags = np.asarray(self._tracking_approach_flags, dtype=float)
        score = np.asarray(self._reachability_score, dtype=float)
        threshold = _last_finite(self._reachability_threshold, default=0.3)
        for index, axis in enumerate(axes):
            _shade_boolean_regions(
                axis,
                time_s,
                approach_flags > 0.5,
                color="0.88",
                alpha=0.28,
            )
            _shade_boolean_regions(
                axis,
                time_s,
                score < threshold,
                color="tab:red",
                alpha=0.08,
            )
            _draw_waypoint_boundaries(
                axis,
                time_s,
                waypoint_indices,
                annotate=(index == 0),
            )

        axes[0].plot(time_s, 1000.0 * np.asarray(self._tracking_error), label="tip error")
        axes[0].plot(
            time_s,
            1000.0 * np.asarray(self._task_reference_jump),
            label="target jump",
        )
        tip_error_xyz = np.asarray(self._tip_error_xyz, dtype=float)
        if tip_error_xyz.ndim == 2 and tip_error_xyz.shape[1] == 3:
            axes[0].plot(time_s, 1000.0 * tip_error_xyz[:, 0], label="tip err x", alpha=0.45)
            axes[0].plot(time_s, 1000.0 * tip_error_xyz[:, 1], label="tip err y", alpha=0.45)
            axes[0].plot(time_s, 1000.0 * tip_error_xyz[:, 2], label="tip err z", alpha=0.45)
        axes[0].set(title="Layer 1: task reference", xlabel="time [s]", ylabel="error [mm]")
        axes[0].legend(loc="upper left", fontsize=8)

        axes[1].plot(
            time_s,
            1000.0 * np.asarray(self._task_space_error),
            label="servo error",
        )
        axes[1].plot(
            time_s,
            1000.0 * np.asarray(self._task_space_velocity),
            label="TCP velocity",
        )
        axes[1].plot(
            time_s,
            np.asarray(self._task_space_speed_limited),
            label="speed limited",
        )
        axes[1].set(title="Layer 2: task-space servo", xlabel="time [s]")
        axes[1].legend(loc="upper right", fontsize=8)

        condition = _finite_positive(self._condition)
        ik_residual = _finite_positive(self._ik_residual)
        projection_residual = _finite_positive(self._ik_projection_residual)
        if np.any(np.isfinite(condition)):
            axes[2].semilogy(time_s, condition, label="condition")
        else:
            axes[2].plot(time_s, condition, label="condition")
        axes[2].semilogy(time_s, ik_residual, label="residual")
        axes[2].semilogy(
            time_s,
            projection_residual,
            label="projection residual",
        )
        axes[2].set(
            title="Layer 3: IK/tendon command",
            xlabel="time [s]",
            ylabel="condition / residual",
        )
        if self._ik_right_axis is not None:
            self._ik_right_axis.plot(
                time_s,
                np.asarray(self._velocity_scale),
                color="tab:orange",
                label="velocity scale",
            )
            self._ik_right_axis.set(ylabel="scale", ylim=(-0.05, 1.05))
            _combined_legend(axes[2], self._ik_right_axis, loc="upper right")
        else:
            axes[2].legend(loc="upper right", fontsize=8)

        axes[3].plot(
            time_s,
            1000.0 * np.asarray(self._tendon_error),
            label="tendon target error",
        )
        axes[3].set(
            title="Layer 4: backend execution",
            xlabel="time [s]",
            ylabel="tendon error [mm]",
        )
        if self._backend_right_axis is not None:
            self._backend_right_axis.plot(
                time_s,
                np.asarray(self._force_utilization),
                color="tab:orange",
                label="force utilization",
            )
            self._backend_right_axis.plot(
                time_s,
                np.asarray(self._saturation_scale),
                color="tab:green",
                label="limit scale",
            )
            self._backend_right_axis.plot(
                time_s,
                np.asarray(self._execution_saturation_active),
                color="tab:red",
                drawstyle="steps-post",
                label="saturation active",
            )
            self._backend_right_axis.set(ylabel="ratio / active")
            _combined_legend(axes[3], self._backend_right_axis, loc="upper left")
        else:
            axes[3].legend(loc="upper right", fontsize=8)

        progress_component = np.asarray(self._reachability_progress_component)
        alignment_component = np.asarray(self._reachability_alignment_component)
        tendon_component = np.asarray(self._reachability_tendon_component)
        model_component = np.asarray(self._reachability_model_component)
        execution_score = np.asarray(self._reachability_execution_score)
        combined_score = np.asarray(self._reachability_combined_score)
        bottleneck = _reachability_bottleneck(
            progress_component,
            alignment_component,
            model_component,
        )
        axes[4].plot(
            time_s,
            score,
            label="reachability",
            linewidth=2.0,
            color="black",
        )
        axes[4].plot(
            time_s,
            progress_component,
            label="progress",
            linewidth=(2.2 if bottleneck == "progress" else 1.2),
        )
        axes[4].plot(
            time_s,
            alignment_component,
            label="alignment",
            linewidth=(2.2 if bottleneck == "alignment" else 1.2),
        )
        axes[4].plot(
            time_s,
            model_component,
            label="model",
            linewidth=(2.2 if bottleneck == "model" else 1.2),
        )
        axes[4].plot(
            time_s,
            execution_score,
            label="execution",
            color="tab:green",
            linestyle="--",
            linewidth=1.4,
        )
        axes[4].plot(
            time_s,
            combined_score,
            label="combined",
            color="0.5",
            linestyle=":",
            linewidth=1.0,
        )
        axes[4].axhline(threshold, color="tab:red", linestyle="--", linewidth=1.0)
        auto_advance = np.asarray(self._reachability_auto_advance_requested)
        if np.any(auto_advance > 0.5):
            event_times = time_s[auto_advance > 0.5]
            axes[4].scatter(
                event_times,
                np.full(event_times.shape, threshold),
                marker="v",
                color="tab:red",
                s=45,
                label="auto advance",
                zorder=5,
            )
        waypoint_advanced = np.asarray(self._waypoint_advanced_flags)
        regular_advance = (waypoint_advanced > 0.5) & ~(auto_advance > 0.5)
        if np.any(regular_advance):
            event_times = time_s[regular_advance]
            axes[4].scatter(
                event_times,
                np.full(event_times.shape, 1.0),
                marker="|",
                color="0.25",
                s=80,
                label="waypoint advance",
                zorder=5,
            )
        axes[4].set(
            title=f"Reachability score (bottleneck: {bottleneck})",
            xlabel="time [s]",
            ylim=(-0.05, 1.05),
        )
        axes[4].legend(loc="upper right", fontsize=8)

        axes[5].plot(
            time_s,
            1000.0 * np.asarray(self._reachability_progress_rate),
            label="progress [mm/s]",
        )
        axes[5].plot(
            time_s,
            1000.0 * np.asarray(self._reachability_model_residual),
            label="model residual [mm/s]",
            color="tab:red",
        )
        axes[5].set(
            title="Reachability drivers",
            xlabel="time [s]",
            ylabel="mm/s",
        )
        if self._drivers_right_axis is not None:
            self._drivers_right_axis.plot(
                time_s,
                np.asarray(self._reachability_alignment),
                color="tab:orange",
                label="alignment",
            )
            self._drivers_right_axis.plot(
                time_s,
                np.asarray(self._reachability_tendon_ratio),
                color="tab:green",
                label="tendon ratio",
            )
            self._drivers_right_axis.plot(
                time_s,
                auto_advance,
                color="tab:purple",
                drawstyle="steps-post",
                label="auto advance",
            )
            self._drivers_right_axis.set(ylabel="ratio / active", ylim=(-0.1, 1.1))
            _combined_legend(axes[5], self._drivers_right_axis, loc="upper right")
        else:
            axes[5].legend(loc="upper right", fontsize=8)

        status_score = _last_finite(self._reachability_score)
        title, title_style = _diagnostics_status_title(
            time_s=_last_finite(self._time),
            phase=self._phase,
            waypoint_index=self._waypoint_index,
            score=status_score,
            bottleneck=bottleneck,
            bottleneck_value=_bottleneck_value(
                bottleneck,
                progress_component,
                alignment_component,
                model_component,
            ),
            execution_score=_last_finite(self._reachability_execution_score),
            low_score_steps=_last_finite(
                self._reachability_low_score_steps,
                default=0.0,
            ),
            low_score_patience_steps=self._reachability_low_score_patience_steps,
            tip_error_m=_last_finite(self._tracking_error),
            tendon_error_m=_last_finite(self._tendon_error),
            threshold=threshold,
        )
        self._figure.suptitle(title, fontsize=10, **title_style)
        self._figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()


def _finite_positive(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[~np.isfinite(result) | (result <= 0.0)] = np.nan
    return result


def _last_finite(values, *, default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float(default)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(finite[-1])


def _combined_legend(left_axis, right_axis, *, loc: str) -> None:
    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc=loc,
        fontsize=8,
    )


def _shade_boolean_regions(
    axis,
    time_s: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> None:
    if time_s.size == 0 or mask.size != time_s.size:
        return
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return
    median_dt = _median_dt(time_s)
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == mask.size - 1):
            end = index - 1 if not active else index
            left = float(time_s[start])
            right = float(time_s[end])
            if right <= left:
                right = left + median_dt
            axis.axvspan(left, right, color=color, alpha=alpha, linewidth=0.0)
            start = None


def _draw_waypoint_boundaries(
    axis,
    time_s: np.ndarray,
    waypoint_indices: np.ndarray,
    *,
    annotate: bool,
) -> None:
    if time_s.size < 2 or waypoint_indices.size != time_s.size:
        return
    previous = waypoint_indices[0]
    for index in range(1, waypoint_indices.size):
        current = waypoint_indices[index]
        if not np.isfinite(previous) or not np.isfinite(current):
            previous = current
            continue
        if int(current) != int(previous):
            time_value = float(time_s[index])
            axis.axvline(
                time_value,
                color="0.35",
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
            )
            if annotate:
                axis.text(
                    time_value,
                    0.98,
                    f"wp {int(current)}",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                    color="0.25",
                    transform=axis.get_xaxis_transform(),
                )
        previous = current


def _median_dt(time_s: np.ndarray) -> float:
    if time_s.size < 2:
        return 1.0e-3
    delta = np.diff(time_s)
    finite = delta[np.isfinite(delta) & (delta > 0.0)]
    if finite.size == 0:
        return 1.0e-3
    return float(np.median(finite))


def _reachability_bottleneck(
    progress: np.ndarray,
    alignment: np.ndarray,
    model: np.ndarray,
) -> str:
    values = {
        "progress": _last_finite(progress),
        "alignment": _last_finite(alignment),
        "model": _last_finite(model),
    }
    finite = {name: value for name, value in values.items() if np.isfinite(value)}
    if not finite:
        return "n/a"
    return min(finite, key=finite.get)


def _bottleneck_value(
    bottleneck: str,
    progress: np.ndarray,
    alignment: np.ndarray,
    model: np.ndarray,
) -> float:
    values = {
        "progress": _last_finite(progress),
        "alignment": _last_finite(alignment),
        "model": _last_finite(model),
    }
    return float(values.get(bottleneck, np.nan))


def _diagnostics_status_title(
    *,
    time_s: float,
    phase: str,
    waypoint_index: int,
    score: float,
    bottleneck: str,
    bottleneck_value: float,
    execution_score: float,
    low_score_steps: float,
    low_score_patience_steps: int,
    tip_error_m: float,
    tendon_error_m: float,
    threshold: float,
) -> tuple[str, dict[str, object]]:
    if np.isfinite(score) and score < threshold:
        background = "#c62828"
        foreground = "white"
    elif np.isfinite(score) and score < 0.7:
        background = "#f9a825"
        foreground = "black"
    elif np.isfinite(score):
        background = "#2e7d32"
        foreground = "white"
    else:
        background = "0.35"
        foreground = "white"
    patience = (
        "n/a"
        if low_score_patience_steps <= 0
        else f"{int(max(low_score_steps, 0.0))}/{low_score_patience_steps}"
    )
    title = (
        f"t={_format_status_value(time_s, 2)}s | phase={phase} | "
        f"wp={waypoint_index} | reach={_format_status_value(score, 3)} | "
        f"exec={_format_status_value(execution_score, 3)} | "
        f"bottleneck={bottleneck}:{_format_status_value(bottleneck_value, 3)} | "
        f"low={patience} | tip={_format_mm(tip_error_m)} | "
        f"tendon_err={_format_mm(tendon_error_m)}"
    )
    return title, {
        "color": foreground,
        "bbox": {
            "facecolor": background,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.35",
            "alpha": 0.95,
        },
    }


def _format_status_value(value: float, precision: int) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _format_mm(value_m: float) -> str:
    if not np.isfinite(value_m):
        return "nan mm"
    return f"{1000.0 * value_m:.2f} mm"


def _last_value(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(values[-1])


def _last_vector(values: list[np.ndarray]) -> str:
    if not values:
        return "[nan, nan, nan]"
    vector = np.asarray(values[-1], dtype=float)
    if vector.shape != (3,):
        return "[nan, nan, nan]"
    return "[" + ", ".join(f"{float(value): .5f}" for value in vector) + "]"


def _tip_target_error_vector(
    state: RobotSystemState,
    metadata: dict[str, object],
) -> np.ndarray | None:
    executor = _executor_arm(state)
    if executor is None:
        return None
    target = _metadata_point(metadata, "executor_target_world")
    if target is None:
        target = _metadata_point(metadata, "engine_navigation_active_target_m")
    if target is None:
        return None
    return target - executor.tip_pose_world.position


class MujocoViewerHook:
    """Optional passive viewer kept outside backend and controller policy."""

    def __init__(self, backend, *, keep_open: bool = False) -> None:
        self.backend = backend
        self.keep_open = keep_open
        self._viewer = None
        self._start_wall_s = 0.0
        self._start_sim_s = 0.0
        self._mujoco = None
        self._overlay_state = _TrackingOverlayState()

    def on_reset(self, state: RobotSystemState) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._overlay_state.clear()
        self._viewer = mujoco.viewer.launch_passive(
            self.backend.physics.model,
            self.backend.physics.data,
        )
        _configure_mujoco_viewer(self._viewer, self.backend.config)
        self._viewer.sync()
        self._start_wall_s = time.perf_counter()
        self._start_sim_s = state.time_s

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del step_index
        if self._viewer is not None:
            self._overlay_state.capture(
                state,
                command,
                max_points=self.backend.config.viewer.overlays.trail_max_points,
            )
            _draw_mujoco_tracking_overlay(
                self._viewer,
                self._mujoco,
                self.backend.config.viewer.overlays,
                self._overlay_state,
                state=state,
            )
            _update_follow_camera(
                getattr(self._viewer, "cam", None),
                self.backend.config.viewer.camera,
                state,
            )
            self._viewer.sync()
            viewer_config = self.backend.config.viewer
            if viewer_config.realtime:
                _sleep_until_simulation_time(
                    self._start_wall_s,
                    self._start_sim_s,
                    state.time_s,
                    viewer_config.realtime_factor,
                )

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return self._viewer is not None and not self._viewer.is_running()

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._viewer is not None:
            while self.keep_open and self._viewer.is_running():
                self._viewer.sync()
                time.sleep(0.03)
            self._viewer.close()
            self._viewer = None


def _configure_mujoco_viewer(viewer, config) -> None:
    """Apply the canonical MuJoCo camera and visibility configuration."""

    cam = getattr(viewer, "cam", None)
    if cam is not None:
        camera = config.viewer.camera
        cam.lookat[:] = camera.lookat
        cam.distance = camera.distance
        cam.azimuth = camera.azimuth
        cam.elevation = camera.elevation
    opt = getattr(viewer, "opt", None)
    if opt is not None and hasattr(opt, "geomgroup"):
        opt.geomgroup[config.visuals.visual_geom_group] = 1
        opt.geomgroup[config.visuals.collision_geom_group] = int(
            config.viewer.show_collision_geoms
        )


def _sleep_until_simulation_time(
    start_wall_s: float,
    start_sim_s: float,
    current_sim_s: float,
    realtime_factor: float,
) -> None:
    if realtime_factor <= 0.0:
        raise ValueError("MuJoCo viewer realtime_factor must be positive.")
    target_elapsed_s = (current_sim_s - start_sim_s) / realtime_factor
    delay_s = target_elapsed_s - (time.perf_counter() - start_wall_s)
    if delay_s > 0.0:
        time.sleep(delay_s)


def _draw_mujoco_tracking_overlay(
    viewer,
    mujoco,
    config,
    overlay_state: _TrackingOverlayState,
    state: RobotSystemState | None = None,
) -> None:
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
    _draw_tracking_overlay_scene(
        scene,
        mujoco,
        config,
        overlay_state,
        state=state,
        reset_scene=True,
    )


def _draw_tracking_overlay_scene(
    scene,
    mujoco,
    config,
    overlay_state: _TrackingOverlayState,
    *,
    state: RobotSystemState | None = None,
    reset_scene: bool,
) -> None:
    if reset_scene:
        scene.ngeom = 0
    navigation_config = config.engine_navigation
    navigation_enabled = bool(
        navigation_config.enabled and overlay_state.navigation_metadata
    )
    if navigation_enabled:
        _draw_engine_navigation_overlay_scene(
            scene,
            mujoco,
            navigation_config,
            config,
            overlay_state,
        )
    elif config.target_marker and overlay_state.target_trail:
        _add_overlay_sphere(
            scene,
            mujoco,
            overlay_state.target_trail[-1],
            config.target_marker_radius,
            config.target_marker_rgba,
        )
    if (
        not navigation_enabled
        and navigation_config.observer_roi
        and overlay_state.observer_roi_world is not None
    ):
        _add_overlay_sphere(
            scene,
            mujoco,
            overlay_state.observer_roi_world,
            navigation_config.observer_roi_radius,
            navigation_config.observer_roi_rgba,
        )
    if not navigation_enabled and config.tip_trail:
        _add_overlay_trail(
            scene,
            mujoco,
            overlay_state.tip_trail[:: config.trail_stride],
            config.tip_trail_radius,
            config.tip_trail_rgba,
        )
    if not navigation_enabled and config.target_trail:
        _add_overlay_trail(
            scene,
            mujoco,
            overlay_state.target_trail[:: config.trail_stride],
            config.target_trail_radius,
            config.target_trail_rgba,
        )
    if state is not None and config.error_vector:
        _draw_error_vector_overlay_scene(
            scene,
            mujoco,
            config,
            overlay_state,
            state,
            navigation_enabled=navigation_enabled,
        )
    if state is not None and config.segment_endpoints:
        _draw_segment_endpoint_overlay_scene(scene, mujoco, state, config)


def _draw_engine_navigation_overlay_scene(
    scene,
    mujoco,
    config,
    shared_config,
    overlay_state: _TrackingOverlayState,
) -> None:
    metadata = overlay_state.navigation_metadata
    active_target = _metadata_point(
        metadata,
        "engine_navigation_active_target_m",
    )
    if config.current_target and active_target is not None:
        target_kind = metadata.get(
            "engine_navigation_active_target_kind",
            "executor",
        )
        if target_kind == "base":
            radius = config.base_target_radius
            rgba = config.base_target_rgba
        else:
            radius = config.executor_target_radius
            rgba = config.executor_target_rgba
        _add_overlay_sphere(scene, mujoco, active_target, radius, rgba)

    pre_entry = _metadata_point(
        metadata,
        "engine_navigation_pre_entry_target_m",
    )
    if config.planned_paths and pre_entry is not None:
        _add_overlay_sphere(
            scene,
            mujoco,
            pre_entry,
            config.pre_entry_target_radius,
            config.pre_entry_target_rgba,
        )

    observer_roi = _metadata_point(
        metadata,
        "engine_navigation_observer_roi_m",
    )
    if config.observer_roi and observer_roi is not None:
        _add_overlay_sphere(
            scene,
            mujoco,
            observer_roi,
            config.observer_roi_radius,
            config.observer_roi_rgba,
        )

    insertion_path = _metadata_path(
        metadata,
        "engine_navigation_insertion_path_m",
    )
    if config.insertion_waypoints and insertion_path is not None:
        for point in _sample_overlay_points(
            insertion_path,
            config.waypoint_stride,
        ):
            _add_overlay_sphere(
                scene,
                mujoco,
                point,
                config.insertion_waypoint_radius,
                config.insertion_waypoint_rgba,
            )

    if config.planned_paths:
        paths = (
            (
                "engine_navigation_base_path_m",
                config.base_path_radius,
                config.base_path_rgba,
            ),
            (
                "engine_navigation_insertion_path_m",
                config.insertion_path_radius,
                config.insertion_path_rgba,
            ),
        )
        for key, radius, rgba in paths:
            points = _metadata_path(metadata, key)
            if points is not None:
                _add_overlay_trail(
                    scene,
                    mujoco,
                    _sample_overlay_points(points, config.path_stride),
                    radius,
                    rgba,
                )
        executor_paths = _metadata_paths(
            metadata,
            "engine_navigation_executor_paths_m",
        )
        if not executor_paths:
            fallback = _metadata_path(
                metadata,
                "engine_navigation_executor_path_m",
            )
            executor_paths = () if fallback is None else (fallback,)
        for points in executor_paths:
            _add_overlay_trail(
                scene,
                mujoco,
                _sample_overlay_points(points, config.path_stride),
                config.executor_path_radius,
                config.executor_path_rgba,
            )

    history_slice = slice(None, None, shared_config.trail_stride)
    if config.base_history:
        _add_overlay_trail(
            scene,
            mujoco,
            overlay_state.base_trail[history_slice],
            config.base_history_radius,
            config.base_history_rgba,
        )
    if config.executor_history:
        _add_overlay_trail(
            scene,
            mujoco,
            overlay_state.tip_trail[history_slice],
            config.executor_history_radius,
            config.executor_history_rgba,
        )
    if config.target_history:
        for target_segment in _split_target_history(
            overlay_state.target_trail,
            overlay_state.target_trail_kinds,
            shared_config.trail_stride,
        ):
            _add_overlay_trail(
                scene,
                mujoco,
                target_segment,
                config.target_history_radius,
                config.target_history_rgba,
            )


def _draw_error_vector_overlay_scene(
    scene,
    mujoco,
    config,
    overlay_state: _TrackingOverlayState,
    state: RobotSystemState,
    *,
    navigation_enabled: bool,
) -> None:
    start: np.ndarray | None = None
    target: np.ndarray | None = None
    if navigation_enabled:
        metadata = overlay_state.navigation_metadata
        target = _metadata_point(metadata, "engine_navigation_active_target_m")
        kind = metadata.get("engine_navigation_active_target_kind", "executor")
        if kind == "base":
            start = state.base.pose.position.copy()
        else:
            executor = _executor_arm(state)
            start = None if executor is None else executor.tip_pose_world.position.copy()
    else:
        target = overlay_state.target_trail[-1] if overlay_state.target_trail else None
        executor = _executor_arm(state)
        start = None if executor is None else executor.tip_pose_world.position.copy()
    if start is None or target is None:
        return
    points = np.asarray([start, target], dtype=float)
    if not np.all(np.isfinite(points)):
        return
    if float(np.linalg.norm(points[1] - points[0])) <= 1.0e-9:
        return
    _add_overlay_trail(
        scene,
        mujoco,
        points,
        config.error_vector_radius,
        config.error_vector_rgba,
    )


def _sample_overlay_points(points: np.ndarray, stride: int) -> np.ndarray:
    sampled = points[::stride]
    if (len(points) - 1) % stride != 0:
        sampled = np.vstack((sampled, points[-1]))
    return sampled


def _split_target_history(
    points: list[np.ndarray],
    kinds: list[str],
    stride: int,
) -> list[list[np.ndarray]]:
    segments: list[list[np.ndarray]] = []
    previous_kind: str | None = None
    for point, kind in zip(points, kinds, strict=True):
        if not segments or kind != previous_kind:
            segments.append([point])
        else:
            segments[-1].append(point)
        previous_kind = kind
    return [
        list(_sample_overlay_points(np.asarray(segment), stride))
        for segment in segments
    ]


def _draw_segment_endpoint_overlay_scene(
    scene,
    mujoco,
    state: RobotSystemState,
    config,
) -> None:
    for arm in state.arms.values():
        if arm.role == "executor":
            rgba = config.executor_segment_endpoint_rgba
        elif arm.role == "observer":
            rgba = config.observer_segment_endpoint_rgba
        else:
            continue
        for pose in arm.segment_poses_world:
            _add_overlay_sphere(
                scene,
                mujoco,
                np.asarray(pose, dtype=float)[:3, 3],
                config.segment_endpoint_radius,
                rgba,
            )


def _add_overlay_sphere(scene, mujoco, position, radius, rgba) -> None:
    if int(scene.ngeom) >= int(scene.maxgeom):
        return
    geom = scene.geoms[int(scene.ngeom)]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )


def _add_overlay_trail(scene, mujoco, points, radius, rgba) -> None:
    for start, end in zip(points[:-1], points[1:]):
        if int(scene.ngeom) >= int(scene.maxgeom):
            return
        geom = scene.geoms[int(scene.ngeom)]
        scene.ngeom += 1
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.ascontiguousarray(start, dtype=np.float64),
            np.ascontiguousarray(end, dtype=np.float64),
        )
        geom.rgba[:] = rgba


class MatplotlibSystemViewerHook:
    """Simple named-arm centerline viewer for analytic or MuJoCo scenarios."""

    def __init__(self, *, keep_open: bool = True) -> None:
        self.keep_open = keep_open
        self._plt = None
        self._figure = None
        self._axes = None
        self._target = None
        self._tip_trail: list[np.ndarray] = []
        self._target_trail: list[np.ndarray] = []

    def on_reset(self, state: RobotSystemState) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure = plt.figure()
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._tip_trail.clear()
        self._target_trail.clear()
        plt.ion()
        self._draw(state)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del step_index
        target = command.metadata.get("executor_target_world")
        if target is not None:
            self._target = np.asarray(target, dtype=float).copy()
            self._target_trail.append(self._target)
        executor = _executor_arm(state)
        if executor is not None:
            self._tip_trail.append(executor.tip_pose_world.position.copy())
        self._draw(state)

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return self._figure is not None and not self._plt.fignum_exists(self._figure.number)

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._plt is None:
            return
        self._plt.ioff()
        if self.keep_open:
            self._plt.show()

    def _draw(self, state: RobotSystemState) -> None:
        axes = self._axes
        axes.clear()
        all_points: list[np.ndarray] = []
        for name, arm in state.arms.items():
            points = arm.centerline_world
            if points is None:
                points = np.asarray(
                    [pose[:3, 3] for pose in arm.segment_poses_world],
                    dtype=float,
                )
                points = np.vstack((state.base.pose.position, points))
            axes.plot(points[:, 0], points[:, 1], points[:, 2], label=name)
            all_points.append(points)
        if all_points:
            values = np.vstack(all_points)
            center = 0.5 * (values.min(axis=0) + values.max(axis=0))
            radius = max(float(np.max(values.max(axis=0) - values.min(axis=0))) * 0.6, 0.05)
            axes.set_xlim(center[0] - radius, center[0] + radius)
            axes.set_ylim(center[1] - radius, center[1] + radius)
            axes.set_zlim(center[2] - radius, center[2] + radius)
        if self._target is not None:
            axes.scatter(*self._target, color="tab:orange", marker="x", s=45, label="target")
        if self._tip_trail:
            trail = np.asarray(self._tip_trail)
            axes.plot(*trail.T, color="tab:blue", linewidth=1.2, label="executor trail")
        if self._target_trail:
            trail = np.asarray(self._target_trail)
            axes.plot(*trail.T, "--", color="tab:orange", linewidth=1.0)
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_zlabel("z [m]")
        axes.legend()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
