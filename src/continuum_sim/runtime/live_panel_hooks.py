"""Interactive live-panel hooks."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

from continuum_sim.runtime.concurrency import LatestValueSlot, TimeRateGate
from continuum_sim.runtime.hook_utils import (
    finite_metadata_float as _finite_metadata_float,
    metadata_max_abs as _metadata_max_abs,
    metadata_norm as _metadata_norm,
    metadata_point as _metadata_point,
    tip_target_error_vector as _tip_target_error_vector,
)
from continuum_sim.runtime.matplotlib_artists import PersistentAxisArtists
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class LiveTendonPanelHook:
    """Optional rich tendon monitor attached to the scenario hook lifecycle."""

    requires_gui_main_thread = True

    def __init__(
        self,
        *,
        stride: int = 1,
        history_points: int = 300,
        display_interval_s: float | None = None,
    ) -> None:
        if stride <= 0:
            raise ValueError("LiveTendonPanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._panel = None
        self._samples = None
        self._sample_version = -1
        self._display_gate = (
            None
            if display_interval_s is None
            else TimeRateGate(display_interval_s)
        )

    def on_reset(self, state: RobotSystemState) -> None:
        self._samples = LatestValueSlot(state)
        self._sample_version = -1
        if self._display_gate is not None:
            self._display_gate.reset(state.time_s)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del command
        due = (
            self._display_gate.due(state.time_s)
            if self._display_gate is not None
            else step_index % self.stride == 0
        )
        if due and self._samples is not None:
            self._samples.publish(state)

    def present_pending(self, *, force: bool = False) -> None:
        if self._samples is None:
            return
        item = self._samples.consume_after(self._sample_version)
        if item is None:
            return
        state, self._sample_version = item
        if self._panel is None:
            from continuum_sim.visualization.system_tendon_debug import (
                SystemTendonMonitorPanel,
            )

            self._panel = SystemTendonMonitorPanel()
            self._panel.show(block=False)
        if self._panel.is_open():
            self._panel.update(state)

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        if self._samples is not None:
            self._samples.publish(state)

    def close_presentation(self) -> None:
        if self._panel is not None:
            _safe_panel_call(self._panel, "close")
            self._panel = None


def _safe_panel_call(panel: object, method_name: str) -> None:
    method = getattr(panel, method_name, None)
    if not callable(method):
        return
    try:
        method()
    except Exception:
        pass


class LiveWipingForcePanelHook:
    """Optional live panel for scenario wiping force/contact metadata."""

    requires_gui_main_thread = True

    def __init__(
        self,
        *,
        stride: int = 1,
        history_points: int = 300,
        display_interval_s: float | None = None,
    ) -> None:
        if stride <= 0:
            raise ValueError("LiveWipingForcePanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._plt = None
        self._figure = None
        self._axes = None
        self._samples = None
        self._sample_version = -1
        self._display_gate = (
            None
            if display_interval_s is None
            else TimeRateGate(display_interval_s)
        )
        self._time: list[float] = []
        self._target_force: list[float] = []
        self._current_force: list[float] = []
        self._force_error: list[float] = []
        self._contact_distance: list[float] = []

    def on_reset(self, state: RobotSystemState) -> None:
        self._samples = LatestValueSlot((state, None))
        self._sample_version = -1
        self._time.clear()
        self._target_force.clear()
        self._current_force.clear()
        self._force_error.clear()
        self._contact_distance.clear()
        if self._display_gate is not None:
            self._display_gate.reset(state.time_s)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        due = (
            self._display_gate.due(state.time_s)
            if self._display_gate is not None
            else step_index % self.stride == 0
        )
        if not due or self._samples is None:
            return
        self._samples.publish((state, command))

    def present_pending(self, *, force: bool = False) -> None:
        if self._samples is None:
            return
        item = self._samples.consume_after(self._sample_version)
        if item is None:
            return
        (state, command), self._sample_version = item
        if command is None:
            return
        if self._figure is None:
            self._create_figure()
        self._time.append(float(state.time_s))
        self._target_force.append(
            float(command.metadata.get("target_normal_force_n", np.nan))
        )
        self._current_force.append(
            _finite_metadata_float(
                command.metadata,
                "measured_normal_force_n",
                fallback_key="estimated_normal_force_n",
            )
        )
        self._force_error.append(float(command.metadata.get("force_error_n", np.nan)))
        self._contact_distance.append(
            float(command.metadata.get("contact_distance_m", np.nan))
        )
        self._trim()
        self._draw()

    def _create_figure(self) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim wiping contact force")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        force_axis, contact_axis = axes
        force_axis.set_ylabel("force [N]")
        force_axis.set_title("Wiping contact force")
        contact_axis.set_xlabel("time [s]")
        contact_axis.set_ylabel("distance [mm]")
        contact_axis.set_title("Contact distance / penetration proxy")
        self._axes = tuple(PersistentAxisArtists(axis) for axis in axes)
        plt.ion()
        plt.show(block=False)

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def close_presentation(self) -> None:
        if self._plt is not None:
            try:
                self._plt.ioff()
                if self._figure is not None:
                    self._plt.close(self._figure)
            except Exception:
                pass
        self._figure = None
        self._axes = None

    def _trim(self) -> None:
        if len(self._time) <= self.history_points:
            return
        excess = len(self._time) - self.history_points
        del self._time[:excess]
        del self._target_force[:excess]
        del self._current_force[:excess]
        del self._force_error[:excess]
        del self._contact_distance[:excess]

    def _draw(self) -> None:
        if self._axes is None or self._figure is None:
            return
        force_axis, contact_axis = self._axes
        for axis in self._axes:
            axis.begin_frame()
        time_s = np.asarray(self._time, dtype=float)
        target_force = np.asarray(self._target_force, dtype=float)
        current_force = np.asarray(self._current_force, dtype=float)
        force_error = np.asarray(self._force_error, dtype=float)
        contact_distance_mm = 1000.0 * np.asarray(self._contact_distance, dtype=float)
        penetration_mm = 1000.0 * np.maximum(
            0.0,
            -np.asarray(self._contact_distance, dtype=float),
        )

        force_axis.plot(time_s, target_force, "--", label="target force [N]")
        force_axis.plot(time_s, current_force, label="current contact force [N]")
        force_axis.plot(time_s, force_error, label="force error [N]")
        force_axis.legend(loc="upper right", fontsize=8)

        contact_axis.plot(time_s, contact_distance_mm, label="contact distance [mm]")
        contact_axis.plot(time_s, penetration_mm, label="penetration proxy [mm]")
        contact_axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        contact_axis.legend(loc="upper right", fontsize=8)
        for axis in self._axes:
            axis.end_frame()
        self._figure.canvas.draw_idle()


class LiveDiagnosticsPanelHook:
    """Optional compact live panel for tracking, safety, and actuator diagnostics."""

    requires_gui_main_thread = True

    def __init__(
        self,
        *,
        stride: int = 5,
        history_points: int = 300,
        display_interval_s: float | None = None,
    ) -> None:
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
        self._snapshot_png: bytes | None = None
        self.errors: list[str] = []
        self._samples = None
        self._sample_version = -1
        self._display_gate = (
            None
            if display_interval_s is None
            else TimeRateGate(display_interval_s)
        )

    def on_reset(self, state: RobotSystemState) -> None:
        self._info_text = None
        self._snapshot_png = None
        self.errors.clear()
        self._clear()
        self._samples = LatestValueSlot((state, None))
        self._sample_version = -1
        if self._display_gate is not None:
            self._display_gate.reset(state.time_s)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        due = (
            self._display_gate.due(state.time_s)
            if self._display_gate is not None
            else step_index % self.stride == 0
        )
        if not due or self._samples is None:
            return
        self._samples.publish((state, command))

    def present_pending(self, *, force: bool = False) -> None:
        if self._samples is None:
            return
        item = self._samples.consume_after(self._sample_version)
        if item is None:
            return
        (state, command), self._sample_version = item
        if self._figure is None:
            self._create_figure()
        self._append(state, command)
        self._trim()
        self._draw()

    def _create_figure(self) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, raw_axes = plt.subplots(3, 2, figsize=(12.0, 9.6))
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim live diagnostics")
        raw_axes = raw_axes.reshape(-1)
        ik_right = raw_axes[2].twinx()
        backend_right = raw_axes[3].twinx()
        drivers_right = raw_axes[5].twinx()
        for axis in raw_axes:
            axis.grid(True, alpha=0.25)
        for axis in (ik_right, backend_right, drivers_right):
            axis.patch.set_alpha(0.0)
        self._axes = tuple(PersistentAxisArtists(axis) for axis in raw_axes)
        self._ik_right_axis = PersistentAxisArtists(ik_right)
        self._backend_right_axis = PersistentAxisArtists(backend_right)
        self._drivers_right_axis = PersistentAxisArtists(drivers_right)
        plt.ion()
        plt.show(block=False)

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def close_presentation(self) -> None:
        if self._plt is None:
            return
        try:
            self._capture_snapshot()
            self._plt.ioff()
            if self._figure is not None:
                self._plt.close(self._figure)
        except Exception as exc:
            self.errors.append(f"live_diagnostics_panel: {type(exc).__name__}: {exc}")
        self._figure = None
        self._axes = None
        self._ik_right_axis = None
        self._backend_right_axis = None
        self._drivers_right_axis = None

    def save_snapshot(self, path: str | Path) -> Path | None:
        """Save the final live diagnostics panel image collected during shutdown."""

        destination = Path(path)
        if self._snapshot_png is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self._snapshot_png)
            return destination
        if self._figure is None:
            return None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._figure.savefig(destination, dpi=160, bbox_inches="tight")
        except Exception as exc:
            self.errors.append(f"live_diagnostics_panel: {type(exc).__name__}: {exc}")
            return None
        return destination

    def _capture_snapshot(self) -> None:
        if self._figure is None:
            return
        try:
            self._draw()
            buffer = BytesIO()
            self._figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
            self._snapshot_png = buffer.getvalue()
        except Exception as exc:
            self.errors.append(f"live_diagnostics_panel: {type(exc).__name__}: {exc}")

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
            axis.begin_frame()
        right_axes = (
            self._ik_right_axis,
            self._backend_right_axis,
            self._drivers_right_axis,
        )
        for axis in right_axes:
            if axis is not None:
                axis.begin_frame()

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
        for axis in (*axes, *right_axes):
            if axis is not None:
                axis.end_frame()
        self._figure.canvas.draw_idle()


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

__all__ = [
    "LiveDiagnosticsPanelHook",
    "LiveTendonPanelHook",
    "LiveWipingForcePanelHook",
]
