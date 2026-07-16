"""Optional observers for scenario-driven simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np

from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


_ENGINE_NAVIGATION_METADATA_KEYS = (
    "engine_navigation_pre_entry_target_m",
    "engine_navigation_base_path_m",
    "engine_navigation_insertion_path_m",
    "engine_navigation_executor_path_m",
    "engine_navigation_executor_paths_m",
    "engine_navigation_observer_roi_m",
    "engine_navigation_active_target_m",
    "engine_navigation_active_target_kind",
)


@dataclass
class _TrackingOverlayState:
    """Shared, bounded tracking data for live and recorded MuJoCo overlays."""

    tip_trail: list[np.ndarray] = field(default_factory=list)
    target_trail: list[np.ndarray] = field(default_factory=list)
    target_trail_kinds: list[str] = field(default_factory=list)
    base_trail: list[np.ndarray] = field(default_factory=list)
    navigation_metadata: dict[str, object] = field(default_factory=dict)

    def clear(self) -> None:
        self.tip_trail.clear()
        self.target_trail.clear()
        self.target_trail_kinds.clear()
        self.base_trail.clear()
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
        if is_engine_navigation:
            self.base_trail.append(state.base.pose.position.copy())
            self.navigation_metadata = {
                key: _copy_overlay_metadata_value(command.metadata[key])
                for key in _ENGINE_NAVIGATION_METADATA_KEYS
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
    tracking_complete: list[bool] = field(default_factory=list)
    tracking_approach: list[bool] = field(default_factory=list)
    arm_saturation_scale: dict[str, list[float]] = field(default_factory=dict)
    arm_tendon_target_error_norm_m: dict[str, list[float]] = field(default_factory=dict)
    arm_tendon_target_error_max_m: dict[str, list[float]] = field(default_factory=dict)
    arm_peak_actuator_force_n: dict[str, list[float]] = field(default_factory=dict)
    min_clearance_m: list[float] = field(default_factory=list)
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
        self.tracking_complete.clear()
        self.tracking_approach.clear()
        self.arm_saturation_scale = {name: [] for name in state.arms}
        self.arm_tendon_target_error_norm_m = {name: [] for name in state.arms}
        self.arm_tendon_target_error_max_m = {name: [] for name in state.arms}
        self.arm_peak_actuator_force_n = {name: [] for name in state.arms}
        self.min_clearance_m.clear()
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
            executor = _executor_arm(state)
            self.target_actual_position_m.append(
                np.full(3, np.nan, dtype=float)
                if executor is None
                else executor.tip_pose_world.position.copy()
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
            self.tracking_complete.append(
                bool(command.metadata.get("tracking_complete", False))
            )
            self.tracking_approach.append(
                bool(command.metadata.get("tracking_approach", False))
            )
            self.min_clearance_m.append(
                float(command.metadata.get("min_clearance_m", np.nan))
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


def _metadata_vector_or_nan(
    metadata: dict[str, Any],
    key: str,
) -> np.ndarray:
    value = metadata.get(key)
    if value is None:
        return np.full(3, np.nan, dtype=float)
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return np.full(3, np.nan, dtype=float)
    return vector.copy()


def _executor_arm(state: RobotSystemState):
    return next((arm for arm in state.arms.values() if arm.role == "executor"), None)


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


class MujocoLiveVideoRecorderHook:
    """Write MuJoCo scene frames during simulation instead of replaying afterward."""

    def __init__(
        self,
        backend: object,
        output_path: str | Path,
        *,
        fps: int = 20,
        stride: int | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        if fps <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook fps must be positive.")
        if stride is not None and stride <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook stride must be positive.")
        self.backend = backend
        self.output_path = Path(output_path)
        self.fps = fps
        self.stride = 1 if stride is None else stride
        self.width = width
        self.height = height
        self.path: Path | None = None
        self.errors: list[str] = []
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writer = None
        self._camera = None
        self._overlay_state = _TrackingOverlayState()

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self.path = None
        self.errors.clear()
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writer = None
        self._camera = None
        self._overlay_state.clear()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            import mujoco

            self._mujoco = mujoco
            model = self.backend.physics.model
            self._renderer = mujoco.Renderer(
                model,
                height=self.height,
                width=self.width,
            )
            self._camera = _mujoco_render_camera(
                mujoco,
                getattr(self.backend.config.viewer, "camera", None),
            )
            self._writer = _open_video_writer(imageio, self.output_path, self.fps)
        except Exception as exc:  # noqa: BLE001 - video must not fail the run.
            self._record_error(f"live video setup failed: {type(exc).__name__}: {exc}")
            self._close_resources()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if self._renderer is None or self._writer is None or self._mujoco is None:
            return
        self._overlay_state.capture(
            state,
            command,
            max_points=self.backend.config.viewer.overlays.trail_max_points,
        )
        if step_index % self.stride != 0:
            return
        try:
            data = self.backend.physics.data
            self._mujoco.mj_forward(self.backend.physics.model, data)
            _update_follow_camera(
                self._camera,
                self.backend.config.viewer.camera,
                state,
            )
            self._renderer.update_scene(data, camera=self._camera)
            _draw_tracking_overlay_scene(
                self._renderer.scene,
                self._mujoco,
                self.backend.config.viewer.overlays,
                self._overlay_state,
                state=state,
                reset_scene=False,
            )
            frame = self._renderer.render().copy()
            self._writer.append_data(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001 - preserve non-video artifacts.
            self._record_error(
                f"live video frame {self.frame_count} failed at step "
                f"{step_index}: {type(exc).__name__}: {exc}"
            )
            self._close_resources()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        self._close_resources()
        if self.frame_count > 0 and not self.errors:
            self.path = self.output_path
        elif self.frame_count > 0:
            self.path = self.output_path
        elif not self.errors:
            self._record_error("live video produced no frames")

    def _record_error(self, message: str) -> None:
        self.errors.append(message)
        error_path = self.output_path.parent / "video_error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if error_path.is_file():
            existing = error_path.read_text(encoding="utf-8")
        error_path.write_text(existing + message + "\n", encoding="utf-8")

    def _close_resources(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - report writer close errors.
                self._record_error(
                    f"live video writer close failed: {type(exc).__name__}: {exc}"
                )
        renderer = self._renderer
        self._renderer = None
        close = getattr(renderer, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - report renderer close errors.
                self._record_error(
                    f"live video renderer close failed: {type(exc).__name__}: {exc}"
                )


def _open_video_writer(imageio, path: Path, fps: int):
    if path.suffix.lower() == ".gif":
        return imageio.get_writer(path, mode="I", duration=1000.0 / fps, loop=0)
    return imageio.get_writer(path, fps=fps)


def _mujoco_render_camera(mujoco, camera: object | None):
    if camera is None:
        return -1
    render_camera = mujoco.MjvCamera()
    render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    render_camera.lookat[:] = getattr(camera, "lookat")
    render_camera.distance = float(getattr(camera, "distance"))
    render_camera.azimuth = float(getattr(camera, "azimuth"))
    render_camera.elevation = float(getattr(camera, "elevation"))
    return render_camera


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


@dataclass
class TendonDiagnosticHook:
    """Collect tendon and singularity snapshots without a GUI dependency."""

    stride: int = 1
    samples: list[dict[str, Any]] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        self.samples.clear()
        self.samples.append(self._snapshot(state, None, -1))

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride == 0:
            self.samples.append(self._snapshot(state, command, step_index))

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _snapshot(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand | None,
        step_index: int,
    ) -> dict[str, Any]:
        return {
            "step_index": step_index,
            "time_s": state.time_s,
            "arms": {
                name: {
                    "tendon_target_m": arm.tendon_target_m.copy(),
                    "tendon_displacement_m": arm.tendon_displacement_m.copy(),
                    "tendon_velocity_mps": arm.tendon_velocity_mps.copy(),
                    "actuator_force_n": arm.actuator_force_n.copy(),
                }
                for name, arm in state.arms.items()
            },
            "command_metadata": {} if command is None else dict(command.metadata),
            "state_metadata": dict(state.metadata),
        }


class LiveTendonPanelHook:
    """Optional rich tendon monitor attached to the scenario hook lifecycle."""

    def __init__(self, *, stride: int = 1, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveTendonPanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._panel = None

    def on_reset(self, state: RobotSystemState) -> None:
        from continuum_sim.visualization.system_tendon_debug import (
            SystemTendonMonitorPanel,
        )

        if self._panel is not None:
            self._panel.close()
        self._panel = SystemTendonMonitorPanel()
        self._panel.update(state, redraw=False)
        self._panel.show(block=False)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del command
        if step_index % self.stride == 0:
            if self._panel is not None and self._panel.is_open():
                self._panel.update(state)
                self._panel.flush_events()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._panel is not None:
            _safe_panel_call(self._panel, "flush_events")
            _safe_panel_call(self._panel, "close")
            self._panel = None


class LiveWipingForcePanelHook:
    """Optional live panel for scenario wiping force/contact metadata."""

    def __init__(self, *, stride: int = 1, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveWipingForcePanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._plt = None
        self._figure = None
        self._axes = None
        self._time: list[float] = []
        self._target_force: list[float] = []
        self._estimated_force: list[float] = []
        self._contact_distance: list[float] = []

    def on_reset(self, state: RobotSystemState) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, self._axes = plt.subplots()
        self._time.clear()
        self._target_force.clear()
        self._estimated_force.clear()
        self._contact_distance.clear()
        plt.ion()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride != 0:
            return
        self._time.append(float(state.time_s))
        self._target_force.append(float(command.metadata.get("target_normal_force_n", np.nan)))
        self._estimated_force.append(
            float(command.metadata.get("estimated_normal_force_n", np.nan))
        )
        self._contact_distance.append(float(command.metadata.get("contact_distance_m", np.nan)))
        self._trim()
        self._draw()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
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
        del self._estimated_force[:excess]
        del self._contact_distance[:excess]

    def _draw(self) -> None:
        if self._axes is None or self._figure is None:
            return
        self._axes.clear()
        self._axes.plot(self._time, self._target_force, label="target force [N]")
        self._axes.plot(self._time, self._estimated_force, label="estimated force [N]")
        self._axes.plot(self._time, self._contact_distance, label="contact distance [m]")
        self._axes.set_xlabel("time [s]")
        self._axes.legend(loc="upper right")
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()


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
        self._phase = ""
        self._observer_mode = ""
        self._waypoint_index = -1
        self._last_task_target: np.ndarray | None = None

    def on_reset(self, state: RobotSystemState) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("continuum_sim live diagnostics")
        self._axes = axes.reshape(-1)
        self._info_text = None
        self._clear()
        self._append(state, None)
        plt.ion()
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
        ):
            values.clear()
        self._tip_error_xyz.clear()
        self._phase = ""
        self._observer_mode = ""
        self._waypoint_index = -1
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
        axes[0].legend(loc="upper right", fontsize=8)

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
        if np.any(np.isfinite(condition)):
            axes[2].semilogy(time_s, condition, label="condition")
        else:
            axes[2].plot(time_s, condition, label="condition")
        axes[2].plot(time_s, np.asarray(self._velocity_scale), label="velocity scale")
        axes[2].plot(
            time_s,
            np.asarray(self._ik_residual),
            label="residual",
        )
        axes[2].plot(
            time_s,
            np.asarray(self._ik_projection_residual),
            label="projection residual",
        )
        axes[2].set(title="Layer 3: IK/tendon command", xlabel="time [s]")
        axes[2].legend(loc="upper right", fontsize=8)

        axes[3].plot(
            time_s,
            1000.0 * np.asarray(self._tendon_error),
            label="tendon target error",
        )
        axes[3].plot(
            time_s,
            np.asarray(self._force_utilization),
            label="force utilization",
        )
        axes[3].plot(
            time_s,
            np.asarray(self._saturation_scale),
            label="limit scale",
        )
        axes[3].plot(
            time_s,
            np.asarray(self._execution_saturation_active),
            label="saturation active",
        )
        axes[3].set(title="Layer 4: backend execution", xlabel="time [s]")
        axes[3].legend(loc="upper right", fontsize=8)

        title = (
            f"t={_last_value(self._time):.3f}s phase={self._phase} "
            f"wp={self._waypoint_index} "
            f"L1 err={_last_value(self._tracking_error):.4g}m "
            f"L2 v={_last_value(self._task_space_velocity):.4g}m/s "
            f"L3 cond={_last_value(self._condition):.3g} "
            f"L4 tendon_err={_last_value(self._tendon_error):.3g}m"
        )
        self._figure.suptitle(title, fontsize=9)
        self._figure.tight_layout()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()


def _finite_positive(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[~np.isfinite(result) | (result <= 0.0)] = np.nan
    return result


def _metadata_norm(value: object) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        return float("nan")
    return float(np.linalg.norm(array))


def _metadata_max_abs(value: object) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float("nan")
    return float(np.max(np.abs(finite)))


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


def _safe_panel_call(panel: object, method_name: str) -> None:
    method = getattr(panel, method_name, None)
    if not callable(method):
        return
    try:
        method()
    except Exception:
        pass


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


def _metadata_point(
    metadata: dict[str, object],
    key: str,
) -> np.ndarray | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point.copy()


def _metadata_path(
    metadata: dict[str, object],
    key: str,
) -> np.ndarray | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) == 0
        or not np.all(np.isfinite(points))
    ):
        return None
    return points.copy()


def _metadata_paths(
    metadata: dict[str, object],
    key: str,
) -> tuple[np.ndarray, ...]:
    value = metadata.get(key)
    if not isinstance(value, list | tuple):
        return ()
    result: list[np.ndarray] = []
    for item in value:
        path = _metadata_path({"path": item}, "path")
        if path is not None:
            result.append(path)
    return tuple(result)


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


class ControllerCompletionHook:
    """Stop a scenario when a waypoint-style controller reports completion."""

    def __init__(self, controller) -> None:
        self.controller = controller

    def on_reset(self, state: RobotSystemState) -> None:
        del state

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state, command, step_index

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return bool(getattr(self.controller, "done", False))

    def on_finish(self, state: RobotSystemState) -> None:
        del state


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
