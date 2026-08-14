"""MuJoCo video recording hooks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from threading import Event, Thread, current_thread

import numpy as np

from continuum_sim.runtime.concurrency import TimeRateGate
from continuum_sim.runtime.hook_utils import finite_metadata_float
from continuum_sim.runtime.matplotlib_artists import PersistentAxisArtists
from continuum_sim.runtime.mujoco_overlay_utils import (
    _TrackingOverlayState,
    _draw_tracking_overlay_scene,
    _update_follow_camera,
)
from continuum_sim.runtime.mujoco_state_copy import capture_mujoco_dynamic_state
from continuum_sim.runtime.video_utils import (
    BoundedFrameQueue,
    mujoco_render_camera as _mujoco_render_camera,
    normalise_output_paths as _normalise_output_paths,
    open_video_writer as _open_video_writer,
)
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


_VIDEO_LAYOUTS = ("scene_only", "scene_and_errors")


@dataclass(frozen=True)
class _TaskErrorHistorySnapshot:
    time_s: np.ndarray
    tracking_error_m: np.ndarray
    tracking_tolerance_m: np.ndarray
    target_force_n: np.ndarray
    measured_force_n: np.ndarray
    force_error_n: np.ndarray


class MujocoLiveVideoRecorderHook:
    """Write MuJoCo scene frames during simulation instead of replaying afterward."""

    def __init__(
        self,
        backend: object,
        output_path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        fps: int = 20,
        stride: int | None = None,
        width: int = 640,
        height: int = 480,
        queue_size: int = 8,
        layout: str = "scene_only",
        split_ratio: float = 0.5,
        show_force: bool = False,
        history_points: int = 600,
    ) -> None:
        if fps <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook fps must be positive.")
        if stride is not None and stride <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook stride must be positive.")
        if layout not in _VIDEO_LAYOUTS:
            raise ValueError(
                f"MujocoLiveVideoRecorderHook layout must be one of {_VIDEO_LAYOUTS}."
            )
        if not np.isfinite(split_ratio) or not 0.0 < split_ratio < 1.0:
            raise ValueError(
                "MujocoLiveVideoRecorderHook split_ratio must be between 0 and 1."
            )
        if history_points <= 0:
            raise ValueError(
                "MujocoLiveVideoRecorderHook history_points must be positive."
            )
        if layout == "scene_and_errors" and width < 2:
            raise ValueError("Composite MuJoCo video width must be at least 2 pixels.")
        self.backend = backend
        self.output_paths = _normalise_output_paths(output_path)
        self.output_path = self.output_paths[0]
        self.fps = fps
        self.stride = None if stride is None else int(stride)
        self._frame_gate = TimeRateGate(1.0 / self.fps)
        self.width = width
        self.height = height
        self.queue_size = int(queue_size)
        self.layout = str(layout)
        self.split_ratio = float(split_ratio)
        self.show_force = bool(show_force)
        self.history_points = int(history_points)
        if self.layout == "scene_and_errors":
            self._scene_width = min(
                self.width - 1,
                max(1, int(round(self.width * self.split_ratio))),
            )
            self._error_width = self.width - self._scene_width
        else:
            self._scene_width = self.width
            self._error_width = 0
        self.path: Path | None = None
        self.paths: list[Path] = []
        self.errors: list[str] = []
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._error_plot = None
        self._writers = []
        self._camera = None
        self._overlay_state = _TrackingOverlayState()
        self._queue = BoundedFrameQueue(maxsize=self.queue_size)
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._error_time: list[float] = []
        self._tracking_error: list[float] = []
        self._tracking_tolerance: list[float] = []
        self._target_force: list[float] = []
        self._measured_force: list[float] = []
        self._force_error: list[float] = []
        self._last_due_step_index: int | None = None
        self._last_written_step_index: int | None = None
        self._last_frame: np.ndarray | None = None

    def on_reset(self, state: RobotSystemState) -> None:
        self._frame_gate.reset(state.time_s)
        self.path = None
        self.paths.clear()
        self.errors.clear()
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._error_plot = None
        self._writers = []
        self._camera = None
        self._overlay_state.clear()
        for values in self._error_series():
            values.clear()
        self._last_due_step_index = None
        self._last_written_step_index = None
        self._last_frame = None
        self._queue = BoundedFrameQueue(maxsize=self.queue_size)
        self._stop_event.clear()
        for output_path in self.output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker = Thread(
            target=self._run_worker,
            name="continuum-sim-live-video",
            daemon=False,
        )
        self._worker.start()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        self._overlay_state.capture(
            state,
            command,
            max_points=self.backend.config.viewer.overlays.trail_max_points,
        )
        due = (
            self._frame_gate.due(state.time_s)
            if self.stride is None
            else step_index % self.stride == 0
        )
        if not due:
            return
        self._last_due_step_index = int(step_index)
        if self.layout == "scene_and_errors":
            self._append_error_sample(state, command)
        payload = (
            capture_mujoco_dynamic_state(self.backend.physics.data),
            state,
            deepcopy(self._overlay_state),
            (
                self._task_error_snapshot()
                if self.layout == "scene_and_errors"
                else None
            ),
        )
        self._queue.submit(step_index, payload)

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join()
        self._worker = None
        if self._queue.overload_count:
            self._record_error(
                "live video backpressure: rejected "
                f"{self._queue.overload_count} frame(s) because the bounded queue "
                "was full"
            )
        self.paths = [path for path in self.output_paths if path.is_file()]
        if self.paths:
            self.path = self.paths[0]
        elif self.frame_count > 0:
            self.path = self.output_path
        elif not self.errors:
            self._record_error("live video produced no frames")

    def _run_worker(self) -> None:
        try:
            self._open_resources()
            while not self._stop_event.is_set() or not self._queue.empty:
                try:
                    step_index, payload = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    self._render_payload(step_index, payload)
                finally:
                    self._queue.task_done()
            self._pad_timing_tail()
        except Exception as exc:  # noqa: BLE001 - video must not fail the run.
            self._record_error(f"live video worker failed: {type(exc).__name__}: {exc}")
        finally:
            self._close_resources()

    def _open_resources(self) -> None:
        import imageio.v2 as imageio
        import mujoco

        self._mujoco = mujoco
        model = self.backend.physics.model
        self._render_data = mujoco.MjData(model)
        self._renderer = mujoco.Renderer(
            model,
            height=self.height,
            width=self._scene_width,
        )
        if self.layout == "scene_and_errors":
            self._error_plot = _TaskErrorVideoPlot(
                width=self._error_width,
                height=self.height,
                show_force=self.show_force,
            )
        self._camera = _mujoco_render_camera(
            mujoco,
            getattr(self.backend.config.viewer, "camera", None),
        )
        for output_path in self.output_paths:
            try:
                writer = _open_video_writer(imageio, output_path, self.fps)
            except Exception as exc:  # noqa: BLE001 - keep other formats alive.
                self._record_error(
                    "live video writer setup failed for "
                    f"{output_path.name}: {type(exc).__name__}: {exc}"
                )
                continue
            self._writers.append((output_path, writer))
        if not self._writers:
            raise RuntimeError("live video setup produced no active writers")

    def _render_payload(self, step_index: int, payload) -> None:
        snapshot, state, overlay_state, task_errors = payload
        snapshot.apply_to(self._render_data)
        self._mujoco.mj_forward(self.backend.physics.model, self._render_data)
        _update_follow_camera(
            self._camera,
            self.backend.config.viewer.camera,
            state,
        )
        self._renderer.update_scene(self._render_data, camera=self._camera)
        _draw_tracking_overlay_scene(
            self._renderer.scene,
            self._mujoco,
            self.backend.config.viewer.overlays,
            overlay_state,
            state=state,
            reset_scene=False,
        )
        frame = self._renderer.render().copy()
        if self._error_plot is not None and task_errors is not None:
            error_frame = self._error_plot.render(task_errors)
            frame = np.concatenate((frame, error_frame), axis=1)
        self._append_timed_frame(step_index, frame)

    def _append_timed_frame(self, step_index: int, frame: np.ndarray) -> None:
        if (
            self.stride is not None
            and self._last_frame is not None
            and self._last_written_step_index is not None
        ):
            next_step = self._last_written_step_index + self.stride
            while next_step < step_index:
                if not self._append_frame(self._last_frame, next_step):
                    return
                self._last_written_step_index = next_step
                next_step += self.stride
        if self._append_frame(frame, step_index):
            self._last_frame = frame
            self._last_written_step_index = int(step_index)

    def _append_frame(self, frame: np.ndarray, step_index: int) -> bool:
        active_writers = []
        for output_path, writer in self._writers:
            try:
                writer.append_data(frame)
            except Exception as exc:  # noqa: BLE001 - keep other formats alive.
                self._record_error(
                    "live video frame append failed for "
                    f"{output_path.name} at step {step_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                try:
                    writer.close()
                except Exception:
                    pass
                continue
            active_writers.append((output_path, writer))
        self._writers = active_writers
        if active_writers:
            self.frame_count += 1
            return True
        return False

    def _pad_timing_tail(self) -> None:
        if (
            self.stride is None
            or self._last_frame is None
            or self._last_written_step_index is None
            or self._last_due_step_index is None
        ):
            return
        next_step = self._last_written_step_index + self.stride
        while next_step <= self._last_due_step_index:
            if not self._append_frame(self._last_frame, next_step):
                return
            self._last_written_step_index = next_step
            next_step += self.stride

    def _record_error(self, message: str) -> None:
        self.errors.append(message)
        error_path = self.output_path.parent / "video_error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if error_path.is_file():
            existing = error_path.read_text(encoding="utf-8")
        error_path.write_text(existing + message + "\n", encoding="utf-8")

    def _append_error_sample(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
    ) -> None:
        metadata = command.metadata
        self._error_time.append(float(state.time_s))
        self._tracking_error.append(
            finite_metadata_float(metadata, "executor_error_m")
        )
        self._tracking_tolerance.append(
            finite_metadata_float(metadata, "waypoint_tolerance_m")
        )
        self._target_force.append(
            finite_metadata_float(metadata, "target_normal_force_n")
        )
        self._measured_force.append(
            finite_metadata_float(
                metadata,
                "measured_normal_force_n",
                fallback_key="estimated_normal_force_n",
            )
        )
        self._force_error.append(
            finite_metadata_float(metadata, "force_error_n")
        )
        excess = len(self._error_time) - self.history_points
        if excess > 0:
            for values in self._error_series():
                del values[:excess]

    def _error_series(self) -> tuple[list[float], ...]:
        return (
            self._error_time,
            self._tracking_error,
            self._tracking_tolerance,
            self._target_force,
            self._measured_force,
            self._force_error,
        )

    def _task_error_snapshot(self) -> _TaskErrorHistorySnapshot:
        return _TaskErrorHistorySnapshot(
            time_s=np.asarray(self._error_time, dtype=float).copy(),
            tracking_error_m=np.asarray(self._tracking_error, dtype=float).copy(),
            tracking_tolerance_m=np.asarray(
                self._tracking_tolerance,
                dtype=float,
            ).copy(),
            target_force_n=np.asarray(self._target_force, dtype=float).copy(),
            measured_force_n=np.asarray(self._measured_force, dtype=float).copy(),
            force_error_n=np.asarray(self._force_error, dtype=float).copy(),
        )

    def _close_resources(self) -> None:
        writers = self._writers
        self._writers = []
        for output_path, writer in writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - report writer close errors.
                self._record_error(
                    "live video writer close failed for "
                    f"{output_path.name}: {type(exc).__name__}: {exc}"
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
        error_plot = self._error_plot
        self._error_plot = None
        if error_plot is not None:
            try:
                error_plot.close()
            except Exception as exc:  # noqa: BLE001 - report plot close errors.
                self._record_error(
                    "live video error plot close failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        self._last_frame = None


class _TaskErrorVideoPlot:
    """Render a task-error panel into an RGB frame without opening a GUI."""

    def __init__(self, *, width: int, height: int, show_force: bool) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.width = int(width)
        self.height = int(height)
        self.show_force = bool(show_force)
        dpi = 100.0
        self.figure = Figure(
            figsize=(self.width / dpi, self.height / dpi),
            dpi=dpi,
            facecolor="#f5f7fa",
        )
        self.canvas = FigureCanvasAgg(self.figure)
        row_count = 2 if self.show_force else 1
        raw_axes = self.figure.subplots(
            row_count,
            1,
            sharex=self.show_force,
        )
        axes = np.atleast_1d(raw_axes).reshape(-1)
        for axis in axes:
            axis.set_facecolor("#fbfcfe")
            axis.grid(True, color="#c7ced8", alpha=0.45, linewidth=0.7)
        self.axes = tuple(PersistentAxisArtists(axis) for axis in axes)
        self.figure.tight_layout(pad=1.4)

    def render(self, history: _TaskErrorHistorySnapshot) -> np.ndarray:
        for axis in self.axes:
            axis.begin_frame()
        if self.show_force:
            force_axis, tracking_axis = self.axes
            force_axis.plot(
                history.time_s,
                history.target_force_n,
                color="#59636f",
                linestyle="--",
                linewidth=1.2,
                label="target force",
            )
            force_axis.plot(
                history.time_s,
                history.measured_force_n,
                color="#168aad",
                linewidth=1.8,
                label="measured force",
            )
            force_axis.plot(
                history.time_s,
                history.force_error_n,
                color="#d1495b",
                linewidth=1.4,
                label="force error",
            )
            force_axis.axhline(0.0, color="#59636f", linewidth=0.8)
            force_axis.set(
                title=(
                    "Force tracking | error "
                    f"{_last_finite(history.force_error_n):.3f} N"
                ),
                ylabel="force [N]",
            )
            force_axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
        else:
            tracking_axis = self.axes[0]

        tracking_error_mm = 1000.0 * history.tracking_error_m
        tolerance_mm = 1000.0 * history.tracking_tolerance_m
        tracking_axis.plot(
            history.time_s,
            tracking_error_mm,
            color="#6f42c1",
            linewidth=2.0,
            label="TCP position error",
        )
        if np.any(np.isfinite(tolerance_mm)):
            tracking_axis.plot(
                history.time_s,
                tolerance_mm,
                color="#59636f",
                linestyle="--",
                linewidth=1.0,
                label="waypoint tolerance",
            )
        tracking_axis.set(
            title=(
                "TCP tracking error | current "
                f"{1000.0 * _last_finite(history.tracking_error_m):.2f} mm"
            ),
            xlabel="time [s]",
            ylabel="error [mm]",
        )
        tracking_axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()
            axis.end_frame()
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        self.figure.clear()


def _last_finite(values: np.ndarray, *, default: float = float("nan")) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return float(finite[-1])


__all__ = ["MujocoLiveVideoRecorderHook"]
