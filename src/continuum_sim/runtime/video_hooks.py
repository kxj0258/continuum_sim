"""MuJoCo video recording hooks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from queue import Empty
from threading import Event, Thread, current_thread

from continuum_sim.runtime.concurrency import TimeRateGate
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
    ) -> None:
        if fps <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook fps must be positive.")
        if stride is not None and stride <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook stride must be positive.")
        self.backend = backend
        self.output_paths = _normalise_output_paths(output_path)
        self.output_path = self.output_paths[0]
        self.fps = fps
        self.stride = None if stride is None else int(stride)
        self._frame_gate = TimeRateGate(1.0 / self.fps)
        self.width = width
        self.height = height
        self.queue_size = int(queue_size)
        self.path: Path | None = None
        self.paths: list[Path] = []
        self.errors: list[str] = []
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writers = []
        self._camera = None
        self._overlay_state = _TrackingOverlayState()
        self._queue = BoundedFrameQueue(maxsize=self.queue_size)
        self._stop_event = Event()
        self._worker: Thread | None = None

    def on_reset(self, state: RobotSystemState) -> None:
        self._frame_gate.reset(state.time_s)
        self.path = None
        self.paths.clear()
        self.errors.clear()
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writers = []
        self._camera = None
        self._overlay_state.clear()
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
        payload = (
            capture_mujoco_dynamic_state(self.backend.physics.data),
            state,
            deepcopy(self._overlay_state),
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
            width=self.width,
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
        snapshot, state, overlay_state = payload
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

    def _record_error(self, message: str) -> None:
        self.errors.append(message)
        error_path = self.output_path.parent / "video_error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if error_path.is_file():
            existing = error_path.read_text(encoding="utf-8")
        error_path.write_text(existing + message + "\n", encoding="utf-8")

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


__all__ = ["MujocoLiveVideoRecorderHook"]
