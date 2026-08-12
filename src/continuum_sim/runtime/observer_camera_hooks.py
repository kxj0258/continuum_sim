"""Observer-mounted camera feedback hooks."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from queue import Empty
from threading import Event, Thread, current_thread

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.concurrency import LatestValueSlot, TimeRateGate
from continuum_sim.runtime.hook_utils import metadata_point as _metadata_point
from continuum_sim.runtime.mujoco_state_copy import (
    capture_mujoco_dynamic_state,
    copy_mujoco_dynamic_state,
)
from continuum_sim.runtime.video_utils import (
    BoundedFrameQueue,
    normalise_output_paths as _normalise_output_paths,
    open_video_writer as _open_video_writer,
)
from continuum_sim.sensing.camera_model import CameraIntrinsicsConfig
from continuum_sim.sensing.visual_feedback import (
    VisualServoFeedback,
    project_roi_to_camera_feedback,
)
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class _ObserverRenderWorker:
    """Own observer renderer/writers and drain a bounded snapshot queue."""

    def __init__(self, owner, *, queue_size: int = 8) -> None:
        self.owner = owner
        self.queue = BoundedFrameQueue(maxsize=queue_size)
        self.frames = LatestValueSlot(None)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self.failure: BaseException | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            name="continuum-sim-observer-render",
            daemon=False,
        )
        self._thread.start()

    def submit(
        self,
        *,
        state: RobotSystemState,
        feedback: VisualServoFeedback | None,
        display: bool,
        record: bool,
        sequence: int,
    ) -> bool:
        payload = (
            capture_mujoco_dynamic_state(self.owner.backend.physics.data),
            state,
            feedback,
            bool(display),
            bool(record),
        )
        return self.queue.submit(sequence, payload)

    def consume_frame_after(self, version: int):
        return self.frames.consume_after(version)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        self._thread = None

    def _run(self) -> None:
        owner = self.owner
        state = None
        try:
            owner._open_render_resources()
            while not self._stop_event.is_set() or not self.queue.empty:
                try:
                    _sequence, payload = self.queue.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    snapshot, state, feedback, display, record = payload
                    snapshot.apply_to(owner._render_data)
                    owner._mujoco.mj_forward(
                        owner.backend.physics.model,
                        owner._render_data,
                    )
                    frame = owner._render_frame(owner._render_data)
                    frame = owner._draw_roi_overlay(frame, feedback=feedback)
                    if display:
                        self.frames.publish(frame)
                    if record:
                        owner._append_video_frame(frame)
                finally:
                    self.queue.task_done()
        except BaseException as exc:  # noqa: BLE001 - isolated presentation failure.
            self.failure = exc
            owner._record_error(
                f"observer camera render worker failed: {type(exc).__name__}: {exc}"
            )
        finally:
            owner._close_video_writers()
            owner._close_renderer()


class MujocoObserverCameraFeedbackHook:
    """Project engine ROI into an observer-mounted MuJoCo camera and show frames."""

    def __init__(
        self,
        backend: object,
        *,
        camera_name: str,
        intrinsics: CameraIntrinsicsConfig,
        fallback_target_world: np.ndarray | None = None,
        show_window: bool = True,
        stride: int = 1,
        display_interval_s: float | None = None,
        video_output_paths: (
            str | Path | list[str | Path] | tuple[str | Path, ...] | None
        ) = None,
        video_fps: int = 20,
        video_stride: int | None = None,
        data_lock=None,
        runtime_timing=None,
        async_render: bool = False,
        render_queue_size: int = 8,
    ) -> None:
        if stride <= 0:
            raise ValueError("MujocoObserverCameraFeedbackHook stride must be positive.")
        if display_interval_s is not None and display_interval_s <= 0.0:
            raise ValueError(
                "MujocoObserverCameraFeedbackHook display_interval_s must be positive."
            )
        if video_fps <= 0:
            raise ValueError("MujocoObserverCameraFeedbackHook video_fps must be positive.")
        if video_stride is not None and video_stride <= 0:
            raise ValueError(
                "MujocoObserverCameraFeedbackHook video_stride must be positive."
            )
        self.backend = backend
        self.camera_name = str(camera_name)
        self.intrinsics = intrinsics
        self.show_window = bool(show_window)
        self.stride = int(stride)
        self.display_interval_s = (
            None if display_interval_s is None else float(display_interval_s)
        )
        self.output_paths = (
            ()
            if video_output_paths is None
            else _normalise_output_paths(video_output_paths)
        )
        self.output_path = self.output_paths[0] if self.output_paths else None
        self.video_fps = int(video_fps)
        self.video_stride = None if video_stride is None else int(video_stride)
        self._video_gate = TimeRateGate(1.0 / self.video_fps)
        self.runtime_timing = runtime_timing
        self.data_lock = data_lock
        self.async_render = bool(async_render)
        self.render_queue_size = int(render_queue_size)
        if self.render_queue_size <= 0:
            raise ValueError("render_queue_size must be positive.")
        self.requires_gui_main_thread = self.show_window and self.async_render
        self._target_world = (
            None
            if fallback_target_world is None
            else np.asarray(fallback_target_world, dtype=float).copy()
        )
        if self._target_world is not None and self._target_world.shape != (3,):
            raise ValueError("fallback_target_world must have shape (3,).")
        self._mujoco = None
        self._renderer = None
        self._render_data = None
        self._cv2 = None
        self._plt = None
        self._figure = None
        self._axis = None
        self._image_artist = None
        self._frame_index = 0
        self._next_display_time_s: float | None = None
        self._last_display_check_time_s: float | None = None
        self._recording_started = False
        self._latest_step_index = -1
        self.frame_count = 0
        self.path: Path | None = None
        self.paths: list[Path] = []
        self._writers = []
        self._render_worker = None
        self._presented_frame_version = -1
        self.last_feedback: VisualServoFeedback | None = None
        self.errors: list[str] = []

    def on_reset(self, state: RobotSystemState) -> None:
        self._video_gate.reset(state.time_s)
        self._frame_index = 0
        self._next_display_time_s = None
        self._last_display_check_time_s = None
        self._recording_started = False
        self._latest_step_index = -1
        self.frame_count = 0
        self.path = None
        self.paths.clear()
        self._writers = []
        self._presented_frame_version = -1
        self.last_feedback = None
        self.errors.clear()
        if self._render_worker is not None:
            self._render_worker.stop()
            self._render_worker = None
        self._close_renderer()
        for output_path in self.output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import mujoco

            self._mujoco = mujoco
            if self.async_render:
                self._render_worker = _ObserverRenderWorker(
                    self,
                    queue_size=self.render_queue_size,
                )
                self._render_worker.start()
                return
            self._open_render_resources()
        except Exception as exc:  # noqa: BLE001 - camera feedback must not stop runs.
            self._record_error(
                f"observer camera setup failed: {type(exc).__name__}: {exc}"
            )
            return
    def _open_render_resources(self) -> None:
        if self._mujoco is None:
            import mujoco

            self._mujoco = mujoco
        self._render_data = (
            self._mujoco.MjData(self.backend.physics.model)
            if self.async_render or self.data_lock is not None
            else None
        )
        self._renderer = self._mujoco.Renderer(
            self.backend.physics.model,
            height=self.intrinsics.height,
            width=self.intrinsics.width,
        )
        if not self.output_paths:
            return
        try:
            import imageio.v2 as imageio
        except Exception as exc:  # noqa: BLE001 - window feedback can still work.
            self._record_error(
                f"observer camera video setup failed: {type(exc).__name__}: {exc}"
            )
            return
        for output_path in self.output_paths:
            try:
                writer = _open_video_writer(imageio, output_path, self.video_fps)
            except Exception as exc:  # noqa: BLE001 - keep other formats alive.
                self._record_error(
                    "observer camera video writer setup failed for "
                    f"{output_path.name}: {type(exc).__name__}: {exc}"
                )
                continue
            self._writers.append((output_path, writer))
        if self.output_paths and not self._writers:
            self._record_error("observer camera video setup produced no active writers")

    def enrich_state(self, state: RobotSystemState) -> RobotSystemState:
        target = self._target_world
        if self._mujoco is None:
            return state
        try:
            display_due = self.show_window and self._display_due(state.time_s)
            record_due = self._recording_started and (
                bool(self.output_paths) if self.async_render else bool(self._writers)
            ) and (
                self._video_gate.due(state.time_s)
                if self.video_stride is None
                else self._latest_step_index % self.video_stride == 0
            )
            model = self.backend.physics.model
            data = (
                self.backend.physics.data
                if self.async_render
                else self._camera_data_snapshot()
            )
            if (display_due or record_due) and not self.async_render:
                with (
                    nullcontext()
                    if self.runtime_timing is None
                    else self.runtime_timing.measure("camera.forward")
                ):
                    self._mujoco.mj_forward(model, data)
            camera_pose = self._camera_pose_world(model, data)
            metadata = {
                **state.metadata,
                "visual_servo_camera_name": self.camera_name,
                "visual_servo_camera_position_world": (
                    camera_pose.position.copy()
                ),
                "visual_servo_camera_quat_world_wxyz": camera_pose.quat.copy(),
            }
            if target is not None:
                feedback = project_roi_to_camera_feedback(
                    target,
                    camera_pose,
                    self.intrinsics,
                    timestamp_s=state.time_s,
                )
                self.last_feedback = feedback
                metadata.update(feedback.as_metadata())
            if self.errors:
                metadata["visual_servo_camera_errors"] = tuple(self.errors)
            if self.async_render:
                worker = self._render_worker
                if worker is not None and (display_due or record_due):
                    worker.submit(
                        state=state,
                        feedback=self.last_feedback,
                        display=display_due,
                        record=record_due,
                        sequence=self._frame_index,
                    )
                self._frame_index += 1
                return replace(state, metadata=metadata)
            frame = None
            if display_due or record_due:
                with (
                    nullcontext()
                    if self.runtime_timing is None
                    else self.runtime_timing.measure("camera.render")
                ):
                    frame = self._render_frame(data)
                    frame = self._draw_roi_overlay(frame)
            if display_due and frame is not None:
                with (
                    nullcontext()
                    if self.runtime_timing is None
                    else self.runtime_timing.measure("camera.present")
                ):
                    self._show_frame(frame)
            if record_due and frame is not None:
                self._append_video_frame(frame)
            self._frame_index += 1
            return replace(state, metadata=metadata)
        except Exception as exc:  # noqa: BLE001 - keep simulation alive.
            self._record_error(
                f"observer camera feedback failed: {type(exc).__name__}: {exc}"
            )
            return replace(
                state,
                metadata={
                    **state.metadata,
                    "visual_servo_target_visible": False,
                    "visual_servo_camera_name": self.camera_name,
                    "visual_servo_camera_errors": tuple(self.errors),
                },
            )

    def _camera_data_snapshot(self):
        source = self.backend.physics.data
        if self._render_data is None:
            return source
        lock = self.data_lock
        with lock:
            destination = self._render_data
            copy_mujoco_dynamic_state(source, destination)
        return destination

    def render_frame(self, state: RobotSystemState) -> np.ndarray:
        """Render one copied state without invoking any GUI presentation API."""

        if self._mujoco is None or self._renderer is None:
            raise RuntimeError("observer camera renderer is not available.")
        model = self.backend.physics.model
        data = self._camera_data_snapshot()
        with (
            nullcontext()
            if self.runtime_timing is None
            else self.runtime_timing.measure("camera.forward")
        ):
            self._mujoco.mj_forward(model, data)
        target = self._target_world
        if target is not None:
            camera_pose = self._camera_pose_world(model, data)
            self.last_feedback = project_roi_to_camera_feedback(
                target,
                camera_pose,
                self.intrinsics,
                timestamp_s=state.time_s,
            )
        with (
            nullcontext()
            if self.runtime_timing is None
            else self.runtime_timing.measure("camera.render")
        ):
            return self._draw_roi_overlay(self._render_frame(data))

    def _display_due(self, time_s: float) -> bool:
        if self.display_interval_s is None:
            return self._frame_index % self.stride == 0
        now = float(time_s)
        tolerance = 1.0e-12
        if (
            self._last_display_check_time_s is not None
            and now + tolerance < self._last_display_check_time_s
        ):
            self._next_display_time_s = None
        self._last_display_check_time_s = now
        if self._next_display_time_s is None:
            self._next_display_time_s = now + self.display_interval_s
            return True
        if now + tolerance < self._next_display_time_s:
            return False
        intervals = max(
            1,
            int(
                (now + tolerance - self._next_display_time_s)
                // self.display_interval_s
            )
            + 1,
        )
        self._next_display_time_s += intervals * self.display_interval_s
        return True

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state
        self._recording_started = True
        self._latest_step_index = int(step_index)
        for key in (
            "engine_navigation_observer_roi_m",
            "visual_servo_roi_world",
            "observer_target_position_world",
            "executor_target_world",
        ):
            target = _metadata_point(command.metadata, key)
            if target is not None:
                self._target_world = target
                return

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._render_worker is not None:
            worker = self._render_worker
            worker.stop()
            if worker.queue.overload_count:
                self._record_error(
                    "observer camera backpressure: rejected "
                    f"{worker.queue.overload_count} frame(s) because the bounded "
                    "render queue was full"
                )
            if not self.requires_gui_main_thread:
                self._render_worker = None
        else:
            self._close_video_writers()
            self._close_renderer()
        if not self.requires_gui_main_thread:
            self._close_window()
        self.paths = [path for path in self.output_paths if path.is_file()]
        if self.paths:
            self.path = self.paths[0]
        elif self.frame_count > 0 and self.output_path is not None:
            self.path = self.output_path
        elif self.output_paths and not self.errors:
            self._record_error("observer camera video produced no frames")

    def present_pending(self, *, force: bool = False) -> None:
        del force
        worker = self._render_worker
        if worker is None:
            return
        item = worker.consume_frame_after(self._presented_frame_version)
        if item is None:
            return
        frame, self._presented_frame_version = item
        if frame is not None:
            self._show_frame(frame)

    def close_presentation(self) -> None:
        self._close_window()
        self._render_worker = None

    def _camera_pose_world(self, model, data) -> Pose6D:
        camera_id = self._mujoco.mj_name2id(
            model,
            self._mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name,
        )
        if camera_id < 0:
            raise ValueError(f"MuJoCo model is missing camera {self.camera_name!r}.")
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = np.asarray(data.cam_xmat[int(camera_id)], dtype=float).reshape(3, 3)
        pose[:3, 3] = np.asarray(data.cam_xpos[int(camera_id)], dtype=float)
        return Pose6D.from_matrix(pose)

    def _render_window(self, data) -> None:
        frame = self._render_frame(data)
        self._show_frame(frame)

    def _render_frame(self, data) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("observer camera renderer is not available.")
        self._renderer.update_scene(data, camera=self.camera_name)
        scene = getattr(self._renderer, "scene", None)
        flags = getattr(scene, "flags", None)
        rnd_flags = getattr(self._mujoco, "mjtRndFlag", None)
        cull_face = getattr(rnd_flags, "mjRND_CULL_FACE", None)
        if flags is not None and cull_face is not None:
            try:
                flags[int(cull_face)] = 0
            except (IndexError, TypeError):
                pass
        return self._renderer.render().copy()

    def _show_frame(self, frame: np.ndarray) -> None:
        if self._show_with_cv2(frame):
            return
        self._show_with_matplotlib(frame)

    def _draw_roi_overlay(
        self,
        frame: np.ndarray,
        *,
        feedback: VisualServoFeedback | None = None,
    ) -> np.ndarray:
        feedback = self.last_feedback if feedback is None else feedback
        if feedback is None:
            return frame
        pixel_error = np.asarray(feedback.pixel_error_px, dtype=float)
        if pixel_error.shape != (2,) or not np.all(np.isfinite(pixel_error)):
            return frame
        height, width = frame.shape[:2]
        center = np.array(
            [(float(width) - 1.0) / 2.0, (float(height) - 1.0) / 2.0],
            dtype=float,
        )
        pixel = np.rint(center + pixel_error).astype(int)
        if feedback.target_visible:
            color = np.array([255, 220, 0], dtype=frame.dtype)
        else:
            color = np.array([255, 120, 0], dtype=frame.dtype)
        x = int(np.clip(pixel[0], 0, width - 1))
        y = int(np.clip(pixel[1], 0, height - 1))
        radius = max(4, min(width, height) // 32)
        thickness = 1
        for offset in range(-radius, radius + 1):
            xx = x + offset
            yy = y + offset
            if 0 <= xx < width:
                frame[
                    max(0, y - thickness) : min(height, y + thickness + 1),
                    xx,
                ] = color
            if 0 <= yy < height:
                frame[
                    yy,
                    max(0, x - thickness) : min(width, x + thickness + 1),
                ] = color
        box_radius = radius + 2
        x0 = max(0, x - box_radius)
        x1 = min(width - 1, x + box_radius)
        y0 = max(0, y - box_radius)
        y1 = min(height - 1, y + box_radius)
        frame[y0 : y0 + 1, x0 : x1 + 1] = color
        frame[y1 : y1 + 1, x0 : x1 + 1] = color
        frame[y0 : y1 + 1, x0 : x0 + 1] = color
        frame[y0 : y1 + 1, x1 : x1 + 1] = color
        return frame

    def _append_video_frame(self, frame: np.ndarray) -> None:
        active_writers = []
        appended = False
        for output_path, writer in self._writers:
            try:
                writer.append_data(frame)
            except Exception as exc:  # noqa: BLE001 - keep other formats alive.
                self._record_error(
                    "observer camera video frame append failed for "
                    f"{output_path.name} at frame {self.frame_count}: "
                    f"{type(exc).__name__}: {exc}"
                )
                try:
                    writer.close()
                except Exception as close_exc:  # noqa: BLE001
                    self._record_error(
                        "observer camera video writer close failed for "
                        f"{output_path.name}: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )
                continue
            appended = True
            active_writers.append((output_path, writer))
        self._writers = active_writers
        if appended:
            self.frame_count += 1

    def _show_with_cv2(self, frame: np.ndarray) -> bool:
        if self._cv2 is False:
            return False
        if self._cv2 is None:
            try:
                import cv2

                self._cv2 = cv2
                cv2.namedWindow(self.camera_name, cv2.WINDOW_NORMAL)
            except Exception as exc:  # noqa: BLE001 - fall back to matplotlib.
                self._cv2 = False
                self._record_error(
                    f"opencv observer camera window unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                return False
        self._cv2.imshow(self.camera_name, frame[:, :, ::-1])
        self._cv2.waitKey(1)
        return True

    def _show_with_matplotlib(self, frame: np.ndarray) -> None:
        if self._plt is None:
            try:
                import matplotlib.pyplot as plt

                self._plt = plt
                plt.ion()
                self._figure, self._axis = plt.subplots()
                self._figure.canvas.manager.set_window_title(self.camera_name)
            except Exception as exc:  # noqa: BLE001 - record once and disable.
                self.show_window = False
                self._record_error(
                    f"matplotlib observer camera window unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                return
        if self._image_artist is None:
            self._image_artist = self._axis.imshow(frame)
            self._axis.set_axis_off()
        else:
            self._image_artist.set_data(frame)
        self._figure.canvas.draw_idle()

    def _close_renderer(self) -> None:
        renderer = self._renderer
        self._renderer = None
        self._render_data = None
        close = getattr(renderer, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                self._record_error(
                    f"observer camera renderer close failed: {type(exc).__name__}: {exc}"
                )

    def _close_video_writers(self) -> None:
        writers = self._writers
        self._writers = []
        for output_path, writer in writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                self._record_error(
                    "observer camera video writer close failed for "
                    f"{output_path.name}: {type(exc).__name__}: {exc}"
                )

    def _close_window(self) -> None:
        if self._cv2 not in (None, False):
            try:
                self._cv2.destroyWindow(self.camera_name)
            except Exception:
                pass
        if self._plt is not None and self._figure is not None:
            try:
                self._plt.close(self._figure)
            except Exception:
                pass
        self._image_artist = None

    def _record_error(self, message: str) -> None:
        if not self.errors or self.errors[-1] != message:
            self.errors.append(message)


__all__ = ["MujocoObserverCameraFeedbackHook"]
