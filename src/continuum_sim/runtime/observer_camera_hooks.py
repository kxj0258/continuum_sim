"""Observer-mounted camera feedback hooks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.hook_utils import metadata_point as _metadata_point
from continuum_sim.runtime.video_utils import (
    normalise_output_paths as _normalise_output_paths,
    open_video_writer as _open_video_writer,
)
from continuum_sim.sensing.camera_model import CameraIntrinsicsConfig
from continuum_sim.sensing.visual_feedback import (
    VisualServoFeedback,
    project_roi_to_camera_feedback,
)
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


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
        video_output_paths: (
            str | Path | list[str | Path] | tuple[str | Path, ...] | None
        ) = None,
        video_fps: int = 20,
        video_stride: int | None = None,
    ) -> None:
        if stride <= 0:
            raise ValueError("MujocoObserverCameraFeedbackHook stride must be positive.")
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
        self.output_paths = (
            ()
            if video_output_paths is None
            else _normalise_output_paths(video_output_paths)
        )
        self.output_path = self.output_paths[0] if self.output_paths else None
        self.video_fps = int(video_fps)
        self.video_stride = 1 if video_stride is None else int(video_stride)
        self._target_world = (
            None
            if fallback_target_world is None
            else np.asarray(fallback_target_world, dtype=float).copy()
        )
        if self._target_world is not None and self._target_world.shape != (3,):
            raise ValueError("fallback_target_world must have shape (3,).")
        self._mujoco = None
        self._renderer = None
        self._cv2 = None
        self._plt = None
        self._figure = None
        self._axis = None
        self._frame_index = 0
        self._recording_started = False
        self._latest_step_index = -1
        self.frame_count = 0
        self.path: Path | None = None
        self.paths: list[Path] = []
        self._writers = []
        self.last_feedback: VisualServoFeedback | None = None
        self.errors: list[str] = []

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self._frame_index = 0
        self._recording_started = False
        self._latest_step_index = -1
        self.frame_count = 0
        self.path = None
        self.paths.clear()
        self._writers = []
        self.last_feedback = None
        self.errors.clear()
        self._close_renderer()
        for output_path in self.output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import mujoco

            self._mujoco = mujoco
            self._renderer = mujoco.Renderer(
                self.backend.physics.model,
                height=self.intrinsics.height,
                width=self.intrinsics.width,
            )
        except Exception as exc:  # noqa: BLE001 - camera feedback must not stop runs.
            self._record_error(
                f"observer camera setup failed: {type(exc).__name__}: {exc}"
            )
            return
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
            model = self.backend.physics.model
            data = self.backend.physics.data
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
            display_due = self.show_window and self._frame_index % self.stride == 0
            record_due = self._recording_started and bool(self._writers) and (
                self._latest_step_index % self.video_stride == 0
            )
            if display_due or record_due:
                frame = self._render_frame(data)
                frame = self._draw_roi_overlay(frame)
                if display_due:
                    self._show_frame(frame)
                if record_due:
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
        self._close_video_writers()
        self._close_renderer()
        self._close_window()
        self.paths = [path for path in self.output_paths if path.is_file()]
        if self.paths:
            self.path = self.paths[0]
        elif self.frame_count > 0 and self.output_path is not None:
            self.path = self.output_path
        elif self.output_paths and not self.errors:
            self._record_error("observer camera video produced no frames")

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
        return self._renderer.render().copy()

    def _show_frame(self, frame: np.ndarray) -> None:
        if self._show_with_cv2(frame):
            return
        self._show_with_matplotlib(frame)

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        feedback = self.last_feedback
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
        self._axis.clear()
        self._axis.imshow(frame)
        self._axis.set_axis_off()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def _close_renderer(self) -> None:
        renderer = self._renderer
        self._renderer = None
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

    def _record_error(self, message: str) -> None:
        if not self.errors or self.errors[-1] != message:
            self.errors.append(message)


__all__ = ["MujocoObserverCameraFeedbackHook"]
