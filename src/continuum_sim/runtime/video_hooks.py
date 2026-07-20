"""MuJoCo video recording hooks."""

from __future__ import annotations

from pathlib import Path

from continuum_sim.runtime.mujoco_overlay_utils import (
    _TrackingOverlayState,
    _draw_tracking_overlay_scene,
    _update_follow_camera,
)
from continuum_sim.runtime.video_utils import (
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
    ) -> None:
        if fps <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook fps must be positive.")
        if stride is not None and stride <= 0:
            raise ValueError("MujocoLiveVideoRecorderHook stride must be positive.")
        self.backend = backend
        self.output_paths = _normalise_output_paths(output_path)
        self.output_path = self.output_paths[0]
        self.fps = fps
        self.stride = 1 if stride is None else stride
        self.width = width
        self.height = height
        self.path: Path | None = None
        self.paths: list[Path] = []
        self.errors: list[str] = []
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writers = []
        self._camera = None
        self._overlay_state = _TrackingOverlayState()

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self.path = None
        self.paths.clear()
        self.errors.clear()
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writers = []
        self._camera = None
        self._overlay_state.clear()
        for output_path in self.output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
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
                self._record_error("live video setup produced no active writers")
                self._close_resources()
        except Exception as exc:  # noqa: BLE001 - video must not fail the run.
            self._record_error(f"live video setup failed: {type(exc).__name__}: {exc}")
            self._close_resources()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if self._renderer is None or not self._writers or self._mujoco is None:
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
            active_writers = []
            for output_path, writer in self._writers:
                try:
                    writer.append_data(frame)
                except Exception as exc:  # noqa: BLE001 - keep other formats alive.
                    self._record_error(
                        "live video frame append failed for "
                        f"{output_path.name} at frame {self.frame_count}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    try:
                        writer.close()
                    except Exception as close_exc:  # noqa: BLE001
                        self._record_error(
                            "live video writer close failed for "
                            f"{output_path.name}: "
                            f"{type(close_exc).__name__}: {close_exc}"
                        )
                    continue
                active_writers.append((output_path, writer))
            self._writers = active_writers
            self.frame_count += 1
            if not self._writers:
                self._close_resources()
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
        self.paths = [path for path in self.output_paths if path.is_file()]
        if self.paths:
            self.path = self.paths[0]
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
