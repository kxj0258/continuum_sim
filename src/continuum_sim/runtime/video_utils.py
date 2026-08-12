"""Shared helpers for runtime video hooks."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Full, Queue
from typing import Generic, TypeVar


T = TypeVar("T")


class BoundedFrameQueue(Generic[T]):
    """Ordered bounded queue with explicit overload accounting."""

    def __init__(self, *, maxsize: int = 8) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive.")
        self._queue: Queue[tuple[int, T]] = Queue(maxsize=maxsize)
        self.overload_count = 0

    def submit(self, sequence: int, payload: T) -> bool:
        try:
            self._queue.put_nowait((int(sequence), payload))
        except Full:
            self.overload_count += 1
            return False
        return True

    def get(self, timeout: float | None = None) -> tuple[int, T]:
        if timeout is None:
            return self._queue.get()
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> tuple[int, T]:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    @property
    def empty(self) -> bool:
        return self._queue.empty()


def open_video_writer(imageio, path: Path, fps: int):
    """Open an imageio writer with the project's GIF/MP4 defaults."""

    if path.suffix.lower() == ".gif":
        return imageio.get_writer(path, mode="I", duration=1000.0 / fps, loop=0)
    return imageio.get_writer(path, fps=fps, macro_block_size=1)


def normalise_output_paths(
    output_path: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> tuple[Path, ...]:
    """Normalize one or many video output paths."""

    if isinstance(output_path, (list, tuple)):
        paths = tuple(Path(path) for path in output_path)
    else:
        paths = (Path(output_path),)
    if not paths:
        raise ValueError("Video recorder requires at least one output path.")
    return paths


def mujoco_render_camera(mujoco, camera: object | None):
    """Build a MuJoCo render camera from the scenario camera config."""

    if camera is None:
        return -1
    render_camera = mujoco.MjvCamera()
    render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    render_camera.lookat[:] = getattr(camera, "lookat")
    render_camera.distance = float(getattr(camera, "distance"))
    render_camera.azimuth = float(getattr(camera, "azimuth"))
    render_camera.elevation = float(getattr(camera, "elevation"))
    return render_camera
