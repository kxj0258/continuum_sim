"""Shared helpers for runtime video hooks."""

from __future__ import annotations

from pathlib import Path


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
