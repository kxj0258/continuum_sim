"""Replay video export for saved rollout results."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np


def save_replay_video(
    result: object,
    output_path: str | Path,
    *,
    enabled: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    stride: int | None = None,
    camera: object | None = None,
) -> Path | None:
    """Save a replay animation, preferring MuJoCo offscreen replay when possible."""

    if _is_mujoco_result(result):
        path = save_mujoco_replay_video(
            result,
            output_path,
            enabled=enabled,
            width=width,
            height=height,
            fps=fps,
            stride=stride,
            camera=camera,
        )
        if path is not None or not enabled or not (
            hasattr(result, "target_position") and _has_tip_position(result)
        ):
            return path
        fallback_path = save_matplotlib_replay_video(
            result,
            output_path,
            enabled=enabled,
            fps=fps,
            stride=stride,
        )
        if fallback_path is not None:
            _annotate_video_fallback(Path(output_path), fallback_path)
        return fallback_path
    return save_matplotlib_replay_video(
        result,
        output_path,
        enabled=enabled,
        fps=fps,
        stride=stride,
    )


def save_mujoco_replay_video(
    result: object,
    output_path: str | Path,
    *,
    enabled: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    stride: int | None = None,
    camera: object | None = None,
) -> Path | None:
    """Save a replay animation from recorded MuJoCo qpos/qvel, or write an error note."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not enabled:
        _write_video_error(path, "video export disabled")
        return None
    if not _is_mujoco_result(result):
        _write_video_error(path, "result does not contain scene_xml_path, qpos, and qvel")
        return None

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        _write_video_error(path, f"imageio is not installed: {exc}")
        return None

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        _write_video_error(path, f"MuJoCo is not installed: {exc}")
        return None

    qpos = np.asarray(getattr(result, "qpos"), dtype=float)
    qvel = np.asarray(getattr(result, "qvel"), dtype=float)
    mocap_pos = _optional_history(result, "mocap_pos")
    mocap_quat = _optional_history(result, "mocap_quat")
    scene_xml_path = Path(getattr(result, "scene_xml_path")).resolve()
    if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] == 0:
        _write_video_error(path, f"invalid qpos/qvel history shapes: {qpos.shape}, {qvel.shape}")
        return None
    if not scene_xml_path.exists():
        _write_video_error(path, f"scene XML does not exist: {scene_xml_path}")
        return None

    frame_indices = _video_frame_indices(qpos.shape[0], fps=fps, stride=stride)
    temp_dir = None
    renderer = None
    stage = "loading scene XML"
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        stage = "checking offscreen framebuffer"
        offscreen_width, offscreen_height = _model_offscreen_size(model)
        if offscreen_width < width or offscreen_height < height:
            temp_dir = TemporaryDirectory()
            stage = "patching offscreen framebuffer"
            patched_scene_xml = _patched_offscreen_xml(
                scene_xml_path,
                width=width,
                height=height,
                output_dir=Path(temp_dir.name),
            )
            stage = "loading patched scene XML"
            model = mujoco.MjModel.from_xml_path(str(patched_scene_xml))
        stage = "creating MuJoCo data"
        data = mujoco.MjData(model)
        stage = "creating MuJoCo renderer"
        renderer = mujoco.Renderer(model, height=height, width=width)
        stage = "configuring MuJoCo camera"
        render_camera = _mujoco_render_camera(mujoco, camera)
        stage = "opening video writer"
        with _open_video_writer(imageio, path, fps) as writer:
            for frame_number, index in enumerate(frame_indices):
                try:
                    stage = f"rendering frame {frame_number} (sample {index})"
                    _restore_state(model, data, qpos[index], qvel[index])
                    _restore_mocap_state(data, mocap_pos, mocap_quat, index)
                    mujoco.mj_forward(model, data)
                    _update_follow_camera(
                        mujoco,
                        model,
                        data,
                        render_camera,
                        camera,
                        result,
                        index,
                    )
                    renderer.update_scene(data, camera=render_camera)
                    frame = renderer.render().copy()
                except Exception as exc:  # noqa: BLE001 - include frame context in artifact error.
                    raise RuntimeError(
                        f"render failed at frame {frame_number} (sample {index}): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                try:
                    stage = f"encoding frame {frame_number} (sample {index})"
                    writer.append_data(frame)
                except Exception as exc:  # noqa: BLE001 - include frame context in artifact error.
                    raise RuntimeError(
                        f"encode failed at frame {frame_number} (sample {index}): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
    except Exception as exc:  # noqa: BLE001 - artifact export must not fail the CLI run.
        _write_video_error(
            path,
            f"video export failed during {stage}: {type(exc).__name__}: {exc}",
        )
        return None
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
        if temp_dir is not None:
            temp_dir.cleanup()
    return path


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
    mujoco,
    model,
    data,
    render_camera: object | None,
    camera: object | None,
    result: object,
    index: int,
) -> None:
    if render_camera in (None, -1) or camera is None:
        return
    target = _follow_camera_target(mujoco, model, data, camera, result, index)
    if target is not None:
        render_camera.lookat[:] = target


def _follow_camera_target(
    mujoco,
    model,
    data,
    camera: object,
    result: object,
    index: int,
) -> np.ndarray | None:
    follow = str(getattr(camera, "follow", "none"))
    if follow == "base":
        return _base_follow_target(mujoco, model, data, result, index)
    if follow == "executor_tip":
        return _history_point(result, "tip_position", index)
    return None


def _base_follow_target(
    mujoco,
    model,
    data,
    result: object,
    index: int,
) -> np.ndarray | None:
    target = _history_point(result, "base_position_m", index)
    if target is not None:
        return target
    target = _history_point(result, "mujoco_mobile_base_frame_pose", index)
    if target is not None:
        return target
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "mobile_base_freejoint",
    )
    if joint_id >= 0:
        qpos_addr = int(model.jnt_qposadr[int(joint_id)])
        return np.asarray(data.qpos[qpos_addr : qpos_addr + 3], dtype=float).copy()
    for site_name in ("mobile_base_frame", "executor_base_site", "base_site"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            return np.asarray(data.site_xpos[int(site_id)], dtype=float).copy()
    return None


def _history_point(result: object, name: str, index: int) -> np.ndarray | None:
    if not hasattr(result, name):
        return None
    values = np.asarray(getattr(result, name), dtype=float)
    if values.ndim == 2 and values.shape[1] == 3 and values.shape[0] > 0:
        point = values[min(index, values.shape[0] - 1)]
    elif values.ndim == 3 and values.shape[1:] == (4, 4) and values.shape[0] > 0:
        point = values[min(index, values.shape[0] - 1), :3, 3]
    else:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return np.asarray(point, dtype=float).copy()


def _model_offscreen_size(model) -> tuple[int, int]:
    global_visual = getattr(getattr(model, "vis", None), "global_", None)
    if global_visual is None:
        return (0, 0)
    return (
        int(getattr(global_visual, "offwidth", 0)),
        int(getattr(global_visual, "offheight", 0)),
    )


def _open_video_writer(imageio, path: Path, fps: int):
    if path.suffix.lower() == ".gif":
        return imageio.get_writer(path, mode="I", duration=1000.0 / fps, loop=0)
    return imageio.get_writer(path, fps=fps)


def _patched_offscreen_xml(
    scene_xml_path: Path,
    *,
    width: int,
    height: int,
    output_dir: Path,
) -> Path:
    tree = ElementTree.parse(scene_xml_path)
    root = tree.getroot()
    _rebase_asset_file_paths(root, scene_xml_path.parent, output_dir)
    visual = root.find("visual")
    if visual is None:
        visual = ElementTree.Element("visual")
        insert_index = 1 if root.find("option") is not None else 0
        root.insert(insert_index, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ElementTree.Element("global")
        visual.insert(0, global_visual)
    current_width = int(float(global_visual.get("offwidth", "0")))
    current_height = int(float(global_visual.get("offheight", "0")))
    global_visual.set("offwidth", str(max(current_width, int(width))))
    global_visual.set("offheight", str(max(current_height, int(height))))
    patched_path = output_dir / scene_xml_path.name
    tree.write(patched_path, encoding="utf-8", xml_declaration=False)
    return patched_path


def _rebase_asset_file_paths(
    root: ElementTree.Element,
    base_xml_dir: Path,
    output_xml_dir: Path,
) -> None:
    asset = root.find("asset")
    if asset is None:
        return
    for element in asset.iter():
        raw_file = element.attrib.get("file")
        if not raw_file:
            continue
        raw_path = Path(raw_file)
        if raw_path.is_absolute():
            continue
        source_path = (base_xml_dir / raw_path).resolve()
        element.set(
            "file",
            Path(os.path.relpath(source_path, output_xml_dir.resolve())).as_posix(),
        )


def save_matplotlib_replay_video(
    result: object,
    output_path: str | Path,
    *,
    enabled: bool = True,
    fps: int = 30,
    stride: int | None = None,
) -> Path | None:
    """Save a generic target/tip trajectory animation for non-MuJoCo results."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not enabled:
        _write_video_error(path, "video export disabled")
        return None
    if not (hasattr(result, "target_position") and _has_tip_position(result)):
        _write_video_error(path, "result does not contain target/tip position history")
        return None
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        _write_video_error(path, f"video plotting dependency is not installed: {exc}")
        return None

    try:
        target = np.asarray(getattr(result, "target_position"), dtype=float)
        tip = _tip_position(result)
        if target.ndim != 2 or target.shape[1] != 3 or tip.ndim != 2 or tip.shape[1] != 3:
            raise ValueError(f"invalid target/tip shapes: {target.shape}, {tip.shape}")
        sample_count = min(target.shape[0], tip.shape[0])
        frame_indices = _video_frame_indices(sample_count, fps=fps, stride=stride)
        bounds = _axis_bounds(np.vstack((target[:sample_count], tip[:sample_count])))

        fig = plt.figure(figsize=(7.0, 7.0), dpi=120)
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        set_proj_type = getattr(axis, "set_proj_type", None)
        if set_proj_type is not None:
            set_proj_type("ortho")
        with _open_video_writer(imageio, path, fps) as writer:
            for index in frame_indices:
                axis.clear()
                axis.plot(
                    target[: index + 1, 0],
                    target[: index + 1, 1],
                    target[: index + 1, 2],
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.4,
                    label="target",
                )
                axis.plot(
                    tip[: index + 1, 0],
                    tip[: index + 1, 1],
                    tip[: index + 1, 2],
                    color="tab:blue",
                    linewidth=1.6,
                    label="tip",
                )
                axis.scatter(*target[index], color="tab:orange", s=24)
                axis.scatter(*tip[index], color="tab:blue", s=28)
                _apply_axis_bounds(axis, bounds)
                axis.set_title(f"Replay t={_time_value(result, index):.2f} s")
                axis.set_xlabel("x [m]")
                axis.set_ylabel("y [m]")
                axis.set_zlabel("z [m]")
                axis.legend(loc="upper left")
                axis.grid(True, alpha=0.35)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                writer.append_data(rgba[:, :, :3].copy())
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - export should not fail the CLI run.
        _write_video_error(path, f"video export failed: {type(exc).__name__}: {exc}")
        return None
    return path


def _is_mujoco_result(result: object) -> bool:
    return (
        hasattr(result, "scene_xml_path")
        and hasattr(result, "qpos")
        and hasattr(result, "qvel")
    )


def _has_tip_position(result: object) -> bool:
    return hasattr(result, "tip_position") or hasattr(result, "tip_pose")


def _tip_position(result: object) -> np.ndarray:
    if hasattr(result, "tip_position"):
        return np.asarray(getattr(result, "tip_position"), dtype=float)
    tip_pose = np.asarray(getattr(result, "tip_pose"), dtype=float)
    if tip_pose.ndim != 3 or tip_pose.shape[1:] != (4, 4):
        raise ValueError(f"Expected tip_pose with shape (N, 4, 4), got {tip_pose.shape}.")
    return tip_pose[:, :3, 3]


def _restore_state(model, data, qpos: np.ndarray, qvel: np.ndarray) -> None:
    qpos_count = min(data.qpos.shape[0], qpos.shape[0])
    qvel_count = min(data.qvel.shape[0], qvel.shape[0])
    data.qpos[:qpos_count] = qpos[:qpos_count]
    data.qvel[:qvel_count] = qvel[:qvel_count]
    if qpos_count < data.qpos.shape[0]:
        data.qpos[qpos_count:] = 0.0
    if qvel_count < data.qvel.shape[0]:
        data.qvel[qvel_count:] = 0.0


def _optional_history(result: object, name: str) -> np.ndarray | None:
    if not hasattr(result, name):
        return None
    values = np.asarray(getattr(result, name), dtype=float)
    if values.ndim < 2 or values.shape[0] == 0:
        return None
    return values


def _restore_mocap_state(
    data,
    mocap_pos: np.ndarray | None,
    mocap_quat: np.ndarray | None,
    index: int,
) -> None:
    if mocap_pos is not None and getattr(data, "mocap_pos", None) is not None:
        count = min(data.mocap_pos.shape[0], mocap_pos.shape[1])
        if count:
            data.mocap_pos[:count] = mocap_pos[index, :count]
    if mocap_quat is not None and getattr(data, "mocap_quat", None) is not None:
        count = min(data.mocap_quat.shape[0], mocap_quat.shape[1])
        if count:
            data.mocap_quat[:count] = mocap_quat[index, :count]


def _video_frame_indices(sample_count: int, *, fps: int, stride: int | None) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if stride is None:
        # Keep long runs compact while preserving the final frame.
        stride = max(1, sample_count // 600)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    indices = np.arange(0, sample_count, stride, dtype=int)
    if indices[-1] != sample_count - 1:
        indices = np.append(indices, sample_count - 1)
    return indices


def _axis_bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    return center, max(0.5 * span * 1.15, 0.01)


def _apply_axis_bounds(axis, bounds: tuple[np.ndarray, float]) -> None:
    center, half = bounds
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _time_value(result: object, index: int) -> float:
    if not hasattr(result, "time"):
        return float(index)
    time = np.asarray(getattr(result, "time"), dtype=float)
    if time.shape[0] == 0:
        return float(index)
    return float(time[min(index, time.shape[0] - 1)])


def _write_video_error(output_path: Path, message: str) -> Path:
    error_path = output_path.parent / "video_error.txt"
    error_path.write_text(message + "\n", encoding="utf-8")
    return error_path


def _annotate_video_fallback(output_path: Path, fallback_path: Path) -> None:
    error_path = output_path.parent / "video_error.txt"
    if not error_path.is_file():
        return
    message = error_path.read_text(encoding="utf-8").rstrip()
    error_path.write_text(
        message
        + "\n"
        + f"fallback_saved: matplotlib trajectory animation at {fallback_path}\n",
        encoding="utf-8",
    )
