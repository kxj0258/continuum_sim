"""Shared MuJoCo runtime helpers used across viewer-backed scripts."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.backends import BackendState, MujocoBackend
from continuum_sim.model import ThreeSegmentRobotParams
from continuum_sim.visualization.mujoco_tendon_path_overlay import (
    draw_tendon_path_overlay,
)


@dataclass(frozen=True)
class TendonOverlayContext:
    """Viewer-only context for explanatory fixed-tendon overlays."""

    backend: MujocoBackend
    params: ThreeSegmentRobotParams
    physical_tendons: tuple
    links_per_segment: int


def _generated_visual_xml_path(config) -> Path:
    if config.control_mode == "position_joint":
        return config.generated_xml_path
    if config.control_mode == "tendon_position":
        return config.tendon_generated_xml_path
    raise ValueError(f"Unsupported MuJoCo control_mode {config.control_mode!r}.")


def _resolve_visual_xml_path(config, use_segment_visuals: bool) -> Path | None:
    if not use_segment_visuals:
        return None
    visual_xml_path = _generated_visual_xml_path(config)
    if not visual_xml_path.is_file():
        if getattr(config.model, "type", "distributed_links") == "segment_2dof_followers":
            hint = "Run scripts/build_mujoco_segment_2dof_model.py."
        else:
            hint = (
                "Run scripts/build_mujoco_with_segment_visuals.py after placing all "
                "segmented STL files and enabling visuals."
            )
        raise FileNotFoundError(
            "Segmented visual XML does not exist: "
            f"{visual_xml_path}. "
            f"{hint}"
        )
    return visual_xml_path


def _configure_viewer_groups(viewer, config, show_collision_geoms: bool) -> None:
    opt = getattr(viewer, "opt", None)
    if opt is None or not hasattr(opt, "geomgroup"):
        return
    opt.geomgroup[config.visuals.visual_geom_group] = 1
    opt.geomgroup[config.visuals.collision_geom_group] = int(show_collision_geoms)


def _configure_viewer_camera(viewer, config) -> None:
    cam = getattr(viewer, "cam", None)
    if cam is None:
        return
    camera = config.viewer.camera
    cam.lookat[:] = camera.lookat
    cam.distance = camera.distance
    cam.azimuth = camera.azimuth
    cam.elevation = camera.elevation


def draw_tendon_path_overlay_if_enabled(
    viewer,
    backend: MujocoBackend,
    config,
    params: ThreeSegmentRobotParams | None,
    physical_tendons: Sequence | None,
) -> None:
    """Draw the configured tendon path overlay into a passive viewer scene."""
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
    scene.ngeom = 0
    if not config.viewer.overlays.tendon_paths or params is None:
        return
    draw_tendon_path_overlay(
        scene,
        backend._mujoco,
        backend.model,
        backend.data,
        params,
        tuple(physical_tendons or ()),
        links_per_segment=config.links_per_segment,
        radius=config.viewer.overlays.tendon_path_radius,
        stride=config.viewer.overlays.tendon_path_stride,
    )


def sleep_for_realtime(
    start_wall: float,
    start_sim: float,
    current_sim: float,
    realtime_factor: float,
) -> None:
    """Sleep until wall time catches up with simulation time."""
    target_wall_elapsed = (current_sim - start_sim) / realtime_factor
    sleep_s = target_wall_elapsed - (time.perf_counter() - start_wall)
    if sleep_s > 0.0:
        time.sleep(sleep_s)


def compute_mujoco_control_substeps(controller_dt: float, mujoco_timestep: float) -> int:
    """Return MuJoCo substeps that make one control step span controller_dt."""

    if controller_dt <= 0.0:
        raise ValueError(f"controller_dt must be positive, got {controller_dt}.")
    if mujoco_timestep <= 0.0:
        raise ValueError(f"mujoco_timestep must be positive, got {mujoco_timestep}.")
    return max(1, round(controller_dt / mujoco_timestep))


def _clip_tendon_position_control(
    tendon_delta_command: np.ndarray,
    ctrlrange_m: tuple[float, float],
) -> np.ndarray:
    lower, upper = ctrlrange_m
    return np.clip(np.asarray(tendon_delta_command, dtype=float), lower, upper)


def _append_trail_sample(
    trail: list[np.ndarray],
    point: np.ndarray,
    sample_index: int,
    stride: int,
    max_points: int,
) -> None:
    if stride <= 0:
        raise ValueError(f"trail_stride must be positive, got {stride}.")
    if max_points <= 0:
        raise ValueError(f"trail_max_points must be positive, got {max_points}.")
    if sample_index % stride != 0:
        return
    trail.append(np.asarray(point, dtype=float).copy())
    overflow = len(trail) - max_points
    if overflow > 0:
        del trail[:overflow]


def _select_trail_segments(
    points: Sequence[np.ndarray],
    max_segments: int,
    *,
    min_distance: float = 1.0e-9,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if max_segments <= 0 or len(points) < 2:
        return []
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(points) - 1, 0, -1):
        start = np.asarray(points[index - 1], dtype=float)
        end = np.asarray(points[index], dtype=float)
        if float(np.linalg.norm(end - start)) <= min_distance:
            continue
        segments.append((start.copy(), end.copy()))
        if len(segments) >= max_segments:
            break
    segments.reverse()
    return segments


def _draw_tracking_overlays(
    viewer,
    mujoco_module,
    tendon_overlay: TendonOverlayContext | None,
    overlay_config,
    target_position: np.ndarray,
    tip_trail: Sequence[np.ndarray],
    target_trail: Sequence[np.ndarray],
) -> None:
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
    scene.ngeom = 0
    if overlay_config.target_marker:
        _draw_target_marker(
            scene,
            mujoco_module,
            target_position,
            overlay_config.target_marker_radius,
            overlay_config.target_marker_rgba,
        )
    if overlay_config.tip_trail:
        _draw_capsule_trail(
            scene,
            mujoco_module,
            tip_trail,
            overlay_config.tip_trail_radius,
            overlay_config.tip_trail_rgba,
        )
    if overlay_config.target_trail:
        _draw_capsule_trail(
            scene,
            mujoco_module,
            target_trail,
            overlay_config.target_trail_radius,
            overlay_config.target_trail_rgba,
        )
    if overlay_config.tendon_paths and tendon_overlay is not None:
        draw_tendon_path_overlay(
            scene,
            mujoco_module,
            tendon_overlay.backend.model,
            tendon_overlay.backend.data,
            tendon_overlay.params,
            tendon_overlay.physical_tendons,
            links_per_segment=tendon_overlay.links_per_segment,
            radius=overlay_config.tendon_path_radius,
            stride=overlay_config.tendon_path_stride,
        )


def _draw_target_marker(
    scene,
    mujoco_module,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    geom = _next_overlay_geom(scene)
    if geom is None:
        return
    mujoco_module.mjv_initGeom(
        geom,
        mujoco_module.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius, 0.0, 0.0), dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )


def _draw_capsule_trail(
    scene,
    mujoco_module,
    points: Sequence[np.ndarray],
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    for start, end in _select_trail_segments(points, _available_overlay_slots(scene)):
        geom = _next_overlay_geom(scene)
        if geom is None:
            return
        _connect_capsule_geom(mujoco_module, geom, radius, start, end)
        geom.rgba[:] = rgba


def _connect_capsule_geom(
    mujoco_module,
    geom,
    radius: float,
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    geom_type = mujoco_module.mjtGeom.mjGEOM_CAPSULE
    connector = getattr(mujoco_module, "mjv_connector", None)
    if connector is not None:
        connector(
            geom,
            geom_type,
            float(radius),
            np.ascontiguousarray(np.asarray(start, dtype=np.float64).reshape(3)),
            np.ascontiguousarray(np.asarray(end, dtype=np.float64).reshape(3)),
        )
        return

    make_connector = getattr(mujoco_module, "mjv_makeConnector", None)
    if make_connector is None:
        raise AttributeError(
            "MuJoCo module has neither mjv_connector nor mjv_makeConnector."
        )
    make_connector(
        geom,
        geom_type,
        float(radius),
        float(start[0]),
        float(start[1]),
        float(start[2]),
        float(end[0]),
        float(end[1]),
        float(end[2]),
    )


def _next_overlay_geom(scene):
    if int(scene.ngeom) >= _scene_maxgeom(scene):
        return None
    geom = scene.geoms[int(scene.ngeom)]
    scene.ngeom += 1
    return geom


def _available_overlay_slots(scene) -> int:
    return max(0, _scene_maxgeom(scene) - int(scene.ngeom))


def _scene_maxgeom(scene) -> int:
    maxgeom = getattr(scene, "maxgeom", None)
    if maxgeom is not None:
        return int(maxgeom)
    return len(getattr(scene, "geoms", ()))


def _history_trail_points(
    points: Sequence[np.ndarray],
    end_index: int,
    stride: int,
    max_points: int,
) -> list[np.ndarray]:
    selected = [
        np.asarray(point, dtype=float).copy()
        for sample_index, point in enumerate(points[: end_index + 1])
        if sample_index % stride == 0
    ]
    return selected[-max_points:]


def _effective_realtime_factor(realtime_factor: float, viewer_speed: float) -> float:
    return max(1.0e-9, realtime_factor * viewer_speed)


def _sync_tracking_viewer(
    viewer,
    step_index: int,
    sync_interval_steps: int,
    realtime: bool,
    realtime_start_wall: float,
    controller_dt: float,
    realtime_factor: float,
    viewer_speed: float,
    mujoco_module,
    tendon_overlay: TendonOverlayContext | None,
    overlay_config,
    target_position: np.ndarray,
    tip_trail: Sequence[np.ndarray],
    target_trail: Sequence[np.ndarray],
) -> None:
    if viewer is None:
        return
    if (step_index + 1) % sync_interval_steps == 0:
        if mujoco_module is not None:
            _draw_tracking_overlays(
                viewer,
                mujoco_module,
                tendon_overlay,
                overlay_config,
                target_position,
                tip_trail,
                target_trail,
            )
        viewer.sync()
    if realtime:
        target_wall_elapsed = (
            (step_index + 1)
            * controller_dt
            / _effective_realtime_factor(realtime_factor, viewer_speed)
        )
        sleep_s = target_wall_elapsed - (time.perf_counter() - realtime_start_wall)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
