"""Optional observers for scenario-driven simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from continuum_sim.runtime.hook_utils import (
    executor_arm as _executor_arm,
    metadata_path as _metadata_path,
    metadata_paths as _metadata_paths,
    metadata_point as _metadata_point,
    sample_overlay_points as _sample_overlay_points,
    split_target_history as _split_target_history,
)
from continuum_sim.runtime.metadata_schema import ENGINE_NAVIGATION_OVERLAY_METADATA
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass
class _TrackingOverlayState:
    """Shared, bounded tracking data for live and recorded MuJoCo overlays."""

    tip_trail: list[np.ndarray] = field(default_factory=list)
    target_trail: list[np.ndarray] = field(default_factory=list)
    target_trail_kinds: list[str] = field(default_factory=list)
    base_trail: list[np.ndarray] = field(default_factory=list)
    observer_roi_world: np.ndarray | None = None
    navigation_metadata: dict[str, object] = field(default_factory=dict)

    def clear(self) -> None:
        self.tip_trail.clear()
        self.target_trail.clear()
        self.target_trail_kinds.clear()
        self.base_trail.clear()
        self.observer_roi_world = None
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
        observer_roi = _metadata_point(command.metadata, "visual_servo_roi_world")
        if observer_roi is None:
            observer_roi = _metadata_point(
                command.metadata,
                "engine_navigation_observer_roi_m",
            )
        self.observer_roi_world = None if observer_roi is None else observer_roi.copy()
        if is_engine_navigation:
            self.base_trail.append(state.base.pose.position.copy())
            self.navigation_metadata = {
                key: _copy_overlay_metadata_value(command.metadata[key])
                for key in ENGINE_NAVIGATION_OVERLAY_METADATA
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
    if (
        not navigation_enabled
        and navigation_config.observer_roi
        and overlay_state.observer_roi_world is not None
    ):
        _add_overlay_sphere(
            scene,
            mujoco,
            overlay_state.observer_roi_world,
            navigation_config.observer_roi_radius,
            navigation_config.observer_roi_rgba,
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
