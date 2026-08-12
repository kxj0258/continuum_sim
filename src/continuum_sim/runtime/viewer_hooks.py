"""Interactive viewer hooks and MuJoCo overlay helpers."""

from __future__ import annotations

import time

import numpy as np

from continuum_sim.runtime.hook_utils import (
    executor_arm as _executor_arm,
    metadata_path as _metadata_path,
    metadata_paths as _metadata_paths,
    metadata_point as _metadata_point,
    sample_overlay_points as _sample_overlay_points,
    split_target_history as _split_target_history,
)
from continuum_sim.runtime.mujoco_overlay_utils import (
    _TrackingOverlayState,
    _draw_mujoco_tracking_overlay,
    _update_follow_camera,
)
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


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
    # Engine work is observed from both outside and inside the closed CAD
    # shell. Rendering back faces prevents interior surfaces disappearing.
    scene = getattr(viewer, "user_scn", None)
    flags = getattr(scene, "flags", None)
    if flags is not None:
        try:
            import mujoco

            flags[int(mujoco.mjtRndFlag.mjRND_CULL_FACE)] = 0
        except (ImportError, IndexError, TypeError):
            pass


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


__all__ = [
    "MatplotlibSystemViewerHook",
    "MujocoViewerHook",
    "_TrackingOverlayState",
    "_configure_mujoco_viewer",
    "_metadata_path",
    "_metadata_paths",
    "_metadata_point",
    "_sample_overlay_points",
    "_split_target_history",
]
