"""Interactive viewer hooks and MuJoCo overlay helpers."""

from __future__ import annotations

import time
from threading import Event, Thread, current_thread

import numpy as np

from continuum_sim.runtime.concurrency import LatestValueSlot, TimeRateGate
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
from continuum_sim.runtime.mujoco_state_copy import capture_mujoco_dynamic_state
from continuum_sim.runtime.matplotlib_artists import PersistentAxisArtists
from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


class RealtimePacerHook:
    """Pace simulation time independently from any presentation refresh rate."""

    def __init__(
        self,
        *,
        realtime_factor: float,
        clock=time.perf_counter,
        sleeper=None,
    ) -> None:
        if realtime_factor <= 0.0:
            raise ValueError("realtime_factor must be positive.")
        self.realtime_factor = float(realtime_factor)
        self._clock = clock
        self._sleeper = _sleep_until_simulation_time if sleeper is None else sleeper
        self._start_wall_s = 0.0
        self._start_sim_s = 0.0

    def on_reset(self, state: RobotSystemState) -> None:
        self._start_wall_s = float(self._clock())
        self._start_sim_s = float(state.time_s)

    def on_step(self, state, command, step_index) -> None:
        del command, step_index
        self._sleeper(
            self._start_wall_s,
            self._start_sim_s,
            float(state.time_s),
            self.realtime_factor,
        )

    def should_stop(self, state, step_index) -> bool:
        del state, step_index
        return False

    def on_finish(self, state) -> None:
        del state


class MujocoViewerHook:
    """Optional passive viewer kept outside backend and controller policy."""

    def __init__(
        self,
        backend,
        *,
        keep_open: bool = False,
        display_interval_s: float = 1.0 / 15.0,
    ) -> None:
        self.backend = backend
        self.keep_open = keep_open
        self._viewer = None
        self._mujoco = None
        self._viewer_data = None
        self._overlay_state = _TrackingOverlayState()
        self._display_gate = TimeRateGate(display_interval_s)
        self._requests = LatestValueSlot(None)
        self._request_event = Event()
        self._stop_event = Event()
        self._ready_event = Event()
        self._first_sync_event = Event()
        self._worker: Thread | None = None
        self._worker_failure: BaseException | None = None
        self._viewer_running = True

    def on_reset(self, state: RobotSystemState) -> None:
        self._overlay_state.clear()
        self._display_gate.reset(float(state.time_s))
        self._worker_failure = None
        self._viewer_running = True
        self._stop_event.clear()
        self._ready_event.clear()
        self._first_sync_event.clear()
        self._request_event.clear()
        self._requests = LatestValueSlot(
            (capture_mujoco_dynamic_state(self.backend.physics.data), state, None)
        )
        self._worker = Thread(
            target=self._run_viewer,
            name="continuum-sim-mujoco-viewer",
            daemon=False,
        )
        self._worker.start()
        self._ready_event.wait()
        self._first_sync_event.wait()

    def _open_viewer(self):
        import mujoco
        import mujoco.viewer

        data = mujoco.MjData(self.backend.physics.model)
        viewer = mujoco.viewer.launch_passive(
            self.backend.physics.model,
            data,
        )
        return mujoco, viewer, data

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del step_index
        if self._display_gate.due(state.time_s):
            self._requests.publish(
                (
                    capture_mujoco_dynamic_state(self.backend.physics.data),
                    state,
                    command,
                )
            )
            self._request_event.set()

    def _draw_overlay(self, state, command) -> None:
        if command is not None:
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

    def _sync_snapshot(self, snapshot, state, command) -> None:
        snapshot.apply_to(self._viewer_data)
        self._mujoco.mj_forward(self.backend.physics.model, self._viewer_data)
        if command is not None:
            self._draw_overlay(state, command)
        self._viewer.sync()

    def _run_viewer(self) -> None:
        consumed_version = -1
        try:
            self._mujoco, self._viewer, self._viewer_data = self._open_viewer()
            _configure_mujoco_viewer(self._viewer, self.backend.config)
            self._ready_event.set()
            while True:
                item = self._requests.consume_after(consumed_version)
                if item is None:
                    if self._stop_event.is_set():
                        break
                    self._request_event.wait(0.05)
                    self._request_event.clear()
                    continue
                payload, consumed_version = item
                snapshot, state, command = payload
                self._sync_snapshot(snapshot, state, command)
                self._first_sync_event.set()
                self._viewer_running = self._viewer.is_running()
        except BaseException as exc:  # noqa: BLE001 - surfaced through stop state.
            self._worker_failure = exc
            self._viewer_running = False
            self._ready_event.set()
            self._first_sync_event.set()
        finally:
            viewer = self._viewer
            if viewer is not None:
                while self.keep_open and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
                viewer.close()
            self._viewer = None
            self._viewer_data = None

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return not self._viewer_running

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        self._stop_event.set()
        self._request_event.set()
        worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join()
        self._worker = None


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

    requires_gui_main_thread = True

    def __init__(self, *, keep_open: bool = True) -> None:
        self.keep_open = keep_open
        self._plt = None
        self._figure = None
        self._axes = None
        self._artists = None
        self._latest_state = None
        self._target = None
        self._tip_trail: list[np.ndarray] = []
        self._target_trail: list[np.ndarray] = []
        self._presentation_closed = False

    def on_reset(self, state: RobotSystemState) -> None:
        self._tip_trail.clear()
        self._target_trail.clear()
        self._latest_state = state
        self._presentation_closed = False

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
        self._latest_state = state

    def present_pending(self, *, force: bool = False) -> None:
        del force
        state = self._latest_state
        if state is None:
            return
        if self._figure is None:
            import matplotlib.pyplot as plt

            self._plt = plt
            self._figure = plt.figure()
            axis = self._figure.add_subplot(111, projection="3d")
            self._artists = PersistentAxisArtists(axis)
            self._axes = self._artists
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_zlabel("z [m]")
            plt.ion()
            plt.show(block=False)
        self._presentation_closed = not self._plt.fignum_exists(self._figure.number)
        self._draw(state)

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return self._presentation_closed

    def on_finish(self, state: RobotSystemState) -> None:
        self._latest_state = state

    def close_presentation(self) -> None:
        if self._plt is None:
            return
        self._plt.ioff()
        if self.keep_open and self._figure is not None:
            self._plt.show()
        elif self._figure is not None:
            self._plt.close(self._figure)
        self._figure = None

    def _draw(self, state: RobotSystemState) -> None:
        axes = self._axes
        axes.begin_frame()
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
        axes.legend()
        axes.end_frame()
        self._figure.canvas.draw_idle()


__all__ = [
    "MatplotlibSystemViewerHook",
    "MujocoViewerHook",
    "RealtimePacerHook",
    "_TrackingOverlayState",
    "_configure_mujoco_viewer",
    "_metadata_path",
    "_metadata_paths",
    "_metadata_point",
    "_sample_overlay_points",
    "_split_target_history",
]
