"""Optional observers for scenario-driven simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np

from continuum_sim.system.types import RobotSystemCommand, RobotSystemState


@dataclass
class StateRecorderHook:
    """Record compact named state samples independently from backend details."""

    time_s: list[float] = field(default_factory=list)
    base_position_m: list[np.ndarray] = field(default_factory=list)
    arm_tip_position_m: dict[str, list[np.ndarray]] = field(default_factory=dict)
    target_position_m: list[np.ndarray] = field(default_factory=list)
    waypoint_index: list[int] = field(default_factory=list)
    tracking_error_m: list[float] = field(default_factory=list)
    min_clearance_m: list[float] = field(default_factory=list)
    contact_distance_m: list[float] = field(default_factory=list)
    target_force_n: list[float] = field(default_factory=list)
    estimated_force_n: list[float] = field(default_factory=list)
    force_error_n: list[float] = field(default_factory=list)
    contact_error_m: list[float] = field(default_factory=list)
    task_phase: list[str] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        self.time_s.clear()
        self.base_position_m.clear()
        self.arm_tip_position_m = {name: [] for name in state.arms}
        self.target_position_m.clear()
        self.waypoint_index.clear()
        self.tracking_error_m.clear()
        self.min_clearance_m.clear()
        self.contact_distance_m.clear()
        self.target_force_n.clear()
        self.estimated_force_n.clear()
        self.force_error_n.clear()
        self.contact_error_m.clear()
        self.task_phase.clear()
        self._append(state)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        self._append(state)
        target = command.metadata.get("executor_target_world")
        if target is not None:
            self.target_position_m.append(np.asarray(target, dtype=float).copy())
            self.waypoint_index.append(int(command.metadata.get("waypoint_index", 0)))
            self.tracking_error_m.append(
                float(command.metadata.get("executor_error_m", np.nan))
            )
            self.min_clearance_m.append(
                float(command.metadata.get("min_clearance_m", np.nan))
            )
            self.contact_distance_m.append(
                float(command.metadata.get("contact_distance_m", np.nan))
            )
            self.target_force_n.append(
                float(command.metadata.get("target_normal_force_n", np.nan))
            )
            self.estimated_force_n.append(
                float(command.metadata.get("estimated_normal_force_n", np.nan))
            )
            self.force_error_n.append(
                float(command.metadata.get("force_error_n", np.nan))
            )
            self.contact_error_m.append(
                float(command.metadata.get("contact_error_m", np.nan))
            )
            self.task_phase.append(str(command.metadata.get("wiping_phase", "")))

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _append(self, state: RobotSystemState) -> None:
        self.time_s.append(float(state.time_s))
        self.base_position_m.append(state.base.pose.position.copy())
        for name, arm in state.arms.items():
            self.arm_tip_position_m.setdefault(name, []).append(
                arm.tip_pose_world.position.copy()
            )


@dataclass
class MujocoReplayRecorderHook:
    """Record generalized state needed for deterministic offscreen replay."""

    backend: object
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    mocap_pos: list[np.ndarray] = field(default_factory=list)
    mocap_quat: list[np.ndarray] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self.qpos.clear()
        self.qvel.clear()
        self.mocap_pos.clear()
        self.mocap_quat.clear()
        self._append()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state, command, step_index
        self._append()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _append(self) -> None:
        data = self.backend.physics.data
        self.qpos.append(np.asarray(data.qpos, dtype=float).copy())
        self.qvel.append(np.asarray(data.qvel, dtype=float).copy())
        self.mocap_pos.append(np.asarray(data.mocap_pos, dtype=float).copy())
        self.mocap_quat.append(np.asarray(data.mocap_quat, dtype=float).copy())


class MujocoLiveVideoRecorderHook:
    """Write MuJoCo scene frames during simulation instead of replaying afterward."""

    def __init__(
        self,
        backend: object,
        output_path: str | Path,
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
        self.output_path = Path(output_path)
        self.fps = fps
        self.stride = 1 if stride is None else stride
        self.width = width
        self.height = height
        self.path: Path | None = None
        self.errors: list[str] = []
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writer = None
        self._camera = None
        self._tip_trail: list[np.ndarray] = []
        self._target_trail: list[np.ndarray] = []

    def on_reset(self, state: RobotSystemState) -> None:
        del state
        self.path = None
        self.errors.clear()
        self.frame_count = 0
        self._mujoco = None
        self._renderer = None
        self._writer = None
        self._camera = None
        self._tip_trail.clear()
        self._target_trail.clear()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
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
            self._writer = _open_video_writer(imageio, self.output_path, self.fps)
        except Exception as exc:  # noqa: BLE001 - video must not fail the run.
            self._record_error(f"live video setup failed: {type(exc).__name__}: {exc}")
            self._close_resources()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if self._renderer is None or self._writer is None or self._mujoco is None:
            return
        target = command.metadata.get("executor_target_world")
        if target is not None:
            self._target_trail.append(np.asarray(target, dtype=float).copy())
        executor = next(
            (arm for arm in state.arms.values() if arm.role == "executor"),
            None,
        )
        if executor is not None:
            self._tip_trail.append(executor.tip_pose_world.position.copy())
        if step_index % self.stride != 0:
            return
        try:
            data = self.backend.physics.data
            self._mujoco.mj_forward(self.backend.physics.model, data)
            self._renderer.update_scene(data, camera=self._camera)
            _draw_tracking_overlay_scene(
                self._renderer.scene,
                self._mujoco,
                self.backend.config.viewer.overlays,
                self._target_trail,
                self._tip_trail,
                state=state,
                reset_scene=False,
            )
            frame = self._renderer.render().copy()
            self._writer.append_data(frame)
            self.frame_count += 1
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
        if self.frame_count > 0 and not self.errors:
            self.path = self.output_path
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
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - report writer close errors.
                self._record_error(
                    f"live video writer close failed: {type(exc).__name__}: {exc}"
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


def _open_video_writer(imageio, path: Path, fps: int):
    if path.suffix.lower() == ".gif":
        return imageio.get_writer(path, mode="I", duration=1000.0 / fps, loop=0)
    return imageio.get_writer(path, fps=fps)


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


@dataclass
class TendonDiagnosticHook:
    """Collect tendon and singularity snapshots without a GUI dependency."""

    stride: int = 1
    samples: list[dict[str, Any]] = field(default_factory=list)

    def on_reset(self, state: RobotSystemState) -> None:
        self.samples.clear()
        self.samples.append(self._snapshot(state, None, -1))

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride == 0:
            self.samples.append(self._snapshot(state, command, step_index))

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state

    def _snapshot(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand | None,
        step_index: int,
    ) -> dict[str, Any]:
        return {
            "step_index": step_index,
            "time_s": state.time_s,
            "arms": {
                name: {
                    "tendon_target_m": arm.tendon_target_m.copy(),
                    "tendon_displacement_m": arm.tendon_displacement_m.copy(),
                    "tendon_velocity_mps": arm.tendon_velocity_mps.copy(),
                    "actuator_force_n": arm.actuator_force_n.copy(),
                }
                for name, arm in state.arms.items()
            },
            "command_metadata": {} if command is None else dict(command.metadata),
            "state_metadata": dict(state.metadata),
        }


class LiveTendonPanelHook:
    """Optional rich tendon monitor attached to the scenario hook lifecycle."""

    def __init__(self, *, stride: int = 1, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveTendonPanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._panel = None

    def on_reset(self, state: RobotSystemState) -> None:
        from continuum_sim.visualization.system_tendon_debug import (
            SystemTendonMonitorPanel,
        )

        if self._panel is not None:
            self._panel.close()
        self._panel = SystemTendonMonitorPanel()
        self._panel.update(state, redraw=False)
        self._panel.show(block=False)

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del command
        if step_index % self.stride == 0:
            if self._panel is not None and self._panel.is_open():
                self._panel.update(state)
                self._panel.flush_events()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._panel is not None:
            self._panel.flush_events()


class LiveWipingForcePanelHook:
    """Optional live panel for scenario wiping force/contact metadata."""

    def __init__(self, *, stride: int = 1, history_points: int = 300) -> None:
        if stride <= 0:
            raise ValueError("LiveWipingForcePanelHook stride must be positive.")
        self.stride = stride
        self.history_points = history_points
        self._plt = None
        self._figure = None
        self._axes = None
        self._time: list[float] = []
        self._target_force: list[float] = []
        self._estimated_force: list[float] = []
        self._contact_distance: list[float] = []

    def on_reset(self, state: RobotSystemState) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._figure, self._axes = plt.subplots()
        self._time.clear()
        self._target_force.clear()
        self._estimated_force.clear()
        self._contact_distance.clear()
        plt.ion()

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        if step_index % self.stride != 0:
            return
        self._time.append(float(state.time_s))
        self._target_force.append(float(command.metadata.get("target_normal_force_n", np.nan)))
        self._estimated_force.append(
            float(command.metadata.get("estimated_normal_force_n", np.nan))
        )
        self._contact_distance.append(float(command.metadata.get("contact_distance_m", np.nan)))
        self._trim()
        self._draw()

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return False

    def on_finish(self, state: RobotSystemState) -> None:
        del state
        if self._plt is not None:
            self._plt.ioff()

    def _trim(self) -> None:
        if len(self._time) <= self.history_points:
            return
        excess = len(self._time) - self.history_points
        del self._time[:excess]
        del self._target_force[:excess]
        del self._estimated_force[:excess]
        del self._contact_distance[:excess]

    def _draw(self) -> None:
        if self._axes is None or self._figure is None:
            return
        self._axes.clear()
        self._axes.plot(self._time, self._target_force, label="target force [N]")
        self._axes.plot(self._time, self._estimated_force, label="estimated force [N]")
        self._axes.plot(self._time, self._contact_distance, label="contact distance [m]")
        self._axes.set_xlabel("time [s]")
        self._axes.legend(loc="upper right")
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()


class MujocoViewerHook:
    """Optional passive viewer kept outside backend and controller policy."""

    def __init__(self, backend, *, keep_open: bool = False) -> None:
        self.backend = backend
        self.keep_open = keep_open
        self._viewer = None
        self._start_wall_s = 0.0
        self._start_sim_s = 0.0
        self._mujoco = None
        self._tip_trail: list[np.ndarray] = []
        self._target_trail: list[np.ndarray] = []

    def on_reset(self, state: RobotSystemState) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._tip_trail.clear()
        self._target_trail.clear()
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
            target = command.metadata.get("executor_target_world")
            if target is not None:
                self._target_trail.append(np.asarray(target, dtype=float).copy())
            executor = next(
                (arm for arm in state.arms.values() if arm.role == "executor"),
                None,
            )
            if executor is not None:
                self._tip_trail.append(executor.tip_pose_world.position.copy())
            _draw_mujoco_tracking_overlay(
                self._viewer,
                self._mujoco,
                self.backend.config.viewer.overlays,
                self._target_trail,
                self._tip_trail,
                state=state,
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
    target_trail: list[np.ndarray],
    tip_trail: list[np.ndarray],
    state: RobotSystemState | None = None,
) -> None:
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
    _draw_tracking_overlay_scene(
        scene,
        mujoco,
        config,
        target_trail,
        tip_trail,
        state=state,
        reset_scene=True,
    )


def _draw_tracking_overlay_scene(
    scene,
    mujoco,
    config,
    target_trail: list[np.ndarray],
    tip_trail: list[np.ndarray],
    *,
    state: RobotSystemState | None = None,
    reset_scene: bool,
) -> None:
    if reset_scene:
        scene.ngeom = 0
    if config.target_marker and target_trail:
        _add_overlay_sphere(
            scene,
            mujoco,
            target_trail[-1],
            config.target_marker_radius,
            config.target_marker_rgba,
        )
    if config.tip_trail:
        _add_overlay_trail(
            scene,
            mujoco,
            tip_trail[-config.trail_max_points :: config.trail_stride],
            config.tip_trail_radius,
            config.tip_trail_rgba,
        )
    if config.target_trail:
        _add_overlay_trail(
            scene,
            mujoco,
            target_trail[-config.trail_max_points :: config.trail_stride],
            config.target_trail_radius,
            config.target_trail_rgba,
        )
    if state is not None and config.segment_endpoints:
        _draw_segment_endpoint_overlay_scene(scene, mujoco, state, config)


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


class ControllerCompletionHook:
    """Stop a scenario when a waypoint-style controller reports completion."""

    def __init__(self, controller) -> None:
        self.controller = controller

    def on_reset(self, state: RobotSystemState) -> None:
        del state

    def on_step(
        self,
        state: RobotSystemState,
        command: RobotSystemCommand,
        step_index: int,
    ) -> None:
        del state, command, step_index

    def should_stop(self, state: RobotSystemState, step_index: int) -> bool:
        del state, step_index
        return bool(getattr(self.controller, "done", False))

    def on_finish(self, state: RobotSystemState) -> None:
        del state


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
        executor = next(
            (arm for arm in state.arms.values() if arm.role == "executor"),
            None,
        )
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
