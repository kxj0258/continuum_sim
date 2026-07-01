"""Optional observers for scenario-driven simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
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
                    "tendon_displacement_m": arm.tendon_displacement_m.copy(),
                    "tendon_velocity_mps": arm.tendon_velocity_mps.copy(),
                }
                for name, arm in state.arms.items()
            },
            "command_metadata": {} if command is None else dict(command.metadata),
            "state_metadata": dict(state.metadata),
        }


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
) -> None:
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
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
