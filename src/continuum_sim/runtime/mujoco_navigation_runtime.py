"""Run structured obstacle navigation through the MuJoCo backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.actuation.motor_mapping import (
    motor_position_to_tendon_delta,
    motor_velocity_to_tendon_velocity,
    tendon_delta_to_motor_position,
)
from continuum_sim.backends import BackendState, MujocoBackend, pcc_q_to_joint_targets
from continuum_sim.config import load_mujoco_config
from continuum_sim.control.navigation_controller import (
    compute_navigation_motor_velocity_command,
    compute_navigation_motor_velocity_command_from_observation,
)
from continuum_sim.model import (
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
)
from continuum_sim.runtime import mujoco_runtime_utils as _mujoco_runtime_utils
from continuum_sim.runtime.mujoco_viewer_runtime import (
    ViewerControlState,
    _create_tendon_live_panel,
    _make_viewer_key_callback,
    _maybe_hold_tracking_viewer_open_after_run,
    _state_mocap_pos,
    _state_mocap_quat,
    _sync_tendon_live_panel,
)
from continuum_sim.scenes.scene_builder import build_mujoco_scene_xml
from continuum_sim.scenes.scene_config import load_navigation_scene_config
from continuum_sim.tasks.navigation_config import load_mujoco_navigation_config


@dataclass(frozen=True)
class MujocoNavigationResult:
    """Recorded samples from a structured MuJoCo navigation rollout."""

    time: np.ndarray
    target_position: np.ndarray
    waypoint_index: np.ndarray
    tip_pose: np.ndarray
    segment_poses: np.ndarray
    error_norm: np.ndarray
    min_clearance_m: np.ndarray
    clearance_source_id: tuple[str, ...]
    clearance_point: np.ndarray
    clearance_normal: np.ndarray
    motor_position: np.ndarray
    motor_velocity: np.ndarray
    tendon_delta: np.ndarray
    q_est: np.ndarray
    joint_targets: np.ndarray
    mujoco_control: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    tendon_length: np.ndarray
    tendon_velocity: np.ndarray
    actuator_force: np.ndarray
    scene_xml_path: Path

    @property
    def tip_position(self) -> np.ndarray:
        return self.tip_pose[:, :3, 3]


def run_mujoco_navigation(
    task_config_path: str | Path = Path("configs/tasks/mujoco_navigation_rocket.yaml"),
    mujoco_config_path: str | Path = Path("configs/mujoco.yaml"),
    *,
    show: bool | None = None,
) -> MujocoNavigationResult:
    """Run ordered inspection navigation in a structured MuJoCo scene."""

    task_config = load_mujoco_navigation_config(task_config_path)
    mujoco_config = load_mujoco_config(mujoco_config_path)
    scene_config = load_navigation_scene_config(task_config.scene_config_path)
    target_positions = scene_config.target_positions(task_config.mission.waypoint_ids)
    scene_xml_path = build_mujoco_scene_xml(
        _base_xml_path(mujoco_config),
        scene_config,
        task_config.generated_scene_xml_path,
    )

    show_viewer = mujoco_config.viewer.show if show is None else show
    sync_interval = mujoco_config.viewer.sync_interval_steps
    sleep_realtime = mujoco_config.viewer.realtime
    sleep_factor = mujoco_config.viewer.realtime_factor
    if sync_interval <= 0:
        raise ValueError(f"sync_interval_steps must be positive, got {sync_interval}.")
    if sleep_factor <= 0.0:
        raise ValueError(f"realtime_factor must be positive, got {sleep_factor}.")

    params = ThreeSegmentRobotParams.from_yaml(task_config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(task_config.robot_config_path)
    motor_params = load_motor_params_from_yaml(task_config.robot_config_path)
    n_substeps = compute_mujoco_control_substeps(
        task_config.simulation.dt,
        mujoco_config.solver.timestep,
    )
    backend = MujocoBackend.from_config(
        mujoco_config,
        override_xml_path=scene_xml_path,
    )
    observed_state = backend.reset()
    tendon_overlay = TendonOverlayContext(
        backend=backend,
        params=params,
        physical_tendons=physical_tendons,
        links_per_segment=mujoco_config.links_per_segment,
    )

    motor_position = task_config.simulation.initial_motor_position_rad.copy()
    waypoint_index = 0
    times: list[float] = []
    target_history: list[np.ndarray] = []
    waypoint_history: list[int] = []
    tip_pose_history: list[np.ndarray] = []
    segment_pose_history: list[np.ndarray] = []
    error_history: list[float] = []
    clearance_history: list[float] = []
    clearance_source_history: list[str] = []
    clearance_point_history: list[np.ndarray] = []
    clearance_normal_history: list[np.ndarray] = []
    motor_position_history: list[np.ndarray] = []
    motor_velocity_history: list[np.ndarray] = []
    tendon_delta_history: list[np.ndarray] = []
    q_history: list[np.ndarray] = []
    joint_target_history: list[np.ndarray] = []
    mujoco_control_history: list[np.ndarray] = []
    qpos_history: list[np.ndarray] = []
    qvel_history: list[np.ndarray] = []
    mocap_pos_history: list[np.ndarray] = []
    mocap_quat_history: list[np.ndarray] = []
    tendon_length_history: list[np.ndarray] = []
    tendon_velocity_history: list[np.ndarray] = []
    actuator_force_history: list[np.ndarray] = []
    tip_trail: list[np.ndarray] = []
    target_trail: list[np.ndarray] = []
    tendon_monitor_panel = None

    def run_loop(
        viewer=None,
        mujoco_module=None,
        viewer_control: ViewerControlState | None = None,
    ) -> None:
        nonlocal motor_position, observed_state, waypoint_index
        realtime_start_wall = time.perf_counter()
        step_index = 0
        while step_index < task_config.simulation.max_steps:
            if viewer is not None and not viewer.is_running():
                break
            current_target = target_positions[waypoint_index]
            if (
                viewer is not None
                and mujoco_module is not None
                and viewer_control is not None
                and viewer_control.paused
                and not viewer_control.step_once
            ):
                if tendon_monitor_panel is not None:
                    tendon_monitor_panel.flush_events()
                _draw_tracking_overlays(
                    viewer,
                    mujoco_module,
                    tendon_overlay,
                    mujoco_config.viewer.overlays,
                    current_target,
                    tip_trail,
                    target_trail,
                )
                viewer.sync()
                time.sleep(0.02)
                continue
            if viewer_control is not None and viewer_control.step_once:
                viewer_control.step_once = False

            if mujoco_config.control_mode == "position_joint":
                controller_motor_position = motor_position.copy()
                motor_velocity_cmd, info = compute_navigation_motor_velocity_command(
                    controller_motor_position,
                    current_target,
                    params,
                    physical_tendons,
                    motor_params,
                    scene_config.clearance_primitives,
                    task_config.controller,
                )
                tendon_delta = motor_position_to_tendon_delta(
                    controller_motor_position,
                    motor_params,
                )
                q_est = np.asarray(info["q_est"], dtype=float)
                joint_targets = pcc_q_to_joint_targets(
                    q_est,
                    params,
                    mujoco_config.links_per_segment,
                )
                mujoco_control = joint_targets
            elif mujoco_config.control_mode == "tendon_position":
                if task_config.mujoco.feedback_mode == "mujoco_actual":
                    tendon_delta = observed_state.tendon_length.copy()
                    controller_motor_position = tendon_delta_to_motor_position(
                        tendon_delta,
                        motor_params,
                    )
                    motor_velocity_cmd, info = (
                        compute_navigation_motor_velocity_command_from_observation(
                            observed_state.tip_pose[:3, 3],
                            tendon_delta,
                            current_target,
                            params,
                            physical_tendons,
                            motor_params,
                            scene_config.clearance_primitives,
                            task_config.controller,
                        )
                    )
                    q_est = np.asarray(info["q_est"], dtype=float)
                    tendon_velocity_cmd = motor_velocity_to_tendon_velocity(
                        motor_velocity_cmd,
                        motor_params,
                    )
                    mujoco_control = _clip_tendon_position_control(
                        tendon_delta + task_config.simulation.dt * tendon_velocity_cmd,
                        mujoco_config.actuators.tendon_position.ctrlrange_m,
                    )
                else:
                    controller_motor_position = motor_position.copy()
                    motor_velocity_cmd, info = compute_navigation_motor_velocity_command(
                        controller_motor_position,
                        current_target,
                        params,
                        physical_tendons,
                        motor_params,
                        scene_config.clearance_primitives,
                        task_config.controller,
                    )
                    tendon_delta = motor_position_to_tendon_delta(
                        controller_motor_position,
                        motor_params,
                    )
                    q_est = np.asarray(info["q_est"], dtype=float)
                    mujoco_control = _clip_tendon_position_control(
                        tendon_delta,
                        mujoco_config.actuators.tendon_position.ctrlrange_m,
                    )
                joint_targets = np.zeros((0,), dtype=float)
            else:
                raise ValueError(
                    f"Unsupported MuJoCo control_mode {mujoco_config.control_mode!r}."
                )

            state = backend.step(mujoco_control, n_substeps=n_substeps)
            observed_state = state
            _sync_tendon_live_panel(
                tendon_monitor_panel,
                task_config.mujoco.live_tendon_panel_stride,
                mujoco_control,
                state,
                step_index,
            )

            tip_position = state.tip_pose[:3, 3]
            error_norm = float(np.linalg.norm(current_target - tip_position))
            min_clearance = float(info["min_clearance_m"])
            times.append(step_index * task_config.simulation.dt)
            target_history.append(current_target.copy())
            waypoint_history.append(waypoint_index)
            tip_pose_history.append(state.tip_pose.copy())
            segment_pose_history.append(state.segment_poses.copy())
            error_history.append(error_norm)
            clearance_history.append(min_clearance)
            clearance_source_history.append(str(info["clearance_source_id"]))
            clearance_point_history.append(np.asarray(info["clearance_point"], dtype=float).copy())
            clearance_normal_history.append(np.asarray(info["clearance_normal"], dtype=float).copy())
            motor_position_history.append(controller_motor_position.copy())
            motor_velocity_history.append(motor_velocity_cmd.copy())
            tendon_delta_history.append(tendon_delta.copy())
            q_history.append(q_est.copy())
            joint_target_history.append(joint_targets.copy())
            mujoco_control_history.append(mujoco_control.copy())
            qpos_history.append(state.qpos.copy())
            qvel_history.append(state.qvel.copy())
            mocap_pos_history.append(_state_mocap_pos(state))
            mocap_quat_history.append(_state_mocap_quat(state))
            tendon_length_history.append(state.tendon_length.copy())
            tendon_velocity_history.append(state.tendon_velocity.copy())
            actuator_force_history.append(state.actuator_force.copy())
            sample_index = len(times) - 1
            _append_trail_sample(
                tip_trail,
                tip_position,
                sample_index,
                mujoco_config.viewer.overlays.trail_stride,
                mujoco_config.viewer.overlays.trail_max_points,
            )
            _append_trail_sample(
                target_trail,
                current_target,
                sample_index,
                mujoco_config.viewer.overlays.trail_stride,
                mujoco_config.viewer.overlays.trail_max_points,
            )

            if min_clearance < task_config.controller.clearance_min_m:
                if task_config.mission.terminate_on_clearance_violation:
                    _sync_tracking_viewer(
                        viewer,
                        step_index,
                        sync_interval,
                        sleep_realtime,
                        realtime_start_wall,
                        task_config.simulation.dt,
                        sleep_factor,
                        viewer_control.speed if viewer_control is not None else 1.0,
                        mujoco_module,
                        tendon_overlay,
                        mujoco_config.viewer.overlays,
                        current_target,
                        tip_trail,
                        target_trail,
                    )
                    break

            if error_norm <= task_config.controller.position_tolerance_m:
                if waypoint_index < len(target_positions) - 1:
                    waypoint_index += 1
                elif task_config.simulation.stop_on_completion:
                    _sync_tracking_viewer(
                        viewer,
                        step_index,
                        sync_interval,
                        sleep_realtime,
                        realtime_start_wall,
                        task_config.simulation.dt,
                        sleep_factor,
                        viewer_control.speed if viewer_control is not None else 1.0,
                        mujoco_module,
                        tendon_overlay,
                        mujoco_config.viewer.overlays,
                        current_target,
                        tip_trail,
                        target_trail,
                    )
                    break

            if (
                mujoco_config.control_mode == "position_joint"
                or task_config.mujoco.feedback_mode == "pcc_command"
            ):
                motor_position = motor_position + motor_velocity_cmd * task_config.simulation.dt
                motor_position = np.clip(
                    motor_position,
                    -task_config.simulation.position_limit_rad,
                    task_config.simulation.position_limit_rad,
                )
            _sync_tracking_viewer(
                viewer,
                step_index,
                sync_interval,
                sleep_realtime,
                realtime_start_wall,
                task_config.simulation.dt,
                sleep_factor,
                viewer_control.speed if viewer_control is not None else 1.0,
                mujoco_module,
                tendon_overlay,
                mujoco_config.viewer.overlays,
                current_target,
                tip_trail,
                target_trail,
            )
            step_index += 1

    if show_viewer:
        import mujoco.viewer

        viewer_control = ViewerControlState()
        key_callback = _make_viewer_key_callback(viewer_control)
        with mujoco.viewer.launch_passive(
            backend.model,
            backend.data,
            key_callback=key_callback,
        ) as viewer:
            _configure_viewer_groups(
                viewer,
                mujoco_config,
                mujoco_config.viewer.show_collision_geoms,
            )
            _configure_viewer_camera(viewer, mujoco_config)
            tendon_monitor_panel = _create_tendon_live_panel(
                show_viewer=show_viewer,
                control_mode=mujoco_config.control_mode,
                show_live_tendon_panel=task_config.mujoco.show_live_tendon_panel,
                config=mujoco_config,
                params=params,
                physical_tendons=physical_tendons,
                initial_state=observed_state,
            )
            try:
                _draw_tracking_overlays(
                    viewer,
                    mujoco,
                    tendon_overlay,
                    mujoco_config.viewer.overlays,
                    target_positions[waypoint_index],
                    tip_trail,
                    target_trail,
                )
                viewer.sync()
                run_loop(viewer, mujoco, viewer_control)
                _maybe_hold_tracking_viewer_open_after_run(
                    keep_viewer_open_after_run=task_config.mujoco.hold_viewer_open_after_run,
                    viewer=viewer,
                    backend=backend,
                    mujoco_module=mujoco,
                    control=viewer_control,
                    tendon_overlay=tendon_overlay,
                    tendon_monitor_panel=tendon_monitor_panel,
                    overlay_config=mujoco_config.viewer.overlays,
                    target_history=target_history,
                    tip_pose_history=tip_pose_history,
                    qpos_history=qpos_history,
                    qvel_history=qvel_history,
                    mujoco_control_history=mujoco_control_history,
                    tendon_length_history=tendon_length_history,
                    actuator_force_history=actuator_force_history,
                    controller_dt=task_config.simulation.dt,
                    realtime=sleep_realtime,
                    realtime_factor=sleep_factor,
                )
            finally:
                if tendon_monitor_panel is not None:
                    tendon_monitor_panel.close()
    else:
        run_loop()

    return MujocoNavigationResult(
        time=np.asarray(times, dtype=float),
        target_position=np.asarray(target_history, dtype=float),
        waypoint_index=np.asarray(waypoint_history, dtype=int),
        tip_pose=np.asarray(tip_pose_history, dtype=float),
        segment_poses=np.asarray(segment_pose_history, dtype=float),
        error_norm=np.asarray(error_history, dtype=float),
        min_clearance_m=np.asarray(clearance_history, dtype=float),
        clearance_source_id=tuple(clearance_source_history),
        clearance_point=np.asarray(clearance_point_history, dtype=float),
        clearance_normal=np.asarray(clearance_normal_history, dtype=float),
        motor_position=np.asarray(motor_position_history, dtype=float),
        motor_velocity=np.asarray(motor_velocity_history, dtype=float),
        tendon_delta=np.asarray(tendon_delta_history, dtype=float),
        q_est=np.asarray(q_history, dtype=float),
        joint_targets=np.asarray(joint_target_history, dtype=float),
        mujoco_control=np.asarray(mujoco_control_history, dtype=float),
        qpos=np.asarray(qpos_history, dtype=float),
        qvel=np.asarray(qvel_history, dtype=float),
        mocap_pos=np.asarray(mocap_pos_history, dtype=float),
        mocap_quat=np.asarray(mocap_quat_history, dtype=float),
        tendon_length=np.asarray(tendon_length_history, dtype=float),
        tendon_velocity=np.asarray(tendon_velocity_history, dtype=float),
        actuator_force=np.asarray(actuator_force_history, dtype=float),
        scene_xml_path=scene_xml_path,
    )


def _base_xml_path(config) -> Path:
    visual_xml_path = _resolve_visual_xml_path(config, config.viewer.use_segment_visuals)
    if visual_xml_path is not None:
        return visual_xml_path
    if config.control_mode == "position_joint":
        return config.xml_path
    if config.control_mode == "tendon_position":
        return config.tendon_xml_path
    raise ValueError(f"Unsupported MuJoCo control_mode {config.control_mode!r}.")


TendonOverlayContext = _mujoco_runtime_utils.TendonOverlayContext
_resolve_visual_xml_path = _mujoco_runtime_utils._resolve_visual_xml_path
_configure_viewer_groups = _mujoco_runtime_utils._configure_viewer_groups
_configure_viewer_camera = _mujoco_runtime_utils._configure_viewer_camera
compute_mujoco_control_substeps = _mujoco_runtime_utils.compute_mujoco_control_substeps
_clip_tendon_position_control = _mujoco_runtime_utils._clip_tendon_position_control
_append_trail_sample = _mujoco_runtime_utils._append_trail_sample
_draw_tracking_overlays = _mujoco_runtime_utils._draw_tracking_overlays
_sync_tracking_viewer = _mujoco_runtime_utils._sync_tracking_viewer
