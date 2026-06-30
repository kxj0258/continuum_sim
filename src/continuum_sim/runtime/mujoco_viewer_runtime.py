"""Shared MuJoCo passive viewer runtime helpers."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from continuum_sim.backends.base_types import BackendState
from continuum_sim.control.mobile_base_controller import (
    MobileBaseState,
    reset_mobile_base_state,
    set_mobile_base_locked,
)
from continuum_sim.model import ThreeSegmentRobotParams
from continuum_sim.runtime import mujoco_runtime_utils as _mujoco_runtime_utils


@dataclass
class ViewerControlState:
    """Mutable keyboard state for the passive MuJoCo viewer."""

    paused: bool = False
    step_once: bool = False
    speed: float = 1.0
    replay_requested: bool = False
    replay_index: int = 0
    base_state: MobileBaseState = field(default_factory=reset_mobile_base_state)
    arm_locked: bool = False
    fine_motion: bool = False


def _create_tendon_live_panel(
    *,
    show_viewer: bool,
    control_mode: str,
    show_live_tendon_panel: bool,
    config,
    params: ThreeSegmentRobotParams,
    physical_tendons,
    initial_state: BackendState,
):
    if not show_viewer or control_mode != "tendon_position" or not show_live_tendon_panel:
        return None

    from continuum_sim.visualization.mujoco_tendon_debug_viewer import (
        MujocoTendonMonitorPanel,
    )

    tendon_indices = None
    if getattr(config.model, "type", None) == "dual_distributed_links":
        from continuum_sim.model.dual_arm_robot import load_dual_arm_robot_config

        dual_robot = load_dual_arm_robot_config(config.robot_config_path)
        arm_index = dual_robot.arm_names.index(dual_robot.default_arm)
        start = arm_index * len(tuple(physical_tendons))
        tendon_indices = np.arange(start, start + len(tuple(physical_tendons)), dtype=int)
    panel = MujocoTendonMonitorPanel(
        config,
        params,
        tuple(physical_tendons),
        title="continuum_sim MuJoCo tendon tracking monitor",
        tendon_indices=tendon_indices,
    )
    panel.update_from_state(
        np.zeros(len(tuple(physical_tendons)), dtype=float),
        initial_state,
        redraw=False,
    )
    panel.show(block=False)
    return panel


def _sync_tendon_live_panel(
    panel,
    stride: int,
    commanded_tendon_delta: np.ndarray,
    state: BackendState,
    sample_index: int,
) -> None:
    if panel is None:
        return
    if stride <= 0:
        raise ValueError(f"live_tendon_panel_stride must be positive, got {stride}.")
    if sample_index % stride == 0:
        panel.update_from_state(commanded_tendon_delta, state, redraw=True)
    panel.flush_events()


def _update_tendon_live_panel_from_history(
    panel,
    *,
    index: int,
    controller_dt: float,
    mujoco_control_history: Sequence[np.ndarray],
    tip_pose_history: Sequence[np.ndarray],
    qpos_history: Sequence[np.ndarray],
    qvel_history: Sequence[np.ndarray],
    tendon_length_history: Sequence[np.ndarray],
    actuator_force_history: Sequence[np.ndarray],
) -> None:
    if panel is None:
        return
    tendon_length = np.asarray(tendon_length_history[index], dtype=float).copy()
    panel.update_from_state(
        np.asarray(mujoco_control_history[index], dtype=float).copy(),
        BackendState(
            time=index * controller_dt,
            tip_pose=np.asarray(tip_pose_history[index], dtype=float).copy(),
            segment_poses=np.zeros((3, 4, 4), dtype=float),
            qpos=np.asarray(qpos_history[index], dtype=float).copy(),
            qvel=np.asarray(qvel_history[index], dtype=float).copy(),
            tendon_length=tendon_length,
            tendon_velocity=np.zeros_like(tendon_length),
            actuator_force=np.asarray(actuator_force_history[index], dtype=float).copy(),
        ),
        redraw=True,
    )
    panel.flush_events()


def _make_viewer_key_callback(control: ViewerControlState):
    def key_callback(keycode: int) -> None:
        _handle_viewer_key(control, keycode)

    return key_callback


def _handle_viewer_key(control: ViewerControlState, keycode: int) -> None:
    if keycode == 32:
        control.paused = not control.paused
        return
    try:
        key = chr(keycode).lower()
    except (TypeError, ValueError):
        return
    if key == ".":
        control.paused = True
        control.step_once = True
    elif key in ("+", "="):
        control.speed = min(control.speed * 1.25, 8.0)
    elif key in ("-", "_"):
        control.speed = max(control.speed / 1.25, 0.125)
    elif key == "r":
        control.replay_requested = True
        control.replay_index = 0
    elif key == "b":
        control.base_state = set_mobile_base_locked(control.base_state, not control.base_state.locked)
    elif key == "h":
        control.base_state = reset_mobile_base_state()


def _maybe_hold_tracking_viewer_open_after_run(
    *,
    keep_viewer_open_after_run: bool,
    viewer,
    backend,
    mujoco_module,
    control: ViewerControlState,
    tendon_overlay: TendonOverlayContext | None,
    tendon_monitor_panel,
    overlay_config,
    target_history: Sequence[np.ndarray],
    tip_pose_history: Sequence[np.ndarray],
    qpos_history: Sequence[np.ndarray],
    qvel_history: Sequence[np.ndarray],
    mujoco_control_history: Sequence[np.ndarray],
    tendon_length_history: Sequence[np.ndarray],
    actuator_force_history: Sequence[np.ndarray],
    controller_dt: float,
    realtime: bool,
    realtime_factor: float,
) -> None:
    if not keep_viewer_open_after_run:
        return
    _idle_tracking_viewer(
        viewer,
        backend,
        mujoco_module,
        control,
        tendon_overlay,
        tendon_monitor_panel,
        overlay_config,
        target_history,
        tip_pose_history,
        qpos_history,
        qvel_history,
        mujoco_control_history,
        tendon_length_history,
        actuator_force_history,
        controller_dt,
        realtime,
        realtime_factor,
    )


def _idle_tracking_viewer(
    viewer,
    backend,
    mujoco_module,
    control: ViewerControlState,
    tendon_overlay: TendonOverlayContext | None,
    tendon_monitor_panel,
    overlay_config,
    target_history: Sequence[np.ndarray],
    tip_pose_history: Sequence[np.ndarray],
    qpos_history: Sequence[np.ndarray],
    qvel_history: Sequence[np.ndarray],
    mujoco_control_history: Sequence[np.ndarray],
    tendon_length_history: Sequence[np.ndarray],
    actuator_force_history: Sequence[np.ndarray],
    controller_dt: float,
    realtime: bool,
    realtime_factor: float,
) -> None:
    if not target_history or not tip_pose_history or not qpos_history:
        return
    while viewer.is_running():
        if control.replay_requested:
            _run_tracking_replay(
                viewer,
                backend,
                mujoco_module,
                control,
                tendon_overlay,
                tendon_monitor_panel,
                overlay_config,
                target_history,
                tip_pose_history,
                qpos_history,
                qvel_history,
                mujoco_control_history,
                tendon_length_history,
                actuator_force_history,
                controller_dt,
                realtime_factor,
            )
            continue
        _draw_replay_overlays(
            viewer,
            mujoco_module,
            tendon_overlay,
            overlay_config,
            len(qpos_history) - 1,
            target_history,
            tip_pose_history,
        )
        if tendon_monitor_panel is not None:
            tendon_monitor_panel.flush_events()
        viewer.sync()
        time.sleep(0.03 if realtime else 0.01)


def _run_tracking_replay(
    viewer,
    backend,
    mujoco_module,
    control: ViewerControlState,
    tendon_overlay: TendonOverlayContext | None,
    tendon_monitor_panel,
    overlay_config,
    target_history: Sequence[np.ndarray],
    tip_pose_history: Sequence[np.ndarray],
    qpos_history: Sequence[np.ndarray],
    qvel_history: Sequence[np.ndarray],
    mujoco_control_history: Sequence[np.ndarray],
    tendon_length_history: Sequence[np.ndarray],
    actuator_force_history: Sequence[np.ndarray],
    controller_dt: float,
    realtime_factor: float,
) -> None:
    control.replay_requested = False
    control.replay_index = min(control.replay_index, len(qpos_history) - 1)
    while viewer.is_running() and control.replay_index < len(qpos_history):
        if control.replay_requested:
            control.replay_requested = False
            control.replay_index = 0
        index = control.replay_index
        _restore_replay_frame(backend, qpos_history, qvel_history, index)
        _draw_replay_overlays(
            viewer,
            mujoco_module,
            tendon_overlay,
            overlay_config,
            index,
            target_history,
            tip_pose_history,
        )
        _update_tendon_live_panel_from_history(
            tendon_monitor_panel,
            index=index,
            controller_dt=controller_dt,
            mujoco_control_history=mujoco_control_history,
            tip_pose_history=tip_pose_history,
            qpos_history=qpos_history,
            qvel_history=qvel_history,
            tendon_length_history=tendon_length_history,
            actuator_force_history=actuator_force_history,
        )
        viewer.sync()
        if control.paused and not control.step_once:
            time.sleep(0.02)
            continue
        if control.step_once:
            control.step_once = False
        control.replay_index += 1
        time.sleep(controller_dt / _effective_realtime_factor(realtime_factor, control.speed))
    if control.replay_index >= len(qpos_history):
        control.replay_index = 0


def _restore_replay_frame(
    backend,
    qpos_history: Sequence[np.ndarray],
    qvel_history: Sequence[np.ndarray],
    index: int,
) -> None:
    backend.data.qpos[:] = qpos_history[index]
    backend.data.qvel[:] = qvel_history[index]
    update_followers = getattr(backend, "update_follower_poses", None)
    if update_followers is not None:
        update_followers()
    backend._mujoco.mj_forward(backend.model, backend.data)


def _state_mocap_pos(state: BackendState) -> np.ndarray:
    values = getattr(state, "mocap_pos", None)
    if values is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(values, dtype=float).copy()


def _state_mocap_quat(state: BackendState) -> np.ndarray:
    values = getattr(state, "mocap_quat", None)
    if values is None:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(values, dtype=float).copy()


def _draw_replay_overlays(
    viewer,
    mujoco_module,
    tendon_overlay: TendonOverlayContext | None,
    overlay_config,
    index: int,
    target_history: Sequence[np.ndarray],
    tip_pose_history: Sequence[np.ndarray],
) -> None:
    tip_positions = [
        np.asarray(pose[:3, 3], dtype=float)
        for pose in tip_pose_history[: index + 1]
    ]
    target_positions = [
        np.asarray(target, dtype=float)
        for target in target_history[: index + 1]
    ]
    _draw_tracking_overlays(
        viewer,
        mujoco_module,
        tendon_overlay,
        overlay_config,
        target_history[index],
        _history_trail_points(
            tip_positions,
            index,
            overlay_config.trail_stride,
            overlay_config.trail_max_points,
        ),
        _history_trail_points(
            target_positions,
            index,
            overlay_config.trail_stride,
            overlay_config.trail_max_points,
        ),
    )


TendonOverlayContext = _mujoco_runtime_utils.TendonOverlayContext
_draw_tracking_overlays = _mujoco_runtime_utils._draw_tracking_overlays
_history_trail_points = _mujoco_runtime_utils._history_trail_points
_effective_realtime_factor = _mujoco_runtime_utils._effective_realtime_factor
