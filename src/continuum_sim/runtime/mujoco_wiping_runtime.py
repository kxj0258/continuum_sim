"""Run hybrid force-position wiping through the MuJoCo backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.actuation.motor_mapping import (
    motor_position_to_tendon_delta,
    motor_velocity_to_tendon_velocity,
    tendon_delta_to_motor_position,
)
from continuum_sim.backends import MujocoBackend
from continuum_sim.config import load_mujoco_config
from continuum_sim.control.adaptive_impedance import (
    AdaptiveImpedanceConfig,
    compute_dynamic_wiping_motor_velocity_command_from_state,
)
from continuum_sim.control.hybrid_force_position import (
    compute_wiping_motor_velocity_command_from_observation,
    compute_wiping_motor_velocity_command_from_state,
)
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model import (
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
)
from continuum_sim.dynamics import load_pcc_dynamics_config
from continuum_sim.runtime.mujoco_contact_projection import (
    FOLLOWER_CONTACT_SOURCE,
    apply_projected_qfrc,
    project_follower_contacts,
)
from continuum_sim.model.tendon_coupling import physical_tendon_delta_to_q
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
from continuum_sim.scenes.scene_builder import build_mujoco_wiping_xml
from continuum_sim.scenes.scene_config import load_navigation_scene_config
from continuum_sim.tasks.wiping_config import (
    build_raster_wiping_path,
    load_mujoco_wiping_config,
)


@dataclass(frozen=True)
class MujocoWipingResult:
    """Recorded samples from a MuJoCo hybrid wiping rollout."""

    time: np.ndarray
    target_position: np.ndarray
    target_pose: np.ndarray
    phase: tuple[str, ...]
    waypoint_index: np.ndarray
    tip_pose: np.ndarray
    segment_poses: np.ndarray
    error_norm: np.ndarray
    normal_force_n: np.ndarray
    contact_proxy_m: np.ndarray
    force_error_n: np.ndarray
    contact_source: tuple[str, ...]
    in_contact: np.ndarray
    motor_position: np.ndarray
    motor_velocity: np.ndarray
    tendon_delta: np.ndarray
    q_est: np.ndarray
    predicted_q: np.ndarray
    predicted_qdot: np.ndarray
    predicted_qddot: np.ndarray
    stiffness_diag: np.ndarray
    damping_diag: np.ndarray
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


def run_mujoco_wiping(
    task_config_path: str | Path = Path("configs/tasks/mujoco_wiping_board.yaml"),
    mujoco_config_path: str | Path = Path("configs/mujoco.yaml"),
    *,
    show: bool | None = None,
) -> MujocoWipingResult:
    """Run raster wiping against a structured work surface."""

    task_config = load_mujoco_wiping_config(task_config_path)
    mujoco_config = load_mujoco_config(mujoco_config_path)
    scene_config = load_navigation_scene_config(task_config.scene.config_path)
    surface = scene_config.work_surface(task_config.motion.surface_id)
    wipe_path = build_raster_wiping_path(
        task_config.motion,
        surface,
        contact_radius_m=task_config.tool.radius_m,
    )
    scene_xml_path = build_mujoco_wiping_xml(
        _base_xml_path(mujoco_config),
        scene_config,
        task_config.tool.to_xml_config(),
        task_config.scene.generated_xml_path,
        tip_site_name=mujoco_config.site_names.tip,
        mobile_base_config_path=mujoco_config.mobile_base_config_path,
        offscreen_size=(
            mujoco_config.rendering.offscreen_width,
            mujoco_config.rendering.offscreen_height,
        ),
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
    adaptive_config = _adaptive_impedance_config(task_config, params)
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
    contact_loss_steps = 0
    times: list[float] = []
    target_history: list[np.ndarray] = []
    target_pose_history: list[np.ndarray] = []
    phase_history: list[str] = []
    waypoint_history: list[int] = []
    tip_pose_history: list[np.ndarray] = []
    segment_pose_history: list[np.ndarray] = []
    error_history: list[float] = []
    normal_force_history: list[float] = []
    contact_proxy_history: list[float] = []
    force_error_history: list[float] = []
    contact_source_history: list[str] = []
    in_contact_history: list[bool] = []
    motor_position_history: list[np.ndarray] = []
    motor_velocity_history: list[np.ndarray] = []
    tendon_delta_history: list[np.ndarray] = []
    q_history: list[np.ndarray] = []
    predicted_q_history: list[np.ndarray] = []
    predicted_qdot_history: list[np.ndarray] = []
    predicted_qddot_history: list[np.ndarray] = []
    stiffness_diag_history: list[np.ndarray] = []
    damping_diag_history: list[np.ndarray] = []
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
    force_monitor_panel = None

    def run_loop(
        viewer=None,
        mujoco_module=None,
        viewer_control: ViewerControlState | None = None,
    ) -> None:
        nonlocal motor_position, observed_state, waypoint_index, contact_loss_steps
        realtime_start_wall = time.perf_counter()
        step_index = 0
        while step_index < task_config.simulation.max_steps:
            if viewer is not None and not viewer.is_running():
                break
            current_target = wipe_path.target_position[waypoint_index]
            current_pose = wipe_path.target_pose[waypoint_index]
            current_phase = wipe_path.phase[waypoint_index]
            if (
                viewer is not None
                and mujoco_module is not None
                and viewer_control is not None
                and viewer_control.paused
                and not viewer_control.step_once
            ):
                if tendon_monitor_panel is not None:
                    tendon_monitor_panel.flush_events()
                if force_monitor_panel is not None:
                    force_monitor_panel.flush_events()
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

            projected_contact = _projected_follower_contact_for_wiping(
                backend,
                mujoco_module,
                mujoco_config,
                params,
                surface,
            )
            tool_contact_force = None
            if projected_contact is not None and projected_contact.contact_count > 0:
                measured_force = projected_contact.normal_force_n
            else:
                tool_contact_force = _normal_force_from_mujoco_contacts(
                    backend,
                    mujoco_module,
                    task_config.tool.geom_name,
                    f"scene_{_safe_name(surface.primitive_id)}",
                )
                measured_force = tool_contact_force
            if (
                projected_contact is not None
                and projected_contact.contact_count > 0
                and mujoco_config.model.apply_projected_qfrc
            ):
                apply_projected_qfrc(
                    backend.data,
                    projected_contact.projected_generalized_force_q,
                )
            if measured_force is None:
                measured_force = tool_contact_force
            contact_position = _tool_contact_position(
                backend,
                mujoco_module,
                task_config.tool.contact_site_name,
                observed_state.tip_pose,
                task_config.tool.offset_m,
            )
            if mujoco_config.control_mode != "tendon_position":
                raise ValueError(
                    "MuJoCo wiping currently requires control_mode='tendon_position'."
                )

            if task_config.mujoco.feedback_mode == "mujoco_actual":
                tendon_delta = observed_state.tendon_length.copy()
                controller_motor_position = tendon_delta_to_motor_position(
                    tendon_delta,
                    motor_params,
                )
                if task_config.controller.type == "dynamic_adaptive_impedance":
                    q_est = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)
                    qdot_est = physical_tendon_delta_to_q(
                        observed_state.tendon_velocity,
                        params,
                        physical_tendons,
                    )
                    motor_velocity_cmd, info = (
                        compute_dynamic_wiping_motor_velocity_command_from_state(
                            contact_position,
                            q_est,
                            qdot_est,
                            target_position=current_target,
                            surface=surface,
                            params=params,
                            physical_tendons=physical_tendons,
                            motor_params=motor_params,
                            wiping_config=task_config.controller,
                            adaptive_config=adaptive_config,
                            measured_normal_force_n=measured_force,
                            dt=task_config.simulation.dt,
                            contact_radius_m=task_config.tool.radius_m,
                            force_control_enabled=current_phase == "contact",
                        )
                    )
                else:
                    motor_velocity_cmd, info = (
                        compute_wiping_motor_velocity_command_from_observation(
                        contact_position,
                        tendon_delta,
                        current_target,
                        surface,
                        params,
                        physical_tendons,
                        motor_params,
                        task_config.controller,
                        measured_normal_force_n=measured_force,
                        contact_radius_m=task_config.tool.radius_m,
                        force_control_enabled=current_phase == "contact",
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
                tendon_delta = motor_position_to_tendon_delta(
                    controller_motor_position,
                    motor_params,
                )
                q_est = physical_tendon_delta_to_q(tendon_delta, params, physical_tendons)
                fk = forward_kinematics(q_est, params)
                contact_position = (
                    fk.tip_pose[:3, 3]
                    + fk.tip_pose[:3, :3] @ task_config.tool.offset_m
                )
                if task_config.controller.type == "dynamic_adaptive_impedance":
                    motor_velocity_cmd, info = (
                        compute_dynamic_wiping_motor_velocity_command_from_state(
                            contact_position,
                            q_est,
                            np.zeros(params.q_size, dtype=float),
                            target_position=current_target,
                            surface=surface,
                            params=params,
                            physical_tendons=physical_tendons,
                            motor_params=motor_params,
                            wiping_config=task_config.controller,
                            adaptive_config=adaptive_config,
                            measured_normal_force_n=None,
                            dt=task_config.simulation.dt,
                            contact_radius_m=task_config.tool.radius_m,
                            force_control_enabled=current_phase == "contact",
                        )
                    )
                else:
                    motor_velocity_cmd, info = compute_wiping_motor_velocity_command_from_state(
                        contact_position,
                        q_est,
                        current_target,
                        surface,
                        params,
                        physical_tendons,
                        motor_params,
                        task_config.controller,
                        measured_normal_force_n=None,
                        contact_radius_m=task_config.tool.radius_m,
                        force_control_enabled=current_phase == "contact",
                    )
                mujoco_control = _clip_tendon_position_control(
                    tendon_delta,
                    mujoco_config.actuators.tendon_position.ctrlrange_m,
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
            error_norm = float(np.linalg.norm(current_target - contact_position))
            normal_force = float(info["normal_force_n"])
            contact_proxy = float(info["contact_proxy_m"])
            force_error = float(info["force_error_n"])
            contact_source = _contact_source_from_feedback(projected_contact, info)
            in_contact = bool(info["in_contact"])
            sample_time = step_index * task_config.simulation.dt
            waypoint_id = int(wipe_path.waypoint_index[waypoint_index])
            _sync_force_live_panel(
                force_monitor_panel,
                task_config.mujoco.live_force_panel_stride,
                sample_index=step_index,
                time_s=sample_time,
                normal_force_n=normal_force,
                target_normal_force_n=task_config.controller.target_normal_force_n,
                force_error_n=force_error,
                contact_proxy_m=contact_proxy,
                phase=current_phase,
                waypoint_index=waypoint_id,
                contact_source=contact_source,
                in_contact=in_contact,
            )
            if current_phase == "contact" and not in_contact:
                contact_loss_steps += 1
            else:
                contact_loss_steps = 0

            times.append(sample_time)
            target_history.append(current_target.copy())
            target_pose_history.append(current_pose.copy())
            phase_history.append(current_phase)
            waypoint_history.append(waypoint_id)
            tip_pose_history.append(state.tip_pose.copy())
            segment_pose_history.append(state.segment_poses.copy())
            error_history.append(error_norm)
            normal_force_history.append(normal_force)
            contact_proxy_history.append(contact_proxy)
            force_error_history.append(force_error)
            contact_source_history.append(contact_source)
            in_contact_history.append(in_contact)
            motor_position_history.append(controller_motor_position.copy())
            motor_velocity_history.append(motor_velocity_cmd.copy())
            tendon_delta_history.append(tendon_delta.copy())
            q_history.append(q_est.copy())
            predicted_q_history.append(_info_vector(info, "predicted_q", params.q_size))
            predicted_qdot_history.append(_info_vector(info, "predicted_qdot", params.q_size))
            predicted_qddot_history.append(_info_vector(info, "predicted_qddot", params.q_size))
            stiffness_diag_history.append(_info_vector(info, "stiffness_diag", params.q_size))
            damping_diag_history.append(_info_vector(info, "damping_diag", params.q_size))
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

            should_stop = False
            if normal_force > task_config.controller.max_contact_force_n:
                should_stop = True
            if contact_loss_steps > task_config.controller.contact_loss_tolerance_steps:
                should_stop = True
            if error_norm <= task_config.motion.waypoint_tolerance_m:
                if waypoint_index < len(wipe_path.target_position) - 1:
                    waypoint_index += 1
                elif task_config.simulation.stop_on_completion:
                    should_stop = True

            if task_config.mujoco.feedback_mode == "pcc_command":
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
            if should_stop:
                break
            step_index += 1

    if show_viewer:
        import mujoco
        import mujoco.viewer

        viewer_control = ViewerControlState()
        key_callback = _make_viewer_key_callback(viewer_control)
        with mujoco.viewer.launch_passive(
            backend.model,
            backend.data,
            key_callback=key_callback,
        ) as viewer:
            try:
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
                force_monitor_panel = _create_force_live_panel(
                    show_viewer=show_viewer,
                    show_live_force_panel=task_config.mujoco.show_live_force_panel,
                    target_normal_force_n=task_config.controller.target_normal_force_n,
                    history_points=task_config.mujoco.live_force_panel_history_points,
                )
                _draw_tracking_overlays(
                    viewer,
                    mujoco,
                    tendon_overlay,
                    mujoco_config.viewer.overlays,
                    wipe_path.target_position[waypoint_index],
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
                if force_monitor_panel is not None:
                    force_monitor_panel.close()
                if tendon_monitor_panel is not None:
                    tendon_monitor_panel.close()
    else:
        run_loop()

    return MujocoWipingResult(
        time=np.asarray(times, dtype=float),
        target_position=np.asarray(target_history, dtype=float),
        target_pose=np.asarray(target_pose_history, dtype=float),
        phase=tuple(phase_history),
        waypoint_index=np.asarray(waypoint_history, dtype=int),
        tip_pose=np.asarray(tip_pose_history, dtype=float),
        segment_poses=np.asarray(segment_pose_history, dtype=float),
        error_norm=np.asarray(error_history, dtype=float),
        normal_force_n=np.asarray(normal_force_history, dtype=float),
        contact_proxy_m=np.asarray(contact_proxy_history, dtype=float),
        force_error_n=np.asarray(force_error_history, dtype=float),
        contact_source=tuple(contact_source_history),
        in_contact=np.asarray(in_contact_history, dtype=bool),
        motor_position=np.asarray(motor_position_history, dtype=float),
        motor_velocity=np.asarray(motor_velocity_history, dtype=float),
        tendon_delta=np.asarray(tendon_delta_history, dtype=float),
        q_est=np.asarray(q_history, dtype=float),
        predicted_q=np.asarray(predicted_q_history, dtype=float),
        predicted_qdot=np.asarray(predicted_qdot_history, dtype=float),
        predicted_qddot=np.asarray(predicted_qddot_history, dtype=float),
        stiffness_diag=np.asarray(stiffness_diag_history, dtype=float),
        damping_diag=np.asarray(damping_diag_history, dtype=float),
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


def _normal_force_from_mujoco_contacts(
    backend,
    mujoco_module,
    tool_geom_name: str,
    surface_geom_name: str,
) -> float | None:
    """Return summed normal contact force when MuJoCo contact APIs are available."""

    module = mujoco_module or getattr(backend, "_mujoco", None)
    model = getattr(backend, "model", None)
    data = getattr(backend, "data", None)
    if module is None or model is None or data is None:
        return None
    name2id = getattr(module, "mj_name2id", None)
    obj = getattr(module, "mjtObj", None)
    contact_force = getattr(module, "mj_contactForce", None)
    if name2id is None or obj is None or contact_force is None:
        return None
    tool_id = int(name2id(model, obj.mjOBJ_GEOM, tool_geom_name))
    surface_id = int(name2id(model, obj.mjOBJ_GEOM, surface_geom_name))
    if tool_id < 0 or surface_id < 0:
        return None
    total = 0.0
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if geom_pair != {tool_id, surface_id}:
            continue
        force = np.zeros(6, dtype=float)
        contact_force(model, data, index, force)
        total += abs(float(force[0]))
    return total


def _projected_follower_contact_for_wiping(
    backend,
    mujoco_module,
    mujoco_config,
    params: ThreeSegmentRobotParams,
    surface,
):
    """Return follower contact projection when enabled and MuJoCo APIs exist."""

    if (
        mujoco_config.model.type != "segment_2dof_followers"
        or not mujoco_config.model.contact_force_projection
    ):
        return None
    module = mujoco_module or getattr(backend, "_mujoco", None)
    model = getattr(backend, "model", None)
    data = getattr(backend, "data", None)
    if module is None or model is None or data is None:
        return None
    qpos = np.asarray(getattr(data, "qpos", np.zeros(0, dtype=float)), dtype=float)
    if qpos.shape[0] < 6:
        return None
    return project_follower_contacts(
        mujoco_module=module,
        model=model,
        data=data,
        q_segment=qpos[:6],
        params=params,
        samples_per_segment=mujoco_config.model.follower_samples_per_segment,
        surface=surface,
    )


def _contact_source_from_feedback(projected_contact, info: dict) -> str:
    if projected_contact is not None and projected_contact.contact_count > 0:
        return FOLLOWER_CONTACT_SOURCE
    return str(info["contact_source"])


def _adaptive_impedance_config(task_config, params: ThreeSegmentRobotParams) -> AdaptiveImpedanceConfig:
    path = getattr(task_config.controller, "dynamics_config_path", None)
    if path is None:
        return AdaptiveImpedanceConfig.default(params)
    return AdaptiveImpedanceConfig(load_pcc_dynamics_config(path, params))


def _info_vector(info: dict, name: str, size: int) -> np.ndarray:
    value = info.get(name)
    if value is None:
        return np.full(size, np.nan, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"info[{name!r}] must have shape ({size},), got {array.shape}.")
    return array.copy()


def _tool_contact_position(
    backend,
    mujoco_module,
    contact_site_name: str,
    tip_pose: np.ndarray,
    tool_offset_m: np.ndarray,
) -> np.ndarray:
    """Return the tool contact site world position, falling back to tip pose."""

    module = mujoco_module or getattr(backend, "_mujoco", None)
    model = getattr(backend, "model", None)
    data = getattr(backend, "data", None)
    if module is not None and model is not None and data is not None:
        name2id = getattr(module, "mj_name2id", None)
        obj = getattr(module, "mjtObj", None)
        site_xpos = getattr(data, "site_xpos", None)
        if name2id is not None and obj is not None and site_xpos is not None:
            site_id = int(name2id(model, obj.mjOBJ_SITE, contact_site_name))
            if site_id >= 0:
                return np.asarray(site_xpos[site_id], dtype=float).copy()
    pose = np.asarray(tip_pose, dtype=float)
    offset = np.asarray(tool_offset_m, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected tip_pose with shape (4, 4), got {pose.shape}.")
    if offset.shape != (3,):
        raise ValueError(f"Expected tool_offset_m with shape (3,), got {offset.shape}.")
    return pose[:3, 3] + pose[:3, :3] @ offset


def _create_force_live_panel(
    *,
    show_viewer: bool,
    show_live_force_panel: bool,
    target_normal_force_n: float,
    history_points: int,
):
    if not show_viewer or not show_live_force_panel:
        return None

    from continuum_sim.visualization.wiping_force_panel import WipingForceMonitorPanel

    panel = WipingForceMonitorPanel(
        target_normal_force_n=target_normal_force_n,
        history_points=history_points,
    )
    panel.show(block=False)
    return panel


def _sync_force_live_panel(
    panel,
    stride: int,
    *,
    sample_index: int,
    time_s: float,
    normal_force_n: float,
    target_normal_force_n: float,
    force_error_n: float,
    contact_proxy_m: float,
    phase: str,
    waypoint_index: int,
    contact_source: str,
    in_contact: bool,
) -> None:
    if panel is None:
        return
    if stride <= 0:
        raise ValueError(f"live_force_panel_stride must be positive, got {stride}.")
    if sample_index % stride == 0:
        panel.update(
            time_s=time_s,
            normal_force_n=normal_force_n,
            target_normal_force_n=target_normal_force_n,
            force_error_n=force_error_n,
            contact_proxy_m=contact_proxy_m,
            phase=phase,
            waypoint_index=waypoint_index,
            contact_source=contact_source,
            in_contact=in_contact,
            redraw=True,
        )
    panel.flush_events()


def _base_xml_path(config) -> Path:
    visual_xml_path = _resolve_visual_xml_path(config, config.viewer.use_segment_visuals)
    if visual_xml_path is not None:
        return visual_xml_path
    if config.control_mode == "position_joint":
        return config.xml_path
    if config.control_mode == "tendon_position":
        return config.tendon_xml_path
    raise ValueError(f"Unsupported MuJoCo control_mode {config.control_mode!r}.")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


TendonOverlayContext = _mujoco_runtime_utils.TendonOverlayContext
_resolve_visual_xml_path = _mujoco_runtime_utils._resolve_visual_xml_path
_configure_viewer_groups = _mujoco_runtime_utils._configure_viewer_groups
_configure_viewer_camera = _mujoco_runtime_utils._configure_viewer_camera
compute_mujoco_control_substeps = _mujoco_runtime_utils.compute_mujoco_control_substeps
_clip_tendon_position_control = _mujoco_runtime_utils._clip_tendon_position_control
_append_trail_sample = _mujoco_runtime_utils._append_trail_sample
_draw_tracking_overlays = _mujoco_runtime_utils._draw_tracking_overlays
_sync_tracking_viewer = _mujoco_runtime_utils._sync_tracking_viewer
