"""Native artifact export for scenario-driven system simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.io.mujoco_pcc_diagnostics import (
    build_mujoco_pcc_diagnostic_arrays,
)
from continuum_sim.visualization.mujoco_video import save_replay_video


@dataclass(frozen=True)
class ScenarioArtifactPaths:
    run_dir: Path
    result_npz: Path
    metadata_json: Path
    plots_dir: Path
    videos_dir: Path


def save_scenario_artifacts(application, result) -> ScenarioArtifactPaths | None:
    """Persist one named-system run without depending on legacy result types."""

    config = application.config
    settings = config.artifacts
    if not settings.enabled:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _unique_dir(settings.output_root / f"{config.name}_{stamp}")
    paths = ScenarioArtifactPaths(
        run_dir=run_dir,
        result_npz=run_dir / "result.npz",
        metadata_json=run_dir / "metadata.json",
        plots_dir=run_dir / "plots",
        videos_dir=run_dir / "videos",
    )
    paths.plots_dir.mkdir(parents=True)
    paths.videos_dir.mkdir(parents=True)
    arrays = _flatten_result(application, result)
    errors: list[str] = []
    if config.backend.type == "mujoco":
        try:
            arrays.update(
                build_mujoco_pcc_diagnostic_arrays(
                    application.loop.backend.assembly,
                    result.states,
                    result.commands,
                )
            )
        except Exception as exc:  # Diagnostics must not discard the base artifacts.
            errors.append(
                f"mujoco_pcc_diagnostics: {type(exc).__name__}: {exc}"
            )
    if settings.save_npz:
        np.savez_compressed(paths.result_npz, **arrays)
    config_dir = run_dir / "configs"
    config_dir.mkdir()
    shutil.copy2(config.path, config_dir / "scenario.yaml")
    shutil.copy2(config.assembly_config_path, config_dir / "assembly.yaml")
    if config.low_level_control_path is not None:
        shutil.copy2(
            config.low_level_control_path,
            config_dir / "low_level_control.yaml",
        )
    if config.backend.mujoco_config_path is not None:
        shutil.copy2(config.backend.mujoco_config_path, config_dir / "mujoco.yaml")
    scene_xml = _copy_scene_model(application, run_dir / "model") if settings.save_model else None
    plot_files: list[str] = []
    if settings.save_plots:
        try:
            plot_files = [str(path) for path in _save_plots(arrays, paths.plots_dir)]
        except Exception as exc:  # Artifact failure must not discard simulation data.
            errors.append(f"plots: {type(exc).__name__}: {exc}")
    metadata = {
        "scenario": config.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": config.backend.type,
        "task": config.task.type,
        "arms": sorted(result.states[-1].arms),
        "samples": len(result.states),
        "commands": len(result.commands),
        "stopped_early": result.stopped_early,
        "stop_reason": result.metadata.get("stop_reason", ""),
        "metrics": _metrics(arrays),
        "npz_keys": sorted(arrays) if settings.save_npz else [],
        "plots": plot_files,
        "video": None,
        "video_status": "disabled" if not settings.save_gif else "pending",
        "video_mode": settings.video_mode if settings.save_gif else None,
        "video_frames": None,
        "model": None if scene_xml is None else str(scene_xml),
        "errors": errors,
    }
    _write_metadata(paths.metadata_json, metadata)
    video_path = None
    if settings.save_gif:
        if settings.video_mode == "live_mujoco":
            video_path = _collect_live_mujoco_video(
                application,
                paths.videos_dir / "simulation.gif",
                errors,
            )
        else:
            try:
                replay = _replay_result(application, arrays, scene_xml)
                video_path = save_replay_video(
                    replay,
                    paths.videos_dir / "simulation.gif",
                    fps=settings.video_fps,
                    stride=settings.video_stride,
                    camera=(
                        None
                        if config.backend.type != "mujoco"
                        else application.loop.backend.config.viewer.camera
                    ),
                )
            except Exception as exc:  # Video errors are reported alongside successful data.
                errors.append(f"video: {type(exc).__name__}: {exc}")
        if video_path is None:
            _collect_video_error_file(paths.videos_dir / "simulation.gif", errors)
    metadata["video"] = None if video_path is None else str(video_path)
    metadata["video_status"] = (
        "disabled"
        if not settings.save_gif
        else "ok"
        if video_path is not None
        else "failed"
    )
    metadata["video_frames"] = _video_frame_count(application)
    metadata["errors"] = errors
    _write_metadata(paths.metadata_json, metadata)
    return paths


def _video_frame_count(application) -> int | None:
    recorder = application.hooks_by_name.get("live_mujoco_video")
    if recorder is None:
        return None
    return int(getattr(recorder, "frame_count", 0))


def _collect_live_mujoco_video(application, destination: Path, errors: list[str]) -> Path | None:
    recorder = application.hooks_by_name.get("live_mujoco_video")
    if recorder is None:
        errors.append("video: live_mujoco recorder was not attached")
        return None
    for error in getattr(recorder, "errors", []):
        errors.append(f"video: {error}")
    source = getattr(recorder, "path", None)
    if source is None:
        source = getattr(recorder, "output_path", None)
    source_path = None if source is None else Path(source)
    if source_path is None or not source_path.is_file():
        if not getattr(recorder, "errors", []):
            errors.append("video: live_mujoco recorder produced no video file")
        _write_live_video_error(destination, recorder)
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), destination)
    _write_live_video_error(destination, recorder)
    return destination


def _write_live_video_error(destination: Path, recorder: object) -> None:
    messages = list(getattr(recorder, "errors", []))
    pending_error = Path(getattr(recorder, "output_path", destination)).parent / "video_error.txt"
    if pending_error.is_file():
        for line in pending_error.read_text(encoding="utf-8").splitlines():
            if line and line not in messages:
                messages.append(line)
        pending_error.unlink()
    if not messages:
        return
    (destination.parent / "video_error.txt").write_text(
        "\n".join(messages) + "\n",
        encoding="utf-8",
    )


def _collect_video_error_file(destination: Path, errors: list[str]) -> None:
    error_path = destination.parent / "video_error.txt"
    if not error_path.is_file():
        if not any(error.startswith("video:") for error in errors):
            errors.append("video: export produced no video file")
        return
    for line in error_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        message = f"video: {line}"
        if message not in errors:
            errors.append(message)


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _flatten_result(application, result) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "time_s": np.asarray([state.time_s for state in result.states], dtype=float),
        "command_time_s": np.asarray(
            [state.time_s for state in result.states[: len(result.commands)]],
            dtype=float,
        ),
        "base_position_m": np.asarray(
            [state.base.pose.position for state in result.states], dtype=float
        ),
        "base_quat_wxyz": np.asarray(
            [state.base.pose.quat for state in result.states], dtype=float
        ),
        "base_twist_world": np.asarray(
            [state.base.twist_world for state in result.states], dtype=float
        ),
    }
    mujoco_mobile_base_pose_rpy = _state_metadata_array(
        result.states,
        "mujoco_mobile_base_pose_rpy",
        shape=(6,),
    )
    if mujoco_mobile_base_pose_rpy is not None:
        arrays["mujoco_mobile_base_pose_rpy"] = mujoco_mobile_base_pose_rpy
    mujoco_mobile_base_frame_pose = _state_metadata_array(
        result.states,
        "mujoco_mobile_base_frame_pose",
        shape=(4, 4),
    )
    if mujoco_mobile_base_frame_pose is not None:
        arrays["mujoco_mobile_base_frame_pose"] = mujoco_mobile_base_frame_pose
    for arm_name in sorted(result.states[-1].arms):
        prefix = f"arm_{arm_name}"
        arrays[f"{prefix}_tip_position_m"] = np.asarray(
            [state.arms[arm_name].tip_pose_world.position for state in result.states]
        )
        arrays[f"{prefix}_tip_quat_wxyz"] = np.asarray(
            [state.arms[arm_name].tip_pose_world.quat for state in result.states]
        )
        tendon_displacement = np.asarray(
            [state.arms[arm_name].tendon_displacement_m for state in result.states]
        )
        tendon_velocity = np.asarray(
            [state.arms[arm_name].tendon_velocity_mps for state in result.states]
        )
        tendon_target = np.asarray(
            [state.arms[arm_name].tendon_target_m for state in result.states]
        )
        actuator_force = np.asarray(
            [state.arms[arm_name].actuator_force_n for state in result.states]
        )
        arrays[f"{prefix}_tendon_displacement_m"] = tendon_displacement
        arrays[f"{prefix}_tendon_velocity_mps"] = tendon_velocity
        arrays[f"{prefix}_tendon_velocity_sensor_raw_mps"] = tendon_velocity.copy()
        arrays[f"{prefix}_tendon_target_m"] = tendon_target
        arrays[f"{prefix}_actuator_force_n"] = actuator_force
        arrays[f"{prefix}_command_rate_mps"] = np.asarray(
            [command.arms[arm_name].tendon_rate_mps for command in result.commands]
        )
        command_count = len(result.commands)
        tendon_shape = (tendon_displacement.shape[1],)
        inner_loop_mode = _arm_transition_metadata_array(
            result.states,
            command_count,
            arm_name,
            "inner_loop_mode",
            shape=(),
            dtype="<U32",
            default="unknown",
        )
        arrays[f"{prefix}_tendon_inner_loop_mode"] = inner_loop_mode
        arrays[f"{prefix}_tendon_target_mode"] = _arm_transition_metadata_array(
            result.states,
            command_count,
            arm_name,
            "target_mode",
            shape=(),
            dtype="<U32",
            default="unknown",
        )
        arrays[f"{prefix}_tendon_servo_evaluated"] = (
            inner_loop_mode == "bending_rate_servo"
        )
        constrained_rate = _arm_transition_metadata_array(
            result.states,
            command_count,
            arm_name,
            "constrained_rate_mps",
            shape=tendon_shape,
            aliases=("applied_rate_mps",),
        )
        arrays[f"{prefix}_constrained_command_rate_mps"] = constrained_rate
        arrays[f"{prefix}_applied_rate_mps"] = constrained_rate.copy()
        arrays[f"{prefix}_tendon_target_rate_fd_mps"] = _finite_difference_rate(
            tendon_target,
            arrays["time_s"],
        )[:command_count]
        arrays[f"{prefix}_tendon_realized_rate_fd_mps"] = _finite_difference_rate(
            tendon_displacement,
            arrays["time_s"],
        )[:command_count]
        arrays[f"{prefix}_tendon_measured_rate_filtered_mps"] = (
            _arm_transition_metadata_array(
                result.states,
                command_count,
                arm_name,
                "measured_rate_mps",
                shape=tendon_shape,
            )
        )
        arrays[f"{prefix}_tendon_target_lead_m"] = (
            tendon_target[1 : command_count + 1]
            - tendon_displacement[1 : command_count + 1]
        )
        for key, metadata_key in (
            ("tendon_target_lead_utilization", "target_lead_utilization"),
            ("tendon_rate_error_integral_m", "rate_error_integral_m"),
            ("tendon_anti_windup_correction_m", "anti_windup_correction_m"),
            ("actuator_force_utilization", "actuator_force_utilization"),
        ):
            arrays[f"{prefix}_{key}"] = _arm_transition_metadata_array(
                result.states,
                command_count,
                arm_name,
                metadata_key,
                shape=tendon_shape,
            )
        for key, metadata_key in (
            ("rate_saturated", "rate"),
            ("tendon_target_rate_saturated", "target_rate"),
            ("displacement_saturated", "displacement"),
            ("tendon_lead_saturated", "lead"),
            ("tendon_force_constraint_active", "force_constraint_active"),
            ("tendon_anti_windup_active", "anti_windup_active"),
            ("actuator_force_at_limit", "actuator_force_at_limit"),
        ):
            arrays[f"{prefix}_{key}"] = _arm_transition_metadata_array(
                result.states,
                command_count,
                arm_name,
                metadata_key,
                shape=tendon_shape,
                dtype=bool,
            )
        arrays[f"{prefix}_tendon_guard_feasible"] = (
            _arm_transition_metadata_array(
                result.states,
                command_count,
                arm_name,
                "guard_feasible",
                shape=(),
                dtype=bool,
                default=False,
            )
        )
        for key, metadata_key in (
            (
                "tendon_compatibility_bypassed_for_safety",
                "compatibility_bypassed_for_safety",
            ),
            ("tendon_hold_target_retained", "hold_target_retained"),
        ):
            arrays[f"{prefix}_{key}"] = _arm_transition_metadata_array(
                result.states,
                command_count,
                arm_name,
                metadata_key,
                shape=(),
                dtype=bool,
            )
        arrays[f"{prefix}_tendon_max_constraint_violation_m"] = (
            _arm_transition_metadata_array(
                result.states,
                command_count,
                arm_name,
                "max_constraint_violation_m",
                shape=(),
            )
        )
    recorder = application.hooks_by_name.get("recorder")
    if recorder is not None and getattr(recorder, "engine_navigation_phase", ()):
        arrays["engine_navigation_phase"] = np.asarray(
            recorder.engine_navigation_phase,
            dtype=str,
        )
        arrays["engine_navigation_terminal_reason"] = np.asarray(
            recorder.engine_navigation_terminal_reason,
            dtype=str,
        )
        arrays["engine_navigation_progress"] = np.asarray(
            recorder.engine_navigation_progress,
            dtype=float,
        )
        arrays["base_target_position_m"] = np.asarray(
            recorder.base_target_position_m,
            dtype=float,
        )
        arrays["base_position_error_m"] = np.asarray(
            recorder.base_position_error_m,
            dtype=float,
        )
        arrays["base_orientation_error_rad"] = np.asarray(
            recorder.base_orientation_error_rad,
            dtype=float,
        )
    if recorder is not None and recorder.target_position_m:
        arrays["target_position_m"] = np.asarray(recorder.target_position_m)
        target_count = len(recorder.target_position_m)
        for key, attribute, dtype in (
            (
                "target_actual_position_m",
                "target_actual_position_m",
                float,
            ),
            (
                "target_engine_local_path_name",
                "target_engine_local_path_name",
                str,
            ),
            (
                "target_engine_local_path_type",
                "target_engine_local_path_type",
                str,
            ),
            (
                "target_engine_executor_subphase",
                "target_engine_executor_subphase",
                str,
            ),
            (
                "target_engine_local_path_center_m",
                "target_engine_local_path_center_m",
                float,
            ),
            (
                "target_engine_insertion_direction_world",
                "target_engine_insertion_direction_world",
                float,
            ),
        ):
            values = getattr(recorder, attribute, ())
            if len(values) == target_count:
                arrays[key] = np.asarray(values, dtype=dtype)
        arrays["tracking_error_m"] = np.asarray(recorder.tracking_error_m)
        arrays["achieved_waypoint_error_m"] = np.asarray(
            getattr(recorder, "achieved_waypoint_error_m", ()),
            dtype=float,
        )
        arrays["waypoint_advanced"] = np.asarray(
            getattr(recorder, "waypoint_advanced", ()),
            dtype=bool,
        )
        arrays["tracking_complete"] = np.asarray(
            getattr(recorder, "tracking_complete", ()),
            dtype=bool,
        )
        arrays["tracking_approach"] = np.asarray(
            getattr(recorder, "tracking_approach", ()),
            dtype=bool,
        )
        arrays["waypoint_index"] = np.asarray(recorder.waypoint_index, dtype=int)
        arrays["min_clearance_m"] = np.asarray(recorder.min_clearance_m)
        arrays["contact_distance_m"] = np.asarray(recorder.contact_distance_m)
        arrays["contact_error_m"] = np.asarray(recorder.contact_error_m)
        arrays["target_force_n"] = np.asarray(recorder.target_force_n)
        arrays["estimated_force_n"] = np.asarray(recorder.estimated_force_n)
        arrays["force_error_n"] = np.asarray(recorder.force_error_n)
        arrays["measured_force_n"] = np.asarray(
            getattr(recorder, "measured_force_n", ()),
            dtype=float,
        )
        arrays["normal_force_source"] = np.asarray(
            getattr(recorder, "normal_force_source", ()),
            dtype=str,
        )
        arrays["admittance_position_m"] = np.asarray(
            getattr(recorder, "admittance_position_m", ()),
            dtype=float,
        )
        arrays["admittance_velocity_m_s"] = np.asarray(
            getattr(recorder, "admittance_velocity_m_s", ()),
            dtype=float,
        )
        arrays["dynamic_normal_correction_m"] = np.asarray(
            getattr(recorder, "dynamic_normal_correction_m", ()),
            dtype=float,
        )
        arrays["wiping_dynamic_active"] = np.asarray(
            getattr(recorder, "wiping_dynamic_active", ()),
            dtype=bool,
        )
        arrays["task_phase"] = np.asarray(recorder.task_phase, dtype=str)
        for arm_name in sorted(result.states[-1].arms):
            prefix = f"arm_{arm_name}"
            for key, attribute in (
                ("saturation_scale", "arm_saturation_scale"),
                ("tendon_target_error_norm_m", "arm_tendon_target_error_norm_m"),
                ("tendon_target_error_max_m", "arm_tendon_target_error_max_m"),
                ("peak_actuator_force_n", "arm_peak_actuator_force_n"),
            ):
                by_arm = getattr(recorder, attribute, {})
                if arm_name in by_arm:
                    arrays[f"{prefix}_{key}"] = np.asarray(
                        by_arm[arm_name],
                        dtype=float,
                    )
    replay = application.hooks_by_name.get("mujoco_replay")
    if replay is not None:
        arrays["qpos"] = np.asarray(replay.qpos)
        arrays["qvel"] = np.asarray(replay.qvel)
        arrays["mocap_pos"] = np.asarray(replay.mocap_pos)
        arrays["mocap_quat"] = np.asarray(replay.mocap_quat)
    reports = [
        command.metadata.get("whole_body_singularity")
        for command in result.commands
    ]
    if reports and all(report is not None for report in reports):
        arrays["whole_body_rank"] = np.asarray([report.rank for report in reports])
        arrays["whole_body_min_singular_value"] = np.asarray(
            [report.minimum_singular_value for report in reports]
        )
        arrays["whole_body_condition_number"] = np.asarray(
            [report.condition_number for report in reports]
        )
        arrays["whole_body_damping"] = np.asarray(
            [report.damping for report in reports]
        )
        arrays["whole_body_velocity_scale"] = np.asarray(
            [report.velocity_scale for report in reports]
        )
        arrays["whole_body_residual_norm"] = np.asarray(
            [command.metadata.get("residual_norm", np.nan) for command in result.commands]
        )
        for arm_name in sorted(result.states[-1].arms):
            control_reports = [
                command.metadata.get("arm_singularities", {}).get(arm_name)
                for command in result.commands
            ]
            if all(report is not None for report in control_reports):
                arrays[f"arm_{arm_name}_control_min_singular_value"] = np.asarray(
                    [report.minimum_singular_value for report in control_reports]
                )
                arrays[f"arm_{arm_name}_control_condition_number"] = np.asarray(
                    [report.condition_number for report in control_reports]
                )
                arrays[f"arm_{arm_name}_control_damping"] = np.asarray(
                    [report.damping for report in control_reports]
                )
                arrays[f"arm_{arm_name}_control_velocity_scale"] = np.asarray(
                    [report.velocity_scale for report in control_reports]
                )
            arm_reports = [
                command.metadata.get("tendon_mapping_singularity", {}).get(arm_name)
                for command in result.commands
            ]
            if all(report is not None for report in arm_reports):
                arrays[f"arm_{arm_name}_mapping_rank"] = np.asarray(
                    [report.rank for report in arm_reports]
                )
                arrays[f"arm_{arm_name}_mapping_condition_number"] = np.asarray(
                    [report.condition_number for report in arm_reports]
                )
    if result.commands:
        command_metadata = [command.metadata for command in result.commands]
        for key, dtype in (
            ("observer_control_mode", str),
            ("observer_collision_active", bool),
            ("observer_tracking_active", bool),
            ("inter_arm_safety_mode", str),
            ("inter_arm_executor_frozen", bool),
            ("inter_arm_critical_distance", bool),
            ("inter_arm_hard_stop", bool),
            ("inter_arm_closest_observer_index", int),
            ("inter_arm_closest_executor_index", int),
            ("inter_arm_distance_m", float),
            ("inter_arm_min_distance_m", float),
            ("inter_arm_influence_distance_m", float),
            ("inter_arm_hard_stop_distance_m", float),
            ("inter_arm_release_margin_m", float),
            ("observer_avoidance_desired_speed_mps", float),
            ("observer_residual_norm", float),
            ("executor_feedforward_gain", float),
        ):
            arrays[key] = np.asarray(
                [metadata.get(key, _metadata_default(dtype)) for metadata in command_metadata],
                dtype=dtype,
            )
        for key in (
            "executor_target_velocity_world",
            "task_intent_velocity_world",
            "executor_scaled_feedforward_velocity_world",
            "observer_target_position_world",
            "observer_target_error_world",
            "observer_target_velocity_world",
            "inter_arm_closest_observer_point_world",
            "inter_arm_closest_executor_point_world",
        ):
            arrays[key] = np.asarray(
                [
                    metadata.get(key, np.full(3, np.nan, dtype=float))
                    for metadata in command_metadata
                ],
                dtype=float,
            )
    return arrays


def _metadata_default(dtype):
    if dtype is str:
        return ""
    if dtype is bool:
        return False
    if dtype is int:
        return -1
    return np.nan


def _state_metadata_array(
    states: list[object],
    key: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if not states or key not in states[0].metadata:
        return None
    fallback = np.full(shape, np.nan, dtype=float)
    values = []
    for state in states:
        raw = state.metadata.get(key)
        if raw is None:
            values.append(fallback.copy())
            continue
        array = np.asarray(raw, dtype=float)
        values.append(array.copy() if array.shape == shape else fallback.copy())
    return np.asarray(values, dtype=float)


def _arm_transition_metadata_array(
    states: list[object],
    command_count: int,
    arm_name: str,
    key: str,
    *,
    shape: tuple[int, ...],
    aliases: tuple[str, ...] = (),
    dtype=float,
    default: object | None = None,
) -> np.ndarray:
    fallback_value = (
        default
        if default is not None
        else False
        if dtype is bool
        else np.nan
    )
    fallback = np.full(shape, fallback_value, dtype=dtype)
    values: list[np.ndarray] = []
    for state in states[1 : command_count + 1]:
        metadata = state.metadata.get("saturation", {}).get(arm_name, {})
        raw = metadata.get(key)
        if raw is None:
            for alias in aliases:
                raw = metadata.get(alias)
                if raw is not None:
                    break
        if raw is None:
            values.append(fallback.copy())
            continue
        array = np.asarray(raw, dtype=dtype)
        values.append(array.copy() if array.shape == shape else fallback.copy())
    if not values:
        return np.empty((0, *shape), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _finite_difference_rate(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    count = min(len(values), len(time_s))
    if count < 2:
        return np.empty((0, *values.shape[1:]), dtype=float)
    delta_time = np.diff(time_s[:count])
    delta_values = np.diff(values[:count], axis=0)
    rate = np.full_like(delta_values, np.nan, dtype=float)
    valid = np.isfinite(delta_time) & (delta_time > 0.0)
    if np.any(valid):
        denominator_shape = (int(np.count_nonzero(valid)),) + (1,) * (
            delta_values.ndim - 1
        )
        rate[valid] = delta_values[valid] / delta_time[valid].reshape(
            denominator_shape
        )
    return rate


def _metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    error = arrays.get("tracking_error_m")
    finite = (
        np.asarray([], dtype=float)
        if error is None
        else error[np.isfinite(error)]
    )
    if finite.size:
        metrics.update(
            final_tracking_error_m=float(finite[-1]),
            mean_tracking_error_m=float(np.mean(finite)),
            max_tracking_error_m=float(np.max(finite)),
            rms_tracking_error_m=float(np.sqrt(np.mean(finite**2))),
        )
    achieved = arrays.get("achieved_waypoint_error_m")
    if achieved is not None:
        achieved_finite = achieved[np.isfinite(achieved)]
        if achieved_finite.size:
            metrics.update(
                final_achieved_waypoint_error_m=float(achieved_finite[-1]),
                mean_achieved_waypoint_error_m=float(np.mean(achieved_finite)),
                max_achieved_waypoint_error_m=float(np.max(achieved_finite)),
            )
    approach = arrays.get("tracking_approach")
    if (
        error is not None
        and approach is not None
        and approach.shape == error.shape
        and np.any(~approach.astype(bool) & np.isfinite(error))
    ):
        path_error = error[~approach.astype(bool) & np.isfinite(error)]
        metrics["mean_path_tracking_error_m"] = float(np.mean(path_error))
        metrics["max_path_tracking_error_m"] = float(np.max(path_error))
    clearance = arrays.get("min_clearance_m")
    if clearance is not None and np.any(np.isfinite(clearance)):
        metrics["minimum_clearance_m"] = float(np.nanmin(clearance))
    condition = arrays.get("whole_body_condition_number")
    if condition is not None and condition.size:
        metrics["maximum_whole_body_condition_number"] = float(
            np.nanmax(condition)
        )
    diagnostic_suffixes = {
        "_pcc_mujoco_tip_error_norm_m": "pcc_mujoco_tip_error_m",
        "_pcc_jacobian_linearization_residual_norm_mps": (
            "pcc_jacobian_linearization_residual_mps"
        ),
        "_pcc_mujoco_model_velocity_residual_norm_mps": (
            "pcc_mujoco_model_velocity_residual_mps"
        ),
        "_pcc_command_mujoco_velocity_residual_norm_mps": (
            "pcc_command_mujoco_velocity_residual_mps"
        ),
    }
    for key, values in arrays.items():
        for suffix, metric_name in diagnostic_suffixes.items():
            if not key.startswith("arm_") or not key.endswith(suffix):
                continue
            arm_name = key[len("arm_") : -len(suffix)]
            diagnostic_finite = values[np.isfinite(values)]
            if not diagnostic_finite.size:
                break
            metric_prefix = f"{arm_name}_{metric_name}"
            metrics[f"final_{metric_prefix}"] = float(diagnostic_finite[-1])
            metrics[f"mean_{metric_prefix}"] = float(
                np.mean(diagnostic_finite)
            )
            metrics[f"max_{metric_prefix}"] = float(np.max(diagnostic_finite))
            break
    constrained_suffix = "_constrained_command_rate_mps"
    for key, constrained_rate in arrays.items():
        if not key.startswith("arm_") or not key.endswith(constrained_suffix):
            continue
        arm_name = key[len("arm_") : -len(constrained_suffix)]
        prefix = f"arm_{arm_name}"
        inner_loop_mode = arrays.get(f"{prefix}_tendon_inner_loop_mode")
        servo_mask = (
            inner_loop_mode == "bending_rate_servo"
            if inner_loop_mode is not None
            else np.zeros(len(constrained_rate), dtype=bool)
        )
        realized_rate = arrays.get(f"{prefix}_tendon_realized_rate_fd_mps")
        if realized_rate is not None and realized_rate.shape == constrained_rate.shape:
            rate_error = constrained_rate - realized_rate
            finite_rate_error = rate_error[np.isfinite(rate_error)]
            if finite_rate_error.size:
                metrics[
                    f"rms_{arm_name}_constrained_realized_rate_error_mps"
                ] = float(np.sqrt(np.mean(finite_rate_error**2)))
        lead = arrays.get(f"{prefix}_tendon_target_lead_m")
        if lead is not None and np.any(np.isfinite(lead)):
            metrics[f"max_{arm_name}_tendon_target_lead_m"] = float(
                np.nanmax(np.abs(lead))
            )
        for signal_name, metric_name in (
            ("tendon_anti_windup_active", "anti_windup_active_fraction"),
            ("tendon_force_constraint_active", "force_constraint_active_fraction"),
        ):
            signal = arrays.get(f"{prefix}_{signal_name}")
            if (
                signal is not None
                and signal.size
                and servo_mask.shape == (len(signal),)
                and np.any(servo_mask)
            ):
                metrics[f"{arm_name}_{metric_name}"] = float(
                    np.mean(np.any(signal[servo_mask], axis=1))
                )
        force_utilization = arrays.get(f"{prefix}_actuator_force_utilization")
        if force_utilization is not None and np.any(np.isfinite(force_utilization)):
            metrics[f"max_{arm_name}_actuator_force_utilization"] = float(
                np.nanmax(force_utilization)
            )
        guard_feasible = arrays.get(f"{prefix}_tendon_guard_feasible")
        if (
            guard_feasible is not None
            and guard_feasible.size
            and servo_mask.shape == guard_feasible.shape
            and np.any(servo_mask)
        ):
            metrics[f"{arm_name}_guard_infeasible_fraction"] = float(
                np.mean(~guard_feasible[servo_mask].astype(bool))
            )
    return metrics


def _save_plots(arrays: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    saved: list[Path] = []
    target = arrays.get("target_position_m")
    executor = arrays.get("arm_executor_tip_position_m")
    if target is not None and executor is not None:
        aligned_executor = arrays.get("target_actual_position_m")
        if (
            aligned_executor is not None
            and aligned_executor.shape == target.shape
        ):
            count = len(target)
            executor_trace = aligned_executor
        else:
            count = min(len(target), len(executor) - 1)
            executor_trace = executor[1 : count + 1]
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(*target[:count].T, "--", label="target")
        axis.plot(*executor_trace[:count].T, label="executor")
        local_names = arrays.get("target_engine_local_path_name")
        is_engine_target_series = (
            local_names is not None
            and any(str(value) for value in local_names)
        )
        if not is_engine_target_series:
            for key in sorted(arrays):
                if (
                    key.startswith("arm_")
                    and key.endswith("_tip_position_m")
                    and key != "arm_executor_tip_position_m"
                ):
                    axis.plot(*arrays[key][1 : count + 1].T, label=key[4:-15])
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.legend()
        path = output_dir / "trajectory.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    saved.extend(
        _save_engine_navigation_local_path_plots(
            arrays,
            output_dir,
            plt,
        )
    )
    base_target = arrays.get("base_target_position_m")
    base_position = arrays.get("base_position_m")
    if base_target is not None and base_position is not None:
        count = min(len(base_target), len(base_position) - 1)
        if count > 0:
            fig = plt.figure(figsize=(8, 6))
            axis = fig.add_subplot(111, projection="3d")
            axis.plot(*base_target[:count].T, "--", label="base target")
            axis.plot(*base_position[1 : count + 1].T, label="base")
            axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
            axis.legend()
            path = output_dir / "engine_navigation_base_path.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    error = arrays.get("tracking_error_m")
    if error is not None:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(error)
        axis.set(xlabel="control step", ylabel="error [m]", title="Tracking error")
        axis.grid(alpha=0.3)
        path = output_dir / "tracking_error.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    for key in sorted(arrays):
        if not key.endswith("_tendon_displacement_m"):
            continue
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(arrays[key])
        axis.set(
            xlabel="control step",
            ylabel="displacement [m]",
            title=key.removesuffix("_tendon_displacement_m"),
        )
        axis.grid(alpha=0.3)
        path = output_dir / f"{key}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    for key, title, ylabel in (
        ("min_clearance_m", "Minimum clearance", "clearance [m]"),
        ("contact_distance_m", "Wiping contact distance", "distance [m]"),
        ("contact_error_m", "Wiping contact error", "error [m]"),
        ("target_force_n", "Target normal force", "force [N]"),
        ("estimated_force_n", "Estimated normal force", "force [N]"),
        ("force_error_n", "Normal force error", "force [N]"),
    ):
        values = arrays.get(key)
        if values is None or not np.any(np.isfinite(values)):
            continue
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(values)
        axis.set(xlabel="control step", ylabel=ylabel, title=title)
        axis.grid(alpha=0.3)
        path = output_dir / f"{key}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    condition = arrays.get("whole_body_condition_number")
    if condition is not None:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.semilogy(condition)
        axis.set(
            xlabel="control step",
            ylabel="condition number",
            title="Whole-body singularity",
        )
        axis.grid(alpha=0.3)
        path = output_dir / "whole_body_singularity.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    saved.extend(_save_mujoco_pcc_diagnostic_plots(arrays, output_dir, plt))
    saved.extend(_save_dual_arm_control_plots(arrays, output_dir, plt))
    return saved


def _save_mujoco_pcc_diagnostic_plots(
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    plt,
) -> list[Path]:
    state_time = arrays.get("time_s")
    command_time = arrays.get("command_time_s")
    if state_time is None or command_time is None:
        return []
    suffix = "_pcc_mujoco_tip_error_norm_m"
    arm_names = sorted(
        key[len("arm_") : -len(suffix)]
        for key in arrays
        if key.startswith("arm_") and key.endswith(suffix)
    )
    saved: list[Path] = []
    component_names = ("x", "y", "z")
    for arm_name in arm_names:
        prefix = f"arm_{arm_name}"
        mujoco_tip = arrays[f"{prefix}_mujoco_tip_position_mount_m"]
        pcc_tip = arrays[f"{prefix}_pcc_tip_position_mount_m"]
        position_error = arrays[f"{prefix}_pcc_mujoco_tip_error_mount_m"]
        position_error_norm = arrays[f"{prefix}_pcc_mujoco_tip_error_norm_m"]

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
        for component, axis in enumerate(axes.flat[:3]):
            axis.plot(
                state_time,
                1000.0 * mujoco_tip[:, component],
                label="MuJoCo",
            )
            axis.plot(
                state_time,
                1000.0 * pcc_tip[:, component],
                "--",
                label="PCC FK",
            )
            axis.set(
                title=f"mount-frame {component_names[component]}",
                ylabel="position [mm]",
            )
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        for component, component_name in enumerate(component_names):
            axes[1, 1].plot(
                state_time,
                1000.0 * position_error[:, component],
                label=f"{component_name} error",
            )
        axes[1, 1].plot(
            state_time,
            1000.0 * position_error_norm,
            "k--",
            linewidth=1.2,
            label="norm",
        )
        axes[1, 1].set(
            title="PCC FK - MuJoCo",
            ylabel="error [mm]",
        )
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend(fontsize=8)
        for axis in axes[1, :]:
            axis.set_xlabel("time [s]")
        fig.suptitle(f"{arm_name}: PCC and MuJoCo tip position")
        path = output_dir / f"{prefix}_pcc_mujoco_position.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        mujoco_velocity = arrays[f"{prefix}_mujoco_tip_velocity_fd_mount_mps"]
        pcc_fk_velocity = arrays[f"{prefix}_pcc_tip_velocity_fd_mount_mps"]
        jacobian_velocity = arrays[
            f"{prefix}_pcc_tip_velocity_from_tendon_delta_mount_mps"
        ]
        command_velocity = arrays[
            f"{prefix}_pcc_command_arm_tip_velocity_mount_mps"
        ]
        measured_tendon_velocity = arrays[
            f"{prefix}_pcc_tip_velocity_from_measured_tendon_mount_mps"
        ]
        jacobian_residual = arrays[
            f"{prefix}_pcc_jacobian_linearization_residual_norm_mps"
        ]
        model_residual = arrays[
            f"{prefix}_pcc_mujoco_model_velocity_residual_norm_mps"
        ]
        command_residual = arrays[
            f"{prefix}_pcc_command_mujoco_velocity_residual_norm_mps"
        ]
        velocity_count = min(
            len(command_time),
            len(mujoco_velocity),
            len(pcc_fk_velocity),
            len(jacobian_velocity),
            len(command_velocity),
            max(len(measured_tendon_velocity) - 1, 0),
        )
        velocity_time = command_time[:velocity_count]
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
        for component, axis in enumerate(axes.flat[:3]):
            axis.plot(
                velocity_time,
                1000.0 * mujoco_velocity[:velocity_count, component],
                label="MuJoCo FD",
            )
            axis.plot(
                velocity_time,
                1000.0 * pcc_fk_velocity[:velocity_count, component],
                "--",
                label="PCC FK FD",
            )
            axis.plot(
                velocity_time,
                1000.0 * jacobian_velocity[:velocity_count, component],
                ":",
                label="PCC J from actual tendon delta",
            )
            axis.plot(
                velocity_time,
                1000.0 * command_velocity[:velocity_count, component],
                alpha=0.65,
                label="PCC J from command",
            )
            axis.plot(
                velocity_time,
                1000.0
                * measured_tendon_velocity[1 : velocity_count + 1, component],
                "-.",
                alpha=0.65,
                label="PCC J from measured tendon velocity",
            )
            axis.set(
                title=f"mount-frame {component_names[component]}",
                ylabel="velocity [mm/s]",
            )
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        residual_count = min(
            velocity_count,
            len(jacobian_residual),
            len(model_residual),
            len(command_residual),
        )
        axes[1, 1].plot(
            velocity_time[:residual_count],
            1000.0 * jacobian_residual[:residual_count],
            label="Jacobian vs PCC FK",
        )
        axes[1, 1].plot(
            velocity_time[:residual_count],
            1000.0 * model_residual[:residual_count],
            label="PCC FK vs MuJoCo",
        )
        axes[1, 1].plot(
            velocity_time[:residual_count],
            1000.0 * command_residual[:residual_count],
            label="command vs MuJoCo (world)",
        )
        axes[1, 1].set(
            title="velocity residual norms",
            ylabel="residual [mm/s]",
        )
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend(fontsize=8)
        for axis in axes[1, :]:
            axis.set_xlabel("time [s]")
        fig.suptitle(f"{arm_name}: PCC velocity and command response")
        path = output_dir / f"{prefix}_pcc_mujoco_velocity.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        singular_values = arrays[f"{prefix}_pcc_jacobian_singular_values"]
        row_norms = arrays[f"{prefix}_pcc_jacobian_world_row_norm"]
        rank = arrays[f"{prefix}_pcc_jacobian_rank"]
        condition_number = arrays[f"{prefix}_pcc_jacobian_condition_number"]
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
        for index in range(singular_values.shape[1]):
            axes[0].semilogy(
                state_time,
                singular_values[:, index],
                label=f"sigma {index + 1}",
            )
        axes[0].set(
            title="bending Jacobian singular values",
            xlabel="time [s]",
            ylabel="singular value",
        )
        axes[0].legend(fontsize=8)
        for component, component_name in enumerate(component_names):
            axes[1].semilogy(
                state_time,
                row_norms[:, component],
                label=f"world {component_name} row",
            )
        axes[1].set(
            title="world-axis sensitivity",
            xlabel="time [s]",
            ylabel="row norm",
        )
        axes[1].legend(fontsize=8)
        finite_condition = np.where(
            np.isfinite(condition_number),
            condition_number,
            np.nan,
        )
        axes[2].semilogy(
            state_time,
            finite_condition,
            label="condition number",
        )
        rank_axis = axes[2].twinx()
        rank_axis.plot(
            state_time,
            rank,
            color="tab:orange",
            alpha=0.7,
            label="rank",
        )
        axes[2].set(
            title="conditioning and rank",
            xlabel="time [s]",
            ylabel="condition number",
        )
        rank_axis.set_ylabel("rank")
        lines = axes[2].lines + rank_axis.lines
        axes[2].legend(lines, [line.get_label() for line in lines], fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.3)
        fig.suptitle(f"{arm_name}: PCC Jacobian conditioning")
        path = output_dir / f"{prefix}_pcc_jacobian.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def _save_dual_arm_control_plots(
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    plt,
) -> list[Path]:
    state_time = arrays.get("time_s")
    command_time = arrays.get("command_time_s")
    if state_time is None or command_time is None:
        return []
    saved: list[Path] = []
    for arm_name in ("executor", "observer"):
        prefix = f"arm_{arm_name}"
        target = arrays.get(f"{prefix}_tendon_target_m")
        actual = arrays.get(f"{prefix}_tendon_displacement_m")
        velocity = arrays.get(f"{prefix}_tendon_velocity_sensor_raw_mps")
        if velocity is None:
            velocity = arrays.get(f"{prefix}_tendon_velocity_mps")
        requested_rate = arrays.get(f"{prefix}_command_rate_mps")
        constrained_rate = arrays.get(f"{prefix}_constrained_command_rate_mps")
        if constrained_rate is None:
            constrained_rate = arrays.get(f"{prefix}_applied_rate_mps")
        target_rate = arrays.get(f"{prefix}_tendon_target_rate_fd_mps")
        realized_rate = arrays.get(f"{prefix}_tendon_realized_rate_fd_mps")
        filtered_rate = arrays.get(f"{prefix}_tendon_measured_rate_filtered_mps")
        target_lead = arrays.get(f"{prefix}_tendon_target_lead_m")
        servo_evaluated = arrays.get(f"{prefix}_tendon_servo_evaluated")
        has_servo_transitions = bool(
            servo_evaluated is not None and np.any(servo_evaluated)
        )
        rate_integral = arrays.get(f"{prefix}_tendon_rate_error_integral_m")
        anti_windup_active = arrays.get(f"{prefix}_tendon_anti_windup_active")
        force_constraint_active = arrays.get(
            f"{prefix}_tendon_force_constraint_active"
        )
        force_utilization = arrays.get(f"{prefix}_actuator_force_utilization")
        actuator_force_at_limit = arrays.get(f"{prefix}_actuator_force_at_limit")
        force = arrays.get(f"{prefix}_actuator_force_n")
        if any(
            value is None
            for value in (
                target,
                actual,
                velocity,
                requested_rate,
                constrained_rate,
                force,
            )
        ):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
        for tendon_index in range(actual.shape[1]):
            line = axes[0, 0].plot(
                state_time,
                1000.0 * actual[:, tendon_index],
                linewidth=1.0,
                label=("actual" if tendon_index == 0 else None),
            )[0]
            axes[0, 0].plot(
                state_time,
                1000.0 * target[:, tendon_index],
                "--",
                color=line.get_color(),
                linewidth=0.8,
                alpha=0.7,
                label=("target" if tendon_index == 0 else None),
            )
        axes[0, 0].set(
            title=f"{arm_name} tendon target vs actual",
            xlabel="time [s]",
            ylabel="displacement [mm]",
        )
        axes[0, 0].legend(loc="upper right", fontsize=8)

        if target_lead is not None and len(target_lead):
            lead_count = min(len(command_time), len(target_lead))
            axes[0, 1].plot(
                command_time[:lead_count],
                1000.0 * np.max(np.abs(target_lead[:lead_count]), axis=1),
                label="target lead",
            )
            if has_servo_transitions and anti_windup_active is not None:
                active_count = min(lead_count, len(anti_windup_active))
                active_mask = np.any(anti_windup_active[:active_count], axis=1)
                if np.any(active_mask):
                    axes[0, 1].scatter(
                        command_time[:active_count][active_mask],
                        1000.0
                        * np.max(np.abs(target_lead[:active_count]), axis=1)[
                            active_mask
                        ],
                        marker="x",
                        s=18,
                        label="anti-windup",
                    )
            if has_servo_transitions and rate_integral is not None:
                integral_count = min(lead_count, len(rate_integral))
                axes[0, 1].plot(
                    command_time[:integral_count],
                    1000.0 * np.max(np.abs(rate_integral[:integral_count]), axis=1),
                    label="rate-error integral",
                )
            axes[0, 1].set(
                title=(
                    "Tendon target lead and servo state"
                    if has_servo_transitions
                    else "Tendon target lead"
                ),
                xlabel="time [s]",
                ylabel="max absolute value [mm]",
            )
            axes[0, 1].legend(loc="upper right", fontsize=8)
        else:
            tendon_error = np.linalg.norm(target - actual, axis=1)
            axes[0, 1].plot(state_time, 1000.0 * tendon_error)
            axes[0, 1].set(
                title="Tendon target error norm",
                xlabel="time [s]",
                ylabel="error [mm]",
            )

        command_count = min(
            len(command_time),
            len(requested_rate),
            len(constrained_rate),
            max(0, len(velocity) - 1),
        )
        if command_count:
            axes[1, 0].plot(
                command_time[:command_count],
                1000.0 * np.max(np.abs(requested_rate[:command_count]), axis=1),
                label="requested rate",
            )
            axes[1, 0].plot(
                command_time[:command_count],
                1000.0 * np.max(np.abs(constrained_rate[:command_count]), axis=1),
                label="constrained command",
            )
            for values, label, style in (
                (target_rate, "target FD", "-."),
                (realized_rate, "realized FD", "-"),
                (filtered_rate, "realized filtered", ":"),
            ):
                if values is None:
                    continue
                count = min(command_count, len(values))
                axes[1, 0].plot(
                    command_time[:count],
                    1000.0 * np.max(np.abs(values[:count]), axis=1),
                    style,
                    label=label,
                )
            axes[1, 0].plot(
                command_time[:command_count],
                1000.0 * np.max(
                    np.abs(velocity[1 : command_count + 1]),
                    axis=1,
                ),
                alpha=0.45,
                label="raw sensor",
            )
        axes[1, 0].set(
            title="Tendon rate envelope",
            xlabel="time [s]",
            ylabel="max absolute rate [mm/s]",
        )
        if command_count:
            axes[1, 0].legend(loc="upper right", fontsize=8)

        force_line = axes[1, 1].plot(
            state_time,
            np.max(np.abs(force), axis=1),
            label="peak actuator force",
        )[0]
        axes[1, 1].set(
            title="Actuator load",
            xlabel="time [s]",
            ylabel="force [N]",
        )
        force_lines = [force_line]
        if force_utilization is not None and len(force_utilization):
            utilization_count = min(len(command_time), len(force_utilization))
            utilization_axis = axes[1, 1].twinx()
            utilization_line = utilization_axis.plot(
                command_time[:utilization_count],
                np.max(force_utilization[:utilization_count], axis=1),
                color="tab:orange",
                label="force utilization",
            )[0]
            utilization_axis.set_ylabel("utilization")
            force_lines.append(utilization_line)
        for active, marker, label in (
            (force_constraint_active, "x", "force guard"),
            (actuator_force_at_limit, "o", "actuator force limit"),
        ):
            if active is None or not len(active):
                continue
            active_count = min(len(command_time), len(active), max(0, len(force) - 1))
            active_mask = np.any(active[:active_count], axis=1)
            if np.any(active_mask):
                marker_line = axes[1, 1].scatter(
                    command_time[:active_count][active_mask],
                    np.max(np.abs(force[1 : active_count + 1]), axis=1)[active_mask],
                    marker=marker,
                    s=20,
                    label=label,
                )
                force_lines.append(marker_line)
        axes[1, 1].legend(
            force_lines,
            [line.get_label() for line in force_lines],
            loc="upper right",
            fontsize=8,
        )
        for axis in axes.reshape(-1):
            axis.grid(alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"{prefix}_synchronized_control.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    distance = arrays.get("inter_arm_distance_m")
    observer_velocity = arrays.get("observer_target_velocity_world")
    collision_active = arrays.get("observer_collision_active")
    if (
        "arm_observer_tendon_displacement_m" in arrays
        and distance is not None
        and observer_velocity is not None
    ):
        count = min(len(command_time), len(distance), len(observer_velocity))
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
        axes[0].plot(
            command_time[:count],
            1000.0 * distance[:count],
            label="inter-arm distance",
        )
        for key, label in (
            ("inter_arm_influence_distance_m", "influence"),
            ("inter_arm_min_distance_m", "minimum"),
            ("inter_arm_hard_stop_distance_m", "critical"),
        ):
            values = arrays.get(key)
            if values is not None and len(values):
                axes[0].axhline(
                    1000.0 * float(values[0]),
                    linestyle="--",
                    label=label,
                )
        axes[0].set(ylabel="distance [mm]", title="Dual-arm safety distance")
        axes[0].legend(loc="upper right", fontsize=8)
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            axes[1].plot(
                command_time[:count],
                1000.0 * observer_velocity[:count, axis_index],
                label=f"observer v {axis_name}",
            )
        if collision_active is not None and count:
            axes[1].fill_between(
                command_time[:count],
                0.0,
                1.0,
                where=np.asarray(collision_active[:count], dtype=bool),
                transform=axes[1].get_xaxis_transform(),
                alpha=0.12,
                label="avoidance active",
            )
        axes[1].set(
            xlabel="time [s]",
            ylabel="target velocity [mm/s]",
            title="Observer collision-avoidance command",
        )
        axes[1].legend(loc="upper right", fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.3)
        fig.tight_layout()
        path = output_dir / "dual_arm_synchronized_safety.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def _save_engine_navigation_local_path_plots(
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    plt,
) -> list[Path]:
    target = arrays.get("target_position_m")
    actual = arrays.get("target_actual_position_m")
    names = arrays.get("target_engine_local_path_name")
    subphases = arrays.get("target_engine_executor_subphase")
    centers = arrays.get("target_engine_local_path_center_m")
    directions = arrays.get("target_engine_insertion_direction_world")
    required = (target, actual, names, subphases, centers, directions)
    if any(value is None for value in required):
        return []
    count = len(target)
    if any(len(value) != count for value in required):
        return []
    if (
        np.asarray(target).shape != (count, 3)
        or np.asarray(actual).shape != (count, 3)
        or np.asarray(centers).shape != (count, 3)
        or np.asarray(directions).shape != (count, 3)
        or np.asarray(names).shape != (count,)
        or np.asarray(subphases).shape != (count,)
    ):
        return []

    saved: list[Path] = []
    for name in dict.fromkeys(str(value) for value in names if str(value)):
        mask = (names == name) & (subphases == "path")
        if not np.any(mask):
            continue
        target_path = np.asarray(target[mask], dtype=float)
        actual_path = np.asarray(actual[mask], dtype=float)
        center_path = np.asarray(centers[mask], dtype=float)
        direction_path = np.asarray(directions[mask], dtype=float)
        finite = (
            np.all(np.isfinite(target_path), axis=1)
            & np.all(np.isfinite(actual_path), axis=1)
            & np.all(np.isfinite(center_path), axis=1)
            & np.all(np.isfinite(direction_path), axis=1)
        )
        if not np.any(finite):
            continue
        target_path = target_path[finite]
        actual_path = actual_path[finite]
        center_path = center_path[finite]
        direction = direction_path[finite][0]
        basis = _transverse_plot_basis(direction)
        if basis is None:
            continue
        axis_u, axis_v = basis
        target_delta = target_path - center_path
        actual_delta = actual_path - center_path
        target_local = np.column_stack(
            (target_delta @ axis_u, target_delta @ axis_v)
        )
        actual_local = np.column_stack(
            (actual_delta @ axis_u, actual_delta @ axis_v)
        )
        error_mm = 1000.0 * np.linalg.norm(actual_path - target_path, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        trajectory_axis, error_axis = axes
        trajectory_axis.plot(
            1000.0 * target_local[:, 0],
            1000.0 * target_local[:, 1],
            "--",
            label="target",
        )
        trajectory_axis.plot(
            1000.0 * actual_local[:, 0],
            1000.0 * actual_local[:, 1],
            label="actual",
        )
        trajectory_axis.scatter(
            1000.0 * target_local[0, 0],
            1000.0 * target_local[0, 1],
            marker="o",
            s=28,
            label="start",
        )
        trajectory_axis.scatter(
            1000.0 * actual_local[-1, 0],
            1000.0 * actual_local[-1, 1],
            marker="x",
            s=36,
            label="actual end",
        )
        trajectory_axis.set(
            xlabel="local x [mm]",
            ylabel="local y [mm]",
            title=f"{name}: local trajectory",
        )
        trajectory_axis.set_aspect("equal", adjustable="box")
        trajectory_axis.grid(alpha=0.3)
        trajectory_axis.legend()

        error_axis.plot(error_mm)
        error_axis.set(
            xlabel="target sample",
            ylabel="3D tracking error [mm]",
            title="Same-step tracking error",
        )
        error_axis.grid(alpha=0.3)
        mean_error = float(np.mean(error_mm))
        rms_error = float(np.sqrt(np.mean(error_mm**2)))
        max_error = float(np.max(error_mm))
        error_axis.text(
            0.98,
            0.95,
            f"mean: {mean_error:.3f} mm\n"
            f"RMS: {rms_error:.3f} mm\n"
            f"max: {max_error:.3f} mm",
            transform=error_axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )
        fig.tight_layout()
        safe_name = _safe_plot_name(name)
        path = output_dir / f"engine_navigation_local_path_{safe_name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def _transverse_plot_basis(
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    z_axis = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(z_axis))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    z_axis = z_axis / norm
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    axis_u = reference - float(reference @ z_axis) * z_axis
    axis_u_norm = float(np.linalg.norm(axis_u))
    if axis_u_norm <= 1.0e-12:
        return None
    axis_u = axis_u / axis_u_norm
    axis_v = np.cross(z_axis, axis_u)
    axis_v_norm = float(np.linalg.norm(axis_v))
    if axis_v_norm <= 1.0e-12:
        return None
    return axis_u, axis_v / axis_v_norm


def _safe_plot_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe or "unnamed"


def _replay_result(application, arrays: dict[str, np.ndarray], scene_xml: Path | None):
    target = arrays.get("target_position_m")
    tip = arrays.get("arm_executor_tip_position_m")
    count = 0 if target is None or tip is None else min(len(target), len(tip) - 1)
    values = {
        "time": arrays["time_s"][1 : count + 1],
        "target_position": np.zeros((0, 3)) if target is None else target[:count],
        "tip_position": np.zeros((0, 3)) if tip is None else tip[1 : count + 1],
    }
    if "base_position_m" in arrays:
        values["base_position_m"] = arrays["base_position_m"]
    if "mujoco_mobile_base_frame_pose" in arrays:
        values["mujoco_mobile_base_frame_pose"] = arrays[
            "mujoco_mobile_base_frame_pose"
        ]
    if scene_xml is not None and "qpos" in arrays:
        values.update(
            scene_xml_path=scene_xml,
            qpos=arrays["qpos"],
            qvel=arrays["qvel"],
            mocap_pos=arrays["mocap_pos"],
            mocap_quat=arrays["mocap_quat"],
        )
    return SimpleNamespace(**values)


def _copy_scene_model(application, model_dir: Path) -> Path | None:
    backend_config = application.config.backend
    source = backend_config.generated_xml_path
    if source is None or not source.exists():
        return None
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / "scene.xml"
    tree = ET.parse(source)
    for element in tree.getroot().iter():
        raw_file = element.get("file")
        if not raw_file:
            continue
        resolved = Path(raw_file)
        if not resolved.is_absolute():
            resolved = (source.parent / resolved).resolve()
        element.set("file", Path(os.path.relpath(resolved, model_dir)).as_posix())
    tree.write(destination, encoding="utf-8", xml_declaration=False)
    return destination


def _unique_dir(path: Path) -> Path:
    candidate = path.resolve()
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{suffix:03d}").resolve()
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate
