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
    if settings.save_npz:
        np.savez_compressed(paths.result_npz, **arrays)
    config_dir = run_dir / "configs"
    config_dir.mkdir()
    shutil.copy2(config.path, config_dir / "scenario.yaml")
    shutil.copy2(config.assembly_config_path, config_dir / "assembly.yaml")
    if config.backend.mujoco_config_path is not None:
        shutil.copy2(config.backend.mujoco_config_path, config_dir / "mujoco.yaml")
    scene_xml = _copy_scene_model(application, run_dir / "model") if settings.save_model else None
    errors: list[str] = []
    plot_files: list[str] = []
    if settings.save_plots:
        try:
            plot_files = [str(path) for path in _save_plots(arrays, paths.plots_dir)]
        except Exception as exc:  # Artifact failure must not discard simulation data.
            errors.append(f"plots: {type(exc).__name__}: {exc}")
    video_path = None
    if settings.save_gif:
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
    metrics = _metrics(arrays)
    metadata = {
        "scenario": config.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": config.backend.type,
        "task": config.task.type,
        "arms": sorted(result.states[-1].arms),
        "samples": len(result.states),
        "commands": len(result.commands),
        "stopped_early": result.stopped_early,
        "metrics": metrics,
        "npz_keys": sorted(arrays) if settings.save_npz else [],
        "plots": plot_files,
        "video": None if video_path is None else str(video_path),
        "model": None if scene_xml is None else str(scene_xml),
        "errors": errors,
    }
    paths.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def _flatten_result(application, result) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "time_s": np.asarray([state.time_s for state in result.states], dtype=float),
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
    for arm_name in sorted(result.states[-1].arms):
        prefix = f"arm_{arm_name}"
        arrays[f"{prefix}_tip_position_m"] = np.asarray(
            [state.arms[arm_name].tip_pose_world.position for state in result.states]
        )
        arrays[f"{prefix}_tip_quat_wxyz"] = np.asarray(
            [state.arms[arm_name].tip_pose_world.quat for state in result.states]
        )
        arrays[f"{prefix}_tendon_displacement_m"] = np.asarray(
            [state.arms[arm_name].tendon_displacement_m for state in result.states]
        )
        arrays[f"{prefix}_tendon_velocity_mps"] = np.asarray(
            [state.arms[arm_name].tendon_velocity_mps for state in result.states]
        )
        arrays[f"{prefix}_command_rate_mps"] = np.asarray(
            [command.arms[arm_name].tendon_rate_mps for command in result.commands]
        )
    recorder = application.hooks_by_name.get("recorder")
    if recorder is not None and recorder.target_position_m:
        arrays["target_position_m"] = np.asarray(recorder.target_position_m)
        arrays["tracking_error_m"] = np.asarray(recorder.tracking_error_m)
        arrays["waypoint_index"] = np.asarray(recorder.waypoint_index, dtype=int)
        arrays["min_clearance_m"] = np.asarray(recorder.min_clearance_m)
        arrays["contact_distance_m"] = np.asarray(recorder.contact_distance_m)
        arrays["task_phase"] = np.asarray(recorder.task_phase, dtype=str)
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
    return arrays


def _metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    error = arrays.get("tracking_error_m")
    if error is None or error.size == 0:
        return {}
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return {}
    metrics = {
        "final_tracking_error_m": float(finite[-1]),
        "mean_tracking_error_m": float(np.mean(finite)),
        "max_tracking_error_m": float(np.max(finite)),
        "rms_tracking_error_m": float(np.sqrt(np.mean(finite**2))),
    }
    clearance = arrays.get("min_clearance_m")
    if clearance is not None and np.any(np.isfinite(clearance)):
        metrics["minimum_clearance_m"] = float(np.nanmin(clearance))
    condition = arrays.get("whole_body_condition_number")
    if condition is not None and condition.size:
        metrics["maximum_whole_body_condition_number"] = float(
            np.nanmax(condition)
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
        count = min(len(target), len(executor) - 1)
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(*target[:count].T, "--", label="target")
        axis.plot(*executor[1 : count + 1].T, label="executor")
        for key in sorted(arrays):
            if key.startswith("arm_") and key.endswith("_tip_position_m") and key != "arm_executor_tip_position_m":
                axis.plot(*arrays[key][1 : count + 1].T, label=key[4:-15])
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.legend()
        path = output_dir / "trajectory.png"
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
    return saved


def _replay_result(application, arrays: dict[str, np.ndarray], scene_xml: Path | None):
    target = arrays.get("target_position_m")
    tip = arrays.get("arm_executor_tip_position_m")
    count = 0 if target is None or tip is None else min(len(target), len(tip) - 1)
    values = {
        "time": arrays["time_s"][1 : count + 1],
        "target_position": np.zeros((0, 3)) if target is None else target[:count],
        "tip_position": np.zeros((0, 3)) if tip is None else tip[1 : count + 1],
    }
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
