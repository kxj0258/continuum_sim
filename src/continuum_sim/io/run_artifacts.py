"""Save simulation rollout artifacts produced by CLI commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

from continuum_sim.config import load_mujoco_config, load_yaml
from continuum_sim.visualization.mujoco_video import save_replay_video
from continuum_sim.visualization.run_plots import save_run_plots


DEFAULT_RUNS_ROOT = Path("output") / "runs"
DEFAULT_REPLAY_VIDEO_SIZE = (640, 480)


@dataclass(frozen=True)
class RunArtifactPaths:
    """Filesystem locations created for one saved CLI run."""

    run_dir: Path
    result_npz: Path
    metadata_json: Path
    config_dir: Path
    model_dir: Path
    plots_dir: Path
    videos_dir: Path


def save_run_artifacts(
    *,
    command: str,
    result: object,
    task_config_path: str | Path,
    main_config_path: str | Path | None = None,
    mujoco_config_path: str | Path | None = None,
    task_name: str | None = None,
    output_root: str | Path = DEFAULT_RUNS_ROOT,
    timestamp: datetime | None = None,
    save_video: bool = True,
) -> RunArtifactPaths:
    """Save NPZ data, metadata, configs, plots, and optional replay video."""

    resolved_task = Path(task_config_path).resolve()
    resolved_main = None if main_config_path is None else Path(main_config_path).resolve()
    resolved_mujoco = None if mujoco_config_path is None else Path(mujoco_config_path).resolve()
    name = _safe_run_name(task_name or _task_name_from_yaml(resolved_task) or command)
    run_dir = _create_run_dir(Path(output_root), name, timestamp or datetime.now())
    paths = RunArtifactPaths(
        run_dir=run_dir,
        result_npz=run_dir / "result.npz",
        metadata_json=run_dir / "metadata.json",
        config_dir=run_dir / "configs",
        model_dir=run_dir / "model",
        plots_dir=run_dir / "plots",
        videos_dir=run_dir / "videos",
    )
    for directory in (
        paths.config_dir,
        paths.model_dir,
        paths.plots_dir,
        paths.videos_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    npz_keys = save_result_npz(result, paths.result_npz)
    copied_configs = _copy_input_configs(
        paths.config_dir,
        task_config_path=resolved_task,
        main_config_path=resolved_main,
        mujoco_config_path=resolved_mujoco,
    )
    copied_models = _copy_model_artifacts(paths.model_dir, result)
    plot_paths = save_run_plots(result, paths.plots_dir, task_name=name)
    video_width, video_height, video_camera = _replay_video_settings(resolved_mujoco)
    video_path = _save_replay_video_artifact(
        result=result,
        result_npz_path=paths.result_npz,
        output_path=paths.videos_dir / "simulation.gif",
        copied_models=copied_models,
        enabled=save_video,
        width=video_width,
        height=video_height,
        camera=video_camera,
    )
    metadata = {
        "command": command,
        "task_name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(paths.run_dir),
        "result_npz": str(paths.result_npz),
        "npz_keys": npz_keys,
        "configs": [str(path) for path in copied_configs],
        "models": [str(path) for path in copied_models],
        "plots": [str(path) for path in plot_paths],
        "video": None if video_path is None else str(video_path),
        "samples": _sample_count(result),
    }
    paths.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def save_result_npz(result: object, path: str | Path) -> list[str]:
    """Save dataclass result fields to a compressed NPZ file."""

    data = _result_to_npz_mapping(result)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return sorted(data)


def _result_to_npz_mapping(result: object) -> dict[str, np.ndarray]:
    if not is_dataclass(result):
        raise TypeError(f"Expected a dataclass result, got {type(result).__name__}.")
    data: dict[str, np.ndarray] = {}
    for field in fields(result):
        value = getattr(result, field.name)
        encoded = _to_npz_array(value)
        if encoded is not None:
            data[field.name] = encoded
    tip_position = getattr(result, "tip_position", None)
    if tip_position is not None and "tip_position" not in data:
        data["tip_position"] = np.asarray(tip_position)
    return data


def _to_npz_array(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Path):
        return np.asarray(str(value))
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return np.asarray(value, dtype=str)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return np.asarray(value, dtype=str)
    if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
        return np.asarray(value)
    return None


def _create_run_dir(root: Path, task_name: str, timestamp: datetime) -> Path:
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{task_name}_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{task_name}_{stamp}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _copy_input_configs(
    config_dir: Path,
    *,
    task_config_path: Path,
    main_config_path: Path | None,
    mujoco_config_path: Path | None,
) -> list[Path]:
    copied: list[Path] = []
    seen: set[Path] = set()
    for label, source in (
        ("main_config", main_config_path),
        ("task_config", task_config_path),
        ("mujoco_config", mujoco_config_path),
    ):
        if source is None:
            continue
        resolved = source.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        copied.append(_copy_file(resolved, config_dir / f"{label}{resolved.suffix}"))

    raw_task = load_yaml(task_config_path)
    for section_name, field_name, label in (
        ("robot", "config_path", "robot_config"),
        ("scene", "config_path", "scene_config"),
    ):
        section = raw_task.get(section_name)
        if not isinstance(section, dict) or field_name not in section:
            continue
        resolved = _resolve_relative(task_config_path, section[field_name])
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        copied.append(_copy_file(resolved, config_dir / f"{label}{resolved.suffix}"))
    return copied


def _copy_model_artifacts(model_dir: Path, result: object) -> list[Path]:
    copied: list[Path] = []
    scene_xml_path = getattr(result, "scene_xml_path", None)
    if scene_xml_path is None:
        return copied
    source = Path(scene_xml_path).resolve()
    if not source.exists():
        return copied
    copied.append(_copy_mujoco_xml(source, model_dir / "scene.xml"))
    return copied


def _copy_mujoco_xml(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ElementTree.parse(source)
    _rebase_asset_file_paths(tree.getroot(), source.parent, destination.parent)
    tree.write(destination, encoding="utf-8", xml_declaration=False)
    return destination


def _rebase_asset_file_paths(
    root: ElementTree.Element,
    base_xml_dir: Path,
    output_xml_dir: Path,
) -> None:
    asset = root.find("asset")
    if asset is None:
        return
    for element in asset.iter():
        raw_file = element.attrib.get("file")
        if not raw_file:
            continue
        raw_path = Path(raw_file)
        if raw_path.is_absolute():
            continue
        source_path = (base_xml_dir / raw_path).resolve()
        element.set(
            "file",
            Path(os.path.relpath(source_path, output_xml_dir.resolve())).as_posix(),
        )


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _resolve_relative(config_path: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    parent_candidate = (config_path.parent / path).resolve()
    if parent_candidate.exists():
        return parent_candidate
    return path.resolve()


def _task_name_from_yaml(path: Path) -> str | None:
    try:
        raw = load_yaml(path)
    except (FileNotFoundError, ValueError):
        return None
    value = raw.get("name")
    if value is None:
        return None
    return str(value)


def _replay_video_settings(mujoco_config_path: Path | None) -> tuple[int, int, object | None]:
    if mujoco_config_path is None:
        return (*DEFAULT_REPLAY_VIDEO_SIZE, None)
    config = load_mujoco_config(
        mujoco_config_path,
        require_xml=False,
        require_tendon_xml=False,
        require_visual_meshes=False,
    )
    return (
        config.rendering.offscreen_width,
        config.rendering.offscreen_height,
        config.viewer.camera,
    )


def _save_replay_video_artifact(
    *,
    result: object,
    result_npz_path: Path,
    output_path: Path,
    copied_models: list[Path],
    enabled: bool,
    width: int,
    height: int,
    camera: object | None,
) -> Path | None:
    if not enabled or not _is_mujoco_result(result):
        return save_replay_video(
            result,
            output_path,
            enabled=enabled,
            width=width,
            height=height,
            camera=camera,
        )
    return _save_replay_video_subprocess(
        result_npz_path=result_npz_path,
        output_path=output_path,
        scene_xml_path=copied_models[0] if copied_models else None,
        width=width,
        height=height,
        camera=camera,
    )


def _save_replay_video_subprocess(
    *,
    result_npz_path: Path,
    output_path: Path,
    scene_xml_path: Path | None,
    width: int,
    height: int,
    camera: object | None,
) -> Path | None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "export_replay_video.py"
    command = [
        sys.executable,
        str(script_path),
        "--result-npz",
        str(result_npz_path),
        "--output",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    if camera is not None:
        command.extend(
            [
                "--camera-lookat",
                *[str(value) for value in getattr(camera, "lookat")],
                "--camera-distance",
                str(getattr(camera, "distance")),
                "--camera-azimuth",
                str(getattr(camera, "azimuth")),
                "--camera-elevation",
                str(getattr(camera, "elevation")),
            ]
        )
    if scene_xml_path is not None:
        command.extend(["--scene-xml", str(scene_xml_path)])

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and output_path.is_file():
        return output_path
    error_path = output_path.parent / "video_error.txt"
    if not error_path.is_file():
        message = (completed.stderr or completed.stdout or "video export subprocess failed").strip()
        error_path.write_text(message + "\n", encoding="utf-8")
    return output_path if output_path.is_file() else None


def _is_mujoco_result(result: object) -> bool:
    return (
        hasattr(result, "scene_xml_path")
        and hasattr(result, "qpos")
        and hasattr(result, "qvel")
    )


def _safe_run_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "run"


def _sample_count(result: object) -> int:
    time = getattr(result, "time", None)
    if time is None:
        return 0
    return int(np.asarray(time).shape[0])
