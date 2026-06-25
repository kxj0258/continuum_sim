from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.tasks.engine_cleaning_config import (
    EngineCleaningTaskConfig,
    load_engine_cleaning_task_config,
    validate_engine_cleaning_task_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "engine_surface_path.yaml"


def test_load_engine_cleaning_task_config_reads_surface_patch_and_path() -> None:
    config = load_engine_cleaning_task_config(TASK_CONFIG)

    assert isinstance(config, EngineCleaningTaskConfig)
    assert config.task_type == "engine_surface_path"
    assert config.target_region == "carbon_deposit_region"
    assert config.executor_arm == "executor"
    assert config.observer_arm == "observer"
    assert config.tool_config_path.name == "carbon_remover.yaml"
    assert config.surface_patch.name == "carbon_deposit_patch"
    assert config.path.num_passes_u == 5
    assert config.path.num_passes_v == 4


def test_validate_engine_cleaning_task_config_requires_target_region(tmp_path: Path) -> None:
    config_path = _write_task_yaml(
        tmp_path,
        {
            "type": "engine_surface_path",
            "scene_config_path": "configs/scenes/engine_cleaning.yaml",
            "surface_patch": _surface_patch_values(),
            "path": _path_values(),
        }
    )

    with pytest.raises(ValueError, match="target_region"):
        load_engine_cleaning_task_config(config_path)


def test_validate_engine_cleaning_task_config_rejects_missing_surface_patch_section(
    tmp_path: Path,
) -> None:
    config_path = _write_task_yaml(
        tmp_path,
        {
            "type": "engine_surface_path",
            "scene_config_path": "configs/scenes/engine_cleaning.yaml",
            "target_region": "carbon_deposit_region",
            "path": _path_values(),
        }
    )

    with pytest.raises(ValueError, match="surface_patch"):
        load_engine_cleaning_task_config(config_path)


def test_validate_engine_cleaning_task_config_rejects_missing_path_section(
    tmp_path: Path,
) -> None:
    config_path = _write_task_yaml(
        tmp_path,
        {
            "type": "engine_surface_path",
            "scene_config_path": "configs/scenes/engine_cleaning.yaml",
            "target_region": "carbon_deposit_region",
            "surface_patch": _surface_patch_values(),
        }
    )

    with pytest.raises(ValueError, match="path"):
        load_engine_cleaning_task_config(config_path)


def _write_task_yaml(tmp_path: Path, task_values: dict[str, object]) -> Path:
    path = tmp_path / "engine_cleaning_task.yaml"
    path.write_text(yaml.safe_dump({"task": task_values}, sort_keys=False), encoding="utf-8")
    return path


def _surface_patch_values() -> dict[str, object]:
    return {
        "name": "carbon_deposit_patch",
        "type": "sphere_patch",
        "sphere_center": [0.18, 0.02, 0.35],
        "radius_m": 0.08,
        "patch_center": [0.18, 0.02, 0.43],
        "normal": [0.0, 0.0, 1.0],
        "tangent_u": [1.0, 0.0, 0.0],
        "size_u_m": 0.06,
        "size_v_m": 0.04,
    }


def _path_values() -> dict[str, object]:
    return {
        "num_passes_u": 5,
        "num_passes_v": 4,
        "approach_distance_m": 0.04,
        "retreat_distance_m": 0.05,
        "target_force_n": 1.0,
        "standoff_distance_m": 0.02,
        "snake_pattern": True,
    }
