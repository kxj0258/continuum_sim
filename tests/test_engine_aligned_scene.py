from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.scenes.engine_scene import load_engine_scene_config, validate_engine_scene_config
from scripts.check_engine_assets import collect_engine_scene_diagnostics
from scripts.preview_engine_scene_mujoco import build_engine_preview_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning_aligned.yaml"


def test_aligned_scene_config_loads_and_preserves_grounded_pose() -> None:
    config = load_engine_scene_config(ALIGNED_CONFIG)

    assert config.engine.scale == pytest.approx(0.001)
    assert config.engine.pose.position_m.tolist() == pytest.approx([-1.12127, -1.32342, -3.48744])


def test_aligned_scene_allows_non_strict_asset_validation() -> None:
    config = load_engine_scene_config(ALIGNED_CONFIG)

    validate_engine_scene_config(config, strict_assets=False)


def test_aligned_scene_diagnostics_include_primitive_collision_hint_from_temp_config(tmp_path: Path) -> None:
    config_path = _write_tiny_aligned_scene(tmp_path)
    diagnostics = collect_engine_scene_diagnostics(config_path, strict_assets=False)

    assert diagnostics.primitive_hint_reports
    hint = diagnostics.primitive_hint_reports[0]
    assert hint.name == "nozzle_collision_hint"
    assert hint.enabled is False
    assert hint.type == "capsule"
    assert hint.bbox_min is not None


def test_preview_mjcf_can_show_disabled_primitive_collision_hints(tmp_path: Path) -> None:
    config_path = _write_tiny_aligned_scene(tmp_path)
    xml_text = build_engine_preview_mjcf(config_path, show_disabled_hints=True)

    assert 'name="hint_nozzle_collision_hint"' in xml_text
    assert 'name="region_entry_port"' in xml_text


def _write_tiny_aligned_scene(tmp_path: Path) -> Path:
    visual_mesh = tmp_path / "visual.obj"
    collision_mesh = tmp_path / "collision.obj"
    mesh_text = """v 0 0 0
v 1 1 1
v 0 1 0
f 1 2 3
"""
    visual_mesh.write_text(mesh_text, encoding="utf-8")
    collision_mesh.write_text(mesh_text, encoding="utf-8")
    config_path = tmp_path / "engine_cleaning_aligned.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "engine_cleaning_aligned",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": {
                        "visual_mesh": visual_mesh.name,
                        "collision_mesh": collision_mesh.name,
                    },
                    "scale": 1.0,
                    "pose": {
                        "position_m": [0.0, 0.0, 0.0],
                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "primitive_collision_geoms": [
                    {
                        "name": "nozzle_collision_hint",
                        "type": "capsule",
                        "enabled": False,
                        "position_m": [0.0, 0.0, 0.4],
                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "radius_m": 0.03,
                        "length_m": 0.4,
                    }
                ],
                "regions": {
                    "entry_port": {
                        "type": "circular_port",
                        "center_m": [0.0, 0.0, 0.0],
                        "normal": [1.0, 0.0, 0.0],
                        "radius_m": 0.1,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
