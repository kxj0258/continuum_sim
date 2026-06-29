from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.scenes.engine_scene import (
    iter_primitive_collision_geoms,
    load_engine_scene_config,
    validate_engine_scene_config,
)
from scripts.check_engine_assets import collect_engine_scene_diagnostics
from scripts.preview_engine_scene_mujoco import build_engine_preview_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning.yaml"


def test_engine_scene_config_loads_with_identity_engine_pose() -> None:
    config = load_engine_scene_config(SCENE_CONFIG)

    assert config.engine.scale == pytest.approx(0.001)
    assert config.engine.pose.position_m.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert config.engine.pose.quat_wxyz.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_engine_scene_allows_non_strict_asset_validation() -> None:
    config = load_engine_scene_config(SCENE_CONFIG)

    validate_engine_scene_config(config, strict_assets=False)


def test_engine_scene_preserves_entry_region_and_exploration_metadata() -> None:
    config = load_engine_scene_config(SCENE_CONFIG)

    hints = list(iter_primitive_collision_geoms(config))

    assert config.engine.scale == pytest.approx(0.001)
    assert config.engine.pose.position_m.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert config.engine.pose.quat_wxyz.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert set(config.regions) == {"entry_port"}
    assert len(hints) == 1
    assert hints[0].name == "debug_box_1"
    assert hints[0].frame == "world"
    assert hints[0].enabled is True
    assert config.engine.assets.collision_mesh_offset_m.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert config.exploration_start is not None
    assert config.exploration_start.frame == "engine"
    assert config.exploration_start.point_m.tolist() == pytest.approx([0.442, 1.58169, 1.74693])
    assert config.exploration_start.normal.tolist() == pytest.approx([0.0, -0.96436878, 0.26456163])
    assert len(config.exploration_paths) == 1
    path = config.exploration_paths[0]
    assert path.name == "nozzle_axis_entry"
    assert path.frame == "engine"
    assert path.points_m[0].tolist() == pytest.approx([0.442, 1.58169, 1.74693])
    assert path.points_m[1].tolist() == pytest.approx([0.442, 0.37281, 2.07857])


def test_preview_mjcf_includes_engine_axes_and_exploration_overlays_for_repository_scene() -> None:
    xml_text = build_engine_preview_mjcf(SCENE_CONFIG)

    assert 'name="engine_x_axis"' in xml_text
    assert 'name="engine_y_axis"' in xml_text
    assert 'name="engine_z_axis"' in xml_text
    assert 'name="exploration_nozzle_axis_entry_segment_0"' in xml_text
    assert 'name="exploration_start_point"' in xml_text
    assert 'name="exploration_start_normal"' in xml_text


def test_engine_scene_loader_exposes_primitive_collision_geoms(tmp_path: Path) -> None:
    config_path = _write_tiny_aligned_scene(tmp_path)
    config = load_engine_scene_config(config_path)

    hints = list(iter_primitive_collision_geoms(config))

    assert len(hints) == 1
    assert hints[0].name == "nozzle_collision_hint"
    assert hints[0].enabled is False


def test_aligned_scene_diagnostics_include_primitive_collision_hint_from_temp_config(tmp_path: Path) -> None:
    config_path = _write_tiny_aligned_scene(tmp_path)
    diagnostics = collect_engine_scene_diagnostics(config_path, strict_assets=False)

    assert diagnostics.primitive_hint_reports
    hint = diagnostics.primitive_hint_reports[0]
    assert hint.name == "nozzle_collision_hint"
    assert hint.enabled is False
    assert hint.type == "capsule"
    assert hint.bbox_min is not None
    assert hint.intersects_visual_bbox is True
    assert hint.intersects_collision_bbox is True


def test_preview_mjcf_can_show_disabled_primitive_collision_hints(tmp_path: Path) -> None:
    config_path = _write_tiny_aligned_scene(tmp_path)
    xml_text = build_engine_preview_mjcf(
        config_path,
        show_primitive_collision=True,
        show_disabled_hints=True,
    )

    assert 'name="hint_nozzle_collision_hint"' in xml_text
    assert 'name="region_entry_port"' in xml_text


def test_preview_mjcf_can_hide_mesh_collision_when_showing_primitive(tmp_path: Path) -> None:
    config_path = _write_tiny_aligned_scene(tmp_path)

    xml_text = build_engine_preview_mjcf(
        config_path,
        show_primitive_collision=True,
        show_disabled_hints=True,
        hide_mesh_collision=True,
    )

    assert 'name="engine_collision"' not in xml_text
    assert 'name="hint_nozzle_collision_hint"' in xml_text


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
    config_path = tmp_path / "engine_cleaning.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "engine_cleaning",
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
