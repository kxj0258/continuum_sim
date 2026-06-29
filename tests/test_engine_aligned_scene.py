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
ALIGNED_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning_aligned.yaml"
NOZZLE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning_nozzle_collision.yaml"


def test_aligned_scene_config_loads_and_preserves_y_forward_pose() -> None:
    config = load_engine_scene_config(ALIGNED_CONFIG)

    assert config.engine.scale == pytest.approx(0.001)
    assert config.engine.pose.position_m.tolist() == pytest.approx([4.043, 1.12127, 0.0])
    assert config.engine.pose.quat_wxyz.tolist() == pytest.approx([0.5, 0.5, -0.5, -0.5])


def test_aligned_scene_allows_non_strict_asset_validation() -> None:
    config = load_engine_scene_config(ALIGNED_CONFIG)

    validate_engine_scene_config(config, strict_assets=False)


def test_nozzle_collision_candidate_keeps_aligned_scene_and_disabled_hints() -> None:
    config = load_engine_scene_config(NOZZLE_CONFIG)

    hints = list(iter_primitive_collision_geoms(config))

    assert config.engine.scale == pytest.approx(0.001)
    assert config.engine.pose.position_m.tolist() == pytest.approx([4.043, 1.12127, 0.0])
    assert config.engine.pose.quat_wxyz.tolist() == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert set(config.regions) >= {
        "entry_port",
        "inspection_roi",
        "carbon_deposit_region",
        "forbidden_zone",
    }
    assert [hint.name for hint in hints] == [
        "nozzle_collision_capsule_hint",
        "nozzle_collision_box_hint",
    ]
    assert [hint.enabled for hint in hints] == [False, False]
    assert config.engine.assets.collision_mesh_offset_m.tolist() == pytest.approx(
        [0.959563457031, 0.0, 0.0]
    )
    assert hints[0].fromto_m.tolist() == pytest.approx(
        [-4.394531e-06, -0.838058312988, 1.78673425293, -4.394531e-06, 0.838058312989, 1.78673425293]
    )


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
