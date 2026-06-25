from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.scenes.engine_scene import load_engine_scene_config
from scripts.suggest_nozzle_collision import build_nozzle_collision_hints, main as suggest_nozzle_main


def test_build_nozzle_collision_hints_from_collision_bbox() -> None:
    bbox_min = (-1.89074, -0.112827, 0.0)
    bbox_max = (-0.0283876, 1.03946, 1.11113)

    hints = build_nozzle_collision_hints(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        axis="longest",
        radius_scale=0.15,
        enabled=False,
    )

    capsule = hints[0]
    box = hints[1]
    assert capsule["name"] == "nozzle_collision_capsule_hint"
    assert capsule["enabled"] is False
    assert capsule["fromto_m"] == pytest.approx(
        [-1.79762238, 0.4633165, 0.555565, -0.12150522, 0.4633165, 0.555565]
    )
    assert capsule["radius_m"] == pytest.approx(0.1666695)
    assert box["position_m"] == pytest.approx([-0.9595638, 0.4633165, 0.555565])
    assert box["size_m"] == pytest.approx([0.9311762, 0.5761435, 0.555565])


def test_suggest_nozzle_collision_writes_candidate_config(tmp_path: Path) -> None:
    config_path = _write_tiny_scene(tmp_path)
    output_path = tmp_path / "engine_cleaning_nozzle_collision.yaml"

    result = suggest_nozzle_main(
        [
            "--config",
            str(config_path),
            "--source",
            "collision",
            "--primitive",
            "capsule",
            "--output-config",
            str(output_path),
        ]
    )

    assert result == 0
    raw = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert raw["metadata"]["generated_from"] == config_path.as_posix()
    assert raw["metadata"]["collision_hint_source"] == "collision"
    assert raw["primitive_collision_geoms"][0]["type"] == "capsule"
    assert raw["primitive_collision_geoms"][0]["enabled"] is False
    load_engine_scene_config(output_path)


def test_suggest_nozzle_collision_can_enable_hint(tmp_path: Path) -> None:
    config_path = _write_tiny_scene(tmp_path)
    output_path = tmp_path / "enabled.yaml"

    result = suggest_nozzle_main(
        [
            "--config",
            str(config_path),
            "--source",
            "collision",
            "--primitive",
            "box",
            "--output-config",
            str(output_path),
            "--enable-hint",
        ]
    )

    assert result == 0
    raw = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert raw["primitive_collision_geoms"][0]["enabled"] is True
    assert raw["primitive_collision_geoms"][0]["type"] == "box"


def _write_tiny_scene(tmp_path: Path) -> Path:
    visual_mesh = tmp_path / "visual.obj"
    collision_mesh = tmp_path / "collision.obj"
    visual_mesh.write_text(
        """v 0 0 0
v 1 1 1
v 0 1 0
f 1 2 3
""",
        encoding="utf-8",
    )
    collision_mesh.write_text(
        """v -2 0 0
v 2 1 1
v -2 1 0
f 1 2 3
""",
        encoding="utf-8",
    )
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
