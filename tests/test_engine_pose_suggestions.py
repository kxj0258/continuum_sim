from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.suggest_engine_pose import compute_pose_suggestion, main as suggest_pose_main


def test_compute_pose_suggestion_recenters_and_grounds_visual_mesh(tmp_path: Path) -> None:
    mesh_path = tmp_path / "visual.obj"
    mesh_path.write_text(
        """v 2 4 6
v 4 8 10
v 2 8 6
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(tmp_path, mesh_path.name, scale=0.5)

    suggestion = compute_pose_suggestion(
        config_path,
        target_center=(1.0, 2.0, 3.0),
        ground_z=0.25,
    )

    assert suggestion.current_pose_position == pytest.approx((0.0, 0.0, 0.0))
    assert suggestion.bbox_center_scaled == pytest.approx((1.5, 3.0, 4.0))
    assert suggestion.recenter_pose_position == pytest.approx((-0.5, -1.0, -1.0))
    assert suggestion.grounded_pose_position == pytest.approx((-1.5, -3.0, -2.75))


def test_suggest_pose_writes_candidate_yaml_without_overwriting_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mesh_path = tmp_path / "visual.obj"
    mesh_path.write_text(
        """v 0 0 0
v 2 2 2
v 0 2 0
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(tmp_path, mesh_path.name, scale=1.0)
    output_path = tmp_path / "engine_cleaning.suggested.yaml"

    result = suggest_pose_main(
        [
            "--config",
            str(config_path),
            "--target-center",
            "0",
            "0",
            "0",
            "--write-suggested-config",
            str(output_path),
        ]
    )

    assert result == 0
    assert output_path.exists()
    original = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    suggested = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert original["engine"]["pose"]["position_m"] == [0.0, 0.0, 0.0]
    assert suggested["engine"]["pose"]["position_m"] == [-1.0, -1.0, -1.0]
    assert "Suggested recenter pose.position_m" in capsys.readouterr().out


def test_suggest_pose_writes_aligned_config_with_grounded_pose_and_metadata(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "visual.obj"
    mesh_path.write_text(
        """v 2 4 6
v 4 8 10
v 2 8 6
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = _write_scene_config(tmp_path, mesh_path.name, scale=0.5)
    output_path = tmp_path / "engine_cleaning_grounded.generated.yaml"

    result = suggest_pose_main(
        [
            "--config",
            str(config_path),
            "--mode",
            "grounded",
            "--write-aligned-config",
            str(output_path),
        ]
    )

    assert result == 0
    aligned = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert aligned["name"] == "engine_cleaning_grounded.generated"
    assert aligned["engine"]["pose"]["position_m"] == [-1.5, -3.0, -3.0]
    assert aligned["metadata"]["generated_from"] == str(config_path)
    assert aligned["metadata"]["alignment_mode"] == "grounded"
    assert aligned["primitive_collision_geoms"][0]["enabled"] is False


def _write_scene_config(tmp_path: Path, visual_mesh: str, *, scale: float) -> Path:
    config_path = tmp_path / "engine_scene.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "test_engine",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": {"visual_mesh": visual_mesh},
                    "scale": scale,
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
