from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.scenes.engine_scene import load_engine_scene_config
from scripts.check_engine_assets import collect_engine_scene_diagnostics
from scripts.preview_engine_scene_mujoco import _engine_local_to_world, build_engine_preview_mjcf


def test_identity_quaternion_keeps_legacy_scaled_translated_bbox(tmp_path: Path) -> None:
    config_path = _write_scene_config(
        tmp_path,
        bbox_max=(2.0, 4.0, 6.0),
        scale=0.5,
        position_m=(1.0, 2.0, 3.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    visual = collect_engine_scene_diagnostics(config_path).asset_reports[0]

    assert visual.bbox_min_world == pytest.approx((1.0, 2.0, 3.0))
    assert visual.bbox_max_world == pytest.approx((2.0, 4.0, 6.0))
    assert visual.bbox_size_world == pytest.approx((1.0, 2.0, 3.0))


def test_bbox_world_size_applies_non_identity_rotation(tmp_path: Path) -> None:
    config_path = _write_scene_config(
        tmp_path,
        bbox_max=(2.0, 4.0, 6.0),
        scale=0.5,
        position_m=(1.0, 2.0, 3.0),
        quat_wxyz=(0.70710678118, 0.70710678118, 0.0, 0.0),
    )

    visual = collect_engine_scene_diagnostics(config_path).asset_reports[0]

    assert visual.bbox_size_scaled == pytest.approx((1.0, 2.0, 3.0))
    assert visual.bbox_size_world == pytest.approx((1.0, 3.0, 2.0))
    assert visual.bbox_size_world != pytest.approx(visual.bbox_size_scaled)


def test_yforward_pose_rotates_visual_bbox_size_to_expected_axes(tmp_path: Path) -> None:
    config_path = _write_scene_config(
        tmp_path,
        bbox_max=(2214.0, 2646.0, 1111.0),
        scale=0.001,
        position_m=(4.043, 1.12127, 0.0),
        quat_wxyz=(0.5, 0.5, -0.5, -0.5),
    )

    visual = collect_engine_scene_diagnostics(config_path).asset_reports[0]

    assert visual.bbox_size_scaled == pytest.approx((2.214, 2.646, 1.111))
    assert visual.bbox_size_world == pytest.approx((1.111, 2.214, 2.646))


def test_preview_bbox_marker_uses_rotation_aware_diagnostics(tmp_path: Path) -> None:
    config_path = _write_scene_config(
        tmp_path,
        bbox_max=(2.0, 4.0, 6.0),
        scale=0.5,
        position_m=(1.0, 2.0, 3.0),
        quat_wxyz=(0.70710678118, 0.70710678118, 0.0, 0.0),
    )

    xml_text = build_engine_preview_mjcf(config_path, show_bbox=True, show_regions=False, show_axes=False)

    assert 'fromto="1 -1 3 2 -1 3"' in xml_text
    assert 'fromto="2 -1 3 2 -1 5"' in xml_text


def test_engine_frame_primitive_local_point_applies_scale_rotation_and_translation(tmp_path: Path) -> None:
    config_path = _write_scene_config(
        tmp_path,
        bbox_max=(1.0, 1.0, 1.0),
        scale=0.5,
        position_m=(1.0, 2.0, 3.0),
        quat_wxyz=(0.70710678118, 0.70710678118, 0.0, 0.0),
    )
    config = load_engine_scene_config(config_path)

    world_point = _engine_local_to_world((0.0, 2.0, 0.0), config)

    assert world_point == pytest.approx((1.0, 2.0, 4.0))


def test_engine_frame_primitive_diagnostics_intersect_rotation_aware_visual_bbox(tmp_path: Path) -> None:
    config_path = _write_scene_config(
        tmp_path,
        bbox_max=(2.0, 4.0, 6.0),
        scale=0.5,
        position_m=(1.0, 2.0, 3.0),
        quat_wxyz=(0.70710678118, 0.70710678118, 0.0, 0.0),
        primitive_collision_geoms=[
            {
                "name": "engine_frame_sphere",
                "type": "sphere",
                "enabled": False,
                "frame": "engine",
                "position_m": [0.0, 2.0, 0.0],
                "radius_m": 0.2,
            }
        ],
    )

    diagnostics = collect_engine_scene_diagnostics(config_path)
    hint = diagnostics.primitive_hint_reports[0]

    assert hint.bbox_min == pytest.approx((0.9, 1.9, 3.9))
    assert hint.bbox_max == pytest.approx((1.1, 2.1, 4.1))
    assert hint.intersects_visual_bbox is True


def _write_scene_config(
    tmp_path: Path,
    *,
    bbox_max: tuple[float, float, float],
    scale: float,
    position_m: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    primitive_collision_geoms: list[dict[str, object]] | None = None,
) -> Path:
    visual_mesh = tmp_path / "visual.obj"
    visual_mesh.write_text(
        f"""v 0 0 0
v {bbox_max[0]} {bbox_max[1]} {bbox_max[2]}
v 0 {bbox_max[1]} 0
f 1 2 3
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "engine_scene.yaml"
    raw_config = {
        "name": "test_engine",
        "scene_type": "engine_cleaning",
        "engine": {
            "assets": {"visual_mesh": visual_mesh.name},
            "scale": scale,
            "pose": {
                "position_m": list(position_m),
                "quat_wxyz": list(quat_wxyz),
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
    }
    if primitive_collision_geoms is not None:
        raw_config["primitive_collision_geoms"] = primitive_collision_geoms
    config_path.write_text(
        yaml.safe_dump(
            raw_config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
