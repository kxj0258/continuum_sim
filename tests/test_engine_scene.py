from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.scenes.engine_scene import (
    EngineAssetConfig,
    EngineModelConfig,
    EnginePoseConfig,
    EngineSceneConfig,
    iter_engine_regions,
    load_engine_scene_config,
    resolve_engine_asset_paths,
    validate_engine_scene_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "engine_cleaning.yaml"


def test_load_engine_scene_config_reads_engine_pose_scale_and_regions() -> None:
    config = load_engine_scene_config(SCENE_CONFIG)

    assert isinstance(config, EngineSceneConfig)
    assert config.scene_type == "engine_cleaning"
    assert config.engine.scale == pytest.approx(0.001)
    assert config.engine.pose.position_m.tolist() == pytest.approx([0.35, 0.0, 0.12])
    assert config.engine.pose.quat_wxyz.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert tuple(config.regions) == (
        "entry_port",
        "inspection_roi",
        "carbon_deposit_region",
        "forbidden_zone",
    )


def test_load_engine_scene_config_parses_region_fields() -> None:
    config = load_engine_scene_config(SCENE_CONFIG)
    entry_port = config.regions["entry_port"]
    forbidden_zone = config.regions["forbidden_zone"]

    assert entry_port.type == "circular_port"
    assert entry_port.center_m.tolist() == pytest.approx([0.28, 0.0, 0.1])
    assert entry_port.normal.tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert entry_port.radius_m == pytest.approx(0.045)

    assert forbidden_zone.type == "box"
    assert forbidden_zone.center_m.tolist() == pytest.approx([0.42, 0.0, 0.18])
    assert forbidden_zone.size_m.tolist() == pytest.approx([0.08, 0.06, 0.04])


def test_iter_engine_regions_preserves_named_region_order() -> None:
    config = load_engine_scene_config(SCENE_CONFIG)

    names = [name for name, _region in iter_engine_regions(config)]

    assert names == [
        "entry_port",
        "inspection_roi",
        "carbon_deposit_region",
        "forbidden_zone",
    ]


def test_validate_engine_scene_config_allows_missing_assets_when_not_strict(tmp_path: Path) -> None:
    config = load_engine_scene_config(_write_missing_asset_scene_config(tmp_path))

    with pytest.warns(UserWarning, match="does not exist"):
        validate_engine_scene_config(config, strict_assets=False)


def test_validate_engine_scene_config_rejects_missing_assets_when_strict(tmp_path: Path) -> None:
    config = load_engine_scene_config(_write_missing_asset_scene_config(tmp_path))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_engine_scene_config(config, strict_assets=True)


def test_load_engine_scene_config_requires_engine_section(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_engine.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "missing_engine",
                "scene_type": "engine_cleaning",
                "regions": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="engine"):
        load_engine_scene_config(config_path)


def test_load_engine_scene_config_rejects_non_mapping_regions(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_regions.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "invalid_regions",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": {
                        "visual_mesh": "assets/placeholder/engine_visual.obj",
                        "collision_mesh": "assets/placeholder/engine_collision.obj",
                    },
                    "scale": 1.0,
                    "pose": {
                        "position_m": [0.0, 0.0, 0.0],
                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "regions": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="regions"):
        load_engine_scene_config(config_path)


def test_load_engine_scene_config_rejects_unknown_region_type(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown_region.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "unknown_region",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": {
                        "visual_mesh": "assets/placeholder/engine_visual.obj",
                        "collision_mesh": "assets/placeholder/engine_collision.obj",
                    },
                    "scale": 1.0,
                    "pose": {
                        "position_m": [0.0, 0.0, 0.0],
                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "regions": {
                    "entry_port": {
                        "type": "mystery_shape",
                        "center_m": [0.0, 0.0, 0.0],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="region type"):
        load_engine_scene_config(config_path)


def test_resolve_engine_asset_paths_reanchors_relative_assets(tmp_path: Path) -> None:
    config = EngineSceneConfig(
        path=tmp_path / "engine_scene.yaml",
        name="engine_scene",
        scene_type="engine_cleaning",
        engine=EngineModelConfig(
            assets=EngineAssetConfig(
                visual_mesh=Path("assets/visual.obj"),
                collision_mesh=Path("assets/collision.obj"),
                collision_geoms=None,
            ),
            scale=1.0,
            pose=EnginePoseConfig(
                position_m=[0.0, 0.0, 0.0],
                quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            ),
        ),
        regions={},
    )

    resolved = resolve_engine_asset_paths(config, tmp_path)

    assert resolved.visual_mesh == (tmp_path / "assets" / "visual.obj").resolve()
    assert resolved.collision_mesh == (tmp_path / "assets" / "collision.obj").resolve()


def _write_missing_asset_scene_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "missing_assets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "missing_assets",
                "scene_type": "engine_cleaning",
                "engine": {
                    "assets": {
                        "visual_mesh": "assets/missing_visual.obj",
                        "collision_mesh": "assets/missing_collision.obj",
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
