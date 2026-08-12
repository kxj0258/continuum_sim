from __future__ import annotations

from pathlib import Path

from numpy.testing import assert_allclose
import pytest
import yaml

from continuum_sim.model.multi_arm import load_multi_arm_config
from continuum_sim.tools.attachments import (
    AttachmentConfig,
    get_attachment,
    load_attachment_config,
    load_attachment_registry,
    validate_attachment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARBON_TOOL_CONFIG = PROJECT_ROOT / "configs" / "tools" / "carbon_remover.yaml"
CAMERA_AIRGUN_CONFIG = PROJECT_ROOT / "configs" / "tools" / "eye_camera_air_gun.yaml"
DUAL_CONTINUUM_CONFIG = PROJECT_ROOT / "configs" / "robots" / "dual_continuum.yaml"


def test_load_carbon_remover_reads_contact_tool_fields() -> None:
    config = load_attachment_config(CARBON_TOOL_CONFIG)

    assert isinstance(config, AttachmentConfig)
    assert config.name == "carbon_removal_tool"
    assert config.type == "contact_sphere_tool"
    assert config.enabled is True
    assert_allclose(config.tip_to_attachment.position, [0.0, 0.0, 0.004])
    assert_allclose(config.tcp_pose.position, [0.0, 0.0, 0.014])
    assert config.collision.type == "sphere"
    assert config.collision.radius_m == pytest.approx(0.009)
    assert_allclose(config.collision.position, [0.0, 0.0, 0.005])
    assert config.contact.target_normal_force_n == pytest.approx(1.0)
    assert config.contact.max_normal_force_n == pytest.approx(3.0)
    assert config.contact.standoff_distance_m == pytest.approx(0.02)
    assert config.mass_kg == pytest.approx(0.01)
    assert config.force_torque_sensor is not None
    assert_allclose(config.force_torque_sensor.size_m, [0.015, 0.015, 0.008])
    assert config.force_torque_sensor.mass_kg == pytest.approx(0.03)
    assert config.force_torque_sensor.filter_cutoff_hz == pytest.approx(15.0)


def test_load_eye_camera_air_gun_reads_camera_nozzle_and_airgun_fields() -> None:
    config = load_attachment_config(CAMERA_AIRGUN_CONFIG)

    assert config.name == "eye_camera_air_gun"
    assert config.type == "camera_airgun"
    assert config.camera.name == "observer_eye_camera"
    assert config.camera.intrinsics.width == 640
    assert config.camera.intrinsics.height == 480
    assert_allclose(config.camera.tip_to_camera.position, [0.0, 0.0, 0.004])
    assert config.camera_visual is not None
    assert config.camera_visual.shape == "hemisphere"
    assert config.camera_visual.radius_m == pytest.approx(0.00375)
    assert config.camera_visual.lens_radius_m == pytest.approx(0.0015)
    assert_allclose(config.nozzle_pose.position, [0.0, 0.015, 0.045])
    assert config.airgun.standoff_distance_m == pytest.approx(0.05)


def test_load_attachment_config_rejects_invalid_attachment_type(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_type.yaml"
    _write_tool_config(config_path, {"name": "bad", "type": "laser_pointer"})

    with pytest.raises(ValueError, match="type"):
        load_attachment_config(config_path)


def test_load_attachment_config_rejects_nonpositive_radius(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_radius.yaml"
    values = _contact_tool_values()
    values["collision"]["radius_m"] = 0.0
    _write_tool_config(config_path, values)

    with pytest.raises(ValueError, match="radius"):
        load_attachment_config(config_path)


def test_load_attachment_config_rejects_nonpositive_max_force(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_force.yaml"
    values = _contact_tool_values()
    values["contact"]["max_normal_force_n"] = 0.0
    _write_tool_config(config_path, values)

    with pytest.raises(ValueError, match="max_normal_force_n"):
        load_attachment_config(config_path)


def test_validate_attachment_config_allows_missing_visual_mesh_when_not_strict(tmp_path: Path) -> None:
    config_path = tmp_path / "mesh_placeholder.yaml"
    values = _contact_tool_values()
    values["visual_mesh_path"] = "missing/tool.obj"
    _write_tool_config(config_path, values)
    config = load_attachment_config(config_path)

    with pytest.warns(UserWarning, match="does not exist"):
        validate_attachment_config(config, strict_assets=False)


def test_validate_attachment_config_rejects_missing_visual_mesh_when_strict(tmp_path: Path) -> None:
    config_path = tmp_path / "mesh_placeholder.yaml"
    values = _contact_tool_values()
    values["visual_mesh_path"] = "missing/tool.obj"
    _write_tool_config(config_path, values)
    config = load_attachment_config(config_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_attachment_config(config, strict_assets=True)


def test_attachment_registry_queries_tools_by_name() -> None:
    registry = load_attachment_registry([CARBON_TOOL_CONFIG, CAMERA_AIRGUN_CONFIG])

    assert get_attachment(registry, "carbon_removal_tool").type == "contact_sphere_tool"
    assert get_attachment(registry, "eye_camera_air_gun").type == "camera_airgun"


def test_dual_continuum_attachment_names_match_registry() -> None:
    multi_arm = load_multi_arm_config(DUAL_CONTINUUM_CONFIG)
    registry = load_attachment_registry([CARBON_TOOL_CONFIG, CAMERA_AIRGUN_CONFIG])

    for arm in multi_arm.arms.values():
        assert arm.attachment is not None
        assert get_attachment(registry, arm.attachment).name == arm.attachment


def _write_tool_config(path: Path, values: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"tool": values}, sort_keys=False), encoding="utf-8")


def _contact_tool_values() -> dict[str, object]:
    return {
        "name": "carbon_removal_tool",
        "type": "contact_sphere_tool",
        "enabled": True,
        "tip_to_attachment": {
            "position": [0.0, 0.0, 0.02],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        "tcp_pose": {
            "position": [0.0, 0.0, 0.045],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        "collision": {
            "type": "sphere",
            "radius_m": 0.018,
        },
        "contact": {
            "target_normal_force_n": 1.0,
            "max_normal_force_n": 3.0,
            "standoff_distance_m": 0.02,
        },
    }
