from pathlib import Path

import yaml
from numpy.testing import assert_allclose

from continuum_sim.scenes import build_mujoco_wiping_xml, load_navigation_scene_config
from continuum_sim.tasks import build_raster_wiping_path, load_mujoco_wiping_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_wiping_board.yaml"
BASE_XML = PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon.xml"


def test_load_mujoco_wiping_config_from_yaml() -> None:
    config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(config.scene.config_path)

    assert config.robot_config_path.is_file()
    assert config.scene.config_path.name == "wiping_board.yaml"
    assert config.tool.to_xml_config().type == "sphere"
    assert config.tool.contact_site_name == "tool_contact_site"
    assert config.simulation.initial_motor_position_rad.shape == (9,)
    assert config.controller.target_normal_force_n > 0.0
    assert config.motion.surface_id == "board_surface"
    assert config.mujoco.show_live_force_panel is True
    assert config.mujoco.live_force_panel_stride == 1
    assert config.mujoco.live_force_panel_history_points == 300
    assert len(scene.work_surfaces) == 1
    assert len(scene.wipe_patches) == 1


def test_load_mujoco_wiping_config_uses_live_force_panel_defaults(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"].pop("show_live_force_panel", None)
    raw["mujoco"].pop("live_force_panel_stride", None)
    raw["mujoco"].pop("live_force_panel_history_points", None)
    config_path = tmp_path / "wiping_without_force_panel_fields.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_mujoco_wiping_config(config_path)

    assert config.mujoco.show_live_force_panel is False
    assert config.mujoco.live_force_panel_stride == 1
    assert config.mujoco.live_force_panel_history_points == 300


def test_load_mujoco_wiping_config_accepts_dynamic_controller_type(tmp_path: Path) -> None:
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    raw["controller"]["type"] = "dynamic_adaptive_impedance"
    raw["controller"]["dynamics_config_path"] = str(PROJECT_ROOT / "configs" / "dynamics" / "pcc_reduced.yaml")
    config_path = tmp_path / "dynamic_wiping.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_mujoco_wiping_config(config_path)

    assert config.controller.type == "dynamic_adaptive_impedance"
    assert config.controller.dynamics_config_path.name == "pcc_reduced.yaml"


def test_dynamic_mujoco_config_uses_lower_tendon_position_kp() -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "mujoco.yaml").read_text(encoding="utf-8"))

    assert raw["actuators"]["tendon_position"]["kp"] == 500000.0
    assert raw["actuators"]["joint_position"]["kp"] == 2.0


def test_raster_wipe_path_uses_surface_frame_and_boustrophedon_order() -> None:
    config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(config.scene.config_path)
    surface = scene.work_surface(config.motion.surface_id)
    path = build_raster_wiping_path(
        config.motion,
        surface,
        contact_radius_m=config.tool.radius_m,
    )

    expected_count = 1 + config.motion.line_count * config.motion.samples_per_line
    assert path.target_position.shape == (expected_count, 3)
    assert path.target_pose.shape == (expected_count, 4, 4)
    assert path.phase[0] == "approach"
    assert set(path.phase[1:]) == {"contact"}

    contact_origin = config.motion.center_m + (
        config.tool.radius_m + config.motion.contact_offset_m
    ) * surface.normal
    first_contact = contact_origin - 0.5 * config.motion.width_m * surface.tangent_u - 0.5 * config.motion.height_m * surface.tangent_v
    second_line_start_index = 1 + config.motion.samples_per_line
    second_line_start = contact_origin + 0.5 * config.motion.width_m * surface.tangent_u - 0.25 * config.motion.height_m * surface.tangent_v

    assert_allclose(path.target_position[1], first_contact)
    assert_allclose(path.target_position[second_line_start_index], second_line_start)
    assert_allclose(path.target_pose[0, :3, 2], surface.normal)


def test_wiping_xml_builder_injects_board_and_tool_pad(tmp_path: Path) -> None:
    config = load_mujoco_wiping_config(TASK_CONFIG)
    scene = load_navigation_scene_config(config.scene.config_path)
    output_path = build_mujoco_wiping_xml(
        BASE_XML,
        scene,
        config.tool.to_xml_config(),
        tmp_path / "wiping.xml",
    )

    import xml.etree.ElementTree as ET

    root = ET.parse(output_path).getroot()
    geoms = {
        geom.attrib["name"]: geom
        for geom in root.findall(".//geom")
        if "name" in geom.attrib
    }
    sites = {
        site.attrib["name"]: site
        for site in root.findall(".//site")
        if "name" in site.attrib
    }

    assert geoms["scene_board_surface_geom"].attrib["type"] == "box"
    assert geoms["tool_contact_pad"].attrib["type"] == "sphere"
    assert "tool_contact_site" in sites
