from pathlib import Path
import xml.etree.ElementTree as ET

from continuum_sim.scenes import (
    build_mujoco_scene_xml,
    build_mujoco_wiping_xml,
    load_navigation_scene_config,
)
from continuum_sim.tasks import load_mujoco_wiping_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_XML = PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon_with_visuals.xml"
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "rocket_nozzle_entry.yaml"
WIPING_TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_wiping_board.yaml"


def test_scene_builder_injects_shell_obstacles_and_targets(tmp_path: Path) -> None:
    scene = load_navigation_scene_config(SCENE_CONFIG)
    output_path = build_mujoco_scene_xml(BASE_XML, scene, tmp_path / "scene.xml")

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

    assert any(name.startswith("scene_nozzle_entry_shell_s") for name in geoms)
    assert "scene_sensor_boss_obstacle" in geoms
    assert geoms["scene_sensor_boss_obstacle"].attrib["type"] == "cylinder"
    assert geoms["scene_sensor_boss_obstacle"].attrib["quat"]
    assert "scene_target_entry_wall_30deg" in sites

    mesh = root.find(".//mesh[@name='segment_2_link_3_visual_mesh']")
    assert mesh is not None
    mesh_path = (output_path.parent / mesh.attrib["file"]).resolve()
    assert mesh_path == (
        PROJECT_ROOT
        / "assets"
        / "meshes"
        / "mujoco_visual_segments"
        / "segment_2_link_3_visual.stl"
    )
    assert mesh_path.is_file()


def test_wiping_xml_builder_injects_board_and_tip_contact_pad(tmp_path: Path) -> None:
    task_config = load_mujoco_wiping_config(WIPING_TASK_CONFIG)
    scene = load_navigation_scene_config(task_config.scene.config_path)
    output_path = build_mujoco_wiping_xml(
        PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon.xml",
        scene,
        task_config.tool.to_xml_config(),
        tmp_path / "wiping.xml",
    )

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
    bodies = {
        body.attrib["name"]: body
        for body in root.findall(".//body")
        if "name" in body.attrib
    }

    assert "scene_board_surface_geom" in geoms
    assert geoms["scene_board_surface_geom"].attrib["type"] == "box"
    assert "scene_board_frame_top" in geoms
    assert "tool_contact_pad" in geoms
    assert geoms["tool_contact_pad"].attrib["type"] == "sphere"
    assert "tool_contact_site" in sites
    assert "tool_contact_pad_body" in bodies
