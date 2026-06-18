from pathlib import Path
from xml.etree import ElementTree

from continuum_sim import load_mujoco_config
from continuum_sim.model import ThreeSegmentRobotParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"
SEGMENTED_VISUALS_TEMPLATE = PROJECT_ROOT / "assets" / "mujoco" / "segmented_visuals_template.xml"


def test_mujoco_xml_asset_exists_and_is_referenced_by_config() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)

    assert config.xml_path == PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml"
    assert config.xml_path.is_file()


def test_mujoco_xml_contains_reduced_order_chain_structure() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    root = ElementTree.parse(config.xml_path).getroot()

    joints = [joint for joint in root.findall(".//joint") if joint.get("name") is not None]
    actuator_root = root.find("actuator")
    assert actuator_root is not None
    actuators = list(actuator_root)
    geoms = root.findall(".//geom[@type='capsule']") + [
        geom for geom in root.findall(".//geom") if geom.get("fromto") is not None
    ]
    sites = {site.get("name") for site in root.findall(".//site")}

    assert len(joints) == 3 * config.links_per_segment * 2
    assert len(actuators) == len(joints)
    assert len(geoms) >= 3 * config.links_per_segment
    assert {
        "base_site",
        "segment_1_tip",
        "segment_2_tip",
        "segment_3_tip",
        "tip",
    }.issubset(sites)

    link_length = 0.01
    assert 3 * config.links_per_segment * link_length == sum(params.segment_lengths)
    default_geom = root.find("./default/geom")
    base_geom = root.find(".//geom[@name='base_geom']")
    assert default_geom is not None
    assert base_geom is not None
    assert default_geom.get("group") == str(config.visuals.collision_geom_group)
    assert base_geom.get("group") == str(config.visuals.collision_geom_group)


def test_mujoco_xml_visual_background_does_not_add_physics_geoms() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)
    root = ElementTree.parse(config.xml_path).getroot()

    visual = root.find("./visual")
    skybox = root.find("./asset/texture[@type='skybox']")
    planes = root.findall(".//geom[@type='plane']")

    assert visual is not None
    assert visual.find("./headlight") is not None
    assert skybox is None
    assert planes == []


def test_segmented_visual_template_matches_configured_meshes() -> None:
    config = load_mujoco_config(MUJOCO_CONFIG)
    root = ElementTree.parse(SEGMENTED_VISUALS_TEMPLATE).getroot()

    mesh_assets = root.findall("./asset/mesh")
    mesh_files = {Path(str(mesh.get("file"))).name for mesh in mesh_assets}

    assert SEGMENTED_VISUALS_TEMPLATE.is_file()
    assert mesh_files == set(config.visuals.expected_meshes)
    assert all(mesh.get("scale") == "0.001 0.001 0.001" for mesh in mesh_assets)


def test_segmented_visual_template_geoms_are_non_collision_visuals() -> None:
    root = ElementTree.parse(SEGMENTED_VISUALS_TEMPLATE).getroot()
    visual_geoms = root.findall(".//geom[@type='mesh']")

    assert len(visual_geoms) == 13
    assert all(geom.get("contype") == "0" for geom in visual_geoms)
    assert all(geom.get("conaffinity") == "0" for geom in visual_geoms)
    assert all(geom.get("density") == "0" for geom in visual_geoms)
    assert all(geom.get("group") == "1" for geom in visual_geoms)
    assert all(str(geom.get("name", "")).endswith("_visual_geom") for geom in visual_geoms)
