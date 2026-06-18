from pathlib import Path

import yaml

from continuum_sim.tasks import load_mujoco_navigation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_navigation_rocket.yaml"


def test_mujoco_navigation_config_loads_paths_and_mission() -> None:
    config = load_mujoco_navigation_config(TASK_CONFIG)

    assert config.robot_config_path.is_file()
    assert config.scene_config_path.is_file()
    assert config.generated_scene_xml_path.name == "rocket_nozzle_entry_navigation.xml"
    assert config.controller.type == "navigation_differential_ik"
    assert config.controller.clearance_min_m < config.controller.avoidance_influence_m
    assert config.mission.waypoint_ids == (
        "entry_wall_30deg",
        "rib_gap_center",
        "throat_wall_210deg",
    )


def test_mujoco_navigation_config_accepts_cbf_qp_controller(tmp_path: Path) -> None:
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    raw["controller"]["type"] = "navigation_cbf_qp"
    config_path = tmp_path / "navigation_cbf.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_mujoco_navigation_config(config_path)

    assert config.controller.type == "navigation_cbf_qp"
