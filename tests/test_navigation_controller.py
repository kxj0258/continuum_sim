from pathlib import Path

import numpy as np

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.control import compute_navigation_motor_velocity_command
from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml
from continuum_sim.scenes import load_navigation_scene_config
from continuum_sim.tasks import load_mujoco_navigation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_navigation_rocket.yaml"
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "rocket_nozzle_entry.yaml"


def test_navigation_controller_reports_clearance_and_avoidance_term() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    motor_params = load_motor_params_from_yaml(ROBOT_CONFIG)
    task_config = load_mujoco_navigation_config(TASK_CONFIG)
    scene = load_navigation_scene_config(SCENE_CONFIG)
    target = scene.target_positions(("entry_wall_30deg",))[0]

    command, info = compute_navigation_motor_velocity_command(
        np.zeros(9, dtype=float),
        target,
        params,
        physical_tendons,
        motor_params,
        scene.clearance_primitives,
        task_config.controller,
    )

    assert command.shape == (9,)
    assert np.all(np.isfinite(command))
    assert np.isfinite(float(info["min_clearance_m"]))
    assert np.asarray(info["centerline"]).shape[1] == 3
    assert np.asarray(info["avoidance_motor_velocity"]).shape == (9,)
