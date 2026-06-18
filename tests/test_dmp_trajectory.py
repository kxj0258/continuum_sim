from pathlib import Path

import numpy as np
import yaml
from numpy.testing import assert_allclose

from continuum_sim.model import ThreeSegmentRobotParams
from continuum_sim.tasks import build_target_positions, load_tracking_config
from continuum_sim.tasks.dmp_trajectory import DiscreteDMP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "pcc_trajectory_tracking.yaml"


def test_discrete_dmp_rollout_adapts_start_and_goal() -> None:
    time = np.linspace(0.0, 1.0, 60)
    demo = np.column_stack(
        (
            0.04 * time,
            0.01 * np.sin(np.pi * time),
            0.12 + 0.02 * time,
        )
    )
    dmp = DiscreteDMP(basis_count=16, samples=80).imitate(time, demo)

    start = np.array([0.02, -0.01, 0.10], dtype=float)
    goal = np.array([0.08, 0.01, 0.18], dtype=float)
    rollout = dmp.rollout(start, goal, tau=1.5)

    assert rollout.position.shape == (80, 3)
    assert np.all(np.isfinite(rollout.position))
    assert_allclose(rollout.position[0], start, atol=1.0e-12)
    assert_allclose(rollout.position[-1], goal, atol=3.0e-3)
    assert float(np.max(rollout.position[:, 1]) - np.min(rollout.position[:, 1])) > 0.005


def test_tracking_config_builds_dmp_targets_from_csv(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRACKING_CONFIG.read_text(encoding="utf-8"))
    demo_path = tmp_path / "demo.csv"
    time = np.linspace(0.0, 1.0, 20)
    demo = np.column_stack((time, 0.02 * time, 0.005 * np.sin(np.pi * time), 0.12 + 0.01 * time))
    np.savetxt(demo_path, demo, delimiter=",")
    raw["trajectory"] = {
        "type": "dmp",
        "samples": 32,
        "demo_path": str(demo_path),
        "start_xyz_m": [0.01, 0.0, 0.10],
        "goal_xyz_m": [0.04, 0.0, 0.15],
        "tau": 1.2,
        "basis_count": 10,
    }
    config_path = tmp_path / "dmp_tracking.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_tracking_config(config_path)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    targets = build_target_positions(config, params)

    assert targets.shape == (32, 3)
    assert_allclose(targets[0], [0.01, 0.0, 0.10], atol=1.0e-12)
    assert_allclose(targets[-1], [0.04, 0.0, 0.15], atol=4.0e-3)
