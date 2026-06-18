from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.scenes import (
    BoxObstaclePrimitive,
    CylinderObstaclePrimitive,
    InteriorShellPrimitive,
    load_navigation_scene_config,
    nearest_clearance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "rocket_nozzle_entry.yaml"


def test_navigation_scene_config_loads_wall_targets_and_primitives() -> None:
    scene = load_navigation_scene_config(SCENE_CONFIG)

    assert scene.name == "rocket_nozzle_entry"
    assert len(scene.primitives) == 5
    assert len(scene.inspection_targets) == 3
    targets = scene.target_positions(("entry_wall_30deg", "rib_gap_center"))

    assert targets.shape == (2, 3)
    assert targets[0, 2] == 0.075
    assert_allclose(targets[1], [0.025, -0.022, 0.095])


def test_clearance_primitives_report_wall_and_obstacle_distances() -> None:
    shell = InteriorShellPrimitive(
        id="shell",
        z_min_m=0.0,
        z_max_m=0.2,
        radius_start_m=0.05,
        radius_end_m=0.05,
    )
    cylinder = CylinderObstaclePrimitive(
        id="post",
        center_m=(0.02, 0.0, 0.08),
        radius_m=0.005,
        half_length_m=0.02,
        axis="z",
    )
    box = BoxObstaclePrimitive(
        id="rib",
        center_m=(0.0, 0.03, 0.08),
        half_size_m=(0.004, 0.004, 0.02),
    )

    wall_query = shell.clearance(np.array([0.045, 0.0, 0.08]))
    assert_allclose(wall_query.distance_m, 0.005)
    assert_allclose(wall_query.normal, [-1.0, 0.0, 0.0])

    cylinder_query = cylinder.clearance(np.array([0.02, 0.0, 0.08]))
    assert cylinder_query.distance_m < 0.0

    nearest = nearest_clearance(
        np.array([0.0, 0.028, 0.08]),
        (shell, cylinder, box),
    )
    assert nearest.source_id == "rib"
