from pathlib import Path
import math

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.model import ThreeSegmentRobotParams, load_physical_tendons_from_yaml
from continuum_sim.visualization.mujoco_tendon_path_overlay import (
    iter_tendon_body_names,
    tendon_path_polyline_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robot_3seg.yaml"


def test_iter_tendon_body_names_covers_three_segment_chain_once() -> None:
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)

    body_names = iter_tendon_body_names(physical_tendons, links_per_segment=4)

    assert len(body_names) == 12
    assert body_names[0] == "segment_1_link_1"
    assert body_names[-1] == "segment_3_link_4"
    assert len(set(body_names)) == len(body_names)


def test_tendon_path_polyline_points_follow_straight_robot_geometry() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    tendon = physical_tendons[6]
    body_poses = _straight_body_poses(params, links_per_segment=4)

    points = tendon_path_polyline_points(
        body_poses,
        params,
        tendon,
        links_per_segment=4,
    )

    assert len(points) == 13
    theta = math.radians(tendon.angle_deg)
    expected_offset = np.array(
        [
            tendon.radial_offset * math.cos(theta),
            tendon.radial_offset * math.sin(theta),
        ],
        dtype=float,
    )
    assert_allclose(points[0], [expected_offset[0], expected_offset[1], 0.0], atol=1.0e-12)
    assert_allclose(points[-1], [expected_offset[0], expected_offset[1], 0.12], atol=1.0e-12)


def test_tendon_path_polyline_points_respect_tendon_angle_offset() -> None:
    params = ThreeSegmentRobotParams.from_yaml(ROBOT_CONFIG)
    physical_tendons = load_physical_tendons_from_yaml(ROBOT_CONFIG)
    tendon = physical_tendons[1]
    body_poses = _straight_body_poses(params, links_per_segment=4)

    points = tendon_path_polyline_points(
        body_poses,
        params,
        tendon,
        links_per_segment=4,
    )

    assert_allclose(points[0], [-0.0025, 0.00433012701892, 0.0], atol=1.0e-12)
    assert_allclose(points[-1], [-0.0025, 0.00433012701892, 0.04], atol=1.0e-12)


def _straight_body_poses(
    params: ThreeSegmentRobotParams,
    *,
    links_per_segment: int,
) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    z_origin = 0.0
    for segment_index, segment in enumerate(params.segments):
        link_length = segment.length / float(links_per_segment)
        for link_index in range(links_per_segment):
            pose = np.eye(4, dtype=float)
            pose[:3, 3] = [0.0, 0.0, z_origin]
            poses[f"segment_{segment_index + 1}_link_{link_index + 1}"] = pose
            z_origin += link_length
    return poses
