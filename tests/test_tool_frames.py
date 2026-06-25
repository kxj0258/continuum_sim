from __future__ import annotations

from pathlib import Path

from numpy.testing import assert_allclose

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.tools.attachments import load_attachment_config
from continuum_sim.tools.tool_frames import (
    compute_all_attachment_frames,
    compute_attachment_pose,
    compute_camera_pose,
    compute_nozzle_pose,
    compute_tool_tcp_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARBON_TOOL_CONFIG = PROJECT_ROOT / "configs" / "tools" / "carbon_remover.yaml"
CAMERA_AIRGUN_CONFIG = PROJECT_ROOT / "configs" / "tools" / "eye_camera_air_gun.yaml"


def test_compute_tool_tcp_pose_identity_tip_returns_expected_tcp() -> None:
    tool = load_attachment_config(CARBON_TOOL_CONFIG)

    world_tcp = compute_tool_tcp_pose(Pose6D.identity(), tool)

    assert_allclose(world_tcp.position, [0.0, 0.0, 0.065])


def test_compute_tool_tcp_pose_translates_with_world_tip() -> None:
    tool = load_attachment_config(CARBON_TOOL_CONFIG)
    world_tip = Pose6D.from_dict(
        {
            "position": [0.3, -0.2, 0.4],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    world_tcp = compute_tool_tcp_pose(world_tip, tool)

    assert_allclose(world_tcp.position, [0.3, -0.2, 0.465])


def test_compute_camera_and_nozzle_pose_compose_from_tip_frame() -> None:
    camera_airgun = load_attachment_config(CAMERA_AIRGUN_CONFIG)
    world_tip = Pose6D.from_dict(
        {
            "position": [0.1, 0.2, 0.3],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    )

    world_camera = compute_camera_pose(world_tip, camera_airgun)
    world_nozzle = compute_nozzle_pose(world_tip, camera_airgun)

    assert_allclose(world_camera.position, [0.1, 0.2, 0.34])
    assert_allclose(world_nozzle.position, [0.1, 0.215, 0.345])


def test_compute_all_attachment_frames_returns_supported_frames() -> None:
    tool = load_attachment_config(CARBON_TOOL_CONFIG)
    camera_airgun = load_attachment_config(CAMERA_AIRGUN_CONFIG)

    tool_frames = compute_all_attachment_frames(Pose6D.identity(), tool)
    camera_frames = compute_all_attachment_frames(Pose6D.identity(), camera_airgun)

    assert set(tool_frames) == {"attachment", "tcp"}
    assert_allclose(tool_frames["attachment"].position, [0.0, 0.0, 0.02])
    assert_allclose(tool_frames["tcp"].position, [0.0, 0.0, 0.065])
    assert set(camera_frames) == {"attachment", "camera", "nozzle"}
    assert_allclose(camera_frames["attachment"].position, [0.0, 0.0, 0.025])
    assert_allclose(camera_frames["camera"].position, [0.0, 0.0, 0.04])
    assert_allclose(camera_frames["nozzle"].position, [0.0, 0.015, 0.045])


def test_compute_attachment_pose_uses_tip_to_attachment_transform() -> None:
    camera_airgun = load_attachment_config(CAMERA_AIRGUN_CONFIG)

    attachment_pose = compute_attachment_pose(Pose6D.identity(), camera_airgun)

    assert_allclose(attachment_pose.position, [0.0, 0.0, 0.025])
