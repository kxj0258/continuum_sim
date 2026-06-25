from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.sensing.camera_model import (
    CameraConfig,
    CameraIntrinsicsConfig,
    camera_matrix,
    focal_lengths_px,
    horizontal_fov_rad,
    principal_point_px,
    vertical_fov_rad,
)


def test_camera_intrinsics_compute_fovs_focal_lengths_and_matrix() -> None:
    intrinsics = CameraIntrinsicsConfig(
        width=640,
        height=480,
        fovy_deg=60.0,
        near=0.02,
        far=2.0,
    )

    fy = 0.5 * 480.0 / np.tan(np.deg2rad(60.0) / 2.0)
    fx = fy

    assert vertical_fov_rad(intrinsics) == pytest.approx(np.deg2rad(60.0))
    expected_horizontal_fov = 2.0 * np.arctan(
        (640.0 / 480.0) * np.tan(np.deg2rad(60.0) / 2.0)
    )

    assert horizontal_fov_rad(intrinsics) == pytest.approx(expected_horizontal_fov)
    assert_allclose(focal_lengths_px(intrinsics), (fx, fy))
    assert_allclose(principal_point_px(intrinsics), (319.5, 239.5))
    assert_allclose(
        camera_matrix(intrinsics),
        [
            [fx, 0.0, 319.5],
            [0.0, fy, 239.5],
            [0.0, 0.0, 1.0],
        ],
    )


def test_camera_config_stores_tip_to_camera_pose() -> None:
    camera = CameraConfig(
        name="observer_eye_camera",
        intrinsics=CameraIntrinsicsConfig(width=640, height=480, fovy_deg=60.0, near=0.02, far=2.0),
        tip_to_camera=Pose6D.from_dict(
            {
                "position": [0.0, 0.0, 0.04],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        ),
    )

    assert camera.name == "observer_eye_camera"
    assert_allclose(camera.tip_to_camera.position, [0.0, 0.0, 0.04])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"width": 0, "height": 480, "fovy_deg": 60.0, "near": 0.02, "far": 2.0}, "width"),
        ({"width": 640, "height": 0, "fovy_deg": 60.0, "near": 0.02, "far": 2.0}, "height"),
        ({"width": 640, "height": 480, "fovy_deg": 0.5, "near": 0.02, "far": 2.0}, "fovy_deg"),
        ({"width": 640, "height": 480, "fovy_deg": 179.5, "near": 0.02, "far": 2.0}, "fovy_deg"),
        ({"width": 640, "height": 480, "fovy_deg": 60.0, "near": 0.0, "far": 2.0}, "near"),
        ({"width": 640, "height": 480, "fovy_deg": 60.0, "near": 2.0, "far": 2.0}, "far"),
    ],
)
def test_camera_intrinsics_reject_invalid_values(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CameraIntrinsicsConfig(**kwargs)
