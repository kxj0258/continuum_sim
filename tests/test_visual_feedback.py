from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.sensing.camera_model import CameraIntrinsicsConfig
from continuum_sim.sensing.visual_feedback import project_roi_to_camera_feedback


def test_project_roi_to_camera_feedback_centers_forward_roi() -> None:
    intrinsics = CameraIntrinsicsConfig(
        width=640,
        height=480,
        fovy_deg=60.0,
        near=0.02,
        far=2.0,
    )

    feedback = project_roi_to_camera_feedback(
        np.array([0.0, 0.0, 0.5], dtype=float),
        Pose6D.identity(),
        intrinsics,
        timestamp_s=1.25,
    )

    assert feedback.target_visible is True
    assert feedback.timestamp_s == 1.25
    assert feedback.depth_m == 0.5
    assert_allclose(feedback.pixel_error_px, [0.0, 0.0], atol=1.0e-12)
    assert_allclose(feedback.normalized_error, [0.0, 0.0], atol=1.0e-12)


def test_project_roi_to_camera_feedback_reports_off_center_error() -> None:
    intrinsics = CameraIntrinsicsConfig(
        width=640,
        height=480,
        fovy_deg=60.0,
        near=0.02,
        far=2.0,
    )

    feedback = project_roi_to_camera_feedback(
        np.array([0.05, 0.0, 0.5], dtype=float),
        Pose6D.identity(),
        intrinsics,
    )

    assert feedback.target_visible is True
    assert feedback.pixel_error_px[0] > 0.0
    assert feedback.normalized_error[0] > 0.0
    assert feedback.normalized_error[1] == 0.0


def test_project_roi_to_camera_feedback_marks_behind_camera_invisible() -> None:
    intrinsics = CameraIntrinsicsConfig(
        width=640,
        height=480,
        fovy_deg=60.0,
        near=0.02,
        far=2.0,
    )

    feedback = project_roi_to_camera_feedback(
        np.array([0.0, 0.0, -0.5], dtype=float),
        Pose6D.identity(),
        intrinsics,
    )

    assert feedback.target_visible is False
    assert feedback.depth_m == -0.5
