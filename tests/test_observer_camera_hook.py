from __future__ import annotations

from pathlib import Path

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.hooks import MujocoObserverCameraFeedbackHook
from continuum_sim.sensing.camera_model import CameraIntrinsicsConfig
from continuum_sim.sensing.visual_feedback import VisualServoFeedback
from continuum_sim.system.types import (
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)


def test_observer_camera_hook_renders_even_without_roi_target() -> None:
    hook = MujocoObserverCameraFeedbackHook(
        _Backend(),
        camera_name="observer_eye_camera",
        intrinsics=CameraIntrinsicsConfig(
            width=64,
            height=48,
            fovy_deg=60.0,
            near=0.02,
            far=2.0,
        ),
        fallback_target_world=None,
        show_window=True,
        stride=1,
    )
    renderer = _Renderer()
    hook._mujoco = _Mujoco()
    hook._renderer = renderer
    hook._show_with_cv2 = lambda frame: True

    enriched = hook.enrich_state(
        RobotSystemState(
            time_s=0.0,
            base=BaseSystemState(pose=Pose6D.identity()),
            arms={},
        )
    )

    assert renderer.camera_names == ["observer_eye_camera"]
    assert enriched.metadata["visual_servo_camera_name"] == "observer_eye_camera"
    assert "visual_servo_target_visible" not in enriched.metadata


def test_observer_camera_hook_records_rendered_frames_to_own_writers(
    tmp_path: Path,
) -> None:
    hook = MujocoObserverCameraFeedbackHook(
        _Backend(),
        camera_name="observer_eye_camera",
        intrinsics=CameraIntrinsicsConfig(
            width=64,
            height=48,
            fovy_deg=60.0,
            near=0.02,
            far=2.0,
        ),
        fallback_target_world=None,
        show_window=False,
        stride=1,
        video_output_paths=(
            tmp_path / "observer_eye_camera.gif",
            tmp_path / "observer_eye_camera.mp4",
        ),
        video_fps=10,
        video_stride=1,
    )
    gif_writer = _Writer(tmp_path / "observer_eye_camera.gif")
    mp4_writer = _Writer(tmp_path / "observer_eye_camera.mp4")
    hook._mujoco = _Mujoco()
    hook._renderer = _Renderer()
    hook._writers = (
        (gif_writer.path, gif_writer),
        (mp4_writer.path, mp4_writer),
    )
    hook.on_step(
        _state(),
        RobotSystemCommand.zeros({}),
        0,
    )

    hook.enrich_state(_state())
    hook.on_finish(_state())

    assert gif_writer.frame_count == 1
    assert mp4_writer.frame_count == 1
    assert gif_writer.closed is True
    assert mp4_writer.closed is True
    assert hook.frame_count == 1


def test_observer_camera_roi_overlay_marks_projected_roi_pixel() -> None:
    hook = MujocoObserverCameraFeedbackHook(
        _Backend(),
        camera_name="observer_eye_camera",
        intrinsics=CameraIntrinsicsConfig(
            width=64,
            height=48,
            fovy_deg=60.0,
            near=0.02,
            far=2.0,
        ),
        show_window=False,
    )
    hook.last_feedback = VisualServoFeedback(
        target_visible=True,
        pixel_error_px=np.array([0.0, 0.0]),
        normalized_error=np.array([0.0, 0.0]),
        depth_m=0.2,
        target_camera_m=np.array([0.0, 0.0, -0.2]),
        target_world_m=np.zeros(3),
        camera_position_world_m=np.zeros(3),
        camera_quat_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_s=0.0,
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    marked = hook._draw_roi_overlay(frame)

    assert marked[24, 32].tolist() == [255, 220, 0]


class _Backend:
    def __init__(self) -> None:
        self.physics = _Physics()


class _Physics:
    def __init__(self) -> None:
        self.model = object()
        self.data = _Data()


class _Data:
    cam_xmat = np.asarray([np.eye(3).reshape(-1)], dtype=float)
    cam_xpos = np.zeros((1, 3), dtype=float)


class _Mujoco:
    class mjtObj:
        mjOBJ_CAMERA = object()

    def mj_forward(self, model, data) -> None:
        del model, data

    def mj_name2id(self, model, obj_type, name) -> int:
        del model, obj_type, name
        return 0


class _Renderer:
    def __init__(self) -> None:
        self.camera_names: list[str] = []

    def update_scene(self, data, *, camera: str) -> None:
        del data
        self.camera_names.append(camera)

    def render(self) -> np.ndarray:
        return np.zeros((48, 64, 3), dtype=np.uint8)


class _Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frame_count = 0
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        assert frame.shape == (48, 64, 3)
        self.frame_count += 1

    def close(self) -> None:
        self.closed = True
        self.path.write_bytes(b"video")


def _state() -> RobotSystemState:
    return RobotSystemState(
        time_s=0.0,
        base=BaseSystemState(pose=Pose6D.identity()),
        arms={},
    )
