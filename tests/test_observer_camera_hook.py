from __future__ import annotations

from pathlib import Path
from threading import RLock

import numpy as np

from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.observer_camera_hooks import MujocoObserverCameraFeedbackHook
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


def test_observer_camera_hook_schedules_display_by_simulation_time() -> None:
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
        show_window=True,
        display_interval_s=0.05,
    )
    renderer = _Renderer()
    mujoco = _Mujoco()
    hook._mujoco = mujoco
    hook._renderer = renderer
    hook._show_with_cv2 = lambda frame: True

    for time_s in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):
        hook.enrich_state(_state(time_s=time_s))

    assert renderer.camera_names == [
        "observer_eye_camera",
        "observer_eye_camera",
        "observer_eye_camera",
    ]
    assert mujoco.forward_calls == 3

    hook.enrich_state(_state(time_s=0.0))

    assert len(renderer.camera_names) == 4
    assert mujoco.forward_calls == 4


def test_matplotlib_camera_reuses_single_image_artist_without_flushing() -> None:
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
    )
    axis = _Axis()
    figure = _Figure()
    hook._plt = object()
    hook._axis = axis
    hook._figure = figure

    first = np.zeros((48, 64, 3), dtype=np.uint8)
    second = np.ones((48, 64, 3), dtype=np.uint8)
    hook._show_with_matplotlib(first)
    hook._show_with_matplotlib(second)

    assert axis.imshow_calls == 1
    assert axis.axis_off_calls == 1
    assert axis.image.set_data_calls == 1
    assert np.array_equal(axis.image.data, second)
    assert figure.canvas.draw_idle_calls == 2


def test_camera_snapshot_copies_dynamic_state_under_lock() -> None:
    backend = _Backend()
    source = backend.physics.data
    source.time = 1.25
    source.qpos = np.array([1.0, 2.0])
    source.qvel = np.array([3.0, 4.0])
    hook = MujocoObserverCameraFeedbackHook(
        backend,
        camera_name="observer_eye_camera",
        intrinsics=CameraIntrinsicsConfig(
            width=64,
            height=48,
            fovy_deg=60.0,
            near=0.02,
            far=2.0,
        ),
        data_lock=RLock(),
    )
    hook._render_data = _SnapshotData()

    snapshot = hook._camera_data_snapshot()

    assert snapshot is hook._render_data
    assert snapshot.time == 1.25
    assert np.array_equal(snapshot.qpos, [1.0, 2.0])
    assert np.array_equal(snapshot.qvel, [3.0, 4.0])


def test_render_frame_for_presentation_does_not_present_from_worker() -> None:
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
    hook._mujoco = _Mujoco()
    hook._renderer = _Renderer()
    hook._show_frame = lambda frame: (_ for _ in ()).throw(
        AssertionError("render worker must not call GUI presentation")
    )

    frame = hook.render_frame(_state())

    assert frame.shape == (48, 64, 3)
    assert hook._mujoco.forward_calls == 1


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
    time = 0.0
    qpos = np.zeros(2)
    qvel = np.zeros(2)
    cam_xmat = np.asarray([np.eye(3).reshape(-1)], dtype=float)
    cam_xpos = np.zeros((1, 3), dtype=float)


class _SnapshotData:
    def __init__(self) -> None:
        self.time = 0.0
        self.qpos = np.zeros(2)
        self.qvel = np.zeros(2)


class _Mujoco:
    class mjtObj:
        mjOBJ_CAMERA = object()

    def __init__(self) -> None:
        self.forward_calls = 0

    def mj_forward(self, model, data) -> None:
        del model, data
        self.forward_calls += 1

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


class _Image:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.set_data_calls = 0

    def set_data(self, data: np.ndarray) -> None:
        self.data = data
        self.set_data_calls += 1


class _Axis:
    def __init__(self) -> None:
        self.imshow_calls = 0
        self.axis_off_calls = 0
        self.image: _Image | None = None

    def imshow(self, frame: np.ndarray) -> _Image:
        self.imshow_calls += 1
        self.image = _Image(frame)
        return self.image

    def set_axis_off(self) -> None:
        self.axis_off_calls += 1


class _Canvas:
    def __init__(self) -> None:
        self.draw_idle_calls = 0

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1


class _Figure:
    def __init__(self) -> None:
        self.canvas = _Canvas()


def _state(*, time_s: float = 0.0) -> RobotSystemState:
    return RobotSystemState(
        time_s=time_s,
        base=BaseSystemState(pose=Pose6D.identity()),
        arms={},
    )
