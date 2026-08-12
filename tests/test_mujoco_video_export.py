import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
from numpy.testing import assert_allclose

import continuum_sim.visualization.mujoco_video as mujoco_video
import continuum_sim.runtime.video_hooks as video_hooks
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.video_hooks import MujocoLiveVideoRecorderHook
from continuum_sim.runtime.video_utils import BoundedFrameQueue
from continuum_sim.system.types import (
    BaseSystemState,
    RobotSystemCommand,
    RobotSystemState,
)
from continuum_sim.visualization.mujoco_video import _restore_mocap_state
from continuum_sim.visualization.mujoco_video import _patched_offscreen_xml


def test_patched_offscreen_xml_preserves_asset_paths_and_expands_framebuffer(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "patched"
    mesh_dir = tmp_path / "meshes"
    model_dir.mkdir()
    output_dir.mkdir()
    mesh_dir.mkdir()
    mesh_path = mesh_dir / "segment.stl"
    mesh_path.write_text("solid segment\nendsolid segment\n", encoding="utf-8")
    scene_path = model_dir / "scene.xml"
    scene_path.write_text(
        (
            '<mujoco model="test">'
            '<asset><mesh name="segment" file="../meshes/segment.stl"/></asset>'
            '<visual><global offwidth="640" offheight="480"/></visual>'
            "<worldbody/>"
            "</mujoco>"
        ),
        encoding="utf-8",
    )

    patched_path = _patched_offscreen_xml(
        scene_path,
        width=1280,
        height=720,
        output_dir=output_dir,
    )

    root = ElementTree.parse(patched_path).getroot()
    global_visual = root.find("./visual/global")
    mesh = root.find("./asset/mesh")

    assert global_visual is not None
    assert global_visual.get("offwidth") == "1280"
    assert global_visual.get("offheight") == "720"
    assert mesh is not None
    assert (patched_path.parent / str(mesh.get("file"))).resolve() == mesh_path.resolve()


def test_restore_mocap_state_replays_follower_pose_history() -> None:
    data = type(
        "FakeData",
        (),
        {
            "mocap_pos": np.zeros((2, 3), dtype=float),
            "mocap_quat": np.zeros((2, 4), dtype=float),
        },
    )()
    mocap_pos = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=float,
    )
    mocap_quat = np.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )

    _restore_mocap_state(data, mocap_pos, mocap_quat, index=1)

    assert_allclose(data.mocap_pos, mocap_pos[1])
    assert_allclose(data.mocap_quat, mocap_quat[1])


def test_mujoco_replay_copies_frame_and_reports_render_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text("<mujoco/>", encoding="utf-8")
    rendered = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    written: list[np.ndarray] = []
    camera_values: dict[str, object] = {}

    class FakeRenderer:
        def __init__(self, model, *, height: int, width: int) -> None:
            del model, height, width
            self.render_count = 0

        def update_scene(self, data, camera=-1) -> None:
            del data
            if camera != -1:
                camera_values["lookat"] = tuple(camera.lookat)
                camera_values["distance"] = camera.distance
                camera_values["azimuth"] = camera.azimuth
                camera_values["elevation"] = camera.elevation

        def render(self) -> np.ndarray:
            self.render_count += 1
            if self.render_count == 2:
                raise OSError("boom")
            return rendered

        def close(self) -> None:
            pass

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

        def append_data(self, frame: np.ndarray) -> None:
            written.append(frame)

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjModel = SimpleNamespace(
        from_xml_path=lambda path: SimpleNamespace(
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=16, offheight=16))
        )
    )
    fake_mujoco.MjData = lambda model: SimpleNamespace(
        qpos=np.zeros(1, dtype=float),
        qvel=np.zeros(1, dtype=float),
    )
    fake_mujoco.Renderer = FakeRenderer
    fake_mujoco.mj_forward = lambda model, data: None
    fake_mujoco.MjvCamera = lambda: SimpleNamespace(
        type=None,
        lookat=np.zeros(3, dtype=float),
        distance=0.0,
        azimuth=0.0,
        elevation=0.0,
    )
    fake_mujoco.mjtCamera = SimpleNamespace(mjCAMERA_FREE=0)

    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setattr(
        mujoco_video,
        "_open_video_writer",
        lambda imageio, path, fps: FakeWriter(),
    )

    result = SimpleNamespace(
        scene_xml_path=scene_path,
        qpos=np.zeros((2, 1), dtype=float),
        qvel=np.zeros((2, 1), dtype=float),
    )

    assert (
        mujoco_video.save_mujoco_replay_video(
            result,
            tmp_path / "simulation.gif",
            width=16,
            height=16,
            camera=SimpleNamespace(
                lookat=(0.025, 0.0, 0.095),
                distance=0.20,
                azimuth=315.0,
                elevation=-25.0,
            ),
        )
        is None
    )
    assert len(written) == 1
    assert camera_values == {
        "lookat": (0.025, 0.0, 0.095),
        "distance": 0.20,
        "azimuth": 315.0,
        "elevation": -25.0,
    }
    assert not np.shares_memory(written[0], rendered)
    assert "render failed at frame 1 (sample 1): OSError: boom" in (
        tmp_path / "video_error.txt"
    ).read_text(encoding="utf-8")


def test_mujoco_replay_reports_renderer_creation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text("<mujoco/>", encoding="utf-8")

    class BrokenRenderer:
        def __init__(self, model, *, height: int, width: int) -> None:
            del model, height, width
            raise OSError("renderer boom")

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjModel = SimpleNamespace(
        from_xml_path=lambda path: SimpleNamespace(
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=16, offheight=16))
        )
    )
    fake_mujoco.MjData = lambda model: SimpleNamespace(
        qpos=np.zeros(1, dtype=float),
        qvel=np.zeros(1, dtype=float),
    )
    fake_mujoco.Renderer = BrokenRenderer

    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    result = SimpleNamespace(
        scene_xml_path=scene_path,
        qpos=np.zeros((1, 1), dtype=float),
        qvel=np.zeros((1, 1), dtype=float),
    )

    assert (
        mujoco_video.save_mujoco_replay_video(
            result,
            tmp_path / "simulation.gif",
            width=16,
            height=16,
        )
        is None
    )
    assert "video export failed during creating MuJoCo renderer: OSError: renderer boom" in (
        tmp_path / "video_error.txt"
    ).read_text(encoding="utf-8")


def test_save_replay_video_falls_back_to_matplotlib_when_mujoco_export_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "simulation.gif"
    calls: list[Path] = []

    monkeypatch.setattr(
        mujoco_video,
        "save_mujoco_replay_video",
        lambda *args, **kwargs: (tmp_path / "video_error.txt").write_text(
            "video export failed during creating MuJoCo renderer: OSError: renderer boom\n",
            encoding="utf-8",
        )
        and None,
    )

    def fake_matplotlib_export(result, path, **kwargs):
        del result, kwargs
        calls.append(Path(path))
        Path(path).write_bytes(b"gif")
        return Path(path)

    monkeypatch.setattr(mujoco_video, "save_matplotlib_replay_video", fake_matplotlib_export)

    result = SimpleNamespace(
        scene_xml_path=tmp_path / "scene.xml",
        qpos=np.zeros((1, 1), dtype=float),
        qvel=np.zeros((1, 1), dtype=float),
        target_position=np.zeros((1, 3), dtype=float),
        tip_position=np.zeros((1, 3), dtype=float),
    )

    assert mujoco_video.save_replay_video(result, output_path) == output_path
    assert calls == [output_path]
    assert output_path.is_file()
    assert "fallback_saved: matplotlib trajectory animation" in (
        tmp_path / "video_error.txt"
    ).read_text(encoding="utf-8")


def test_bounded_frame_queue_preserves_order_and_reports_backpressure() -> None:
    queue = BoundedFrameQueue(maxsize=2)

    assert queue.submit(0, "zero") is True
    assert queue.submit(1, "one") is True
    assert queue.submit(2, "two") is False
    assert queue.overload_count == 1
    assert queue.get() == (0, "zero")
    assert queue.get() == (1, "one")


def test_live_video_worker_drains_frames_in_order_and_owns_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writer = _OrderedWriter(tmp_path / "live.gif")
    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjData = lambda model: _DynamicData(0.0)
    fake_mujoco.Renderer = _WorkerRenderer
    fake_mujoco.mj_forward = lambda model, data: None
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setattr(
        video_hooks,
        "_open_video_writer",
        lambda imageio, path, fps: writer,
    )
    monkeypatch.setattr(video_hooks, "_mujoco_render_camera", lambda *args: -1)
    monkeypatch.setattr(video_hooks, "_update_follow_camera", lambda *args: None)
    monkeypatch.setattr(
        video_hooks,
        "_draw_tracking_overlay_scene",
        lambda *args, **kwargs: None,
    )
    backend = SimpleNamespace(
        physics=SimpleNamespace(model=object(), data=_DynamicData(0.0)),
        config=SimpleNamespace(
            viewer=SimpleNamespace(
                camera=None,
                overlays=SimpleNamespace(trail_max_points=8),
            )
        ),
    )
    hook = MujocoLiveVideoRecorderHook(
        backend,
        writer.path,
        stride=1,
        queue_size=4,
        width=4,
        height=3,
    )
    initial = _video_state(0.0)
    hook.on_reset(initial)
    for index in range(3):
        time_s = 0.02 * (index + 1)
        backend.physics.data.time = time_s
        backend.physics.data.qpos[:] = index + 1
        hook.on_step(
            _video_state(time_s),
            RobotSystemCommand.zeros({}),
            index,
        )
    hook.on_finish(_video_state(0.06))

    assert writer.values == [1, 2, 3]
    assert writer.closed is True
    assert hook.frame_count == 3


class _DynamicData:
    def __init__(self, value: float) -> None:
        self.time = value
        self.qpos = np.full(1, value)
        self.qvel = np.zeros(1)
        self.act = np.zeros(0)
        self.ctrl = np.zeros(0)
        self.mocap_pos = np.zeros((0, 3))
        self.mocap_quat = np.zeros((0, 4))
        self.userdata = np.zeros(0)


class _WorkerRenderer:
    def __init__(self, model, *, height: int, width: int) -> None:
        del model
        self.height = height
        self.width = width
        self.value = 0
        self.scene = object()

    def update_scene(self, data, camera=-1) -> None:
        del camera
        self.value = int(data.qpos[0])

    def render(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), self.value, dtype=np.uint8)

    def close(self) -> None:
        pass


class _OrderedWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: list[int] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.values.append(int(frame[0, 0, 0]))

    def close(self) -> None:
        self.closed = True
        self.path.write_bytes(b"video")


def _video_state(time_s: float) -> RobotSystemState:
    return RobotSystemState(
        time_s=time_s,
        base=BaseSystemState(pose=Pose6D.identity()),
        arms={},
    )
