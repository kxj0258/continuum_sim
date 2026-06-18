from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml

import continuum_sim.io.run_artifacts as run_artifacts
from continuum_sim.io.run_artifacts import save_run_artifacts


@dataclass(frozen=True)
class FakeResult:
    time: np.ndarray
    target_position: np.ndarray
    tip_position: np.ndarray
    error_norm: np.ndarray
    motor_position: np.ndarray
    motor_velocity: np.ndarray
    q_est: np.ndarray


@dataclass(frozen=True)
class FakeMujocoResult:
    time: np.ndarray
    target_position: np.ndarray
    tip_position: np.ndarray
    error_norm: np.ndarray
    motor_position: np.ndarray
    motor_velocity: np.ndarray
    q_est: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    scene_xml_path: Path


def test_save_run_artifacts_writes_npz_metadata_configs_and_plots(tmp_path: Path) -> None:
    task_config = tmp_path / "task.yaml"
    robot_config = tmp_path / "robot.yaml"
    robot_config.write_text("schema_version: 1\n", encoding="utf-8")
    task_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "fake_task",
                "robot": {"config_path": str(robot_config)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = FakeResult(
        time=np.array([0.0, 0.02], dtype=float),
        target_position=np.array([[0.0, 0.0, 0.1], [0.01, 0.0, 0.1]], dtype=float),
        tip_position=np.array([[0.0, 0.0, 0.09], [0.008, 0.0, 0.1]], dtype=float),
        error_norm=np.array([0.01, 0.002], dtype=float),
        motor_position=np.zeros((2, 9), dtype=float),
        motor_velocity=np.ones((2, 9), dtype=float),
        q_est=np.zeros((2, 9), dtype=float),
    )

    paths = save_run_artifacts(
        command="run-tracking",
        result=result,
        task_config_path=task_config,
        output_root=tmp_path / "output" / "runs",
        timestamp=datetime(2026, 6, 15, 12, 30, 5),
        save_video=True,
    )

    assert paths.run_dir.name == "fake_task_20260615_123005"
    assert paths.result_npz.is_file()
    assert paths.metadata_json.is_file()
    assert (paths.config_dir / "task_config.yaml").is_file()
    assert (paths.config_dir / "robot_config.yaml").is_file()
    assert (paths.plots_dir / "trajectory.png").is_file()
    assert (paths.plots_dir / "error.png").is_file()
    assert (paths.plots_dir / "motor_velocity.png").is_file()
    assert (
        (paths.videos_dir / "simulation.gif").is_file()
        or (paths.videos_dir / "video_error.txt").is_file()
    )

    with np.load(paths.result_npz) as data:
        assert "time" in data
        assert "target_position" in data
        assert "tip_position" in data
        np.testing.assert_allclose(data["error_norm"], result.error_norm)


def test_save_run_artifacts_uses_mujoco_render_size_for_video_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    task_config = tmp_path / "task.yaml"
    robot_config = project_root / "configs" / "robot_3seg.yaml"
    task_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "fake_task",
                "robot": {"config_path": str(robot_config)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    raw_mujoco = yaml.safe_load((project_root / "configs" / "mujoco.yaml").read_text(encoding="utf-8"))
    raw_mujoco["robot_config_path"] = str(robot_config)
    raw_mujoco["xml_path"] = str(tmp_path / "unused.xml")
    raw_mujoco["tendon_xml_path"] = str(tmp_path / "unused_tendon.xml")
    raw_mujoco["generated_xml_path"] = str(tmp_path / "unused_with_visuals.xml")
    raw_mujoco["tendon_generated_xml_path"] = str(tmp_path / "unused_tendon_with_visuals.xml")
    raw_mujoco["rendering"]["offscreen_width"] = 320
    raw_mujoco["rendering"]["offscreen_height"] = 240
    mujoco_config = tmp_path / "mujoco.yaml"
    mujoco_config.write_text(yaml.safe_dump(raw_mujoco), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_save_replay_video(result, path, **kwargs):
        del result
        captured["path"] = Path(path)
        captured["width"] = kwargs["width"]
        captured["height"] = kwargs["height"]
        captured["camera"] = kwargs["camera"]
        Path(path).write_bytes(b"gif")
        return Path(path)

    monkeypatch.setattr(run_artifacts, "save_replay_video", fake_save_replay_video)

    result = FakeResult(
        time=np.array([0.0, 0.02], dtype=float),
        target_position=np.array([[0.0, 0.0, 0.1], [0.01, 0.0, 0.1]], dtype=float),
        tip_position=np.array([[0.0, 0.0, 0.09], [0.008, 0.0, 0.1]], dtype=float),
        error_norm=np.array([0.01, 0.002], dtype=float),
        motor_position=np.zeros((2, 9), dtype=float),
        motor_velocity=np.ones((2, 9), dtype=float),
        q_est=np.zeros((2, 9), dtype=float),
    )

    save_run_artifacts(
        command="run-tracking",
        result=result,
        task_config_path=task_config,
        mujoco_config_path=mujoco_config,
        output_root=tmp_path / "output" / "runs",
        timestamp=datetime(2026, 6, 15, 12, 30, 5),
        save_video=True,
    )

    assert captured["path"] == (
        tmp_path / "output" / "runs" / "fake_task_20260615_123005" / "videos" / "simulation.gif"
    )
    assert captured["width"] == 320
    assert captured["height"] == 240
    assert captured["camera"].azimuth == 315.0


def test_save_run_artifacts_rebases_copied_mujoco_mesh_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_config = tmp_path / "task.yaml"
    robot_config = tmp_path / "robot.yaml"
    mesh_dir = tmp_path / "meshes"
    model_dir = tmp_path / "generated"
    mesh_dir.mkdir()
    model_dir.mkdir()
    mesh_path = mesh_dir / "segment.stl"
    mesh_path.write_text("solid segment\nendsolid segment\n", encoding="utf-8")
    scene_path = model_dir / "scene.xml"
    scene_path.write_text(
        (
            '<mujoco model="test">'
            '<asset><mesh name="segment" file="../meshes/segment.stl"/></asset>'
            "<worldbody/>"
            "</mujoco>"
        ),
        encoding="utf-8",
    )
    robot_config.write_text("schema_version: 1\n", encoding="utf-8")
    task_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "fake_mujoco_task",
                "robot": {"config_path": str(robot_config)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_artifacts,
        "_save_replay_video_subprocess",
        lambda **kwargs: kwargs["output_path"],
    )
    result = FakeMujocoResult(
        time=np.array([0.0], dtype=float),
        target_position=np.zeros((1, 3), dtype=float),
        tip_position=np.zeros((1, 3), dtype=float),
        error_norm=np.zeros(1, dtype=float),
        motor_position=np.zeros((1, 9), dtype=float),
        motor_velocity=np.zeros((1, 9), dtype=float),
        q_est=np.zeros((1, 9), dtype=float),
        qpos=np.zeros((1, 1), dtype=float),
        qvel=np.zeros((1, 1), dtype=float),
        scene_xml_path=scene_path,
    )

    paths = save_run_artifacts(
        command="run-mujoco-test",
        result=result,
        task_config_path=task_config,
        output_root=tmp_path / "output" / "runs",
        timestamp=datetime(2026, 6, 15, 12, 30, 5),
        save_video=True,
    )

    copied_scene = paths.model_dir / "scene.xml"
    copied_mesh = ElementTree.parse(copied_scene).getroot().find("./asset/mesh")

    assert copied_mesh is not None
    assert (copied_scene.parent / str(copied_mesh.get("file"))).resolve() == mesh_path.resolve()


def test_save_run_artifacts_uses_subprocess_for_mujoco_video_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_config = tmp_path / "task.yaml"
    robot_config = tmp_path / "robot.yaml"
    scene_path = tmp_path / "scene.xml"
    robot_config.write_text("schema_version: 1\n", encoding="utf-8")
    scene_path.write_text("<mujoco/>", encoding="utf-8")
    task_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "fake_mujoco_task",
                "robot": {"config_path": str(robot_config)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_subprocess_export(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"gif")
        return kwargs["output_path"]

    monkeypatch.setattr(run_artifacts, "_save_replay_video_subprocess", fake_subprocess_export)
    result = FakeMujocoResult(
        time=np.array([0.0], dtype=float),
        target_position=np.zeros((1, 3), dtype=float),
        tip_position=np.zeros((1, 3), dtype=float),
        error_norm=np.zeros(1, dtype=float),
        motor_position=np.zeros((1, 9), dtype=float),
        motor_velocity=np.zeros((1, 9), dtype=float),
        q_est=np.zeros((1, 9), dtype=float),
        qpos=np.zeros((1, 1), dtype=float),
        qvel=np.zeros((1, 1), dtype=float),
        scene_xml_path=scene_path,
    )

    paths = save_run_artifacts(
        command="run-mujoco-test",
        result=result,
        task_config_path=task_config,
        output_root=tmp_path / "output" / "runs",
        timestamp=datetime(2026, 6, 15, 12, 30, 5),
        save_video=True,
    )

    assert len(calls) == 1
    assert calls[0]["result_npz_path"] == paths.result_npz
    assert calls[0]["scene_xml_path"] == paths.model_dir / "scene.xml"
    assert calls[0]["output_path"] == paths.videos_dir / "simulation.gif"
    assert calls[0]["camera"] is None
