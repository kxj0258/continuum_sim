import importlib
import sys
import types
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from continuum_sim.backends import BackendState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_wiping_board.yaml"


def _load_wiping_runtime_module():
    module = importlib.import_module("continuum_sim.runtime.mujoco_wiping_runtime")
    return importlib.reload(module)


def test_mujoco_wiping_result_exposes_force_and_phase_histories() -> None:
    module = _load_wiping_runtime_module()

    fields = set(module.MujocoWipingResult.__dataclass_fields__)

    assert {
        "target_pose",
        "normal_force_n",
        "contact_proxy_m",
        "force_error_n",
        "phase",
        "waypoint_index",
        "predicted_q",
        "stiffness_diag",
        "scene_xml_path",
    }.issubset(fields)
    assert callable(module.run_mujoco_wiping)


def test_mujoco_wiping_runtime_generates_xml_and_uses_actual_feedback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_wiping_runtime_module()
    task_config_path = _write_task_config(tmp_path, max_steps=2)
    mujoco_config_path = _write_mujoco_config(tmp_path)
    state0 = _make_backend_state(
        tip_position=np.array([0.051, 0.0, 0.105], dtype=float),
        tendon_length=np.zeros(9, dtype=float),
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.0505, -0.001, 0.106], dtype=float),
        tendon_length=np.full(9, 0.001, dtype=float),
    )
    state2 = _make_backend_state(
        tip_position=np.array([0.050, -0.002, 0.107], dtype=float),
        tendon_length=np.full(9, 0.002, dtype=float),
    )
    backend = _FakeWipingBackend([state0, state1, state2])
    override_paths: list[Path] = []

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            del config
            override_paths.append(Path(override_xml_path))
            return backend

    controller_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_controller(
        actual_tip_position,
        actual_tendon_delta,
        target_position,
        surface,
        params,
        physical_tendons,
        motor_params,
        config,
        *,
        measured_normal_force_n=None,
        contact_radius_m=0.0,
        force_control_enabled=True,
    ):
        del surface, params, physical_tendons, motor_params, config, measured_normal_force_n
        actual_tip = np.asarray(actual_tip_position, dtype=float)
        target = np.asarray(target_position, dtype=float)
        controller_calls.append((actual_tip.copy(), np.asarray(actual_tendon_delta).copy()))
        return (
            np.full(9, 0.1, dtype=float),
            {
                "q_est": np.zeros(9, dtype=float),
                "tip_position": actual_tip.copy(),
                "target_position": target.copy(),
                "position_error": target - actual_tip,
                "error_norm": float(np.linalg.norm(target - actual_tip)),
                "normal_force_n": 1.25,
                "contact_proxy_m": -0.002,
                "force_error_n": 0.25,
                "contact_radius_m": float(contact_radius_m),
                "force_control_enabled": bool(force_control_enabled),
                "contact_source": "distance_proxy",
                "in_contact": True,
            },
        )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "compute_wiping_motor_velocity_command_from_observation",
        fake_controller,
    )
    monkeypatch.setattr(
        module,
        "compute_wiping_motor_velocity_command_from_state",
        lambda *args, **kwargs: (_unexpected_command_controller_call()),
    )

    result = module.run_mujoco_wiping(
        task_config_path,
        mujoco_config_path,
        show=False,
    )

    assert override_paths == [tmp_path / "wiping_scene.xml"]
    assert override_paths[0].is_file()
    global_visual = ElementTree.parse(override_paths[0]).getroot().find("./visual/global")
    assert global_visual is not None
    raw_mujoco = yaml.safe_load(mujoco_config_path.read_text(encoding="utf-8"))
    assert global_visual.get("offwidth") == str(raw_mujoco["rendering"]["offscreen_width"])
    assert global_visual.get("offheight") == str(raw_mujoco["rendering"]["offscreen_height"])
    assert result.scene_xml_path == override_paths[0]
    assert len(backend.controls) == 2
    assert len(controller_calls) == 2
    assert_allclose(controller_calls[0][0], state0.tip_pose[:3, 3] + [0.0, 0.0, 0.004])
    assert_allclose(controller_calls[1][0], state1.tip_pose[:3, 3] + [0.0, 0.0, 0.004])
    assert_allclose(controller_calls[0][1], state0.tendon_length)
    assert_allclose(controller_calls[1][1], state1.tendon_length)
    assert result.target_pose.shape == (2, 4, 4)
    assert result.motor_velocity.shape == (2, 9)
    assert_allclose(result.normal_force_n, [1.25, 1.25])
    assert_allclose(result.force_error_n, [0.25, 0.25])
    assert result.phase[0] == "approach"


def test_mujoco_wiping_runtime_updates_live_force_panel_with_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_wiping_runtime_module()
    task_config_path = _write_task_config(tmp_path, max_steps=2)
    mujoco_config_path = _write_mujoco_config(tmp_path)
    state0 = _make_backend_state(
        tip_position=np.array([0.051, 0.0, 0.105], dtype=float),
        tendon_length=np.zeros(9, dtype=float),
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.0505, -0.001, 0.106], dtype=float),
        tendon_length=np.full(9, 0.001, dtype=float),
    )
    state2 = _make_backend_state(
        tip_position=np.array([0.050, -0.002, 0.107], dtype=float),
        tendon_length=np.full(9, 0.002, dtype=float),
    )
    backend = _FakeWipingBackend([state0, state1, state2])
    fake_panel = _FakeForcePanel()

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            del config, override_xml_path
            return backend

    def fake_controller(
        actual_tip_position,
        actual_tendon_delta,
        target_position,
        surface,
        params,
        physical_tendons,
        motor_params,
        config,
        *,
        measured_normal_force_n=None,
        contact_radius_m=0.0,
        force_control_enabled=True,
    ):
        del (
            actual_tip_position,
            actual_tendon_delta,
            target_position,
            surface,
            params,
            physical_tendons,
            motor_params,
            config,
            measured_normal_force_n,
            contact_radius_m,
            force_control_enabled,
        )
        sample_index = len(backend.controls)
        return (
            np.full(9, 0.1, dtype=float),
            {
                "q_est": np.zeros(9, dtype=float),
                "tip_position": np.zeros(3, dtype=float),
                "target_position": np.zeros(3, dtype=float),
                "position_error": np.zeros(3, dtype=float),
                "error_norm": 0.0,
                "normal_force_n": 0.25 + sample_index,
                "contact_proxy_m": -0.001 * (sample_index + 1),
                "force_error_n": 1.25 - sample_index,
                "contact_source": "distance_proxy",
                "in_contact": sample_index == 1,
            },
        )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "compute_wiping_motor_velocity_command_from_observation",
        fake_controller,
    )
    monkeypatch.setattr(
        module,
        "compute_wiping_motor_velocity_command_from_state",
        lambda *args, **kwargs: (_unexpected_command_controller_call()),
    )
    monkeypatch.setattr(module, "_create_tendon_live_panel", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "_create_force_live_panel",
        lambda **kwargs: fake_panel,
    )
    monkeypatch.setattr(module, "_draw_tracking_overlays", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_sync_tracking_viewer", lambda *args, **kwargs: None)
    _install_fake_mujoco_viewer(monkeypatch)

    result = module.run_mujoco_wiping(
        task_config_path,
        mujoco_config_path,
        show=True,
    )

    assert fake_panel.closed is True
    assert fake_panel.flush_count == 2
    assert len(fake_panel.updates) == 2
    assert_allclose(result.normal_force_n, [0.25, 1.25])
    assert_allclose(result.contact_proxy_m, [-0.001, -0.002])
    assert_allclose(result.force_error_n, [1.25, 0.25])
    assert result.in_contact.tolist() == [False, True]
    assert [
        update["normal_force_n"] for update in fake_panel.updates
    ] == pytest.approx(result.normal_force_n)
    assert [
        update["contact_proxy_m"] for update in fake_panel.updates
    ] == pytest.approx(result.contact_proxy_m)
    assert [
        update["force_error_n"] for update in fake_panel.updates
    ] == pytest.approx(result.force_error_n)


def test_mujoco_wiping_runtime_records_follower_contact_projection_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_wiping_runtime_module()
    task_config_path = _write_task_config(tmp_path, max_steps=1)
    mujoco_config_path = _write_mujoco_config(tmp_path)
    raw = yaml.safe_load(mujoco_config_path.read_text(encoding="utf-8"))
    raw["model"]["type"] = "segment_2dof_followers"
    raw["model"]["contact_force_projection"] = True
    raw["model"]["follower_samples_per_segment"] = 4
    mujoco_config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    state0 = _make_backend_state(
        tip_position=np.array([0.051, 0.0, 0.105], dtype=float),
        tendon_length=np.zeros(9, dtype=float),
        qpos=np.zeros(6, dtype=float),
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.0505, -0.001, 0.106], dtype=float),
        tendon_length=np.full(9, 0.001, dtype=float),
        qpos=np.zeros(6, dtype=float),
    )
    backend = _FakeWipingBackend([state0, state1])

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            del config, override_xml_path
            return backend

    def fake_controller(
        actual_tip_position,
        actual_tendon_delta,
        target_position,
        surface,
        params,
        physical_tendons,
        motor_params,
        config,
        *,
        measured_normal_force_n=None,
        contact_radius_m=0.0,
        force_control_enabled=True,
    ):
        del (
            actual_tip_position,
            actual_tendon_delta,
            target_position,
            surface,
            params,
            physical_tendons,
            motor_params,
            config,
            contact_radius_m,
            force_control_enabled,
        )
        assert measured_normal_force_n == pytest.approx(3.0)
        return (
            np.zeros(9, dtype=float),
            {
                "q_est": np.zeros(9, dtype=float),
                "tip_position": np.zeros(3, dtype=float),
                "target_position": np.zeros(3, dtype=float),
                "position_error": np.zeros(3, dtype=float),
                "error_norm": 0.0,
                "normal_force_n": float(measured_normal_force_n),
                "contact_proxy_m": -0.001,
                "force_error_n": -2.0,
                "contact_source": "mujoco_contact_force",
                "in_contact": True,
            },
        )

    projection = types.SimpleNamespace(
        normal_force_n=3.0,
        contact_count=1,
        projected_generalized_force_q=np.zeros(6, dtype=float),
    )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "compute_wiping_motor_velocity_command_from_observation",
        fake_controller,
    )
    monkeypatch.setattr(
        module,
        "_projected_follower_contact_for_wiping",
        lambda *args, **kwargs: projection,
    )

    result = module.run_mujoco_wiping(
        task_config_path,
        mujoco_config_path,
        show=False,
    )

    assert result.qpos.shape == (1, 6)
    assert result.mujoco_control.shape == (1, 9)
    assert result.contact_source == ("mujoco_follower_contact_projection",)


def test_mujoco_wiping_runtime_uses_dynamic_controller_when_selected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_wiping_runtime_module()
    task_config_path = _write_task_config(tmp_path, max_steps=1)
    raw_task = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
    raw_task["controller"]["type"] = "dynamic_adaptive_impedance"
    task_config_path.write_text(yaml.safe_dump(raw_task), encoding="utf-8")
    mujoco_config_path = _write_mujoco_config(tmp_path)
    state0 = _make_backend_state(
        tip_position=np.array([0.051, 0.0, 0.105], dtype=float),
        tendon_length=np.zeros(9, dtype=float),
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.0505, -0.001, 0.106], dtype=float),
        tendon_length=np.full(9, 0.001, dtype=float),
    )
    backend = _FakeWipingBackend([state0, state1])

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            del config, override_xml_path
            return backend

    dynamic_calls: list[np.ndarray] = []

    def fake_dynamic_controller(
        tip_position,
        q_est,
        qdot_est,
        *,
        target_position,
        surface,
        params,
        physical_tendons,
        motor_params,
        wiping_config,
        adaptive_config,
        measured_normal_force_n,
        dt,
        contact_radius_m=0.0,
        force_control_enabled=True,
    ):
        del (
            tip_position,
            qdot_est,
            target_position,
            surface,
            params,
            physical_tendons,
            motor_params,
            wiping_config,
            adaptive_config,
            measured_normal_force_n,
            dt,
            contact_radius_m,
            force_control_enabled,
        )
        dynamic_calls.append(np.asarray(q_est, dtype=float).copy())
        return (
            np.full(9, 0.05, dtype=float),
            {
                "q_est": np.asarray(q_est, dtype=float),
                "tip_position": np.zeros(3, dtype=float),
                "target_position": np.zeros(3, dtype=float),
                "position_error": np.zeros(3, dtype=float),
                "error_norm": 0.0,
                "normal_force_n": 1.0,
                "contact_proxy_m": -0.001,
                "force_error_n": 0.5,
                "contact_source": "distance_proxy",
                "in_contact": True,
                "predicted_q": np.ones(9, dtype=float),
                "predicted_qdot": np.full(9, 0.2, dtype=float),
                "predicted_qddot": np.full(9, 0.3, dtype=float),
                "stiffness_diag": np.full(9, 0.4, dtype=float),
                "damping_diag": np.full(9, 0.5, dtype=float),
            },
        )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "compute_dynamic_wiping_motor_velocity_command_from_state",
        fake_dynamic_controller,
    )
    monkeypatch.setattr(
        module,
        "compute_wiping_motor_velocity_command_from_observation",
        lambda *args, **kwargs: (_unexpected_command_controller_call()),
    )

    result = module.run_mujoco_wiping(task_config_path, mujoco_config_path, show=False)

    assert len(dynamic_calls) == 1
    assert_allclose(result.motor_velocity[0], np.full(9, 0.05))
    assert_allclose(result.predicted_q[0], np.ones(9))
    assert_allclose(result.stiffness_diag[0], np.full(9, 0.4))


def _write_mujoco_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(MUJOCO_CONFIG.read_text(encoding="utf-8"))
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["xml_path"] = str(PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml")
    raw["tendon_xml_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon.xml"
    )
    raw["generated_xml_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_with_visuals.xml"
    )
    raw["tendon_generated_xml_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon_with_visuals.xml"
    )
    raw["visuals"]["directory"] = str(
        PROJECT_ROOT / "assets" / "meshes" / "mujoco_visual_segments"
    )
    raw["visuals"]["template_path"] = str(
        PROJECT_ROOT / "assets" / "mujoco" / "segmented_visuals_template.xml"
    )
    raw["viewer"]["show"] = False
    raw["viewer"]["sync_interval_steps"] = 1
    raw["viewer"]["realtime"] = False
    raw["viewer"]["use_segment_visuals"] = False
    raw["visuals"]["enabled"] = False
    raw["control_mode"] = "tendon_position"
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _write_task_config(tmp_path: Path, *, max_steps: int) -> Path:
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    raw["robot"]["config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["scene"]["config_path"] = str(
        PROJECT_ROOT / "configs" / "scenes" / "wiping_board.yaml"
    )
    raw["scene"]["generated_xml_path"] = str(tmp_path / "wiping_scene.xml")
    raw["simulation"]["max_steps"] = max_steps
    raw["simulation"]["stop_on_completion"] = False
    raw["motion"]["line_count"] = 1
    raw["motion"]["samples_per_line"] = 3
    raw["motion"]["approach_offset_m"] = 0.0
    raw["tool"]["offset_m"] = [0.0, 0.0, 0.004]
    raw["mujoco"]["show_live_tendon_panel"] = False
    raw["mujoco"]["show_live_force_panel"] = True
    raw["mujoco"]["live_force_panel_stride"] = 1
    raw["mujoco"]["live_force_panel_history_points"] = 12
    raw["mujoco"]["feedback_mode"] = "mujoco_actual"
    raw["visualization"]["show"] = False
    config_path = tmp_path / "wiping.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


class _FakeWipingBackend:
    def __init__(self, states: list[BackendState]) -> None:
        self._states = states
        self._index = 0
        self.controls: list[np.ndarray] = []
        self.model = object()
        self.data = types.SimpleNamespace(qpos=np.asarray(states[0].qpos, dtype=float).copy())

    def reset(self) -> BackendState:
        self._index = 0
        self.data.qpos = np.asarray(self._states[0].qpos, dtype=float).copy()
        return self._states[0]

    def step(self, control: np.ndarray, n_substeps: int = 20) -> BackendState:
        del n_substeps
        self.controls.append(np.asarray(control, dtype=float).copy())
        self._index = min(self._index + 1, len(self._states) - 1)
        self.data.qpos = np.asarray(self._states[self._index].qpos, dtype=float).copy()
        return self._states[self._index]


def _make_backend_state(
    *,
    tip_position: np.ndarray,
    tendon_length: np.ndarray,
    qpos: np.ndarray | None = None,
) -> BackendState:
    tip_pose = np.eye(4, dtype=float)
    tip_pose[:3, 3] = np.asarray(tip_position, dtype=float)
    return BackendState(
        time=0.0,
        tip_pose=tip_pose,
        segment_poses=np.repeat(np.eye(4, dtype=float)[None, :, :], 3, axis=0),
        qpos=np.zeros(24, dtype=float) if qpos is None else np.asarray(qpos, dtype=float),
        qvel=np.zeros(24, dtype=float) if qpos is None else np.zeros_like(np.asarray(qpos, dtype=float)),
        tendon_length=np.asarray(tendon_length, dtype=float).copy(),
        tendon_velocity=np.zeros_like(np.asarray(tendon_length, dtype=float)),
        actuator_force=np.zeros(9, dtype=float),
    )


class _FakeForcePanel:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.flush_count = 0
        self.closed = False

    def update(self, **kwargs):
        self.updates.append(dict(kwargs))

    def flush_events(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class _FakeViewer:
    def __init__(self) -> None:
        self.opt = types.SimpleNamespace(geomgroup=np.zeros(6, dtype=int))
        self.cam = types.SimpleNamespace(
            lookat=np.zeros(3, dtype=float),
            distance=0.0,
            azimuth=0.0,
            elevation=0.0,
        )
        self.user_scn = types.SimpleNamespace(ngeom=0, maxgeom=0, geoms=[])

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


class _FakeViewerContext:
    def __init__(self, viewer: _FakeViewer) -> None:
        self.viewer = viewer

    def __enter__(self) -> _FakeViewer:
        return self.viewer

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


def _install_fake_mujoco_viewer(monkeypatch) -> None:
    fake_mujoco = types.ModuleType("mujoco")
    fake_mujoco.__path__ = []
    fake_viewer_module = types.ModuleType("mujoco.viewer")
    fake_viewer_module.launch_passive = (
        lambda model, data, key_callback=None, **kwargs: _FakeViewerContext(
            _FakeViewer()
        )
    )
    fake_mujoco.viewer = fake_viewer_module
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer_module)


def _unexpected_command_controller_call():
    raise AssertionError("Commanded wiping helper should not run in mujoco_actual mode.")
