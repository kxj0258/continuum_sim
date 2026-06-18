import importlib
from pathlib import Path

import numpy as np
import yaml
from numpy.testing import assert_allclose

from continuum_sim.backends.mujoco_backend import BackendState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
TASK_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "mujoco_navigation_rocket.yaml"


def _load_navigation_runtime_module():
    module = importlib.import_module("continuum_sim.runtime.mujoco_navigation_runtime")
    return importlib.reload(module)


def test_mujoco_navigation_result_exposes_clearance_histories() -> None:
    module = _load_navigation_runtime_module()

    fields = set(module.MujocoNavigationResult.__dataclass_fields__)

    assert {
        "min_clearance_m",
        "clearance_source_id",
        "clearance_point",
        "clearance_normal",
        "waypoint_index",
        "scene_xml_path",
    }.issubset(fields)
    assert callable(module.run_mujoco_navigation)


def test_mujoco_navigation_runtime_uses_scene_xml_and_actual_feedback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_navigation_runtime_module()
    task_config_path = _write_task_config(tmp_path, max_steps=2)
    mujoco_config_path = _write_mujoco_config(tmp_path)
    state0 = _make_backend_state(
        tip_position=np.array([0.0, 0.0, 0.12], dtype=float),
        tendon_length=np.zeros(9, dtype=float),
    )
    state1 = _make_backend_state(
        tip_position=np.array([0.001, 0.0, 0.119], dtype=float),
        tendon_length=np.full(9, 0.002, dtype=float),
    )
    state2 = _make_backend_state(
        tip_position=np.array([0.0015, 0.0, 0.1185], dtype=float),
        tendon_length=np.full(9, 0.003, dtype=float),
    )
    backend = _FakeNavigationBackend([state0, state1, state2])
    override_paths: list[Path] = []

    class BackendFactory:
        @staticmethod
        def from_config(config, *, override_xml_path=None):
            del config
            override_paths.append(Path(override_xml_path))
            return backend

    def fake_controller(
        actual_tip_position,
        actual_tendon_delta,
        target_position,
        params,
        physical_tendons,
        motor_params,
        scene_primitives,
        config,
    ):
        del params, physical_tendons, motor_params, scene_primitives, config
        return (
            np.full(9, 0.1, dtype=float),
            {
                "q_est": np.zeros(9, dtype=float),
                "tip_position": np.asarray(actual_tip_position, dtype=float).copy(),
                "target_position": np.asarray(target_position, dtype=float).copy(),
                "position_error": np.asarray(target_position, dtype=float)
                - np.asarray(actual_tip_position, dtype=float),
                "error_norm": float(
                    np.linalg.norm(
                        np.asarray(target_position, dtype=float)
                        - np.asarray(actual_tip_position, dtype=float)
                    )
                ),
                "min_clearance_m": 0.02,
                "clearance_source_id": "shell",
                "clearance_point": np.array([0.0, 0.0, 0.08], dtype=float),
                "clearance_normal": np.array([1.0, 0.0, 0.0], dtype=float),
            },
        )

    monkeypatch.setattr(module, "MujocoBackend", BackendFactory)
    monkeypatch.setattr(
        module,
        "compute_navigation_motor_velocity_command_from_observation",
        fake_controller,
    )
    monkeypatch.setattr(
        module,
        "compute_navigation_motor_velocity_command",
        lambda *args, **kwargs: (_unexpected_command_controller_call()),
    )

    result = module.run_mujoco_navigation(
        task_config_path,
        mujoco_config_path,
        show=False,
    )

    assert override_paths == [tmp_path / "generated_scene.xml"]
    assert override_paths[0].is_file()
    assert result.scene_xml_path == override_paths[0]
    assert_allclose(result.tendon_delta[0], state0.tendon_length)
    assert_allclose(result.tendon_delta[1], state1.tendon_length)
    assert_allclose(result.min_clearance_m, [0.02, 0.02])
    assert result.clearance_source_id == ("shell", "shell")
    assert len(backend.controls) == 2


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
        PROJECT_ROOT / "configs" / "scenes" / "rocket_nozzle_entry.yaml"
    )
    raw["scene"]["generated_xml_path"] = str(tmp_path / "generated_scene.xml")
    raw["simulation"]["max_steps"] = max_steps
    raw["simulation"]["stop_on_completion"] = False
    raw["mission"]["waypoint_ids"] = ["entry_wall_30deg"]
    raw["mission"]["terminate_on_clearance_violation"] = False
    raw["mujoco"]["show_live_tendon_panel"] = False
    raw["mujoco"]["feedback_mode"] = "mujoco_actual"
    raw["visualization"]["show"] = False
    config_path = tmp_path / "navigation.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


class _FakeNavigationBackend:
    def __init__(self, states: list[BackendState]) -> None:
        self._states = states
        self._index = 0
        self.controls: list[np.ndarray] = []
        self.model = object()
        self.data = object()

    def reset(self) -> BackendState:
        self._index = 0
        return self._states[0]

    def step(self, control: np.ndarray, n_substeps: int = 20) -> BackendState:
        del n_substeps
        self.controls.append(np.asarray(control, dtype=float).copy())
        self._index = min(self._index + 1, len(self._states) - 1)
        return self._states[self._index]


def _make_backend_state(
    *,
    tip_position: np.ndarray,
    tendon_length: np.ndarray,
) -> BackendState:
    tip_pose = np.eye(4, dtype=float)
    tip_pose[:3, 3] = np.asarray(tip_position, dtype=float)
    return BackendState(
        time=0.0,
        tip_pose=tip_pose,
        segment_poses=np.repeat(np.eye(4, dtype=float)[None, :, :], 3, axis=0),
        qpos=np.zeros(24, dtype=float),
        qvel=np.zeros(24, dtype=float),
        tendon_length=np.asarray(tendon_length, dtype=float).copy(),
        tendon_velocity=np.zeros_like(np.asarray(tendon_length, dtype=float)),
        actuator_force=np.zeros(9, dtype=float),
    )


def _unexpected_command_controller_call():
    raise AssertionError("Commanded navigation helper should not run in mujoco_actual mode.")
