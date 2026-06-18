import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_CONFIG = PROJECT_ROOT / "configs" / "mujoco.yaml"
SCRIPT = PROJECT_ROOT / "scripts" / "check_mujoco_offscreen_renderer.py"


def test_probe_offscreen_renderer_uses_config_render_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module(SCRIPT, "check_mujoco_offscreen_renderer_for_test")
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")
    config_path = _write_probe_config(tmp_path, xml_path, width=320, height=240)
    rendered = np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeRenderer:
        def __init__(self, model, *, height: int, width: int) -> None:
            del model
            self.height = height
            self.width = width

        def update_scene(self, data) -> None:
            del data

        def render(self) -> np.ndarray:
            return rendered

        def close(self) -> None:
            pass

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjModel = SimpleNamespace(from_xml_path=lambda path: object())
    fake_mujoco.MjData = lambda model: object()
    fake_mujoco.Renderer = FakeRenderer
    fake_mujoco.mj_forward = lambda model, data: None

    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    summary = module.probe_mujoco_offscreen_renderer(config_path=config_path)

    assert summary["xml_path"] == xml_path.resolve()
    assert summary["width"] == 320
    assert summary["height"] == 240
    assert summary["frame_shape"] == (240, 320, 3)


def test_probe_offscreen_renderer_reports_renderer_creation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module(
        SCRIPT,
        "check_mujoco_offscreen_renderer_failure_for_test",
    )
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")
    config_path = _write_probe_config(tmp_path, xml_path, width=320, height=240)

    class BrokenRenderer:
        def __init__(self, model, *, height: int, width: int) -> None:
            del model, height, width
            raise OSError("renderer boom")

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjModel = SimpleNamespace(from_xml_path=lambda path: object())
    fake_mujoco.MjData = lambda model: object()
    fake_mujoco.Renderer = BrokenRenderer

    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    with pytest.raises(RuntimeError, match="creating MuJoCo renderer: OSError: renderer boom"):
        module.probe_mujoco_offscreen_renderer(config_path=config_path)


def _write_probe_config(
    tmp_path: Path,
    xml_path: Path,
    *,
    width: int,
    height: int,
) -> Path:
    raw = yaml.safe_load(MUJOCO_CONFIG.read_text(encoding="utf-8"))
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["xml_path"] = str(xml_path)
    raw["tendon_xml_path"] = str(xml_path)
    raw["generated_xml_path"] = str(xml_path)
    raw["tendon_generated_xml_path"] = str(xml_path)
    raw["rendering"]["offscreen_width"] = width
    raw["rendering"]["offscreen_height"] = height
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
