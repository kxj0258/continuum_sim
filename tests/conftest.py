from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CORE_TEST_FILES = frozenset(
    {
        "test_base_pose.py",
        "test_bending_space.py",
        "test_differential_ik.py",
        "test_differential_kinematics.py",
        "test_core_chain.py",
        "test_motor_chain_viewer.py",
        "test_motor_mapping.py",
        "test_pcc_fk.py",
        "test_pcc_to_mujoco.py",
        "test_pcc_viewer.py",
        "test_robot_config.py",
        "test_tendon_coupling.py",
        "test_tendon_mapping.py",
    }
)
CLI_SMOKE_TEST_FILES = frozenset(
    {
        "test_mujoco_build_segment_visuals_script.py",
        "test_mujoco_tracking_runtime.py",
        "test_mujoco_segment_visuals_script.py",
        "test_mujoco_tendon_model_asset.py",
    }
)
BASELINE_TEST_FILES = CORE_TEST_FILES | frozenset(
    {
        "test_mujoco_navigation_runtime.py",
        "test_mujoco_tracking_runtime.py",
        "test_mujoco_tendon_debug_viewer.py",
        "test_mujoco_tendon_model_asset.py",
        "test_mujoco_wiping_runtime.py",
    }
)
SLOW_TEST_FILES = CLI_SMOKE_TEST_FILES | frozenset(
    {
        "test_mujoco_tendon_smoke.py",
    }
)
MUJOCO_RELATED_TEST_FILES = frozenset(
    {
        "test_mujoco_backend.py",
        "test_mujoco_build_segment_visuals_script.py",
        "test_mujoco_model_asset.py",
        "test_mujoco_segment_visuals_script.py",
        "test_mujoco_tendon_debug_viewer.py",
        "test_mujoco_tendon_model_asset.py",
        "test_mujoco_tendon_path_overlay.py",
        "test_mujoco_tendon_smoke.py",
        "test_mujoco_tracking_runtime.py",
    }
)
STABLE_CLI_NODEIDS = frozenset(
    {
        "tests/test_mujoco_segment_visuals_script.py::test_check_segment_visuals_reports_bounds_for_existing_meshes",
    }
)

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def script_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_entry = str(SRC_ROOT)
    entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry] if existing_pythonpath else []
    if src_entry not in entries:
        entries.insert(0, src_entry)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUTF8", "1")
    return env


@pytest.fixture
def run_cli(
    project_root: Path,
    script_env: dict[str, str],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def _run(
        args: Sequence[object],
        *,
        check: bool = True,
        cwd: Path | None = None,
        extra_env: Mapping[str, object] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = script_env.copy()
        if extra_env:
            env.update({key: str(value) for key, value in extra_env.items()})
        return subprocess.run(
            [str(arg) for arg in args],
            cwd=cwd or project_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    return _run


@pytest.fixture
def write_yaml_copy(
    project_root: Path,
) -> Callable[[Path, Path, Callable[[dict[str, Any]], None] | None], Path]:
    def _write(
        template_path: Path,
        destination_path: Path,
        mutate: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Expected mapping YAML in {template_path}, got {type(raw).__name__}.")
        if mutate is not None:
            mutate(raw)
        destination_path.write_text(
            yaml.safe_dump(raw, sort_keys=False),
            encoding="utf-8",
        )
        return destination_path

    return _write


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        basename = Path(str(item.fspath)).name
        if basename in CORE_TEST_FILES:
            item.add_marker(pytest.mark.core)
        if basename in CLI_SMOKE_TEST_FILES:
            item.add_marker(pytest.mark.cli_smoke)
        if item.nodeid in STABLE_CLI_NODEIDS:
            item.add_marker(pytest.mark.stable_cli)
        if basename in BASELINE_TEST_FILES:
            item.add_marker(pytest.mark.baseline)
        if basename in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
        if basename in MUJOCO_RELATED_TEST_FILES or "mujoco" in basename:
            item.add_marker(pytest.mark.mujoco)
