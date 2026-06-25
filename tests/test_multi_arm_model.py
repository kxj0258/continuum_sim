from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from continuum_sim.model.multi_arm import (
    ArmConfig,
    MultiArmConfig,
    get_arm,
    get_arms_by_role,
    iter_enabled_arms,
    load_multi_arm_config,
    validate_multi_arm_config,
)
from continuum_sim.model.mount_frame import MountFrameConfig
from continuum_sim.model.base_pose import Pose6D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "robots" / "dual_continuum.yaml"


def test_load_dual_continuum_yaml_parses_observer_and_executor() -> None:
    config = load_multi_arm_config(CONFIG_PATH)

    assert config.base_frame == "mobile_base"
    assert config.default_arm == "executor"
    assert set(config.arms) == {"observer", "executor"}

    observer = get_arm(config, "observer")
    executor = get_arm(config, "executor")

    assert observer.role == "observer"
    assert executor.role == "executor"
    assert observer.attachment == "eye_camera_air_gun"
    assert executor.attachment == "carbon_removal_tool"
    assert_allclose(observer.mount.pose.position, [0.0, 0.035, 0.0])
    assert_allclose(executor.mount.pose.position, [0.0, -0.035, 0.0])


def test_iter_enabled_arms_only_returns_enabled_arms() -> None:
    config = MultiArmConfig(
        base_frame="mobile_base",
        default_arm="executor",
        arms={
            "observer": _arm("observer", "observer", enabled=False),
            "executor": _arm("executor", "executor", enabled=True),
        },
        path=None,
    )

    enabled = list(iter_enabled_arms(config))

    assert [arm.name for arm in enabled] == ["executor"]


def test_get_arms_by_role_returns_matching_observer_arm() -> None:
    config = load_multi_arm_config(CONFIG_PATH)

    observers = get_arms_by_role(config, "observer")

    assert [arm.name for arm in observers] == ["observer"]


def test_validate_multi_arm_config_rejects_invalid_role() -> None:
    config = MultiArmConfig(
        base_frame="mobile_base",
        default_arm=None,
        arms={"observer": _arm("observer", "camera")},
        path=None,
        allow_single_arm_mode=True,
    )

    with pytest.raises(ValueError, match="role"):
        validate_multi_arm_config(config)


def test_validate_multi_arm_config_rejects_empty_arms() -> None:
    config = MultiArmConfig(base_frame="mobile_base", default_arm=None, arms={}, path=None)

    with pytest.raises(ValueError, match="arms"):
        validate_multi_arm_config(config)


def test_validate_multi_arm_config_rejects_mount_parent_frame_mismatch() -> None:
    config = MultiArmConfig(
        base_frame="mobile_base",
        default_arm=None,
        arms={"observer": _arm("observer", "observer", parent_frame="wrong_base")},
        path=None,
        allow_single_arm_mode=True,
    )

    with pytest.raises(ValueError, match="parent_frame"):
        validate_multi_arm_config(config)


def test_validate_multi_arm_config_rejects_missing_required_roles_by_default() -> None:
    config = MultiArmConfig(
        base_frame="mobile_base",
        default_arm="observer",
        arms={"observer": _arm("observer", "observer")},
        path=None,
    )

    with pytest.raises(ValueError, match="executor"):
        validate_multi_arm_config(config)


def test_validate_multi_arm_config_allows_explicit_single_arm_mode() -> None:
    config = MultiArmConfig(
        base_frame="mobile_base",
        default_arm="observer",
        arms={"observer": _arm("observer", "observer")},
        path=None,
        allow_single_arm_mode=True,
    )

    validate_multi_arm_config(config)


def test_strict_paths_rejects_missing_robot_config_path() -> None:
    config = MultiArmConfig(
        base_frame="mobile_base",
        default_arm="observer",
        arms={"observer": _arm("observer", "observer", robot_config_path="missing_robot.yaml")},
        path=PROJECT_ROOT / "configs" / "robots" / "single_test.yaml",
        allow_single_arm_mode=True,
    )

    with pytest.raises(FileNotFoundError, match="Robot config"):
        validate_multi_arm_config(config, strict_paths=True)


def _arm(
    name: str,
    role: str,
    *,
    enabled: bool = True,
    parent_frame: str = "mobile_base",
    robot_config_path: str | None = "configs/robot_3seg.yaml",
) -> ArmConfig:
    return ArmConfig(
        name=name,
        role=role,
        robot_config_path=robot_config_path,
        mount=MountFrameConfig(
            name=f"{name}_mount",
            parent_frame=parent_frame,
            child_frame=f"{name}_continuum_base",
            pose=Pose6D(position=np.zeros(3, dtype=float), quat=np.array([1.0, 0.0, 0.0, 0.0])),
        ),
        attachment=None,
        enabled=enabled,
    )
