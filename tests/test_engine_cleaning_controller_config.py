from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuum_sim.control.engine_cleaning_controller import (
    build_engine_cleaning_gains_from_config,
    load_engine_cleaning_controller_config,
    validate_engine_cleaning_controller_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "control" / "engine_cleaning_controller.yaml"


def test_load_engine_cleaning_controller_config_reads_gains() -> None:
    config = load_engine_cleaning_controller_config(CONFIG_PATH)
    gains = build_engine_cleaning_gains_from_config(config)

    assert config["type"] == "engine_cleaning_task_space"
    assert gains.tangential_position_gain == pytest.approx(1.5)
    assert gains.normal_position_gain == pytest.approx(1.0)
    assert gains.normal_force_gain == pytest.approx(0.02)
    assert gains.approach_position_gain == pytest.approx(1.2)
    assert gains.retreat_position_gain == pytest.approx(1.2)
    assert gains.max_tcp_speed_mps == pytest.approx(0.04)
    assert gains.max_normal_speed_mps == pytest.approx(0.015)
    assert gains.waypoint_tolerance_m == pytest.approx(0.005)
    assert gains.max_contact_force_n == pytest.approx(3.0)
    assert gains.force_deadband_n == pytest.approx(0.05)
    assert gains.min_clearance_m == pytest.approx(0.01)


def test_controller_config_rejects_negative_gain(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"normal_force_gain": -0.1})

    with pytest.raises(ValueError, match="normal_force_gain"):
        build_engine_cleaning_gains_from_config(load_engine_cleaning_controller_config(config_path))


def test_controller_config_rejects_nonpositive_max_speed(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"max_tcp_speed_mps": 0.0})

    with pytest.raises(ValueError, match="max_tcp_speed_mps"):
        build_engine_cleaning_gains_from_config(load_engine_cleaning_controller_config(config_path))


def test_controller_config_rejects_unknown_type(tmp_path: Path) -> None:
    raw = _base_config()
    raw["controller"]["type"] = "unsupported"
    config_path = tmp_path / "controller.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="controller.type"):
        validate_engine_cleaning_controller_config(load_engine_cleaning_controller_config(config_path))


def _write_config(tmp_path: Path, gain_overrides: dict[str, float]) -> Path:
    raw = _base_config()
    raw["controller"]["gains"].update(gain_overrides)
    config_path = tmp_path / "controller.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def _base_config() -> dict[str, object]:
    return {
        "controller": {
            "type": "engine_cleaning_task_space",
            "gains": {
                "tangential_position_gain": 1.5,
                "normal_position_gain": 1.0,
                "normal_force_gain": 0.02,
                "approach_position_gain": 1.2,
                "retreat_position_gain": 1.2,
                "max_tcp_speed_mps": 0.04,
                "max_normal_speed_mps": 0.015,
                "waypoint_tolerance_m": 0.005,
                "max_contact_force_n": 3.0,
                "force_deadband_n": 0.05,
                "min_clearance_m": 0.01,
            },
        }
    }
