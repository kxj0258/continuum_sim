from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from continuum_sim.model import ThreeSegmentRobotParams
from continuum_sim.tasks import (
    build_target_positions,
    load_mujoco_tracking_config,
    load_tracking_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_CONFIG = PROJECT_ROOT / "configs" / "tasks" / "pcc_trajectory_tracking.yaml"
MUJOCO_TRACKING_CONFIG = (
    PROJECT_ROOT / "configs" / "tasks" / "mujoco_trajectory_tracking.yaml"
)


def test_load_tracking_config_from_yaml() -> None:
    raw = yaml.safe_load(TRACKING_CONFIG.read_text(encoding="utf-8"))
    config = load_tracking_config(TRACKING_CONFIG)

    assert config.robot_config_path == (PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    assert config.simulation.dt == raw["simulation"]["dt"]
    assert config.simulation.max_steps == raw["simulation"]["max_steps"]
    assert config.simulation.stop_on_completion is raw["simulation"]["stop_on_completion"]
    assert config.simulation.position_limit_rad == raw["simulation"]["position_limit_rad"]
    assert_allclose(
        config.simulation.initial_motor_position_rad,
        np.asarray(raw["simulation"]["initial_motor_position_rad"], dtype=float),
    )
    assert config.controller.damping == raw["controller"]["damping"]
    assert config.controller.position_gain == raw["controller"]["position_gain"]
    assert (
        config.controller.max_motor_velocity_rad_s
        == raw["controller"]["max_motor_velocity_rad_s"]
    )
    assert config.controller.position_tolerance_m == raw["controller"]["position_tolerance_m"]
    assert config.trajectory.type == raw["trajectory"]["type"]
    assert config.trajectory.samples == raw["trajectory"]["samples"]
    assert config.trajectory.radius_m == raw["trajectory"]["radius_m"]
    assert config.trajectory.center_mode == raw["trajectory"]["placement"]["center_mode"]
    assert config.trajectory.z_mode == raw["trajectory"]["placement"]["z_mode"]
    assert config.trajectory.plane == raw["trajectory"]["placement"]["plane"]
    assert config.trajectory.yaw_deg == raw["trajectory"]["placement"]["yaw_deg"]
    assert_allclose(
        config.trajectory.offset_xyz_m,
        np.asarray(raw["trajectory"]["placement"]["offset_xyz_m"], dtype=float),
    )
    assert "mujoco" not in raw
    assert not hasattr(config, "mujoco")
    assert config.visualization.mode == raw["visualization"]["mode"]
    assert config.visualization.show is raw["visualization"]["show"]
    assert (
        config.visualization.show_summary_after_animation
        is raw["visualization"]["show_summary_after_animation"]
    )
    assert (
        config.visualization.animation_interval_ms
        == raw["visualization"]["animation"]["interval_ms"]
    )
    assert config.visualization.animation_stride == raw["visualization"]["animation"]["stride"]
    assert (
        config.visualization.animation_samples_per_segment
        == raw["visualization"]["animation"]["samples_per_segment"]
    )

    assert config.simulation.dt > 0.0
    assert config.simulation.max_steps > 0
    assert config.simulation.initial_motor_position_rad.shape == (9,)
    assert config.controller.max_motor_velocity_rad_s > 0.0
    assert config.trajectory.type in (
        "circle",
        "figure-eight",
        "ellipse",
        "line",
        "square",
        "lissajous",
        "helix",
    )
    assert config.visualization.mode in ("static", "animation")


def test_load_mujoco_tracking_config_from_yaml() -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    config = load_mujoco_tracking_config(MUJOCO_TRACKING_CONFIG)

    assert config.robot_config_path == (PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    assert config.simulation.max_steps == raw["simulation"]["max_steps"]
    assert config.controller.position_gain == raw["controller"]["position_gain"]
    assert config.trajectory.radius_m == raw["trajectory"]["radius_m"]
    assert config.mujoco.target_advance_mode == raw["mujoco"]["target_advance_mode"]
    assert config.mujoco.feedback_mode == raw["mujoco"]["feedback_mode"]
    assert config.mujoco.show_live_tendon_panel is raw["mujoco"]["show_live_tendon_panel"]
    assert (
        config.mujoco.live_tendon_panel_stride
        == raw["mujoco"]["live_tendon_panel_stride"]
    )
    assert (
        config.mujoco.hold_viewer_open_after_run
        is raw["mujoco"]["hold_viewer_open_after_run"]
    )
    assert config.mujoco.show_summary is raw["mujoco"]["show_summary"]
    assert config.visualization.mode == raw["visualization"]["mode"]


def test_build_target_positions_from_tracking_config() -> None:
    config = load_tracking_config(TRACKING_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)

    targets = build_target_positions(config, params)

    assert targets.shape == (config.trajectory.samples, 3)
    scale = max(
        value
        for value in (
            config.trajectory.radius_m,
            config.trajectory.radius_x_m,
            config.trajectory.radius_y_m,
        )
        if value is not None and value > 0.0
    )
    center = np.array([0.0, 0.0, float(np.sum(params.segment_lengths) - scale)])
    yaw = np.deg2rad(config.trajectory.yaw_deg)
    in_plane_u = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
    radius_x = (
        config.trajectory.radius_x_m
        if config.trajectory.radius_x_m is not None
        else config.trajectory.radius_m
    )
    expected_first_target = center + float(radius_x) * in_plane_u
    assert_allclose(targets[0], expected_first_target, atol=1.0e-14)


def test_load_tracking_config_rejects_negative_dt(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["simulation"]["dt"] = -0.02
    bad_path = tmp_path / "bad_dt.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="dt"):
        load_tracking_config(bad_path)


def test_load_tracking_config_rejects_unknown_trajectory_type(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["trajectory"]["type"] = "spiral"
    bad_path = tmp_path / "bad_trajectory.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="type"):
        load_tracking_config(bad_path)


def test_load_tracking_config_rejects_unknown_mujoco_target_advance_mode(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"]["target_advance_mode"] = "distance"
    bad_path = tmp_path / "bad_mujoco_mode.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="target_advance_mode"):
        load_mujoco_tracking_config(bad_path)


def test_load_tracking_config_defaults_feedback_mode_to_mujoco_actual_when_missing(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"].pop("feedback_mode", None)
    config_path = tmp_path / "default_feedback_mode.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_mujoco_tracking_config(config_path)

    assert config.mujoco.feedback_mode == "mujoco_actual"


def test_load_tracking_config_defaults_live_tendon_panel_settings_when_missing(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"].pop("show_live_tendon_panel", None)
    raw["mujoco"].pop("live_tendon_panel_stride", None)
    config_path = tmp_path / "default_live_tendon_panel.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_mujoco_tracking_config(config_path)

    assert config.mujoco.show_live_tendon_panel is True
    assert config.mujoco.live_tendon_panel_stride == 1


def test_load_tracking_config_defaults_hold_viewer_open_after_run_to_false_when_missing(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"].pop("hold_viewer_open_after_run", None)
    config_path = tmp_path / "default_hold_viewer_open_after_run.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_mujoco_tracking_config(config_path)

    assert config.mujoco.hold_viewer_open_after_run is False


def test_load_tracking_config_rejects_unknown_mujoco_feedback_mode(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"]["feedback_mode"] = "hybrid"
    bad_path = tmp_path / "bad_feedback_mode.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="feedback_mode"):
        load_mujoco_tracking_config(bad_path)


def test_load_tracking_config_rejects_invalid_live_tendon_panel_stride(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(MUJOCO_TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["mujoco"]["live_tendon_panel_stride"] = 0
    bad_path = tmp_path / "bad_live_tendon_panel_stride.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="live_tendon_panel_stride"):
        load_mujoco_tracking_config(bad_path)


def test_figure_eight_target_positions_are_supported() -> None:
    config = load_tracking_config(TRACKING_CONFIG)
    config = replace(
        config,
        trajectory=replace(config.trajectory, type="figure-eight", samples=7),
    )
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)

    targets = build_target_positions(config, params)

    assert targets.shape == (7, 3)
    assert np.max(np.abs(targets[:, 0])) > 0.0
    assert np.max(np.abs(targets[:, 1])) > 0.0


@pytest.mark.parametrize(
    ("trajectory_type", "shape_updates"),
    [
        ("ellipse", {"radius_x_m": 0.06, "radius_y_m": 0.02}),
        ("line", {"length_m": 0.08}),
        ("square", {"side_length_m": 0.08}),
        (
            "lissajous",
            {
                "radius_x_m": 0.05,
                "radius_y_m": 0.03,
                "lissajous_frequency_x": 3,
                "lissajous_frequency_y": 2,
                "lissajous_phase_deg": 45.0,
            },
        ),
        ("helix", {"radius_m": 0.04, "pitch_m": 0.06, "turns": 1.5}),
    ],
)
def test_build_target_positions_supports_extended_trajectory_types(
    trajectory_type: str,
    shape_updates: dict[str, float | int],
) -> None:
    config = load_tracking_config(TRACKING_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    trajectory = replace(
        config.trajectory,
        type=trajectory_type,
        samples=32,
        **shape_updates,
    )

    targets = build_target_positions(replace(config, trajectory=trajectory), params)

    assert targets.shape == (32, 3)
    assert np.all(np.isfinite(targets))


def test_build_target_positions_supports_explicit_center_and_plane_rotation() -> None:
    config = load_tracking_config(TRACKING_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    trajectory = replace(
        config.trajectory,
        type="ellipse",
        samples=24,
        center_mode="explicit",
        center_xyz_m=np.array([0.01, -0.02, 0.18], dtype=float),
        z_mode="center",
        plane="yz",
        yaw_deg=90.0,
        radius_m=0.0,
        radius_x_m=0.03,
        radius_y_m=0.01,
    )

    targets = build_target_positions(replace(config, trajectory=trajectory), params)

    assert targets.shape == (24, 3)
    assert np.allclose(np.mean(targets[:, 0]), 0.01, atol=5.0e-4)
    assert np.max(np.abs(targets[:, 2] - 0.18)) > 0.0


def test_load_tracking_config_supports_legacy_top_level_trajectory_fields(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["trajectory"] = {
        "type": "ellipse",
        "samples": 25,
        "radius_m": 0.04,
        "center_mode": "straight_tip_xy",
        "z_mode": "straight_tip_minus_radius",
        "plane": "xy",
        "yaw_deg": 15.0,
        "radius_x_m": 0.05,
        "radius_y_m": 0.02,
    }
    config_path = tmp_path / "legacy_extended_trajectory.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_tracking_config(config_path)

    assert config.trajectory.type == "ellipse"
    assert config.trajectory.plane == "xy"
    assert config.trajectory.yaw_deg == 15.0
    assert config.trajectory.radius_x_m == 0.05
    assert config.trajectory.radius_y_m == 0.02


def test_load_tracking_config_rejects_missing_explicit_center(tmp_path: Path) -> None:
    raw = yaml.safe_load(TRACKING_CONFIG.read_text(encoding="utf-8"))
    raw["trajectory"]["type"] = "circle"
    raw["trajectory"]["placement"]["center_mode"] = "explicit"
    raw["trajectory"]["placement"]["z_mode"] = "center"
    config_path = tmp_path / "missing_explicit_center.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_tracking_config(config_path)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)

    with pytest.raises(ValueError, match="center_xyz_m"):
        build_target_positions(config, params)


def test_helix_target_positions_span_axial_direction() -> None:
    config = load_tracking_config(TRACKING_CONFIG)
    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    trajectory = replace(
        config.trajectory,
        type="helix",
        samples=40,
        plane="xy",
        radius_m=0.03,
        pitch_m=0.08,
        turns=2.0,
    )

    targets = build_target_positions(replace(config, trajectory=trajectory), params)

    assert targets.shape == (40, 3)
    assert float(np.max(targets[:, 2]) - np.min(targets[:, 2])) > 0.05
