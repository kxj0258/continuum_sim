"""YAML-backed configuration for structured MuJoCo navigation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.config import load_yaml
from continuum_sim.config_validation import (
    bool_field as _bool,
    bool_value as _bool_value,
    choice as _choice,
    choice_value as _choice_value,
    motor_vector as _motor_vector,
    nonnegative_float as _nonnegative_float,
    optional_section as _optional_section,
    positive_float as _positive_float,
    positive_int as _positive_int,
    positive_int_value as _positive_int_value,
    required as _required,
    resolve_path as _resolve_config_path,
    section as _section,
    string_tuple as _string_tuple,
)


NAVIGATION_CONTROLLER_TYPES = ("navigation_differential_ik", "navigation_cbf_qp")
NAVIGATION_MISSION_TYPES = ("ordered_inspection",)
MUJOCO_FEEDBACK_MODES = ("pcc_command", "mujoco_actual")


@dataclass(frozen=True)
class NavigationSimulationConfig:
    """Simulation timing, limits, and initial motor state."""

    dt: float
    max_steps: int
    stop_on_completion: bool
    position_limit_rad: float
    initial_motor_position_rad: np.ndarray


@dataclass(frozen=True)
class NavigationControllerConfig:
    """Differential-IK navigation parameters with clearance regulation."""

    type: str
    damping: float
    position_gain: float
    clearance_gain: float
    clearance_min_m: float
    avoidance_influence_m: float
    max_motor_velocity_rad_s: float
    position_tolerance_m: float
    centerline_samples_per_segment: int
    finite_difference_step_rad: float


@dataclass(frozen=True)
class NavigationMissionConfig:
    """Ordered waypoint mission over named scene inspection targets."""

    type: str
    waypoint_ids: tuple[str, ...]
    terminate_on_clearance_violation: bool


@dataclass(frozen=True)
class NavigationMujocoConfig:
    """MuJoCo runtime behavior for structured navigation."""

    feedback_mode: str
    show_live_tendon_panel: bool
    live_tendon_panel_stride: int
    hold_viewer_open_after_run: bool
    show_summary: bool


@dataclass(frozen=True)
class NavigationVisualizationConfig:
    """Minimal visualization switches shared with CLI smoke tests."""

    show: bool


@dataclass(frozen=True)
class MujocoNavigationConfig:
    """Complete YAML-backed MuJoCo navigation task."""

    path: Path
    robot_config_path: Path
    scene_config_path: Path
    generated_scene_xml_path: Path
    simulation: NavigationSimulationConfig
    controller: NavigationControllerConfig
    mission: NavigationMissionConfig
    mujoco: NavigationMujocoConfig
    visualization: NavigationVisualizationConfig


def load_mujoco_navigation_config(path: str | Path) -> MujocoNavigationConfig:
    """Load and validate a MuJoCo structured-navigation task config."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    robot = _section(raw, "robot")
    scene = _section(raw, "scene")
    simulation = _section(raw, "simulation")
    controller = _section(raw, "controller")
    mission = _section(raw, "mission")
    mujoco = _optional_section(raw, "mujoco")
    visualization = _optional_section(raw, "visualization")

    robot_config_path = _resolve_config_path(config_path, _required(robot, "config_path"))
    motor_count = len(load_motor_params_from_yaml(robot_config_path))
    generated_scene_xml_path = _resolve_config_path(
        config_path,
        scene.get("generated_xml_path", "../assets/mujoco/generated/navigation_scene.xml"),
    )
    controller_type = _choice(
        controller,
        "type",
        NAVIGATION_CONTROLLER_TYPES,
    )
    mission_type = _choice(mission, "type", NAVIGATION_MISSION_TYPES)
    waypoint_ids = _string_tuple(_required(mission, "waypoint_ids"), "mission.waypoint_ids")
    if not waypoint_ids:
        raise ValueError("mission.waypoint_ids must contain at least one id.")
    clearance_min = _nonnegative_float(controller, "clearance_min_m")
    influence = _positive_float(controller, "avoidance_influence_m")
    if influence <= clearance_min:
        raise ValueError(
            "controller.avoidance_influence_m must be greater than "
            "controller.clearance_min_m."
        )

    return MujocoNavigationConfig(
        path=config_path,
        robot_config_path=robot_config_path,
        scene_config_path=_resolve_config_path(config_path, _required(scene, "config_path")),
        generated_scene_xml_path=generated_scene_xml_path,
        simulation=NavigationSimulationConfig(
            dt=_positive_float(simulation, "dt"),
            max_steps=_positive_int(simulation, "max_steps"),
            stop_on_completion=_bool(simulation, "stop_on_completion"),
            position_limit_rad=_positive_float(simulation, "position_limit_rad"),
            initial_motor_position_rad=_motor_vector(
                _required(simulation, "initial_motor_position_rad"),
                "simulation.initial_motor_position_rad",
                expected_size=motor_count,
            ),
        ),
        controller=NavigationControllerConfig(
            type=controller_type,
            damping=_nonnegative_float(controller, "damping"),
            position_gain=_nonnegative_float(controller, "position_gain"),
            clearance_gain=_nonnegative_float(controller, "clearance_gain"),
            clearance_min_m=clearance_min,
            avoidance_influence_m=influence,
            max_motor_velocity_rad_s=_positive_float(
                controller,
                "max_motor_velocity_rad_s",
            ),
            position_tolerance_m=_nonnegative_float(
                controller,
                "position_tolerance_m",
            ),
            centerline_samples_per_segment=_positive_int(
                controller,
                "centerline_samples_per_segment",
            ),
            finite_difference_step_rad=_positive_float(
                controller,
                "finite_difference_step_rad",
            ),
        ),
        mission=NavigationMissionConfig(
            type=mission_type,
            waypoint_ids=waypoint_ids,
            terminate_on_clearance_violation=_bool_value(
                mission.get("terminate_on_clearance_violation", True),
                "mission.terminate_on_clearance_violation",
            ),
        ),
        mujoco=NavigationMujocoConfig(
            feedback_mode=_choice_value(
                mujoco.get("feedback_mode", "mujoco_actual"),
                "mujoco.feedback_mode",
                MUJOCO_FEEDBACK_MODES,
            ),
            show_live_tendon_panel=_bool_value(
                mujoco.get("show_live_tendon_panel", True),
                "mujoco.show_live_tendon_panel",
            ),
            live_tendon_panel_stride=_positive_int_value(
                mujoco.get("live_tendon_panel_stride", 1),
                "mujoco.live_tendon_panel_stride",
            ),
            hold_viewer_open_after_run=_bool_value(
                mujoco.get("hold_viewer_open_after_run", False),
                "mujoco.hold_viewer_open_after_run",
            ),
            show_summary=_bool_value(
                mujoco.get("show_summary", False),
                "mujoco.show_summary",
            ),
        ),
        visualization=NavigationVisualizationConfig(
            show=_bool_value(visualization.get("show", True), "visualization.show"),
        ),
    )


