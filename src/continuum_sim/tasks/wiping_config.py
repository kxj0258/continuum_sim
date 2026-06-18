"""YAML-backed configuration and motion generation for MuJoCo wiping tasks."""

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
    nonnegative_float_value as _nonnegative_float_value,
    nonnegative_int_value as _nonnegative_int_value,
    optional_section as _optional_section,
    position_vector as _position_vector_value,
    positive_float as _positive_float,
    positive_float_value as _positive_float_value,
    positive_int as _positive_int,
    positive_int_value as _positive_int_value,
    required as _required,
    resolve_path as _resolve_config_path,
    rgba_tuple as _rgba_tuple,
    section as _section,
)
from continuum_sim.scenes.contact_surfaces import WorkSurfaceConfig
from continuum_sim.scenes.scene_builder import ToolPadXmlConfig


WIPING_CONTROLLER_TYPES = ("hybrid_force_position", "dynamic_adaptive_impedance")
WIPING_MOTION_TYPES = ("raster_wipe",)
WIPING_TOOL_TYPES = ("spherical_pad", "capsule_pad")
MUJOCO_FEEDBACK_MODES = ("pcc_command", "mujoco_actual")
_MISSING = object()


@dataclass(frozen=True)
class WipingSceneConfig:
    """Scene file and generated XML target for a wiping run."""

    config_path: Path
    generated_xml_path: Path


@dataclass(frozen=True)
class WipingToolConfig:
    """Tip-mounted contact pad settings."""

    type: str
    radius_m: float
    length_m: float
    offset_m: np.ndarray
    rgba: tuple[float, float, float, float]
    geom_name: str
    body_name: str
    contact_site_name: str
    site_radius_m: float
    contype: int
    conaffinity: int

    def to_xml_config(self) -> ToolPadXmlConfig:
        return ToolPadXmlConfig(
            type="sphere" if self.type == "spherical_pad" else "capsule",
            radius_m=self.radius_m,
            length_m=self.length_m,
            offset_m=self.offset_m.copy(),
            rgba=self.rgba,
            geom_name=self.geom_name,
            body_name=self.body_name,
            contact_site_name=self.contact_site_name,
            site_radius_m=self.site_radius_m,
            contype=self.contype,
            conaffinity=self.conaffinity,
        )


@dataclass(frozen=True)
class WipingSimulationConfig:
    """Simulation timing, limits, and initial motor state."""

    dt: float
    max_steps: int
    stop_on_completion: bool
    position_limit_rad: float
    initial_motor_position_rad: np.ndarray


@dataclass(frozen=True)
class WipingControllerConfig:
    """Hybrid tangent-position and normal-force control gains."""

    type: str
    dynamics_config_path: Path | None
    damping: float
    tangent_position_gain: float
    normal_force_gain: float
    normal_position_gain: float
    target_normal_force_n: float
    force_proxy_stiffness_n_m: float
    target_contact_distance_m: float
    max_normal_velocity_m_s: float
    max_tangent_velocity_m_s: float
    max_motor_velocity_rad_s: float
    position_tolerance_m: float
    force_tolerance_n: float
    max_contact_force_n: float
    contact_loss_tolerance_steps: int
    finite_difference_step_rad: float


@dataclass(frozen=True)
class WipingMotionConfig:
    """Raster wiping path settings in a named surface frame."""

    type: str
    surface_id: str
    patch_id: str | None
    center_m: np.ndarray | None
    width_m: float
    height_m: float
    line_count: int
    samples_per_line: int
    approach_offset_m: float
    contact_offset_m: float
    waypoint_tolerance_m: float


@dataclass(frozen=True)
class WipingMujocoConfig:
    """MuJoCo runtime behavior for wiping."""

    feedback_mode: str
    show_live_tendon_panel: bool
    live_tendon_panel_stride: int
    show_live_force_panel: bool
    live_force_panel_stride: int
    live_force_panel_history_points: int
    hold_viewer_open_after_run: bool
    show_summary: bool


@dataclass(frozen=True)
class WipingVisualizationConfig:
    """Minimal visualization switches shared with CLI smoke tests."""

    show: bool


@dataclass(frozen=True)
class MujocoWipingConfig:
    """Complete YAML-backed MuJoCo wiping task."""

    path: Path
    robot_config_path: Path
    scene: WipingSceneConfig
    tool: WipingToolConfig
    simulation: WipingSimulationConfig
    controller: WipingControllerConfig
    motion: WipingMotionConfig
    mujoco: WipingMujocoConfig
    visualization: WipingVisualizationConfig


@dataclass(frozen=True)
class WipingPath:
    """Generated approach/contact waypoints for a wiping task."""

    target_position: np.ndarray
    target_pose: np.ndarray
    phase: tuple[str, ...]
    waypoint_index: np.ndarray


def load_mujoco_wiping_config(path: str | Path) -> MujocoWipingConfig:
    """Load and validate a MuJoCo force-position wiping task config."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    robot = _section(raw, "robot")
    scene = _section(raw, "scene")
    tool = _section(raw, "tool")
    simulation = _section(raw, "simulation")
    controller = _section(raw, "controller")
    motion = _section(raw, "motion")
    mujoco = _optional_section(raw, "mujoco")
    visualization = _optional_section(raw, "visualization")

    robot_config_path = _resolve_config_path(config_path, _required(robot, "config_path"))
    motor_count = len(load_motor_params_from_yaml(robot_config_path))
    controller_type = _choice(
        controller,
        "type",
        WIPING_CONTROLLER_TYPES,
    )
    motion_type = _choice(motion, "type", WIPING_MOTION_TYPES)
    tool_type = _choice(tool, "type", WIPING_TOOL_TYPES)
    radius_m = _positive_float(tool, "radius_m")
    return MujocoWipingConfig(
        path=config_path,
        robot_config_path=robot_config_path,
        scene=WipingSceneConfig(
            config_path=_resolve_config_path(config_path, _required(scene, "config_path")),
            generated_xml_path=_resolve_config_path(
                config_path,
                scene.get("generated_xml_path", "../assets/mujoco/generated/wiping_scene.xml"),
            ),
        ),
        tool=WipingToolConfig(
            type=tool_type,
            radius_m=radius_m,
            length_m=_positive_float_value(
                tool.get("length_m", 2.0 * radius_m),
                "tool.length_m",
            ),
            offset_m=_position_vector_value(
                tool.get("offset_m", (0.0, 0.0, 0.0)),
                "tool.offset_m",
            ),
            rgba=_rgba_tuple(tool.get("rgba", (0.05, 0.15, 0.18, 1.0)), "tool.rgba"),
            geom_name=str(tool.get("geom_name", "tool_contact_pad")),
            body_name=str(tool.get("body_name", "tool_contact_pad_body")),
            contact_site_name=str(tool.get("contact_site_name", "tool_contact_site")),
            site_radius_m=_positive_float_value(
                tool.get("site_radius_m", 0.5 * radius_m),
                "tool.site_radius_m",
            ),
            contype=_nonnegative_int_value(tool.get("contype", 1), "tool.contype"),
            conaffinity=_nonnegative_int_value(
                tool.get("conaffinity", 1),
                "tool.conaffinity",
            ),
        ),
        simulation=WipingSimulationConfig(
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
        controller=WipingControllerConfig(
            type=controller_type,
            dynamics_config_path=_optional_resolved_path(
                config_path,
                controller.get("dynamics_config_path", _MISSING),
            ),
            damping=_nonnegative_float(controller, "damping"),
            tangent_position_gain=_nonnegative_float(controller, "tangent_position_gain"),
            normal_force_gain=_nonnegative_float(controller, "normal_force_gain"),
            normal_position_gain=_nonnegative_float_value(
                controller.get("normal_position_gain", 0.0),
                "controller.normal_position_gain",
            ),
            target_normal_force_n=_nonnegative_float(controller, "target_normal_force_n"),
            force_proxy_stiffness_n_m=_positive_float_value(
                controller.get("force_proxy_stiffness_n_m", 600.0),
                "controller.force_proxy_stiffness_n_m",
            ),
            target_contact_distance_m=float(
                controller.get("target_contact_distance_m", 0.0)
            ),
            max_normal_velocity_m_s=_positive_float(
                controller,
                "max_normal_velocity_m_s",
            ),
            max_tangent_velocity_m_s=_positive_float(
                controller,
                "max_tangent_velocity_m_s",
            ),
            max_motor_velocity_rad_s=_positive_float(
                controller,
                "max_motor_velocity_rad_s",
            ),
            position_tolerance_m=_nonnegative_float(
                controller,
                "position_tolerance_m",
            ),
            force_tolerance_n=_nonnegative_float(controller, "force_tolerance_n"),
            max_contact_force_n=_positive_float(controller, "max_contact_force_n"),
            contact_loss_tolerance_steps=_positive_int_value(
                controller.get("contact_loss_tolerance_steps", 20),
                "controller.contact_loss_tolerance_steps",
            ),
            finite_difference_step_rad=_positive_float(
                controller,
                "finite_difference_step_rad",
            ),
        ),
        motion=WipingMotionConfig(
            type=motion_type,
            surface_id=str(_required(motion, "surface_id")),
            patch_id=(
                None
                if motion.get("patch_id", _MISSING) is _MISSING
                else str(motion["patch_id"])
            ),
            center_m=_optional_position_vector_value(
                motion.get("center_m", _MISSING),
                "motion.center_m",
            ),
            width_m=_positive_float(motion, "width_m"),
            height_m=_positive_float(motion, "height_m"),
            line_count=_positive_int(motion, "line_count"),
            samples_per_line=_positive_int(motion, "samples_per_line"),
            approach_offset_m=_nonnegative_float(motion, "approach_offset_m"),
            contact_offset_m=float(motion.get("contact_offset_m", 0.0)),
            waypoint_tolerance_m=_nonnegative_float_value(
                motion.get("waypoint_tolerance_m", controller.get("position_tolerance_m", 0.0)),
                "motion.waypoint_tolerance_m",
            ),
        ),
        mujoco=WipingMujocoConfig(
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
            show_live_force_panel=_bool_value(
                mujoco.get("show_live_force_panel", False),
                "mujoco.show_live_force_panel",
            ),
            live_force_panel_stride=_positive_int_value(
                mujoco.get("live_force_panel_stride", 1),
                "mujoco.live_force_panel_stride",
            ),
            live_force_panel_history_points=_positive_int_value(
                mujoco.get("live_force_panel_history_points", 300),
                "mujoco.live_force_panel_history_points",
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
        visualization=WipingVisualizationConfig(
            show=_bool_value(visualization.get("show", True), "visualization.show"),
        ),
    )


def build_raster_wiping_path(
    motion: WipingMotionConfig,
    surface: WorkSurfaceConfig,
    *,
    contact_radius_m: float = 0.0,
) -> WipingPath:
    """Generate an approach waypoint followed by a boustrophedon raster path."""

    center = surface.center_m if motion.center_m is None else motion.center_m
    half_width = 0.5 * motion.width_m
    half_height = 0.5 * motion.height_m
    v_offsets = (
        np.array([0.0], dtype=float)
        if motion.line_count == 1
        else np.linspace(-half_height, half_height, motion.line_count)
    )
    positions: list[np.ndarray] = []
    phases: list[str] = []
    waypoint_indices: list[int] = []
    contact_origin = center + (
        max(0.0, float(contact_radius_m)) + motion.contact_offset_m
    ) * surface.normal
    approach = contact_origin + motion.approach_offset_m * surface.normal
    positions.append(approach)
    phases.append("approach")
    waypoint_indices.append(0)

    waypoint_index = 1
    for line_index, v_offset in enumerate(v_offsets):
        u_values = np.linspace(-half_width, half_width, motion.samples_per_line)
        if line_index % 2 == 1:
            u_values = u_values[::-1]
        for u_offset in u_values:
            positions.append(
                contact_origin
                + float(u_offset) * surface.tangent_u
                + float(v_offset) * surface.tangent_v
            )
            phases.append("contact")
            waypoint_indices.append(waypoint_index)
            waypoint_index += 1

    target_position = np.asarray(positions, dtype=float)
    target_pose = np.asarray(
        [surface.target_pose(position) for position in target_position],
        dtype=float,
    )
    return WipingPath(
        target_position=target_position,
        target_pose=target_pose,
        phase=tuple(phases),
        waypoint_index=np.asarray(waypoint_indices, dtype=int),
    )


def _optional_position_vector_value(raw_value: object, name: str) -> np.ndarray | None:
    if raw_value is _MISSING:
        return None
    return _position_vector_value(raw_value, name)


def _optional_resolved_path(config_path: Path, raw_value: object) -> Path | None:
    if raw_value is _MISSING:
        return None
    return _resolve_config_path(config_path, raw_value)
