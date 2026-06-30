"""Configuration loading helpers.

The project stores robot, backend, and task descriptions in YAML files.
Keeping YAML access behind this module gives later validation and path
resolution work a single place to grow from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from continuum_sim.config_validation import (
    bool_field as _bool,
    bool_field as _required_bool,
    choice as _choice,
    choice_value as _choice_value,
    float_tuple as _float_tuple,
    geom_group as _geom_group,
    nonnegative_float as _nonnegative_float,
    nonnegative_float_value as _nonnegative_float_value,
    optional_section as _optional_section,
    positive_float as _positive_float,
    positive_float_value as _positive_float_value,
    positive_int as _positive_int,
    positive_int_value as _positive_int_value,
    range_tuple as _range_tuple,
    required as _required,
    resolve_path as _resolve_path,
    rgba_tuple as _rgba_tuple,
    section as _section,
    string_tuple as _string_tuple,
)


DEFAULT_MUJOCO_VISUAL_MESHES: tuple[str, ...] = (
    "base_visual.stl",
    "segment_1_link_1_visual.stl",
    "segment_1_link_2_visual.stl",
    "segment_1_link_3_visual.stl",
    "segment_1_link_4_visual.stl",
    "segment_2_link_1_visual.stl",
    "segment_2_link_2_visual.stl",
    "segment_2_link_3_visual.stl",
    "segment_2_link_4_visual.stl",
    "segment_3_link_1_visual.stl",
    "segment_3_link_2_visual.stl",
    "segment_3_link_3_visual.stl",
    "segment_3_link_4_visual.stl",
)

MUJOCO_TENDON_PATH_ARM_MODES: tuple[str, ...] = (
    "default",
    "executor",
    "observer",
    "both",
    "none",
)

MUJOCO_CONTROL_MODES: tuple[str, ...] = ("position_joint", "tendon_position")
MUJOCO_TENDON_TYPES: tuple[str, ...] = ("spatial",)
MUJOCO_TENDON_COEFFICIENT_SOURCES: tuple[str, ...] = ("robot_physical_tendons",)
MUJOCO_MODEL_TYPES: tuple[str, ...] = (
    "distributed_links",
    "segment_2dof_followers",
    "dual_distributed_links",
)
MUJOCO_POSE_SOURCES: tuple[str, ...] = ("pcc_fk", "mujoco_site")


@dataclass(frozen=True)
class MujocoModelConfig:
    """High-level MuJoCo model topology and follower-contact options."""

    type: str
    follower_samples_per_segment: int
    follower_collision: bool
    follower_visuals: bool
    contact_force_projection: bool
    apply_projected_qfrc: bool
    pose_source: str


@dataclass(frozen=True)
class MujocoSolverConfig:
    """Numerical solver settings for the reduced-order MuJoCo backend."""

    timestep: float
    integrator: str
    iterations: int


@dataclass(frozen=True)
class MujocoGravityConfig:
    """Runtime gravity settings applied to loaded MuJoCo models."""

    enabled: bool
    vector_m_s2: tuple[float, float, float]


@dataclass(frozen=True)
class MujocoSiteNames:
    """MuJoCo site names used to expose backend poses."""

    base: str
    segments: tuple[str, str, str]
    tip: str


@dataclass(frozen=True)
class MujocoVisualConfig:
    """Optional segmented visual mesh settings for the MuJoCo model."""

    enabled: bool
    frame_mode: str
    cad_origin_mm: tuple[float, float, float]
    mesh_unit: str
    mesh_scale: float
    directory: Path
    template_path: Path
    collision_mode: str
    visual_geom_group: int
    collision_geom_group: int
    expected_meshes: tuple[str, ...]


@dataclass(frozen=True)
class MujocoViewerCameraConfig:
    """Default passive-viewer camera framing."""

    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


@dataclass(frozen=True)
class MujocoViewerOverlayConfig:
    """Trajectory viewer overlay marker and trail settings."""

    target_marker: bool
    target_marker_radius: float
    target_marker_rgba: tuple[float, float, float, float]
    tip_trail: bool
    target_trail: bool
    trail_max_points: int
    trail_stride: int
    tip_trail_radius: float
    target_trail_radius: float
    tip_trail_rgba: tuple[float, float, float, float]
    target_trail_rgba: tuple[float, float, float, float]
    tendon_paths: bool
    tendon_path_radius: float
    tendon_path_stride: int
    tendon_path_arms: str


@dataclass(frozen=True)
class MujocoViewerConfig:
    """Shared MuJoCo passive viewer/runtime settings for scripts."""

    show: bool
    steps: int
    use_segment_visuals: bool
    show_collision_geoms: bool
    sync_interval_steps: int
    realtime: bool
    realtime_factor: float
    camera: MujocoViewerCameraConfig
    overlays: MujocoViewerOverlayConfig


@dataclass(frozen=True)
class MujocoJointHingeConfig:
    """Default hinge joint parameters for generated MuJoCo models."""

    damping: float
    armature: float
    limited: bool
    range_rad: tuple[float, float]
    stiffness: float
    springref: float


@dataclass(frozen=True)
class MujocoJointConfig:
    """Joint defaults used by generated MuJoCo variants."""

    hinge: MujocoJointHingeConfig


@dataclass(frozen=True)
class MujocoTendonModelConfig:
    """YAML-backed spatial tendon model parameters."""

    enabled: bool
    type: str
    count: int
    limited: bool
    length_range_m: tuple[float, float]
    damping: float
    stiffness: float
    coefficient_source: str
    include_axial_strain: bool


@dataclass(frozen=True)
class MujocoTendonPositionActuatorConfig:
    """Position actuator settings for tendon length commands."""

    kp: float
    ctrllimited: bool
    ctrlrange_m: tuple[float, float]
    forcelimited: bool
    forcerange_n: tuple[float, float]


@dataclass(frozen=True)
class MujocoJointPositionActuatorConfig:
    """Position actuator settings for the legacy hinge-joint model."""

    kp: float
    ctrllimited: bool
    ctrlrange_rad: tuple[float, float]
    forcelimited: bool
    forcerange_nm: tuple[float, float]


@dataclass(frozen=True)
class MujocoActuatorConfig:
    """Actuator settings for all supported MuJoCo control modes."""

    tendon_position: MujocoTendonPositionActuatorConfig
    joint_position: MujocoJointPositionActuatorConfig


@dataclass(frozen=True)
class MujocoSensorConfig:
    """Sensor toggles for generated MuJoCo tendon models."""

    tendon_length: bool
    tendon_velocity: bool
    actuator_force: bool


@dataclass(frozen=True)
class MujocoSmokeTestConfig:
    """YAML-backed smoke-test timing, tolerances, and command magnitudes."""

    duration_s: float
    zero_command_tolerance_m: float
    single_tendon_delta_m: float
    symmetric_tendon_delta_m: float


@dataclass(frozen=True)
class MujocoRenderingConfig:
    """MuJoCo offscreen rendering settings used by replay video export."""

    offscreen_width: int
    offscreen_height: int


@dataclass(frozen=True)
class MujocoConfig:
    """Validated YAML-backed configuration for the MuJoCo backend."""

    path: Path
    robot_config_path: Path
    mobile_base_config_path: Path | None
    multi_arm_config_path: Path | None
    dual_arm_mesh_config_path: Path | None
    dual_arm_hole_pattern_config_path: Path | None
    xml_path: Path
    tendon_xml_path: Path
    generated_xml_path: Path
    tendon_generated_xml_path: Path
    asset_scale: float
    links_per_segment: int
    model: MujocoModelConfig
    solver: MujocoSolverConfig
    gravity: MujocoGravityConfig
    control_mode: str
    site_names: MujocoSiteNames
    visuals: MujocoVisualConfig
    viewer: MujocoViewerConfig
    joints: MujocoJointConfig
    tendon_model: MujocoTendonModelConfig
    actuators: MujocoActuatorConfig
    sensors: MujocoSensorConfig
    smoke_tests: MujocoSmokeTestConfig
    rendering: MujocoRenderingConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty documents."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def load_mujoco_config(
    path: str | Path,
    *,
    require_xml: bool = True,
    require_tendon_xml: bool = False,
    require_visual_meshes: bool = True,
) -> MujocoConfig:
    """Load and validate the optional reduced-order MuJoCo backend config.

    This function only validates YAML and filesystem paths. It intentionally
    does not import the optional ``mujoco`` package.
    """

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)

    backend = _required(raw, "backend")
    if backend != "mujoco":
        raise ValueError(f"backend must be 'mujoco', got {backend!r}.")

    robot_config_path = _resolve_path(config_path, _required(raw, "robot_config_path"))
    if not robot_config_path.is_file():
        raise FileNotFoundError(f"Robot config file does not exist: {robot_config_path}")
    physical_tendon_count = _load_robot_physical_tendon_count(robot_config_path)
    robot_config_is_dual = _is_dual_arm_robot_config(robot_config_path)
    mobile_base_config_path = _optional_config_path(config_path, raw.get("mobile_base_config_path"))
    multi_arm_config_path = _optional_config_path(config_path, raw.get("multi_arm_config_path"))
    dual_arm_mesh_config_path = _optional_config_path(
        config_path,
        raw.get("dual_arm_mesh_config_path"),
    )
    dual_arm_hole_pattern_config_path = _optional_config_path(
        config_path,
        raw.get("dual_arm_hole_pattern_config_path"),
    )

    xml_path = _resolve_path(config_path, _required(raw, "xml_path"))
    if require_xml and not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML file does not exist: {xml_path}")
    tendon_xml_path = _resolve_path(config_path, _required(raw, "tendon_xml_path"))
    if require_tendon_xml and not tendon_xml_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo tendon XML file does not exist: {tendon_xml_path}"
        )
    generated_xml_path = _resolve_path(
        config_path,
        raw.get(
            "generated_xml_path",
            "../assets/mujoco/three_segment_arm_with_visuals.xml",
        ),
    )
    tendon_generated_xml_path = _resolve_path(
        config_path,
        raw.get(
            "tendon_generated_xml_path",
            "../assets/mujoco/three_segment_arm_tendon_with_visuals.xml",
        ),
    )

    solver = _section(raw, "solver")
    site_names = _section(raw, "site_names")
    segment_sites = _required(site_names, "segments")
    if not isinstance(segment_sites, list | tuple) or len(segment_sites) != 3:
        raise ValueError("site_names.segments must contain exactly 3 site names.")
    visuals = _load_mujoco_visual_config(
        config_path,
        raw,
        require_meshes=require_visual_meshes,
    )
    gravity = _load_mujoco_gravity_config(raw)
    viewer = _load_mujoco_viewer_config(raw)
    model = _load_mujoco_model_config(raw)
    joints = _load_mujoco_joint_config(raw)
    tendon_model = _load_mujoco_tendon_model_config(
        raw,
        physical_tendon_count,
        model_type=model.type,
        robot_config_is_dual=robot_config_is_dual,
    )
    actuators = _load_mujoco_actuator_config(raw)
    sensors = _load_mujoco_sensor_config(raw)
    smoke_tests = _load_mujoco_smoke_test_config(raw)
    rendering = _load_mujoco_rendering_config(raw)

    return MujocoConfig(
        path=config_path,
        robot_config_path=robot_config_path,
        mobile_base_config_path=mobile_base_config_path,
        multi_arm_config_path=multi_arm_config_path,
        dual_arm_mesh_config_path=dual_arm_mesh_config_path,
        dual_arm_hole_pattern_config_path=dual_arm_hole_pattern_config_path,
        xml_path=xml_path,
        tendon_xml_path=tendon_xml_path,
        generated_xml_path=generated_xml_path,
        tendon_generated_xml_path=tendon_generated_xml_path,
        asset_scale=_positive_float(raw, "asset_scale"),
        links_per_segment=_positive_int(raw, "links_per_segment"),
        model=model,
        solver=MujocoSolverConfig(
            timestep=_positive_float(solver, "timestep"),
            integrator=str(_required(solver, "integrator")),
            iterations=_positive_int(solver, "iterations"),
        ),
        gravity=gravity,
        control_mode=_choice(raw, "control_mode", MUJOCO_CONTROL_MODES),
        site_names=MujocoSiteNames(
            base=str(_required(site_names, "base")),
            segments=tuple(str(name) for name in segment_sites),  # type: ignore[arg-type]
            tip=str(_required(site_names, "tip")),
        ),
        visuals=visuals,
        viewer=viewer,
        joints=joints,
        tendon_model=tendon_model,
        actuators=actuators,
        sensors=sensors,
        smoke_tests=smoke_tests,
        rendering=rendering,
    )


def _load_mujoco_model_config(raw: dict[str, Any]) -> MujocoModelConfig:
    values = _optional_section(raw, "model")
    return MujocoModelConfig(
        type=_choice_value(
            values.get("type", "distributed_links"),
            "model.type",
            MUJOCO_MODEL_TYPES,
        ),
        follower_samples_per_segment=_positive_int_value(
            values.get("follower_samples_per_segment", 4),
            "model.follower_samples_per_segment",
        ),
        follower_collision=_bool(values, "follower_collision", default=True),
        follower_visuals=_bool(values, "follower_visuals", default=True),
        contact_force_projection=_bool(
            values,
            "contact_force_projection",
            default=False,
        ),
        apply_projected_qfrc=_bool(values, "apply_projected_qfrc", default=False),
        pose_source=_choice_value(
            values.get("pose_source", "pcc_fk"),
            "model.pose_source",
            MUJOCO_POSE_SOURCES,
        ),
    )


def _load_mujoco_visual_config(
    config_path: Path,
    raw: dict[str, Any],
    *,
    require_meshes: bool,
) -> MujocoVisualConfig:
    values = _optional_section(raw, "visuals")
    enabled = _bool(values, "enabled", default=False)
    frame_mode = _choice_value(
        values.get("frame_mode", "body_local"),
        "visuals.frame_mode",
        ("body_local", "cad_global"),
    )
    cad_origin_mm = _float_tuple(
        values.get("cad_origin_mm", (0.0, 0.0, 0.0)),
        "visuals.cad_origin_mm",
        length=3,
    )
    mesh_unit = _choice_value(
        values.get("mesh_unit", "mm"),
        "visuals.mesh_unit",
        ("mm", "m"),
    )
    mesh_scale = _positive_float_value(
        values.get("mesh_scale", 0.001),
        "visuals.mesh_scale",
    )
    directory = _resolve_path(
        config_path,
        values.get("directory", "../assets/meshes/mujoco_visual_segments"),
    )
    template_path = _resolve_path(
        config_path,
        values.get("template_path", "../assets/mujoco/segmented_visuals_template.xml"),
    )
    collision_mode = _choice_value(
        values.get("collision_mode", "capsule"),
        "visuals.collision_mode",
        ("capsule",),
    )
    visual_geom_group = _geom_group(
        values.get("visual_geom_group", 1),
        "visuals.visual_geom_group",
    )
    collision_geom_group = _geom_group(
        values.get("collision_geom_group", 0),
        "visuals.collision_geom_group",
    )
    if visual_geom_group == collision_geom_group:
        raise ValueError(
            "visuals.visual_geom_group and visuals.collision_geom_group must differ."
        )
    expected_meshes = _string_tuple(
        values.get("expected_meshes", DEFAULT_MUJOCO_VISUAL_MESHES),
        "visuals.expected_meshes",
        allow_empty=False,
    )

    visual_config = MujocoVisualConfig(
        enabled=enabled,
        frame_mode=frame_mode,
        cad_origin_mm=cad_origin_mm,
        mesh_unit=mesh_unit,
        mesh_scale=mesh_scale,
        directory=directory,
        template_path=template_path,
        collision_mode=collision_mode,
        visual_geom_group=visual_geom_group,
        collision_geom_group=collision_geom_group,
        expected_meshes=expected_meshes,
    )
    if enabled and require_meshes:
        _require_mujoco_visual_meshes(visual_config)
    return visual_config


def _load_mujoco_gravity_config(raw: dict[str, Any]) -> MujocoGravityConfig:
    values = _optional_section(raw, "gravity")
    return MujocoGravityConfig(
        enabled=_bool(values, "enabled", default=True),
        vector_m_s2=_float_tuple(
            values.get("vector_m_s2", (0.0, 0.0, -9.81)),
            "gravity.vector_m_s2",
            length=3,
        ),  # type: ignore[arg-type]
    )


def _load_mujoco_viewer_config(raw: dict[str, Any]) -> MujocoViewerConfig:
    values = _optional_section(raw, "viewer")
    return MujocoViewerConfig(
        show=_bool(values, "show", default=True),
        steps=_positive_int_value(values.get("steps", 100), "viewer.steps"),
        use_segment_visuals=_bool(values, "use_segment_visuals", default=False),
        show_collision_geoms=_bool(values, "show_collision_geoms", default=True),
        sync_interval_steps=_positive_int_value(
            values.get("sync_interval_steps", 1),
            "viewer.sync_interval_steps",
        ),
        realtime=_bool(values, "realtime", default=False),
        realtime_factor=_positive_float_value(
            values.get("realtime_factor", 1.0),
            "viewer.realtime_factor",
        ),
        camera=_load_mujoco_viewer_camera_config(values),
        overlays=_load_mujoco_viewer_overlay_config(values),
    )


def _load_mujoco_viewer_camera_config(
    viewer_values: dict[str, Any],
) -> MujocoViewerCameraConfig:
    values = _optional_section(viewer_values, "camera")
    return MujocoViewerCameraConfig(
        lookat=_float_tuple(
            values.get("lookat", (0.0, 0.0, 0.06)),
            "viewer.camera.lookat",
            length=3,
        ),  # type: ignore[arg-type]
        distance=_positive_float_value(
            values.get("distance", 0.25),
            "viewer.camera.distance",
        ),
        azimuth=float(values.get("azimuth", 135.0)),
        elevation=float(values.get("elevation", -25.0)),
    )


def _load_mujoco_viewer_overlay_config(
    viewer_values: dict[str, Any],
) -> MujocoViewerOverlayConfig:
    values = _optional_section(viewer_values, "overlays")
    return MujocoViewerOverlayConfig(
        target_marker=_bool(values, "target_marker", default=True),
        target_marker_radius=_positive_float_value(
            values.get("target_marker_radius", 0.004),
            "viewer.overlays.target_marker_radius",
        ),
        target_marker_rgba=_rgba_tuple(
            values.get("target_marker_rgba", (1.0, 0.12, 0.08, 1.0)),
            "viewer.overlays.target_marker_rgba",
        ),
        tip_trail=_bool(values, "tip_trail", default=True),
        target_trail=_bool(values, "target_trail", default=True),
        trail_max_points=_positive_int_value(
            values.get("trail_max_points", 250),
            "viewer.overlays.trail_max_points",
        ),
        trail_stride=_positive_int_value(
            values.get("trail_stride", 1),
            "viewer.overlays.trail_stride",
        ),
        tip_trail_radius=_positive_float_value(
            values.get("tip_trail_radius", 0.0012),
            "viewer.overlays.tip_trail_radius",
        ),
        target_trail_radius=_positive_float_value(
            values.get("target_trail_radius", 0.001),
            "viewer.overlays.target_trail_radius",
        ),
        tip_trail_rgba=_rgba_tuple(
            values.get("tip_trail_rgba", (0.05, 0.65, 1.0, 0.75)),
            "viewer.overlays.tip_trail_rgba",
        ),
        target_trail_rgba=_rgba_tuple(
            values.get("target_trail_rgba", (1.0, 0.35, 0.08, 0.45)),
            "viewer.overlays.target_trail_rgba",
        ),
        tendon_paths=_bool(values, "tendon_paths", default=False),
        tendon_path_radius=_positive_float_value(
            values.get("tendon_path_radius", 0.0004),
            "viewer.overlays.tendon_path_radius",
        ),
        tendon_path_stride=_positive_int_value(
            values.get("tendon_path_stride", 1),
            "viewer.overlays.tendon_path_stride",
        ),
        tendon_path_arms=_choice_value(
            values.get("tendon_path_arms", "default"),
            "viewer.overlays.tendon_path_arms",
            MUJOCO_TENDON_PATH_ARM_MODES,
        ),
    )


def _load_robot_physical_tendon_count(robot_config_path: Path) -> int:
    if _is_dual_arm_robot_config(robot_config_path):
        from continuum_sim.model.dual_arm_robot import load_dual_arm_robot_config

        return len(load_dual_arm_robot_config(robot_config_path).physical_tendons)
    robot_config = load_yaml(robot_config_path)
    robot_values = _optional_section(robot_config, "robot")
    declared_count = int(robot_values.get("total_tendon_count", 0))
    physical_tendons = robot_config.get("physical_tendons", [])
    if not isinstance(physical_tendons, list | tuple):
        raise ValueError("robot physical_tendons must be a list.")
    physical_tendon_count = len(physical_tendons)
    if declared_count and declared_count != physical_tendon_count:
        raise ValueError(
            "robot.total_tendon_count must match physical_tendons length, got "
            f"{declared_count} and {physical_tendon_count}."
        )
    return physical_tendon_count


def _optional_config_path(config_path: Path, raw_value: object) -> Path | None:
    if raw_value in (None, ""):
        return None
    resolved = _resolve_path(config_path, raw_value)
    if not resolved.is_file():
        raise FileNotFoundError(f"Optional config file does not exist: {resolved}")
    return resolved


def _load_mujoco_joint_config(raw: dict[str, Any]) -> MujocoJointConfig:
    values = _section(raw, "joints")
    hinge = _section(values, "hinge")
    return MujocoJointConfig(
        hinge=MujocoJointHingeConfig(
            damping=_nonnegative_float(hinge, "damping"),
            armature=_nonnegative_float(hinge, "armature"),
            limited=_required_bool(hinge, "limited"),
            range_rad=_range_tuple(_required(hinge, "range_rad"), "joints.hinge.range_rad"),
            stiffness=_nonnegative_float_value(
                hinge.get("stiffness", 0.0),
                "joints.hinge.stiffness",
            ),
            springref=float(hinge.get("springref", 0.0)),
        )
    )


def _load_mujoco_tendon_model_config(
    raw: dict[str, Any],
    physical_tendon_count: int,
    *,
    model_type: str,
    robot_config_is_dual: bool,
) -> MujocoTendonModelConfig:
    values = _section(raw, "tendon_model")
    count = _positive_int(values, "count")
    expected_count = physical_tendon_count
    if model_type == "dual_distributed_links" and not robot_config_is_dual:
        expected_count *= 2
    if count != expected_count:
        raise ValueError(
            "tendon_model.count must match the expected MuJoCo tendon count, got "
            f"{count} and {expected_count}."
        )
    return MujocoTendonModelConfig(
        enabled=_required_bool(values, "enabled"),
        type=_choice(values, "type", MUJOCO_TENDON_TYPES),
        count=count,
        limited=_required_bool(values, "limited"),
        length_range_m=_range_tuple(
            _required(values, "length_range_m"),
            "tendon_model.length_range_m",
        ),
        damping=_nonnegative_float(values, "damping"),
        stiffness=_nonnegative_float(values, "stiffness"),
        coefficient_source=_choice(
            values,
            "coefficient_source",
            MUJOCO_TENDON_COEFFICIENT_SOURCES,
        ),
        include_axial_strain=_required_bool(values, "include_axial_strain"),
    )


def _is_dual_arm_robot_config(robot_config_path: Path) -> bool:
    from continuum_sim.model.dual_arm_robot import is_dual_arm_robot_config

    return is_dual_arm_robot_config(robot_config_path)


def _load_mujoco_actuator_config(raw: dict[str, Any]) -> MujocoActuatorConfig:
    values = _section(raw, "actuators")
    tendon = _section(values, "tendon_position")
    joint = _section(values, "joint_position")
    return MujocoActuatorConfig(
        tendon_position=MujocoTendonPositionActuatorConfig(
            kp=_positive_float(tendon, "kp"),
            ctrllimited=_required_bool(tendon, "ctrllimited"),
            ctrlrange_m=_range_tuple(
                _required(tendon, "ctrlrange_m"),
                "actuators.tendon_position.ctrlrange_m",
            ),
            forcelimited=_required_bool(tendon, "forcelimited"),
            forcerange_n=_range_tuple(
                _required(tendon, "forcerange_n"),
                "actuators.tendon_position.forcerange_n",
            ),
        ),
        joint_position=MujocoJointPositionActuatorConfig(
            kp=_positive_float(joint, "kp"),
            ctrllimited=_required_bool(joint, "ctrllimited"),
            ctrlrange_rad=_range_tuple(
                _required(joint, "ctrlrange_rad"),
                "actuators.joint_position.ctrlrange_rad",
            ),
            forcelimited=_required_bool(joint, "forcelimited"),
            forcerange_nm=_range_tuple(
                _required(joint, "forcerange_nm"),
                "actuators.joint_position.forcerange_nm",
            ),
        ),
    )


def _load_mujoco_sensor_config(raw: dict[str, Any]) -> MujocoSensorConfig:
    values = _section(raw, "sensors")
    return MujocoSensorConfig(
        tendon_length=_required_bool(values, "tendon_length"),
        tendon_velocity=_required_bool(values, "tendon_velocity"),
        actuator_force=_required_bool(values, "actuator_force"),
    )


def _load_mujoco_smoke_test_config(raw: dict[str, Any]) -> MujocoSmokeTestConfig:
    values = _section(raw, "smoke_tests")
    return MujocoSmokeTestConfig(
        duration_s=_positive_float(values, "duration_s"),
        zero_command_tolerance_m=_nonnegative_float(
            values,
            "zero_command_tolerance_m",
        ),
        single_tendon_delta_m=float(_required(values, "single_tendon_delta_m")),
        symmetric_tendon_delta_m=float(_required(values, "symmetric_tendon_delta_m")),
    )


def _load_mujoco_rendering_config(raw: dict[str, Any]) -> MujocoRenderingConfig:
    values = _optional_section(raw, "rendering")
    return MujocoRenderingConfig(
        offscreen_width=_positive_int_value(
            values.get("offscreen_width", 640),
            "rendering.offscreen_width",
        ),
        offscreen_height=_positive_int_value(
            values.get("offscreen_height", 480),
            "rendering.offscreen_height",
        ),
    )


def _require_mujoco_visual_meshes(visuals: MujocoVisualConfig) -> None:
    missing = [
        name
        for name in visuals.expected_meshes
        if not (visuals.directory / name).is_file()
    ]
    if missing:
        preview = ", ".join(missing)
        raise FileNotFoundError(
            "Segmented MuJoCo visual meshes are enabled, but expected files "
            f"are missing from {visuals.directory}: {preview}"
        )
