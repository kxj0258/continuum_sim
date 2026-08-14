"""Configuration regression tests for the initial three-segment robot."""

from pathlib import Path

import pytest
import yaml

from continuum_sim import load_mujoco_config, load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_robot_config_describes_three_segment_nine_tendon_arm() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "robot_3seg.yaml")

    robot = config["robot"]
    assert config["units"]["system"] == "SI"
    assert robot["segment_count"] == 3
    assert robot["tendons_per_segment"] == 3
    assert robot["total_tendon_count"] == 9
    assert robot["base_frame"] == "base"
    assert robot["tip_frame"] == "tip"

    segments = config["segments"]
    assert len(segments) == 3

    tendon_indices = []
    for segment in segments:
        assert segment["tendon_angles_deg"] == [0.0, 120.0, 240.0]
        assert len(segment["tendons"]) == 3
        assert segment["length"] > 0.0
        assert segment["tendon_radius"] > 0.0
        tendon_indices.extend(tendon["global_index"] for tendon in segment["tendons"])

    assert tendon_indices == list(range(9))


def test_task_configs_reference_robot_and_backend_configs() -> None:
    tasks_dir = PROJECT_ROOT / "configs" / "tasks"

    for task_path in tasks_dir.glob("*.yaml"):
        task = load_yaml(task_path)
        if "robot_config" in task:
            assert (task_path.parent / task["robot_config"]).resolve().is_file()
        elif "robot" in task:
            assert Path(task["robot"]["config_path"]).resolve().is_file()
        else:
            raise AssertionError(f"{task_path} does not define a robot config reference.")

        if "backend_config" in task:
            assert (task_path.parent / task["backend_config"]).resolve().is_file()


def test_mujoco_config_loads_without_importing_optional_backend() -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    config = load_mujoco_config(PROJECT_ROOT / "configs" / "mujoco.yaml", require_xml=False)

    assert config.robot_config_path == PROJECT_ROOT / "configs" / "robot_3seg.yaml"
    assert config.xml_path == PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm.xml"
    assert (
        config.tendon_xml_path
        == PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_tendon.xml"
    )
    assert (
        config.generated_xml_path
        == PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_with_visuals.xml"
    )
    assert (
        config.tendon_generated_xml_path
        == PROJECT_ROOT
        / "assets"
        / "mujoco"
        / "three_segment_arm_tendon_with_visuals.xml"
    )
    assert config.asset_scale == pytest.approx(0.001)
    assert config.links_per_segment == 4
    assert config.model.type == "distributed_links"
    assert config.model.follower_samples_per_segment == 4
    assert config.model.follower_collision is True
    assert config.model.follower_visuals is True
    assert config.model.contact_force_projection is False
    assert config.model.apply_projected_qfrc is False
    assert config.solver.timestep == pytest.approx(0.001)
    assert config.solver.integrator == "implicitfast"
    assert config.solver.iterations == 50
    assert config.gravity.enabled is raw["gravity"]["enabled"]
    assert config.gravity.vector_m_s2 == pytest.approx(tuple(raw["gravity"]["vector_m_s2"]))
    assert config.control_mode == "tendon_position"
    assert config.joints.hinge.damping == pytest.approx(raw["joints"]["hinge"]["damping"])
    assert config.joints.hinge.armature == pytest.approx(
        raw["joints"]["hinge"]["armature"]
    )
    assert config.joints.hinge.limited is raw["joints"]["hinge"]["limited"]
    assert config.joints.hinge.range_rad == pytest.approx(
        tuple(raw["joints"]["hinge"]["range_rad"])
    )
    assert config.joints.hinge.stiffness == pytest.approx(
        raw["joints"]["hinge"]["stiffness"]
    )
    assert config.joints.hinge.springref == pytest.approx(
        raw["joints"]["hinge"]["springref"]
    )
    assert config.tendon_model.enabled is True
    assert config.tendon_model.type == "fixed"
    assert config.tendon_model.count == 9
    assert config.tendon_model.limited is True
    assert config.tendon_model.length_range_m == pytest.approx((-0.020, 0.020))
    assert config.tendon_model.damping == pytest.approx(0.0)
    assert config.tendon_model.stiffness == pytest.approx(0.0)
    assert config.tendon_model.coefficient_source == "robot_physical_tendons"
    assert config.tendon_model.include_axial_strain is False
    assert config.actuators.tendon_position.kp == pytest.approx(
        raw["actuators"]["tendon_position"]["kp"]
    )
    assert (
        config.actuators.tendon_position.ctrllimited
        is raw["actuators"]["tendon_position"]["ctrllimited"]
    )
    assert config.actuators.tendon_position.ctrlrange_m == pytest.approx(
        tuple(raw["actuators"]["tendon_position"]["ctrlrange_m"])
    )
    assert (
        config.actuators.tendon_position.forcelimited
        is raw["actuators"]["tendon_position"]["forcelimited"]
    )
    assert config.actuators.tendon_position.forcerange_n == pytest.approx(
        tuple(raw["actuators"]["tendon_position"]["forcerange_n"])
    )
    assert config.actuators.joint_position.kp == pytest.approx(
        raw["actuators"]["joint_position"]["kp"]
    )
    assert (
        config.actuators.joint_position.ctrllimited
        is raw["actuators"]["joint_position"]["ctrllimited"]
    )
    assert config.actuators.joint_position.ctrlrange_rad == pytest.approx(
        tuple(raw["actuators"]["joint_position"]["ctrlrange_rad"])
    )
    assert (
        config.actuators.joint_position.forcelimited
        is raw["actuators"]["joint_position"]["forcelimited"]
    )
    assert config.actuators.joint_position.forcerange_nm == pytest.approx(
        tuple(raw["actuators"]["joint_position"]["forcerange_nm"])
    )
    assert config.sensors.tendon_length is True
    assert config.sensors.tendon_velocity is True
    assert config.sensors.actuator_force is True
    assert config.smoke_tests.duration_s == pytest.approx(0.25)
    assert config.smoke_tests.zero_command_tolerance_m == pytest.approx(0.002)
    assert config.smoke_tests.single_tendon_delta_m == pytest.approx(-0.003)
    assert config.smoke_tests.symmetric_tendon_delta_m == pytest.approx(-0.003)
    assert config.rendering.offscreen_width == raw["rendering"]["offscreen_width"]
    assert config.rendering.offscreen_height == raw["rendering"]["offscreen_height"]
    assert config.site_names.base == "base_site"
    assert config.site_names.segments == (
        "segment_1_tip",
        "segment_2_tip",
        "segment_3_tip",
    )
    assert config.site_names.tip == "tip"
    assert config.visuals.enabled is True
    assert config.visuals.frame_mode == "cad_global"
    assert config.visuals.cad_origin_mm == pytest.approx(
        (11.160794, 10.092945, 20.345005)
    )
    assert config.visuals.mesh_unit == "mm"
    assert config.visuals.mesh_scale == pytest.approx(0.001)
    assert (
        config.visuals.directory
        == PROJECT_ROOT / "assets" / "meshes" / "mujoco_visual_segments"
    )
    assert (
        config.visuals.template_path
        == PROJECT_ROOT / "assets" / "mujoco" / "segmented_visuals_template.xml"
    )
    assert config.visuals.collision_mode == "capsule"
    assert config.visuals.visual_geom_group == 1
    assert config.visuals.collision_geom_group == 0
    assert len(config.visuals.expected_meshes) == 13
    assert config.visuals.expected_meshes[0] == "base_visual.stl"
    assert config.visuals.expected_meshes[-1] == "segment_3_link_4_visual.stl"
    assert config.viewer.show is True
    assert config.viewer.steps == 1000
    assert config.viewer.use_segment_visuals is True
    assert config.viewer.show_collision_geoms is raw["viewer"]["show_collision_geoms"]
    assert config.viewer.sync_interval_steps == 1
    assert config.viewer.realtime is True
    assert config.viewer.realtime_factor == pytest.approx(1.0)
    assert config.viewer.show_left_ui is False
    assert config.viewer.show_right_ui is False
    assert config.viewer.camera.lookat == pytest.approx((0.025, 0.0, 0.095))
    assert config.viewer.camera.distance == pytest.approx(0.20)
    assert config.viewer.camera.azimuth == pytest.approx(315.0)
    assert config.viewer.camera.elevation == pytest.approx(-25.0)
    assert config.viewer.camera.follow == "none"
    assert config.viewer.overlays.target_marker is True
    assert config.viewer.overlays.target_marker_radius == pytest.approx(0.004)
    assert config.viewer.overlays.target_marker_rgba == pytest.approx(
        (1.0, 0.12, 0.08, 1.0)
    )
    assert config.viewer.overlays.tip_trail is True
    assert config.viewer.overlays.target_trail is True
    assert config.viewer.overlays.trail_max_points == 250
    assert config.viewer.overlays.trail_stride == 1
    assert config.viewer.overlays.tip_trail_radius == pytest.approx(0.0012)
    assert config.viewer.overlays.target_trail_radius == pytest.approx(0.001)
    assert config.viewer.overlays.tip_trail_rgba == pytest.approx(
        (0.05, 0.65, 1.0, 0.75)
    )
    assert config.viewer.overlays.target_trail_rgba == pytest.approx(
        (1.0, 0.35, 0.08, 0.45)
    )
    assert config.viewer.overlays.tendon_paths is True
    assert config.viewer.overlays.tendon_path_radius == pytest.approx(0.0004)
    assert config.viewer.overlays.tendon_path_stride == 1
    assert config.viewer.overlays.engine_navigation.enabled is False
    assert config.viewer.overlays.engine_navigation.planned_paths is True
    assert config.viewer.overlays.engine_navigation.path_stride == 1
    assert config.viewer.overlays.engine_navigation.base_target_radius == pytest.approx(
        0.007
    )


def test_mujoco_segment_2dof_config_loads_without_importing_optional_backend() -> None:
    config = load_mujoco_config(
        PROJECT_ROOT / "configs" / "mujoco_segment_2dof.yaml",
        require_xml=False,
        require_tendon_xml=False,
        require_visual_meshes=False,
    )

    assert config.model.type == "segment_2dof_followers"
    assert config.model.follower_samples_per_segment == 4
    assert config.model.follower_collision is True
    assert config.model.follower_visuals is True
    assert config.model.contact_force_projection is True
    assert config.model.apply_projected_qfrc is False
    assert config.control_mode == "tendon_position"
    assert config.links_per_segment == 1
    assert config.tendon_xml_path == (
        PROJECT_ROOT / "assets" / "mujoco" / "three_segment_arm_2dof_tendon.xml"
    )
    assert config.tendon_generated_xml_path == (
        PROJECT_ROOT
        / "assets"
        / "mujoco"
        / "three_segment_arm_2dof_tendon_with_visuals.xml"
    )


def test_dual_mujoco_config_loads_engine_navigation_overlays() -> None:
    config = load_mujoco_config(
        PROJECT_ROOT / "configs" / "mujoco_dual.yaml",
        require_xml=False,
        require_visual_meshes=False,
    )

    navigation = config.viewer.overlays.engine_navigation
    assert config.viewer.camera.follow == "base"
    assert config.viewer.camera.distance == pytest.approx(0.50)
    assert navigation.enabled is True
    assert navigation.planned_paths is True
    assert navigation.insertion_waypoints is True
    assert navigation.current_target is True
    assert navigation.base_history is True
    assert navigation.executor_history is True
    assert navigation.target_history is True
    assert navigation.base_path_rgba == pytest.approx((0.2, 0.8, 1.0, 0.65))
    assert navigation.executor_target_radius == pytest.approx(0.005)


def test_dual_mujoco_config_matches_committed_mobile_base_model() -> None:
    config = load_mujoco_config(
        PROJECT_ROOT / "configs" / "mujoco_dual.yaml",
        require_visual_meshes=False,
    )

    assert config.mobile_base_xml_path == (
        PROJECT_ROOT
        / "assets"
        / "mujoco"
        / "dual_three_segment_arm_tendon_with_visuals_mobile_base.xml"
    )
    assert config.model.follower_collision is False
    assert config.tendon_model.limited is False
    assert config.actuators.tendon_position.ctrllimited is False
    assert config.actuators.tendon_position.forcerange_n == pytest.approx(
        (-30.0, 30.0)
    )
    assert config.visuals.world_frame.enabled is True
    assert config.visuals.world_frame.axis_length_m == pytest.approx(0.10)
    assert config.visuals.world_frame.axis_radius_m == pytest.approx(0.0015)
    robot = load_yaml(PROJECT_ROOT / "configs" / "robots" / "dual_arm_3seg.yaml")
    assert (
        robot["dual_robot"]["arms"]["executor"]["actuation"]["limits"]["max_tension"]
        == 30.0
    )
    assert (
        robot["dual_robot"]["arms"]["observer"]["actuation"]["limits"]["max_tension"]
        == 30.0
    )


def test_mujoco_config_rejects_unknown_model_type(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["model"]["type"] = "unsupported_model"
    config_path = tmp_path / "bad_model_type.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="model.type"):
        load_mujoco_config(
            config_path,
            require_xml=False,
            require_visual_meshes=False,
        )


def test_mujoco_config_rejects_invalid_follower_sample_count(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco_segment_2dof.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["model"]["follower_samples_per_segment"] = 0
    config_path = tmp_path / "bad_follower_samples.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="follower_samples_per_segment"):
        load_mujoco_config(
            config_path,
            require_xml=False,
            require_tendon_xml=False,
            require_visual_meshes=False,
        )


def test_mujoco_config_resolves_existing_xml_path(tmp_path: Path) -> None:
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["xml_path"] = "arm.xml"
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_mujoco_config(config_path, require_visual_meshes=False)

    assert config.xml_path == xml_path.resolve()


def test_mujoco_config_rejects_missing_xml_when_required(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["xml_path"] = "missing.xml"
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="MuJoCo XML"):
        load_mujoco_config(config_path)


def test_mujoco_config_rejects_unknown_control_mode(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["control_mode"] = "position"
    config_path = tmp_path / "bad_control_mode.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="control_mode"):
        load_mujoco_config(
            config_path,
            require_xml=False,
            require_visual_meshes=False,
        )


def test_mujoco_config_rejects_tendon_count_mismatch(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["tendon_model"]["count"] = 8
    config_path = tmp_path / "bad_tendon_count.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="tendon_model.count"):
        load_mujoco_config(
            config_path,
            require_xml=False,
            require_visual_meshes=False,
        )


def test_mujoco_config_rejects_invalid_range_and_gain(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["tendon_model"]["length_range_m"] = [0.02, -0.02]
    config_path = tmp_path / "bad_range.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="length_range_m"):
        load_mujoco_config(
            config_path,
            require_xml=False,
            require_visual_meshes=False,
        )

    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["actuators"]["tendon_position"]["kp"] = 0.0
    config_path = tmp_path / "bad_gain.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="kp"):
        load_mujoco_config(
            config_path,
            require_xml=False,
            require_visual_meshes=False,
        )


def test_mujoco_visual_config_rejects_missing_enabled_meshes(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    raw["visuals"]["enabled"] = True
    raw["visuals"]["directory"] = str(tmp_path / "missing_visuals")
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Segmented MuJoCo visual meshes"):
        load_mujoco_config(config_path, require_xml=False)


def test_mujoco_visual_config_accepts_enabled_existing_meshes(tmp_path: Path) -> None:
    raw = load_yaml(PROJECT_ROOT / "configs" / "mujoco.yaml")
    raw["robot_config_path"] = str(PROJECT_ROOT / "configs" / "robot_3seg.yaml")
    visual_dir = tmp_path / "visuals"
    visual_dir.mkdir()
    for mesh_name in raw["visuals"]["expected_meshes"]:
        (visual_dir / mesh_name).write_text("", encoding="utf-8")
    raw["visuals"]["enabled"] = True
    raw["visuals"]["directory"] = str(visual_dir)
    config_path = tmp_path / "mujoco.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_mujoco_config(config_path, require_xml=False)

    assert config.visuals.enabled is True
    assert config.visuals.directory == visual_dir.resolve()
    assert config.visuals.expected_meshes == tuple(raw["visuals"]["expected_meshes"])
