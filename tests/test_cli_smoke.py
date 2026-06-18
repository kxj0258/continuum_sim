import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


CLI = Path("cli.py")
PCC_TRACKING_CONFIG = Path("configs/tasks/pcc_trajectory_tracking.yaml")
MUJOCO_TRACKING_CONFIG = Path("configs/tasks/mujoco_trajectory_tracking.yaml")
MUJOCO_NAVIGATION_CONFIG = Path("configs/tasks/mujoco_navigation_rocket.yaml")
MUJOCO_WIPING_CONFIG = Path("configs/tasks/mujoco_wiping_board.yaml")
MUJOCO_CONFIG = Path("configs/mujoco.yaml")


def test_view_pcc_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    config_path = write_yaml_copy(
        project_root / PCC_TRACKING_CONFIG,
        tmp_path / "pcc_view.yaml",
        lambda raw: raw["visualization"].update({"show": False}),
    )

    result = run_cli([sys.executable, project_root / CLI, "view-pcc", "--config", config_path])

    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_view_motor_chain_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    config_path = write_yaml_copy(
        project_root / PCC_TRACKING_CONFIG,
        tmp_path / "motor_chain_view.yaml",
        lambda raw: raw["visualization"].update({"show": False}),
    )

    result = run_cli(
        [sys.executable, project_root / CLI, "view-motor-chain", "--config", config_path]
    )

    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_run_tracking_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    def mutate(raw: dict) -> None:
        raw["simulation"]["max_steps"] = 6
        raw["simulation"]["stop_on_completion"] = False
        raw["trajectory"]["samples"] = 6
        raw["trajectory"]["radius_m"] = 0.001
        raw["visualization"]["mode"] = "static"
        raw["visualization"]["show"] = False
        raw["visualization"]["show_summary_after_animation"] = False
        raw["visualization"]["animation"]["samples_per_segment"] = 5

    config_path = write_yaml_copy(
        project_root / PCC_TRACKING_CONFIG,
        tmp_path / "pcc_tracking.yaml",
        mutate,
    )

    result = run_cli(
        [sys.executable, project_root / CLI, "run-tracking", "--config", config_path]
    )

    assert "final_error_m:" in result.stdout
    assert "mean_error_m:" in result.stdout
    assert "max_error_m:" in result.stdout
    assert not (tmp_path / "output" / "runs").exists()


def test_run_tracking_save_run_creates_artifacts(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    def mutate(raw: dict) -> None:
        raw["name"] = "cli_save_smoke"
        raw["robot"]["config_path"] = str(project_root / "configs" / "robot_3seg.yaml")
        raw["simulation"]["max_steps"] = 4
        raw["simulation"]["stop_on_completion"] = False
        raw["trajectory"]["samples"] = 4
        raw["trajectory"]["radius_m"] = 0.001
        raw["visualization"]["mode"] = "static"
        raw["visualization"]["show"] = False
        raw["visualization"]["show_summary_after_animation"] = False
        raw["visualization"]["animation"]["samples_per_segment"] = 5

    config_path = write_yaml_copy(
        project_root / PCC_TRACKING_CONFIG,
        tmp_path / "pcc_tracking_save.yaml",
        mutate,
    )

    result = run_cli(
        [
            sys.executable,
            project_root / CLI,
            "run-tracking",
            "--config",
            config_path,
            "--save-run",
        ],
        cwd=tmp_path,
    )

    assert "run_dir:" in result.stdout
    run_dir_line = next(line for line in result.stdout.splitlines() if line.startswith("run_dir:"))
    run_dir = Path(run_dir_line.split(":", 1)[1].strip())
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "output" / "runs"
    assert (run_dir / "result.npz").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "configs" / "task_config.yaml").is_file()
    assert (run_dir / "plots" / "trajectory.png").is_file()
    assert (run_dir / "plots" / "error.png").is_file()
    assert (
        (run_dir / "videos" / "simulation.gif").is_file()
        or (run_dir / "videos" / "video_error.txt").is_file()
    )


def test_view_mujoco_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    def mutate(raw: dict) -> None:
        _use_project_mujoco_paths(raw, project_root)
        raw["viewer"]["show"] = False
        raw["viewer"]["steps"] = 2
        raw["viewer"]["use_segment_visuals"] = False
        raw["visuals"]["enabled"] = False

    config_path = write_yaml_copy(
        project_root / MUJOCO_CONFIG,
        tmp_path / "mujoco.yaml",
        mutate,
    )

    result = run_cli(
        [sys.executable, project_root / CLI, "view-mujoco", "--config", config_path]
    )

    assert "view-mujoco skipped" in result.stdout or "tip_position_m:" in result.stdout


def test_debug_mujoco_tendons_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    def mutate(raw: dict) -> None:
        _mutate_mujoco_tracking_backend(raw, project_root)
        raw["viewer"]["show"] = False
        raw["viewer"]["steps"] = 1

    config_path = write_yaml_copy(
        project_root / MUJOCO_CONFIG,
        tmp_path / "mujoco_debug.yaml",
        mutate,
    )

    result = run_cli(
        [
            sys.executable,
            project_root / CLI,
            "debug-mujoco-tendons",
            "--config",
            config_path,
        ]
    )

    assert (
        "debug-mujoco-tendons skipped" in result.stdout
        or "actual_tendon_length_shape:" in result.stdout
    )


def test_debug_mujoco_tendons_interactive_launches_passive_viewer(
    monkeypatch,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    import cli as cli_module
    import continuum_sim.backends as backends

    events: list[str] = []

    class FakeBackend:
        model = object()
        data = object()

    class FakeSimViewer:
        def __init__(self) -> None:
            self.opt = SimpleNamespace(geomgroup=[0] * 8)
            self.cam = SimpleNamespace(
                lookat=[0.0, 0.0, 0.0],
                distance=0.0,
                azimuth=0.0,
                elevation=0.0,
            )
            self.user_scn = SimpleNamespace(ngeom=0)

        def is_running(self) -> bool:
            return True

        def sync(self) -> None:
            events.append("sim_sync")

    class FakeViewerContext:
        def __enter__(self):
            events.append("passive_enter")
            return FakeSimViewer()

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            events.append("passive_exit")

    class FakeDebugViewer:
        def __init__(self, *args, state_update_callback=None, **kwargs) -> None:
            del args, kwargs
            self.state = SimpleNamespace(time=0.0)
            self.state_update_callback = state_update_callback
            events.append("debug_init")
            if self.state_update_callback is not None:
                self.state_update_callback(self.state)

        def show(self) -> None:
            events.append("debug_show")

        def close(self) -> None:
            events.append("debug_close")

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.__path__ = []
    fake_viewer_module = ModuleType("mujoco.viewer")
    fake_viewer_module.launch_passive = lambda model, data: FakeViewerContext()
    fake_mujoco.viewer = fake_viewer_module
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer_module)
    monkeypatch.setattr(
        backends.MujocoBackend,
        "from_config",
        staticmethod(lambda config, override_xml_path=None: FakeBackend()),
    )
    monkeypatch.setattr(cli_module, "MujocoTendonDebugViewer", FakeDebugViewer)

    def mutate(raw: dict) -> None:
        _mutate_mujoco_tracking_backend(raw, project_root)
        raw["viewer"]["show"] = True
        raw["viewer"]["overlays"]["tendon_paths"] = False

    config_path = write_yaml_copy(
        project_root / MUJOCO_CONFIG,
        tmp_path / "mujoco_debug_interactive.yaml",
        mutate,
    )

    assert cli_module.debug_mujoco_tendons(config_path) == 0
    assert events == [
        "passive_enter",
        "debug_init",
        "sim_sync",
        "sim_sync",
        "debug_show",
        "debug_close",
        "passive_exit",
    ]


def test_run_mujoco_tracking_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    mujoco_config_path = write_yaml_copy(
        project_root / MUJOCO_CONFIG,
        tmp_path / "mujoco.yaml",
        lambda raw: _mutate_mujoco_tracking_backend(raw, project_root),
    )

    def mutate(raw: dict) -> None:
        raw["mujoco_backend_config"] = str(mujoco_config_path)
        raw["simulation"]["max_steps"] = 2
        raw["simulation"]["stop_on_completion"] = False
        raw["trajectory"]["samples"] = 3
        raw["trajectory"]["radius_m"] = 0.001
        raw["mujoco"]["show_live_tendon_panel"] = False
        raw["mujoco"]["show_summary"] = False
        raw["visualization"]["show"] = False

    tracking_config_path = write_yaml_copy(
        project_root / MUJOCO_TRACKING_CONFIG,
        tmp_path / "mujoco_tracking.yaml",
        mutate,
    )

    result = run_cli(
        [
            sys.executable,
            project_root / CLI,
            "run-mujoco-tracking",
            "--config",
            tracking_config_path,
        ]
    )

    assert "run-mujoco-tracking skipped" in result.stdout or "final_error_m:" in result.stdout


def test_run_mujoco_navigation_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    mujoco_config_path = write_yaml_copy(
        project_root / MUJOCO_CONFIG,
        tmp_path / "mujoco.yaml",
        lambda raw: _mutate_mujoco_tracking_backend(raw, project_root),
    )

    def mutate(raw: dict) -> None:
        raw["mujoco_backend_config"] = str(mujoco_config_path)
        raw["robot"]["config_path"] = str(project_root / "configs" / "robot_3seg.yaml")
        raw["scene"]["config_path"] = str(
            project_root / "configs" / "scenes" / "rocket_nozzle_entry.yaml"
        )
        raw["scene"]["generated_xml_path"] = str(tmp_path / "navigation_scene.xml")
        raw["simulation"]["max_steps"] = 2
        raw["simulation"]["stop_on_completion"] = False
        raw["mission"]["waypoint_ids"] = ["entry_wall_30deg"]
        raw["mission"]["terminate_on_clearance_violation"] = False
        raw["mujoco"]["show_live_tendon_panel"] = False
        raw["mujoco"]["show_summary"] = False
        raw["visualization"]["show"] = False

    navigation_config_path = write_yaml_copy(
        project_root / MUJOCO_NAVIGATION_CONFIG,
        tmp_path / "mujoco_navigation.yaml",
        mutate,
    )

    result = run_cli(
        [
            sys.executable,
            project_root / CLI,
            "run-mujoco-navigation",
            "--config",
            navigation_config_path,
        ]
    )

    assert "run-mujoco-navigation skipped" in result.stdout or "final_error_m:" in result.stdout


def test_run_mujoco_wiping_headless_yaml_smoke(
    run_cli,
    project_root: Path,
    tmp_path: Path,
    write_yaml_copy,
) -> None:
    mujoco_config_path = write_yaml_copy(
        project_root / MUJOCO_CONFIG,
        tmp_path / "mujoco.yaml",
        lambda raw: _mutate_mujoco_tracking_backend(raw, project_root),
    )

    def mutate(raw: dict) -> None:
        raw["mujoco_backend_config"] = str(mujoco_config_path)
        raw["robot"]["config_path"] = str(project_root / "configs" / "robot_3seg.yaml")
        raw["scene"]["config_path"] = str(
            project_root / "configs" / "scenes" / "wiping_board.yaml"
        )
        raw["scene"]["generated_xml_path"] = str(tmp_path / "wiping_scene.xml")
        raw["simulation"]["max_steps"] = 2
        raw["simulation"]["stop_on_completion"] = False
        raw["motion"]["line_count"] = 1
        raw["motion"]["samples_per_line"] = 2
        raw["motion"]["waypoint_tolerance_m"] = 0.0
        raw["controller"]["contact_loss_tolerance_steps"] = 100
        raw["mujoco"]["show_live_tendon_panel"] = False
        raw["mujoco"]["show_live_force_panel"] = False
        raw["mujoco"]["show_summary"] = False
        raw["visualization"]["show"] = False

    wiping_config_path = write_yaml_copy(
        project_root / MUJOCO_WIPING_CONFIG,
        tmp_path / "mujoco_wiping.yaml",
        mutate,
    )

    result = run_cli(
        [
            sys.executable,
            project_root / CLI,
            "run-mujoco-wiping",
            "--config",
            wiping_config_path,
        ]
    )

    assert "run-mujoco-wiping skipped" in result.stdout or "final_error_m:" in result.stdout


def _use_project_mujoco_paths(raw: dict, project_root: Path) -> None:
    raw["robot_config_path"] = str(project_root / "configs" / "robot_3seg.yaml")
    raw["xml_path"] = str(project_root / "assets" / "mujoco" / "three_segment_arm.xml")
    raw["tendon_xml_path"] = str(
        project_root / "assets" / "mujoco" / "three_segment_arm_tendon.xml"
    )
    raw["generated_xml_path"] = str(
        project_root / "assets" / "mujoco" / "three_segment_arm_with_visuals.xml"
    )
    raw["tendon_generated_xml_path"] = str(
        project_root
        / "assets"
        / "mujoco"
        / "three_segment_arm_tendon_with_visuals.xml"
    )
    raw["visuals"]["directory"] = str(
        project_root / "assets" / "meshes" / "mujoco_visual_segments"
    )
    raw["visuals"]["template_path"] = str(
        project_root / "assets" / "mujoco" / "segmented_visuals_template.xml"
    )


def _mutate_mujoco_tracking_backend(raw: dict, project_root: Path) -> None:
    _use_project_mujoco_paths(raw, project_root)
    raw["viewer"]["show"] = False
    raw["viewer"]["steps"] = 2
    raw["viewer"]["use_segment_visuals"] = False
    raw["visuals"]["enabled"] = False
    raw["control_mode"] = "tendon_position"
