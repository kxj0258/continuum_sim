"""Minimal YAML-first command entry point for continuum_sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.actuation import load_motor_params_from_yaml
from continuum_sim.config import load_mujoco_config, load_yaml
from continuum_sim.control import DifferentialIKConfig
from continuum_sim.io import save_run_artifacts
from continuum_sim.kinematics import ContinuumKinematicsChain
from continuum_sim.model import (
    MobileBaseArmContext,
    ThreeSegmentRobotParams,
    load_physical_tendons_from_yaml,
)
from continuum_sim.runtime.mujoco_tracking_runtime import (
    _show_mujoco_tracking_summary,
    run_mujoco_trajectory_tracking,
)
from continuum_sim.runtime.mujoco_navigation_runtime import run_mujoco_navigation
from continuum_sim.runtime.mujoco_wiping_runtime import run_mujoco_wiping
from continuum_sim.tasks import (
    build_target_positions,
    load_mujoco_navigation_config,
    load_mujoco_tracking_config,
    load_mujoco_wiping_config,
    load_tracking_config,
)
from continuum_sim.visualization.motor_chain_viewer import MotorChainInteractiveViewer
from continuum_sim.visualization.mujoco_tendon_debug_viewer import (
    MujocoTendonDebugViewer,
)
from continuum_sim.visualization.pcc_viewer import PCCInteractiveViewer, named_q
from continuum_sim.visualization.trajectory_tracking_viewer import (
    animate_tracking_result,
    plot_tracking_result,
)


DEFAULT_MAIN_CONFIG = Path("configs/main_config.yaml")
RUN_COMMANDS = (
    "run-tracking",
    "run-mujoco-tracking",
    "run-mujoco-navigation",
    "run-mujoco-wiping",
)
CLI_COMMANDS = (
    "view-pcc",
    "view-motor-chain",
    "run-tracking",
    "view-mujoco",
    "debug-mujoco-tendons",
    "run-mujoco-tracking",
    "run-mujoco-navigation",
    "run-mujoco-wiping",
)

MUJOCO_ROLLOUT_SHAPE_FIELDS = (
    "tip_pose",
    "segment_poses",
    "motor_velocity",
    "tendon_delta",
    "mujoco_control",
    "qpos",
    "qvel",
    "tendon_length",
    "tendon_velocity",
    "actuator_force",
)

MUJOCO_WIPING_SHAPE_FIELDS = (
    "tip_pose",
    "target_pose",
    "segment_poses",
    "motor_velocity",
    "tendon_delta",
    "mujoco_control",
    "tendon_length",
)

MUJOCO_WIPING_RENAMED_SHAPE_FIELDS = (
    ("normal_force", "normal_force_n"),
    ("contact_proxy", "contact_proxy_m"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in CLI_COMMANDS:
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_MAIN_CONFIG,
            help="Path to main_config.yaml or a command-specific YAML config.",
        )
        if name in RUN_COMMANDS:
            subparser.add_argument(
                "--save-run",
                action="store_true",
                help="Save result NPZ, plots, configs, metadata, and replay video.",
            )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command in COMMAND_HANDLERS:
            handler = COMMAND_HANDLERS[args.command]
            if args.command in RUN_COMMANDS:
                return handler(args.config, save_run=args.save_run)
            return handler(args.config)
    except ModuleNotFoundError as exc:
        print(f"{args.command} skipped: {exc}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1
    raise ValueError(f"Unsupported command {args.command!r}.")


def view_pcc(config_path: str | Path) -> int:
    raw, resolved_config = _load_config(config_path)
    robot_config = _robot_config_from_any_yaml(raw, resolved_config)
    visualization = _visualization_values(raw)
    samples_per_segment = int(
        visualization.get(
            "samples_per_segment",
            visualization.get("animation", {}).get("samples_per_segment", 40),
        )
    )
    initial_q_name = str(visualization.get("initial_q", "straight"))
    show = _bool_value(visualization.get("show", True), "visualization.show")

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    params = ThreeSegmentRobotParams.from_yaml(robot_config)
    arm_context = _arm_context_from_any_yaml(raw, resolved_config)
    viewer = PCCInteractiveViewer(
        params,
        initial_q=named_q(initial_q_name),
        samples_per_segment=samples_per_segment,
        arm_context=arm_context,
    )
    viewer.update_plot(redraw=False)
    if not show:
        viewer.close()
        return 0
    viewer.show()
    return 0


def view_motor_chain(config_path: str | Path) -> int:
    raw, resolved_config = _load_config(config_path)
    robot_config = _robot_config_from_any_yaml(raw, resolved_config)
    visualization = _visualization_values(raw)
    motor_view = raw.get("motor_chain_viewer", {})
    if not isinstance(motor_view, dict):
        raise ValueError("motor_chain_viewer must be a mapping when provided.")
    simulation = raw.get("simulation", {})
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be a mapping when provided.")
    show = _bool_value(visualization.get("show", True), "visualization.show")
    samples_per_segment = int(
        visualization.get(
            "samples_per_segment",
            visualization.get("animation", {}).get("samples_per_segment", 40),
        )
    )
    position_limit = float(
        motor_view.get("position_limit_rad", simulation.get("position_limit_rad", 2.0))
    )
    velocity_limit = float(motor_view.get("velocity_limit_rad_s", 1.0))
    dt = float(motor_view.get("dt", simulation.get("dt", 0.02)))
    initial_motor = np.asarray(
        motor_view.get(
            "initial_motor_position_rad",
            simulation.get("initial_motor_position_rad", [0.0] * 9),
        ),
        dtype=float,
    )
    initial_velocity = np.asarray(
        motor_view.get("initial_motor_velocity_rad_s", [0.0] * initial_motor.size),
        dtype=float,
    )

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    params = ThreeSegmentRobotParams.from_yaml(robot_config)
    physical_tendons = load_physical_tendons_from_yaml(robot_config)
    motor_params = load_motor_params_from_yaml(robot_config)
    arm_context = _arm_context_from_any_yaml(raw, resolved_config)
    viewer = MotorChainInteractiveViewer(
        params,
        physical_tendons,
        motor_params,
        position_limit_rad=position_limit,
        velocity_limit_rad_s=velocity_limit,
        dt=dt,
        samples_per_segment=samples_per_segment,
        arm_context=arm_context,
    )
    viewer.set_motor_state(initial_motor, initial_velocity)
    viewer.update_plot(redraw=False)
    if not show:
        viewer.close()
        return 0
    viewer.show()
    return 0


def run_tracking(config_path: str | Path, *, save_run: bool = False) -> int:
    task_config_path = _tracking_config_path(config_path, "pcc_tracking_config")
    config = load_tracking_config(task_config_path)
    show = config.visualization.show
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chain = ContinuumKinematicsChain.from_robot_config(config.robot_config_path)
    target_positions = build_target_positions(config, chain.params)
    arm_context = _arm_context_for_tracking_config(task_config_path)
    result = chain.simulate_tracking(
        config.simulation.initial_motor_position_rad,
        target_positions,
        DifferentialIKConfig(
            dt=config.simulation.dt,
            damping=config.controller.damping,
            position_gain=config.controller.position_gain,
            max_motor_velocity_rad_s=config.controller.max_motor_velocity_rad_s,
            position_tolerance_m=config.controller.position_tolerance_m,
            max_steps=config.simulation.max_steps,
        ),
        position_limit_rad=config.simulation.position_limit_rad,
        stop_on_completion=config.simulation.stop_on_completion,
    )
    if save_run:
        _save_run_and_print(
            command="run-tracking",
            result=result,
            config_path=config_path,
            task_config_path=task_config_path,
        )
    _print_tracking_errors(result.error_norm)

    anim = None
    if config.visualization.mode == "animation":
        fig, anim = animate_tracking_result(
            result,
            chain.params,
            samples_per_segment=config.visualization.animation_samples_per_segment,
            interval_ms=config.visualization.animation_interval_ms,
            stride=config.visualization.animation_stride,
            arm_context=arm_context,
        )
        summary_fig = None
        if config.visualization.show_summary_after_animation:
            summary_fig = plot_tracking_result(
                result,
                chain.params,
                arm_context=arm_context,
            )
    else:
        fig = plot_tracking_result(
            result,
            chain.params,
            arm_context=arm_context,
        )
        summary_fig = None

    if not show:
        fig.canvas.draw()
        if summary_fig is not None:
            summary_fig.canvas.draw()
            plt.close(summary_fig)
        plt.close(fig)
        return 0
    _animation_reference = anim
    _summary_reference = summary_fig
    plt.show()
    _ = _animation_reference
    _ = _summary_reference
    return 0


def view_mujoco(config_path: str | Path) -> int:
    raw, resolved_config = _load_config(config_path)
    mujoco_config_path = _indexed_config(raw, resolved_config, "mujoco_backend_config")
    config = load_mujoco_config(mujoco_config_path)
    try:
        from continuum_sim.backends import MujocoBackend
        from continuum_sim.runtime.mujoco_runtime_utils import (
            _configure_viewer_camera,
            _configure_viewer_groups,
            draw_tendon_path_overlay_if_enabled,
            resolve_runtime_xml_path,
            sleep_for_realtime,
        )
    except ModuleNotFoundError:
        raise

    runtime_xml_path = resolve_runtime_xml_path(config, config.viewer.use_segment_visuals)
    backend = MujocoBackend.from_config(config, override_xml_path=runtime_xml_path)
    state = backend.reset()
    control = np.zeros(backend.model.nu, dtype=float)
    if not config.viewer.show:
        for _ in range(config.viewer.steps):
            state = backend.step(control)
        _print_mujoco_state(state)
        return 0

    import time
    import mujoco.viewer

    overlay_params = None
    overlay_tendons = ()
    if config.viewer.overlays.tendon_paths:
        overlay_params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
        overlay_tendons = load_physical_tendons_from_yaml(config.robot_config_path)

    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        _configure_viewer_groups(viewer, config, config.viewer.show_collision_geoms)
        _configure_viewer_camera(viewer, config)
        realtime_start_wall = time.perf_counter()
        realtime_start_sim = state.time
        for step_index in range(config.viewer.steps):
            state = backend.step(control)
            if (step_index + 1) % config.viewer.sync_interval_steps == 0:
                draw_tendon_path_overlay_if_enabled(
                    viewer,
                    backend,
                    config,
                    overlay_params,
                    overlay_tendons,
                )
                viewer.sync()
            if config.viewer.realtime:
                sleep_for_realtime(
                    realtime_start_wall,
                    realtime_start_sim,
                    state.time,
                    config.viewer.realtime_factor,
                )
            if not viewer.is_running():
                break
    _print_mujoco_state(state)
    return 0


def debug_mujoco_tendons(config_path: str | Path) -> int:
    raw, resolved_config = _load_config(config_path)
    mujoco_config_path = _indexed_config(raw, resolved_config, "mujoco_backend_config")
    config = load_mujoco_config(mujoco_config_path)
    if config.control_mode != "tendon_position":
        raise ValueError("debug-mujoco-tendons requires control_mode='tendon_position'.")

    from continuum_sim.backends import MujocoBackend
    from continuum_sim.runtime.mujoco_runtime_utils import (
        _configure_viewer_camera,
        _configure_viewer_groups,
        draw_tendon_path_overlay_if_enabled,
        resolve_runtime_xml_path,
    )

    params = ThreeSegmentRobotParams.from_yaml(config.robot_config_path)
    physical_tendons = load_physical_tendons_from_yaml(config.robot_config_path)
    runtime_xml_path = resolve_runtime_xml_path(config, config.viewer.use_segment_visuals)
    backend = MujocoBackend.from_config(config, override_xml_path=runtime_xml_path)
    control_dt = max(
        0.02,
        config.solver.timestep * max(1, config.viewer.sync_interval_steps),
    )
    if not config.viewer.show:
        viewer = MujocoTendonDebugViewer(
            backend,
            config,
            params,
            physical_tendons,
            control_dt=control_dt,
        )
        try:
            view_data = viewer.update_view(redraw=False)
            _print_mujoco_tendon_debug_state(view_data)
            return 0
        finally:
            viewer.close()

    import mujoco.viewer

    overlay_params = params if config.viewer.overlays.tendon_paths else None
    overlay_tendons = physical_tendons if config.viewer.overlays.tendon_paths else ()

    with mujoco.viewer.launch_passive(backend.model, backend.data) as sim_viewer:
        _configure_viewer_groups(
            sim_viewer,
            config,
            config.viewer.show_collision_geoms,
        )
        _configure_viewer_camera(sim_viewer, config)

        def sync_sim_viewer(_state) -> None:
            if not sim_viewer.is_running():
                return
            draw_tendon_path_overlay_if_enabled(
                sim_viewer,
                backend,
                config,
                overlay_params,
                overlay_tendons,
            )
            sim_viewer.sync()

        viewer = MujocoTendonDebugViewer(
            backend,
            config,
            params,
            physical_tendons,
            control_dt=control_dt,
            state_update_callback=sync_sim_viewer,
        )
        try:
            sync_sim_viewer(viewer.state)
            viewer.show()
        finally:
            viewer.close()
    return 0


def run_mujoco_tracking(config_path: str | Path, *, save_run: bool = False) -> int:
    task_config_path = _tracking_config_path(config_path, "mujoco_tracking_config")
    raw_task = load_yaml(task_config_path)
    mujoco_config_path = _resolve_mujoco_tracking_backend_config(task_config_path, raw_task)
    result = run_mujoco_trajectory_tracking(task_config_path, mujoco_config_path)
    if result.error_norm.size == 0:
        print("samples: 0")
        return 0
    if save_run:
        _save_run_and_print(
            command="run-mujoco-tracking",
            result=result,
            config_path=config_path,
            task_config_path=task_config_path,
            mujoco_config_path=mujoco_config_path,
        )
    _print_tracking_errors(result.error_norm)
    _print_result_shapes(result, MUJOCO_ROLLOUT_SHAPE_FIELDS)
    task_config = load_mujoco_tracking_config(task_config_path)
    _show_mujoco_tracking_summary(
        result,
        task_config_path,
        show=task_config.mujoco.show_summary,
    )
    return 0


def run_mujoco_navigation_cli(config_path: str | Path, *, save_run: bool = False) -> int:
    task_config_path = _tracking_config_path(config_path, "mujoco_navigation_config")
    raw_task = load_yaml(task_config_path)
    mujoco_config_path = _resolve_mujoco_tracking_backend_config(task_config_path, raw_task)
    result = run_mujoco_navigation(task_config_path, mujoco_config_path)
    if result.error_norm.size == 0:
        print("samples: 0")
        return 0
    if save_run:
        _save_run_and_print(
            command="run-mujoco-navigation",
            result=result,
            config_path=config_path,
            task_config_path=task_config_path,
            mujoco_config_path=mujoco_config_path,
        )
    _print_tracking_errors(result.error_norm)
    print(f"min_clearance_m: {np.min(result.min_clearance_m):.6e}")
    print(f"final_waypoint_index: {result.waypoint_index[-1]}")
    print(f"scene_xml_path: {result.scene_xml_path}")
    _print_result_shapes(result, MUJOCO_ROLLOUT_SHAPE_FIELDS)
    task_config = load_mujoco_navigation_config(task_config_path)
    if task_config.mujoco.show_summary:
        print("navigation_summary: command-line metrics only")
    return 0


def run_mujoco_wiping_cli(config_path: str | Path, *, save_run: bool = False) -> int:
    task_config_path = _tracking_config_path(config_path, "mujoco_wiping_config")
    raw_task = load_yaml(task_config_path)
    mujoco_config_path = _resolve_mujoco_tracking_backend_config(task_config_path, raw_task)
    result = run_mujoco_wiping(task_config_path, mujoco_config_path)
    if result.error_norm.size == 0:
        print("samples: 0")
        return 0
    if save_run:
        _save_run_and_print(
            command="run-mujoco-wiping",
            result=result,
            config_path=config_path,
            task_config_path=task_config_path,
            mujoco_config_path=mujoco_config_path,
        )
    _print_tracking_errors(result.error_norm)
    print(f"max_normal_force_n: {np.max(result.normal_force_n):.6e}")
    print(f"final_force_error_n: {result.force_error_n[-1]:.6e}")
    print(f"final_phase: {result.phase[-1]}")
    print(f"final_waypoint_index: {result.waypoint_index[-1]}")
    print(f"scene_xml_path: {result.scene_xml_path}")
    _print_result_shapes(
        result,
        MUJOCO_WIPING_SHAPE_FIELDS,
        MUJOCO_WIPING_RENAMED_SHAPE_FIELDS,
    )
    task_config = load_mujoco_wiping_config(task_config_path)
    if task_config.mujoco.show_summary:
        print("wiping_summary: command-line metrics only")
    return 0


COMMAND_HANDLERS = {
    "view-pcc": view_pcc,
    "view-motor-chain": view_motor_chain,
    "run-tracking": run_tracking,
    "view-mujoco": view_mujoco,
    "debug-mujoco-tendons": debug_mujoco_tendons,
    "run-mujoco-tracking": run_mujoco_tracking,
    "run-mujoco-navigation": run_mujoco_navigation_cli,
    "run-mujoco-wiping": run_mujoco_wiping_cli,
}


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    return load_yaml(path), path


def _tracking_config_path(config_path: str | Path, main_key: str) -> Path:
    raw, resolved_config = _load_config(config_path)
    if main_key in raw:
        return _resolve_path(resolved_config, raw[main_key])
    return resolved_config


def _indexed_config(raw: dict[str, Any], resolved_config: Path, key: str) -> Path:
    if key in raw:
        return _resolve_path(resolved_config, raw[key])
    return resolved_config


def _robot_config_from_any_yaml(raw: dict[str, Any], resolved_config: Path) -> Path:
    if "robot_config" in raw:
        return _resolve_path(resolved_config, raw["robot_config"])
    if "robot" in raw and isinstance(raw["robot"], dict) and "config_path" in raw["robot"]:
        return _resolve_path(resolved_config, raw["robot"]["config_path"])
    if "robot_config_path" in raw:
        return _resolve_path(resolved_config, raw["robot_config_path"])
    main = load_yaml(PROJECT_ROOT / DEFAULT_MAIN_CONFIG)
    return _resolve_path(PROJECT_ROOT / DEFAULT_MAIN_CONFIG, main["robot_config"])


def _mobile_base_config_from_any_yaml(raw: dict[str, Any], resolved_config: Path) -> Path | None:
    for key in ("mobile_base_config", "mobile_base_config_path"):
        if key in raw and raw[key] not in (None, ""):
            return _resolve_path(resolved_config, raw[key])
    main = load_yaml(PROJECT_ROOT / DEFAULT_MAIN_CONFIG)
    raw_main_value = main.get("mobile_base_config")
    if raw_main_value in (None, ""):
        return None
    return _resolve_path(PROJECT_ROOT / DEFAULT_MAIN_CONFIG, raw_main_value)


def _arm_context_from_any_yaml(raw: dict[str, Any], resolved_config: Path) -> MobileBaseArmContext:
    return MobileBaseArmContext.from_config_path(
        _mobile_base_config_from_any_yaml(raw, resolved_config)
    )


def _arm_context_for_tracking_config(task_config_path: Path) -> MobileBaseArmContext:
    raw, resolved = _load_config(task_config_path)
    return _arm_context_from_any_yaml(raw, resolved)


def _visualization_values(raw: dict[str, Any]) -> dict[str, Any]:
    values = raw.get("visualization", {})
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("visualization must be a mapping when provided.")
    return values


def _resolve_mujoco_tracking_backend_config(task_config_path: Path, raw_task: dict[str, Any]) -> Path:
    if "mujoco_backend_config" in raw_task:
        return _resolve_path(task_config_path, raw_task["mujoco_backend_config"])
    main = load_yaml(PROJECT_ROOT / DEFAULT_MAIN_CONFIG)
    return _resolve_path(PROJECT_ROOT / DEFAULT_MAIN_CONFIG, main["mujoco_backend_config"])


def _resolve_path(config_path: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    parent_candidate = (config_path.parent / path).resolve()
    if parent_candidate.exists():
        return parent_candidate
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return parent_candidate


def _save_run_and_print(
    *,
    command: str,
    result: object,
    config_path: str | Path,
    task_config_path: str | Path,
    mujoco_config_path: str | Path | None = None,
) -> None:
    raw_task = load_yaml(task_config_path)
    paths = save_run_artifacts(
        command=command,
        result=result,
        task_config_path=task_config_path,
        main_config_path=_main_config_path_if_index(config_path),
        mujoco_config_path=mujoco_config_path,
        task_name=str(raw_task.get("name", command)),
    )
    print(f"run_dir: {paths.run_dir}")


def _main_config_path_if_index(config_path: str | Path) -> Path | None:
    raw, resolved = _load_config(config_path)
    index_keys = {
        "pcc_tracking_config",
        "mujoco_tracking_config",
        "mujoco_navigation_config",
        "mujoco_wiping_config",
    }
    if any(key in raw for key in index_keys):
        return resolved
    return None


def _bool_value(raw_value: object, name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}.")


def _print_tracking_errors(error_norm: np.ndarray) -> None:
    print(f"final_error_m: {error_norm[-1]:.6e}")
    print(f"mean_error_m: {np.mean(error_norm):.6e}")
    print(f"max_error_m: {np.max(error_norm):.6e}")


def _print_result_shapes(
    result: object,
    fields: tuple[str, ...],
    renamed_fields: tuple[tuple[str, str], ...] = (),
) -> None:
    for attribute in fields:
        value = getattr(result, attribute)
        print(f"{attribute}_shape: {value.shape}")
    for label, attribute in renamed_fields:
        value = getattr(result, attribute)
        print(f"{label}_shape: {value.shape}")


def _print_mujoco_state(state) -> None:
    tip_position = state.tip_pose[:3, 3]
    print(f"time_s: {state.time:.6f}")
    print(
        "tip_position_m: "
        f"[{tip_position[0]:.6e}, {tip_position[1]:.6e}, {tip_position[2]:.6e}]"
    )
    print(f"segment_poses_shape: {state.segment_poses.shape}")


def _print_mujoco_tendon_debug_state(view_data) -> None:
    tip = view_data.tip_position
    print(f"time_s: {view_data.time_s:.6f}")
    print(
        "tip_position_m: "
        f"[{tip[0]:.6e}, {tip[1]:.6e}, {tip[2]:.6e}]"
    )
    print(f"commanded_tendon_delta_shape: {view_data.commanded_tendon_delta.shape}")
    print(f"actual_tendon_length_shape: {view_data.actual_tendon_length.shape}")
    print(f"actuator_force_shape: {view_data.actuator_force.shape}")


if __name__ == "__main__":
    raise SystemExit(main())
