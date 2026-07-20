"""Runtime hook construction for scenario applications."""

from __future__ import annotations

from pathlib import Path

from continuum_sim.runtime.hooks import (
    ControllerCompletionHook,
    LiveDiagnosticsPanelHook,
    LiveTendonPanelHook,
    LiveWipingForcePanelHook,
    MatplotlibSystemViewerHook,
    MujocoLiveVideoRecorderHook,
    MujocoObserverCameraFeedbackHook,
    MujocoReplayRecorderHook,
    MujocoViewerHook,
    StateRecorderHook,
    TendonDiagnosticHook,
)
from continuum_sim.tools.attachments import load_attachment_config


def build_runtime_hooks(
    *,
    config,
    backend,
    controller,
    assembly,
    observer_camera_target_world,
) -> dict[str, object]:
    """Build named runtime hooks from scenario configuration."""

    hooks_by_name: dict[str, object] = {}
    if config.hooks.recorder:
        hooks_by_name["recorder"] = StateRecorderHook()
    if config.hooks.tendon_debug:
        hooks_by_name["tendon_debug"] = TendonDiagnosticHook(
            stride=config.hooks.tendon_debug_stride
        )
    if config.hooks.show_live_tendon_panel:
        hooks_by_name["live_tendon_panel"] = LiveTendonPanelHook(
            stride=config.hooks.live_tendon_panel_stride,
            history_points=config.hooks.live_force_panel_history_points,
        )
    if config.hooks.show_live_force_panel:
        hooks_by_name["live_force_panel"] = LiveWipingForcePanelHook(
            stride=config.hooks.live_force_panel_stride,
            history_points=config.hooks.live_force_panel_history_points,
        )
    if config.hooks.show_live_diagnostics_panel:
        hooks_by_name["live_diagnostics_panel"] = LiveDiagnosticsPanelHook(
            stride=config.hooks.live_diagnostics_panel_stride,
            history_points=config.hooks.live_diagnostics_panel_history_points,
        )

    observer_camera = observer_camera_attachment_config(assembly)
    observer_camera_video_paths = (
        None
        if observer_camera is None or observer_camera.camera is None
        else observer_camera_pending_video_paths(
            config,
            observer_camera.camera.name,
        )
    )
    if (
        config.backend.type == "mujoco"
        and observer_camera is not None
        and observer_camera.camera is not None
        and (
            config.hooks.show_observer_camera
            or bool(observer_camera_video_paths)
        )
    ):
        hooks_by_name["observer_camera"] = MujocoObserverCameraFeedbackHook(
            backend,
            camera_name=observer_camera.camera.name,
            intrinsics=observer_camera.camera.intrinsics,
            fallback_target_world=observer_camera_target_world,
            show_window=config.hooks.show_observer_camera,
            stride=config.hooks.observer_camera_stride,
            video_output_paths=observer_camera_video_paths,
            video_fps=config.artifacts.video_fps,
            video_stride=config.artifacts.video_stride,
        )

    if (
        config.backend.type == "mujoco"
        and config.artifacts.enabled
        and video_artifacts_enabled(config.artifacts)
        and config.artifacts.video_mode == "live_mujoco"
    ):
        hooks_by_name["live_mujoco_video"] = MujocoLiveVideoRecorderHook(
            backend,
            live_mujoco_pending_video_paths(config),
            fps=config.artifacts.video_fps,
            stride=config.artifacts.video_stride,
            width=backend.config.rendering.offscreen_width,
            height=backend.config.rendering.offscreen_height,
        )
    if (
        config.backend.type == "mujoco"
        and config.artifacts.enabled
        and config.artifacts.video_mode == "replay"
    ):
        hooks_by_name["mujoco_replay"] = MujocoReplayRecorderHook(backend)
    if config.hooks.viewer == "matplotlib":
        hooks_by_name["viewer"] = MatplotlibSystemViewerHook(
            keep_open=config.hooks.keep_viewer_open
        )
    elif config.hooks.viewer == "mujoco":
        if config.backend.type != "mujoco":
            raise ValueError("The MuJoCo viewer requires a MuJoCo backend.")
        hooks_by_name["viewer"] = MujocoViewerHook(
            backend,
            keep_open=config.hooks.keep_viewer_open,
        )
    if hasattr(controller, "done"):
        hooks_by_name["completion"] = ControllerCompletionHook(controller)
    return hooks_by_name


def observer_camera_attachment_config(assembly):
    observer_arms = [arm for arm in assembly.enabled_arms if arm.role == "observer"]
    if len(observer_arms) != 1 or observer_arms[0].attachment is None:
        return None
    path = _attachment_config_path(assembly.path, observer_arms[0].attachment)
    if path is None:
        return None
    config = load_attachment_config(path)
    if config.type != "camera_airgun":
        return None
    return config


def video_artifacts_enabled(artifacts) -> bool:
    return bool(artifacts.save_gif or artifacts.save_mp4)


def live_mujoco_pending_video_paths(config) -> list[Path]:
    return [
        config.artifacts.output_root
        / f"_{config.name}_live_mujoco_pending.{suffix}"
        for suffix in _video_suffixes(config)
    ]


def observer_camera_pending_video_paths(config, camera_name: str) -> list[Path]:
    if (
        config.backend.type != "mujoco"
        or not config.artifacts.enabled
        or not video_artifacts_enabled(config.artifacts)
        or config.artifacts.video_mode != "live_mujoco"
    ):
        return []
    safe_camera_name = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in camera_name
    )
    return [
        config.artifacts.output_root
        / f"_{config.name}_{safe_camera_name}_pending.{suffix}"
        for suffix in _video_suffixes(config)
    ]


def _video_suffixes(config) -> list[str]:
    suffixes: list[str] = []
    if config.artifacts.save_gif:
        suffixes.append("gif")
    if config.artifacts.save_mp4:
        suffixes.append("mp4")
    return suffixes


def _attachment_config_path(assembly_path: Path, attachment_name: str) -> Path | None:
    for parent in (assembly_path.parent, *assembly_path.parents):
        candidate = parent / "tools" / f"{attachment_name}.yaml"
        if candidate.is_file():
            return candidate
    return None
