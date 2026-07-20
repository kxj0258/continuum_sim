"""Path helpers for scenario arm-mode specific configuration."""

from __future__ import annotations

from pathlib import Path

from continuum_sim.config_validation import resolve_path


def arm_mode_assembly_config_path(
    config_path: Path,
    values: dict[str, object],
    task_values: dict[str, object],
    arm_mode: str,
) -> Path:
    raw = values.get("assembly_config_path")
    if raw is None:
        raw = default_assembly_config_path(task_values, arm_mode)
    return resolve_path(config_path, replace_arm_mode_token(str(raw), arm_mode))


def default_assembly_config_path(
    task_values: dict[str, object],
    arm_mode: str,
) -> str:
    task_type = str(task_values.get("type", "idle"))
    mobile = task_type in ("navigation", "engine_navigation")
    prefix = "dual" if arm_mode == "dual" else "single"
    suffix = "_mobile" if mobile else ""
    return f"../robots/assemblies/{prefix}_spatial{suffix}.yaml"


def arm_mode_generated_xml_path(
    backend_values: dict[str, object],
    arm_mode: str,
) -> object | None:
    raw = backend_values.get("generated_xml_path")
    if raw is None:
        return None
    return replace_arm_mode_token(str(raw), arm_mode)


def arm_mode_retain_arm(arm_mode: str) -> str | None:
    return "executor" if arm_mode == "single" else None


def optional_path(config_path: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(config_path, value)


def replace_arm_mode_token(value: str, arm_mode: str) -> str:
    if arm_mode == "single":
        return (
            value.replace("dual_spatial_mobile.yaml", "single_spatial_mobile.yaml")
            .replace("dual_spatial.yaml", "single_spatial.yaml")
            .replace("scenario_dual_", "scenario_single_")
            .replace("dual_", "single_")
        )
    return (
        value.replace("single_spatial_mobile.yaml", "dual_spatial_mobile.yaml")
        .replace("single_spatial.yaml", "dual_spatial.yaml")
        .replace("scenario_single_", "scenario_dual_")
        .replace("single_", "dual_")
    )
