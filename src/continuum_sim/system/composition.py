"""System-level YAML composition for single/dual engine simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.config import load_mujoco_config, load_yaml
from continuum_sim.config_validation import resolve_path
from continuum_sim.model.robot_assembly import (
    RobotAssemblyConfig,
    load_robot_assembly_config,
)
from continuum_sim.scenes.engine_mjcf_adapter import (
    inject_engine_scene,
    rebase_mjcf_file_assets,
    retain_spatial_arm,
)
from continuum_sim.scenes.engine_scene import EngineSceneConfig, load_engine_scene_config


@dataclass(frozen=True)
class EngineSystemComposition:
    """Resolved files required to compose one engine simulation."""

    path: Path
    name: str
    assembly: RobotAssemblyConfig
    mujoco_config_path: Path
    source_robot_xml_path: Path
    engine_scene: EngineSceneConfig
    retain_arm: str | None
    include_visual_mesh: bool
    include_collision_mesh: bool
    include_control_primitives: bool
    generated_xml_path: Path
    controller_dt_s: float
    n_substeps: int
    max_steps: int


def load_engine_system_composition(path: str | Path) -> EngineSystemComposition:
    """Load a single- or dual-arm engine system composition YAML."""

    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    values = raw.get("engine_system")
    if not isinstance(values, dict):
        raise ValueError("engine_system must be a mapping.")
    assembly_path = resolve_path(config_path, _required(values, "assembly_config_path"))
    scene_path = resolve_path(config_path, _required(values, "engine_scene_config_path"))
    retain_arm = values.get("retain_arm")
    return EngineSystemComposition(
        path=config_path,
        name=str(values.get("name", config_path.stem)),
        assembly=load_robot_assembly_config(assembly_path),
        mujoco_config_path=resolve_path(
            config_path,
            _required(values, "mujoco_config_path"),
        ),
        source_robot_xml_path=resolve_path(
            config_path,
            _required(values, "source_robot_xml_path"),
        ),
        engine_scene=load_engine_scene_config(scene_path),
        retain_arm=None if retain_arm is None else str(retain_arm),
        include_visual_mesh=bool(values.get("include_visual_mesh", True)),
        include_collision_mesh=bool(values.get("include_collision_mesh", False)),
        include_control_primitives=bool(values.get("include_control_primitives", True)),
        generated_xml_path=resolve_path(
            config_path,
            values.get("generated_xml_path", "../../output/generated/engine_system.xml"),
        ),
        controller_dt_s=float(values.get("controller_dt_s", 0.02)),
        n_substeps=int(values.get("n_substeps", 20)),
        max_steps=int(values.get("max_steps", 1000)),
    )


def build_engine_system_backend(
    composition: str | Path | EngineSystemComposition,
    generated_xml_path: str | Path | None = None,
) -> MujocoSystemBackend:
    """Compose MJCF and construct the named direct-tendon MuJoCo backend."""

    resolved = (
        composition
        if isinstance(composition, EngineSystemComposition)
        else load_engine_system_composition(composition)
    )
    output_path = (
        resolved.generated_xml_path
        if generated_xml_path is None
        else Path(generated_xml_path).resolve()
    )
    tree = ET.parse(resolved.source_robot_xml_path)
    root = tree.getroot()
    rebase_mjcf_file_assets(
        root,
        resolved.source_robot_xml_path.parent,
        output_path.parent,
    )
    if resolved.retain_arm is not None:
        retain_spatial_arm(root, resolved.retain_arm)
    inject_engine_scene(
        root,
        resolved.engine_scene,
        output_dir=output_path.parent,
        include_visual_mesh=resolved.include_visual_mesh,
        include_collision_mesh=resolved.include_collision_mesh,
        include_control_primitives=resolved.include_control_primitives,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return MujocoSystemBackend(
        load_mujoco_config(resolved.mujoco_config_path),
        resolved.assembly,
        xml_path=output_path,
    )


def _required(values: dict, name: str) -> object:
    if name not in values:
        raise ValueError(f"Missing required engine_system field {name!r}.")
    return values[name]
