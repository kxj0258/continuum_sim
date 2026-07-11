"""Compare analytic straight-arm tip position with MuJoCo reset tip position."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from continuum_sim.application.scenario import load_scenario_config
from continuum_sim.backends.mujoco_system_backend import MujocoSystemBackend
from continuum_sim.config import load_mujoco_config
from continuum_sim.kinematics.pcc import forward_kinematics
from continuum_sim.model.robot_assembly import load_robot_assembly_config
from continuum_sim.scenes.engine_mjcf_adapter import (
    rebase_mjcf_file_assets,
    retain_spatial_arm,
)
from continuum_sim.scenes.scene_builder import lock_mobile_base_freejoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print analytic PCC straight-tip position and MuJoCo reset tip "
            "position for one executor arm."
        )
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="configs/scenarios/single_mujoco_tracking.yaml",
        help="Scenario YAML path.",
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    config = load_scenario_config(scenario_path)
    assembly = load_robot_assembly_config(config.assembly_config_path)
    executor = next(arm for arm in assembly.enabled_arms if arm.role == "executor")

    q0 = np.zeros(executor.spatial_arm.params.q_size, dtype=float)
    local_tip = forward_kinematics(q0, executor.spatial_arm.params).tip_pose[:3, 3]
    world_mount = assembly.base.initial_pose.compose(executor.mount_pose)
    analytic_tip_world = world_mount.transform_point(local_tip)

    xml_path = _prepare_mujoco_xml(config, assembly)
    backend = MujocoSystemBackend(
        load_mujoco_config(config.backend.mujoco_config_path),
        assembly,
        xml_path=xml_path,
    )
    state = backend.reset_system()
    mujoco_tip_world = state.arms[executor.name].tip_pose_world.position
    difference = mujoco_tip_world - analytic_tip_world

    print(f"scenario: {scenario_path}")
    print(f"mujoco_xml: {xml_path}")
    print(f"executor: {executor.name}")
    print(f"mount_position_m: {_format_vector(executor.mount_pose.position)}")
    print(f"analytic_local_straight_tip_m: {_format_vector(local_tip)}")
    print(f"analytic_straight_tip_world_m: {_format_vector(analytic_tip_world)}")
    print(f"mujoco_reset_tip_world_m: {_format_vector(mujoco_tip_world)}")
    print(f"difference_mujoco_minus_analytic_m: {_format_vector(difference)}")
    print(f"difference_norm_m: {float(np.linalg.norm(difference)):.9e}")


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value): .9f}" for value in values) + "]"


def _prepare_mujoco_xml(config, assembly) -> Path:
    backend = config.backend
    if backend.source_xml_path is None or backend.generated_xml_path is None:
        raise ValueError("Scenario backend requires source_xml_path and generated_xml_path.")
    tree = ET.parse(backend.source_xml_path)
    root = tree.getroot()
    rebase_mjcf_file_assets(
        root,
        backend.source_xml_path.parent,
        backend.generated_xml_path.parent,
    )
    if backend.retain_arm is not None:
        retain_spatial_arm(root, backend.retain_arm)
    if assembly.base.control_mode == "fixed":
        lock_mobile_base_freejoint(root)
    backend.generated_xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(backend.generated_xml_path, encoding="utf-8", xml_declaration=False)
    return backend.generated_xml_path


if __name__ == "__main__":
    main()
