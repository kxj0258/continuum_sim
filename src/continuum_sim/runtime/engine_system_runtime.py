"""Headless single/dual spatial-arm engine runtime composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from continuum_sim.control.coordinated_tracking import (
    CoordinatedTrackingController,
    CoordinatedTrackingTarget,
)
from continuum_sim.model.base_pose import Pose6D
from continuum_sim.runtime.simulation_loop import (
    SimulationLoop,
    SimulationLoopConfig,
    SimulationLoopResult,
)
from continuum_sim.scenes.engine_query import EnginePrimitiveSceneQuery
from continuum_sim.scenes.engine_scene import effective_engine_frame_position
from continuum_sim.system.composition import (
    EngineSystemComposition,
    build_engine_system_backend,
    load_engine_system_composition,
)


def run_engine_system(
    composition: str | Path | EngineSystemComposition,
) -> SimulationLoopResult:
    """Run the configured system with executor/observer coordinated tracking."""

    resolved = (
        composition
        if isinstance(composition, EngineSystemComposition)
        else load_engine_system_composition(composition)
    )
    backend = build_engine_system_backend(resolved)
    target_position = _default_engine_target_world(resolved)
    controller = CoordinatedTrackingController(
        resolved.assembly,
        CoordinatedTrackingTarget(
            executor_position_world=target_position,
            observer_roi_position_world=target_position,
        ),
        scene_query=EnginePrimitiveSceneQuery(resolved.engine_scene),
    )
    return SimulationLoop(
        backend,
        controller,
        SimulationLoopConfig(
            controller_dt_s=resolved.controller_dt_s,
            n_substeps=resolved.n_substeps,
            max_steps=resolved.max_steps,
        ),
    ).run()


def _default_engine_target_world(composition: EngineSystemComposition) -> np.ndarray:
    scene = composition.engine_scene
    start = scene.exploration_start
    if start is None:
        return effective_engine_frame_position(scene)
    if start.frame == "world":
        return np.asarray(start.point_m, dtype=float)
    engine_frame = Pose6D(
        position=effective_engine_frame_position(scene),
        quat=scene.engine.pose.quat_wxyz,
    )
    return engine_frame.transform_point(np.asarray(start.point_m, dtype=float))

