"""Structured scene descriptions and MuJoCo XML builders."""

from continuum_sim.scenes.contact_surfaces import (
    SurfaceDistanceQuery,
    WipePatchConfig,
    WorkSurfaceConfig,
)
from continuum_sim.scenes.primitives import (
    BoxObstaclePrimitive,
    ClearancePrimitive,
    CylinderObstaclePrimitive,
    DistanceQuery,
    InteriorShellPrimitive,
    nearest_clearance,
)
from continuum_sim.scenes.scene_builder import (
    ToolPadXmlConfig,
    build_mujoco_scene_xml,
    build_mujoco_wiping_xml,
    inject_mobile_base_wrapper,
    inject_tool_contact_pad,
    lock_mobile_base_freejoint,
)
from continuum_sim.scenes.scene_config import (
    InspectionTargetConfig,
    NavigationSceneConfig,
    SceneBuilderConfig,
    ScenePrimitiveConfig,
    load_navigation_scene_config,
)
from continuum_sim.scenes.engine_mjcf_adapter import (
    build_engine_mujoco_scene_xml,
    inject_engine_scene,
    prepare_mujoco_stl,
    rebase_mjcf_file_assets,
    retain_spatial_arm,
)
from continuum_sim.scenes.engine_query import (
    EnginePrimitiveSceneQuery,
    EngineSceneQueryProtocol,
)
from continuum_sim.scenes.structured_query import StructuredSceneQuery

__all__ = [
    "BoxObstaclePrimitive",
    "ClearancePrimitive",
    "CylinderObstaclePrimitive",
    "DistanceQuery",
    "InspectionTargetConfig",
    "InteriorShellPrimitive",
    "NavigationSceneConfig",
    "SceneBuilderConfig",
    "ScenePrimitiveConfig",
    "SurfaceDistanceQuery",
    "ToolPadXmlConfig",
    "WipePatchConfig",
    "WorkSurfaceConfig",
    "build_mujoco_scene_xml",
    "build_mujoco_wiping_xml",
    "inject_mobile_base_wrapper",
    "inject_tool_contact_pad",
    "lock_mobile_base_freejoint",
    "load_navigation_scene_config",
    "nearest_clearance",
    "build_engine_mujoco_scene_xml",
    "inject_engine_scene",
    "prepare_mujoco_stl",
    "rebase_mjcf_file_assets",
    "retain_spatial_arm",
    "EnginePrimitiveSceneQuery",
    "EngineSceneQueryProtocol",
    "StructuredSceneQuery",
]
