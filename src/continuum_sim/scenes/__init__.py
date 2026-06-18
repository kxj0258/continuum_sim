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
    inject_tool_contact_pad,
)
from continuum_sim.scenes.scene_config import (
    InspectionTargetConfig,
    NavigationSceneConfig,
    SceneBuilderConfig,
    ScenePrimitiveConfig,
    load_navigation_scene_config,
)

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
    "inject_tool_contact_pad",
    "load_navigation_scene_config",
    "nearest_clearance",
]
