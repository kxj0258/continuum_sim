"""Fixed-tendon overlay helpers for MuJoCo passive viewers."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from continuum_sim.model import PhysicalTendonPath, ThreeSegmentRobotParams
from continuum_sim.model.hole_pattern import TendonHolePattern


TENDON_PATH_RGBA_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.92, 0.20, 0.16, 0.90),
    (0.95, 0.55, 0.12, 0.90),
    (0.98, 0.84, 0.20, 0.90),
    (0.36, 0.73, 0.19, 0.90),
    (0.10, 0.73, 0.61, 0.90),
    (0.09, 0.60, 0.92, 0.90),
    (0.29, 0.43, 0.95, 0.90),
    (0.64, 0.33, 0.94, 0.90),
    (0.90, 0.26, 0.70, 0.90),
    (0.55, 0.32, 0.16, 0.90),
    (0.15, 0.15, 0.15, 0.90),
    (0.55, 0.70, 0.90, 0.90),
    (0.69, 0.08, 0.16, 0.90),
    (0.00, 0.50, 0.00, 0.90),
    (0.00, 0.45, 0.70, 0.90),
    (0.84, 0.37, 0.00, 0.90),
    (0.80, 0.47, 0.65, 0.90),
    (0.35, 0.70, 0.90, 0.90),
    (0.94, 0.90, 0.25, 0.90),
    (0.00, 0.62, 0.45, 0.90),
    (0.90, 0.62, 0.00, 0.90),
    (0.34, 0.24, 0.05, 0.90),
    (0.00, 0.45, 0.70, 0.90),
    (0.80, 0.40, 0.00, 0.90),
    (0.80, 0.60, 0.70, 0.90),
)


def draw_tendon_path_overlay(
    scene,
    mujoco_module,
    model,
    data,
    params: ThreeSegmentRobotParams,
    physical_tendons: tuple[PhysicalTendonPath, ...],
    *,
    links_per_segment: int,
    radius: float,
    stride: int,
    arm_name: str | None = None,
    hole_pattern: TendonHolePattern | None = None,
) -> None:
    """Append colored tendon path capsules into ``scene``."""

    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")

    body_names = iter_tendon_body_names(
        physical_tendons,
        links_per_segment,
        arm_name=arm_name,
    )
    body_poses = _body_pose_map_from_mujoco(mujoco_module, model, data, body_names)
    for tendon in physical_tendons:
        rgba = TENDON_PATH_RGBA_PALETTE[
            tendon.global_index % len(TENDON_PATH_RGBA_PALETTE)
        ]
        points = tendon_path_polyline_points(
            body_poses,
            params,
            tendon,
            links_per_segment=links_per_segment,
            arm_name=arm_name,
            hole_pattern=hole_pattern,
        )
        for start, end in _polyline_segments(points, stride=stride):
            geom = _next_overlay_geom(scene)
            if geom is None:
                return
            _connect_capsule_geom(mujoco_module, geom, radius, start, end)
            geom.rgba[:] = rgba


def iter_tendon_body_names(
    physical_tendons: tuple[PhysicalTendonPath, ...],
    links_per_segment: int,
    arm_name: str | None = None,
) -> tuple[str, ...]:
    """Return unique link-body names needed by the overlay, in traversal order."""

    if links_per_segment <= 0:
        raise ValueError(
            f"links_per_segment must be positive, got {links_per_segment}."
        )

    names: list[str] = []
    seen: set[str] = set()
    for tendon in physical_tendons:
        if arm_name is not None:
            base_name = f"{arm_name}_base"
            if base_name not in seen:
                seen.add(base_name)
                names.append(base_name)
        for segment_index in tendon.path_segment_indices:
            for link_index in range(links_per_segment):
                name = _link_body_name(segment_index, link_index, arm_name=arm_name)
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
    return tuple(names)


def tendon_path_polyline_points(
    body_poses: Mapping[str, np.ndarray],
    params: ThreeSegmentRobotParams,
    tendon: PhysicalTendonPath,
    *,
    links_per_segment: int,
    arm_name: str | None = None,
    hole_pattern: TendonHolePattern | None = None,
) -> list[np.ndarray]:
    """Build a world-space polyline for one physical tendon path."""

    if links_per_segment <= 0:
        raise ValueError(
            f"links_per_segment must be positive, got {links_per_segment}."
        )

    points: list[np.ndarray] = []
    if arm_name is not None and hole_pattern is not None and tendon.hole_index is not None:
        base_pose = np.asarray(body_poses[f"{arm_name}_base"], dtype=float)
        base_hole = hole_pattern.base_hole_by_index(tendon.hole_index)
        points.append(
            _transform_point(
                base_pose,
                np.array([base_hole.xy_m[0], base_hole.xy_m[1], base_hole.in_z_m], dtype=float),
            )
        )
        points.append(
            _transform_point(
                base_pose,
                np.array([base_hole.xy_m[0], base_hole.xy_m[1], base_hole.out_z_m], dtype=float),
            )
        )
    for segment_index in tendon.path_segment_indices:
        if segment_index < 0 or segment_index >= len(params.segments):
            raise ValueError(
                f"{tendon.id} has segment index {segment_index} outside robot bounds."
            )
        link_length = params.segments[segment_index].length / float(links_per_segment)
        for link_index in range(links_per_segment):
            link_number = segment_index * links_per_segment + link_index + 1
            local_in_point, local_out_point = _tendon_local_site_points(
                tendon,
                hole_pattern,
                arm_name=arm_name,
                global_link_number=link_number,
                segment_number=segment_index + 1,
                segment_link_number=link_index + 1,
                fallback_link_length=1.0,
            )
            local_start = _scale_fallback_site_point(
                local_in_point,
                link_length,
                hole_pattern,
            )
            local_end = _scale_fallback_site_point(
                local_out_point,
                link_length,
                hole_pattern,
            )
            body_name = _link_body_name(segment_index, link_index, arm_name=arm_name)
            pose = np.asarray(body_poses[body_name], dtype=float)
            if pose.shape != (4, 4):
                raise ValueError(
                    f"body_poses[{body_name!r}] must be a 4x4 pose, got {pose.shape}."
                )
            start = _transform_point(
                pose,
                local_start,
            )
            end = _transform_point(
                pose,
                local_end,
            )
            if not points:
                points.append(start)
            elif float(np.linalg.norm(points[-1] - start)) > 1.0e-9:
                points.append(start)
            points.append(end)
    return points


def _body_pose_map_from_mujoco(
    mujoco_module,
    model,
    data,
    body_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    xpos = getattr(data, "xpos", getattr(data, "body_xpos", None))
    xmat = getattr(data, "xmat", getattr(data, "body_xmat", None))
    if xpos is None or xmat is None:
        raise AttributeError("MuJoCo data does not expose body xpos/xmat arrays.")

    body_type = mujoco_module.mjtObj.mjOBJ_BODY
    poses: dict[str, np.ndarray] = {}
    for body_name in body_names:
        body_id = mujoco_module.mj_name2id(model, body_type, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body {body_name!r}.")
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = np.asarray(xmat[body_id], dtype=float).reshape(3, 3)
        pose[:3, 3] = np.asarray(xpos[body_id], dtype=float).reshape(3)
        poses[body_name] = pose
    return poses


def _polyline_segments(
    points: list[np.ndarray],
    *,
    stride: int,
    min_distance: float = 1.0e-9,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(points) - 1):
        if index % stride != 0:
            continue
        start = np.asarray(points[index], dtype=float)
        end = np.asarray(points[index + 1], dtype=float)
        if float(np.linalg.norm(end - start)) <= min_distance:
            continue
        segments.append((start.copy(), end.copy()))
    return segments


def _transform_point(pose: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    return pose[:3, :3] @ np.asarray(local_point, dtype=float) + pose[:3, 3]


def _tendon_local_offset(
    tendon: PhysicalTendonPath,
    hole_pattern: TendonHolePattern | None,
    *,
    arm_name: str | None,
    global_link_number: int,
    segment_number: int,
    segment_link_number: int,
) -> np.ndarray:
    if hole_pattern is not None and tendon.hole_index is not None:
        endpoint = hole_pattern.link_endpoint_by_index(
            tendon.hole_index,
            arm_name=_required_arm_name(arm_name),
            suffix="in",
            global_link_number=global_link_number,
            segment_number=segment_number,
            segment_link_number=segment_link_number,
        )
        return np.array([endpoint.xy_m[0], endpoint.xy_m[1]], dtype=float)
    theta_rad = math.radians(tendon.angle_deg)
    return np.array(
        [
            tendon.radial_offset * math.cos(theta_rad),
            tendon.radial_offset * math.sin(theta_rad),
        ],
        dtype=float,
    )


def _tendon_local_site_points(
    tendon: PhysicalTendonPath,
    hole_pattern: TendonHolePattern | None,
    *,
    arm_name: str | None,
    global_link_number: int,
    segment_number: int,
    segment_link_number: int,
    fallback_link_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    if hole_pattern is not None and tendon.hole_index is not None:
        in_endpoint = hole_pattern.link_endpoint_by_index(
            tendon.hole_index,
            arm_name=_required_arm_name(arm_name),
            suffix="in",
            global_link_number=global_link_number,
            segment_number=segment_number,
            segment_link_number=segment_link_number,
        )
        out_endpoint = hole_pattern.link_endpoint_by_index(
            tendon.hole_index,
            arm_name=_required_arm_name(arm_name),
            suffix="out",
            global_link_number=global_link_number,
            segment_number=segment_number,
            segment_link_number=segment_link_number,
        )
        return (
            np.array(
                [in_endpoint.xy_m[0], in_endpoint.xy_m[1], in_endpoint.z_m],
                dtype=float,
            ),
            np.array(
                [out_endpoint.xy_m[0], out_endpoint.xy_m[1], out_endpoint.z_m],
                dtype=float,
            ),
        )
    offset = _tendon_local_offset(
        tendon,
        hole_pattern,
        arm_name=arm_name,
        global_link_number=global_link_number,
        segment_number=segment_number,
        segment_link_number=segment_link_number,
    )
    return (
        np.array([offset[0], offset[1], 0.0], dtype=float),
        np.array([offset[0], offset[1], fallback_link_length], dtype=float),
    )


def _required_arm_name(arm_name: str | None) -> str:
    if arm_name is None:
        raise ValueError("arm_name is required when using a dual-arm hole pattern.")
    return arm_name


def _scale_fallback_site_point(
    point: np.ndarray,
    link_length: float,
    hole_pattern: TendonHolePattern | None,
) -> np.ndarray:
    if hole_pattern is not None:
        return point
    scaled = np.asarray(point, dtype=float).copy()
    scaled[2] *= link_length
    return scaled


def _link_body_name(segment_index: int, link_index: int, *, arm_name: str | None = None) -> str:
    base = f"segment_{segment_index + 1}_link_{link_index + 1}"
    if arm_name is None:
        return base
    return f"{arm_name}_{base}"


def _connect_capsule_geom(
    mujoco_module,
    geom,
    radius: float,
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    geom_type = mujoco_module.mjtGeom.mjGEOM_CAPSULE
    connector = getattr(mujoco_module, "mjv_connector", None)
    if connector is not None:
        connector(
            geom,
            geom_type,
            float(radius),
            np.ascontiguousarray(np.asarray(start, dtype=np.float64).reshape(3)),
            np.ascontiguousarray(np.asarray(end, dtype=np.float64).reshape(3)),
        )
        return

    make_connector = getattr(mujoco_module, "mjv_makeConnector", None)
    if make_connector is None:
        raise AttributeError(
            "MuJoCo module has neither mjv_connector nor mjv_makeConnector."
        )
    make_connector(
        geom,
        geom_type,
        float(radius),
        float(start[0]),
        float(start[1]),
        float(start[2]),
        float(end[0]),
        float(end[1]),
        float(end[2]),
    )


def _next_overlay_geom(scene):
    maxgeom = getattr(scene, "maxgeom", None)
    maxgeom_value = int(maxgeom) if maxgeom is not None else len(getattr(scene, "geoms", ()))
    if int(scene.ngeom) >= maxgeom_value:
        return None
    geom = scene.geoms[int(scene.ngeom)]
    scene.ngeom += 1
    return geom


__all__ = [
    "TENDON_PATH_RGBA_PALETTE",
    "draw_tendon_path_overlay",
    "iter_tendon_body_names",
    "tendon_path_polyline_points",
]
