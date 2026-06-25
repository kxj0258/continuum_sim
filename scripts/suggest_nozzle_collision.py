"""Suggest primitive nozzle collision hints from engine mesh bboxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.check_engine_assets import collect_engine_scene_diagnostics  # noqa: E402


def build_nozzle_collision_hints(
    *,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    axis: str,
    radius_scale: float,
    enabled: bool,
    primary: str = "capsule",
) -> list[dict[str, object]]:
    """Build capsule and box primitive hints from a world-space bbox."""

    center = tuple((bbox_min[index] + bbox_max[index]) * 0.5 for index in range(3))
    size = tuple(bbox_max[index] - bbox_min[index] for index in range(3))
    axis_index = _axis_index(axis, size)
    radius = min(size[index] for index in range(3) if index != axis_index) * radius_scale
    capsule_length = size[axis_index] * 0.9
    start = list(center)
    end = list(center)
    start[axis_index] -= capsule_length * 0.5
    end[axis_index] += capsule_length * 0.5
    capsule = {
        "name": "nozzle_collision_capsule_hint",
        "type": "capsule",
        "enabled": enabled,
        "frame": "world",
        "fromto_m": [round(value, 12) for value in [*start, *end]],
        "radius_m": round(radius, 12),
        "rgba": [1.0, 0.2, 0.1, 0.35],
        "note": "Auto-generated initial capsule hint from collision mesh bbox; verify manually in viewer.",
    }
    box = {
        "name": "nozzle_collision_box_hint",
        "type": "box",
        "enabled": enabled,
        "frame": "world",
        "position_m": [round(value, 12) for value in center],
        "size_m": [round(value * 0.5, 12) for value in size],
        "rgba": [0.2, 0.8, 1.0, 0.25],
        "note": "BBox hint only; not recommended as final collision unless manually verified.",
    }
    return [box, capsule] if primary == "box" else [capsule, box]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        diagnostics = collect_engine_scene_diagnostics(args.config, root=args.root)
        source_report = _source_report(diagnostics, args.source)
        if source_report.bbox_min_world is None or source_report.bbox_max_world is None:
            raise ValueError(f"{args.source} mesh world bbox is unavailable.")
        hints = build_nozzle_collision_hints(
            bbox_min=source_report.bbox_min_world,
            bbox_max=source_report.bbox_max_world,
            axis=args.axis,
            radius_scale=args.radius_scale,
            enabled=args.enable_hint,
            primary=args.primitive,
        )
        _write_output_config(
            source_config=args.config,
            output_config=args.output_config,
            source=args.source,
            primitive=args.primitive,
            hints=hints,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"source mesh: {source_report.asset_name}")
    print(f"source bbox min: {_format_tuple(source_report.bbox_min_world)}")
    print(f"source bbox max: {_format_tuple(source_report.bbox_max_world)}")
    print(f"selected axis: {_axis_name(_axis_index(args.axis, _bbox_size(source_report)))}")
    print(f"generated primitive: {args.primitive}")
    output_config_text = args.output_config.as_posix()
    print(f"output config: {output_config_text}")
    print("warning: bbox-based primitive hints require manual viewer confirmation.")
    print(
        "next: python scripts/preview_engine_scene_mujoco.py "
        f"--config {output_config_text} --viewer --show-bbox --show-regions --show-axes "
        "--show-primitive-collision --show-disabled-hints"
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/scenes/engine_cleaning_aligned.yaml"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--source", choices=("collision", "visual"), default="collision")
    parser.add_argument("--primitive", choices=("capsule", "box"), default="capsule")
    parser.add_argument("--axis", choices=("longest", "x", "y", "z"), default="longest")
    parser.add_argument("--radius-scale", type=float, default=0.15)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/scenes/engine_cleaning_nozzle_collision.yaml"),
    )
    parser.add_argument("--enable-hint", action="store_true")
    return parser.parse_args(argv)


def _source_report(diagnostics, source: str):
    asset_name = "collision_mesh" if source == "collision" else "visual_mesh"
    report = next((report for report in diagnostics.asset_reports if report.asset_name == asset_name), None)
    if report is None:
        raise ValueError(f"Configured {asset_name} is unavailable.")
    return report


def _write_output_config(
    *,
    source_config: Path,
    output_config: Path,
    source: str,
    primitive: str,
    hints: list[dict[str, object]],
) -> None:
    source_path = source_config.resolve()
    output_path = output_config.resolve()
    if output_path == source_path:
        raise ValueError("--output-config must not overwrite the source config.")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["name"] = "engine_cleaning_nozzle_collision"
    metadata = dict(raw.get("metadata", {}))
    metadata.update(
        {
            "generated_from": source_config.as_posix(),
            "collision_hint_source": source,
            "collision_hint_primitive": primitive,
            "note": "Derived from aligned mesh bbox diagnostics; verify manually in viewer.",
        }
    )
    raw["metadata"] = metadata
    raw["primitive_collision_geoms"] = hints
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _axis_index(axis: str, size: tuple[float, float, float]) -> int:
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    return max(range(3), key=lambda index: size[index])


def _axis_name(axis_index: int) -> str:
    return ("x", "y", "z")[axis_index]


def _bbox_size(report) -> tuple[float, float, float]:
    if report.bbox_size_world is None:
        return (0.0, 0.0, 0.0)
    return report.bbox_size_world


def _format_tuple(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "<unavailable>"
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
