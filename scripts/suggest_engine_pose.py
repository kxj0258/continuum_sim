"""Suggest safe engine pose offsets from the configured visual mesh bbox."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.check_engine_assets import collect_engine_scene_diagnostics  # noqa: E402


@dataclass(frozen=True)
class PoseSuggestion:
    """Pose suggestion derived from the visual mesh bbox."""

    config_path: Path
    current_pose_position: tuple[float, float, float]
    bbox_min_scaled: tuple[float, float, float]
    bbox_max_scaled: tuple[float, float, float]
    bbox_size_scaled: tuple[float, float, float]
    bbox_center_scaled: tuple[float, float, float]
    bbox_min_world: tuple[float, float, float]
    bbox_max_world: tuple[float, float, float]
    bbox_center_world: tuple[float, float, float]
    recenter_pose_position: tuple[float, float, float]
    grounded_pose_position: tuple[float, float, float]


def compute_pose_suggestion(
    config_path: str | Path = Path("configs/scenes/engine_cleaning.yaml"),
    *,
    root: str | Path | None = None,
    target_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ground_z: float = 0.0,
) -> PoseSuggestion:
    """Compute candidate pose positions without modifying the source config."""

    diagnostics = collect_engine_scene_diagnostics(config_path, root=root, strict_assets=False)
    visual = next((report for report in diagnostics.asset_reports if report.asset_name == "visual_mesh"), None)
    if visual is None or visual.bbox_center_scaled is None:
        raise ValueError("visual_mesh bbox is unavailable; cannot suggest an engine pose.")

    recenter_pose = tuple(
        target_center[index] - visual.bbox_center_scaled[index] for index in range(3)
    )
    grounded_pose = (
        -visual.bbox_center_scaled[0],
        -visual.bbox_center_scaled[1],
        ground_z - visual.bbox_min_scaled[2],
    )
    return PoseSuggestion(
        config_path=diagnostics.config_path,
        current_pose_position=diagnostics.pose_position,
        bbox_min_scaled=visual.bbox_min_scaled,
        bbox_max_scaled=visual.bbox_max_scaled,
        bbox_size_scaled=visual.bbox_size_scaled,
        bbox_center_scaled=visual.bbox_center_scaled,
        bbox_min_world=visual.bbox_min_world,
        bbox_max_world=visual.bbox_max_world,
        bbox_center_world=visual.bbox_center_world,
        recenter_pose_position=recenter_pose,  # type: ignore[arg-type]
        grounded_pose_position=grounded_pose,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target_center = tuple(float(value) for value in args.target_center)
        suggestion = compute_pose_suggestion(
            args.config,
            root=args.root,
            target_center=target_center,  # type: ignore[arg-type]
            ground_z=float(args.ground_z),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_suggestion(suggestion, mode=args.mode)
    if args.write_suggested_config is not None:
        try:
            pose_position = _pose_for_mode(suggestion, args.mode)
            _write_suggested_config(
                source_path=Path(args.config),
                output_path=args.write_suggested_config,
                pose_position=pose_position,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote suggested config: {args.write_suggested_config}")
        print(
            "Preview it with: "
            f"python scripts/preview_engine_scene_mujoco.py --config {args.write_suggested_config} --headless-check"
        )
    if args.write_aligned_config is not None:
        try:
            _write_aligned_config(
                source_path=Path(args.config),
                output_path=args.write_aligned_config,
                pose_position=suggestion.grounded_pose_position,
                generated_from=str(args.config),
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote aligned config: {args.write_aligned_config}")
        print(
            "Preview it with: "
            f"python scripts/preview_engine_scene_mujoco.py --config {args.write_aligned_config} "
            "--headless-check"
        )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenes/engine_cleaning.yaml"),
        help="Engine scene YAML config to inspect.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root used when re-resolving relative configured asset paths.",
    )
    parser.add_argument(
        "--mode",
        choices=("recenter", "grounded"),
        default="recenter",
        help="Pose suggestion mode to print or write for --write-suggested-config.",
    )
    parser.add_argument(
        "--target-center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Target world point for aligning the visual mesh center.",
    )
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.0,
        help="Ground plane z value for the grounded pose suggestion.",
    )
    parser.add_argument(
        "--write-suggested-config",
        type=Path,
        default=None,
        help="Write a new candidate YAML with the recenter pose. The source config is not modified.",
    )
    parser.add_argument(
        "--write-aligned-config",
        type=Path,
        default=None,
        help="Write a derived aligned YAML using the grounded pose and primitive collision hints.",
    )
    return parser.parse_args(argv)


def _print_suggestion(suggestion: PoseSuggestion, *, mode: str) -> None:
    print(f"config: {suggestion.config_path}")
    print(f"mode: {mode}")
    print(f"Current pose.position_m: {_format_tuple(suggestion.current_pose_position)}")
    print(f"Scaled bbox min: {_format_tuple(suggestion.bbox_min_scaled)}")
    print(f"Scaled bbox max: {_format_tuple(suggestion.bbox_max_scaled)}")
    print(f"Scaled bbox size: {_format_tuple(suggestion.bbox_size_scaled)}")
    print(f"Scaled bbox center: {_format_tuple(suggestion.bbox_center_scaled)}")
    print(f"World bbox min: {_format_tuple(suggestion.bbox_min_world)}")
    print(f"World bbox max: {_format_tuple(suggestion.bbox_max_world)}")
    print(f"World bbox center: {_format_tuple(suggestion.bbox_center_world)}")
    print(f"Suggested recenter pose.position_m: {_format_tuple(suggestion.recenter_pose_position)}")
    print(f"Suggested grounded pose.position_m: {_format_tuple(suggestion.grounded_pose_position)}")
    print(f"Selected suggested pose.position_m: {_format_tuple(_pose_for_mode(suggestion, mode))}")


def _pose_for_mode(suggestion: PoseSuggestion, mode: str) -> tuple[float, float, float]:
    if mode == "grounded":
        return suggestion.grounded_pose_position
    return suggestion.recenter_pose_position


def _write_suggested_config(
    *,
    source_path: Path,
    output_path: Path,
    pose_position: tuple[float, float, float],
) -> None:
    source = source_path.resolve()
    output = output_path.resolve()
    if output == source:
        raise ValueError("--write-suggested-config must not overwrite the source config.")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["engine"]["pose"]["position_m"] = [round(value, 12) for value in pose_position]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _write_aligned_config(
    *,
    source_path: Path,
    output_path: Path,
    pose_position: tuple[float, float, float],
    generated_from: str,
) -> None:
    source = source_path.resolve()
    output = output_path.resolve()
    if output == source:
        raise ValueError("--write-aligned-config must not overwrite the source config.")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["name"] = "engine_cleaning_aligned"
    raw["metadata"] = {
        "generated_from": generated_from,
        "alignment_mode": "grounded",
        "note": "Derived from mesh bbox diagnostics; verify in viewer.",
    }
    raw["engine"]["pose"]["position_m"] = [round(value, 12) for value in pose_position]
    raw.setdefault("primitive_collision_geoms", _default_primitive_collision_hints())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _default_primitive_collision_hints() -> list[dict[str, object]]:
    return [
        {
            "name": "nozzle_collision_hint",
            "type": "capsule",
            "enabled": False,
            "position_m": [0.0, 0.0, 0.4],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "radius_m": 0.03,
            "length_m": 0.4,
            "note": "Placeholder only. Needs manual alignment to the nozzle.",
        }
    ]


def _format_tuple(values: tuple[float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
