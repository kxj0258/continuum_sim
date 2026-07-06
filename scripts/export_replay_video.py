"""Export a saved rollout NPZ to a replay animation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.visualization.mujoco_video import save_replay_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--camera-lookat", type=float, nargs=3, default=None)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--camera-azimuth", type=float, default=None)
    parser.add_argument("--camera-elevation", type=float, default=None)
    parser.add_argument("--camera-follow", default="none")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = export_replay_video(
        result_npz_path=args.result_npz,
        output_path=args.output,
        scene_xml_path=args.scene_xml,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=args.stride,
        camera=_camera_from_args(args),
    )
    if path is None:
        print(f"video_export_failed: {args.output}", file=sys.stderr)
        return 1
    print(f"video_path: {path}")
    return 0


def export_replay_video(
    *,
    result_npz_path: Path,
    output_path: Path,
    scene_xml_path: Path | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    stride: int | None = None,
    camera: object | None = None,
) -> Path | None:
    result = _load_result_npz(result_npz_path)
    if scene_xml_path is not None:
        result.scene_xml_path = scene_xml_path.resolve()
    return save_replay_video(
        result,
        output_path,
        enabled=True,
        width=width,
        height=height,
        fps=fps,
        stride=stride,
        camera=camera,
    )


def _camera_from_args(args: argparse.Namespace) -> SimpleNamespace | None:
    if args.camera_lookat is None:
        return None
    if (
        args.camera_distance is None
        or args.camera_azimuth is None
        or args.camera_elevation is None
    ):
        raise ValueError("camera distance, azimuth, and elevation are required with lookat.")
    return SimpleNamespace(
        lookat=tuple(args.camera_lookat),
        distance=args.camera_distance,
        azimuth=args.camera_azimuth,
        elevation=args.camera_elevation,
        follow=args.camera_follow,
    )


def _load_result_npz(path: Path) -> SimpleNamespace:
    values: dict[str, object] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            values[key] = _decode_npz_value(data[key])
    return SimpleNamespace(**values)


def _decode_npz_value(value: np.ndarray) -> object:
    if value.shape == ():
        item = value.item()
        return str(item) if isinstance(item, np.str_) else item
    return value


if __name__ == "__main__":
    raise SystemExit(main())
