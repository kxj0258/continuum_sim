"""Probe whether MuJoCo can create an offscreen renderer for a configured XML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from continuum_sim.config import load_mujoco_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mujoco.yaml"),
        help="Path to the MuJoCo backend YAML config.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Optional XML path override to probe instead of the configured model XML.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional offscreen width override.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional offscreen height override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = probe_mujoco_offscreen_renderer(
            config_path=args.config,
            xml_path=args.xml,
            width=args.width,
            height=args.height,
        )
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"xml_path: {summary['xml_path']}")
    print(f"offscreen_width: {summary['width']}")
    print(f"offscreen_height: {summary['height']}")
    print(f"frame_shape: {summary['frame_shape']}")
    return 0


def probe_mujoco_offscreen_renderer(
    *,
    config_path: Path,
    xml_path: Path | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, object]:
    config = load_mujoco_config(
        config_path,
        require_xml=False,
        require_tendon_xml=False,
        require_visual_meshes=False,
    )
    probe_xml_path = _probe_xml_path(config, xml_path)
    probe_width = _positive_dimension(
        config.rendering.offscreen_width if width is None else width,
        "width",
    )
    probe_height = _positive_dimension(
        config.rendering.offscreen_height if height is None else height,
        "height",
    )

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"MuJoCo is not installed: {exc}") from exc

    renderer = None
    stage = "loading MuJoCo XML"
    try:
        model = mujoco.MjModel.from_xml_path(str(probe_xml_path))
        stage = "creating MuJoCo data"
        data = mujoco.MjData(model)
        stage = "creating MuJoCo renderer"
        renderer = mujoco.Renderer(model, height=probe_height, width=probe_width)
        stage = "rendering frame 0"
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frame = renderer.render().copy()
    except Exception as exc:  # noqa: BLE001 - diagnostic should surface native failures.
        raise RuntimeError(
            f"offscreen renderer probe failed during {stage}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()

    return {
        "xml_path": probe_xml_path,
        "width": probe_width,
        "height": probe_height,
        "frame_shape": tuple(int(value) for value in frame.shape),
    }


def _probe_xml_path(config, xml_path: Path | None) -> Path:
    if xml_path is not None:
        candidate = Path(xml_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"MuJoCo XML file does not exist: {candidate}")
        return candidate
    for candidate in (
        config.tendon_generated_xml_path,
        config.tendon_xml_path,
        config.generated_xml_path,
        config.xml_path,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No MuJoCo XML file is available to probe. Pass --xml or generate one of the "
        "configured XML assets first."
    )


def _positive_dimension(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
