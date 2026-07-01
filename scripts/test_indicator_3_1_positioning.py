"""指标 3.1 定位误差验收脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continuum_sim.validation import configure_chinese_plot_font, positioning_metrics, write_markdown_report


THRESHOLD_M = 0.02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-npz", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/mujoco_navigation_rocket.yaml"))
    parser.add_argument("--mujoco-config", type=Path, default=Path("configs/mujoco.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/acceptance/indicator_3_1_positioning"))
    args = parser.parse_args()

    result = _load_or_run_navigation(args.result_npz, args.config, args.mujoco_config)
    error = np.asarray(result["error_norm"], dtype=float)
    metrics = positioning_metrics(error)
    plot_path = _save_curve(args.output_dir / "positioning_error.png", error, THRESHOLD_M, "error [m]")
    report = write_markdown_report(
        args.output_dir / "report.md",
        title="指标 3.1 定位误差验收",
        metrics=metrics,
        thresholds={"steady_error_m": THRESHOLD_M},
        artifacts=[plot_path],
        notes=["通过判据：末端三维欧氏稳态误差不超过 2 cm。"],
    )
    print(report)


def _load_or_run_navigation(result_npz: Path | None, config: Path, mujoco_config: Path) -> dict[str, np.ndarray]:
    if result_npz is not None:
        return dict(np.load(result_npz, allow_pickle=True))
    print("未提供 --result-npz，将启动完整 MuJoCo navigation 仿真；推荐正式验收时先用 scenario artifacts 生成 result.npz。")
    from continuum_sim.runtime import run_mujoco_navigation

    result = run_mujoco_navigation(config, mujoco_config, show=False)
    return {"error_norm": result.error_norm}


def _save_curve(path: Path, values: np.ndarray, threshold: float, ylabel: str) -> Path:
    import matplotlib.pyplot as plt

    configure_chinese_plot_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(values)
    axis.axhline(threshold, color="tab:red", linestyle="--")
    axis.set_xlabel("采样点")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
