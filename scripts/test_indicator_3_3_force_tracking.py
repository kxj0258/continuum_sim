"""指标 3.3 接触力跟踪验收脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continuum_sim.validation import configure_chinese_plot_font, force_tracking_metrics, write_markdown_report


THRESHOLD_N = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-npz", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/mujoco_wiping_board_dynamic.yaml"))
    parser.add_argument("--mujoco-config", type=Path, default=Path("configs/mujoco.yaml"))
    parser.add_argument("--target-force-n", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=Path("output/acceptance/indicator_3_3_force_tracking"))
    args = parser.parse_args()

    result = _load_or_run_wiping(args.result_npz, args.config, args.mujoco_config)
    force = np.asarray(result["normal_force_n"], dtype=float)
    metrics = force_tracking_metrics(force, target_force_n=args.target_force_n)
    plot_path = _save_force_plot(args.output_dir / "force_tracking.png", force, args.target_force_n)
    report = write_markdown_report(
        args.output_dir / "report.md",
        title="指标 3.3 接触力跟踪验收",
        metrics=metrics,
        thresholds={"rmse_n": THRESHOLD_N},
        artifacts=[plot_path],
        notes=["通过判据：接触力 RMSE 不超过 1 N。"],
    )
    print(report)


def _load_or_run_wiping(result_npz: Path | None, config: Path, mujoco_config: Path) -> dict[str, np.ndarray]:
    if result_npz is not None:
        return dict(np.load(result_npz, allow_pickle=True))
    print("未提供 --result-npz，将启动完整 MuJoCo wiping 仿真；动态阻抗配置会较慢，推荐正式验收时先用 CLI --save-run 生成 result.npz。")
    from continuum_sim.runtime import run_mujoco_wiping

    result = run_mujoco_wiping(config, mujoco_config, show=False)
    return {"normal_force_n": result.normal_force_n}


def _save_force_plot(path: Path, force: np.ndarray, target_force: float) -> Path:
    import matplotlib.pyplot as plt

    configure_chinese_plot_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(force, label="实际法向力")
    axis.axhline(target_force, color="tab:green", linestyle="--", label="目标力")
    axis.set_xlabel("采样点")
    axis.set_ylabel("力 [N]")
    axis.grid(True, alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
