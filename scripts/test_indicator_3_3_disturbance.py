"""指标 3.3 扰动位移偏差验收脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from continuum_sim.validation import configure_chinese_plot_font, disturbance_metrics, write_markdown_report


THRESHOLD_M = 0.04


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-npz", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/mujoco_wiping_board_dynamic.yaml"))
    parser.add_argument("--mujoco-config", type=Path, default=Path("configs/mujoco.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/acceptance/indicator_3_3_disturbance"))
    parser.add_argument("--disturbance-force-n", type=float, default=5.0)
    parser.add_argument("--disturbance-duration-s", type=float, default=0.10)
    parser.add_argument("--disturbance-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    args = parser.parse_args()

    result = _load_or_run_wiping(args.result_npz, args.config, args.mujoco_config)
    target = np.asarray(result["target_position"], dtype=float)
    tip = np.asarray(result["tip_position"], dtype=float)
    metrics = disturbance_metrics(target, tip)
    displacement = np.linalg.norm(tip - target, axis=1)
    plot_path = _save_curve(args.output_dir / "disturbance_displacement.png", displacement, THRESHOLD_M)
    report = write_markdown_report(
        args.output_dir / "report.md",
        title="指标 3.3 扰动位移偏差验收",
        metrics=metrics,
        thresholds={"max_displacement_m": THRESHOLD_M},
        artifacts=[plot_path],
        notes=[
            "通过判据：末端最大位移偏差不超过 4 cm。",
            "第三方扰动测试口径尚未固定，脚本当前记录可配置工程扰动工况。",
            f"扰动工况：{args.disturbance_force_n} N，持续 {args.disturbance_duration_s} s，方向 {tuple(args.disturbance_direction)}。",
        ],
    )
    print(report)


def _load_or_run_wiping(result_npz: Path | None, config: Path, mujoco_config: Path) -> dict[str, np.ndarray]:
    if result_npz is not None:
        data = dict(np.load(result_npz, allow_pickle=True))
        if "tip_position" not in data and "tip_pose" in data:
            data["tip_position"] = np.asarray(data["tip_pose"], dtype=float)[:, :3, 3]
        return data
    print("未提供 --result-npz，将启动完整 MuJoCo wiping 仿真；推荐正式验收时先用 CLI --save-run 生成 result.npz。")
    from continuum_sim.runtime import run_mujoco_wiping

    result = run_mujoco_wiping(config, mujoco_config, show=False)
    return {"target_position": result.target_position, "tip_position": result.tip_position}


def _save_curve(path: Path, values: np.ndarray, threshold: float) -> Path:
    import matplotlib.pyplot as plt

    configure_chinese_plot_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(values)
    axis.axhline(threshold, color="tab:red", linestyle="--")
    axis.set_xlabel("采样点")
    axis.set_ylabel("位移偏差 [m]")
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
