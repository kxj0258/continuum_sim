"""验收脚本使用的轻量指标与报告工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def configure_chinese_plot_font() -> None:
    """为 Matplotlib 图表选择可用中文字体。"""

    import matplotlib
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in (
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def positioning_metrics(error_norm_m: np.ndarray, *, steady_fraction: float = 0.2) -> dict[str, float]:
    """计算最终误差、最大误差和稳态定位误差。"""

    error = _series(error_norm_m, "error_norm_m")
    if steady_fraction <= 0.0 or steady_fraction > 1.0:
        raise ValueError("steady_fraction must be in (0, 1].")
    window = max(1, int(np.ceil(error.size * steady_fraction)))
    steady = error[-window:]
    return {
        "final_error_m": float(error[-1]),
        "max_error_m": float(np.max(error)),
        "steady_error_m": float(np.mean(steady)),
    }


def disturbance_metrics(target_position: np.ndarray, tip_position: np.ndarray) -> dict[str, float]:
    """计算目标位置与实测末端位置之间的位移偏差指标。"""

    target = _positions(target_position, "target_position")
    tip = _positions(tip_position, "tip_position")
    if target.shape != tip.shape:
        raise ValueError(f"target_position and tip_position shapes differ: {target.shape} vs {tip.shape}.")
    displacement = np.linalg.norm(tip - target, axis=1)
    return {
        "max_displacement_m": float(np.max(displacement)),
        "final_displacement_m": float(displacement[-1]),
        "mean_displacement_m": float(np.mean(displacement)),
    }


def force_tracking_metrics(
    normal_force_n: np.ndarray,
    *,
    target_force_n: float,
) -> dict[str, float]:
    """计算接触力跟踪指标。"""

    force = _series(normal_force_n, "normal_force_n")
    error = force - float(target_force_n)
    return {
        "target_force_n": float(target_force_n),
        "mean_error_n": float(np.mean(error)),
        "mean_abs_error_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(error**2))),
    }


def acceptance_passed(value: float, threshold: float) -> bool:
    """判断指标是否满足上界阈值。"""

    return bool(float(value) <= float(threshold))


def write_markdown_report(
    path: str | Path,
    *,
    title: str,
    metrics: dict[str, float],
    thresholds: dict[str, float],
    artifacts: list[Path],
    notes: list[str],
) -> Path:
    """写入带盖章预留区的简洁 Markdown 报告。"""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## 测试指标",
        "| 指标 | 数值 | 阈值 | 结论 |",
        "|--------|-------|-----------|------|",
    ]
    for name, value in metrics.items():
        threshold = thresholds.get(name)
        passed = "" if threshold is None else ("通过" if acceptance_passed(value, threshold) else "未通过")
        threshold_text = "" if threshold is None else f"{threshold:.6g}"
        lines.append(f"| `{name}` | {value:.6g} | {threshold_text} | {passed} |")
    lines.extend(["", "## 输出文件"])
    lines.extend(f"- `{artifact}`" for artifact in artifacts)
    lines.extend(["", "## 说明"])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## CNAS/CMA 盖章区",
            "",
            "| 检测机构 | 盖章 | 日期 |",
            "|-------------|-------|------|",
            "|             |       |      |",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _series(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array, got {array.shape}.")
    return array


def _positions(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}.")
    return array
