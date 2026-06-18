from pathlib import Path

import numpy as np

from continuum_sim.validation.acceptance import (
    acceptance_passed,
    disturbance_metrics,
    force_tracking_metrics,
    positioning_metrics,
    write_markdown_report,
)


def test_acceptance_metrics_compute_thresholds_and_report(tmp_path: Path) -> None:
    error = np.array([0.05, 0.03, 0.015, 0.010], dtype=float)
    positioning = positioning_metrics(error, steady_fraction=0.5)
    disturbance = disturbance_metrics(
        np.zeros((3, 3), dtype=float),
        np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float),
    )
    force = force_tracking_metrics(np.array([1.5, 1.0, 2.0]), target_force_n=1.5)

    assert positioning["steady_error_m"] == 0.0125
    assert disturbance["max_displacement_m"] == 0.02
    assert force["rmse_n"] > 0.0
    assert acceptance_passed(positioning["steady_error_m"], 0.02) is True

    report = write_markdown_report(
        tmp_path / "report.md",
        title="指标测试",
        metrics={"steady_error_m": positioning["steady_error_m"]},
        thresholds={"steady_error_m": 0.02},
        artifacts=[tmp_path / "plot.png"],
        notes=["工程估计值"],
    )

    text = report.read_text(encoding="utf-8")
    assert "指标测试" in text
    assert "CNAS/CMA 盖章区" in text
