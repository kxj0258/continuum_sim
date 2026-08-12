"""Monte Carlo workspace analysis for continuum-arm encoder accuracy.

By default the script samples 10,000 project-valid continuum poses, evaluates
all 64 sign corners of each encoder error bound, writes summary/sample data,
and generates TCP-workspace heatmaps. The result is a sampled-workspace bound;
it is not a mathematical proof for unsampled poses or unmodelled error sources.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from cal_accuracy import (
        DEFAULT_ACCURACY_DEG,
        DEFAULT_ARM_CONFIG,
        DEFAULT_SCENARIO_CONFIG,
        DEFAULT_TOOL_CONFIG,
        ENCODER_CHANNELS,
        SIGNS,
        AccuracyModel,
        batch_encoder_poses,
        load_accuracy_model,
    )
except ModuleNotFoundError:  # Support ``python -m scripts.cal_accuracy_workspace``.
    from scripts.cal_accuracy import (
        DEFAULT_ACCURACY_DEG,
        DEFAULT_ARM_CONFIG,
        DEFAULT_SCENARIO_CONFIG,
        DEFAULT_TOOL_CONFIG,
        ENCODER_CHANNELS,
        SIGNS,
        AccuracyModel,
        batch_encoder_poses,
        load_accuracy_model,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "accuracy_workspace"


@dataclass(frozen=True)
class WorkspaceAccuracyResult:
    accuracy_deg: float
    position_error_mm: np.ndarray
    orientation_error_deg: np.ndarray

    @property
    def maximum_position_error_mm(self) -> float:
        return float(np.max(self.position_error_mm))

    @property
    def mean_position_error_mm(self) -> float:
        return float(np.mean(self.position_error_mm))

    @property
    def p95_position_error_mm(self) -> float:
        return float(np.percentile(self.position_error_mm, 95.0))


def sample_valid_workspace(
    model: AccuracyModel,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-sample angle- and physical-tendon-valid encoder poses."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    accepted: list[np.ndarray] = []
    remaining = sample_count
    attempts = 0
    while remaining:
        batch_size = max(remaining * 2, 1024)
        candidate = rng.uniform(
            model.encoder_angle_lower_deg,
            model.encoder_angle_upper_deg,
            size=(batch_size, ENCODER_CHANNELS),
        )
        bending = np.deg2rad(candidate) / model.flexure_lengths_m
        tendon = bending @ model.bending_model.coupling_matrix.T
        limits = model.arm.limits
        valid = np.all(
            (tendon >= limits.tendon_displacement_min_m)
            & (tendon <= limits.tendon_displacement_max_m),
            axis=1,
        )
        selected = candidate[valid][:remaining]
        accepted.append(selected)
        remaining -= selected.shape[0]
        attempts += batch_size
        if attempts > 1000 * sample_count and remaining:
            raise RuntimeError("Unable to sample enough poses within project limits.")
    return np.vstack(accepted)


def analyze_workspace(
    theta_true_deg: np.ndarray,
    accuracies_deg: Sequence[float],
    model: AccuracyModel,
    *,
    error_model: str = "corners",
    rng: np.random.Generator | None = None,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, tuple[WorkspaceAccuracyResult, ...]]:
    """Return reference TCP poses and per-pose error statistics."""

    theta = np.asarray(theta_true_deg, dtype=float)
    reference_position, reference_rotation = batch_encoder_poses(theta, model)
    rng = rng or np.random.default_rng()
    results = []
    for accuracy in accuracies_deg:
        accuracy = float(accuracy)
        if not np.isfinite(accuracy) or accuracy < 0.0:
            raise ValueError("accuracies_deg values must be finite and non-negative.")
        max_position = np.zeros(theta.shape[0], dtype=float)
        max_orientation = np.zeros(theta.shape[0], dtype=float)
        if error_model == "uniform":
            errors = rng.uniform(-accuracy, accuracy, size=theta.shape)
            measured_position, measured_rotation = batch_encoder_poses(theta + errors, model)
            max_position[:] = 1000.0 * np.linalg.norm(
                measured_position - reference_position, axis=1
            )
            max_orientation[:] = _batch_orientation_error_deg(
                reference_rotation, measured_rotation
            )
        elif error_model == "corners":
            for first in range(0, theta.shape[0], chunk_size):
                stop = min(first + chunk_size, theta.shape[0])
                true_chunk = theta[first:stop]
                measured_theta = (
                    true_chunk[:, None, :] + accuracy * SIGNS[None, :, :]
                ).reshape(-1, ENCODER_CHANNELS)
                measured_position, measured_rotation = batch_encoder_poses(
                    measured_theta, model
                )
                shape = (stop - first, SIGNS.shape[0])
                position_error = 1000.0 * np.linalg.norm(
                    measured_position.reshape(*shape, 3)
                    - reference_position[first:stop, None, :],
                    axis=2,
                )
                orientation_error = _batch_orientation_error_deg(
                    np.repeat(reference_rotation[first:stop], SIGNS.shape[0], axis=0),
                    measured_rotation,
                ).reshape(shape)
                max_position[first:stop] = np.max(position_error, axis=1)
                max_orientation[first:stop] = np.max(orientation_error, axis=1)
        else:
            raise ValueError("error_model must be 'corners' or 'uniform'.")
        results.append(
            WorkspaceAccuracyResult(
                accuracy_deg=accuracy,
                position_error_mm=max_position,
                orientation_error_deg=max_orientation,
            )
        )
    return reference_position, reference_rotation, tuple(results)


def _batch_orientation_error_deg(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    relative = np.einsum("nji,njk->nik", reference, measured)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def write_outputs(
    output_dir: Path,
    theta_deg: np.ndarray,
    tcp_position_m: np.ndarray,
    results: tuple[WorkspaceAccuracyResult, ...],
    *,
    threshold_mm: float,
    seed: int,
    error_model: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "workspace_accuracy_summary.csv"
    samples_path = output_dir / "workspace_accuracy_samples.npz"
    heatmap_path = output_dir / "workspace_accuracy_heatmap.png"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "accuracy_deg",
                "sample_count",
                "mean_position_error_mm",
                "p95_position_error_mm",
                "max_position_error_mm",
                "mean_orientation_error_deg",
                "max_orientation_error_deg",
                f"fraction_le_{threshold_mm:g}_mm",
                "sampled_workspace_pass",
                "error_model",
                "seed",
            )
        )
        for result in results:
            passing = result.position_error_mm <= threshold_mm
            writer.writerow(
                (
                    result.accuracy_deg,
                    result.position_error_mm.size,
                    result.mean_position_error_mm,
                    result.p95_position_error_mm,
                    result.maximum_position_error_mm,
                    float(np.mean(result.orientation_error_deg)),
                    float(np.max(result.orientation_error_deg)),
                    float(np.mean(passing)),
                    result.maximum_position_error_mm <= threshold_mm,
                    error_model,
                    seed,
                )
            )
    np.savez_compressed(
        samples_path,
        theta_true_deg=theta_deg,
        tcp_position_m=tcp_position_m,
        accuracies_deg=np.asarray([result.accuracy_deg for result in results]),
        position_error_mm=np.vstack([result.position_error_mm for result in results]),
        orientation_error_deg=np.vstack(
            [result.orientation_error_deg for result in results]
        ),
        threshold_mm=float(threshold_mm),
        seed=int(seed),
        error_model=error_model,
    )
    _plot_heatmaps(heatmap_path, tcp_position_m, results, threshold_mm)
    return summary_path, samples_path, heatmap_path


def _plot_heatmaps(
    path: Path,
    tcp_position_m: np.ndarray,
    results: tuple[WorkspaceAccuracyResult, ...],
    threshold_mm: float,
) -> None:
    import matplotlib.pyplot as plt

    count = len(results)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.4 * rows), squeeze=False)
    x_mm = 1000.0 * tcp_position_m[:, 0]
    y_mm = 1000.0 * tcp_position_m[:, 1]
    extent = (x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max())
    for axis, result in zip(axes.flat, results, strict=False):
        weighted, x_edges, y_edges = np.histogram2d(
            x_mm, y_mm, bins=45, weights=result.position_error_mm
        )
        counts, _, _ = np.histogram2d(x_mm, y_mm, bins=(x_edges, y_edges))
        mean_error = np.divide(
            weighted,
            counts,
            out=np.full_like(weighted, np.nan),
            where=counts > 0,
        )
        image = axis.imshow(
            mean_error.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="turbo",
            vmin=0.0,
            vmax=max(threshold_mm, float(np.nanmax(mean_error))),
        )
        axis.set_title(
            f"±{result.accuracy_deg:g}° | max={result.maximum_position_error_mm:.3f} mm"
        )
        axis.set_xlabel("TCP x [mm]")
        axis.set_ylabel("TCP y [mm]")
        figure.colorbar(image, ax=axis, label="mean bounded error [mm]")
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    figure.suptitle("Encoder accuracy over sampled TCP workspace")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--accuracy-deg", nargs="+", type=float, default=DEFAULT_ACCURACY_DEG)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--error-model", choices=("corners", "uniform"), default="corners")
    parser.add_argument("--threshold-mm", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arm-config", type=Path, default=DEFAULT_ARM_CONFIG)
    parser.add_argument("--tool-config", type=Path, default=DEFAULT_TOOL_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.threshold_mm <= 0.0:
        raise ValueError("--threshold-mm must be positive.")
    model = load_accuracy_model(args.arm_config, args.tool_config, args.scenario_config)
    rng = np.random.default_rng(args.seed)
    theta = sample_valid_workspace(model, args.samples, rng)
    tcp_position, _, results = analyze_workspace(
        theta,
        args.accuracy_deg,
        model,
        error_model=args.error_model,
        rng=rng,
    )
    paths = write_outputs(
        args.output_dir.resolve(),
        theta,
        tcp_position,
        results,
        threshold_mm=args.threshold_mm,
        seed=args.seed,
        error_model=args.error_model,
    )
    print(
        f"samples={args.samples} | error_model={args.error_model} | "
        f"workspace={model.encoder_angle_lower_deg[0]:.1f}..{model.encoder_angle_upper_deg[0]:.1f} deg"
    )
    for result in results:
        status = "PASS" if result.maximum_position_error_mm <= args.threshold_mm else "FAIL"
        print(
            f"accuracy=+/-{result.accuracy_deg:g} deg | "
            f"mean={result.mean_position_error_mm:.3f} mm | "
            f"p95={result.p95_position_error_mm:.3f} mm | "
            f"max={result.maximum_position_error_mm:.3f} mm | "
            f"sampled <= {args.threshold_mm:g} mm: {status}"
        )
    print("outputs:")
    for path in paths:
        print(path)
    print("Note: PASS applies only to sampled poses and the configured error model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
