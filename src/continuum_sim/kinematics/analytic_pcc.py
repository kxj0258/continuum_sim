"""Analytic PCC position Jacobians.

The PCC state of each segment is ``[kx, ky, eps]``.  The formulas here match
``constant_curvature_transform`` and differentiate the same SE(3) exponential
used by the forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuum_sim.model.robot_params import (
    PCC_VALUES_PER_SEGMENT,
    ThreeSegmentRobotParams,
)
from continuum_sim.utils.math_utils import make_transform, skew


@dataclass(frozen=True)
class SegmentTransformDerivative:
    """One PCC segment transform and derivatives with respect to local q."""

    transform: np.ndarray
    derivatives: tuple[np.ndarray, np.ndarray, np.ndarray]


def analytic_position_jacobian(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    *,
    curvature_tol: float = 1.0e-7,
) -> np.ndarray:
    """Return ``d tip_position / d q`` for the full PCC arm."""

    q_segments = _as_segment_q(q, params)
    segments = [
        segment_transform_with_derivatives(
            segment_q,
            segment.length,
            curvature_tol=curvature_tol,
        )
        for segment_q, segment in zip(q_segments, params.segments, strict=True)
    ]
    return _point_jacobian_from_segment_derivatives(
        segments,
        params,
        active_segment=params.segment_count - 1,
        point_transform=np.eye(4, dtype=float),
        include_active_segment_full=True,
    )


def analytic_bending_position_jacobian(
    q: np.ndarray,
    params: ThreeSegmentRobotParams,
    selection_matrix: np.ndarray,
    *,
    curvature_tol: float = 1.0e-7,
) -> np.ndarray:
    """Return tip-position Jacobian with respect to bending coordinates."""

    selection = np.asarray(selection_matrix, dtype=float)
    if selection.shape[0] != params.q_size:
        raise ValueError(
            f"selection_matrix must have {params.q_size} rows, got {selection.shape}."
        )
    return analytic_position_jacobian(
        q,
        params,
        curvature_tol=curvature_tol,
    ) @ selection


def analytic_centerline_point_jacobian(
    q: np.ndarray,
    centerline_index: int,
    params: ThreeSegmentRobotParams,
    *,
    samples_per_segment: int = 6,
    curvature_tol: float = 1.0e-7,
) -> np.ndarray:
    """Return ``d centerline_point / d q`` for one sampled point."""

    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")
    total_points = params.segment_count * (samples_per_segment - 1) + 1
    if centerline_index < 0 or centerline_index >= total_points:
        raise ValueError("centerline_index is outside the sampled centerline.")

    q_segments = _as_segment_q(q, params)
    segment_index, local_sample = _centerline_sample_location(
        centerline_index,
        samples_per_segment,
    )
    partial_length = (
        params.segments[segment_index].length
        * float(local_sample)
        / float(samples_per_segment - 1)
    )
    full_segments = [
        segment_transform_with_derivatives(
            segment_q,
            segment.length,
            curvature_tol=curvature_tol,
        )
        for segment_q, segment in zip(q_segments, params.segments, strict=True)
    ]
    point_segment = segment_transform_with_derivatives(
        q_segments[segment_index],
        partial_length,
        curvature_tol=curvature_tol,
    )
    return _point_jacobian_from_segment_derivatives(
        full_segments,
        params,
        active_segment=segment_index,
        point_transform=point_segment.transform,
        point_derivatives=point_segment.derivatives,
        include_active_segment_full=False,
    )


def analytic_centerline_point_bending_jacobian(
    q: np.ndarray,
    centerline_index: int,
    params: ThreeSegmentRobotParams,
    selection_matrix: np.ndarray,
    *,
    samples_per_segment: int = 6,
    curvature_tol: float = 1.0e-7,
) -> np.ndarray:
    """Return centerline-point Jacobian with respect to bending coordinates."""

    selection = np.asarray(selection_matrix, dtype=float)
    if selection.shape[0] != params.q_size:
        raise ValueError(
            f"selection_matrix must have {params.q_size} rows, got {selection.shape}."
        )
    return analytic_centerline_point_jacobian(
        q,
        centerline_index,
        params,
        samples_per_segment=samples_per_segment,
        curvature_tol=curvature_tol,
    ) @ selection


def segment_transform_with_derivatives(
    q_segment: np.ndarray,
    length: float,
    *,
    curvature_tol: float = 1.0e-7,
) -> SegmentTransformDerivative:
    """Return one segment transform and ``dT/d[kx, ky, eps]``."""

    kx, ky, eps = _as_segment_vector(q_segment)
    arc_length = float(length) * (1.0 + eps)
    if arc_length < 0.0:
        raise ValueError("Segment arc length must not be negative.")
    if arc_length == 0.0:
        transform = np.eye(4, dtype=float)
        zero = np.zeros((4, 4), dtype=float)
        return SegmentTransformDerivative(transform, (zero, zero, zero))

    omega = np.array([-ky, kx, 0.0], dtype=float)
    omega_hat = skew(omega)
    omega_hat_sq = omega_hat @ omega_hat
    kappa = float(np.linalg.norm(omega))
    coeffs = _curvature_coefficients(kappa, arc_length, curvature_tol)

    rotation = (
        np.eye(3, dtype=float)
        + coeffs.a * omega_hat
        + coeffs.b * omega_hat_sq
    )
    translation_matrix = (
        arc_length * np.eye(3, dtype=float)
        + coeffs.b * omega_hat
        + coeffs.d * omega_hat_sq
    )
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    translation = translation_matrix @ ez
    transform = make_transform(rotation, translation)

    derivatives = tuple(
        _segment_transform_derivative(
            omega_hat,
            omega_hat_sq,
            coeffs,
            kappa,
            arc_length,
            length,
            parameter,
            curvature_tol,
        )
        for parameter in ("kx", "ky", "eps")
    )
    return SegmentTransformDerivative(transform, derivatives)


@dataclass(frozen=True)
class _CurvatureCoefficients:
    a: float
    b: float
    d: float
    da_dkappa: float
    db_dkappa: float
    dd_dkappa: float
    da_ds: float
    db_ds: float
    dd_ds: float


def _curvature_coefficients(
    kappa: float,
    arc_length: float,
    curvature_tol: float,
) -> _CurvatureCoefficients:
    s = float(arc_length)
    k = float(kappa)
    if k < curvature_tol:
        k2 = k * k
        k3 = k2 * k
        k5 = k3 * k2
        return _CurvatureCoefficients(
            a=s - k2 * s**3 / 6.0 + k2 * k2 * s**5 / 120.0,
            b=s**2 / 2.0 - k2 * s**4 / 24.0 + k2 * k2 * s**6 / 720.0,
            d=s**3 / 6.0 - k2 * s**5 / 120.0 + k2 * k2 * s**7 / 5040.0,
            da_dkappa=-k * s**3 / 3.0 + k3 * s**5 / 30.0 - k5 * s**7 / 840.0,
            db_dkappa=-k * s**4 / 12.0 + k3 * s**6 / 180.0 - k5 * s**8 / 6720.0,
            dd_dkappa=-k * s**5 / 60.0 + k3 * s**7 / 1260.0 - k5 * s**9 / 60480.0,
            da_ds=1.0 - k2 * s**2 / 2.0 + k2 * k2 * s**4 / 24.0,
            db_ds=s - k2 * s**3 / 6.0 + k2 * k2 * s**5 / 120.0,
            dd_ds=s**2 / 2.0 - k2 * s**4 / 24.0 + k2 * k2 * s**6 / 720.0,
        )

    theta = k * s
    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))
    return _CurvatureCoefficients(
        a=sin_theta / k,
        b=(1.0 - cos_theta) / k**2,
        d=(theta - sin_theta) / k**3,
        da_dkappa=(k * s * cos_theta - sin_theta) / k**2,
        db_dkappa=(k * s * sin_theta - 2.0 * (1.0 - cos_theta)) / k**3,
        dd_dkappa=(k * s * (1.0 - cos_theta) - 3.0 * (theta - sin_theta))
        / k**4,
        da_ds=cos_theta,
        db_ds=sin_theta / k,
        dd_ds=(1.0 - cos_theta) / k**2,
    )


def _segment_transform_derivative(
    omega_hat: np.ndarray,
    omega_hat_sq: np.ndarray,
    coeffs: _CurvatureCoefficients,
    kappa: float,
    arc_length: float,
    segment_length: float,
    parameter: str,
    curvature_tol: float,
) -> np.ndarray:
    del arc_length
    if parameter == "kx":
        domega = np.array([0.0, 1.0, 0.0], dtype=float)
        dkappa = _safe_curvature_component(domega, omega_hat, kappa, curvature_tol)
        ds = 0.0
    elif parameter == "ky":
        domega = np.array([-1.0, 0.0, 0.0], dtype=float)
        dkappa = _safe_curvature_component(domega, omega_hat, kappa, curvature_tol)
        ds = 0.0
    elif parameter == "eps":
        domega = np.zeros(3, dtype=float)
        dkappa = 0.0
        ds = float(segment_length)
    else:
        raise ValueError(f"Unknown segment parameter {parameter!r}.")

    domega_hat = skew(domega)
    domega_hat_sq = domega_hat @ omega_hat + omega_hat @ domega_hat
    da = coeffs.da_dkappa * dkappa + coeffs.da_ds * ds
    db = coeffs.db_dkappa * dkappa + coeffs.db_ds * ds
    dd = coeffs.dd_dkappa * dkappa + coeffs.dd_ds * ds

    d_rotation = (
        da * omega_hat
        + coeffs.a * domega_hat
        + db * omega_hat_sq
        + coeffs.b * domega_hat_sq
    )
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    d_translation = (
        ds * np.eye(3, dtype=float)
        + db * omega_hat
        + coeffs.b * domega_hat
        + dd * omega_hat_sq
        + coeffs.d * domega_hat_sq
    ) @ ez
    derivative = np.zeros((4, 4), dtype=float)
    derivative[:3, :3] = d_rotation
    derivative[:3, 3] = d_translation
    return derivative


def _safe_curvature_component(
    domega: np.ndarray,
    omega_hat: np.ndarray,
    kappa: float,
    curvature_tol: float,
) -> float:
    if kappa < curvature_tol:
        return 0.0
    omega = np.array(
        [omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]],
        dtype=float,
    )
    return float(np.dot(omega, domega) / kappa)


def _point_jacobian_from_segment_derivatives(
    segments: list[SegmentTransformDerivative],
    params: ThreeSegmentRobotParams,
    *,
    active_segment: int,
    point_transform: np.ndarray,
    point_derivatives: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    include_active_segment_full: bool,
) -> np.ndarray:
    prefixes = [np.eye(4, dtype=float)]
    for segment in segments:
        prefixes.append(prefixes[-1] @ segment.transform)

    suffix = np.eye(4, dtype=float)
    if include_active_segment_full:
        for segment in segments[active_segment + 1 :]:
            suffix = suffix @ segment.transform

    jacobian = np.zeros((3, params.q_size), dtype=float)
    for segment_index in range(active_segment + 1):
        if segment_index == active_segment and not include_active_segment_full:
            derivatives = point_derivatives
            after = np.eye(4, dtype=float)
        else:
            derivatives = segments[segment_index].derivatives
            after = suffix if segment_index == active_segment else _suffix_after(
                segments,
                segment_index,
                active_segment,
                point_transform,
                include_active_segment_full,
            )
        if derivatives is None:
            continue
        before = prefixes[segment_index]
        for local_index, derivative in enumerate(derivatives):
            q_index = segment_index * PCC_VALUES_PER_SEGMENT + local_index
            d_transform = before @ derivative @ after
            jacobian[:, q_index] = d_transform[:3, 3]

    return jacobian


def _suffix_after(
    segments: list[SegmentTransformDerivative],
    segment_index: int,
    active_segment: int,
    point_transform: np.ndarray,
    include_active_segment_full: bool,
) -> np.ndarray:
    suffix = np.eye(4, dtype=float)
    end = active_segment + 1 if include_active_segment_full else active_segment
    for segment in segments[segment_index + 1 : end]:
        suffix = suffix @ segment.transform
    if not include_active_segment_full:
        suffix = suffix @ point_transform
    return suffix


def _centerline_sample_location(
    centerline_index: int,
    samples_per_segment: int,
) -> tuple[int, int]:
    if centerline_index < samples_per_segment:
        return 0, centerline_index
    offset = centerline_index - samples_per_segment
    segment_offset, local_offset = divmod(offset, samples_per_segment - 1)
    return segment_offset + 1, local_offset + 1


def _as_segment_q(q: np.ndarray, params: ThreeSegmentRobotParams) -> np.ndarray:
    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (params.q_size,):
        raise ValueError(f"Expected q with shape ({params.q_size},), got {q_array.shape}.")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q must contain only finite values.")
    return q_array.reshape(params.segment_count, PCC_VALUES_PER_SEGMENT)


def _as_segment_vector(q_segment: np.ndarray) -> np.ndarray:
    values = np.asarray(q_segment, dtype=float)
    if values.shape != (PCC_VALUES_PER_SEGMENT,):
        raise ValueError(
            "q_segment must have shape "
            f"({PCC_VALUES_PER_SEGMENT},), got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("q_segment must contain only finite values.")
    return values
