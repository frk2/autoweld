#!/usr/bin/env python3
"""File-only Fast/Precision tube comparison; never changes XYZ or calibration.

Fits independent infinite cylinders with the user-reported diameter fixed.
Radial consistency is NOT a timestamp/absolute-position accuracy measurement:
rigid shifts can be absorbed, and cylinder-axis translation is invisible.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Scan:
    session: str
    folder: Path
    points_m: np.ndarray
    frame_ids: np.ndarray
    frames: list
    metadata: dict
    excluded_nonfinite: int = 0


def _basis(axis):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    anchor = np.eye(3)[np.argmin(abs(axis))]
    u = np.cross(axis, anchor)
    u /= np.linalg.norm(u)
    return axis, u, np.cross(axis, u)


def radial_residual_mm(points_m, model):
    delta = (np.asarray(points_m, float) - np.asarray(model["axis_point_world_m"])) * 1000.0
    axis = np.asarray(model["axis_direction_world"])
    perpendicular = delta - (delta @ axis)[:, None] * axis
    return np.linalg.norm(perpendicular, axis=1) - model["radius_mm"]


def balance_weights(points_mm, frame_ids, axis, slice_mm=1.0):
    """Each axial slice equal; within it each frame equal; then points equal."""
    bins = np.floor((points_mm @ axis) / slice_mm).astype(np.int64)
    pairs = np.c_[bins, frame_ids]
    unique, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    _, slice_inverse, frames_per_slice = np.unique(unique[:, 0], return_inverse=True, return_counts=True)
    weights = 1.0 / (counts[inverse] * frames_per_slice[slice_inverse[inverse]])
    return weights / weights.mean()


def _sample(points, frame_ids, max_frames=256, points_per_frame=64):
    frames = np.unique(frame_ids)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    selected = []
    # Source arrays are grouped by frame; one pass to avoid O(frames*points).
    frame_values, first, counts = np.unique(frame_ids, return_index=True, return_counts=True)
    for frame in frames:
        position = int(np.searchsorted(frame_values, frame))
        start, count = first[position], counts[position]
        selected.extend(
            (start + np.linspace(0, count - 1, min(points_per_frame, count)).round().astype(int)).tolist()
        )
    selected = np.asarray(selected, int)
    return points[selected], frame_ids[selected], selected


def _functions(points, axis0, u, v, radius, free_radius=False):
    def unpack(parameters):
        raw = axis0 + parameters[0] * u + parameters[1] * v
        axis = raw / np.linalg.norm(raw)
        center = parameters[2] * u + parameters[3] * v
        return axis, center, parameters[4] if free_radius else radius

    def residual(parameters):
        axis, center, r = unpack(parameters)
        delta = points - center
        along = delta @ axis
        normal = delta - along[:, None] * axis
        return np.linalg.norm(normal, axis=1) - r

    def jacobian(parameters):
        axis, center, _ = unpack(parameters)
        length = np.linalg.norm(axis0 + parameters[0] * u + parameters[1] * v)
        da = (u - axis * (axis @ u)) / length
        db = (v - axis * (axis @ v)) / length
        delta = points - center
        along = delta @ axis
        normal = delta - along[:, None] * axis
        distance = np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
        columns = [
            -along * (delta @ da) / distance,
            -along * (delta @ db) / distance,
            -(normal @ u) / distance,
            -(normal @ v) / distance,
        ]
        if free_radius:
            columns.append(-np.ones(len(points)))
        return np.column_stack(columns)

    return unpack, residual, jacobian


def _weighted_loss(weights):
    def loss(z):
        root = np.sqrt(1.0 + z)
        return np.stack((2.0 * weights * (root - 1.0), weights / root, -0.5 * weights / root**3))

    return loss


def _angular_span(normal, axis):
    if len(normal) < 3:
        return 0.0
    _, u, v = _basis(axis)
    angle = np.sort(np.mod(np.arctan2(normal @ v, normal @ u), 2 * np.pi))
    return float(np.degrees(2 * np.pi - np.max(np.diff(np.r_[angle, angle[0] + 2 * np.pi]))))


def fit_cylinder(points_m, frame_ids=None, *, diameter_mm=25.4, robust_scale_mm=0.1, free_radius=False):
    """Known-radius robust geometry fit; metrics always include all valid points."""
    points_m = np.asarray(points_m, float)
    if points_m.ndim != 2 or points_m.shape[1] != 3 or len(points_m) < 30 or not np.isfinite(points_m).all():
        raise ValueError("Cylinder fit requires at least30 finite world XYZ points")
    if not np.isfinite([diameter_mm, robust_scale_mm]).all() or min(diameter_mm, robust_scale_mm) <= 0:
        raise ValueError("Diameter and robust scale must be finite and positive")
    frame_ids = np.zeros(len(points_m), int) if frame_ids is None else np.asarray(frame_ids, int)
    if frame_ids.shape != (len(points_m),) or np.any(np.diff(frame_ids) < 0):
        raise ValueError("Frame IDs must align with points and be grouped in order")
    origin = np.median(points_m, axis=0)
    all_points = (points_m - origin) * 1000.0
    points, ids, selected = _sample(all_points, frame_ids)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    seed_axes = [eigenvectors[:, index] for index in (2, 1, 0)]
    if max(abs(axis @ np.array([0.0, 0.0, 1.0])) for axis in seed_axes) < 0.98:
        seed_axes.append(np.array([0.0, 0.0, 1.0]))
    radius = diameter_mm / 2.0
    weights = balance_weights(points, ids, seed_axes[0])
    candidates = []
    # Deterministic multi-start avoids assuming the tube is vertical or that
    # the largest PCA dimension is necessarily its axis on a partial arc.
    for seed in seed_axes:
        axis0, u, v = _basis(seed)
        xy = np.c_[points @ u, points @ v]
        algebraic = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], np.sum(xy * xy, axis=1), rcond=None)[0][
            :2
        ]
        centers = [algebraic] + [
            np.median(xy, axis=0) + radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
        unpack, residual, jacobian = _functions(points, axis0, u, v, radius)
        for center in centers:
            result = least_squares(
                residual,
                [0.0, 0.0, *center],
                jac=jacobian,
                loss=_weighted_loss(weights),
                f_scale=robust_scale_mm,
                x_scale=[0.2, 0.2, radius, radius],
                max_nfev=120,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            candidates.append((result.cost, result, axis0, u, v))
    _, result, axis0, u, v = min(candidates, key=lambda value: value[0])
    # Rebalance with the fitted axis once; the objective never uses just inliers.
    unpack, residual, jacobian = _functions(points, axis0, u, v, radius)
    axis, old_center, _ = unpack(result.x)
    weights = balance_weights(points, ids, axis)
    # Re-anchor the direction chart at the solution. Otherwise an equivalent
    # axis nearly perpendicular to its seed creates artificial ill-conditioning.
    axis0, u, v = _basis(axis)
    initial = np.array([0.0, 0.0, old_center @ u, old_center @ v])
    unpack, residual, jacobian = _functions(points, axis0, u, v, radius)
    if free_radius:
        parameters = np.r_[initial, radius]
        unpack, residual, jacobian = _functions(points, axis0, u, v, radius, True)
        bounds = ([-np.inf] * 4 + [radius * 0.1], [np.inf] * 4 + [radius * 10.0])
        scale = [0.2, 0.2, radius, radius, radius]
    else:
        parameters = initial
        bounds = (-np.inf, np.inf)
        scale = [0.2, 0.2, radius, radius]
    result = least_squares(
        residual,
        parameters,
        jac=jacobian,
        loss=_weighted_loss(weights),
        f_scale=robust_scale_mm,
        bounds=bounds,
        x_scale=scale,
        max_nfev=300,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    axis, center, r = unpack(result.x)
    # Fix sign for reporting only; an unoriented cylinder has identical ±axis.
    significant = np.flatnonzero(abs(axis) > 1e-8)
    sign = 1.0 if axis[2] > 0.000001 else (-1.0 if axis[2] < -0.000001 else np.sign(axis[significant[0]]))
    axis *= sign
    axis_point = origin + center / 1000.0
    model = dict(
        radius_mm=float(r),
        diameter_mm=float(2 * r),
        axis_direction_world=axis.tolist(),
        axis_point_world_m=axis_point.tolist(),
        axis_point_definition="intersection with fixed seed-axis plane through dataset median; no unique axial center exists",
    )
    residuals = radial_residual_mm(points_m, model)
    threshold = 3 * robust_scale_mm
    inliers = abs(residuals) <= threshold
    delta = (points_m - axis_point) * 1000.0
    along = delta @ axis
    normal = delta - along[:, None] * axis
    span = _angular_span(normal[inliers], axis)
    J = jacobian(result.x) * np.sqrt(weights)[:, None]
    singular = np.linalg.svd(J * np.asarray(scale)[None, :], compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    warnings = []
    if span < 60.0:
        warnings.append(
            "Partial visible arc below60degrees: center/axis and especially free radius may be weakly identifiable."
        )
    if condition > 1e4:
        warnings.append(
            "Cylinder Jacobian is ill-conditioned; fitted parameters may not be uniquely determined."
        )
    if np.ptp(along[inliers]) < diameter_mm if inliers.any() else True:
        warnings.append(
            "Axial support is shorter than one nominal diameter; axis orientation may be weakly constrained."
        )
    if free_radius:
        warnings.append(
            "Free radius is diagnostic only and may absorb calibration/shape bias; it is not a correction to XYZ scale."
        )
    if free_radius and (span < 120.0 or condition > 1e3):
        warnings.append(
            "Free-radius diagnostic on a partial arc can absorb geometry/calibration bias; not a diameter or scale calibration."
        )
    all_weights = balance_weights(all_points, frame_ids, axis)
    frame_mse = []
    for frame in np.unique(frame_ids):
        frame_mse.append(float(np.mean(residuals[frame_ids == frame] ** 2)))
    stats = dict(
        point_count=len(points_m),
        frame_count=len(np.unique(frame_ids)),
        rms_mm=float(np.sqrt(np.mean(residuals**2))),
        p95_abs_mm=float(np.percentile(abs(residuals), 95)),
        max_abs_mm=float(np.max(abs(residuals))),
        bias_mm=float(np.mean(residuals)),
        median_mm=float(np.median(residuals)),
        robust_sigma_mad_mm=float(1.4826 * np.median(abs(residuals - np.median(residuals)))),
        frame_balanced_rms_mm=float(np.sqrt(np.mean(frame_mse))),
        frame_slice_balanced_rms_mm=float(np.sqrt(np.average(residuals**2, weights=all_weights))),
        inlier_threshold_abs_mm=threshold,
        inlier_count=int(inliers.sum()),
        outlier_count=int((~inliers).sum()),
        inlier_rms_mm=float(np.sqrt(np.mean(residuals[inliers] ** 2))) if inliers.any() else None,
        axis_degrees_from_world_vertical=float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0)))),
        axial_span_mm=float(np.ptp(along)),
        inlier_axial_span_mm=float(np.ptp(along[inliers])) if inliers.any() else None,
        inlier_angular_coverage_deg=span,
        scaled_jacobian_condition=condition,
        fit_sample_points=len(points),
        fit_sample_frames=len(np.unique(ids)),
        robust_scale_mm=robust_scale_mm,
        optimizer_success=bool(result.success),
        optimizer_message=result.message,
        warnings=warnings,
    )
    return dict(
        model=model,
        statistics=stats,
        weighting="Equal1mm fitted-axial slices; equal frames per slice; equal sampled points per frame-slice; weighted soft-L1",
        radius_fixed=not free_radius,
        assumed_nominal_diameter_mm=diameter_mm,
    )


def fit_cylinder_ordinary(points_m, initial_model, *, diameter_mm=25.4):
    """Conventional four-DOF, fixed-radius OLS over every supplied point."""
    points_m = np.asarray(points_m, float)
    if points_m.ndim != 2 or points_m.shape[1] != 3 or len(points_m) < 30 or not np.isfinite(points_m).all():
        raise ValueError("Ordinary cylinder fit requires at least 30 finite XYZ points")
    if not np.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("Diameter must be finite and positive")
    origin = np.median(points_m, axis=0)
    points = (points_m - origin) * 1000.0
    radius = diameter_mm / 2.0
    initial_axis, initial_u, initial_v = _basis(initial_model["axis_direction_world"])
    old_center = (np.asarray(initial_model["axis_point_world_m"]) - origin) * 1000.0
    seeds = [
        initial_axis,
        initial_axis + 0.25 * initial_u,
        initial_axis - 0.25 * initial_u,
        initial_axis + 0.25 * initial_v,
        initial_axis - 0.25 * initial_v,
    ]
    candidates = []
    for seed in seeds:
        axis0, u, v = _basis(seed)
        center = old_center - initial_axis * (axis0 @ old_center) / (axis0 @ initial_axis)
        unpack, residual, jacobian = _functions(points, axis0, u, v, radius)
        result = least_squares(
            residual,
            [0.0, 0.0, center @ u, center @ v],
            jac=jacobian,
            loss="linear",
            x_scale=[0.2, 0.2, radius, radius],
            max_nfev=150,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-10,
        )
        candidates.append((result.cost, result, unpack))
    _, result, unpack = min(candidates, key=lambda item: item[0])
    axis, center, _ = unpack(result.x)
    if axis[2] < 0:
        axis = -axis
    model = dict(
        radius_mm=radius,
        diameter_mm=diameter_mm,
        axis_direction_world=axis.tolist(),
        axis_point_world_m=(origin + center / 1000).tolist(),
    )
    errors = radial_residual_mm(points_m, model)
    return dict(
        model=model,
        objective="Unweighted sum of squared geometric radial residuals over ALL finite accepted-selected main-phase points; fixed nominal radius, no sampling, robust loss, trimming or outlier removal.",
        statistics=dict(
            point_count=len(points_m),
            rms_mm=float(np.sqrt(np.mean(errors**2))),
            p95_abs_mm=float(np.percentile(abs(errors), 95)),
            max_abs_mm=float(np.max(abs(errors))),
            bias_mm=float(np.mean(errors)),
            median_mm=float(np.median(errors)),
            sum_squared_radial_mm2=float(errors @ errors),
            axis_degrees_from_world_vertical=float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0, 1)))),
            optimizer_success=bool(result.success),
            optimizer_message=result.message,
            start_count=len(seeds),
            all_start_rms_mm=[float(np.sqrt(2 * x[0] / len(points))) for x in candidates],
        ),
        limitation="Nonlinear multi-start local optimum, not a proof of a global optimum; measures radial consistency rather than timing or absolute accuracy.",
    )


def native_curve_statistics(points_m):
    """Descriptive planar circle approximation; oblique tube cuts may be ellipses."""
    points = np.asarray(points_m, float) * 1000.0
    if len(points) < 12 or not np.isfinite(points).all():
        return dict(native_curve_rms_mm=None, native_apparent_circle_radius_mm=None, native_arc_deg=None)
    center = points.mean(0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    xy = (points - center) @ vectors[:2].T
    first = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], np.sum(xy * xy, 1), rcond=None)[0]
    radius = math.sqrt(max(1e-12, first[2] + first[:2] @ first[:2]))
    result = least_squares(
        lambda p: np.linalg.norm(xy - p[:2], axis=1) - p[2],
        [*first[:2], radius],
        bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1e6]),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=100,
    )
    radial = xy - result.x[:2]
    angle = np.sort(np.mod(np.arctan2(radial[:, 1], radial[:, 0]), 2 * np.pi))
    arc = float(np.degrees(2 * np.pi - max(np.diff(np.r_[angle, angle[0] + 2 * np.pi]))))
    return dict(
        native_curve_rms_mm=float(np.sqrt(np.mean(result.fun**2))),
        native_apparent_circle_radius_mm=float(result.x[2]),
        native_arc_deg=arc,
        native_planarity_rms_mm=float(np.sqrt(np.mean(((points - center) @ vectors[2]) ** 2))),
    )


def native_axis_normal_statistics(points_m, axis_camera, radius_mm):
    """Refit only native cross-section center; axis comes from the global fit."""
    points = np.asarray(points_m, float) * 1000.0
    if len(points) < 12 or not np.isfinite(points).all():
        return dict(native_axis_normal_rms_mm=None, native_axis_normal_p95_abs_mm=None)
    _, u, v = _basis(axis_camera)
    xy = np.c_[points @ u, points @ v]
    initial = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], np.sum(xy * xy, axis=1), rcond=None)[0][:2]
    fitted = least_squares(
        lambda center: np.linalg.norm(xy - center, axis=1) - radius_mm,
        initial,
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=100,
    )
    errors = np.linalg.norm(xy - fitted.x, axis=1) - radius_mm
    return dict(
        native_axis_normal_rms_mm=float(np.sqrt(np.mean(errors**2))),
        native_axis_normal_p95_abs_mm=float(np.percentile(abs(errors), 95)),
    )


def _native_axis_normal_frame(scan, row, model):
    with np.load(scan.folder / row["native_path"], allow_pickle=False) as native:
        selected = native["selected_indices"]
        points = np.asarray(native["xyz_camera_m"])[selected]
        world = np.asarray(native["xyz_world_m"])
        world = world[selected] if row["world_array_layout"] == "all_native" else world
        points = points[np.isfinite(world).all(axis=1)]
        optical = np.asarray(native["base_from_tcp"]) @ np.asarray(native["tcp_from_camera"])
        axis_camera = optical[:3, :3].T @ np.asarray(model["axis_direction_world"])
        return native_axis_normal_statistics(points, axis_camera, model["radius_mm"])


def resolve_session(runtime, session):
    direct = Path(session)
    if direct.is_dir():
        return direct.resolve()
    for name in ("motion-scans", "laser-scans"):
        folder = Path(runtime) / name / session
        if folder.is_dir():
            return folder.resolve()
    raise ValueError(f"Archive session not found: {session}")


def load_scan(runtime, session, *, acquisition_phase=None):
    folder = resolve_session(runtime, session)
    journal = folder / "frames.jsonl"
    moving = journal.exists()
    if not moving:
        journal = folder / "cells.jsonl"
    if not journal.exists():
        raise ValueError("Archive needs frames.jsonl or cells.jsonl with raw NPZ views")
    meta_path = folder / "session.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # Determine whether phase labels exist before selecting anything: mixed
    # labelled/unlabelled sessions must not silently admit unlabelled anchors.
    available_phases = set()
    with journal.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Uncommitted journal tail at line {line_number}")
            record = json.loads(line)
            if record.get("status") in ("accepted", "captured"):
                phase = (record.get("scan_context") or {}).get("acquisition_phase")
                if phase is not None:
                    available_phases.add(phase)
    labelled = bool(available_phases)
    excluded_phase_frames = {}
    excluded_phase_points = {}
    worlds = []
    ids = []
    frames = []
    excluded = 0
    statuses = {}
    seen = set()
    with journal.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Uncommitted journal tail at line {line_number}")
            record = json.loads(line)
            if record.get("kind") == "gap":
                continue
            status = record.get("status")
            statuses[status] = statuses.get(status, 0) + 1
            if status not in ("accepted", "captured"):
                continue
            path = folder / record["raw_file"] if moving else Path(record["raw_snapshot_path"])
            if path.resolve().parent != folder:
                raise ValueError("Raw native view path escapes session directory")
            phase = (record.get("scan_context") or {}).get("acquisition_phase")
            if acquisition_phase is not None and labelled and phase != acquisition_phase:
                phase_name = phase if phase is not None else "unlabelled"
                with np.load(path, allow_pickle=False) as native:
                    count = len(native["selected_indices"])
                excluded_phase_frames[phase_name] = excluded_phase_frames.get(phase_name, 0) + 1
                excluded_phase_points[phase_name] = excluded_phase_points.get(phase_name, 0) + count
                continue
            key = tuple(
                record.get("acquisition_identity") or (record.get("ordinal", record.get("cell_index")),)
            )
            if key in seen:
                raise ValueError("Repeated accepted native acquisition identity")
            seen.add(key)
            with np.load(path, allow_pickle=False) as native:
                indices = np.asarray(native["selected_indices"])
                camera = np.asarray(native["xyz_camera_m"], float)
                world = np.asarray(native["xyz_world_m"], float)
                if (
                    camera.ndim != 2
                    or camera.shape[1] != 3
                    or world.ndim != 2
                    or world.shape[1] != 3
                    or indices.ndim != 1
                    or indices.dtype.kind not in "iu"
                    or len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0)
                    or np.any(indices >= len(camera))
                ):
                    raise ValueError(f"Invalid selected native arrays: {path.name}")
                selected_world = world[indices] if moving else world
                selected_camera = camera[indices]
                if len(selected_world) != len(indices):
                    raise ValueError("Selected camera/world counts differ")
                valid = np.isfinite(selected_world).all(1)
                excluded += int((~valid).sum())
                selected_world, selected_camera = selected_world[valid], selected_camera[valid]
                if not len(selected_world):
                    continue
                optical = np.asarray(native["base_from_tcp"]) @ np.asarray(native["tcp_from_camera"])
                reg = record.get("registration") or record
                cloud = record.get("cloud") or record
                row = dict(
                    frame=len(frames),
                    acquisition_phase=phase,
                    native_path=path.name,
                    world_array_layout="all_native" if moving else "selected",
                    handeye_sha256=hashlib.sha256(
                        np.asarray(native["tcp_from_camera"], dtype="<f8").tobytes()
                    ).hexdigest(),
                    point_count=len(selected_world),
                    device_frame_id=(cloud.get("timing") or {}).get("device_frame_id"),
                    sequence=cloud.get("sequence"),
                    exposure_midpoint_s=reg.get(
                        "exposure_monotonic_s", (cloud.get("timing") or {}).get("exposure_host_mid_s")
                    ),
                    camera_world_z_m=float(optical[2, 3]),
                    source_id=cloud.get("source_id"),
                    optical_intrinsics_id=cloud.get("optical_intrinsics_id"),
                    calibration_id=cloud.get("calibration_id"),
                    acquisition_session_id=cloud.get("acquisition_session_id"),
                    **native_curve_statistics(selected_camera),
                )
                frames.append(row)
                worlds.append(selected_world)
                ids.append(np.full(len(selected_world), len(frames) - 1, int))
    if not worlds:
        raise ValueError(
            f"No accepted finite selected native world points for phase {acquisition_phase!r} in {folder.name}"
        )
    selection = dict(
        requested_phase=acquisition_phase,
        available_accepted_phases=sorted(available_phases),
        mode=(
            "labelled_phase_only"
            if labelled and acquisition_phase is not None
            else "all_accepted_phases_no_filter_requested"
            if labelled
            else "legacy_unlabelled_all_accepted"
        ),
        legacy_unlabelled_fallback=not labelled,
        included_frames=len(frames),
        included_valid_selected_points=sum(len(x) for x in worlds),
        excluded_phase_frame_counts=excluded_phase_frames,
        excluded_phase_selected_point_counts=excluded_phase_points,
        excluded_phase_frames=sum(excluded_phase_frames.values()),
        excluded_phase_selected_points=sum(excluded_phase_points.values()),
    )
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    metadata = {
        **metadata,
        "journal_status_counts": statuses,
        "phase_selection": selection,
        "archive_summary": summary,
    }
    return Scan(folder.name, folder, np.concatenate(worlds), np.concatenate(ids), frames, metadata, excluded)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(scans, output, *, diameter_mm=25.4, robust_scale_mm=0.1, free_radius=False):
    # Enforce the primary comparison protocol even for callers which loaded all
    # accepted phases for exploratory diagnostics.
    scans = dict(scans)
    for label, phase in (("fast", "moving_main"), ("precision", "stationary_main")):
        if label in scans and scans[label].metadata["phase_selection"]["requested_phase"] != phase:
            scans[label] = load_scan(None, scans[label].folder, acquisition_phase=phase)
    for scan in scans.values():
        summary = scan.metadata.get("archive_summary")
        if summary is not None and summary.get("status") != "complete":
            raise ValueError(f"Primary comparison requires complete archive: {scan.session}")
    output = Path(output)
    if any(output.resolve().is_relative_to(scan.folder.resolve()) for scan in scans.values()):
        raise ValueError("Analysis output must be outside raw archive sessions")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    models = {}
    residuals = {}
    frame_rows = []
    slice_rows = []
    for label, scan in scans.items():
        fitted = fit_cylinder(
            scan.points_m, scan.frame_ids, diameter_mm=diameter_mm, robust_scale_mm=robust_scale_mm
        )
        models[label] = fitted["model"]
        residuals[label] = radial_residual_mm(scan.points_m, fitted["model"])
        if free_radius:
            fitted["free_radius_diagnostic"] = fit_cylinder(
                scan.points_m,
                scan.frame_ids,
                diameter_mm=diameter_mm,
                robust_scale_mm=robust_scale_mm,
                free_radius=True,
            )
        identities = {
            key: sorted({row[key] for row in scan.frames if row.get(key) is not None})
            for key in (
                "source_id",
                "optical_intrinsics_id",
                "calibration_id",
                "acquisition_session_id",
                "handeye_sha256",
            )
        }
        results[label] = dict(
            session=scan.session,
            archive_directory=str(scan.folder),
            excluded_nonfinite_selected_points=scan.excluded_nonfinite,
            identities=identities,
            protocol=scan.metadata.get("scan_protocol", scan.metadata.get("settings", {})),
            phase_selection=scan.metadata["phase_selection"],
            archive_summary=scan.metadata.get("archive_summary"),
            **fitted,
        )
        results[label]["statistics"].update(
            phase_selection_mode=scan.metadata["phase_selection"]["mode"],
            requested_acquisition_phase=scan.metadata["phase_selection"]["requested_phase"],
            excluded_phase_frames=scan.metadata["phase_selection"]["excluded_phase_frames"],
            excluded_phase_selected_points=scan.metadata["phase_selection"]["excluded_phase_selected_points"],
        )
        for row in scan.frames:
            r = residuals[label][scan.frame_ids == row["frame"]]
            frame_rows.append(
                dict(
                    dataset=label,
                    **row,
                    **_native_axis_normal_frame(scan, row, fitted["model"]),
                    rms_radial_mm=float(np.sqrt(np.mean(r * r))),
                    bias_radial_mm=float(np.mean(r)),
                    p95_abs_radial_mm=float(np.percentile(abs(r), 95)),
                )
            )
        projected = [
            row["native_axis_normal_rms_mm"]
            for row in frame_rows
            if row["dataset"] == label and row["native_axis_normal_rms_mm"] is not None
        ]
        results[label]["statistics"]["native_axis_normal_frame_balanced_rms_mm"] = (
            float(np.sqrt(np.mean(np.square(projected)))) if projected else None
        )
        model = fitted["model"]
        along = (
            (scan.points_m - np.asarray(model["axis_point_world_m"]))
            @ np.asarray(model["axis_direction_world"])
            * 1000.0
        )
        bins = np.floor(along).astype(int)
        for axial in np.unique(bins):
            r = residuals[label][bins == axial]
            slice_rows.append(
                dict(
                    dataset=label,
                    axial_slice_start_mm=int(axial),
                    point_count=len(r),
                    rms_mm=float(np.sqrt(np.mean(r * r))),
                    bias_mm=float(np.mean(r)),
                    p95_abs_mm=float(np.percentile(abs(r), 95)),
                )
            )
    fast, precision = scans["fast"], scans["precision"]
    fa, pa = (
        np.asarray(models["fast"]["axis_direction_world"]),
        np.asarray(models["precision"]["axis_direction_world"]),
    )
    gauge = float(np.median(np.r_[fast.points_m[:, 2], precision.points_m[:, 2]]))
    separation = None
    if min(abs(fa[2]), abs(pa[2])) > 0.05:
        points = []
        for label, axis in (("fast", fa), ("precision", pa)):
            c = np.asarray(models[label]["axis_point_world_m"])
            points.append(c + axis * (gauge - c[2]) / axis[2])
        separation = float(np.linalg.norm(points[0] - points[1]) * 1000.0)
    common_point = np.median(np.r_[fast.points_m, precision.points_m], axis=0)
    normal_separation = None
    if abs(fa @ pa) > 0.05:
        intersections = []
        for label, axis in (("fast", fa), ("precision", pa)):
            center = np.asarray(models[label]["axis_point_world_m"])
            intersections.append(center + axis * (pa @ (common_point - center)) / (pa @ axis))
        normal_separation = float(np.linalg.norm(intersections[0] - intersections[1]) * 1000.0)
    identity_comparison = {}
    for key in ("source_id", "optical_intrinsics_id", "calibration_id", "handeye_sha256"):
        left, right = results["fast"]["identities"][key], results["precision"]["identities"][key]
        identity_comparison[key] = dict(
            fast=left, precision=right, match=(left == right) if left and right else None
        )
    protocol_comparison = {}
    for key in (
        "comparison_id",
        "comparison_binding",
        "line_start_base_from_camera",
        "line_end_base_from_camera",
        "selected_optical_roll_deg",
        "direction_base",
        "handeye",
        "model_fingerprint",
    ):
        left, right = fast.metadata.get(key), precision.metadata.get(key)
        protocol_comparison[key] = dict(
            fast=left,
            precision=right,
            match=(left == right) if left is not None and right is not None else None,
        )
    cross = radial_residual_mm(fast.points_m, models["precision"])
    report = dict(
        schema="autoweld.tube-comparison.v1",
        diameter_mm=diameter_mm,
        diameter_basis="User-reported nominal diameter treated as assumed actual; not a traceable dimensional calibration",
        coordinate_frame="robot_base",
        xyz_units="m",
        residual_units="mm",
        datasets=results,
        relative_comparison=dict(
            axis_angle_deg=float(np.degrees(np.arccos(np.clip(abs(fa @ pa), 0, 1)))),
            common_world_z_gauge_m=gauge,
            centerline_separation_at_world_z_mm=separation,
            centerline_gauge_well_conditioned=separation is not None,
            common_axis_normal_gauge_point_m=common_point.tolist(),
            centerline_separation_at_axis_normal_gauge_mm=normal_separation,
            acquisition_identity_comparison=identity_comparison,
            frozen_protocol_exact_comparison=protocol_comparison,
            fast_against_precision_model_rms_mm=float(np.sqrt(np.mean(cross * cross))),
            fast_against_precision_model_bias_mm=float(np.mean(cross)),
        ),
        limitations=[
            "All finite accepted-selected world points in the requested main acquisition phase enter the reported raw RMS, including robust-fit outliers; labelled anchors are excluded and counted.",
            "Legacy archives without acquisition-phase labels fall back explicitly to all accepted frames; moving-only versus stationary-only phase separation cannot be certified for that fallback.",
            "Independent free-center/free-axis cylinder RMS measures radial consistency, not timestamp accuracy or absolute position.",
            "Constant-speed translation delay may cause a rigid cloud shift absorbed by the fitted axis/center; tube-axis translation is entirely unobservable.",
            "Reference frame/slice weights prevent dense or repeated frames from dominating the robust fitting objective; all-point and balanced RMS are both reported.",
            "Native per-frame axis-normal RMS projects XYZ perpendicular to the globally fitted world axis transformed by each saved camera rotation, then refits only circle center at fixed nominal radius; it removes between-frame center offsets but is not independent of fitted axis/orientation.",
            "Unconstrained native-plane circles are descriptive section approximations only; oblique cylinder sections can be elliptical.",
            "No XYZ scale, calibration, laser plane, hand-eye or archived measurements are altered.",
        ],
        method_source="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html",
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_csv(output / "frames.csv", frame_rows)
    _write_csv(output / "axial-slices.csv", slice_rows)
    _write_csv(
        output / "summary.csv",
        [
            dict(
                dataset=label,
                radius_mm=result["model"]["radius_mm"],
                **{
                    f"axis_{axis}": value
                    for axis, value in zip("xyz", result["model"]["axis_direction_world"])
                },
                **{
                    f"axis_point_world_{axis}_m": value
                    for axis, value in zip("xyz", result["model"]["axis_point_world_m"])
                },
                warnings="; ".join(result["statistics"]["warnings"]),
                **{k: v for k, v in result["statistics"].items() if not isinstance(v, list)},
            )
            for label, result in results.items()
        ],
    )
    _plot(scans, models, residuals, frame_rows, report, output / "comparison.png")
    return report


def _plot(scans, models, residuals, frame_rows, report, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/autoweld-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300}
    ):
        fig = plt.figure(figsize=(12, 9), layout="constrained")
        xyz = fig.add_subplot(221, projection="3d")
        trace = fig.add_subplot(222)
        hist = fig.add_subplot(223)
        native = fig.add_subplot(224)
        colors = {"fast": "#D55E00", "precision": "#0072B2"}
        common = np.asarray(models["precision"]["axis_point_world_m"])
        axis = np.asarray(models["precision"]["axis_direction_world"])
        for label, scan in scans.items():
            color = colors[label]
            take = np.linspace(0, len(scan.points_m) - 1, min(4000, len(scan.points_m))).round().astype(int)
            points = scan.points_m[take] * 1000.0
            r = residuals[label]
            xyz.scatter(*points.T, s=1.5, alpha=0.35, color=color, label=label.capitalize(), rasterized=True)
            model = models[label]
            a = np.asarray(model["axis_direction_world"])
            c = np.asarray(model["axis_point_world_m"])
            axial = (scan.points_m - c) @ a
            ends = c + np.array([axial.min(), axial.max()])[:, None] * a
            xyz.plot(*((ends * 1000.0).T), color=color, linewidth=2)
            x = (scan.points_m - common) @ axis * 1000.0
            trace.scatter(x[take], r[take], s=2, alpha=0.2, color=color, rasterized=True)
            hist.hist(
                r,
                bins=80,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.5,
                label=label.capitalize(),
            )
            rows = [
                row
                for row in frame_rows
                if row["dataset"] == label and row["native_axis_normal_rms_mm"] is not None
            ]
            if rows:
                z = np.array([row["camera_world_z_m"] for row in rows]) * 1000.0
                native.plot(
                    z,
                    [row["native_axis_normal_rms_mm"] for row in rows],
                    ".",
                    markersize=3,
                    color=color,
                    label=label.capitalize(),
                )
        xyz.set(
            xlabel="World X (mm)",
            ylabel="World Y (mm)",
            zlabel="World Z (mm)",
            title="Independent cylinder axes and native world points",
        )
        # Display all three world axes legibly despite the long narrow tube.
        # The geometric fits use metres; only this view uses unequal axis scales.
        xyz.set_box_aspect((1.0, 1.0, 1.7))
        xyz.set_title("World view (axis scales differ)")
        for coordinate_axis in (xyz.xaxis, xyz.yaxis, xyz.zaxis):
            coordinate_axis.set_major_locator(MaxNLocator(nbins=3))
        xyz.view_init(elev=18, azim=-65)
        xyz.tick_params(labelsize=8)
        xyz.legend(markerscale=3)
        trace.axhline(0, color=".3", linewidth=0.7)
        trace.set(
            xlabel="Axial position (mm; precision axis reference)",
            ylabel="Radial residual (mm)",
            title="All-point residual to robust balanced fit",
        )
        trace.grid(alpha=0.2)
        hist.set(
            xlabel="Radial residual (mm; all selected points)",
            ylabel="Probability density",
            title="All-point residual distributions",
        )
        hist.legend()
        hist.grid(alpha=0.2)
        native.set(
            xlabel="Measured camera world Z (mm)",
            ylabel="Axis-normal, frame-centered RMS (mm)",
            title="Within-frame fixed-radius consistency",
        )
        native.grid(alpha=0.2)
        native.legend()
        stats = report["datasets"]
        line = " | ".join(
            f"{key.capitalize()}: RMS {value['statistics']['rms_mm']:.3f} mm, P95 {value['statistics']['p95_abs_mm']:.3f} mm, n={value['statistics']['point_count']:,}"
            for key, value in stats.items()
        )
        fig.suptitle(
            f"Tube comparison · assumed diameter {report['diameter_mm']:.1f} mm\nAll-point RMS to robust balanced fit\n{line}", fontsize=11
        )
        fig.savefig(path)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--runtime", type=Path, default=Path("runtime/agents-live"))
    parser.add_argument("--diameter-mm", type=float, default=25.4)
    parser.add_argument("--robust-scale-mm", type=float, default=0.1)
    parser.add_argument("--free-radius", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from threadpoolctl import threadpool_limits

        context = threadpool_limits(limits=1)
    except ImportError:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        result = compare(
            {
                name: load_scan(
                    args.runtime,
                    getattr(args, name),
                    acquisition_phase=("moving_main" if name == "fast" else "stationary_main"),
                )
                for name in ("fast", "precision")
            },
            args.output,
            diameter_mm=args.diameter_mm,
            robust_scale_mm=args.robust_scale_mm,
            free_radius=args.free_radius,
        )
    print(json.dumps({key: value["statistics"] for key, value in result["datasets"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
