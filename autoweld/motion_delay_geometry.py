"""Empirical full-pose latency fitting against one observed static surface.

Only actual measured camera poses and native camera points enter this fitter.
It creates no device/model, calls no FK, uses no command targets, and never
extrapolates. A common empirical delay is not proof of six firmware sample ages.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

from autoweld.motion_delay import DelayCalibrationRejected


@dataclass(frozen=True)
class GeometryDelayFit:
    delay_s: float
    uncertainty_s: float | None
    confidence_interval_s: tuple[float, float] | None
    residual_rms_m: float
    heldout_rms_m: float | None
    direction_bias_m: float | None
    samples: int
    cycles: int
    measured_range_m: float
    depth_range_m: float
    method: str = 'full_pose_static_quadratic_surface'
    heldout_passed: bool = False
    diagnostic_only: bool = False
    qualified: bool = False
    boundary_hint: bool = False
    points_per_frame_min: int = 0
    points_per_frame_max: int = 0

    def as_dict(self):
        return asdict(self)


def _rigid(value, label):
    value = np.asarray(value, float)
    if (value.shape[-2:] != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise DelayCalibrationRejected(f'Need finite measured {label} transforms')
    rotations = value[..., :3, :3]
    if (not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-6, rtol=0)
            or np.any(abs(np.linalg.det(rotations) - 1.) > 1e-6)):
        raise DelayCalibrationRejected(f'Need proper measured {label} rotations')
    return value


def _surface_fit(stats, base_weights, selected):
    """IRLS sufficient statistics, each frame totals ONE observation weight."""
    gram, rhs, squared, mean_x, mean_z = stats
    base = base_weights * selected
    if np.count_nonzero(base) < 20:
        raise DelayCalibrationRejected('Too few independent frames for quadratic surface fitting')
    current = base.copy()
    for _ in range(4):
        matrix = np.einsum('f,fij->ij', current, gram, optimize=False)
        target = np.einsum('f,fi->i', current, rhs, optimize=False)
        # A laser line need not identify all six surface coefficients. SVD
        # solves the observable quadratic subspace without inventing unsupported
        # coefficients. Delay identifiability is tested independently by the
        # profile interval and held-out cycles, never by matrix rank alone.
        coefficients = np.linalg.lstsq(matrix, target, rcond=1e-10)[0]
        residual_squared = np.maximum(0., squared - 2 * (rhs @ coefficients)
                                      + np.einsum('i,fij,j->f', coefficients, gram, coefficients, optimize=False))
        mean_residual = mean_z - mean_x @ coefficients
        center = np.median(mean_residual[selected])
        sigma = max(2e-5, 1.4826 * float(np.median(abs(mean_residual[selected] - center))))
        robust = np.minimum(1., 1.5 * sigma / np.maximum(np.sqrt(residual_squared), 1e-12))
        current = base * robust
    # All frames remain present; robust weights are per-frame, never per-pixel.
    cost = float(current @ residual_squared)
    return cost, coefficients, residual_squared, mean_residual, current


def fit_geometry_motion_delay(frame_times, native_points_camera_m, cycle_ids, output_times,
                              camera_poses, *, baseline_optical, weights=None,
                              search_s=(-.150, .150), max_uncertainty_s=.002,
                              max_residual_m=.00025, max_direction_bias_m=.00015,
                              max_gap_s=.020, max_points_per_frame=50, diagnostic_only=False, diagnostics=None):
    """Fit measured camera pose(exposure+delay) using all native XYZ components.

    Static height is quadratic in FIXED baseline optical XY. Actual interpolated
    orientation/translation transform each frame's native points before fitting.
    Native point counts do not multiply statistical independence: every frame
    contributes its mean point residual once. Leave-one-cycle-out estimates and
    a frame-based profile/jackknife interval qualify the delay only when <=2 ms.

    diagnostic_only=True produces an explicitly NONQUALIFIED coarse hint without
    CI/uncertainty. A caller may use filled trace holes only for such diagnostics;
    qualification must use original valid measured brackets over a fixed entire
    narrow search window. This function never removes frames for a trial delay.
    """
    times = np.asarray(frame_times, float)
    cycles = np.asarray(cycle_ids)
    outputs = np.asarray(output_times, float)
    poses = _rigid(camera_poses, 'camera pose')
    baseline = _rigid(baseline_optical, 'baseline optical')
    if (times.ndim != 1 or len(times) < 60 or cycles.shape != times.shape
            or outputs.ndim != 1 or len(outputs) < 20 or poses.shape != (len(outputs), 4, 4)
            or baseline.shape != (4, 4) or not np.isfinite(times).all() or not np.isfinite(outputs).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(outputs) <= 0)):
        raise DelayCalibrationRejected('Need ordered finite exposure frames and dense measured camera poses')
    low, high = map(float, search_s)
    settings = np.asarray([low, high, max_uncertainty_s, max_residual_m, max_direction_bias_m, max_gap_s], float)
    if (not np.isfinite(settings).all() or low >= high or high - low > .6000000001
            or min(settings[2:]) <= 0 or max_uncertainty_s > .002 or max_gap_s > .020):
        raise DelayCalibrationRejected('Invalid full-pose delay search/qualification limits')
    if type(max_points_per_frame) is not int or not 6 <= max_points_per_frame <= 128:
        raise DelayCalibrationRejected('Bound native points per frame to [6,128]')
    if times[0] + low < outputs[0] or times[-1] + high > outputs[-1]:
        raise DelayCalibrationRejected('Measured poses must bracket the full fixed delay search; extrapolation is forbidden')
    # Fixed per-frame trial windows may be disjoint. A bad trace gap BETWEEN
    # those windows is irrelevant; ANY gap queried by ANY allowed delay rejects.
    used_outputs = np.zeros(len(outputs), bool)
    for stamp in times:
        left = max(0, int(np.searchsorted(outputs, stamp + low, side='right')) - 1)
        right = min(len(outputs) - 1, int(np.searchsorted(outputs, stamp + high, side='left')))
        if np.any(np.diff(outputs[left:right + 1]) > max_gap_s + 1e-12):
            raise DelayCalibrationRejected('Measured pose gap exceeds 20 ms within a fixed frame delay search window')
        used_outputs[left:right + 1] = True
    unique_cycles = np.unique(cycles)
    if len(unique_cycles) < 3:
        raise DelayCalibrationRejected('At least three independent probe cycles are required')
    weight = np.ones(len(times)) if weights is None else np.asarray(weights, float)
    if weight.shape != times.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise DelayCalibrationRejected('Need positive finite per-frame weights')
    weight = weight / np.mean(weight)
    if len(native_points_camera_m) != len(times):
        raise DelayCalibrationRejected('Every retained frame needs its own measured native XYZ')
    clouds = []
    for points in native_points_camera_m:
        points = np.asarray(points, float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 6
                or not np.isfinite(points).all() or np.any(points[:, 2] <= 0)):
            raise DelayCalibrationRejected('Every retained frame needs >=6 finite positive-depth native points')
        if len(points) > max_points_per_frame:
            points = points[np.linspace(0, len(points) - 1, max_points_per_frame).astype(int)]
        clouds.append(points)
    counts = np.array([len(points) for points in clouds])
    native = np.zeros((len(times), int(counts.max()), 3))
    point_weights = np.zeros(native.shape[:2])
    for index, points in enumerate(clouds):
        native[index, :len(points)] = points
        point_weights[index, :len(points)] = 1. / len(points)
    mask = point_weights > 0
    slerp = Slerp(outputs, Rotation.from_matrix(poses[:, :3, :3].copy()))
    translations = poses[:, :3, 3]
    normal_z = (translations - baseline[:3, 3]) @ baseline[:3, 2]
    normal_speed = np.zeros(len(outputs))
    segments = np.split(np.arange(len(outputs)), np.flatnonzero(np.diff(outputs) > max_gap_s) + 1)
    for segment in segments:
        if len(segment) > 1:
            normal_speed[segment] = np.gradient(normal_z[segment], outputs[segment])
    measured_range = float(np.ptp(normal_z[used_outputs]))
    if measured_range < .003:
        raise DelayCalibrationRejected('Measured normal probe range is below observable 3 mm')

    def transformed(delay):
        query = times + delay
        rotation = slerp(query).as_matrix()
        translation = np.column_stack([np.interp(query, outputs, translations[:, axis]) for axis in range(3)])
        world = np.einsum('fij,fpj->fpi', rotation, native, optimize=False) + translation[:, None, :]
        return (world - baseline[:3, 3]) @ baseline[:3, :3]

    reference = transformed((low + high) / 2)
    center = np.median(reference[mask], axis=0)
    scale = np.maximum(np.ptp(reference[mask, :2], axis=0), .005)

    @lru_cache(maxsize=16)
    def moments(delay):
        points = transformed(delay)
        xy = (points[..., :2] - center[:2]) / scale
        x, y = xy[..., 0], xy[..., 1]
        design = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)
        z = points[..., 2] - center[2]
        return (np.einsum('fpi,fpj,fp->fij', design, design, point_weights, optimize=False),
                np.einsum('fpi,fp,fp->fi', design, z, point_weights, optimize=False),
                np.sum(point_weights*z*z, axis=1),
                np.einsum('fp,fpi->fi', point_weights, design, optimize=False),
                np.sum(point_weights*z, axis=1))

    all_frames = np.ones(len(times), bool)

    def evaluate(delay, selected=all_frames):
        return _surface_fit(moments(float(delay)), weight, selected)

    grid = np.linspace(low, high, max(81, int(np.ceil((high - low) / .002)) + 1))
    selections = [all_frames] if diagnostic_only else [all_frames, *[cycles != cycle for cycle in unique_cycles]]
    costs = np.empty((len(selections), len(grid)))
    for index, offset in enumerate(grid):
        stats = moments(float(offset))
        for partition, selected in enumerate(selections):
            costs[partition, index] = _surface_fit(stats, weight, selected)[0]

    def optimize(partition):
        index = int(np.argmin(costs[partition]))
        boundary = index in (0, len(grid) - 1)
        if boundary:
            if not diagnostic_only:
                raise DelayCalibrationRejected('Full-pose delay estimate reached the search boundary')
            return float(grid[index]), True
        optimized = minimize_scalar(lambda d: evaluate(d, selections[partition])[0],
            bounds=(grid[index - 1], grid[index + 1]), method='bounded', options={'xatol': 1e-7})
        return float(optimized.x), False

    delay, boundary = optimize(0)
    cost, coefficients, rss, residual, robust_weight = evaluate(delay)
    rms = float(np.sqrt(np.average(rss, weights=robust_weight)))
    common = dict(delay_s=delay, residual_rms_m=rms, samples=len(times), cycles=len(unique_cycles),
                  measured_range_m=measured_range,
                  depth_range_m=float(np.ptp([np.median(points[:, 2]) for points in clouds])),
                  points_per_frame_min=int(counts.min()), points_per_frame_max=int(counts.max()))
    if diagnostic_only:
        return GeometryDelayFit(**common, uncertainty_s=None, confidence_interval_s=None,
                                heldout_rms_m=None, direction_bias_m=None, diagnostic_only=True,
                                boundary_hint=boundary)

    speed = np.interp(times + delay, outputs, normal_speed)
    forward, reverse = speed > .002, speed < -.002
    if min(np.sum(forward), np.sum(reverse)) < 15:
        raise DelayCalibrationRejected('Full-pose probe needs measured motion in both directions')
    direction_bias = abs(float(np.median(residual[forward]) - np.median(residual[reverse])))
    heldout, independent_delays, cycle_diagnostics = [], [], []
    for partition, cycle in enumerate(unique_cycles, start=1):
        training, test = cycles != cycle, cycles == cycle
        if np.sum(training) < 40 or np.sum(test) < 10:
            raise DelayCalibrationRejected('Too few independent frames in held-out probe cycles')
        estimate, _ = optimize(partition)
        independent_delays.append(estimate)
        _, model, _, _, _ = evaluate(estimate, training)
        gram, rhs, squared, mean_x, mean_z = moments(estimate)
        error_squared = np.maximum(0., squared - 2 * (rhs @ model)
                                   + np.einsum('i,fij,j->f', model, gram, model, optimize=False))
        heldout.append(float(np.sqrt(np.average(error_squared[test], weights=weight[test]))))
        held_residual = mean_z - mean_x @ model
        held_speed = np.interp(times + estimate, outputs, normal_speed)
        fwd, rev = test & (held_speed > .002), test & (held_speed < -.002)
        if min(np.sum(fwd), np.sum(rev)) < 5:
            raise DelayCalibrationRejected('Each held-out cycle needs both measured directions')
        held_bias = abs(float(np.median(held_residual[fwd]) - np.median(held_residual[rev])))
        direction_bias = max(direction_bias, held_bias)
        cycle_diagnostics.append(dict(cycle=str(cycle), delay_s=estimate, heldout_rms_m=heldout[-1],
                                      direction_bias_m=held_bias, frames=int(np.sum(test))))
    heldout_rms = max(heldout)
    # Effective observations are FRAMES, never native pixels/stripe points.
    variance = max((2e-5)**2, cost / max(1., float(np.sum(robust_weight)) - 7))
    threshold = cost + 3.841459 * variance
    fine = np.unique(np.r_[np.linspace(low, high, int(np.ceil((high - low) / .0001)) + 1), delay])
    supported = fine[np.array([evaluate(float(value))[0] <= threshold for value in fine])]
    if not len(supported) or supported[0] == low or supported[-1] == high:
        raise DelayCalibrationRejected('Full-pose delay profile is unbounded/unobservable in the search')
    independent = np.asarray(independent_delays)
    jackknife = np.sqrt((len(unique_cycles) - 1) / len(unique_cycles) * np.sum((independent - independent.mean())**2))
    uncertainty = max(delay - supported[0], supported[-1] - delay, .00005,
                      1.96 * float(jackknife), float(np.max(abs(independent - delay))))
    diagnostic = dict(**common, uncertainty_s=uncertainty,
        confidence_interval_s=[delay - uncertainty, delay + uncertainty], heldout_rms_m=heldout_rms,
        direction_bias_m=direction_bias, profile_support_s=[float(supported[0]), float(supported[-1])],
        independent_cycles=cycle_diagnostics, limits=dict(max_residual_m=max_residual_m,
        max_direction_bias_m=max_direction_bias_m, max_uncertainty_s=max_uncertainty_s),
        surface_coefficients=coefficients.tolist(), surface_center_m=center.tolist(),
        surface_scale_xy_m=scale.tolist())
    if diagnostics is not None:
        diagnostics(dict(summary=diagnostic, frame_times=times.copy(), cycle_ids=cycles.copy(),
            frame_mean_residual_m=residual.copy(), frame_rms_m=np.sqrt(rss),
            frame_weights=robust_weight.copy(), normal_speed_m_s=speed.copy(),
            transformed_points_baseline_m=transformed(delay), point_valid_mask=mask.copy()))
    if rms > max_residual_m or heldout_rms > max_residual_m or direction_bias > max_direction_bias_m:
        error = DelayCalibrationRejected('Full-pose static surface failed held-out/direction consistency')
        error.diagnostic = diagnostic
        raise error
    if uncertainty > max_uncertainty_s:
        error = DelayCalibrationRejected('Full-pose delay confidence interval exceeds motion registration budget')
        error.diagnostic = diagnostic
        raise error
    return GeometryDelayFit(**common, uncertainty_s=uncertainty,
        confidence_interval_s=(delay - uncertainty, delay + uncertainty), heldout_rms_m=heldout_rms,
        direction_bias_m=direction_bias, heldout_passed=True, qualified=True)
