"""Smooth travel-depth field from measured scout returns, with explicit holes.

This module creates no robot/camera/model. World points must already be accepted,
acquisition-time registered native measurements of the selected object. A depth
field is a planning reference, never a pointcloud or evidence of observed geometry.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DepthEstimate:
    camera_z_m: float
    supported: bool
    surface_z_m: float | None
    uncertainty_m: float | None
    support_cells: int
    support_points: int
    nearest_observation_m: float | None
    confidence: float
    reason: str


class SurfaceDepthField:
    """Frozen C4 scalar camera-Z reference in original Tube optical coordinates.

    Observed surface height minus target standoff defines each node's camera Z.
    Robust bin medians reject local mixed depth returns. A compact C4 kernel and
    strictly positive last-safe prior give a smooth bounded convex combination;
    no nearest-neighbour jumps or hard support cutoffs enter the motion callback.
    Where support is insufficient, query() reports a hole, surface_z_m=None, and
    camera_z_m is ONLY a smooth travel reference toward frozen last-safe depth.
    It must not authorize captures, prove clearance, or create object geometry.

    laser_offset_camera_m is the measured/calibrated median stripe footprint at
    target standoff, expressed in the chosen fixed camera frame. It accounts for
    the illumination footprint's XY displacement, not exposure/publication delay.
    """

    def __init__(self, world_points_m, baseline_optical, *, fixed_optical_rotation=None,
                 laser_offset_camera_m=(0., 0., 0.), target_standoff_m=.140,
                 last_safe_camera_z_m=0., z_bounds_m=(-.060, .060), bin_size_m=.002,
                 support_radius_m=.025, nearest_limit_m=.015, min_cells=3, min_points=6,
                 max_uncertainty_m=.008, max_bin_depth_span_m=.006, prior_weight=.001):
        self.baseline = self._transform(baseline_optical, 'baseline optical')
        rotation = self.baseline[:3, :3] if fixed_optical_rotation is None else np.asarray(fixed_optical_rotation, float)
        transform = np.eye(4)
        if rotation.shape != (3, 3):
            raise ValueError('Expected fixed optical rotation 3x3')
        transform[:3, :3] = rotation
        self.rotation = self._transform(transform, 'fixed optical rotation')[:3, :3]
        if np.linalg.norm(self.rotation[:, 2] - self.baseline[:3, 2]) > 1e-6:
            raise ValueError('Surface depth requires the fixed original optical axis')
        laser_offset = np.asarray(laser_offset_camera_m, float)
        if laser_offset.shape != (3,) or not np.isfinite(laser_offset).all():
            raise ValueError('Expected finite measured laser offset vector in metres')
        self.footprint_xy = (self.baseline[:3, :3].T @ self.rotation @ laser_offset)[:2]
        for label, value in (('standoff', target_standoff_m), ('bin size', bin_size_m),
                             ('support radius', support_radius_m), ('nearest limit', nearest_limit_m),
                             ('uncertainty limit', max_uncertainty_m), ('bin depth span', max_bin_depth_span_m),
                             ('prior weight', prior_weight)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{label} must be finite and positive')
        if not isinstance(min_cells, int) or min_cells < 1 or not isinstance(min_points, int) or min_points < 1:
            raise ValueError('Minimum support counts must be positive integers')
        bounds = np.asarray(z_bounds_m, float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError('Expected increasing finite camera Z bounds')
        if bounds[0] < -.060 - 1e-12 or bounds[1] > .060 + 1e-12:
            raise ValueError('Surface camera Z bounds cannot exceed original ±60 mm envelope')
        if not np.isfinite(last_safe_camera_z_m) or not bounds[0] <= last_safe_camera_z_m <= bounds[1]:
            raise ValueError('Last-safe camera Z must lie inside depth bounds')
        points = np.asarray(world_points_m, float)
        if points.size == 0:
            points = np.empty((0, 3))
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite Nx3 measured world points in metres')
        self.target_standoff_m = float(target_standoff_m)
        self.fallback = float(last_safe_camera_z_m)
        self.bounds = bounds.copy()
        self.radius = float(support_radius_m)
        self.nearest_limit = float(nearest_limit_m)
        self.min_cells, self.min_points = min_cells, min_points
        self.max_uncertainty = float(max_uncertainty_m)
        self.prior = float(prior_weight)
        local = (points - self.baseline[:3, 3]) @ self.baseline[:3, :3]
        camera_z = local[:, 2] - self.target_standoff_m
        allowed = (camera_z >= bounds[0]) & (camera_z <= bounds[1])
        local = local[allowed]
        self.input_points = len(points)
        self.out_of_bounds_points = int(len(points) - np.sum(allowed))
        self.rejected_bins = 0
        nodes, counts, scatter = [], [], []
        if len(local):
            _, inverse = np.unique(np.floor(local[:, :2] / bin_size_m).astype(np.int64), axis=0, return_inverse=True)
            order = np.argsort(inverse, kind='stable')
            groups = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
            for indices in groups:
                group = local[indices]
                z = group[:, 2] - self.target_standoff_m
                if np.ptp(np.quantile(z, [.1, .9])) > max_bin_depth_span_m:
                    self.rejected_bins += 1
                    continue
                median = float(np.median(z))
                nodes.append([*np.median(group[:, :2], axis=0), median])
                counts.append(len(group))
                scatter.append(float(np.median(abs(z - median)) * 1.4826))
        self.nodes = np.asarray(nodes, float).reshape(-1, 3)
        self.counts = np.asarray(counts, int)
        self.scatter = np.asarray(scatter, float)
        self.tree = cKDTree(self.nodes[:, :2]) if len(self.nodes) else None
        for array in (self.baseline, self.rotation, self.footprint_xy, self.bounds,
                      self.nodes, self.counts, self.scatter):
            array.flags.writeable = False

    @staticmethod
    def _transform(value, label):
        value = np.array(value, float, copy=True)
        if (value.shape != (4, 4) or not np.isfinite(value).all()
                or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-7, rtol=0)):
            raise ValueError(f'Expected proper finite {label} transform')
        return value

    @classmethod
    def from_native_views(cls, views, baseline_optical, tcp_from_camera, **kwargs):
        """Register supplied measured views; caller owns all freshness/quality gates.

        Each view has xyz_m in native camera frame and actual measured
        base_from_tcp at its exposure. Reference/commanded poses are inadmissible.
        Already registered world points should use the regular constructor.
        """
        handeye = cls._transform(tcp_from_camera, 'TCP from camera')
        world = []
        for view in views:
            pose = cls._transform(view['base_from_tcp'], 'measured base from TCP') @ handeye
            xyz = np.asarray(view['xyz_m'], float)
            if xyz.size == 0:
                continue
            if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                raise ValueError('Expected finite native Nx3 XYZ in metres')
            world.append(xyz @ pose[:3, :3].T + pose[:3, 3])
        return cls(np.concatenate(world) if world else np.empty((0, 3)), baseline_optical, **kwargs)

    def _weights(self, x_m, y_m):
        query = np.asarray([x_m, y_m], float)
        if not np.isfinite(query).all():
            raise ValueError('Surface XY query must be finite')
        if self.tree is None:
            return np.empty(0, int), np.empty(0), np.empty(0)
        query += self.footprint_xy
        indices = np.asarray(self.tree.query_ball_point(query, self.radius), int)
        distance = np.linalg.norm(self.nodes[indices, :2] - query, axis=1)
        r = np.minimum(1., distance / self.radius)
        # Wendland C4 radial basis in R²: value + four derivatives vanish at R.
        weights = (1 - r)**6 * (35 * r**2 + 18 * r + 3) / 3
        return indices, weights, distance

    def depth(self, x_m, y_m):
        """Cheap bounded C4 planning callback; support must be checked separately."""
        indices, weights, _ = self._weights(x_m, y_m)
        return float((self.prior * self.fallback + weights @ self.nodes[indices, 2])
                     / (self.prior + np.sum(weights)))

    __call__ = depth

    def query(self, x_m, y_m):
        indices, weights, distance = self._weights(x_m, y_m)
        total = float(np.sum(weights))
        z = float((self.prior * self.fallback + weights @ self.nodes[indices, 2]) / (self.prior + total))
        if not len(indices) or total <= 1e-15:
            return DepthEstimate(z, False, None, None, 0, 0, None, 0., 'no observed support; last-safe travel depth')
        confidence = total / (self.prior + total)
        mean = float(weights @ self.nodes[indices, 2] / total)
        scatter = np.sqrt(float(weights @ ((self.nodes[indices, 2] - mean)**2 + self.scatter[indices]**2) / total))
        nearest = float(np.min(distance))
        uncertainty = float(scatter + .05 * nearest + (1 - confidence) * abs(mean - self.fallback))
        effective = weights > .01
        cells = int(np.sum(effective))
        points = int(np.sum(self.counts[indices[effective]]))
        failures = []
        if cells < self.min_cells or points < self.min_points:
            failures.append('insufficient observed cells/points')
        if nearest > self.nearest_limit:
            failures.append('gap from nearest observation')
        if confidence < .99:
            failures.append('weak kernel support')
        if uncertainty > self.max_uncertainty:
            failures.append('depth spread/uncertainty')
        supported = not failures
        return DepthEstimate(z, supported, z + self.target_standoff_m if supported else None,
                             uncertainty, cells, points, nearest, confidence,
                             'measured neighbourhood' if supported else '; '.join(failures) + '; travel depth only')

    def state(self):
        return {'input_points': self.input_points, 'support_cells': len(self.nodes),
                'out_of_bounds_points': self.out_of_bounds_points, 'rejected_mixed_bins': self.rejected_bins,
                'target_standoff_m': self.target_standoff_m, 'last_safe_camera_z_m': self.fallback,
                'z_bounds_m': self.bounds.tolist(), 'support_radius_m': self.radius,
                'laser_footprint_xy_m': self.footprint_xy.tolist(),
                'uncertainty_kind': 'heuristic neighbourhood depth spread plus distance/prior penalty'}
