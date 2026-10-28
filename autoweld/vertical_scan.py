"""True fixed-view world-minus-Z planning, with no hardware or model ownership.

The caller supplies an existing worker-private planner sharing the service model.
The line uses the raster builder's common IK/interior/retiming validator, without
constructing a raster or introducing lateral camera motion.
"""
from types import SimpleNamespace

import numpy as np

from autoweld.continuous_raster import RasterPlanBuilder


class VerticalLine:
    """One monotonic line with C3 rest endpoints and bounded requested peak feed."""

    def __init__(self, length_m=.080, feed_m_s=.010):
        if (not np.isfinite([length_m, feed_m_s]).all()
                or not 0 < length_m <= .100 or not 0 < feed_m_s <= .030):
            raise ValueError('Finite bounded vertical line length/feed required')
        self.length_m = float(length_m)
        self.duration = self.length_m / feed_m_s * 2.1875
        # Common validator report dimensions describe this single line, not a
        # second row or an area-coverage assertion. Replaced by line scope below.
        self.rows = self.coverage_rows = 1
        self.columns = self.coverage_columns = round(length_m / .001) + 1
        self.lane_pitch_m = 0.
        self.settings = SimpleNamespace(width_m=self.length_m, height_m=0.)

    def fraction(self, t):
        if not np.isfinite(t):
            raise ValueError('Line elapsed time must be finite')
        u = np.clip(float(t) / self.duration, 0., 1.)
        return float(35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7)

    def sample_times(self, period):
        if not np.isfinite(period) or period <= 0:
            raise ValueError('Line sample period must be finite and positive')
        return np.linspace(0., self.duration, int(np.ceil(self.duration / period)) + 1)


class WorldLinePlanBuilder(RasterPlanBuilder):
    """Common full-nine-limit validator applied to a true three-dimensional line.

    Constructor callbacks/budgets match RasterPlanBuilder. baseline_optical is
    the frozen start base-from-camera transform; fixed_optical_rotation must be
    exactly its rotation. No surface/depth callback is accepted for this path.
    """

    def __init__(self, line, seed, **kwargs):
        if not isinstance(line, VerticalLine):
            raise ValueError('World line builder requires genuine VerticalLine geometry')
        if kwargs.get('depth') is not None:
            raise ValueError('World vertical line cannot accept a depth-field deformation')
        super().__init__(line, seed, **kwargs)
        if not np.allclose(self.rotation, self.baseline[:3, :3], atol=1e-10, rtol=0):
            raise ValueError('World vertical line must retain its frozen start camera rotation')

    def target(self, original_time):
        camera = self.baseline.copy()
        camera[:3, 3] += [0., 0., -self.raster.length_m * self.raster.fraction(original_time)]
        return camera @ self.inverse_handeye

    def _finish_validation(self):
        super()._finish_validation()
        if self.result is None:
            return
        report = self.result.report
        end = self.baseline.copy()
        end[:3, 3] += [0., 0., -self.raster.length_m]
        report.update(kind='Continuous laser vertical line', scan_profile='vertical_down',
                      sweep_length_mm=self.raster.length_m * 1000., direction_base=[0., 0., -1.],
                      station_count=self.raster.columns)
        report['validation_proof']['schema'] = 'autoweld.continuous_world_line.v1'
        report['scan_scope'] = dict(
            scan_profile='vertical_down', start_base_from_camera=self.baseline.tolist(),
            end_base_from_camera=end.tolist(), tcp_from_camera=self.handeye.tolist(),
            fixed_optical_rotation=self.rotation.tolist(), direction_base=[0., 0., -1.],
            length_m=self.raster.length_m)
        for key in ('actual_lane_spacing_m', 'coverage_rows', 'coverage_columns',
                    'coverage_cells', 'rows', 'columns'):
            report.pop(key, None)
        report['verification'] = report['verification'].replace('Raster', 'World vertical line')


def build_vertical_plan(planner, *, start_base_from_camera, tcp_from_camera, seed_q, bounds,
                        gripper=0., length_mm=80., feed_mm_s=10., max_total_s=900.,
                        cancelled=None, progress=None):
    """Validate one frozen line using only an existing worker-private planner."""
    from autoweld.scan_plan_probe import _contacts_collide

    start = WorldLinePlanBuilder._transform(start_base_from_camera, 'Frozen vertical start')
    handeye = WorldLinePlanBuilder._transform(tcp_from_camera, 'Measured hand-eye')
    limits = np.asarray(bounds, float).copy()
    seed = np.asarray(seed_q, float)
    seed = np.r_[seed, gripper] if seed.shape == (6,) else seed.copy()
    if (seed.shape != (7,) or not np.isfinite(seed).all() or not np.isfinite(gripper)
            or not np.isclose(seed[6], gripper, atol=1e-12, rtol=0)):
        raise ValueError('Expected finite six/seven-joint seed retaining captured gripper')
    if (limits.shape != (6, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError('Expected finite increasing installed six-joint bounds')
    if np.any(seed[:6] < limits[:, 0]) or np.any(seed[:6] > limits[:, 1]):
        raise ValueError('Vertical line seed exceeds installed joint bounds')

    def validate(q):
        if (q.shape != (7,) or not np.isfinite(q).all() or abs(q[6] - gripper) > 1e-12
                or np.any(q[:6] < limits[:, 0]) or np.any(q[:6] > limits[:, 1])):
            raise ValueError('Vertical line violates installed joint/gripper constraints')

    # A collision query already computes forward kinematics. Reuse its exact
    # pose for the immediately following residual check; never reuse contacts
    # after a Jacobian call has changed private kinematics DATA.
    collision_q, collision_pose = None, None

    def collision(q):
        nonlocal collision_q, collision_pose
        collision_pose = np.array(planner.fk(q), float, copy=True)
        collision_q = q.copy()
        return _contacts_collide(planner)

    def cached_fk(q):
        if collision_q is not None and np.array_equal(q, collision_q):
            return collision_pose.copy()
        return planner.fk(q)

    builder = WorldLinePlanBuilder.from_planner(
        planner, VerticalLine(length_mm / 1000., feed_mm_s / 1000.), seed,
        baseline_optical=start, fixed_optical_rotation=start[:3, :3], tcp_from_camera=handeye,
        bounds=limits, validate=validate, collision=collision, fk=cached_fk,
        max_total_s=max_total_s, max_duration_s=min(900., max_total_s),
        max_preflight_s=min(90., max_total_s))
    plan = builder.run(cancelled=cancelled, progress=progress)
    plan.report['entry_pose'] = (start @ np.linalg.inv(handeye)).tolist()
    plan.report['entry_q'] = plan.path.positions[0].tolist()
    plan.report['feed_mm_s'] = feed_mm_s
    return plan
