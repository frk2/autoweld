"""Pure, locally timed C3 raster geometry; no model, camera or hardware ownership.

All units are metres/seconds. The trajectory describes the optical camera origin
in the ORIGINAL Tube optical XY frame. Fixed chosen view rotation and measured
hand-eye remain the caller's responsibility. A 1 mm row change uses a rounded
hairpin extending slightly beyond the square: validate this full curve, not just
nominal grid endpoints. This module does not authorize motion.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial

_SMOOTH = Polynomial([0, 0, 0, 0, 35, -84, 70, -20])
_INTEGRAL = _SMOOTH.integ()
_TURN_X = Polynomial([0, 1, 0, 0, -5, 6, -2])


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RasterSettings:
    width_m: float = .100
    height_m: float = .100
    spacing_m: float = .001
    cruise_m_s: float = .020
    turn_m_s: float = .001
    ramp_s: float = .600
    turn_s: float = .800
    lane_spacing_m: float | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == 'lane_spacing_m' and value is None:
                continue
            _positive(name, value)
        if self.lane_spacing_m is not None and self.lane_spacing_m > .020:
            raise ValueError('Illumination lane spacing must be at most20 mm')
        if self.turn_m_s >= self.cruise_m_s:
            raise ValueError("Turn speed must be below cruise speed")
        for length in (self.width_m, self.height_m):
            if not np.isclose(length / self.spacing_m, round(length / self.spacing_m), atol=1e-9):
                raise ValueError("Raster dimensions must be integral multiples of spacing")
        if self.width_m <= self.ramp_s * (self.cruise_m_s + self.turn_m_s):
            raise ValueError("Raster width is too short for cruise and both speed ramps")


@dataclass(frozen=True)
class _Segment:
    start: float
    duration: float
    x: Polynomial
    y: Polynomial
    row: int
    kind: str
    _derivatives: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_derivatives", tuple((self.x.deriv(n), self.y.deriv(n)) for n in range(4)))

    def sample(self, elapsed):
        u = np.clip((elapsed - self.start) / self.duration, 0., 1.)
        return tuple(np.array([px(u), py(u)]) / self.duration**n
                     for n, (px, py) in enumerate(self._derivatives))


class ContinuousRaster:
    """Constant feed on row interiors, positive internal speed, start/end at rest.

    Degree-eight row speed ramps join degree-seven XY turns with continuous
    position, velocity, acceleration and jerk. Joins do not restart motion clocks.
    Settings are a process feed request, not a replacement for installed limits.
    Full joint/TCP derivative, IK, collision and measured-feedback validation is
    required by the service before executing any resulting timed joint spline.
    """

    def __init__(self, settings=None):
        self.settings = settings or RasterSettings()
        cfg = self.settings
        self.coverage_rows = round(cfg.height_m / cfg.spacing_m) + 1
        self.coverage_columns = round(cfg.width_m / cfg.spacing_m) + 1
        self.rows = (self.coverage_rows if cfg.lane_spacing_m is None
                     else int(np.ceil(cfg.height_m / cfg.lane_spacing_m - 1e-12)) + 1)
        self.columns = self.coverage_columns
        self.lane_pitch_m = cfg.height_m / (self.rows - 1)
        # A larger illumination-lane step needs a longer smooth turn; its peak
        # cross-lane speed stays at the requested process feed. Also bound the
        # X bulge to0.9mm inside the existing+1mm path envelope.
        self.turn_duration_s = max(cfg.turn_s, 2.1875 * self.lane_pitch_m / cfg.cruise_m_s)
        self.turn_speed_m_s = min(cfg.turn_m_s, .0009 / (.34375 * self.turn_duration_s))
        self.segments = []
        self.row_intervals = []
        t = 0.
        for row in range(self.rows):
            direction = 1 if row % 2 == 0 else -1
            x = -direction * cfg.width_m / 2
            y = -cfg.height_m / 2 + row * self.lane_pitch_m
            row_start = t
            v0 = 0. if row == 0 else self.turn_speed_m_s
            v1 = 0. if row == self.rows - 1 else self.turn_speed_m_s
            for va, vb, duration, kind in (
                (v0, cfg.cruise_m_s, cfg.ramp_s, 'accelerate'),
                (cfg.cruise_m_s, cfg.cruise_m_s,
                 (cfg.width_m - cfg.ramp_s * (cfg.cruise_m_s + (v0 + v1) / 2)) / cfg.cruise_m_s,
                 'cruise'),
                (cfg.cruise_m_s, v1, cfg.ramp_s, 'decelerate'),
            ):
                displacement = duration * (va * Polynomial([0, 1]) + (vb - va) * _INTEGRAL)
                px = Polynomial([x]) + direction * displacement
                self.segments.append(_Segment(t, duration, px, Polynomial([y]), row, kind))
                x = float(px(1))
                t += duration
            self.row_intervals.append((row_start, t))
            if row != self.rows - 1:
                px = Polynomial([x]) + direction * self.turn_speed_m_s * self.turn_duration_s * _TURN_X
                py = Polynomial([y]) + self.lane_pitch_m * _SMOOTH
                self.segments.append(_Segment(t, self.turn_duration_s, px, py, row, 'turn'))
                t += self.turn_duration_s
        self.duration = t
        self._ends = np.array([s.start + s.duration for s in self.segments])

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Elapsed time must be finite")
        if elapsed < 0 or elapsed > self.duration:
            p = self.segments[0 if elapsed < 0 else -1].sample(0 if elapsed < 0 else self.duration)[0]
            return p, np.zeros(2), np.zeros(2), np.zeros(2)
        index = min(int(np.searchsorted(self._ends, elapsed, side='right')), len(self.segments) - 1)
        return self.segments[index].sample(elapsed)

    def position(self, elapsed):
        """Position-only geometry evaluation without unnecessary derivatives."""
        if not np.isfinite(elapsed):
            raise ValueError("Elapsed time must be finite")
        t = min(max(float(elapsed), 0.), self.duration)
        index = min(int(np.searchsorted(self._ends, t, side='right')), len(self.segments) - 1)
        segment = self.segments[index]
        u = (t - segment.start) / segment.duration
        return np.array([segment.x(u), segment.y(u)])

    def pose(self, elapsed, baseline_optical, fixed_optical_rotation, depth_offset_m=0.):
        """Optical pose; XY remains original Tube frame even after fixed roll."""
        baseline = np.asarray(baseline_optical, float)
        rotation = np.asarray(fixed_optical_rotation, float)
        if baseline.shape != (4, 4) or rotation.shape != (3, 3):
            raise ValueError("Expected baseline 4x4 and fixed optical rotation 3x3")
        if not np.isfinite(baseline).all() or not np.isfinite(rotation).all() or not np.isfinite(depth_offset_m):
            raise ValueError("Pose inputs must be finite")
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = baseline[:3, 3] + baseline[:3, :3] @ np.r_[self.position(elapsed), depth_offset_m]
        return pose

    def sample_times(self, max_period_s=.05):
        """Include every join plus bounded temporal intervals for validation/IK.

        At default 20 mm/s, .05 s is at most 1 mm of straight travel. A service
        must refine joint/TCP interiors further if its error/limit gates require.
        """
        _positive('sample period', max_period_s)
        return np.unique(np.concatenate([np.linspace(s.start, s.start + s.duration,
                            max(2, int(np.ceil(s.duration / max_period_s)) + 1)) for s in self.segments]))

    def lookahead(self, elapsed, horizon_s, max_period_s=.05):
        """Bounded future reference samples; no authority to modify an active path.

        A scout can inspect these XY targets and propose a separately validated
        future C3 depth splice. Never replace a committed target with a stale
        camera pose. Registration must interpolate measured SDK history at the
        actual exposure interval, independent of decode/publication delay.
        """
        _positive('horizon', horizon_s)
        _positive('sample period', max_period_s)
        if not np.isfinite(elapsed) or not 0 <= elapsed <= self.duration:
            raise ValueError("Elapsed must be inside the trajectory")
        end = min(self.duration, elapsed + horizon_s)
        times = np.linspace(elapsed, end, max(2, int(np.ceil((end - elapsed) / max_period_s)) + 1))
        return times, np.array([self.position(t) for t in times])

    def cell_events(self):
        """Carrier sample crossings (time,lane,column,xy), never capture Holds.

        These are coverage bins only. Use timestamped actual acquisitions and
        actual measured poses to populate bins; a planned crossing proves no
        observation, depth, association, or coverage on its own.
        """
        events = []
        for row, (start, end) in enumerate(self.row_intervals):
            direction = 1 if row % 2 == 0 else -1
            targets = direction * (-self.settings.width_m / 2 + np.arange(self.columns) * self.settings.spacing_m)
            lo, hi = np.full(self.columns, start), np.full(self.columns, end)
            row_segments = self.segments[row * 4:row * 4 + 3]
            for _ in range(45):
                middle = (lo + hi) / 2
                positions = np.empty(self.columns)
                for segment in row_segments:
                    selected = (middle >= segment.start) & (middle <= segment.start + segment.duration)
                    positions[selected] = segment.x((middle[selected] - segment.start) / segment.duration)
                before = direction * (positions - targets) < 0
                lo = np.where(before, middle, lo)
                hi = np.where(before, hi, middle)
            times = (lo + hi) / 2
            y = -self.settings.height_m / 2 + row * self.lane_pitch_m
            events.extend((float(t), row, column, np.array([float(x), y]))
                          for column, (t, x) in enumerate(zip(times, targets)))
        return events



def raster_path_digest(path):
    """Bind a validation report to exact local-time knots, positions and duration."""
    import hashlib

    digest = hashlib.sha256()
    for array in (np.asarray([path.duration], '<f8'), np.asarray(path.knots, '<f8'),
                  np.asarray(path.positions, '<f8')):
        digest.update(np.asarray(array.shape, '<i8').tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class RasterPlanBuilder:
    """Incremental, hardware-free IK/interior preflight using injected callbacks.

    The worker driving this builder must own private kinematic DATA and reuse the
    existing robot MODEL. Callbacks must not read hardware or mutate the running
    controller/model data. This class creates no model, owner, thread or executor.
    `advance` bounds batches; `run` is convenient inside an existing executor.
    All times exposed by geometry and depth are original (unretimed) seconds.
    """

    def __init__(self, raster, seed, *, baseline_optical, fixed_optical_rotation,
                 tcp_from_camera, solve, fk, jacobian, bounds, limits=None,
                 validate=None, collision=None, depth=None, ik_period_s=.05,
                 validation_period_s=.01, max_duration_s=900., max_preflight_s=120.,
                 max_total_s=900.):
        import time
        from dataclasses import asdict

        from autoweld.motion import MotionLimits

        self.raster = raster
        self.seed = np.array(seed, float, copy=True)
        self.baseline = self._transform(baseline_optical, 'baseline optical')
        self.handeye = self._transform(tcp_from_camera, 'TCP from camera')
        self.inverse_handeye = np.linalg.inv(self.handeye)
        rotation = np.asarray(fixed_optical_rotation, float)
        view = np.eye(4)
        if rotation.shape != (3, 3):
            raise ValueError('Expected fixed optical rotation 3x3')
        view[:3, :3] = rotation
        self.rotation = self._transform(view, 'fixed optical rotation')[:3, :3]
        self.bounds = np.array(bounds, float, copy=True)
        if self.seed.shape != (7,) or not np.isfinite(self.seed).all():
            raise ValueError('Expected finite seven-joint seed')
        if (self.bounds.shape != (6, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('Expected six finite increasing installed joint bounds')
        for name, value in (('IK period', ik_period_s), ('validation period', validation_period_s),
                            ('duration budget', max_duration_s), ('preflight budget', max_preflight_s),
                            ('total budget', max_total_s)):
            _positive(name, value)
        if validation_period_s > .01:
            raise ValueError('Interior validation period may not exceed 10 ms')
        self.solve, self.fk, self.jacobian = solve, fk, jacobian
        self.validate, self.collision, self.depth = validate, collision, depth
        self.limits = limits or MotionLimits()
        self.limit_dict = asdict(self.limits)
        self.limit_values = np.array(list(self.limit_dict.values()))
        self.max_duration_s, self.max_preflight_s = max_duration_s, max_preflight_s
        self.max_total_s = max_total_s
        self.validation_period_s = validation_period_s
        self.times = raster.sample_times(ik_period_s)
        if len(self.times) < 8:
            self.times = np.unique(np.r_[self.times, np.linspace(0, raster.duration, 8)])
        self.positions = []
        self.phase = 'ik'
        self.index = 0
        self.validation_index = 0
        self.validation_count = 0
        self.scale = 1.
        self.retimes = 0
        self.peaks = np.zeros(9)
        self.joint_peaks = np.zeros((3, 6))
        self.residual = np.zeros(2)
        self.result = None
        self.path = None
        self.error = None
        self.started = time.monotonic()
        self.finished = None
        self.validation_times = np.array([])
        self.validation_started = None

    @classmethod
    def from_planner(cls, planner, raster, seed, **kwargs):
        """Use an EXISTING private-data MotionPlanner; never compile a model.

        Construct that planner in the planning worker using shared_kinematics on
        the running robot's model. The normal installed bounds, collision and
        command-constraint callbacks remain required constructor arguments.
        """
        from autoweld.diagnostics import solve_pose

        builder = None

        def solve(target, previous):
            fraction = float(builder.times[builder.index] / raster.duration)
            solved = solve_pose(planner.kin, target, planner.site_name,
                                planner._configuration(previous), stage='raster_ik', path_fraction=fraction)
            return np.r_[solved[:6], previous[6]]

        kwargs.setdefault('limits', planner.limits)
        kwargs.setdefault('fk', planner.fk)
        kwargs.setdefault('jacobian', planner.jacobian)
        builder = cls(raster, seed, solve=solve, **kwargs)
        return builder

    @staticmethod
    def _transform(value, label):
        value = np.array(value, float, copy=True)
        if (value.shape != (4, 4) or not np.isfinite(value).all()
                or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-7, rtol=0)):
            raise ValueError(f'Expected proper finite {label} transform')
        return value

    def target(self, original_time):
        xy = self.raster.position(original_time)
        depth = 0. if self.depth is None else float(self.depth(float(xy[0]), float(xy[1])))
        if not np.isfinite(depth) or abs(depth) > .060 + 1e-12:
            self._reject('depth_envelope', 'Depth field must remain within original optical Z ±60 mm',
                         time_s=float(original_time), depth_offset_m=depth)
        optical = np.eye(4)
        optical[:3, :3] = self.rotation
        optical[:3, 3] = self.baseline[:3, 3] + self.baseline[:3, :3] @ np.r_[xy, depth]
        return optical @ self.inverse_handeye

    def _reject(self, code, message, **details):
        from autoweld.diagnostics import MotionRejection

        raise MotionRejection('raster_preflight', code, message, **details)

    def _configuration_checks(self, q, elapsed, stage):
        if q.shape != (7,) or not np.isfinite(q).all():
            self._reject('joint_nonfinite', 'Non-finite or malformed raster joint reference', time_s=float(elapsed))
        if not np.isclose(q[6], self.seed[6], atol=1e-12, rtol=0):
            self._reject('gripper_motion', 'Removed gripper must remain fixed')
        bad = np.flatnonzero((q[:6] < self.bounds[:, 0] - 1e-10) | (q[:6] > self.bounds[:, 1] + 1e-10))
        if len(bad):
            j = int(bad[0])
            self._reject('joint_bounds', f'Raster {stage} exceeds installed J{j + 1} bounds',
                         time_s=float(elapsed), joint=j + 1, value_rad=float(q[j]),
                         bounds_rad=self.bounds[j].tolist())
        if self.validate is not None:
            self.validate(q)
        if self.collision is not None and self.collision(q):
            self._reject('self_collision', f'Raster {stage} self-collision', time_s=float(elapsed))

    def _ik_one(self):
        t = float(self.times[self.index])
        seed = self.seed if not self.positions else self.positions[-1]
        solved = np.asarray(self.solve(self.target(t), seed.copy()), float)
        if solved.shape == (6,):
            solved = np.r_[solved, self.seed[6]]
        q = np.array(solved, float, copy=True)
        self._configuration_checks(q, t, 'IK knot')
        if self.positions:
            step = abs(q[:6] - self.positions[-1][:6])
            if np.max(step) > .05:
                j = int(np.argmax(step))
                self._reject('joint_step', f'Raster IK branch/continuity jump on J{j + 1}',
                             time_s=t, joint=j + 1, step_rad=float(step[j]), limit_rad=.05)
        self.positions.append(q)
        self.index += 1
        if self.index == len(self.times):
            self._begin_validation()

    def _begin_validation(self):
        from i2rt.robots.waypoint_motion import TimedJointPath

        duration = self.raster.duration * self.scale
        if duration > self.max_duration_s:
            self._reject('duration_budget', 'Validated raster cannot fit requested motion budget',
                         duration_s=float(duration), budget_s=float(self.max_duration_s))
        self.path = TimedJointPath(self.times / self.raster.duration, np.asarray(self.positions), duration)
        # Reserve physical-time sample density for EVERY globally retimed path
        # that could still fit the motion/total budget. Geometry is unchanged by
        # duration: no duplicate collision/FK/limit-validation pass is necessary.
        maximum_duration = max(duration, min(self.max_duration_s, self.max_total_s, duration * 2))
        self.validation_times = np.unique(np.r_[
            np.linspace(0., duration, int(np.ceil(maximum_duration / self.validation_period_s)) + 1),
            self.times * self.scale,
        ])
        self._validation_pending = self.validation_times
        self.validation_index = 0
        self.peaks[:] = 0
        self.joint_peaks[:] = 0
        self.phase = 'validation'

    def _validation_one(self):
        from scipy.spatial.transform import Rotation

        t = float(self._validation_pending[self.validation_index])
        q, velocity, acceleration, jerk = self.path.sample(t)
        self._configuration_checks(q, t, 'spline interior')
        actual = np.array(self.fk(q), float, copy=True)
        desired = self.target(t / self.scale)
        if actual.shape != (4, 4) or not np.isfinite(actual).all():
            self._reject('fk_nonfinite', 'Non-finite/malformed raster forward kinematics', time_s=t)
        error = np.array([np.linalg.norm(actual[:3, 3] - desired[:3, 3]),
                          np.linalg.norm(Rotation.from_matrix(actual[:3, :3] @ desired[:3, :3].T).as_rotvec())])
        self.residual = np.maximum(self.residual, error)
        if np.any(error > .0002):
            self._reject('tcp_residual', 'Raster spline exceeds 0.2 mm / 0.0002 rad target residual',
                         time_s=t, position_error_m=float(error[0]), orientation_error_rad=float(error[1]))
        # Same analytic chain rule as MotionPlanner; J time derivatives use
        # central differences along THIS absolute-time joint reference.
        h = 1e-4
        J = np.array(self.jacobian(q), float, copy=True)
        Jp = np.array(self.jacobian(self.path.position(t + h)), float, copy=True)
        Jm = np.array(self.jacobian(self.path.position(t - h)), float, copy=True)
        if any(j.shape != (6, 6) or not np.isfinite(j).all() for j in (J, Jp, Jm)):
            self._reject('jacobian_nonfinite', 'Non-finite/malformed raster Jacobian', time_s=t)
        if not all(np.isfinite(v).all() for v in (velocity, acceleration, jerk)):
            self._reject('derivative_nonfinite', 'Non-finite raster joint derivatives', time_s=t)
        Jt, Jtt = (Jp - Jm) / (2 * h), (Jp - 2 * J + Jm) / h**2
        v, a, j = velocity[:6], acceleration[:6], jerk[:6]
        tcp = np.array([J @ v, J @ a + Jt @ v, J @ j + 2 * Jt @ a + Jtt @ v])
        if not np.isfinite(tcp).all():
            self._reject('derivative_nonfinite', 'Non-finite raster TCP derivatives', time_s=t)
        joints = abs(np.array([v, a, j]))
        peaks = np.r_[np.linalg.norm(tcp[:, :3], axis=1), np.linalg.norm(tcp[:, 3:], axis=1),
                      np.max(joints, axis=1)]
        self.peaks = np.maximum(self.peaks, peaks)
        self.joint_peaks = np.maximum(self.joint_peaks, joints)
        self.validation_index += 1
        self.validation_count += 1
        if self.validation_index == len(self._validation_pending):
            self._finish_validation()

    def _finish_validation(self):
        import time

        from autoweld.motion import MotionPlan

        if np.any(self.peaks > self.limit_values * .95):
            from i2rt.robots.waypoint_motion import TimedJointPath

            orders = np.tile([1, 2, 3], 3)
            factor = max(1.01, float(1.10 * np.max((self.peaks / self.limit_values) ** (1 / orders))))
            duration = self.path.duration * factor
            if duration > self.max_duration_s or duration > self.max_total_s:
                self._reject('duration_budget', 'Validated raster cannot fit requested motion budget',
                             duration_s=float(duration), budget_s=float(min(self.max_duration_s, self.max_total_s)))
            # TimedJointPath coefficients depend ONLY on normalized local-time
            # knots/positions. A larger duration keeps every geometric sample
            # exact and divides v/a/j by factor/factor²/factor³. The initial
            # reserve guarantees the stretched samples still have <=10 ms gaps.
            self.scale *= factor
            self.retimes += 1
            if self.retimes > 3:
                self._reject('derivative_limit', 'Raster requires more than three conservative retiming refinements')
            self.path = TimedJointPath(self.path.knots, self.path.positions, duration)
            self.validation_times *= factor
            self.peaks /= factor**orders
            self.joint_peaks /= factor**np.arange(1, 4)[:, None]
            if not np.isfinite(self.peaks).all() or np.any(self.peaks > self.limit_values * .95):
                self._reject('derivative_limit', 'Conservative global retiming failed its derivative proof')
            # Small candidate/test/scout paths cap initial reserve at2x to avoid
            # spending the whole900s sample budget on a few-second path. If a
            # larger legal retime needs more density, validate ONLY inserted
            # interior samples; all previous geometry/derivatives remain proved.
            additions = []
            for left, right in zip(self.validation_times[:-1], self.validation_times[1:]):
                subdivisions = int(np.ceil((right - left) / self.validation_period_s - 1e-10))
                if subdivisions > 1:
                    additions.extend(np.linspace(left, right, subdivisions + 1)[1:-1])
            if additions:
                self._validation_pending = np.asarray(additions)
                self.validation_times = np.sort(np.r_[self.validation_times, self._validation_pending])
                self.validation_index = 0
                return
        preflight_s = time.monotonic() - self.started
        if self.path.duration + preflight_s > self.max_total_s:
            self._reject('total_budget', 'Raster motion plus preflight exceeds requested total budget',
                         duration_s=float(self.path.duration), preflight_s=preflight_s,
                         budget_s=float(self.max_total_s))
        report = {
            'kind': 'Continuous laser raster', 'duration_s': float(self.path.duration),
            'original_duration_s': float(self.raster.duration), 'retime_factor': self.scale,
            'retimes': self.retimes, 'limits': self.limit_dict.copy(),
            'planned_peaks': dict(zip(self.limit_dict, self.peaks.tolist())),
            'joint_peak_velocity_rad_s': self.joint_peaks[0].tolist(),
            'joint_peak_acceleration_rad_s2': self.joint_peaks[1].tolist(),
            'joint_peak_jerk_rad_s3': self.joint_peaks[2].tolist(),
            'tcp_path_error_m': float(self.residual[0]), 'tcp_path_error_rad': float(self.residual[1]),
            'ik_knots': len(self.times), 'validation_samples': self.validation_count,
            'validation_max_period_s': self.validation_period_s,
            'validation_actual_max_period_s': float(np.max(np.diff(self.validation_times))),
            'geometric_validation_passes': 1, 'preflight_s': preflight_s,
            'rows': self.raster.rows, 'columns': self.raster.columns,
            'coverage_cells': self.raster.coverage_rows * self.raster.coverage_columns,
            'coverage_rows': self.raster.coverage_rows, 'coverage_columns': self.raster.coverage_columns,
            'actual_lane_spacing_m': self.raster.lane_pitch_m,
            'validation_proof': {
                'schema': 'autoweld.continuous_raster.v1', 'path_sha256': raster_path_digest(self.path),
                'bounds_rad': self.bounds.tolist(), 'collision_checked': self.collision is not None,
                'command_constraints_checked': self.validate is not None,
                'max_period_s': self.validation_period_s, 'ik_step_limit_rad': .05,
                'position_tolerance_m': .0002, 'orientation_tolerance_rad': .0002,
            },
            'scan_scope': {
                'baseline_optical': self.baseline.tolist(), 'fixed_optical_rotation': self.rotation.tolist(),
                'tcp_from_camera': self.handeye.tolist(), 'width_m': self.raster.settings.width_m,
                'height_m': self.raster.settings.height_m, 'depth_bound_m': .060,
            },
            'verification': 'Every IK knot and spline interior at <=10 ms after retiming; finite-difference Jacobian derivatives; '
                            'exact duration-invariant geometry and algebraic v/a/jerk scaling; '
                            '5% acceptance reserve; measured-motion protections still required',
        }
        self.result = MotionPlan(self.path, report)
        self.finished = time.monotonic()
        self.phase = 'ready'

    def advance(self, max_samples=64, max_wall_s=.02, *, cancelled=None):
        """Process bounded work in a worker; a single IK solve cannot be preempted.

        Returns compact progress. Exceptions terminate the candidate and propagate;
        no partial path is executable. Use a cancellation Event callback so shutdown
        can invalidate preflight without waiting for every remaining sample.
        """
        import time

        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError('max_samples must be a positive integer')
        _positive('advance wall budget', max_wall_s)
        if self.phase in ('ready', 'error', 'cancelled'):
            return self.state()
        end = time.monotonic() + max_wall_s
        try:
            for _ in range(max_samples):
                if cancelled is not None and cancelled():
                    self._reject('cancelled', 'Raster preflight cancelled before motion')
                if time.monotonic() - self.started > self.max_preflight_s:
                    self._reject('preflight_budget', 'Raster preflight exceeded its time budget',
                                 budget_s=float(self.max_preflight_s))
                if self.phase == 'ik':
                    self._ik_one()
                elif self.phase == 'validation':
                    self._validation_one()
                if self.phase == 'ready' or time.monotonic() >= end:
                    break
        except Exception as exc:
            self.error = getattr(exc, 'diagnostic', {'message': str(exc), 'exception_type': type(exc).__name__})
            self.phase = 'cancelled' if self.error.get('code') == 'cancelled' else 'error'
            self.result = None
            self.finished = time.monotonic()
            raise
        return self.state()

    def run(self, *, cancelled=None, progress=None):
        """Run ONLY in the service's asynchronous planning worker, never SDK loop."""
        import time

        while self.phase not in ('ready', 'error', 'cancelled'):
            status = self.advance(cancelled=cancelled)
            if progress is not None:
                progress(status)
            time.sleep(0)  # Explicitly yield to the independent SDK/safety owner.
        if self.result is None:
            raise RuntimeError('Raster preflight has no validated result')
        return self.result

    def state(self):
        import time

        return {'phase': self.phase, 'ik_index': self.index, 'ik_total': len(self.times),
                'validation_index': self.validation_index, 'validation_total': len(getattr(self, '_validation_pending', self.validation_times)),
                'validation_samples_completed': self.validation_count,
                'retime_factor': self.scale, 'retimes': self.retimes,
                'duration_s': float(self.raster.duration * self.scale),
                'elapsed_s': (self.finished or time.monotonic()) - self.started,
                'ready': self.phase == 'ready', 'error': self.error}
