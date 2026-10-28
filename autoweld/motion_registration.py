"""Measured joint history and exposure-time cloud registration, without device I/O.

SDK timestamps are SocketCAN CLOCK_REALTIME receive times, not motor sample
clocks. Request/response brackets plus an explicitly evidenced firmware sample-age
bound qualify them. Camera exposure_host_* timestamps are CLOCK_MONOTONIC. This
module never interpolates command targets and never extrapolates measured poses.
"""
import threading
import time
import uuid
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

import numpy as np


class RegistrationRejected(ValueError):
    """Data cannot establish a sufficiently precise measured exposure pose."""


@dataclass(frozen=True)
class HostClockAnchor:
    wall_s: float
    monotonic_s: float
    uncertainty_s: float

    @classmethod
    def capture(cls):
        before = time.monotonic()
        wall = time.time()
        after = time.monotonic()
        return cls(wall, (before+after)/2, (after-before)/2)

    def to_monotonic(self, wall_s):
        return np.asarray(wall_s, float) + (self.monotonic_s-self.wall_s)


@dataclass(frozen=True)
class JointInterpolation:
    monotonic_s: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_error_rad: np.ndarray
    bracket_monotonic_s: np.ndarray
    max_gap_s: float
    timing_uncertainty_s: float
    generation: int
    calibration_only: bool = False
    query_offset_s: float = 0.


class JointOutputHistory:
    """Bounded measured-only per-motor history, safe for one SDK producer.

    ``observe`` attaches directly to MotorChainRobot.add_joint_state_listener.
    No worker, robot, model, SDK calls or unbounded queue is created here. A
    rejection clears the current epoch so no interpolation crosses a fault,
    encoder discontinuity or wall-clock step. Default unknown firmware sample
    age is deliberately unqualified for moving registration. Optional kernel RX
    is a calibration-only event reference; an empirical COMMON delay fit does
    not establish six independent firmware sample ages. Its scope must be
    checked with representative moving-direction/overlap evidence.
    """
    def __init__(self, joint_bounds, *, max_samples=8192, retention_s=30.,
                 max_gap_s=.020, max_timing_uncertainty_s=.002,
                 encoder_sample_age_bound_s=None, max_joint_acceleration_rad_s2=1.,
                 encoder_quantization_rad=25./65535., clock_jump_s=.002, max_joint_speed_rad_s=2.,
                 reference_clock='request_midpoint'):
        self.bounds = np.array(joint_bounds, float, copy=True)
        if self.bounds.shape != (6, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError('Six finite installed joint bounds are required')
        from autoweld.joint_calibration import j6_operating_range
        installed_j6 = j6_operating_range()
        # MuJoCo's XML decimal round-trip can differ from radians by one ULP.
        # Accept that representation error and intersect, never expand, bounds.
        if self.bounds[5, 0] < installed_j6[0]-1e-12 or self.bounds[5, 1] > installed_j6[1]+1e-12:
            raise ValueError('Installed fixed J6 encoder branch is required')
        self.bounds[5, 0] = max(self.bounds[5, 0], installed_j6[0])
        self.bounds[5, 1] = min(self.bounds[5, 1], installed_j6[1])
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 2 <= max_samples <= 65536:
            raise ValueError('History max_samples must be in [2,65536]')
        values = [retention_s, max_gap_s, max_timing_uncertainty_s, max_joint_acceleration_rad_s2, clock_jump_s, max_joint_speed_rad_s]
        if not np.isfinite(values).all() or min(values) <= 0 or not np.isfinite(encoder_quantization_rad) or encoder_quantization_rad < 0:
            raise ValueError('History limits must be finite and positive')
        if encoder_sample_age_bound_s is not None and (not np.isfinite(encoder_sample_age_bound_s) or encoder_sample_age_bound_s < 0):
            raise ValueError('Encoder sample-age bound must be finite and nonnegative')
        self.retention_s, self.max_gap_s = retention_s, max_gap_s
        self.max_timing_uncertainty_s = max_timing_uncertainty_s
        if reference_clock not in ('request_midpoint', 'kernel_receive'):
            raise ValueError('Reference clock must be request_midpoint or kernel_receive')
        if reference_clock == 'kernel_receive' and encoder_sample_age_bound_s is not None:
            raise ValueError('Kernel receive reference is only available for unknown sample age calibration')
        self._reference_clock = reference_clock
        self.encoder_sample_age_bound_s = encoder_sample_age_bound_s
        self.acceleration = max_joint_acceleration_rad_s2
        self.quantization = encoder_quantization_rad
        self.max_joint_speed = max_joint_speed_rad_s
        self.clock_jump_s = clock_jump_s
        self.history_id = uuid.uuid4().hex
        self._samples = [deque(maxlen=max_samples) for _ in range(6)]
        self._lock = threading.Lock()
        self._offset = None
        self._receive_anchors = deque(maxlen=256)
        self._generation = 0
        self._last_error = None
        self._accepted = self._rejected = 0

    @property
    def reference_clock(self):
        """Constructor-only choice: switching requires a fresh history/qualification."""
        return self._reference_clock

    def _invalidate(self, reason):
        self._receive_anchors.clear()
        for samples in self._samples:
            samples.clear()
        self._generation += 1
        self._last_error = str(reason)
        self._rejected += 1

    def _kernel_receive_reference(self, received, recv_mono, rb, anchor, offset):
        """Calibratable RX event reference, never an asserted firmware sample time.

        Preserve historical wall/monotonic offset variation over the receipt's
        anchor interval. Bracket width is a startup fallback until a preceding
        clock anchor exists; normal GIL return latency is not RX-event uncertainty.
        """
        anchors = list(self._receive_anchors)
        anchors.append((anchor.monotonic_s, offset, anchor.uncertainty_s))
        times = np.array([row[0] for row in anchors])
        offsets = np.array([row[1] for row in anchors])
        errors = np.array([row[2] for row in anchors])
        uncertainty = np.empty(6)
        # SO_TIMESTAMPNS is nanosecond resolution, but python-can stores epoch
        # seconds in binary64. Include both timestamp quantization and rounding
        # in the receive value and the wall/monotonic conversion anchor.
        resolution = (.5 * (np.abs(np.spacing(received)) + 1e-9)
                      + .5 * (abs(np.spacing(anchor.wall_s)) + abs(np.spacing(anchor.monotonic_s))))
        for axis, stamp in enumerate(recv_mono):
            preceding = int(np.searchsorted(times[:-1], stamp, side='right')) - 1
            if preceding < 0:
                # No clock observation before this event: its validated request
                # interval remains the conservative event-time fallback.
                uncertainty[axis] = max(anchor.uncertainty_s, abs(stamp - rb[axis, 0]),
                                        abs(rb[axis, 1] - stamp)) + resolution[axis]
            else:
                # Include one more earlier anchor to cover an RX estimate close
                # to an anchor boundary while the offset itself is uncertain.
                first = max(0, preceding - 1)
                lower = np.min(offsets[first:] - errors[first:])
                upper = np.max(offsets[first:] + errors[first:])
                uncertainty[axis] = max(offset - lower, upper - offset) + resolution[axis]
        return recv_mono, uncertainty

    def observe(self, state, *, motion_fault=None):
        """SDK callback: validation failures remain telemetry, never writer errors."""
        try:
            return self.append(state.pos[:6], state.vel[:6], state.motor_timestamps[:6],
                               errors=state.motor_errors[:6] if state.motor_errors else None,
                               request_bounds_monotonic_s=state.motor_request_bounds_monotonic[:6],
                               motion_fault=motion_fault)
        except (AttributeError, TypeError, IndexError) as error:
            with self._lock:
                self._invalidate(error)
            return False

    def append(self, joint_pos, joint_vel, receive_wall_s, *, errors,
               request_bounds_monotonic_s, motion_fault=None, anchor=None):
        anchor = HostClockAnchor.capture() if anchor is None else anchor
        with self._lock:
            try:
                q, v, received = (np.asarray(x, float) for x in (joint_pos, joint_vel, receive_wall_s))
                rb = np.asarray(request_bounds_monotonic_s, float)
                if any(x.shape != (6,) for x in (q, v, received)) or rb.shape != (6, 2):
                    raise RegistrationRejected('Expected six measured outputs and six request brackets')
                if not np.isfinite(np.r_[q, v, received, rb.ravel(), anchor.wall_s, anchor.monotonic_s, anchor.uncertainty_s]).all() or anchor.uncertainty_s < 0:
                    raise RegistrationRejected('Nonfinite measured output or timing')
                if motion_fault is not None or errors is None or len(errors) != 6 or any(str(e).lower() not in ('0x1', '1') for e in errors):
                    raise RegistrationRejected('All six measured motors must be enabled and fault-free')
                if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
                    raise RegistrationRejected('Measured output violates installed bounds/fixed encoder branch')
                offset = anchor.monotonic_s-anchor.wall_s
                if self._offset is not None and abs(offset-self._offset) > self.clock_jump_s:
                    self._offset = offset
                    raise RegistrationRejected('Host wall/monotonic clock mapping stepped')
                self._offset = offset
                recv_mono = anchor.to_monotonic(received)
                if np.any(rb[:, 1] < rb[:, 0]) or np.any(recv_mono < rb[:, 0]-anchor.uncertainty_s) or np.any(recv_mono > rb[:, 1]+anchor.uncertainty_s):
                    raise RegistrationRejected('Receive timestamp lies outside its SDK request/response bracket')
                if np.any(anchor.monotonic_s-rb[:, 1] > .1) or np.any(rb[:, 1] > anchor.monotonic_s+anchor.uncertainty_s):
                    raise RegistrationRejected('SDK publication is stale or in the future')
                age = 0. if self.encoder_sample_age_bound_s is None else self.encoder_sample_age_bound_s
                # Physical samples are bounded in MONOTONIC directly; wall
                # conversion validates receipt provenance but never tightens this
                # bound. Thus wall-clock slew cannot fabricate sample precision.
                if self.reference_clock == 'kernel_receive':
                    centers, uncertainty = self._kernel_receive_reference(received, recv_mono, rb, anchor, offset)
                else:
                    lower = rb[:, 0]-age
                    upper = rb[:, 1]
                    centers, uncertainty = (lower+upper)/2, (upper-lower)/2
                for i, samples in enumerate(self._samples):
                    if samples:
                        old = samples[-1]
                        if received[i] < old[4] or (received[i] > old[4] and centers[i] <= old[0]):
                            raise RegistrationRejected('Measured motor timestamp moved backwards')
                        if received[i] == old[4] and (q[i] != old[1] or v[i] != old[2]):
                            raise RegistrationRejected('Cached motor timestamp changed its measured value')
                        if received[i] > old[4]:
                            physical_gap = centers[i]-old[0]+uncertainty[i]+old[3]
                            if abs(q[i]-old[1]) > self.max_joint_speed*physical_gap+self.quantization:
                                raise RegistrationRejected('Measured encoder discontinuity exceeds installed speed ceiling')
                for i, samples in enumerate(self._samples):
                    if not samples or received[i] != samples[-1][4]:
                        samples.append((float(centers[i]), float(q[i]), float(v[i]), float(uncertainty[i]), float(received[i])))
                    while len(samples) > 2 and samples[0][0] < anchor.monotonic_s-self.retention_s:
                        samples.popleft()
                if self.reference_clock == 'kernel_receive':
                    self._receive_anchors.append((anchor.monotonic_s, offset, anchor.uncertainty_s))
                self._accepted += 1
                self._last_error = None
                return True
            except (ValueError, TypeError, AttributeError) as error:
                self._invalidate(error)
                return False

    def interpolate(self, monotonic_s, *, timing_uncertainty_s=0.):
        return self._interpolate(monotonic_s, timing_uncertainty_s=timing_uncertainty_s)

    def provisional_interpolate(self, monotonic_s, *, timing_uncertainty_s=0.):
        """CALIBRATION ONLY: measured receive-reference curve, NOT physical pose.

        Unknown firmware age is not guessed. These reference-time samples allow
        an independent bidirectional motion/optical experiment to fit the delay.
        Every other rejection/epoch/bracket gate remains in force.
        """
        return self._interpolate(monotonic_s, timing_uncertainty_s=timing_uncertainty_s, provisional=True)

    def interpolate_receive_reference(self, monotonic_s, *, timing_uncertainty_s=0., max_gap_s=.100):
        """Exploratory kernel-RX reference, with its own explicit gap policy.

        This does not alter the strict/experimental physical interpolation gap
        or assert firmware sample timing. No extrapolation is permitted.
        """
        if self.reference_clock != 'kernel_receive' or self.encoder_sample_age_bound_s is not None:
            raise RegistrationRejected('Receive-reference interpolation requires unknown-age kernel_receive history')
        if not np.isfinite(max_gap_s) or not 0 < max_gap_s <= .100:
            raise RegistrationRejected('Receive-reference gap limit must be in (0,100] ms')
        return self._interpolate(monotonic_s, timing_uncertainty_s=timing_uncertainty_s,
                                 provisional=True, max_gap_s=max_gap_s)

    def validate_delay_qualification(self, record):
        if not isinstance(record, dict) or record.get('schema') != 'autoweld.motion-delay.v1' or record.get('status') != 'qualified' or record.get('heldout_passed') is not True:
            raise RegistrationRejected('Motion delay needs a qualified held-out experimental record')
        for key in ('qualification_id', 'history_id', 'source_id', 'optical_intrinsics_id',
                    'acquisition_session_id', 'clock_mapping_id', 'evidence_path', 'method'):
            if not isinstance(record.get(key), str) or not record[key]:
                raise RegistrationRejected(f'Motion delay qualification requires {key}')
        if record['history_id'] != self.history_id:
            raise RegistrationRejected('Motion delay qualification belongs to a different SDK history owner')
        reference = record.get('reference_clock', 'request_midpoint')
        if reference != self.reference_clock:
            raise RegistrationRejected('Motion delay qualification reference clock does not match this history')
        try:
            delay, uncertainty = float(record['delay_s']), float(record['uncertainty_s'])
            ci = np.asarray(record['confidence_interval_s'], float)
        except (KeyError, TypeError, ValueError) as error:
            raise RegistrationRejected('Finite delay estimate, uncertainty and confidence interval are required') from error
        if ci.shape != (2,) or not np.isfinite(np.r_[ci, delay, uncertainty]).all() or uncertainty < 0 or not ci[0] <= delay <= ci[1] or uncertainty+1e-12 < max(delay-ci[0], ci[1]-delay):
            raise RegistrationRejected('Delay uncertainty must cover its finite confidence interval')
        if uncertainty > self.max_timing_uncertainty_s:
            raise RegistrationRejected('Experimental delay uncertainty exceeds timing limit')
        return delay, uncertainty

    def interpolate_with_delay(self, monotonic_s, qualification, *, timing_uncertainty_s=0.):
        """Positive measured sample age means query history at exposure+delay."""
        delay, uncertainty = self.validate_delay_qualification(qualification)
        if self.encoder_sample_age_bound_s is not None:
            raise RegistrationRejected('Experimental delay and firmware-age correction cannot both be applied')
        got = self._interpolate(monotonic_s+delay,
            timing_uncertainty_s=timing_uncertainty_s+uncertainty, provisional=True)
        return JointInterpolation(got.monotonic_s, got.joint_pos, got.joint_vel,
            got.joint_error_rad, got.bracket_monotonic_s, got.max_gap_s,
            got.timing_uncertainty_s, got.generation, False, delay)

    def raw_series(self, start_monotonic_s, end_monotonic_s):
        """Bounded copied current-epoch outputs for delay fitting/diagnostics only."""
        if not np.isfinite([start_monotonic_s, end_monotonic_s]).all() or start_monotonic_s > end_monotonic_s:
            raise RegistrationRejected('Finite ordered diagnostic interval is required')
        with self._lock:
            rows = [np.array([r for r in samples if start_monotonic_s <= r[0] <= end_monotonic_s], float).reshape(-1, 5)
                    for samples in self._samples]
            for row in rows:
                row.flags.writeable = False
            return dict(calibration_only=True, motion_pairing_supported=False, history_id=self.history_id,
                generation=self._generation, reference_clock=self.reference_clock,
                columns=('reference_monotonic_s', 'joint_pos_rad', 'joint_vel_rad_s',
                         'receive_clock_uncertainty_s' if self.reference_clock == 'kernel_receive' else 'request_bracket_uncertainty_s',
                         'receive_wall_s'), axes=tuple(rows), timestamp_semantics=self._reference_semantics())

    def _interpolate(self, monotonic_s, *, timing_uncertainty_s=0., provisional=False, max_gap_s=None):
        if not np.isfinite([monotonic_s, timing_uncertainty_s]).all() or timing_uncertainty_s < 0:
            raise RegistrationRejected('Finite exposure time and uncertainty are required')
        with self._lock:
            if self.encoder_sample_age_bound_s is None and not provisional:
                raise RegistrationRejected('Firmware encoder sample-age bound is unknown; receive time alone is insufficient')
            # Copy only bracket endpoints while holding the producer lock.
            brackets = []
            for samples in self._samples:
                index = bisect_left(samples, monotonic_s, key=lambda row: row[0])
                if index < len(samples) and samples[index][0] == monotonic_s:
                    left = right = samples[index]
                elif index == 0 or index == len(samples):
                    raise RegistrationRejected('Exposure has no measured motor bracket; extrapolation is forbidden')
                else:
                    left, right = samples[index-1], samples[index]
                brackets.append((left, right))
            generation = self._generation
        gap_limit = self.max_gap_s if max_gap_s is None else max_gap_s
        q, v, error, times, uncertainties = [], [], [], [], []
        for left, right in brackets:
            gap = right[0]-left[0]
            uncertainty = max(left[3], right[3])+timing_uncertainty_s
            if gap > gap_limit:
                raise RegistrationRejected('Measured motor bracket exceeds maximum gap')
            if uncertainty > self.max_timing_uncertainty_s:
                raise RegistrationRejected('Combined exposure/motor timing uncertainty exceeds limit')
            fraction = 0. if gap == 0 else (monotonic_s-left[0])/gap
            q.append(left[1]+fraction*(right[1]-left[1]))
            v.append(left[2]+fraction*(right[2]-left[2]))
            # Conservative acceleration-limited linear interpolation error, plus
            # encoder quantization and time uncertainty. Limits are NOT relaxed.
            speed_bound = max(abs(left[2]), abs(right[2])) + self.acceleration*gap
            error.append(self.quantization/2 + speed_bound*uncertainty + self.acceleration*gap*gap/8)
            times.append((left[0], right[0]))
            uncertainties.append(uncertainty)
        return JointInterpolation(float(monotonic_s), np.array(q), np.array(v), np.array(error),
                                  np.array(times), max(b-a for a, b in times), max(uncertainties), generation, provisional)

    def _reference_semantics(self):
        if self.reference_clock == 'kernel_receive':
            return 'Calibratable SocketCAN kernel RX reference mapped to monotonic; firmware sample age unknown'
        if self.encoder_sample_age_bound_s is None:
            return 'SDK request/response midpoint reference; firmware sample age unknown'
        return 'SDK monotonic request/response interval; SocketCAN wall receive provenance'

    def state(self):
        with self._lock:
            return dict(history_id=self.history_id, samples_per_joint=[len(s) for s in self._samples], generation=self._generation,
                        accepted=self._accepted, rejected=self._rejected, last_error=self._last_error,
                        encoder_sample_age_bound_s=self.encoder_sample_age_bound_s,
                        reference_clock=self.reference_clock, timestamp_semantics=self._reference_semantics(),
                        motion_pairing_supported=(self.encoder_sample_age_bound_s is not None and self._last_error is None and all(len(s) >= 2 for s in self._samples)))


class SharedModelFK:
    """Worker-local MjData; compiled model is supplied by the service singleton."""
    def __init__(self, model, site='weld_tcp'):
        import mujoco
        self._mj = mujoco
        self.model = model
        if model.nq < 6 or model.nv < 6:
            raise ValueError('Six robot joints are required')
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise ValueError('TCP site not found in supplied model')
        self.data = mujoco.MjData(model)
        self._lock = threading.Lock()

    def pose_and_jacobian(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RegistrationRejected('Finite six-axis measured pose is required')
        with self._lock:
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[:6] = q
            self._mj.mj_kinematics(self.model, self.data)
            self._mj.mj_comPos(self.model, self.data)
            pose = np.eye(4)
            pose[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
            pose[:3, 3] = self.data.site_xpos[self.site]
            linear, angular = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            self._mj.mj_jacSite(self.model, self.data, linear, angular, self.site)
            return pose, np.vstack((linear[:, :6], angular[:, :6]))


class MovingCloudRegistrar:
    """Pure worker API; native XYZ and calibration remain immutable inputs.

    A caller can retry a missing right bracket after another SDK publication.
    Camera receipt/read-start times never substitute for actual exposure times.
    Encoder RX reference is available only through the explicitly labeled
    receive-reference method. Optical/source identity is checked on every cloud.
    """
    def __init__(self, history, fk, handeye, *, max_spatial_error_m=.0005, max_exposure_s=.020,
                 max_reference_gap_s=.100, max_reference_spatial_error_m=.001):
        self.history, self.fk = history, fk
        self._delay_qualification = None
        self.handeye = dict(handeye)
        self.tcp_from_camera = np.array(handeye['tcp_from_camera'], float, copy=True)
        if handeye.get('units') != 'm' or handeye.get('coordinate_frame') != 'tcp':
            raise ValueError('Measured hand-eye must be TCP frame in meters')
        if self.tcp_from_camera.shape != (4, 4) or not np.isfinite(self.tcp_from_camera).all() or not np.allclose(self.tcp_from_camera[3], [0, 0, 0, 1]) or not np.allclose(self.tcp_from_camera[:3, :3].T @ self.tcp_from_camera[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(self.tcp_from_camera[:3, :3]) < .999999:
            raise ValueError('Hand-eye must be a finite rigid transform')
        if not np.isfinite([max_spatial_error_m, max_exposure_s]).all() or min(max_spatial_error_m, max_exposure_s) <= 0:
            raise ValueError('Registration limits must be finite and positive')
        self.max_spatial_error_m, self.max_exposure_s = max_spatial_error_m, max_exposure_s
        if (not np.isfinite([max_reference_gap_s, max_reference_spatial_error_m]).all()
                or not 0 < max_reference_gap_s <= .100 or not 0 < max_reference_spatial_error_m <= .001):
            raise ValueError('Receive-reference limits must be positive and at most100 ms /1 mm')
        self.max_reference_gap_s = max_reference_gap_s
        self.max_reference_spatial_error_m = max_reference_spatial_error_m

    def set_delay_qualification(self, record):
        """Install the fitter's evidenced result; never silently qualify a guess."""
        import copy
        self.history.validate_delay_qualification(record)
        for key in ('source_id', 'optical_intrinsics_id'):
            if record[key] != self.handeye.get(key, self.handeye.get('calibration_id') if key == 'optical_intrinsics_id' else None):
                raise RegistrationRejected('Delay experiment and installed hand-eye identity differ')
        self._delay_qualification = copy.deepcopy(record)

    @staticmethod
    def _point_error_bound(world, pose, jacobian, joint_error):
        # Evaluate each point's actual velocity Jacobian. Separately summing
        # TCP translation and angular lever norms discards their directions and
        # can reject an accurate scan before any exposure motion is considered.
        lever = world-pose[:3, 3]
        point_jacobian = jacobian[:3].T[None, :, :] + np.cross(
            jacobian[3:].T[None, :, :], lever[:, None, :])
        return np.linalg.norm(point_jacobian, axis=2) @ joint_error

    def register(self, cloud):
        """Strict physical/experimentally qualified branch, unchanged."""
        return self._register(cloud, receive_reference=False)

    def register_receive_reference(self, cloud):
        """Register measured kernel-RX angles at exposure time with unknown age.

        This explicit exploratory quality tier compensates known exposure/host
        processing delay. It applies NO empirical offset and certifies neither
        firmware sample timing nor physical spatial accuracy. Reference estimates
        retain their existing computational timing/gap/blur limits separately.
        """
        if (self.history.reference_clock != 'kernel_receive'
                or self.history.encoder_sample_age_bound_s is not None):
            raise RegistrationRejected('Receive-reference registration requires unknown-age kernel_receive history')
        return self._register(cloud, receive_reference=True)

    def _register(self, cloud, *, receive_reference):
        timing = cloud.get('timing', {})
        if (timing.get('motion_pairing_supported') is not True or timing.get('clock_mapping', {}).get('status') != 'qualified'
                or timing.get('host_clock_domain') != 'host_monotonic'):
            raise RegistrationRejected('Native exposure timing is not qualified in host_monotonic')
        center = timing.get('exposure_host_midpoint_s', timing.get('exposure_host_mid_s'))
        if 'exposure_host_midpoint_s' in timing and 'exposure_host_mid_s' in timing:
            if timing['exposure_host_midpoint_s'] != timing['exposure_host_mid_s']:
                raise RegistrationRejected('Native exposure midpoint aliases disagree')
        start, end = timing.get('exposure_host_start_s'), timing.get('exposure_host_end_s')
        uncertainty = timing.get('exposure_time_uncertainty_s')
        if any(x is None for x in (center, start, end, uncertainty)) or not np.isfinite([center, start, end, uncertainty]).all() or uncertainty < 0 or not start <= center <= end or end-start > self.max_exposure_s:
            raise RegistrationRejected('Invalid native exposure interval/uncertainty')
        if abs(center-(start+end)/2) > 1e-6:
            raise RegistrationRejected('Exposure midpoint disagrees with native interval')
        actual_exposure = timing.get('exposure_duration_us')
        if (timing.get('sensor_shutter_mode') != 'global' or actual_exposure is None
                or not np.isfinite(actual_exposure) or actual_exposure <= 0
                or abs((end-start)-actual_exposure*1e-6) > 1e-6):
            raise RegistrationRejected('Verified global shutter and actual chunk exposure matching the interval are required')
        optical = cloud.get('optical_intrinsics_id', cloud.get('calibration_id'))
        if not cloud.get('source_id') or not optical or cloud.get('source_id') != self.handeye.get('source_id') or optical != self.handeye.get('optical_intrinsics_id', self.handeye.get('calibration_id')):
            raise RegistrationRejected('Cloud and measured hand-eye optical/source identities differ')
        if cloud.get('coordinate_frame') != 'camera' or cloud.get('xyz_m_units', cloud.get('units')) != 'm':
            raise RegistrationRejected('Explicit camera-frame XYZ meter units are required')
        if receive_reference:
            device = timing.get('device_frame_id')
            session = cloud.get('acquisition_session_id')
            mapping_id = timing['clock_mapping'].get('mapping_id')
            if (type(device) is not int or not 0 <= device < 2**64
                    or timing.get('chunk_frame_id') != device or timing.get('exposure_end_frame_id') != device):
                raise RegistrationRejected('Exact native device/chunk/exposure-event frame identity required')
            if (not isinstance(session, str) or not session or not isinstance(mapping_id, str) or not mapping_id
                    or timing.get('acquisition_session_id', session) != session):
                raise RegistrationRejected('Native acquisition session and clock mapping identities are required')
            if (type(cloud.get('sequence')) is not int or cloud['sequence'] < 0
                    or type(cloud.get('generation')) is not int or cloud['generation'] < 0
                    or (cloud.get('frame_product') or {}).get('kind') != 'single_frame'):
                raise RegistrationRejected('Native single-frame generation/sequence provenance is required')
        points = np.asarray(cloud.get('xyz_m'), float)
        if receive_reference and points.size == 0 and points.shape == (0,):
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > 20000 or not np.isfinite(points).all():
            raise RegistrationRejected('Native XYZ must be finite Nx3 with at most 20000 points')
        qualification = None if receive_reference else self._delay_qualification
        if receive_reference:
            def sample(stamp):
                return self.history.interpolate_receive_reference(stamp, timing_uncertainty_s=uncertainty,
                                                                  max_gap_s=self.max_reference_gap_s)
        elif qualification is not None:
            for key, actual in (('source_id', cloud.get('source_id')), ('optical_intrinsics_id', optical),
                                ('acquisition_session_id', cloud.get('acquisition_session_id')),
                                ('clock_mapping_id', timing['clock_mapping'].get('mapping_id'))):
                if qualification[key] != actual:
                    raise RegistrationRejected('Cloud belongs to a different timing-qualification session/identity')
            def sample(stamp):
                return self.history.interpolate_with_delay(stamp, qualification, timing_uncertainty_s=uncertainty)
        else:
            def sample(stamp):
                return self.history.interpolate(stamp, timing_uncertainty_s=uncertainty)
        result, first, last = (sample(stamp) for stamp in (center, start, end))
        if len({result.generation, first.generation, last.generation}) != 1:
            raise RegistrationRejected('Robot history epoch changed during exposure')
        base_from_tcp, jacobian = self.fk.pose_and_jacobian(result.joint_pos)
        camera = base_from_tcp @ self.tcp_from_camera
        world = points @ camera[:3, :3].T + camera[:3, 3]
        error_bound = 0.
        if len(points):
            errors = self._point_error_bound(world, base_from_tcp, jacobian, result.joint_error_rad)
            error_bound = float(errors.max())
            for endpoint in (first, last):
                endpoint_pose, endpoint_jacobian = self.fk.pose_and_jacobian(endpoint.joint_pos)
                endpoint_camera = endpoint_pose @ self.tcp_from_camera
                endpoint_world = points @ endpoint_camera[:3, :3].T + endpoint_camera[:3, 3]
                errors = self._point_error_bound(endpoint_world, endpoint_pose,
                    endpoint_jacobian, endpoint.joint_error_rad)
                error_bound = max(error_bound, float((np.linalg.norm(endpoint_world-world, axis=1)+errors).max()))
        spatial_limit = self.max_reference_spatial_error_m if receive_reference else self.max_spatial_error_m
        if error_bound > spatial_limit:
            if receive_reference:
                raise RegistrationRejected('Receive-reference exposure motion/pose estimate exceeds reference spatial metric')
            raise RegistrationRejected('Exposure motion/pose uncertainty exceeds spatial error limit')
        registered = dict(points_world_m=world, base_from_tcp=base_from_tcp, joint_pos=result.joint_pos,
                    timing_qualified=True, no_extrapolation=True,
                    max_spatial_error_m=self.max_spatial_error_m,
                    max_timing_uncertainty_s=self.history.max_timing_uncertainty_s,
                    max_exposure_s=self.max_exposure_s,
                    exposure_monotonic_s=center, exposure_interval_s=[start, end],
                    timing_uncertainty_s=result.timing_uncertainty_s,
                    joint_error_rad=result.joint_error_rad, spatial_error_bound_m=error_bound,
                    robot_brackets_monotonic_s=result.bracket_monotonic_s,
                    robot_max_gap_s=result.max_gap_s, history_generation=result.generation,
                    history_id=self.history.history_id,
                    registration_method='measured_output_interpolation',
                    delay_qualification_id=None if qualification is None else qualification['qualification_id'],
                    applied_delay_s=result.query_offset_s, clock_mapping_id=timing['clock_mapping'].get('mapping_id'),
                    source_id=cloud['source_id'], optical_intrinsics_id=optical,
                    acquisition_session_id=cloud.get('acquisition_session_id'), sequence=cloud.get('sequence'))
        if receive_reference:
            registered.update(
                reference_registered=True, reference_timing_qualified=True,
                device_frame_id=timing['device_frame_id'], generation=cloud['generation'],
                hardware_exposure_timing_qualified=True, timing_qualified=False,
                physical_sample_timing_qualified=False, firmware_sample_age_s=None,
                physical_timing_uncertainty_s=None, physical_spatial_error_bound_m=None,
                registration_reference='measured_kernel_rx', reference_clock='kernel_receive',
                registration_method='measured_output_receive_reference',
                applied_empirical_delay_s=0., applied_delay_s=0., delay_qualification_id=None,
                reference_timing_uncertainty_s=result.timing_uncertainty_s,
                reference_clock_uncertainty_s=max(0., result.timing_uncertainty_s-uncertainty),
                camera_clock_uncertainty_s=uncertainty,
                reference_spatial_error_bound_m=error_bound,
                reference_joint_error_rad=result.joint_error_rad,
                reference_error_model='Quantization, configured-acceleration interpolation and reference-clock uncertainty; '
                                      'unknown firmware sample age excluded',
                max_reference_timing_uncertainty_s=self.history.max_timing_uncertainty_s,
                max_reference_spatial_error_m=self.max_reference_spatial_error_m,
                max_reference_gap_s=self.max_reference_gap_s,
                timing_uncertainty_s=None, spatial_error_bound_m=None, joint_error_rad=None,
                max_timing_uncertainty_s=None, max_spatial_error_m=None)
        return registered
