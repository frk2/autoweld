"""Command policy independent of transport; all hardware I/O stays in i2rt."""

import copy
import logging
import threading
import time
from dataclasses import replace

import numpy as np

from autoweld.diagnostics import MotionRejection
from autoweld.tracking_limits import POSITION_TOLERANCE_M

ANGLE_TOL = np.deg2rad([0.2, 0.2, 0.2, 1.0, 1.0, 1.0])
REST_TOL = float(np.deg2rad(0.5))
TEST_EXCURSION = 0.49  # Default commissioning envelope; wider tests require explicit configuration.
RETURN_STEP_TIMEOUT = 15.0
# Original PD and fault recovery always retain the factory gains.
FACTORY_KP = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
FACTORY_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
# User-selected gravity-calibrated candidate; overshoot limits: evidence/smooth/REPORT.md.
PID_KP = np.array([80.0, 96.0, 152.8823808, 29.85984, 14.4, 61.917364224])
PID_KI = np.zeros(6)  # User selection: native motor PD only; host I disabled.
PID_KD = np.array([5.0, 5.0, 5.0, 2.592, 1.5, 2.16])
# The installed receiver/camera is represented in the shared dynamics model.
# Historical gripper compensation factors must not carry over to this tool.
DEFAULT_GRAVITY_FACTORS = np.ones(6)
PID_LIMIT = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
COOLDOWN_C = 60.0
COOLDOWN_RESUME_C = 55.0
MAX_POSITION_SPEED_LIMIT = 3.0  # Matches SDK PositionSpeedRobot's accepted ceiling.


class Controller:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, robot_factory, limits, *, live=False, reconnect_factory=None):
        """Acquire the process owner without constructing another SDK connection.

        Browser clients must share this owner; construction of the SDK is inside
        the lock so even simultaneous first connections cannot open two chains.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(robot_factory(), limits, live=live,
                                    robot_factory=reconnect_factory)
            return cls._instance

    def __init__(self, robot, limits, live=False, robot_factory=None):
        self.robot = robot
        self._service_owner_thread = threading.get_ident()
        self.control_law = "POS_VEL"
        self.position_speed_limit = 2.0
        self.test_excursion = TEST_EXCURSION
        self.ki = PID_KI.copy()
        self.gravity_factors = DEFAULT_GRAVITY_FACTORS.copy()
        self.gravity_model = "sdk"
        self.friction_enabled = False  # Latest user selection: omit friction feedforward.
        self.friction_transition = np.zeros(6)
        self.friction_coefficients = np.array([.3, .3, .3, .06, .06, .06])
        self.integral_deadband_deg = 0.06
        self.kp = PID_KP.copy()
        self.kd = PID_KD.copy()
        self.live = live
        self.command_quantization = False
        self.robot_factory = robot_factory
        initial_q = np.asarray(robot.get_joint_pos())
        if initial_q.shape not in ((6,), (7,)):
            raise RuntimeError("Expected six arm joints and an optional gripper from SDK")
        self.sdk_joint_count = len(initial_q)
        self.gripper_limits = np.asarray(robot.get_robot_info().get("gripper_limits", [0., 1.])).copy()
        if self.gripper_limits.shape != (2,):
            self.gripper_limits = np.array([0., 1.])
        self.limits = np.asarray(limits)
        self.strict_joint_limits = set()
        self.closed = False
        self.cached_rest_display = False
        self.rest = self.positions()
        self.gripper_hold_closed = False
        self.target = self.rest.copy()
        self.mode = "Hold"
        self.closed = False
        self.rest_verified = False
        self.settled_since = None
        self.disable_pending = False
        self.disable_warning = None
        self.return_automatic = False
        self.return_step_started = None
        self.disabled_q = None
        self.motion_planner = None
        self.motion_collision = None
        self.motion_plan = None
        self.last_motion = None
        self.motion_started = None
        self.motion_automatic = False
        self.feedback_warning = None
        self.cooldown_active = False
        self.cooldown_max_c = None
        self._last_feedback_warning = 0.0
        self.mode_transition = False
        self.mode_transition_error = None
        self._mode_lock = threading.RLock()
        self.shutdown_requested = False
        self.shutdown_phase = 'idle'
        self.shutdown_error = None
        self._shutdown_phase_started = None

    @property
    def shutdown_status(self):
        return {'requested': self.shutdown_requested,
                'active': self.shutdown_requested and not self.closed,
                'phase': self.shutdown_phase, 'error': self.shutdown_error,
                'retryable': self.shutdown_phase == 'blocked',
                'rest_verified': self.rest_verified}

    def _guard_shutdown(self):
        if self.shutdown_requested:
            raise RuntimeError('Shutdown requested; use shutdown retry or pause shutdown')

    def shutdown(self, *, retry=False):
        """Always accept exit intent, independently of application motion gates.

        This does not relax physical checks or close the SDK. The dedicated tick
        captures measured hold, validates a captured-rest path at configured limits and verifies
        fresh stationary rest before the normal SDK close. A blocked request
        needs an explicit retry and never repeatedly replans by itself.
        """
        if self.shutdown_requested and not (retry and self.shutdown_phase == 'blocked'):
            return self.shutdown_status
        self.shutdown_requested = True
        self.shutdown_phase = 'disabled' if self.closed else 'stopping'
        self.shutdown_error = None
        self._shutdown_phase_started = time.monotonic()
        self.disable_pending = False
        self.disable_warning = None
        self.return_automatic = False
        self.motion_automatic = False
        self.motion_plan = None
        self.return_step_started = None
        if not self.closed:
            self.rest_verified = False
            self.settled_since = None
        return self.shutdown_status

    def _shutdown_capture_hold(self):
        if not self._feedback_health():
            raise RuntimeError('Fresh SDK feedback required for shutdown measured hold')
        if self.motion_fault():
            # The SDK already enforces its native measured fault hold. Never
            # acknowledge it, switch modes, or submit a competing reference.
            self.motion_plan = None
            return
        self._ensure_position_speed()
        q = self.positions()
        q[6] = self.gripper_target
        self.validate(q, automatic=False)
        self._send(q, 'Shutdown stopping')

    def _block_shutdown(self, error):
        self.shutdown_phase = 'blocked'
        self.shutdown_error = str(error)
        self.disable_pending = False
        self.motion_plan = None
        self.rest_verified = False
        self.settled_since = None
        try:
            self._shutdown_capture_hold()
        except Exception as hold_error:
            self.shutdown_error += f'; measured hold unavailable: {hold_error}'
        self.mode = 'Shutdown blocked'
        logging.warning('Shutdown blocked; inspect SDK control and motor power status: %s', self.shutdown_error)
        return self.shutdown_status

    def pause_shutdown(self, reason='Stopped by operator'):
        """Stop a shutdown return without dropping its intent or motor power."""
        self.shutdown_requested = True
        if self.closed:
            self.shutdown_phase = 'disabled'
            return self.shutdown_status
        return self._block_shutdown(reason)

    def _shutdown_temperature(self):
        if hasattr(self.robot, 'get_temperature_state'):
            state = self.robot.get_temperature_state()
            values = np.asarray(list(state.get('mos_c', []))+list(state.get('rotor_c', [])), float)
            if not values.size or not np.isfinite(values).all():
                raise RuntimeError('Valid SDK temperatures required for shutdown motion')
            self.cooldown_max_c = float(values.max())
            if self.cooldown_max_c >= COOLDOWN_C:
                self.cooldown_active = True  # The sole shutdown motion already returns to cooldown rest.

    def _tick_shutdown(self):
        if self.closed:
            self.shutdown_phase = 'disabled'
            return
        if self.shutdown_phase == 'blocked':
            return
        if not self._feedback_health():
            raise RuntimeError('Fresh SDK feedback required for shutdown')
        self._shutdown_temperature()
        now = time.monotonic()
        fault = self.motion_fault()
        if fault and not self.at_rest():
            raise RuntimeError('SDK motion fault blocks shutdown return; native powered hold retained')
        if self.shutdown_phase == 'stopping':
            self._shutdown_capture_hold()
            self.shutdown_phase = 'waiting'
            self._shutdown_phase_started = time.monotonic()
            return
        if self.shutdown_phase == 'waiting':
            if not self._quiet():
                if now-self._shutdown_phase_started > RETURN_STEP_TIMEOUT:
                    raise RuntimeError('Shutdown measured hold did not become quiet; retry after inspection')
                return
            if self.at_rest():
                self.shutdown_phase = 'settling'
                self._shutdown_phase_started = now
                self.mode = 'Returning to rest'
            else:
                target = self.rest.copy()
                target[6] = self.gripper_target
                target = self.validate(target, automatic=False)
                if self.motion_planner is None and self.live:
                    raise RuntimeError('Validated motion planner required for shutdown return')
                planner = self.planner_for_controller() if self.motion_planner is not None else None
                self._start_waypoint(target, 'Returning to rest', automatic=False, planner=planner)
                self.shutdown_phase = 'returning'
                self._shutdown_phase_started = time.monotonic()
                return
        if self.shutdown_phase == 'returning':
            if fault:
                self.motion_plan = None
            if self.motion_plan is not None:
                elapsed = now-self.motion_started
                if not self.live:
                    self.robot.command_joint_pos(self._sdk_positions(self.motion_plan.path.sample(elapsed)[0]))
                if elapsed < self.motion_plan.path.duration+(.02 if self.live else 0.):
                    return
                self.motion_plan = None
            self.shutdown_phase = 'settling'
            self._shutdown_phase_started = now
        if self.shutdown_phase == 'settling':
            if self.at_rest() and self._quiet():
                if self.settled_since is None:
                    self.settled_since = now
                if now-self.settled_since >= 1.:
                    self.rest_verified = True
                    self._disable_at_rest()
                    self.shutdown_phase = 'disabled'
                    return
            else:
                self.rest_verified = False
                self.settled_since = None
            if now-self._shutdown_phase_started > RETURN_STEP_TIMEOUT:
                raise RuntimeError('Shutdown rest verification timed out; motors remain enabled')

    def configure_long_cruise_test(self, enabled):
        """Explicit, nonpersistent exception for the user-approved 500 mm sweep."""
        if not self.closed:
            raise RuntimeError("Change the test excursion exception only with motors disabled")
        self.test_excursion = 1.55 if enabled else TEST_EXCURSION

    def configure_gravity_test_envelope(self, excursion_rad):
        """Nonpersistent wider commissioning envelope, changed only while off."""
        if not self.closed:
            raise RuntimeError('Change the gravity test envelope only with motors disabled')
        if not np.isfinite(excursion_rad) or not TEST_EXCURSION <= excursion_rad <= 1.8:
            raise ValueError('Gravity test excursion must be in [0.49, 1.8] rad')
        self.test_excursion = float(excursion_rad)

    def positions(self):
        if self.closed:
            return self.disabled_q.copy()
        return self._logical_positions(self.robot.get_joint_pos())

    def record_disabled_pose(self, q, *, capture_rest=False):
        """Commit already-verified disabled feedback; no SDK/motor call is made."""
        if not self.closed or self.motion_plan is not None or self.shutdown_requested or self.disable_pending:
            raise RuntimeError('Disabled pose recording requires an idle disabled controller')
        q = np.asarray(q, float)
        if (q.shape != (7,) or not np.isfinite(q).all() or not np.isclose(q[6], self.gripper_target)
                or np.any(q[:6] < self.limits[:, 0]) or np.any(q[:6] > self.limits[:, 1])):
            raise ValueError('Disabled measured pose must remain inside all installed joint bounds')
        self.disabled_q, self.target = q.copy(), q.copy()
        self.cached_rest_display = False
        self.mode = 'Disabled'
        self.rest_verified = bool(capture_rest)
        if capture_rest:
            self.rest = q.copy()

    def _logical_positions(self, values):
        q = np.array(values, dtype=float, copy=True)
        if self.sdk_joint_count == 6 and q.shape == (6,):
            q = np.r_[q, 0.]
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid SDK joint feedback")
        return q

    def _sdk_positions(self, q):
        return np.asarray(q)[:self.sdk_joint_count].copy()

    def _sdk_path(self, path):
        if self.sdk_joint_count == 7:
            return path
        return replace(path, positions=path.positions[:, :6])

    def configure_command_quantization(self, enabled):
        if self.control_law in ("POS_VEL", "MIXED_J45") and enabled:
            raise ValueError("MIT command compensation is unavailable in position-speed mode")
        if self.motion_plan is not None or self.disable_pending:
            raise RuntimeError("Wait for motion before changing command encoding")
        if self.motion_fault():
            raise RuntimeError("Acknowledge the fault before changing command encoding")
        if self.live and not self.closed:
            self.robot.configure_command_quantization(bool(enabled))
        self.command_quantization = bool(enabled)

    def motion_fault(self):
        return self.robot.get_motion_fault() if hasattr(self.robot, "get_motion_fault") else None

    def acknowledge_fault(self):
        self._guard_shutdown()
        if not self._feedback_health():
            raise RuntimeError('Fresh SDK feedback required to acknowledge a motion fault')
        self.robot.acknowledge_motion_fault()
        self._ensure_position_speed()
        self.ki.fill(0.0)
        self.kp = FACTORY_KP.copy()
        self.kd = FACTORY_KD.copy()
        self.target = self.positions()
        self.mode = "Hold"

    def health(self):
        self._feedback_health()
        if self.motion_fault():
            raise RuntimeError("Oscillation fault latched; acknowledge after the arm is quiet")

    def sdk_status(self):
        """Read cached SDK diagnostics without opening a device or authorizing motion."""
        required = self.live and not self.closed
        status = {'required': required, 'control_alive': None, 'chain_running': None,
                  'motor_worker_alive': None, 'server_thread_alive': None,
                  'feedback_fresh': False, 'feedback_timestamp': None,
                  'feedback_age_s': None, 'failure': None}
        if not required:
            return status
        chain = getattr(self.robot, 'motor_chain', None)
        worker = getattr(chain, '_control_thread', None)
        server = getattr(self.robot, '_server_thread', None)
        status.update(chain_running=bool(getattr(chain, 'running', False)),
                      motor_worker_alive=worker.is_alive() if worker is not None else None,
                      server_thread_alive=server.is_alive() if server is not None else False)
        status['control_alive'] = bool(status['chain_running'] and status['server_thread_alive']
                                      and status['motor_worker_alive'] is not False)
        if hasattr(chain, 'get_control_failure'):
            status['failure'] = chain.get_control_failure()
            if status['failure'] is not None:
                status['control_alive'] = False
        try:
            feedback = self.robot.get_feedback_diagnostics()
            for source, target in (('timestamp', 'feedback_timestamp'), ('age_s', 'feedback_age_s')):
                value = feedback.get(source)
                if value is not None and np.isfinite(value):
                    status[target] = float(value)
            age = status['feedback_age_s']
            status['feedback_fresh'] = bool(status['control_alive'] and age is not None and 0 <= age <= .5)
        except Exception as error:
            status['diagnostic_error'] = str(error)
        return status

    def _feedback_health(self):
        """Check feedback without clearing a fault or authorizing new motion."""
        if self.closed:
            raise RuntimeError("Motors are disabled; click Enable motors")
        if self.live:
            status = self.sdk_status()
            if not status['control_alive']:
                self.rest_verified = False
                self.settled_since = None
                self.feedback_warning = 'SDK control stopped; joint, motor and temperature readings are cached; power state unverified'
                detail = (status['failure'] or {}).get('message')
                raise RuntimeError('SDK control thread stopped; inspect hardware before continuing'
                                   + (f': {detail}' if detail else ''))
            age = self.robot.get_feedback_diagnostics()["age_s"]
            fresh = bool(np.isfinite(age) and 0 <= age <= .5)
            self.feedback_warning = f"SDK feedback is stale ({age:.3f} s); current motor state unverified" if not fresh else None
            if self.feedback_warning and time.monotonic()-self._last_feedback_warning >= 5.:
                logging.warning(self.feedback_warning)
                self._last_feedback_warning = time.monotonic()
            return fresh
        self.feedback_warning = None
        return True

    def effective_joint_bounds(self):
        bounds = self.limits.copy()
        for joint in range(6):
            if joint not in self.strict_joint_limits:
                bounds[joint, 0] = min(bounds[joint, 0], self.rest[joint])
                bounds[joint, 1] = max(bounds[joint, 1], self.rest[joint])
        return bounds

    def validate(self, target, *, automatic=False):
        if self.closed:
            raise RuntimeError("Motors are disabled")
        self.health()
        q = np.asarray(target, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite arm radians plus the fixed gripper position")
        if not np.isclose(q[6], self.gripper_target, atol=1e-12, rtol=0):
            raise ValueError("Gripper stays at its captured position during arm testing")
        # Captured rest can lie just outside the nominal URDF range because of encoder zero offsets.
        bounds = self.effective_joint_bounds()
        lo, hi = bounds[:, 0], bounds[:, 1]
        if np.any(q[:6] < lo) or np.any(q[:6] > hi):
            joint = int(np.flatnonzero((q[:6] < lo) | (q[:6] > hi))[0])
            raise MotionRejection("endpoint_validation", "joint_bounds",
                f"Target exceeds the URDF joint range: J{joint+1} = {q[joint]:.6g} rad; "
                f"allowed [{lo[joint]:.6g}, {hi[joint]:.6g}] rad.", joint=joint+1,
                value_rad=float(q[joint]), bounds_rad=[float(lo[joint]), float(hi[joint])])
        if self.live and automatic:
            self._validate_automatic_excursion(q)
        return q.copy()

    def _validate_automatic_excursion(self, q):
        for name, reference in (("current", self.positions()), ("rest", self.rest), ("prior target", self.target)):
            if np.max(np.abs(q[:6] - reference[:6])) >= self.test_excursion:
                joint = int(np.argmax(np.abs(q[:6] - reference[:6])))
                delta = float(abs(q[joint] - reference[joint]))
                raise MotionRejection("endpoint_validation", "automatic_test_excursion",
                    f"Test target must stay within {self.test_excursion:g} rad of current, rest, and prior target: "
                    f"J{joint+1} differs from {name} by {delta:.6g} rad (requires <{self.test_excursion:g} rad).",
                    joint=joint+1, reference=name, excursion_rad=delta, exclusive_limit_rad=self.test_excursion)

    def command(self, target, mode="Joint", *, automatic=False, cartesian=None):
        self._guard_shutdown()
        if self.cooldown_active:
            raise RuntimeError("Cooldown active; wait for motors below 55 C at rest")
        if self.disable_pending:
            raise RuntimeError("Returning to rest for disable; wait for completion")
        if self.motion_plan is not None:
            raise RuntimeError("Motion in progress; wait for the waypoint to finish or select Hold")
        q = self.validate(target, automatic=automatic)
        if (self.motion_planner is not None and np.array_equal(q, self.target)
                and self.mode != "Gravity compensation"):
            self.mode = mode  # An identical operator target preserves the holding integral.
            return
        self._start_waypoint(q, mode, automatic=automatic, cartesian=cartesian)

    def ready_for_waypoint(self):
        if self.shutdown_requested:
            return False
        if self.closed or self.motion_plan is not None or self.disable_pending or self.motion_fault() or self.cooldown_active:
            return False
        if not self.live:
            return True
        self.health()
        return bool(np.max(np.abs(self.robot.get_observations()["joint_vel"][:6])) <= .05)

    def command_tracking(self, tracking, *, automatic=False):
        self._guard_shutdown()
        if self.cooldown_active:
            raise RuntimeError("Cooldown active; wait for motors below 55 C at rest")
        self.health()
        if not self.ready_for_waypoint():
            raise RuntimeError("Tracking needs enabled, idle motors and no pending disable or fault")
        self._ensure_position_speed()
        plan = tracking.motion
        if self.control_law in ("POS_VEL", "MIXED_J45") and plan.report["planned_peaks"]["joint_velocity"] > self.position_speed_limit:
            raise ValueError("Tracking reference exceeds the position-speed commissioning ceiling")
        start = self.positions()
        for s in np.linspace(0, 1, 1001):
            self.validate(plan.path.spline(s), automatic=automatic)
        self._wait_for_fresh_motion_start()
        start = self.positions()
        if np.max(abs(start[:6]-plan.path.positions[0, :6])) > .01:
            raise ValueError("Move to path start before running tracking (joint mismatch >0.01 rad)")
        actual = self.motion_planner.fk(start).copy()
        if tracking.curve.distance(actual[:3, 3]) > POSITION_TOLERANCE_M:
            raise ValueError(f"Path start edge distance must be at most {POSITION_TOLERANCE_M*1000:g} mm; wait for convergence or approach again")
        if self.live:
            self._apply_position_speed_limit()
            self.robot.command_joint_motion(self._sdk_path(plan.path),
                np.zeros(self.sdk_joint_count), self._sdk_positions(np.r_[PID_LIMIT, 0.]),
                kp=self._sdk_positions(np.r_[FACTORY_KP, 5.]),
                kd=self._sdk_positions(np.r_[FACTORY_KD, 0.]),
                integral_deadband=np.deg2rad(self.integral_deadband_deg))
        self.motion_plan = plan
        self.last_motion = plan.report
        self.motion_started = time.monotonic()
        self.motion_automatic = automatic
        self.target = plan.path.positions[-1].copy()
        self.mode = "Path tracking"
        self.rest_verified = False
        self.settled_since = None

    def command_continuous_scan(self, plan):
        """Start a bounded, fully prevalidated named laser raster or vertical line.

        Planning runs on private kinematic data in a worker. This main-owner
        submission verifies its exact artifact/proof against current limits and
        captures fresh quiet SDK state; it never repeats dense planning here.
        """
        from dataclasses import asdict

        from i2rt.robots.waypoint_motion import TimedJointPath
        from scipy.spatial.transform import Rotation

        from autoweld.continuous_raster import RasterPlanBuilder, raster_path_digest
        from autoweld.motion import MotionPlan

        if threading.get_ident() != self._service_owner_thread:
            raise RuntimeError('Continuous scan submission must execute on the service main owner')
        self._guard_shutdown()
        self.health()
        if not self.ready_for_waypoint():
            raise RuntimeError('Continuous scanning needs enabled idle motors without shutdown, cooldown, disable or fault')
        if self.motion_planner is None or type(plan) is not MotionPlan or type(plan.path) is not TimedJointPath:
            raise ValueError('Continuous scanning requires a prevalidated named TimedJointPath MotionPlan')
        report = copy.deepcopy(plan.report)
        kinds = {'Continuous laser raster': 'autoweld.continuous_raster.v1',
                 'Continuous laser vertical line': 'autoweld.continuous_world_line.v1'}
        if not isinstance(report, dict) or report.get('kind') not in kinds:
            raise ValueError('Only a validated continuous laser raster or vertical line may use this submission path')
        vertical_line = report['kind'] == 'Continuous laser vertical line'
        duration = plan.path.duration
        if not np.isfinite(duration) or not 0 < duration <= 900 or report.get('duration_s') != duration:
            raise ValueError('Continuous scan duration must match its validated <=900 s path')
        configured = asdict(self.planner_for_controller().limits)
        limits, peaks = report.get('limits'), report.get('planned_peaks')
        if (not isinstance(limits, dict) or set(limits) != set(configured)
                or not isinstance(peaks, dict) or set(peaks) != set(configured)):
            raise ValueError('Continuous scan must prove all nine configured motion limits')
        for name, expected in configured.items():
            if (not np.isfinite([limits[name], peaks[name]]).all() or limits[name] != expected
                    or peaks[name] < 0 or peaks[name] > expected + 1e-10):
                raise ValueError(f'Continuous scan proof does not match configured {name}')
        if peaks['joint_velocity'] > self.position_speed_limit:
            raise ValueError('Continuous scan exceeds the configured position-speed ceiling')
        proof = report.get('validation_proof', {})
        if (not isinstance(proof, dict) or proof.get('schema') != kinds[report['kind']]
                or proof.get('path_sha256') != raster_path_digest(plan.path)
                or proof.get('collision_checked') is not True or proof.get('command_constraints_checked') is not True):
            raise ValueError('Continuous scan requires matching path, collision and command-constraint proof')
        for name, maximum in (('max_period_s', .01), ('ik_step_limit_rad', .05),
                              ('position_tolerance_m', .0002), ('orientation_tolerance_rad', .0002)):
            value = proof.get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f'Continuous scan has invalid {name} validation proof')
        bounds = self.effective_joint_bounds()
        proved_bounds = np.asarray(proof.get('bounds_rad'), float)
        if proved_bounds.shape != bounds.shape or not np.array_equal(proved_bounds, bounds):
            raise ValueError('Installed joint bounds changed since continuous scan validation')
        positions = plan.path.positions
        if positions.ndim != 2 or positions.shape[1] != 7 or not np.isfinite(positions).all():
            raise ValueError('Continuous scan needs finite seven-joint validated positions')
        if (np.any(positions[:, :6] < bounds[:, 0]) or np.any(positions[:, :6] > bounds[:, 1])
                or not np.allclose(positions[:, 6], self.gripper_target, atol=1e-12, rtol=0)):
            raise ValueError('Continuous scan knots exceed installed bounds or change the captured gripper')
        samples = report.get('validation_samples')
        if not isinstance(samples, int) or samples < int(np.ceil(duration / proof['max_period_s'])) + 1:
            raise ValueError('Continuous scan lacks complete timed interior validation')
        for name in ('tcp_path_error_m', 'tcp_path_error_rad'):
            value = report.get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= .0002:
                raise ValueError('Continuous scan lacks valid Cartesian residual proof')
        scope = report.get('scan_scope', {})
        if not isinstance(scope, dict):
            raise ValueError('Continuous scan is missing its bounded scan scope')
        if not vertical_line:
            for name, maximum in (('width_m', .100), ('height_m', .100), ('depth_bound_m', .060)):
                value = scope.get(name)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= maximum + 1e-12:
                    raise ValueError(f'Continuous scan exceeds its bounded {name} scope')
        baseline = RasterPlanBuilder._transform(scope.get('start_base_from_camera') if vertical_line
            else scope.get('baseline_optical'), 'original scan optical')
        handeye = RasterPlanBuilder._transform(scope.get('tcp_from_camera'), 'scan hand-eye')
        fixed = np.eye(4)
        rotation = np.asarray(scope.get('fixed_optical_rotation'), float)
        if rotation.shape != (3, 3):
            raise ValueError('Continuous scan is missing its fixed optical rotation')
        fixed[:3, :3] = rotation
        RasterPlanBuilder._transform(fixed, 'scan fixed optical')
        if np.linalg.norm(rotation[:, 2] - baseline[:3, 2]) > 1e-6:
            raise ValueError('Continuous scan must retain its original fixed optical axis')
        if vertical_line:
            length = scope.get('length_m')
            direction = np.asarray(scope.get('direction_base'), float)
            if (scope.get('scan_profile') != 'vertical_down' or isinstance(length, bool)
                    or not isinstance(length, (int, float)) or not np.isfinite(length) or not 0 < length <= .100
                    or direction.shape != (3,) or not np.array_equal(direction, [0., 0., -1.])):
                raise ValueError('Continuous vertical line requires a bounded world minus-Z scope')
            endpoint = RasterPlanBuilder._transform(scope.get('end_base_from_camera'), 'vertical scan endpoint')
            if (not np.allclose(endpoint[:3, 3], baseline[:3, 3] + direction * length, atol=1e-9, rtol=0)
                    or not np.allclose(rotation, baseline[:3, :3], atol=1e-9, rtol=0)
                    or not np.allclose(endpoint[:3, :3], rotation, atol=1e-9, rtol=0)):
                raise ValueError('Continuous vertical line endpoint or fixed view disagrees with its scope')
            # Bind scope metadata to the proved joint geometry. This uses the
            # current single model and performs no new trajectory planning.
            for index, q in enumerate(positions):
                camera = np.asarray(self.motion_planner.fk(q.copy())) @ handeye
                offset = camera[:3, 3] - baseline[:3, 3]
                angle = Rotation.from_matrix(camera[:3, :3] @ rotation.T).magnitude()
                if (not np.isfinite(camera).all() or np.linalg.norm(offset[:2]) > .0002
                        or not -.0002 <= -offset[2] <= length + .0002 or angle > .0002
                        or (index == 0 and np.linalg.norm(offset) > .0002)
                        or (index == len(positions) - 1 and np.linalg.norm(offset - direction * length) > .0002)):
                    raise ValueError('Continuous vertical line joint knots disagree with the declared line scope')
        # Own the submission artifact. SDK also clones it, retaining local time.
        submitted = MotionPlan(TimedJointPath(plan.path.knots, positions, duration), report)
        self.validate(submitted.path.positions[0], automatic=False)
        self.validate(submitted.path.positions[-1], automatic=False)
        self._ensure_position_speed()
        self._wait_for_fresh_motion_start()
        self.health()
        if not self.ready_for_waypoint():
            raise RuntimeError('Continuous scan lost fresh quiet readiness after validation')
        start = self.positions()
        if np.max(abs(start[:6] - submitted.path.positions[0, :6])) > REST_TOL:
            raise ValueError('Move to continuous scan start first (joint mismatch exceeds 0.5 degrees)')
        actual = self.motion_planner.fk(start).copy()
        expected = self.motion_planner.fk(submitted.path.positions[0].copy()).copy()
        position_error = np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        angle_error = np.linalg.norm(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).as_rotvec())
        if position_error > .0005 or angle_error > .005:
            raise ValueError('Continuous scan start must converge within 0.5 mm / 0.005 rad')
        if self.live:
            self._apply_position_speed_limit()
            self.robot.command_joint_motion(self._sdk_path(submitted.path),
                np.zeros(self.sdk_joint_count), self._sdk_positions(np.r_[PID_LIMIT, 0.]),
                kp=self._sdk_positions(np.r_[FACTORY_KP, 5.]),
                kd=self._sdk_positions(np.r_[FACTORY_KD, 0.]),
                integral_deadband=np.deg2rad(self.integral_deadband_deg))
        self.motion_plan = submitted
        self.last_motion = report
        self.motion_started = time.monotonic()
        self.motion_automatic = False
        self.target = submitted.path.positions[-1].copy()
        self.mode = 'Laser scanning'
        self.rest_verified = False
        self.settled_since = None

    def _start_waypoint(self, q, mode, *, automatic=False, cartesian=None, planner=None):
        self._ensure_position_speed()
        if self.motion_planner is None:
            self._send(q, mode)
            return
        start = self.positions()
        start[6] = self.gripper_target
        if np.max(np.abs(q-start)) < 1e-10:
            self._send(q, mode)
            return
        if self.live:
            velocity = np.asarray(self.robot.get_observations()["joint_vel"], float)[:6]
            if velocity.shape != (6,) or not np.isfinite(velocity).all() or np.max(abs(velocity)) > .05:
                # A cached quiet admission can race the next measured velocity
                # sample. Use the same bounded fresh/quiet gate as after planning
                # rather than permanently rejecting one transient idle sample.
                self._wait_for_fresh_motion_start()
                start = self.positions()
                start[6] = self.gripper_target
        bounds = self.effective_joint_bounds()
        plan = (planner if planner is not None else self.planner_for_controller()).plan(start, q, cartesian=cartesian, bounds=bounds,
            validate=lambda sample: self.validate(sample, automatic=automatic), collision=self.motion_collision)
        self._wait_for_fresh_motion_start()
        self.health()
        if np.max(np.abs(self.positions()[:6]-start[:6])) > REST_TOL:
            raise RuntimeError("Arm moved during planning; retry from the current pose")
        if self.live:
            self._apply_position_speed_limit()
            self.robot.command_joint_motion(self._sdk_path(plan.path),
                np.zeros(self.sdk_joint_count), self._sdk_positions(np.r_[PID_LIMIT, 0.]),
                kp=self._sdk_positions(np.r_[FACTORY_KP, 5.]),
                kd=self._sdk_positions(np.r_[FACTORY_KD, 0.]),
                integral_deadband=np.deg2rad(self.integral_deadband_deg))
        self.motion_plan = plan
        self.last_motion = plan.report
        self.motion_started = time.monotonic()
        self.motion_automatic = automatic
        self.target = plan.path.positions[-1].copy()
        self.mode = mode
        self.rest_verified = False
        self.settled_since = None

    def _wait_for_fresh_motion_start(self):
        """Do not start a reference on feedback aged by CPU-heavy validation.

        Require a new SDK exchange after validation, not just the normal idle
        health threshold. The independent 400 ms running watchdog is unchanged.
        """
        if not self.live:
            return
        after = time.time()
        deadline = time.monotonic()+.5
        saw_nonquiet = False
        while time.monotonic() < deadline:
            self.health()
            feedback = self.robot.get_feedback_diagnostics()
            if (np.isfinite([feedback['timestamp'], feedback['age_s']]).all()
                    and feedback['timestamp'] > after and 0 <= feedback['age_s'] <= .02):
                velocity = np.asarray(self.robot.get_observations()['joint_vel'], float)[:6]
                if velocity.shape == (6,) and np.isfinite(velocity).all() and np.max(abs(velocity)) <= .05:
                    return
                saw_nonquiet = True
            time.sleep(.002)
        if saw_nonquiet:
            raise RuntimeError("Arm moved during validation; no fresh quiet feedback within 0.5 s; no motion started")
        raise RuntimeError("No fresh SDK feedback after path validation; no motion started")

    def planner_for_controller(self):
        if self.control_law not in ("POS_VEL", "MIXED_J45"):
            return self.motion_planner
        planner = copy.copy(self.motion_planner)
        planner.limits = replace(planner.limits,
            joint_velocity=min(planner.limits.joint_velocity, self.position_speed_limit))
        return planner

    def _apply_position_speed_limit(self):
        if self.live and self.control_law in ("POS_VEL", "MIXED_J45") and hasattr(self.robot, "configure_position_speed_limit"):
            limit = self.position_speed_limit
            if self.motion_planner:
                limit = min(limit, self.motion_planner.limits.joint_velocity)
            self.robot.configure_position_speed_limit(limit)

    def configure_position_speed_limit(self, limit):
        self._guard_shutdown()
        if self.control_law not in ("POS_VEL", "MIXED_J45"):
            raise ValueError("Select position-speed before changing its speed ceiling")
        if self.motion_plan is not None or self.disable_pending or self.motion_fault():
            raise RuntimeError("Wait for healthy idle control before changing the speed ceiling")
        if not np.isfinite(limit) or not 0 < limit <= MAX_POSITION_SPEED_LIMIT:
            raise ValueError("Position-speed ceiling must be in (0, 3] rad/s")
        if self.motion_planner and limit > self.motion_planner.limits.joint_velocity:
            raise ValueError("Position-speed ceiling exceeds the operator joint velocity limit")
        if not self.closed:
            self.health()
            if self.live and np.max(abs(self.robot.get_observations()["joint_vel"][:6])) > .05:
                raise RuntimeError("Wait for the arm to stop before changing its speed ceiling")
        self.position_speed_limit = float(limit)
        self._apply_position_speed_limit()

    def _send(self, q, mode):
        self._ensure_position_speed()
        self.motion_plan = None
        self._apply_position_speed_limit()
        if hasattr(self.robot, "disable_gravity_comp"):
            self.robot.disable_gravity_comp()  # SDK sim API; live gravity feedforward stays on.
        self.robot.command_joint_pos(self._sdk_positions(q))
        self.target = q
        self.mode = mode
        self.rest_verified = False
        self.settled_since = None

    def select_controller(self, name, ki=None, deadband_deg=None, kp=None, kd=None):
        self._guard_shutdown()
        if name != "POS_VEL":
            raise ValueError("Only position-speed (POS_VEL) is supported for position moves; select Gravity compensation for torque mode")
        if any(value is not None for value in (ki, kp, kd, deadband_deg)):
            raise ValueError("Position-speed uses onboard gains; MIT tuning is unavailable")
        if not self.closed and self.control_law != "POS_VEL":
            self.hold()
        elif self.closed:
            self.control_law = "POS_VEL"
        self.command_quantization = False

    def configure_gravity(self, factors, model="sdk"):
        if self.motion_plan is not None:
            raise RuntimeError("Wait for motion to finish before changing feedforward")
        gains = np.asarray(factors, dtype=float)
        if gains.shape != (6,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Expected six finite positive gravity factors")
        if model not in ("sdk", "linear"):
            raise ValueError("Gravity model must be sdk or linear")
        if self.disable_pending:
            raise RuntimeError("Wait for rest and disable to complete")
        if self.cooldown_active:
            raise RuntimeError("Wait for motor cooldown before changing gravity compensation")
        if not self.closed:
            self.health()
            if self.control_law == "MIT":
                self.robot.configure_gravity(gains, model=model, reference=self.rest[:6])
        self.gravity_factors = gains.copy()
        self.gravity_model = model

    def hold(self, *, automatic=False):
        self._guard_shutdown()
        # The SDK may need a damped stop while leaving manual gravity mode.
        # Capture after that stop, rather than pulling back to its earlier pose.
        self.health()
        self._ensure_position_speed()
        q = self.positions()
        q[6] = self.gripper_target
        self.validate(q, automatic=automatic)
        self._send(q, "Hold")

    @property
    def gripper_target(self):
        return 0.0 if self.gripper_hold_closed else float(self.rest[6])

    def close_gripper(self):
        self._guard_shutdown()
        """Clamp a held board using the SDK's existing force limiter and gains."""
        if self.sdk_joint_count == 6:
            raise RuntimeError("No gripper motor is installed on the receiver/camera mount")
        self.health()
        if not self.ready_for_waypoint():
            raise RuntimeError("Close the gripper only with the arm idle")
        if self.live and self.robot.get_robot_info().get("limit_gripper_effort", -1) <= 0:
            raise RuntimeError("Closing requires the SDK gripper force limiter")
        self.gripper_hold_closed = True
        self.hold()

    def configure_friction(self, enabled=True, transition_rad_s=None, coefficients=None):
        if self.control_law in ("POS_VEL", "MIXED_J45") and enabled:
            raise ValueError("Position-speed has no friction torque feedforward")
        if self.motion_plan is not None:
            raise RuntimeError("Wait for motion to finish before changing feedforward")
        width = self.friction_transition.copy() if transition_rad_s is None else np.asarray(transition_rad_s, dtype=float)
        coeff = self.friction_coefficients.copy() if coefficients is None else np.asarray(coefficients, dtype=float)
        if not isinstance(enabled, bool):
            raise ValueError("Friction enabled must be boolean")
        for values in (width, coeff):
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Expected six finite nonnegative friction values")
        if self.disable_pending:
            raise RuntimeError("Wait for rest and disable to complete")
        if not self.closed:
            self.health()
            if self.live:
                self.robot.configure_friction(enabled=enabled, transition_rad_s=width, coefficients=coeff)
        self.friction_enabled = enabled
        self.friction_transition = width.copy()
        self.friction_coefficients = coeff.copy()

    def start_hold(self):
        """Apply selected feedforward and gains at the current pose on app startup."""
        self.configure_friction(self.friction_enabled, self.friction_transition, self.friction_coefficients)
        self.hold()

    def gravity(self):
        self._guard_shutdown()
        if self.disable_pending:
            raise RuntimeError("Returning to rest for disable; wait for completion")
        if self.cooldown_active:
            raise RuntimeError("Cooldown active; gravity compensation is unavailable until motors cool")
        self.health()
        if self.sdk_joint_count != 6:
            raise ValueError("Gravity compensation requires the installed six-axis receiver/camera robot")
        if self.feedback_warning:
            raise RuntimeError("Fresh feedback is required before entering gravity compensation")
        if self.near_gravity_joint_limit():
            raise RuntimeError('Move inward from the calibrated joint limit before selecting gravity compensation')
        if self.motion_plan is not None:
            # Cancel the future reference at the measured pose. The SDK then
            # captures a damped hold and verifies quiet before committing MIT.
            self.hold()
        if self.mode == "Gravity compensation" and self.control_law == "MIT":
            return
        self._switch_arm_control_mode("MIT")
        try:
            self.robot.configure_gravity(self.gravity_factors, model=self.gravity_model, reference=self.rest[:6])
            if hasattr(self.robot, "settle_measured_hold"):
                self.robot.settle_measured_hold()
            if hasattr(self.robot, "enter_gravity_comp_idle"):
                self.robot.enter_gravity_comp_idle()
            elif not self.live and hasattr(self.robot, "enable_gravity_comp"):
                self.robot.enable_gravity_comp()
            else:
                raise RuntimeError("SDK gravity compensation API is unavailable")
        except Exception as error:
            try:
                self._switch_arm_control_mode("POS_VEL")
                self._send(self.positions(), "Hold")
                self.mode_transition_error = str(error)
            except Exception as rollback_error:
                self.mode_transition_error = f"Gravity entry failed: {error}; position-speed recovery failed: {rollback_error}"
                self.mode = "Mode transition failed"
            raise
        self.motion_plan = None
        self.target = self.positions()
        self.mode = "Gravity compensation"
        self.rest_verified = False
        self.settled_since = None

    def _switch_arm_control_mode(self, mode):
        """Only the SDK may change motor modes or coordinate its worker threads."""
        if mode not in ("MIT", "POS_VEL"):
            raise ValueError("Arm control mode must be MIT or POS_VEL")
        with self._mode_lock:
            actual = (self.robot.get_arm_control_modes() if hasattr(self.robot, "get_arm_control_modes")
                      else [self.control_law] * 6)
            if self.control_law == mode and all(value == mode for value in actual):
                return
            self.health()
            if self.feedback_warning:
                raise RuntimeError("Fresh feedback is required to switch arm control mode")
            self.mode_transition = True
            self.mode_transition_error = None
            try:
                if mode == "MIT" and hasattr(self.robot, "prepare_gravity_profile"):
                    # Cache the selected model/factors before the firmware
                    # switch so its first powered MIT hold uses that profile.
                    self.robot.prepare_gravity_profile(self.gravity_factors, model=self.gravity_model,
                                                       reference=self.rest[:6])
                if hasattr(self.robot, "switch_arm_control_mode"):
                    self.robot.switch_arm_control_mode(mode)
                elif self.live:
                    raise RuntimeError("SDK in-place arm mode switching is unavailable; current control remains active")
                self.control_law = mode
                self.target = self.positions()
                self.motion_plan = None
                self.rest_verified = False
                self.settled_since = None
            except Exception as error:
                self.mode_transition_error = str(error)
                raise
            finally:
                self.mode_transition = False

    def _ensure_position_speed(self):
        actual = (self.robot.get_arm_control_modes() if hasattr(self.robot, "get_arm_control_modes")
                  else [self.control_law] * 6)
        if self.control_law != "POS_VEL" or any(value != "POS_VEL" for value in actual):
            self._switch_arm_control_mode("POS_VEL")

    def return_rest(self, *, automatic=False):
        self._guard_shutdown()
        self.health()
        if self.disable_pending:
            return
        self._ensure_position_speed()
        if self.motion_plan is not None:
            raise RuntimeError("Wait for the current waypoint to finish before returning to rest")
        self.return_automatic = automatic
        self._next_return_step()

    def _next_return_step(self):
        target = self.rest.copy()
        target[6] = self.gripper_target
        q = self.validate(target, automatic=self.return_automatic)
        self._start_waypoint(q, "Returning to rest", automatic=self.return_automatic)
        self.return_step_started = time.monotonic()

    def _preflight_disable_return(self, automatic):
        """Reject an inadmissible return before setting the motion barrier."""
        target = self.rest.copy()
        target[6] = self.gripper_target
        self.validate(target, automatic=automatic)
        if self.motion_planner is not None and (self.motion_plan is not None or not self._quiet()):
            # A queued return starts at the existing profile's endpoint. Recheck
            # against actual feedback before execution, since coasting can drift.
            start = (self.motion_plan.path.positions[-1].copy() if self.motion_plan is not None
                     else self.positions())
            start[6] = self.gripper_target
            self.planner_for_controller().plan(start, target, bounds=self.effective_joint_bounds(),
                validate=lambda q: self.validate(q, automatic=automatic), collision=self.motion_collision)

    def _abort_disable_return(self, error):
        self.disable_pending = False
        self.return_automatic = False
        self.rest_verified = False
        self.settled_since = None
        self.return_step_started = None
        self.disable_warning = f'Return for disable rejected; request cancelled and motors remain enabled: {error}'
        try:
            self.hold()  # Measured-position stop only; never an out-of-scope rest move.
        except (ValueError, RuntimeError) as hold_error:
            self.disable_warning += f'; SDK measured hold could not be refreshed: {hold_error}'
        logging.warning(self.disable_warning)

    def request_disable(self, *, automatic=False):
        self._guard_shutdown()
        if self.closed:
            return
        if self.disable_pending:
            if automatic or not self.return_automatic:
                return  # Duplicate requests never restart the settling timer.
            # An explicit operator retry may take ownership of an automatic
            # return, without changing the scope of any other automatic motion.
            if not self.motion_fault():
                try:
                    self._preflight_disable_return(False)
                except (ValueError, RuntimeError) as error:
                    self._abort_disable_return(error)
                    raise
            else:
                self._feedback_health()
            self.return_automatic = False
            self.disable_warning = None
            if self.mode == 'Returning to rest':
                self.motion_automatic = False
            return
        if self.motion_fault():
            self._feedback_health()
            if automatic and self.live:
                target = self.rest.copy()
                target[6] = self.gripper_target
                self._validate_automatic_excursion(target)
            self.disable_pending = True
            self.return_automatic = automatic
            self.rest_verified = False
            self.settled_since = None
            return
        self.health()
        self._preflight_disable_return(automatic)
        self._ensure_position_speed()
        if self.motion_plan is not None or not self._quiet():
            self.disable_pending = True
            self.return_automatic = automatic
            self.disable_warning = None
            self.rest_verified = False
            self.settled_since = None
            return
        # Path planning, collision checks and SDK submission finish before the
        # pending flag is published. A rejection leaves normal controls usable.
        self.return_rest(automatic=automatic)
        self.disable_pending = True
        self.disable_warning = None

    def _within(self, target):
        q = self.positions()
        arm_ok = np.max(np.abs(q[:6] - target[:6])) <= REST_TOL
        if self.gripper_hold_closed:
            # Object contact intentionally prevents reaching the closed target.
            # The requested disable still powers off all motors after arm rest.
            return bool(arm_ok)
        span = abs(np.diff(self.gripper_limits)[0])
        return bool(arm_ok and abs(q[6] - target[6]) * span <= REST_TOL)

    def at_rest(self):
        return self._within(self.rest)

    def _quiet(self):
        return not self.live or bool(np.max(np.abs(self.robot.get_observations()["joint_vel"][:6])) <= .05)

    def _check_cooldown(self):
        """Use cached SDK thermometry; cool while holding the captured rest.

        The SDK restores position-speed control before returning to rest.
        Firmware remains the final thermal protection.
        """
        if not hasattr(self.robot, "get_temperature_state"):
            return
        state = self.robot.get_temperature_state()
        values = np.asarray(list(state.get("mos_c", [])) + list(state.get("rotor_c", [])), float)
        if not values.size or not np.isfinite(values).all():
            return
        self.cooldown_max_c = float(values.max())
        if self.cooldown_max_c >= COOLDOWN_C and not self.cooldown_active:
            self.cooldown_active = True
            logging.warning("Motor temperature %.1f C: returning to captured rest for cooldown", self.cooldown_max_c)
            if not self.feedback_warning and (self.control_law == "MIT" or self._quiet()):
                self._ensure_position_speed()
                self._send(self.positions(), "Cooldown stopping")
            else:
                self.mode = "Cooldown stopping"
        if not self.cooldown_active:
            return
        if (self.mode != "Returning to rest" and not self.rest_verified
                and self._quiet() and not self.feedback_warning):
            self.return_automatic = False
            self._next_return_step()
        elif self.rest_verified and self.at_rest():
            if self.cooldown_max_c < COOLDOWN_RESUME_C and not self.disable_pending:
                self.cooldown_active = False
                self.mode = "Hold"

    def select_mode(self, mode):
        """Switch operation through the single SDK owner without reconnecting."""
        self._guard_shutdown()
        if mode == "Hold":
            self.hold()
        elif mode in ("IK", "Inverse kinematics"):
            self.hold()
            self.mode = "IK"
        elif mode in ("Gravity compensation", "Gravity comp"):
            self.gravity()
        else:
            raise ValueError("Mode must be Hold, IK or Gravity compensation")

    def near_gravity_joint_limit(self):
        if not self.strict_joint_limits:
            return False
        q = self.positions()[:6]
        velocity = (np.asarray(self.robot.get_observations()['joint_vel'])[:6]
                    if self.live else np.zeros(6))
        projected = q + .2*velocity
        margin = np.deg2rad(2.)
        return any(min(q[i], projected[i]) <= self.limits[i, 0]+margin
                   or max(q[i], projected[i]) >= self.limits[i, 1]-margin
                   for i in self.strict_joint_limits)

    def tick(self):
        if self.shutdown_requested:
            try:
                self._tick_shutdown()
            except Exception as error:
                self._block_shutdown(error)
            return
        pending = self.disable_pending
        try:
            self._tick()
        except (ValueError, RuntimeError) as error:
            if (pending or self.disable_pending) and not self.closed and not self.motion_fault():
                self._abort_disable_return(error)
            raise

    def _tick(self):
        fault = self.motion_fault()
        if fault and not self.closed:
            self.motion_plan = None
            self.mode = "Oscillation fault"
            self.target = self._logical_positions(fault["hold_q"])
            if self.disable_pending:
                # A latched fault blocks new motion, not a verified-rest power-off.
                # Away from rest, keep the request pending for explicit recovery.
                self._feedback_health()
                quiet = (not self.live or
                         np.max(np.abs(self.robot.get_observations()['joint_vel'][:6])) <= .05)
                if self.at_rest() and quiet and not self.feedback_warning:
                    if self.settled_since is None:
                        self.settled_since = time.monotonic()
                    if time.monotonic() - self.settled_since >= 1.0:
                        self.rest_verified = True
                        self._disable_at_rest()
                else:
                    self.rest_verified = False
                    self.settled_since = None
            else:
                self.rest_verified = False
                self.settled_since = None
            return
        if self.closed:
            return
        self.health()
        if self.mode == 'Gravity compensation' and self.near_gravity_joint_limit():
            self.hold()
            logging.warning('Calibrated joint travel margin reached; gravity compensation switched to Hold')
        self._check_cooldown()
        if self.motion_plan is not None:
            elapsed = time.monotonic()-self.motion_started
            if self.live and self.motion_automatic and np.max(np.abs(self.positions()[:6]-self.rest[:6])) >= self.test_excursion:
                self._send(self.positions(), "Hold")
                raise RuntimeError(f"Measured automatic excursion exceeded {self.test_excursion:g} rad; holding measured pose")
            if not self.live:
                self.robot.command_joint_pos(self._sdk_positions(self.motion_plan.path.sample(elapsed)[0]))
            if elapsed < self.motion_plan.path.duration + (.02 if self.live else 0):
                return
            self.motion_plan = None
            if self.mode == "Returning to rest":
                self.return_step_started = time.monotonic()  # Settling timeout starts after the profile.
        if self.disable_pending and self.mode != "Returning to rest":
            if self.live and np.max(np.abs(self.robot.get_observations()["joint_vel"][:6])) > .05:
                return
            self._next_return_step()
            return
        if (self.mode == "Returning to rest" and self.at_rest()
                and self._quiet() and not self.feedback_warning):
            if self.settled_since is None:
                self.settled_since = time.monotonic()
            if time.monotonic() - self.settled_since >= 1.0:
                self.rest_verified = True
                if self.disable_pending:
                    self._disable_at_rest()
                return
        else:
            self.settled_since = None
            self.rest_verified = False
        if (
            self.mode == "Returning to rest"
            and self.return_step_started is not None
            and time.monotonic() - self.return_step_started > RETURN_STEP_TIMEOUT
        ):
            self.disable_pending = False
            self.mode = "Hold"
            raise RuntimeError(
                "Rest was not reached within 15 seconds; motors remain enabled at the last target"
            )

    def _disable_at_rest(self):
        if self.motion_fault():
            self._feedback_health()
        else:
            self.health()
        if not self.rest_verified or not self.at_rest() or not self._quiet() or self.feedback_warning:
            raise RuntimeError("Rest verification failed; motors remain enabled")
        self.disabled_q = self.positions()
        self.robot.close()
        self.closed = True
        self.disable_pending = False
        self.mode = "Disabled"

    def enable(self):
        self._guard_shutdown()
        if not self.closed:
            return self.robot
        if self.robot_factory is None:
            raise RuntimeError("No SDK reconnect factory configured")
        # Factory initializes SDK hold at freshly measured positions, never an old motion target.
        kwargs = {"arm_control_mode": "POS_VEL"}
        robot = self.robot_factory(self.gripper_limits.copy(), self._sdk_positions(self.disabled_q), **kwargs)
        q = np.array(robot.get_joint_pos(), dtype=float, copy=True)
        if self.sdk_joint_count == 6 and q.shape == (6,):
            q = np.r_[q, 0.]
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid SDK feedback after enabling")
        self.robot = robot
        self.control_law = "POS_VEL"
        self.closed = False
        self.cached_rest_display = False
        self.target = q
        self.mode = "Hold"
        self.disable_pending = False
        self.rest_verified = False
        self.settled_since = None
        self.return_step_started = None
        if self.live and self.control_law != "POS_VEL" and hasattr(robot, "configure_gravity"):
            robot.configure_gravity(self.gravity_factors, model=self.gravity_model, reference=self.rest[:6])
        if self.live:
            robot.configure_friction(enabled=self.friction_enabled, transition_rad_s=self.friction_transition,
                                     coefficients=self.friction_coefficients)
        if self.live and hasattr(robot, "configure_command_quantization"):
            robot.configure_command_quantization(self.command_quantization)
        self._send(q.copy(), "Hold")
        if self.gripper_hold_closed:
            self.hold()
        return robot
