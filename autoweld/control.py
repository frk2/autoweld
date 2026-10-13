"""Command policy independent of transport; all hardware I/O stays in i2rt."""

import copy
import time
from dataclasses import replace

import numpy as np

from autoweld.diagnostics import MotionRejection
from autoweld.tracking_limits import POSITION_TOLERANCE_M

ANGLE_TOL = np.deg2rad([0.2, 0.2, 0.2, 1.0, 1.0, 1.0])
REST_TOL = float(np.deg2rad(0.5))
TEST_EXCURSION = 0.49  # Strictly below the user's 0.5 rad test authorization.
RETURN_STEP_TIMEOUT = 15.0
# Original PD and fault recovery always retain the factory gains.
FACTORY_KP = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
FACTORY_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
# User-selected gravity-calibrated candidate; overshoot limits: evidence/smooth/REPORT.md.
PID_KP = np.array([80.0, 96.0, 152.8823808, 29.85984, 14.4, 61.917364224])
PID_KI = np.zeros(6)  # User selection: native motor PD only; host I disabled.
PID_KD = np.array([5.0, 5.0, 5.0, 2.592, 1.5, 2.16])
DEFAULT_GRAVITY_FACTORS = np.array([1.0, 1.04, 1.31, 1.32, 1.0, 1.0])
PID_LIMIT = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])


class Controller:
    def __init__(self, robot, limits, live=False, robot_factory=None):
        self.robot = robot
        self.control_law = "PID"
        self.position_speed_limit = .15
        self.test_excursion = TEST_EXCURSION
        self.ki = PID_KI.copy()
        self.gravity_factors = DEFAULT_GRAVITY_FACTORS.copy()
        self.gravity_model = "linear"
        self.friction_enabled = False  # Latest user selection: omit friction feedforward.
        self.friction_transition = np.zeros(6)
        self.friction_coefficients = np.array([.3, .3, .3, .06, .06, .06])
        self.integral_deadband_deg = 0.06
        self.kp = PID_KP.copy()
        self.kd = PID_KD.copy()
        self.live = live
        self.command_quantization = False
        self.robot_factory = robot_factory
        self.gripper_limits = np.asarray(robot.get_robot_info()["gripper_limits"]).copy()
        self.limits = np.asarray(limits)
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
        self.return_automatic = False
        self.return_step_started = None
        self.disabled_q = None
        self.motion_planner = None
        self.motion_collision = None
        self.motion_plan = None
        self.last_motion = None
        self.motion_started = None
        self.motion_automatic = False

    def configure_long_cruise_test(self, enabled):
        """Explicit, nonpersistent exception for the user-approved 500 mm sweep."""
        if not self.closed:
            raise RuntimeError("Change the test excursion exception only with motors disabled")
        self.test_excursion = 1.55 if enabled else TEST_EXCURSION

    def positions(self):
        if self.closed:
            return self.disabled_q.copy()
        q = np.array(self.robot.get_joint_pos(), dtype=float, copy=True)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid SDK joint feedback")
        return q

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
        fault = self.motion_fault()
        self.robot.acknowledge_motion_fault()
        if self.control_law not in ("POS_VEL", "MIXED_J45"):
            self.control_law = "PD"
        self.ki.fill(0.0)
        self.kp = FACTORY_KP.copy()
        self.kd = FACTORY_KD.copy()
        self.target = np.array(fault["hold_q"]) if fault else self.positions()
        self.mode = "Hold"

    def health(self):
        self._feedback_health()
        if self.motion_fault():
            raise RuntimeError("Oscillation fault latched; acknowledge after the arm is quiet")

    def _feedback_health(self):
        """Check feedback without clearing a fault or authorizing new motion."""
        if self.closed:
            raise RuntimeError("Motors are disabled; click Enable motors")
        if self.live:
            if not self.robot.motor_chain.running or not self.robot._server_thread.is_alive():
                raise RuntimeError("SDK control thread stopped; inspect hardware before continuing")
            with self.robot._state_lock:
                timestamp = self.robot._joint_state.timestamp
            if time.time() - timestamp > 0.5:
                raise RuntimeError("SDK feedback is stale")

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
        lo = np.minimum(self.limits[:, 0], self.rest[:6])
        hi = np.maximum(self.limits[:, 1], self.rest[:6])
        if np.any(q[:6] < lo) or np.any(q[:6] > hi):
            joint = int(np.flatnonzero((q[:6] < lo) | (q[:6] > hi))[0])
            raise MotionRejection("endpoint_validation", "joint_bounds",
                f"Target exceeds the URDF joint range: J{joint+1} = {q[joint]:.6g} rad; "
                f"allowed [{lo[joint]:.6g}, {hi[joint]:.6g}] rad.", joint=joint+1,
                value_rad=float(q[joint]), bounds_rad=[float(lo[joint]), float(hi[joint])])
        if self.live and automatic:
            for name, reference in (("current", self.positions()), ("rest", self.rest), ("prior target", self.target)):
                if np.max(np.abs(q[:6] - reference[:6])) >= self.test_excursion:
                    joint = int(np.argmax(np.abs(q[:6] - reference[:6])))
                    delta = float(abs(q[joint] - reference[joint]))
                    raise MotionRejection("endpoint_validation", "automatic_test_excursion",
                        f"Test target must stay within {self.test_excursion:g} rad of current, rest, and prior target: "
                        f"J{joint+1} differs from {name} by {delta:.6g} rad (requires <{self.test_excursion:g} rad).",
                        joint=joint+1, reference=name, excursion_rad=delta, exclusive_limit_rad=self.test_excursion)
        return q.copy()

    def command(self, target, mode="Joint", *, automatic=False, cartesian=None):
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
        if self.closed or self.motion_plan is not None or self.disable_pending or self.motion_fault():
            return False
        if not self.live:
            return True
        self.health()
        return bool(np.max(np.abs(self.robot.get_observations()["joint_vel"][:6])) <= .05)

    def command_tracking(self, tracking, *, automatic=False):
        self.health()
        if not self.ready_for_waypoint():
            raise RuntimeError("Tracking needs enabled, idle motors and no pending disable or fault")
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
            self.robot.command_joint_motion(plan.path,
                np.r_[self.ki if self.control_law in ("PID", "MIXED_J45") else np.zeros(6), 0.], np.r_[PID_LIMIT, 0.],
                kp=np.r_[self.kp if self.control_law in ("PID", "MIXED_J45") else FACTORY_KP, 5.],
                kd=np.r_[self.kd if self.control_law in ("PID", "MIXED_J45") else FACTORY_KD, 0.],
                integral_deadband=np.deg2rad(self.integral_deadband_deg))
        self.motion_plan = plan
        self.last_motion = plan.report
        self.motion_started = time.monotonic()
        self.motion_automatic = automatic
        self.target = plan.path.positions[-1].copy()
        self.mode = "Path tracking"
        self.rest_verified = False
        self.settled_since = None

    def _start_waypoint(self, q, mode, *, automatic=False, cartesian=None):
        if self.motion_planner is None:
            self._send(q, mode)
            return
        start = self.positions()
        start[6] = self.gripper_target
        if np.max(np.abs(q-start)) < 1e-10:
            self._send(q, mode)
            return
        if self.live and np.max(np.abs(self.robot.get_observations()["joint_vel"][:6])) > .05:
            raise RuntimeError("Wait for the arm to stop before a rest-to-rest waypoint")
        bounds = self.limits.copy()
        bounds[:, 0] = np.minimum(bounds[:, 0], self.rest[:6])
        bounds[:, 1] = np.maximum(bounds[:, 1], self.rest[:6])
        plan = self.planner_for_controller().plan(start, q, cartesian=cartesian, bounds=bounds,
            validate=lambda sample: self.validate(sample, automatic=automatic), collision=self.motion_collision)
        self._wait_for_fresh_motion_start()
        self.health()
        if np.max(np.abs(self.positions()[:6]-start[:6])) > REST_TOL:
            raise RuntimeError("Arm moved during planning; retry from the current pose")
        if self.live:
            self._apply_position_speed_limit()
            self.robot.command_joint_motion(plan.path,
                np.r_[self.ki if self.control_law in ("PID", "MIXED_J45") else np.zeros(6), 0.], np.r_[PID_LIMIT, 0.],
                kp=np.r_[self.kp if self.control_law in ("PID", "MIXED_J45") else FACTORY_KP, 5.],
                kd=np.r_[self.kd if self.control_law in ("PID", "MIXED_J45") else FACTORY_KD, 0.],
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
        health threshold. The independent 100 ms running watchdog is unchanged.
        """
        if not self.live:
            return
        after = time.time()
        deadline = time.monotonic()+.5
        while time.monotonic() < deadline:
            self.health()
            feedback = self.robot.get_feedback_diagnostics()
            if feedback['timestamp'] > after and feedback['age_s'] <= .02:
                if np.max(abs(self.robot.get_observations()['joint_vel'][:6])) > .05:
                    raise RuntimeError("Arm moved during validation; wait for it to stop")
                return
            time.sleep(.002)
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
        if self.control_law not in ("POS_VEL", "MIXED_J45"):
            raise ValueError("Select position-speed before changing its speed ceiling")
        if self.motion_plan is not None or self.disable_pending or self.motion_fault():
            raise RuntimeError("Wait for healthy idle control before changing the speed ceiling")
        if not np.isfinite(limit) or not 0 < limit <= 1.:
            raise ValueError("Position-speed ceiling must be in (0, 1] rad/s")
        if self.motion_planner and limit > self.motion_planner.limits.joint_velocity:
            raise ValueError("Position-speed ceiling exceeds the operator joint velocity limit")
        if not self.closed:
            self.health()
            if self.live and np.max(abs(self.robot.get_observations()["joint_vel"][:6])) > .05:
                raise RuntimeError("Wait for the arm to stop before changing its speed ceiling")
        self.position_speed_limit = float(limit)
        self._apply_position_speed_limit()

    def _send(self, q, mode):
        self.motion_plan = None
        self._apply_position_speed_limit()
        if hasattr(self.robot, "disable_gravity_comp"):
            self.robot.disable_gravity_comp()  # SDK sim API; live gravity feedforward stays on.
        if self.control_law in ("PID", "MIXED_J45") and self.live:
            self.robot.command_joint_pos_pid(
                q.copy(),
                np.r_[self.ki, 0.0],
                np.r_[PID_LIMIT, 0.0],
                integral_deadband=np.deg2rad(self.integral_deadband_deg),
                integral_delay=0.0,
                kp=np.r_[self.kp, 5.0],
                kd=np.r_[self.kd, 0.0],
            )
        else:
            self.robot.command_joint_pos(q.copy())
        self.target = q
        self.mode = mode
        self.rest_verified = False
        self.settled_since = None

    def select_controller(self, name, ki=None, deadband_deg=None, kp=None, kd=None):
        if name not in ("PD", "PID", "POS_VEL", "MIXED_J45"):
            raise ValueError("Controller must be PD, PID, POS_VEL or MIXED_J45")
        if ({"PD": "MIT", "PID": "MIT"}.get(name, name)
                != {"PD": "MIT", "PID": "MIT"}.get(self.control_law, self.control_law)):
            if self.live and not self.closed:
                raise RuntimeError("Disable motors at verified rest, select the controller, then Enable motors")
        if name == "POS_VEL" and any(x is not None for x in (kp, kd, deadband_deg)):
            raise ValueError("Position-speed uses onboard cascade gains; MIT tuning is unavailable")
        if name in ("POS_VEL", "MIXED_J45") and ki is not None and np.any(np.asarray(ki) != 0):
            raise ValueError("Position-speed has no host integral torque input")
        if self.disable_pending:
            raise RuntimeError("Wait for rest and disable to complete")
        if self.motion_plan is not None:
            raise RuntimeError("Wait for motion to finish before changing gains")
        gains = self.ki.copy() if ki is None else np.asarray(ki, dtype=float)
        if gains.shape != (6,) or not np.isfinite(gains).all() or np.any(gains < 0):
            raise ValueError("Expected six finite nonnegative integral gains")
        p_gain = self.kp.copy() if kp is None else np.asarray(kp, dtype=float)
        if p_gain.shape != (6,) or not np.isfinite(p_gain).all() or np.any(p_gain <= 0):
            raise ValueError("Expected six finite positive P gains")
        if np.any(np.abs(p_gain / self.kp - 1.0) > 0.2 + 1e-12):
            raise ValueError("Each P gain may change by at most 20% of its current setting")
        d_gain = self.kd.copy() if kd is None else np.asarray(kd, dtype=float)
        if d_gain.shape != (6,) or not np.isfinite(d_gain).all() or np.any(d_gain <= 0) or np.any(d_gain > 5):
            raise ValueError("Expected six native D gains in (0, 5]")
        if np.any(np.abs(d_gain / self.kd - 1.0) > 0.2 + 1e-12):
            raise ValueError("Each D gain may change by at most 20%")
        deadband = self.integral_deadband_deg if deadband_deg is None else float(deadband_deg)
        if not np.isfinite(deadband) or not 0 <= deadband <= float(np.rad2deg(ANGLE_TOL).min()):
            raise ValueError("Integral deadband must be between 0 and the smallest joint accuracy tolerance")
        if not self.closed:
            self.health()
        if (name in ("PID", "MIXED_J45") and self.live and not self.closed
                and not hasattr(self.robot, "command_joint_pos_pid")):
            raise RuntimeError("Install the project's SDK integral-control patch")
        self.control_law = name
        if name in ("POS_VEL", "MIXED_J45"):
            self.command_quantization = False
        self.ki = gains.copy()
        self.kp = p_gain.copy()
        self.kd = d_gain.copy()
        self.integral_deadband_deg = deadband
        if not self.closed and self.mode != "Gravity compensation":
            self._send(self.target.copy(), self.mode)

    def configure_gravity(self, factors, model="linear"):
        if self.control_law == "POS_VEL":
            raise ValueError("Position-speed has no gravity torque feedforward; select MIT at verified rest")
        if self.motion_plan is not None:
            raise RuntimeError("Wait for motion to finish before changing feedforward")
        gains = np.asarray(factors, dtype=float)
        if gains.shape != (6,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Expected six finite positive gravity factors")
        if model not in ("sdk", "linear"):
            raise ValueError("Gravity model must be sdk or linear")
        if self.disable_pending:
            raise RuntimeError("Wait for rest and disable to complete")
        if not self.closed:
            self.health()
            if self.live:
                self.robot.configure_gravity(gains, model=model, reference=self.rest[:6])
        self.gravity_factors = gains.copy()
        self.gravity_model = model

    def hold(self, *, automatic=False):
        q = self.positions()
        q[6] = self.gripper_target
        self.validate(q, automatic=automatic)
        self._send(q, "Hold")

    @property
    def gripper_target(self):
        return 0.0 if self.gripper_hold_closed else float(self.rest[6])

    def close_gripper(self):
        """Clamp a held board using the SDK's existing force limiter and gains."""
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
        self.configure_gravity(self.gravity_factors, self.gravity_model)
        self.configure_friction(self.friction_enabled, self.friction_transition, self.friction_coefficients)
        self.hold()

    def gravity(self):
        if self.control_law in ("POS_VEL", "MIXED_J45"):
            raise ValueError("Gravity mode requires a verified-rest switch to MIT")
        self.health()
        if self.disable_pending:
            raise RuntimeError("Returning to rest for disable; wait for completion")
        if self.motion_plan is not None:
            raise RuntimeError("Wait for motion to stop before gravity mode")
        if hasattr(self.robot, "enter_gravity_comp_idle"):
            self.robot.enter_gravity_comp_idle()
        elif hasattr(self.robot, "enable_gravity_comp"):
            self.robot.enable_gravity_comp()
        self.mode = "Gravity compensation"
        self.rest_verified = False
        self.settled_since = None

    def return_rest(self, *, automatic=False):
        self.health()
        if self.disable_pending:
            return
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

    def request_disable(self, *, automatic=False):
        if self.closed or self.disable_pending:
            return
        if self.motion_fault():
            self._feedback_health()
            self.disable_pending = True
            self.return_automatic = automatic
            self.rest_verified = False
            self.settled_since = None
            return
        self.health()
        if (self.motion_plan is not None
                or (self.live and np.max(np.abs(self.robot.get_observations()['joint_vel'][:6])) > .05)):
            self.disable_pending = True
            self.return_automatic = automatic
            return
        self.return_rest(automatic=automatic)
        self.disable_pending = True

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

    def tick(self):
        fault = self.motion_fault()
        if fault and not self.closed:
            self.motion_plan = None
            self.mode = "Oscillation fault"
            if self.control_law not in ("POS_VEL", "MIXED_J45"):
                self.control_law = "PD"
            self.target = np.array(fault["hold_q"])
            if self.disable_pending:
                # A latched fault blocks new motion, not a verified-rest power-off.
                # Away from rest, keep the request pending for explicit recovery.
                self._feedback_health()
                quiet = (not self.live or
                         np.max(np.abs(self.robot.get_observations()['joint_vel'][:6])) <= .05)
                if self.at_rest() and quiet:
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
        if self.motion_plan is not None:
            elapsed = time.monotonic()-self.motion_started
            if self.live and self.motion_automatic and np.max(np.abs(self.positions()[:6]-self.rest[:6])) >= self.test_excursion:
                self._send(self.positions(), "Hold")
                raise RuntimeError(f"Measured automatic excursion exceeded {self.test_excursion:g} rad; holding measured pose")
            if not self.live:
                self.robot.command_joint_pos(self.motion_plan.path.sample(elapsed)[0])
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
        if self.mode == "Returning to rest" and self.at_rest():
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
        if not self.rest_verified or not self.at_rest():
            raise RuntimeError("Rest verification failed; motors remain enabled")
        self.disabled_q = self.positions()
        self.robot.close()
        self.closed = True
        self.disable_pending = False
        self.mode = "Disabled"

    def enable(self):
        if not self.closed:
            return self.robot
        if self.robot_factory is None:
            raise RuntimeError("No SDK reconnect factory configured")
        # Factory initializes SDK hold at freshly measured positions, never an old motion target.
        kwargs = {}
        if self.control_law in ("POS_VEL", "MIXED_J45") or getattr(self.robot, "position_speed_limit", None) is not None:
            kwargs["arm_control_mode"] = self.control_law if self.control_law in ("POS_VEL", "MIXED_J45") else "MIT"
        robot = self.robot_factory(self.gripper_limits.copy(), self.disabled_q.copy(), **kwargs)
        q = np.array(robot.get_joint_pos(), dtype=float, copy=True)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid SDK feedback after enabling")
        self.robot = robot
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
