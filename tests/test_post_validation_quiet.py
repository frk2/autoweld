"""Simulated post-validation samples; no model, hardware or SDK owner."""
from types import SimpleNamespace

import numpy as np
import pytest

import autoweld.control as module

Controller, REST_TOL = module.Controller, module.REST_TOL


class Clock:
    elapsed = 0.
    def wall(self):
        return 1000.+self.elapsed
    def monotonic(self):
        return 100.+self.elapsed
    def sleep(self, duration):
        self.elapsed += duration


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(module.time, 'time', clock.wall)
    monkeypatch.setattr(module.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(module.time, 'sleep', clock.sleep)

    class Robot:
        def __init__(self):
            self.q = np.zeros(7)
            self.commands = []
            self.diagnostics = 0
            self.samples = []
            self.planned = False
            self.velocity = lambda: np.zeros(7)
            self.feedback = lambda: {'timestamp': clock.wall(), 'age_s': 0.}
        def get_joint_pos(self):
            return self.q.copy()
        def get_robot_info(self):
            return {'gripper_limits': [-1., 1.]}
        def get_observations(self):
            value = np.asarray(self.velocity(), float)
            self.samples.append((clock.elapsed, value.copy()))
            return {'joint_vel': value, 'joint_pos': self.q.copy()}
        def get_feedback_diagnostics(self):
            self.diagnostics += 1
            return self.feedback()
        def command_joint_motion(self, *args, **kwargs):
            self.commands.append((clock.elapsed, args, kwargs))

    robot = Robot()
    ctrl = Controller(robot, np.tile([-3., 3.], (6, 1)), live=True)
    ctrl.health = lambda: None
    return SimpleNamespace(clock=clock, robot=robot, ctrl=ctrl)


def test_first_fresh_sample_moving_then_quiet_passes_same_bounded_wait(rig):
    r, c = rig.robot, rig.ctrl
    r.velocity = lambda: np.full(7, .08 if len(r.samples) == 0 else .05)
    c._wait_for_fresh_motion_start()
    assert [v[0] for _, v in r.samples] == [.08, .05]
    assert 0 < rig.clock.elapsed < .5
    assert not r.commands


def test_permanent_movement_rejects_at_original_deadline(rig):
    rig.robot.velocity = lambda: np.full(7, .050001)
    with pytest.raises(RuntimeError, match='Arm moved.*no motion started'):
        rig.ctrl._wait_for_fresh_motion_start()
    assert .5 <= rig.clock.elapsed < .5021
    assert not rig.robot.commands and rig.ctrl.motion_plan is None


@pytest.mark.parametrize('age', [.020001, -.001, float('nan'), float('inf')])
def test_stale_or_invalid_age_never_qualifies(rig, age):
    rig.robot.feedback = lambda: {'timestamp': rig.clock.wall(), 'age_s': age}
    with pytest.raises(RuntimeError, match='No fresh SDK feedback'):
        rig.ctrl._wait_for_fresh_motion_start()
    assert .5 <= rig.clock.elapsed < .5021
    assert not rig.robot.samples and not rig.robot.commands


def test_timestamp_must_be_newer_than_validation_start(rig):
    rig.robot.feedback = lambda: {'timestamp': 1000., 'age_s': 0.}
    with pytest.raises(RuntimeError, match='No fresh SDK feedback'):
        rig.ctrl._wait_for_fresh_motion_start()
    assert not rig.robot.samples and not rig.robot.commands


def test_twenty_ms_feedback_and_point_zero_five_rad_s_are_still_accepted(rig):
    rig.robot.feedback = lambda: {'timestamp': rig.clock.wall(), 'age_s': .02}
    rig.robot.velocity = lambda: np.full(7, .05)
    rig.ctrl._wait_for_fresh_motion_start()
    assert rig.clock.elapsed == pytest.approx(.002)
    assert not rig.robot.commands


@pytest.mark.parametrize('value', [np.full(7, float('nan')), np.full(7, float('inf')), np.zeros(5)])
def test_invalid_velocity_cannot_authorize_a_reference(rig, value):
    rig.robot.velocity = lambda: value
    with pytest.raises(RuntimeError, match='no fresh quiet feedback'):
        rig.ctrl._wait_for_fresh_motion_start()
    assert not rig.robot.commands


def test_health_fault_interrupts_wait_immediately(rig):
    checks = []
    def health():
        checks.append(rig.clock.elapsed)
        if rig.clock.elapsed >= .002:
            raise RuntimeError('Native hardware fault')
    rig.ctrl.health = health
    with pytest.raises(RuntimeError, match='Native hardware fault'):
        rig.ctrl._wait_for_fresh_motion_start()
    assert rig.clock.elapsed == pytest.approx(.002)
    assert len(checks) == 2 and not rig.robot.commands


def planner(rig, *, drift=0.):
    def plan(start, goal, **kwargs):
        rig.robot.planned = True
        rig.robot.q[0] += drift
        return SimpleNamespace(path=SimpleNamespace(positions=np.stack((start, goal)), duration=.2), report={})
    p = SimpleNamespace(plan=plan, limits=SimpleNamespace(joint_velocity=2.))
    rig.ctrl.motion_planner = p
    return p


def test_no_sdk_command_until_first_moving_postplan_sample_becomes_fresh_and_quiet(rig):
    p = planner(rig)
    post_plan = []
    def velocity():
        if not rig.robot.planned:
            return np.zeros(7)
        post_plan.append(rig.clock.elapsed)
        return np.full(7, .08 if len(post_plan) == 1 else .0)
    rig.robot.velocity = velocity
    target = rig.robot.q.copy()
    target[0] += .01
    rig.ctrl._start_waypoint(target, 'IK', planner=p)
    assert len(rig.robot.commands) == 1
    assert rig.robot.commands[0][0] == pytest.approx(.004)
    assert len(post_plan) == 2
    assert rig.ctrl.motion_plan is not None


def test_persistent_moving_postplan_samples_never_send_sdk_command(rig):
    p = planner(rig)
    rig.robot.velocity = lambda: np.full(7, .08) if rig.robot.planned else np.zeros(7)
    target = rig.robot.q.copy()
    target[0] += .01
    with pytest.raises(RuntimeError, match='Arm moved.*no motion started'):
        rig.ctrl._start_waypoint(target, 'IK', planner=p)
    assert .5 <= rig.clock.elapsed < .5021
    assert not rig.robot.commands and rig.ctrl.motion_plan is None


def test_transient_idle_velocity_spike_waits_before_planning_and_submission(rig):
    p = planner(rig)
    rig.robot.velocity = lambda: np.full(7, .08 if not rig.robot.samples else 0.)
    target = rig.robot.q.copy()
    target[0] += .01
    rig.ctrl._start_waypoint(target, 'IK', planner=p)
    assert rig.robot.planned and len(rig.robot.commands) == 1
    assert 0 < rig.clock.elapsed < .5


def test_persistent_idle_motion_is_rejected_before_any_planning(rig):
    p = planner(rig)
    rig.robot.velocity = lambda: np.full(7, .08)
    target = rig.robot.q.copy()
    target[0] += .01
    with pytest.raises(RuntimeError, match='Arm moved.*no motion started'):
        rig.ctrl._start_waypoint(target, 'IK', planner=p)
    assert not rig.robot.planned and not rig.robot.commands
    assert .5 <= rig.clock.elapsed < .5021


def test_post_plan_rest_tolerance_drift_guard_still_rejects_before_sdk_command(rig):
    p = planner(rig, drift=REST_TOL+.00001)
    target = rig.robot.q.copy()
    target[0] += .01
    with pytest.raises(RuntimeError, match='Arm moved during planning'):
        rig.ctrl._start_waypoint(target, 'IK', planner=p)
    assert not rig.robot.commands and rig.ctrl.motion_plan is None


def test_shutdown_internal_rest_path_is_not_blocked_by_new_application_guard(rig):
    # The dedicated shutdown's own _start_waypoint needs this private helper
    # while shutdown_requested=True. Public new commands remain rejected.
    rig.ctrl.shutdown_requested = True
    rig.ctrl._wait_for_fresh_motion_start()
    assert rig.clock.elapsed < .5 and not rig.robot.commands
    with pytest.raises(RuntimeError, match='Shutdown requested'):
        rig.ctrl.command(rig.robot.q.copy())


def test_deadline_is_not_extended_for_a_sample_that_only_becomes_quiet_after_timeout(rig):
    rig.robot.velocity = lambda: np.full(7, .1 if rig.clock.elapsed < .5 else 0.)
    with pytest.raises(RuntimeError, match='Arm moved'):
        rig.ctrl._wait_for_fresh_motion_start()
    assert .5 <= rig.clock.elapsed < .5021 and not rig.robot.commands
