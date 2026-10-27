"""Dedicated shutdown accepts exit intent without bypassing physical safety."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.control import RETURN_STEP_TIMEOUT, Controller


@dataclass
class Limits:
    linear_velocity: float = 3.
    joint_velocity: float = .35


@dataclass
class Path:
    positions: np.ndarray
    duration: float = 2.

    def sample(self, elapsed):
        s = np.clip(elapsed/self.duration, 0., 1.)
        return self.positions[0]+s*(self.positions[-1]-self.positions[0]), np.zeros(7)


class Planner:
    limits = Limits()

    def __init__(self):
        self.calls = []
        self.error = None

    def plan(self, start, target, **kwargs):
        self.calls.append((start.copy(), target.copy(), self.limits, kwargs))
        if self.error:
            raise ValueError(self.error)
        kwargs['validate'](start)
        kwargs['validate'](target)
        if kwargs['collision'] and kwargs['collision']((start+target)/2):
            raise ValueError('Collision on shutdown path')
        return SimpleNamespace(path=Path(np.array([start, target])), report={'duration_s': 2.})


class Robot:
    def __init__(self):
        self.q = np.array([-.05, .2, .3, .1, -.1, -2.8])
        self.velocity = np.zeros(6)
        self.commands = []
        self.paths = []
        self.events = []
        self.fault = None
        self.age = .01
        self.temperature = 35.
        self.mode = 'POS_VEL'
        self.motor_chain = SimpleNamespace(running=True)
        self._server_thread = SimpleNamespace(is_alive=lambda: True)
        self.close_count = 0

    def get_joint_pos(self):
        return self.q.copy()

    def get_robot_info(self):
        return {'gripper_limits': [0., 1.]}

    def get_feedback_diagnostics(self):
        return {'age_s': self.age}

    def get_motion_fault(self):
        return self.fault

    def get_observations(self):
        return {'joint_vel': self.velocity.copy()}

    def get_temperature_state(self):
        return {'mos_c': [self.temperature]*6, 'rotor_c': [30.]*6}

    def get_arm_control_modes(self):
        return [self.mode]*6

    def switch_arm_control_mode(self, mode):
        self.events.append(('switch', mode))
        self.mode = mode

    def configure_position_speed_limit(self, limit):
        self.events.append(('speed', limit))

    def command_joint_pos(self, q):
        assert len(q) == 6
        self.commands.append(q.copy())  # A real motor need not follow immediately.

    def command_joint_motion(self, path, *args, **kwargs):
        assert path.positions.shape[1] == 6
        self.paths.append(path)

    def close(self):
        self.close_count += 1


@pytest.fixture
def c(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('autoweld.control.time.monotonic', lambda: clock[0])
    controller = Controller(Robot(), np.array([[-3., 3.]]*5+[[-5.37, -2.087]]), live=True)
    controller.strict_joint_limits = {5}
    controller.motion_planner = Planner()
    controller.motion_collision = lambda q: False
    controller._wait_for_fresh_motion_start = lambda: None
    controller.clock = clock
    return controller


def start_return(c):
    c.robot.q[0] -= 1.8
    c.shutdown()
    c.tick()  # Cancel old reference and capture measured hold.
    assert c.shutdown_phase == 'waiting'
    c.tick()  # Fresh, quiet, fully validated captured-rest plan.
    assert c.shutdown_phase == 'returning'


def reach_rest(c):
    c.robot.q[:] = c.rest[:6]
    c.clock[0] += 2.1
    c.tick()
    assert c.shutdown_phase == 'settling'


def test_shutdown_ignores_stale_application_latches_and_prior_target(c):
    c.robot.q[0] -= 1.8
    measured = c.robot.q.copy()
    rest = c.rest.copy()
    c.target[:] = 99.
    c.disable_pending = c.return_automatic = c.motion_automatic = True
    c.cooldown_active = True
    c.motion_plan = object()
    result = c.shutdown()
    assert result['requested'] and result['phase'] == 'stopping'
    assert not c.disable_pending and not c.return_automatic and not c.motion_automatic
    assert c.motion_plan is None and not c.robot.commands
    c.tick()
    np.testing.assert_array_equal(c.robot.commands[-1], measured)
    assert not c.robot.paths and c.robot.close_count == 0
    c.tick()
    assert c.shutdown_phase == 'returning'
    call = c.motion_planner.calls[0]
    np.testing.assert_array_equal(call[0][:6], measured)
    np.testing.assert_array_equal(call[1], rest)
    assert call[2].linear_velocity == 3. and call[2].joint_velocity == .35
    assert c.motion_planner.limits.linear_velocity == 3.
    assert c.motion_planner.limits.joint_velocity == .35
    assert call[3]['collision'] is c.motion_collision
    assert c.test_excursion == .49 and not c.motion_automatic
    assert not c.disable_pending


def test_shutdown_uses_requested_default_speeds_without_hidden_reduction(c):
    c.motion_planner.limits = Limits(linear_velocity=3., joint_velocity=2.)
    c.position_speed_limit = 2.
    start_return(c)
    limits = c.motion_planner.calls[-1][2]
    assert limits.linear_velocity == 3. and limits.joint_velocity == 2.
    assert c.motion_planner.limits == Limits(linear_velocity=3., joint_velocity=2.)


def test_return_disables_only_after_sustained_fresh_quiet_captured_rest(c):
    start_return(c)
    reach_rest(c)
    c.clock[0] += .99
    c.tick()
    assert not c.closed and not c.rest_verified
    c.clock[0] += .02
    c.tick()
    assert c.closed and c.rest_verified and c.robot.close_count == 1
    assert c.shutdown_status['phase'] == 'disabled'
    c.tick()
    c.shutdown(retry=True)
    assert c.robot.close_count == 1


def test_duplicate_requests_preserve_profile_and_settling_timer(c):
    start_return(c)
    plan, started = c.motion_plan, c.motion_started
    assert c.shutdown(retry=True)['phase'] == 'returning'
    assert c.motion_plan is plan and c.motion_started == started
    assert len(c.robot.paths) == 1
    reach_rest(c)
    settled = c.settled_since
    c.shutdown(retry=True)
    assert c.settled_since == settled and c.motion_plan is None


def test_waits_for_quiet_before_planning_without_replaying_prior_target(c):
    c.robot.q[0] -= 1.
    c.robot.velocity[0] = .1
    c.shutdown()
    c.tick()
    c.tick()
    assert c.shutdown_phase == 'waiting' and not c.robot.paths
    c.robot.velocity.fill(0.)
    c.tick()
    assert c.shutdown_phase == 'returning'


@pytest.mark.parametrize('failure', ['collision', 'bounds', 'planner', 'feedback', 'thread', 'temperature'])
def test_physical_rejection_blocks_once_in_powered_hold_and_can_retry(c, failure):
    c.robot.q[0] -= 1.
    c.shutdown()
    c.tick()
    if failure == 'collision':
        c.motion_collision = lambda q: True
    elif failure == 'bounds':
        c.rest[5] = -.1  # Calibrated J6 cannot be expanded by a captured pose.
    elif failure == 'planner':
        c.motion_planner.error = 'Planner refused'
    elif failure == 'feedback':
        c.robot.age = .6
    elif failure == 'thread':
        c.robot.motor_chain.running = False
    else:
        c.robot.temperature = np.nan
    c.tick()  # Errors are published as shutdown status, never a tick exception loop.
    assert c.shutdown_status['phase'] == 'blocked'
    assert c.shutdown_status['retryable'] and c.shutdown_status['error']
    assert not c.disable_pending and not c.closed and c.robot.close_count == 0
    count = len(c.robot.commands)
    for _ in range(4):
        c.tick()
    assert len(c.robot.commands) == count
    c.shutdown()
    assert c.shutdown_phase == 'blocked'
    c.motion_collision = lambda q: False
    c.rest[5] = -2.8
    c.motion_planner.error = None
    c.robot.age, c.robot.temperature, c.robot.motor_chain.running = .01, 35., True
    c.shutdown(retry=True)
    c.tick()
    c.tick()
    assert c.shutdown_phase == 'returning' and len(c.robot.paths) == 1


def test_hot_shutdown_continues_only_validated_rest_return(c):
    c.robot.temperature = 60.
    start_return(c)
    assert c.cooldown_active and c.cooldown_max_c == 60.
    reach_rest(c)
    c.clock[0] += 1.1
    c.tick()
    assert c.closed


def test_fault_away_from_rest_retains_native_hold_without_clear_switch_or_off(c):
    c.robot.q[0] -= 1.
    c.robot.fault = {'hold_q': c.robot.q.tolist()}
    fault = c.robot.fault
    assert c.shutdown()['requested']
    c.tick()
    assert c.shutdown_phase == 'blocked' and c.robot.fault is fault
    assert not c.robot.commands and not c.robot.paths and not c.robot.events
    assert c.robot.close_count == 0


def test_native_fault_at_rest_can_verify_and_disable_without_acknowledgement(c):
    fault = {'hold_q': c.robot.q.tolist()}
    c.robot.fault = fault
    c.shutdown()
    c.tick()
    c.tick()
    c.clock[0] += 1.1
    c.tick()
    assert c.closed and c.robot.fault is fault
    assert not c.robot.commands and not c.robot.paths and not c.robot.events


def test_gravity_shutdown_switches_through_sdk_then_captures_measured_hold(c):
    c.control_law = c.robot.mode = 'MIT'
    c.robot.q[0] -= .5
    measured = c.robot.q.copy()
    c.shutdown()
    c.tick()
    assert ('switch', 'POS_VEL') in c.robot.events
    np.testing.assert_array_equal(c.robot.commands[-1], measured)
    assert c.robot.close_count == 0


def test_operator_stop_preserves_shutdown_intent_and_retry_starts_at_new_measurement(c):
    start_return(c)
    c.robot.q[0] += .2
    stopped = c.robot.q.copy()
    result = c.pause_shutdown()
    assert result['phase'] == 'blocked' and result['requested'] and result['retryable']
    assert 'Stopped by operator' in result['error']
    np.testing.assert_array_equal(c.robot.commands[-1], stopped)
    assert c.motion_plan is None
    c.tick()
    assert c.robot.close_count == 0 and len(c.robot.paths) == 1
    c.shutdown(retry=True)
    c.tick()
    c.tick()
    assert len(c.robot.paths) == 2
    np.testing.assert_array_equal(c.motion_planner.calls[-1][0][:6], stopped)


def test_stop_during_fault_preserves_native_sdk_hold(c):
    start_return(c)
    c.robot.fault = {'hold_q': c.robot.q.tolist()}
    commands = len(c.robot.commands)
    c.pause_shutdown()
    assert len(c.robot.commands) == commands
    assert c.shutdown_phase == 'blocked' and c.robot.close_count == 0


@pytest.mark.parametrize('action', ['command', 'return_rest', 'hold', 'gravity', 'enable',
                                   'request_disable', 'select_mode', 'select_controller', 'command_tracking',
                                   'acknowledge_fault', 'configure_position_speed_limit'])
def test_other_motion_cannot_steal_shutdown_return(c, action):
    c.shutdown()
    args = {'command': (c.rest,), 'select_mode': ('Hold',), 'select_controller': ('POS_VEL',),
            'command_tracking': (object(),), 'configure_position_speed_limit': (.1,)}.get(action, ())
    with pytest.raises(RuntimeError, match='Shutdown requested'):
        getattr(c, action)(*args)
    assert not c.ready_for_waypoint()


def test_settle_timeout_leaves_retryable_powered_hold(c):
    start_return(c)
    c.clock[0] += 2.1
    c.tick()
    assert c.shutdown_phase == 'settling' and not c.at_rest()
    c.clock[0] += RETURN_STEP_TIMEOUT+.1
    c.tick()
    assert c.shutdown_phase == 'blocked' and not c.closed
    np.testing.assert_array_equal(c.robot.commands[-1], c.robot.q)


def test_sdk_close_rechecks_freshness_at_last_moment(c, monkeypatch):
    c.shutdown()
    c.tick()
    c.tick()
    original = c._disable_at_rest
    def stale_at_close():
        c.robot.age = .6
        original()
    monkeypatch.setattr(c, '_disable_at_rest', stale_at_close)
    c.clock[0] += 1.1
    c.tick()
    assert c.shutdown_phase == 'blocked' and c.robot.close_count == 0


def test_already_disabled_shutdown_never_reconnects(c):
    c.closed = True
    c.disabled_q = c.rest.copy()
    assert c.shutdown()['phase'] == 'disabled'
    c.tick()
    assert not c.robot.commands and not c.robot.paths and c.robot.close_count == 0
