"""A rejected rest return must never strand the enabled controller behind a latch."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.control import Controller
from autoweld.diagnostics import MotionRejection


@pytest.fixture
def c():
    controller = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)), live=True)
    controller.health = lambda: None
    return controller


def rejection():
    return MotionRejection('path_validation', 'self_collision', 'Rest path blocked')


@pytest.mark.parametrize('far_reference', ['current', 'future_target'])
def test_automatic_shutdown_preflight_rejects_before_latching_during_motion(c, far_reference):
    endpoint = c.rest.copy()
    endpoint[0] += 1.865
    c.motion_plan = SimpleNamespace(path=SimpleNamespace(positions=np.array([c.rest, endpoint])))
    if far_reference == 'current':
        c.robot.q = endpoint.copy()
    else:
        c.target = endpoint.copy()
    old_plan = c.motion_plan
    with pytest.raises(MotionRejection, match='0.49'):
        c.request_disable(automatic=True)
    assert not c.disable_pending
    assert c.motion_plan is old_plan
    assert c.test_excursion == .49
    assert not c.closed and not c.robot.closed and not c.robot.commands


def test_return_path_preflight_checks_collision_before_queuing_disable(c):
    endpoint = c.rest.copy()
    endpoint[0] += .1
    c.motion_plan = SimpleNamespace(path=SimpleNamespace(positions=np.array([c.rest, endpoint])))
    calls = []
    def plan(start, goal, **kwargs):
        calls.append((start.copy(), goal.copy(), kwargs))
        raise rejection()
    c.motion_planner = SimpleNamespace(limits=SimpleNamespace(joint_velocity=.1), plan=plan)
    # Isolate this policy test from MotionLimits dataclass copying.
    c.planner_for_controller = lambda: c.motion_planner
    c.motion_collision = lambda q: True
    with pytest.raises(MotionRejection, match='blocked'):
        c.request_disable(automatic=True)
    np.testing.assert_array_equal(calls[0][0], endpoint)
    np.testing.assert_array_equal(calls[0][1], c.rest)
    assert calls[0][2]['collision'] is c.motion_collision
    assert not c.disable_pending and not c.robot.closed and not c.robot.commands


def queue_small_coasting_return(c):
    c.robot.q[0] += .1
    c.robot.velocity = np.r_[.1, np.zeros(6)]
    c.request_disable(automatic=True)
    assert c.disable_pending and c.return_automatic
    assert not c.robot.commands


def test_queued_return_rejection_clears_barrier_and_refreshes_measured_powered_hold(c, caplog):
    queue_small_coasting_return(c)
    c.robot.q[0] = c.rest[0]+1.865
    c.robot.velocity.fill(0.)
    measured = c.robot.q.copy()
    with pytest.raises(MotionRejection, match='0.49'):
        c.tick()
    assert not c.disable_pending and not c.return_automatic
    assert not c.rest_verified and c.settled_since is None
    assert c.mode == 'Hold'
    np.testing.assert_array_equal(c.target, measured)
    np.testing.assert_array_equal(c.robot.commands[-1], measured)
    assert 'motors remain enabled' in c.disable_warning
    assert 'request cancelled' in caplog.text
    assert not c.closed and not c.robot.closed
    count = len(c.robot.commands)
    c.tick()  # No repeated MotionRejection/disable loop on subsequent ticks.
    assert len(c.robot.commands) == count and c.ready_for_waypoint()
    with pytest.raises(MotionRejection, match='0.49'):
        c.request_disable(automatic=True)
    assert not c.disable_pending and c.test_excursion == .49


def test_explicit_operator_retry_can_promote_pending_return_without_relaxing_agent_guard(c):
    queue_small_coasting_return(c)
    c.robot.q[0] = c.rest[0]+1.865
    c.robot.velocity.fill(0.)
    c.request_disable(automatic=False)
    assert c.disable_pending and not c.return_automatic
    assert c.test_excursion == .49
    with pytest.raises(RuntimeError, match='Returning to rest'):
        c.command(c.rest)
    c.tick()
    assert c.mode == 'Returning to rest'
    np.testing.assert_array_equal(c.target, c.rest)
    c.tick()
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.closed and c.robot.closed and c.rest_verified


def test_operator_retry_after_rejection_still_uses_verified_rest_disable(c):
    queue_small_coasting_return(c)
    c.robot.q[0] = c.rest[0]+1.865
    c.robot.velocity.fill(0.)
    with pytest.raises(MotionRejection):
        c.tick()
    c.request_disable(automatic=False)
    assert c.disable_pending and c.mode == 'Returning to rest'
    assert not c.robot.closed
    c.tick()
    assert not c.closed
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.closed and c.rest_verified


def test_automatic_duplicate_never_promotes_scope_or_restarts_settling(c):
    c.request_disable(automatic=True)
    c.tick()
    started = c.settled_since
    commands = len(c.robot.commands)
    c.request_disable(automatic=True)
    assert c.disable_pending and c.return_automatic
    assert c.settled_since == started and len(c.robot.commands) == commands


def test_normal_duplicate_does_not_restart_verified_rest_timer(c):
    c.request_disable(automatic=False)
    c.tick()
    started = c.settled_since
    commands = len(c.robot.commands)
    c.request_disable(automatic=False)
    assert c.settled_since == started and len(c.robot.commands) == commands
    assert c.disable_pending and not c.return_automatic


def test_runtime_planner_rejection_releases_pending_latch(c, monkeypatch):
    queue_small_coasting_return(c)
    c.robot.velocity.fill(0.)
    def blocked(*args, **kwargs):
        raise rejection()
    monkeypatch.setattr(c, '_start_waypoint', blocked)
    with pytest.raises(MotionRejection, match='blocked'):
        c.tick()
    assert not c.disable_pending and c.mode == 'Hold'
    assert not c.robot.closed and c.ready_for_waypoint()


def test_measured_automatic_excursion_during_disable_stops_once_without_poweroff(c):
    c.disable_pending = c.return_automatic = True
    c.motion_plan = SimpleNamespace(path=SimpleNamespace(duration=10.))
    c.motion_started = time.monotonic()
    c.motion_automatic = True
    c.robot.q[0] += .49
    with pytest.raises(RuntimeError, match='Measured automatic excursion'):
        c.tick()
    assert not c.disable_pending and c.motion_plan is None and c.mode == 'Hold'
    assert c.target[0] == pytest.approx(.49)
    assert not c.closed and not c.robot.closed
    c.tick()


def test_fault_latch_is_not_cleared_or_switched_by_disable_recovery(c):
    c.live = False
    c.robot.q[0] += .1
    fault = {'hold_q': c.robot.q.tolist()}
    c.robot.get_motion_fault = lambda: fault
    c.request_disable()
    c.tick()
    assert c.disable_pending and c.motion_fault() is fault
    assert not c.robot.commands and not c.closed


def test_automatic_faulted_disable_still_rejects_out_of_scope_rest_before_latching(c):
    c.robot.q[0] += 1.865
    c.robot.get_motion_fault = lambda: {'hold_q': c.robot.q.tolist()}
    c._feedback_health = lambda: True
    with pytest.raises(MotionRejection, match='0.49'):
        c.request_disable(automatic=True)
    assert not c.disable_pending and not c.robot.commands and not c.closed
