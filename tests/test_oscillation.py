"""Replay and fault-path tests; these never open hardware."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
from i2rt.robots.oscillation_guard import OscillationGuard, QuietWindow
from test_integral import sdk_stub

from autoweld.control import FACTORY_KD, FACTORY_KP, Controller


def test_detector_replays_failed_move_before_communication_loss():
    rows = [json.loads(x) for x in Path('evidence/damping/d311-aborted.jsonl').read_text().splitlines()]
    guard = OscillationGuard()
    for row in rows:
        fault = guard.observe(row['timestamp'], row['q'], row['velocity'])
        if fault:
            break
    assert fault['joint'] in (5, 6)
    assert 1.0 <= fault['timestamp'] - rows[0]['timestamp'] < 1.5
    assert rows[-1]['timestamp'] - fault['timestamp'] > 10


@pytest.mark.parametrize('name', ['d216-out', 'd216-reverse', 'd259-out', 'os-p246-w24-out', 'os-p41-i5-out'])
def test_known_comparison_trials_do_not_trip(name):
    guard = OscillationGuard()
    for line in Path(f'evidence/pid/{name}.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert guard.observe(row['timestamp'], row['q'], row['velocity']) is None


def test_monotonic_motion_encoder_noise_duplicate_samples_and_slow_reversals():
    for speed, amplitude, frequency in [(1, 0, 0), (0, np.deg2rad(.025), 20), (0, .03, 1)]:
        guard = OscillationGuard()
        for t in np.arange(0, 3, .004):
            q = np.full(6, speed*t + amplitude*np.sin(2*np.pi*frequency*t))
            v = np.full(6, speed + amplitude*2*np.pi*frequency*np.cos(2*np.pi*frequency*t))
            assert guard.observe(t, q, v) is None
            assert guard.observe(t, q, -v) is None


def test_sdk_latch_cancels_pid_blocks_subsequent_commands_and_requires_quiet_ack(monkeypatch):
    r = sdk_stub()
    wall = time.time()
    now = [100.]
    monkeypatch.setattr('i2rt.robots.motor_chain_robot.time.monotonic', lambda: now[0])
    r._oscillation_guard.observe = lambda *args: {'reason': 'Rapid oscillation on J5', 'joint': 5}
    r._joint_state.pos = np.array([.1, .1, .1, .1, .1, .1, .5])
    r.command_joint_pos_pid(np.ones(7)*.2, np.ones(7)*100, np.ones(7),
                            kp=r._kp*2, kd=np.full(7, 3.))
    r._integral_torque.fill(.8)
    r.update()
    fault = r.get_motion_fault()
    assert fault and fault['joint'] == 5
    np.testing.assert_equal(r.sent[1].pos, r._joint_state.pos)
    np.testing.assert_equal(r.sent[1].kp, r._kp)
    np.testing.assert_equal(r.sent[1].kd, r._kd)
    np.testing.assert_equal(r.get_integral_torque(), 0)
    assert r.sent[1].integral_ki is None
    with pytest.raises(RuntimeError):
        r.acknowledge_motion_fault()
    # Both original PD and PID commands remain physically overridden by the latch.
    for send in [lambda: r.command_joint_pos(np.ones(7)),
                 lambda: r.command_joint_pos_pid(np.ones(7), np.ones(7)*100, np.ones(7))]:
        send()
        r.update()
        np.testing.assert_equal(r.sent[1].pos, fault['hold_q'])
        np.testing.assert_equal(r.sent[1].kd, r._kd)
    for step in range(1, 57):
        now[0] = 100. + step*.02
        r._joint_state.timestamp = wall + step*.02
        r.update()
    r._joint_state.timestamp = wall-2
    with pytest.raises(RuntimeError):
        r.acknowledge_motion_fault()  # Elapsed time cannot substitute for fresh feedback.
    r._joint_state.timestamp = time.time()
    r.acknowledge_motion_fault()
    assert r.get_motion_fault() is None
    np.testing.assert_equal(r._commands.kp, r._kp)


def test_controller_fault_recovery_preserves_position_speed_mode():
    r = sdk_stub()
    r.get_joint_pos = lambda: np.zeros(7)
    r.get_robot_info = lambda: {'gripper_limits': [-5., 5.]}
    c = Controller(r, np.tile([-3., 3.], (6, 1)))
    with pytest.raises(ValueError, match='Only position-speed'):
        c.select_controller('PID', kd=c.kd*.8)
    r._motion_fault = {'reason': 'Rapid oscillation on J5', 'hold_q': [0.]*7}
    with pytest.raises(RuntimeError, match='latched'):
        c.command(np.zeros(7))
    c.tick()
    assert c.mode == 'Oscillation fault'
    assert c.control_law == 'POS_VEL' and not c.rest_verified
    r.acknowledge_motion_fault = lambda: None
    c.acknowledge_fault()
    np.testing.assert_equal(c.kp, FACTORY_KP)
    np.testing.assert_equal(c.kd, FACTORY_KD)
    np.testing.assert_equal(c.ki, 0)
    assert c.control_law == 'POS_VEL'


def test_quiet_window_accepts_encoder_noise_but_rejects_drift_speed_and_gaps():
    quiet = QuietWindow()
    for t in np.arange(0, 1.1, .02):
        ready = quiet.observe(t, np.full(6, np.deg2rad(.02)*np.sin(t*100)), np.full(6, .051))
    assert ready
    assert not quiet.observe(1.1, np.full(6, .01), np.zeros(6))  # Actual drift.
    quiet.reset()
    for t in np.arange(0, 1.1, .02):
        ready = quiet.observe(t, np.zeros(6), np.full(6, .11))
    assert not ready
    quiet.reset()
    for _ in range(100):
        assert not quiet.observe(1., np.zeros(6), np.zeros(6))
    assert not quiet.observe(2., np.zeros(6), np.zeros(6))  # Missing data cannot prove quiet.


def test_one_second_guard_ignores_false_alarm_and_short_settling_burst():
    guard = OscillationGuard()
    for line in Path('evidence/retune/live-guard-response.jsonl').read_text().splitlines():
        r = json.loads(line)
        assert guard.observe(r['timestamp'], r['q'], r['velocity']) is None
    guard = OscillationGuard()
    for t in np.arange(0, 2, .004):
        amplitude = .01*np.exp(-t/.1)
        q = np.full(6, amplitude*np.sin(2*np.pi*10*t))
        v = np.full(6, amplitude*(2*np.pi*10*np.cos(2*np.pi*10*t)-10*np.sin(2*np.pi*10*t)))
        assert guard.observe(t, q, v) is None


def test_one_second_guard_trips_only_after_full_sustained_window():
    guard = OscillationGuard()
    for t in np.arange(0, 1.2, .004):
        q = np.full(6, .01*np.sin(2*np.pi*8*t))
        v = np.full(6, .01*2*np.pi*8*np.cos(2*np.pi*8*t))
        fault = guard.observe(t, q, v)
        if t < 1:
            assert fault is None
        if fault:
            break
    assert fault and 1 <= fault['timestamp'] < 1.2
