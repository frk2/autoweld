"""SDK commissioning probes always end in a powered measured-pose PD hold."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot
from i2rt.robots.position_speed_robot import PositionSpeedRobot


def robot(monkeypatch):
    r = PositionSpeedRobot.__new__(PositionSpeedRobot)
    r._command_lock, r._state_lock = threading.Lock(), threading.Lock()
    r.motor_chain = [None] * 6
    r._mit_indices = lambda: tuple(range(6))
    r.get_motion_fault = lambda: None
    r._joint_state = SimpleNamespace(pos=np.arange(6)*.01, vel=np.zeros(6), timestamp=100.)
    r._kp, r._kd = np.array([80, 80, 80, 10, 10, 10.]), np.array([5, 5, 5, 1.5, 1.5, 1.5])
    r._commands = JointCommands.init_all_zero(6)
    r.remapper = SimpleNamespace(to_robot_joint_pos_space=lambda q: q.copy())
    r.enter_gravity_comp_idle = lambda: None
    monkeypatch.setattr(MotorChainRobot, 'update', lambda self: None)
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.time', lambda: 100.)
    return r


@pytest.mark.parametrize('reason', ['duration complete', 'excursion limit', 'speed limit', 'stale feedback'])
def test_probe_restores_measured_hold_without_disabling(monkeypatch, reason):
    now = [10.]
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.monotonic', lambda: now[0])
    r = robot(monkeypatch)
    r.start_gravity_trial(.5, .04, .3)
    r.update()
    assert r._gravity_trial['active']
    if reason == 'duration complete':
        now[0] += .6
    elif reason == 'excursion limit':
        r._joint_state.pos[2] += .05
    elif reason == 'speed limit':
        r._joint_state.vel[3] = .31
    else:
        r._joint_state.timestamp -= .3
    r.update()
    assert not r._gravity_trial['active']
    assert r._gravity_trial['reason'] == reason
    np.testing.assert_equal(r._commands.pos, r._joint_state.pos)
    np.testing.assert_equal(r._commands.kp, r._kp)
    np.testing.assert_equal(r._commands.kd, r._kd)
    np.testing.assert_equal(r._commands.vel, 0)


def test_invalid_probe_does_not_change_commands(monkeypatch):
    r = robot(monkeypatch)
    for args in ((0, .04, .3), (3, .04, .3), (.5, .5, .3), (.5, .04, .8)):
        with pytest.raises(ValueError):
            r.start_gravity_trial(*args)
    r._mit_indices = lambda: ()
    with pytest.raises(RuntimeError, match='MIT'):
        r.start_gravity_trial()


def test_entry_waits_for_real_position_stability_despite_velocity_bins(monkeypatch):
    r = robot(monkeypatch)
    now = [100.]
    r.motor_chain = SimpleNamespace(running=True)
    r._joint_state.timestamp = now[0]
    r._joint_state.vel[:] = .02
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.time', lambda: now[0])
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.monotonic', lambda: now[0])
    def advance(dt):
        now[0] += dt
        r._joint_state.timestamp = now[0]
        if now[0] < 100.3:
            r._joint_state.pos[5] += .03*dt
        else:
            r._joint_state.vel[:] = -.007326  # SDK quantization noise at rest.
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.sleep', advance)
    r.settle_measured_hold()
    assert now[0] >= 100.5


def test_entry_timeout_preserves_powered_hold(monkeypatch):
    r = robot(monkeypatch)
    now = [100.]
    r.motor_chain = SimpleNamespace(running=True)
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.time', lambda: now[0])
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.monotonic', lambda: now[0])
    def advance(dt):
        now[0] += dt
        r._joint_state.timestamp = now[0]
        r._joint_state.pos[3] += .03*dt
        r._joint_state.vel[3] = .03
    monkeypatch.setattr('i2rt.robots.position_speed_robot.time.sleep', advance)
    before = r._commands
    with pytest.raises(RuntimeError, match='did not settle'):
        r.settle_measured_hold(timeout_s=.6)
    assert r._commands is before
    assert r.motor_chain.running
