"""Current six-axis position-speed policy and motor lifecycle regressions."""
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots.waypoint_motion import FeedJointPath
from test_control import FakeRobot

from autoweld.control import Controller
from autoweld.motion import MotionLimits


def controller():
    robot = FakeRobot()
    robot.q = robot.q[:6]
    return Controller(robot, np.tile([-3., 3.], (6, 1)))


def test_six_axis_sdk_never_receives_phantom_gripper():
    c = controller()
    assert c.positions().shape == (7,)
    assert c.rest[6] == 0
    q = c.rest.copy()
    q[0] += .01
    c.command(q)
    assert c.robot.commands[-1].shape == (6,)
    with pytest.raises(RuntimeError, match="No gripper"):
        c.close_gripper()
    path = FeedJointPath([0., 1.], [c.rest, q], 2., ramp_fraction=.2)
    sdk_path = c._sdk_path(path)
    assert type(sdk_path) is FeedJointPath
    assert sdk_path.ramp_fraction == .2
    assert sdk_path.positions.shape == (2, 6)
    np.testing.assert_allclose(sdk_path.sample(.7)[0], path.sample(.7)[0][:6])


def test_position_speed_only_and_default_tcp_speed():
    c = controller()
    assert c.control_law == 'POS_VEL'
    assert MotionLimits().linear_velocity == 3.
    for mode in ('PD', 'PID', 'MIT', 'MIXED_J45'):
        with pytest.raises(ValueError, match='Only position-speed'):
            c.select_controller(mode)
    np.testing.assert_array_equal(c.gravity_factors, np.ones(6))
    assert c.gravity_model == 'sdk'
    assert c.mode == 'Hold' and not c.robot.gravity
    c.select_mode('IK')
    assert c.mode == 'IK' and not c.robot.closed
    c.select_mode('Hold')
    assert c.mode == 'Hold'


def test_singleton_factory_runs_once_for_concurrent_clients(monkeypatch):
    monkeypatch.setattr(Controller, '_instance', None)
    calls = []
    def factory():
        calls.append(True)
        return FakeRobot()
    with ThreadPoolExecutor(max_workers=8) as executor:
        controllers = list(executor.map(lambda _: Controller.get_instance(factory, np.tile([-3., 3.], (6, 1))), range(20)))
    assert len(calls) == 1
    assert all(c is controllers[0] for c in controllers)


def test_hot_motor_cancels_target_returns_rest_and_keeps_power():
    c = controller()
    c.robot.q[0] += .1
    temperatures = [60.]
    c.robot.get_temperature_state = lambda: {'mos_c': temperatures*6, 'rotor_c': [35.]*6}
    c.tick()
    assert c.cooldown_active and c.cooldown_max_c == 60.
    np.testing.assert_allclose(c.target, c.rest)
    assert c.mode == 'Returning to rest' and not c.closed
    with pytest.raises(RuntimeError, match='Cooldown'):
        c.command(c.rest)
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.rest_verified and c.cooldown_active
    temperatures[0] = 56.
    c.tick()
    assert c.cooldown_active
    temperatures[0] = 54.
    c.tick()
    assert not c.cooldown_active and c.mode == 'Hold'
    assert not c.robot.closed


def test_stale_feedback_warns_without_fault_or_poweroff_and_never_verifies_rest():
    c = controller()
    c.live = True
    c.robot.motor_chain = SimpleNamespace(running=True)
    c.robot._server_thread = SimpleNamespace(is_alive=lambda: True)
    age = [1.]
    c.robot.get_feedback_diagnostics = lambda: {'age_s': age[0]}
    c.health()
    assert 'stale' in c.feedback_warning
    assert not c.closed and not c.motion_fault()
    c.mode = 'Returning to rest'
    c.return_step_started = time.monotonic()
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert not c.rest_verified and c.settled_since is None
    age[0] = .01
    c.tick()
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.rest_verified and c.feedback_warning is None


def test_rest_poweroff_requires_stationary_arm():
    c = controller()
    c.live = True
    c.health = lambda: None
    c.robot.velocity = np.full(6, .1)
    c.rest_verified = True
    with pytest.raises(RuntimeError, match='Rest verification'):
        c._disable_at_rest()
    assert not c.robot.closed
