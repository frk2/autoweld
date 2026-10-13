"""Default startup and reconnect routing through SDK stubs, without hardware."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from test_integral import sdk_stub

from autoweld.control import Controller


def robot_at(q):
    robot = sdk_stub()
    robot.get_joint_pos = lambda: q.copy()
    robot.get_robot_info = lambda: {"gripper_limits": [-5., 5.]}
    robot.kdl = SimpleNamespace(
        compute_inverse_dynamics=lambda q, v, a: np.sin(q),
        model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0, 0, -9.81]))),
    )
    return robot


def assert_profile(c, robot, q):
    profile = json.loads(Path('evidence/smooth/selected-profile.json').read_text())
    assert c.control_law == 'PID'
    for name, command in [('kp', robot._commands.kp), ('ki', robot._commands.integral_ki),
                          ('kd', robot._commands.kd)]:
        np.testing.assert_allclose(command[:6], np.zeros(6) if name == 'ki' else profile[name])
    np.testing.assert_allclose(c.target, q)
    np.testing.assert_allclose(robot.gravity_comp_factor[:6], profile['gravity_factors'])
    assert c.gravity_model == robot.get_gravity_info()['model'] == 'linear'
    np.testing.assert_allclose(robot.get_gravity_info()['reference_rad'], c.rest[:6])
    assert robot.use_gravity_comp and not robot.use_coulomb_friction
    assert robot._commands.integral_delay == 0
    assert robot._commands.integral_deadband == np.deg2rad(profile['integral_deadband_deg'])


def test_startup_applies_default_gains_and_gravity_at_current_pose():
    q = np.array([.01, .02, .03, .04, .05, .06, .4])
    robot = robot_at(q)
    c = Controller(robot, np.tile([-3., 3.], (6, 1)), live=True)
    c.health = lambda: None
    c.start_hold()
    assert_profile(c, robot, q)
    np.testing.assert_equal(c.rest, q)


def test_disabled_selection_sends_nothing_and_reconnect_applies_profile():
    q = np.array([.01, .02, .03, .04, .05, .06, .4])
    robot = robot_at(q)
    c = Controller(robot, np.tile([-3., 3.], (6, 1)), live=True)
    c.closed = True
    c.disabled_q = q.copy()
    previous_command = robot._commands
    previous_gravity = robot.gravity_comp_factor.copy()
    c.select_controller('PID')
    c.configure_gravity(c.gravity_factors, c.gravity_model)
    assert robot._commands is previous_command
    np.testing.assert_equal(robot.gravity_comp_factor, previous_gravity)
    fresh_q = q + np.r_[np.full(6, .02), 0.]
    new_robot = robot_at(fresh_q)
    c.robot_factory = lambda limits, prior: new_robot
    c.enable()
    assert_profile(c, new_robot, fresh_q)
    np.testing.assert_equal(c.rest, q)
