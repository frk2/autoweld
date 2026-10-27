"""Default startup and reconnect routing through SDK stubs, without hardware."""
from types import SimpleNamespace

import numpy as np
from test_integral import sdk_stub

from autoweld.control import Controller
from autoweld.motion import MotionLimits


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
    assert c.control_law == 'POS_VEL'
    assert c.position_speed_limit == 2.
    assert robot.applied_speed_limit == 2.
    assert robot._commands.integral_ki is None
    np.testing.assert_allclose(c.target, q)
    assert not robot.use_coulomb_friction
    np.testing.assert_equal(robot.gravity_comp_factor, 1)


def test_startup_applies_default_gains_and_gravity_at_current_pose():
    q = np.array([.01, .02, .03, .04, .05, .06, .4])
    robot = robot_at(q)
    robot.configure_position_speed_limit = lambda value: setattr(robot, 'applied_speed_limit', value)
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
    c.select_controller('POS_VEL')
    assert robot._commands is previous_command
    np.testing.assert_equal(robot.gravity_comp_factor, previous_gravity)
    fresh_q = q + np.r_[np.full(6, .02), 0.]
    new_robot = robot_at(fresh_q)
    new_robot.configure_position_speed_limit = lambda value: setattr(new_robot, 'applied_speed_limit', value)
    c.robot_factory = lambda limits, prior, **kwargs: new_robot
    c.enable()
    assert_profile(c, new_robot, fresh_q)
    np.testing.assert_equal(c.rest, q)


def test_default_speed_changes_preserve_acceleration_and_jerk_limits():
    limits = MotionLimits()
    assert limits.linear_velocity == 3.
    assert limits.joint_velocity == 2.
    assert (limits.linear_acceleration, limits.linear_jerk) == (.15, 1.)
    assert (limits.joint_acceleration, limits.joint_jerk) == (1., 5.)
    assert (limits.angular_velocity, limits.angular_acceleration, limits.angular_jerk) == (.25, 1., 5.)
