"""Exercise the actual SDK update path without a CAN interface or physical robot."""

import inspect
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots.integral_control import integral_torque
from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot
from i2rt.robots.oscillation_guard import OscillationGuard, QuietWindow

from autoweld.control import Controller


def test_integral_time_units_windup_and_reverse():
    tau = np.zeros(2)
    for _ in range(100):
        tau = integral_torque(np.array([0.1, -0.2]), np.array([10.0, 10.0]), 0.01, tau, np.ones(2))
    np.testing.assert_allclose(tau, [1.0, -1.0])
    tau = integral_torque(np.array([-0.1, 0.2]), np.array([10.0, 10.0]), 0.01, tau, np.ones(2))
    np.testing.assert_allclose(tau, [0.99, -0.98])


@pytest.mark.parametrize("dt", [0.0, -0.1, 0.1, float("nan")])
def test_stale_interval_clears_integral(dt):
    np.testing.assert_equal(integral_torque(np.ones(2), np.ones(2), dt, np.ones(2), np.ones(2)), 0)


def sdk_stub():
    r = MotorChainRobot.__new__(MotorChainRobot)
    r.motor_chain = [None] * 7
    r._gripper_index = 6
    r._joint_limits = None
    r._command_lock = threading.Lock()
    r._state_lock = threading.Lock()
    r._commands = JointCommands.init_all_zero(7)
    r._kp = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0, 5.0])
    r._kd = np.array([5., 5., 5., 1.5, 1.5, 1.5, 1.])
    r._grav_comp_kd = np.full(7, 0.1)
    r.remapper = SimpleNamespace(to_robot_joint_pos_space=lambda q: q.copy(),
                                to_robot_joint_vel_space=lambda q: q.copy())
    r._oscillation_guard = OscillationGuard()
    r._motion_fault = None
    r._fault_hold = None
    r._quiet_guard = QuietWindow()
    r._linear_gravity = None
    r._joint_motion = None
    r._motion_reference = None
    r._torque_terms = None
    r.use_gravity_comp = True
    r._integral_session = None
    r._integral_time = None
    r._integral_torque = np.zeros(7)
    r._joint_state = SimpleNamespace(pos=np.zeros(7), vel=np.zeros(7), eff=np.zeros(7), timestamp=time.time())
    r._compute_gravity_compensation = lambda _: np.zeros(7)
    r.use_coulomb_friction = True
    r._coulomb_friction = np.full(7, 0.3)
    r._friction_transition_rad_s = np.zeros(7)
    r.gravity_comp_factor = np.ones(7)
    r._clip_motor_torque = np.inf
    r._limit_gripper_force = 0
    r._gripper_limits = [-5.0, 5.0]
    r._update_joint_state = lambda tau, commands: setattr(r, "sent", (tau.copy(), commands))
    return r


def test_sdk_pid_then_pd_and_gravity_are_isolated():
    r = sdk_stub()
    target = np.full(7, 0.1)
    r.command_joint_pos_pid(target, np.ones(7) * 10, np.ones(7))
    r.update()
    r._integral_time -= 0.01
    r.update()
    assert np.all(r.sent[0][:6] > 0)
    assert r.sent[0][6] == 0  # gripper always original PD
    np.testing.assert_equal(r.sent[1].kp, r._kp)
    session = r._commands.integral_session
    r.command_joint_pos_pid(target, np.ones(7) * 10, np.ones(7))
    assert r._commands.integral_session == session
    r.command_joint_pos_pid(target * 2, np.ones(7) * 10, np.ones(7))
    assert r._commands.integral_session != session
    r.update()
    np.testing.assert_equal(r.get_integral_torque(), 0)
    r.command_joint_pos(target)
    r.update()
    np.testing.assert_equal(r.sent[0], 0)
    np.testing.assert_equal(r.get_integral_torque(), 0)
    r.command_joint_pos_pid(target, np.ones(7) * 10, np.ones(7))
    assert r._commands.integral_session != session
    r.enter_gravity_comp_idle()
    r.update()
    assert r._commands.integral_session is None
    np.testing.assert_equal(r.sent[0], 0)


def test_original_pd_function_is_byte_for_byte_unchanged():
    source = subprocess.check_output(
        ["git", "-C", "vendor/i2rt", "show", "HEAD:i2rt/robots/motor_chain_robot.py"], text=True
    )
    original = source[
        source.index("    def command_joint_pos(") : source.index("    def command_joint_state(")
    ]
    assert inspect.getsource(MotorChainRobot.command_joint_pos).strip() == original.strip()


def test_sdk_rejects_invalid_gains_before_changing_command():
    r = sdk_stub()
    original = r._commands
    with pytest.raises(ValueError):
        r.command_joint_pos_pid(np.zeros(7), np.full(7, -1), np.ones(7))
    assert r._commands is original


def test_ui_pid_routing_and_unchanged_target():
    r = sdk_stub()
    r.get_joint_pos = lambda: np.zeros(7)
    r.get_robot_info = lambda: {"gripper_limits": [-5.0, 5.0]}
    c = Controller(r, np.tile([-3.0, 3.0], (6, 1)), live=True)
    c.health = lambda: None
    c.select_controller("PID", [20.0] * 6)
    assert r._commands.integral_ki is not None
    c.select_controller("PD")
    assert r._commands.integral_ki is None
    np.testing.assert_equal(c.target, c.rest)


def test_deadband_holds_bias_and_zero_gain_clears_it():
    result = integral_torque(
        np.array([0.0005, 0.1]), np.array([10.0, 0.0]), 0.01, np.array([0.3, 0.3]), np.ones(2), deadband=0.001
    )
    np.testing.assert_allclose(result, [0.3, 0.0])


def test_pid_p_is_separate_while_d_and_original_pd_remain_factory():
    r = sdk_stub()
    original_p = r._kp.copy()
    original_d = r._kd.copy()
    r.command_joint_pos_pid(np.ones(7) * 0.1, np.ones(7), np.ones(7), kp=original_p * 1.2)
    np.testing.assert_allclose(r._commands.kp[:6], original_p[:6] * 1.2)
    assert r._commands.kp[6] == original_p[6]
    np.testing.assert_equal(r._commands.kd, original_d)
    np.testing.assert_equal(r._kp, original_p)
    r.command_joint_pos(np.zeros(7))
    np.testing.assert_equal(r._commands.kp, original_p)
    np.testing.assert_equal(r._commands.kd, original_d)


def test_native_d_override_preserves_factory_pd_and_gripper():
    r = sdk_stub()
    r.command_joint_pos_pid(np.zeros(7), np.zeros(7), np.ones(7), kd=np.array([4,4,4,1.8,1.8,1.8,0]))
    np.testing.assert_equal(r._commands.kd, [4,4,4,1.8,1.8,1.8,1])
    r.command_joint_pos(np.zeros(7))
    np.testing.assert_equal(r._commands.kd, r._kd)
    before = r._commands
    with pytest.raises(ValueError):
        r.command_joint_pos_pid(np.zeros(7), np.zeros(7), np.ones(7), kd=np.full(7, 5.01))
    assert r._commands is before


def test_p_changes_are_bounded_against_current_setting_and_atomic():
    r = sdk_stub()
    r.get_joint_pos = lambda: np.zeros(7)
    r.get_robot_info = lambda: {"gripper_limits": [-5.0, 5.0]}
    c = Controller(r, np.tile([-3.0, 3.0], (6, 1)), live=True)
    c.health = lambda: None
    assert c.integral_deadband_deg == 0.06
    for factor in (1.2, 1.2, 0.8):
        expected = c.kp * factor
        c.select_controller("PID", kp=expected)
        np.testing.assert_allclose(r._commands.kp[:6], expected)
        np.testing.assert_equal(r._commands.kd[:6], c.kd)
    previous = c.kp.copy()
    command = r._commands
    for invalid in (previous * 1.201, previous * 0.799, np.zeros(6), np.full(6, np.nan)):
        with pytest.raises(ValueError):
            c.select_controller("PID", ki=np.ones(6), kp=invalid)
        np.testing.assert_equal(c.kp, previous)
        assert r._commands is command


def test_app_integral_deadband_stays_within_accuracy_criterion():
    r = sdk_stub()
    r.get_joint_pos = lambda: np.zeros(7)
    r.get_robot_info = lambda: {"gripper_limits": [-5.0, 5.0]}
    c = Controller(r, np.tile([-3.0, 3.0], (6, 1)), live=True)
    c.health = lambda: None
    c.select_controller("PID", deadband_deg=0.09)
    assert r._commands.integral_deadband == pytest.approx(np.deg2rad(0.09))
    np.testing.assert_equal(r._commands.kp[:6], c.kp)
    assert r._commands.kp[6] == r._kp[6]
    np.testing.assert_equal(r._commands.kd[:6], c.kd)
    for invalid in (0.201, -0.01, float("nan")):
        with pytest.raises(ValueError):
            c.select_controller("PID", deadband_deg=invalid)
    assert c.integral_deadband_deg == 0.09


def test_integral_delay_keeps_initial_move_factory_pd(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: now[0])
    r = sdk_stub()
    r.command_joint_pos_pid(np.ones(7) * 0.1, np.ones(7) * 10, np.ones(7), integral_delay=0.5)
    r.update()
    now[0] = 100.49
    r.update()
    np.testing.assert_equal(r.sent[0], 0)
    now[0] = 100.51
    r.update()
    np.testing.assert_allclose(r.sent[0][:6], 0.01)
    r.command_joint_pos_pid(np.ones(7) * 0.2, np.ones(7) * 10, np.ones(7), integral_delay=0.5)
    now[0] = 100.52
    r.update()
    np.testing.assert_equal(r.sent[0], 0)


def test_ui_integrates_immediately_and_resets_only_changed_waypoints(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: now[0])
    r = sdk_stub()
    r.get_joint_pos = lambda: np.zeros(7)
    r.get_robot_info = lambda: {"gripper_limits": [-5.0, 5.0]}
    c = Controller(r, np.tile([-3.0, 3.0], (6, 1)), live=True)
    c.health = lambda: None
    c.select_controller("PID", [10.0] * 6)
    target = np.r_[np.full(6, 0.1), 0.0]
    c.command(target)
    assert r._commands.integral_delay == 0.0
    r.update()
    now[0] += 0.004
    r.update()
    np.testing.assert_allclose(r.get_integral_torque()[:6], 0.004)
    c.command(target.copy())
    now[0] += 0.004
    r.update()
    np.testing.assert_allclose(r.get_integral_torque()[:6], 0.008)
    c.command(-target)
    r.update()
    np.testing.assert_equal(r.get_integral_torque(), 0)
    now[0] += 0.004
    r.update()
    np.testing.assert_allclose(r.get_integral_torque()[:6], -0.004)
    assert r.get_integral_torque()[6] == 0
