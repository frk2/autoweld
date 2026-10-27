"""Hardware-free transport, startup, executor and UI policy regressions."""

import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.motor_drivers.dm_driver import ControlMode, DMChainCanInterface, DMSingleMotorCanInterface, MotorCmd
from i2rt.motor_drivers.position_speed import prepare_modes
from i2rt.robots.position_speed_robot import PositionSpeedRobot
from test_control import FakeRobot
from test_integral import sdk_stub

from autoweld.control import Controller
from autoweld.motion import MotionLimits


def interface(mode="POS_VEL"):
    i = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    i.control_mode = mode
    i.cmd_idoffset = ControlMode.get_id_offset(mode)
    i.parse_recv_message = lambda msg, *args, **kwargs: msg
    i._send_message_get_response = lambda *args, **kwargs: args
    return i


def test_float_encoding_and_mixed_mode_gripper():
    i = interface()
    result = i.set_control(2, "DM4340", -.456789, .15, 0, 0, 0)
    assert result[:2] == (0x102, 2)
    assert struct.unpack("<ff", result[2]) == pytest.approx((-.456789, .15))
    i.motor_modes = {7: "MIT"}
    assert i.set_control(7, "DM4310", .1, 0, 5, 1, 0)[0] == 7


@pytest.mark.parametrize("values", [(0, 0, 0, 0, 0), (0, -.1, 0, 0, 0),
    (float("nan"), .1, 0, 0, 0), (13, .1, 0, 0, 0), (0, .1, 1, 0, 0),
    (0, .1, 0, 1, 0), (0, .1, 0, 0, 1)])
def test_invalid_pos_speed_never_sent(values):
    with pytest.raises(ValueError):
        interface().set_control(1, "DM4340", *values)


def test_chain_direction_changes_position_but_not_speed_ceiling():
    c = DMChainCanInterface.__new__(DMChainCanInterface)
    c.motor_list = [(1, "DM4340"), (7, "DM4310")]
    c.motor_offset = np.array([.4, .5])
    c.motor_direction = np.array([-1., -1.])
    c.position_speed_motor_ids = {1}
    seen = []
    c.motor_interface = SimpleNamespace(set_control=lambda **kw: (seen.append(kw), SimpleNamespace(error_code="0x1"))[1])
    c._set_commands([MotorCmd(torque=0, pos=.1, vel=.15), MotorCmd(torque=0, vel=.2)])
    assert seen[0]["pos"] == pytest.approx(.3)
    assert seen[0]["vel"] == .15
    assert seen[1]["vel"] == -.2


def test_mode_transition_requires_all_disabled_before_any_write(monkeypatch):
    from i2rt.motor_drivers import position_speed as ps
    writes = []
    fake = SimpleNamespace(read_motor_status=lambda i, _: SimpleNamespace(error_code="0x1" if i == 2 else "0x0", velocity=0), close=lambda: None)
    monkeypatch.setattr("i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface", lambda **kw: fake)
    defaults = dict(CTRL_MODE=1, Gr=40, PMAX=12.5, VMAX=10, TMAX=28, ACC=1, DEC=1,
                    MAX_SPD=10, TIMEOUT=1000, OT_Value=80, Damp=4, KP_ASR=1, KI_ASR=1, KP_APR=1, KI_APR=0)
    from i2rt.motor_drivers.utils import MotorType
    constants = MotorType.get_motor_constants("DM4340")
    defaults.update(VMAX=constants.VELOCITY_MAX, TMAX=constants.TORQUE_MAX)
    monkeypatch.setattr(ps, "read_register", lambda _, i, reg: defaults[reg])
    monkeypatch.setattr(ps, "write_register", lambda *args: writes.append(args))
    with pytest.raises(RuntimeError, match="disabled"):
        prepare_modes("unused", [(1, "DM4340"), (2, "DM4340")], {1: "POS_VEL", 2: "POS_VEL"})
    assert not writes


def test_ram_transition_checks_readback_and_accepts_signed_deceleration(monkeypatch):
    from i2rt.motor_drivers import position_speed as ps
    from i2rt.motor_drivers.utils import MotorType
    const = MotorType.get_motor_constants("DM4340")
    registers = dict(CTRL_MODE=1, Gr=40, PMAX=12.5, VMAX=const.VELOCITY_MAX,
        TMAX=const.TORQUE_MAX, ACC=2, DEC=-2, MAX_SPD=600, TIMEOUT=8000, OT_Value=80,
        Damp=.0002, KP_ASR=.00384, KI_ASR=.002, KP_APR=54, KI_APR=0)
    writes = []
    fake = SimpleNamespace(read_motor_status=lambda *args: SimpleNamespace(error_code="0x0", velocity=.002), close=lambda: None)
    monkeypatch.setattr("i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface", lambda **kw: fake)
    monkeypatch.setattr(ps, "read_register", lambda _, i, reg: registers[reg])

    def write(_, i, reg, value):
        writes.append((i, reg, value))
        registers[reg] = value

    monkeypatch.setattr(ps, "write_register", write)
    result = prepare_modes("unused", [(1, "DM4340")], {1: "POS_VEL"})
    assert writes == [(1, "CTRL_MODE", 2)]
    assert result[1]["selected_mode"] == 2
    assert result[1]["CTRL_MODE"] == 1
    registers["CTRL_MODE"] = 1
    monkeypatch.setattr(ps, "write_register", lambda *args: None)
    with pytest.raises(RuntimeError, match="read-back"):
        prepare_modes("unused", [(1, "DM4340")], {1: "POS_VEL"})


def test_startup_seeds_measured_position_before_enable(monkeypatch):
    from i2rt.motor_drivers import dm_driver
    calls = []
    state = SimpleNamespace(error_code="0x0", velocity=.002, position=.4321)
    c = DMChainCanInterface.__new__(DMChainCanInterface)
    c.motor_list = [(1, "DM4340")]
    c.position_speed_motor_ids = {1}
    c.motor_interface = SimpleNamespace(_drain_bus=lambda **kw: None,
        read_motor_status=lambda *args: state,
        set_control=lambda *args, **kw: calls.append(("seed", args, kw)),
        motor_on=lambda *args: (calls.append(("enable", args)), state)[1])
    c._update_absolute_positions = lambda *args: None
    monkeypatch.setattr(dm_driver.time, "sleep", lambda *args: None)
    c._motor_on()
    assert [x[0] for x in calls] == ["seed", "enable"]
    assert calls[0][1][2:4] == (.4321, .02)
    assert calls[0][2] == {"ignore_error": True}


def test_partial_enable_failure_disables_every_motor_and_closes_socket(monkeypatch):
    from i2rt.motor_drivers import dm_driver, position_speed
    events = []
    state = SimpleNamespace(error_code="0x0", velocity=0., position=.1)

    def enable(mid, kind):
        events.append(('enable', mid))
        if mid == 2:
            raise RuntimeError('simulated enable failure')
        return state

    fake = SimpleNamespace(_drain_bus=lambda **kw: None, read_motor_status=lambda *args: state,
        set_control=lambda *args, **kw: None, motor_on=enable,
        motor_off=lambda mid: events.append(('disable', mid)),
        close=lambda: events.append(('close',)))
    monkeypatch.setattr(dm_driver, 'DMSingleMotorCanInterface', lambda **kw: fake)
    monkeypatch.setattr(dm_driver, 'run_startup_checks', lambda *args, **kw: None)
    monkeypatch.setattr(position_speed, 'prepare_modes', lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match='simulated enable failure'):
        DMChainCanInterface([(1,'DM4340'),(2,'DM4340')], np.zeros(2), np.ones(2),
            channel='can0', start_thread=False, motor_modes={1:'POS_VEL',2:'POS_VEL'})
    assert events == [('enable',1),('enable',2),('disable',1),('disable',2),('close',)]


def test_position_speed_actual_update_clears_feedforward_and_keeps_gripper(monkeypatch):
    r = sdk_stub()
    r.__class__ = PositionSpeedRobot
    r.position_speed_limit = .15
    r.use_gravity_comp = False
    r.use_coulomb_friction = False
    # Let the real subclass hook process the output without a real chain.
    del r._update_joint_state
    from i2rt.robots.motor_chain_robot import MotorChainRobot
    monkeypatch.setattr(MotorChainRobot, "_update_joint_state", lambda self, tau, cmd, enc=None: setattr(self, "sent", (tau, cmd)))
    r.command_joint_pos(np.full(7, .1))
    r.update()
    tau, cmd = r.sent
    np.testing.assert_equal(tau[:6], 0)
    np.testing.assert_equal(cmd.kp[:6], 0)
    np.testing.assert_equal(cmd.kd[:6], 0)
    np.testing.assert_equal(cmd.vel[:6], .15)
    assert cmd.kp[6] == 5
    assert r.get_torque_breakdown()["onboard_loop_terms_available"] is False
    # A fault holds the measured position with the same onboard controller.
    r._oscillation_guard.observe = lambda *args: {"reason": "test", "timestamp": time.time()}
    r.update()
    np.testing.assert_equal(r.sent[1].pos, 0)
    assert "position-speed hold" in r.get_motion_fault()["response"]
    with pytest.raises(RuntimeError, match="Gravity mode"):
        r.enter_gravity_comp_idle()


def test_ui_uses_only_position_speed_and_reconnects_in_same_mode():
    c = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    c.select_controller('POS_VEL')
    with pytest.raises(ValueError, match='Only position-speed'):
        c.select_controller('PID')
    c.closed = True
    c.disabled_q = c.rest.copy()
    calls = []
    c.robot_factory = lambda *args, **kwargs: (calls.append(kwargs), FakeRobot())[1]
    c.enable()
    assert calls == [{'arm_control_mode': 'POS_VEL'}]
    with pytest.raises(ValueError, match='six-axis'):
        c.gravity()


def test_position_speed_planner_caps_without_changing_operator_limits():
    c = Controller(FakeRobot(), np.tile([-3, 3], (6, 1)))
    c.select_controller("POS_VEL")
    c.motion_planner = SimpleNamespace(limits=MotionLimits(joint_velocity=3))
    assert c.planner_for_controller().limits.joint_velocity == 2.
    assert c.motion_planner.limits.joint_velocity == 3.


def test_temporary_speed_ceiling_bounded_by_operator_and_idle_state():
    c = Controller(FakeRobot(), np.tile([-3, 3], (6, 1)))
    c.select_controller("POS_VEL")
    c.motion_planner = SimpleNamespace(limits=MotionLimits(joint_velocity=.8))
    c.configure_position_speed_limit(.6)
    assert c.planner_for_controller().limits.joint_velocity == .6
    assert c.motion_planner.limits.joint_velocity == .8
    for value in (0, -.1, float('nan'), 3.01, .9):
        with pytest.raises(ValueError):
            c.configure_position_speed_limit(value)
    c.motion_plan = object()
    with pytest.raises(RuntimeError, match="idle"):
        c.configure_position_speed_limit(.15)
    assert c.position_speed_limit == .6
    r = sdk_stub()
    r.__class__ = PositionSpeedRobot
    r.configure_position_speed_limit(.6)
    assert r.position_speed_limit == .6
    with pytest.raises(ValueError):
        r.configure_position_speed_limit(3.01)


def test_controller_accepts_two_and_sdk_maximum_but_preserves_operator_ceiling():
    c = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    c.motion_planner = SimpleNamespace(limits=MotionLimits())
    c.configure_position_speed_limit(2.)
    assert c.position_speed_limit == c.planner_for_controller().limits.joint_velocity == 2.
    with pytest.raises(ValueError, match='operator joint velocity'):
        c.configure_position_speed_limit(2.1)
    c.motion_planner.limits = MotionLimits(joint_velocity=3.)
    c.configure_position_speed_limit(3.)
    assert c.position_speed_limit == 3.
    with pytest.raises(ValueError, match=r'\(0, 3\]'):
        c.configure_position_speed_limit(3.0001)
    assert c.position_speed_limit == 3.


def test_motion_start_waits_for_post_validation_feedback_or_rejects(monkeypatch):
    import autoweld.control as control
    c = Controller(FakeRobot(), np.tile([-3, 3], (6, 1)))
    c.live = True
    c.health = lambda: None
    now = [10.]
    monkeypatch.setattr(control.time, 'time', lambda: now[0])
    monkeypatch.setattr(control.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(control.time, 'sleep', lambda dt: now.__setitem__(0, now[0]+dt))
    c.robot.get_feedback_diagnostics = lambda: {'timestamp':9.99,'age_s':now[0]-9.99}
    c.robot.get_observations = lambda: {'joint_vel':np.zeros(7)}
    with pytest.raises(RuntimeError, match='No fresh SDK feedback'):
        c._wait_for_fresh_motion_start()
    assert not c.robot.commands
    initial = now[0]
    c.robot.get_feedback_diagnostics = lambda: {'timestamp':now[0],'age_s':0.}
    c._wait_for_fresh_motion_start()
    assert now[0] > initial
    c.robot.get_observations = lambda: {'joint_vel':np.full(7,.1)}
    with pytest.raises(RuntimeError, match='Arm moved'):
        c._wait_for_fresh_motion_start()


def test_disabled_viewer_preserves_normalized_gripper_with_real_calibration():
    from i2rt.robots.get_robot import get_yam_robot
    r = get_yam_robot(sim=True, gripper_limits_override=np.array([.2599755856, -2.70218204]))
    try:
        q = np.array([.038, .005, .023, -.07, .007, 0., .9855763])
        r.command_joint_pos(q)
        np.testing.assert_allclose(r.get_joint_pos(), q)
        c = Controller(r, np.tile([-3., 3.], (6, 1)))
        assert c.rest[6] == q[6]
        np.testing.assert_allclose(c.gripper_limits, [.2599755856, -2.70218204])
    finally:
        r.close()
