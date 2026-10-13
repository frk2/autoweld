"""Mixed J4/J5 MIT routing and lifecycle, with no hardware access."""
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.position_speed_robot import PositionSpeedRobot
from test_control import FakeRobot
from test_integral import sdk_stub

from autoweld.control import Controller


def mixed_stub(monkeypatch):
    r = sdk_stub()
    r.__class__ = PositionSpeedRobot
    r.mit_arm_joint_indices = (3, 4)
    r.position_speed_limit = .25
    r.use_coulomb_friction = False
    r.kdl = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0., 0., -9.81]))))
    r._compute_gravity_compensation = lambda _: np.r_[np.arange(1, 7)*.1, 0]
    del r._update_joint_state
    monkeypatch.setattr(MotorChainRobot, '_update_joint_state',
                        lambda self, tau, cmd, enc=None: setattr(self, 'sent', (tau, cmd)))
    return r


def test_mixed_masks_only_position_speed_fields_and_preserves_signed_mit_velocity(monkeypatch):
    r = mixed_stub(monkeypatch)
    r.command_joint_pos_pid(np.full(7, .1), np.zeros(7), np.ones(7),
                           kp=np.r_[np.full(6, 12.), 5.], kd=np.r_[np.full(6, 2.), 0.])
    r._commands.vel[:] = -.07
    r.update()
    tau, cmd = r.sent
    pv = [0, 1, 2, 5]
    np.testing.assert_equal(tau[pv], 0)
    np.testing.assert_allclose(tau[[3, 4]], [.4, .5])
    np.testing.assert_equal(cmd.kp[pv], 0)
    np.testing.assert_equal(cmd.kd[pv], 0)
    np.testing.assert_equal(cmd.vel[pv], .25)
    np.testing.assert_equal(cmd.kp[[3, 4]], 12)
    np.testing.assert_equal(cmd.kd[[3, 4]], 2)
    np.testing.assert_equal(cmd.vel[[3, 4]], -.07)
    assert cmd.kp[6] == 5 and tau[6] == 0
    terms = r.get_torque_breakdown()
    assert terms['controller'] == 'MIXED_J45'
    assert terms['host_terms_valid_arm_joints'] == [4, 5]
    assert r.get_gravity_info()['active_arm_joints'] == [4, 5]
    np.testing.assert_equal(np.asarray(terms['gravity_nm'])[pv], 0)
    np.testing.assert_allclose(np.asarray(terms['gravity_nm'])[[3, 4]], [.4, .5])


def test_mixed_fault_holds_position_speed_and_factory_mit_without_mode_switch(monkeypatch):
    r = mixed_stub(monkeypatch)
    r.command_joint_pos(np.full(7, .1))
    r._oscillation_guard.observe = lambda *args: {'reason': 'test'}
    r.update()
    tau, cmd = r.sent
    np.testing.assert_equal(cmd.pos[:6], 0)
    np.testing.assert_equal(cmd.kp[[0, 1, 2, 5]], 0)
    np.testing.assert_equal(cmd.kp[[3, 4]], 10)
    np.testing.assert_equal(cmd.kd[[3, 4]], 1.5)
    np.testing.assert_equal(cmd.vel[[3, 4]], 0)
    np.testing.assert_equal(cmd.vel[[0, 1, 2, 5]], .25)
    np.testing.assert_allclose(tau[[3, 4]], [.4, .5])
    assert 'mixed-mode' in r.get_motion_fault()['response']


def test_constructor_enables_gravity_only_for_mixed(monkeypatch):
    monkeypatch.setattr(MotorChainRobot, '__init__', lambda self, *a, **kw: setattr(self, 'options', kw))
    pure = PositionSpeedRobot()
    mixed = PositionSpeedRobot(mit_arm_joint_indices=(3, 4))
    assert not pure.options['use_gravity_comp']
    assert mixed.options['use_gravity_comp'] and not mixed.options['use_coulomb_friction']
    with pytest.raises(ValueError):
        PositionSpeedRobot(mit_arm_joint_indices=(0, 1))


def test_mixed_requires_disabled_transition_and_routes_reconnect_and_tuned_pd():
    c = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)), live=True)
    c.health = lambda: None
    with pytest.raises(RuntimeError, match='Disable motors'):
        c.select_controller('MIXED_J45')
    c.closed = True
    c.disabled_q = c.rest.copy()
    c.select_controller('MIXED_J45')
    calls = []
    r = FakeRobot()
    r.configure_gravity = lambda *args, **kw: calls.append(('gravity', args, kw))
    r.configure_friction = lambda **kw: None
    r.command_joint_pos_pid = lambda q, ki, limits, **kw: calls.append(('pd', q, ki, kw))
    c.robot_factory = lambda *args, **kw: (calls.append(('connect', kw)), r)[1]
    c.enable()
    assert calls[0] == ('connect', {'arm_control_mode': 'MIXED_J45'})
    assert calls[1][0] == 'gravity'
    assert calls[2][0] == 'pd'
    np.testing.assert_allclose(calls[2][3]['kp'][:6], c.kp)
    np.testing.assert_allclose(calls[2][3]['kd'][:6], c.kd)
    for mode in ('POS_VEL', 'PID'):
        with pytest.raises(RuntimeError, match='Disable motors'):
            c.select_controller(mode)
    with pytest.raises(ValueError, match='Gravity mode'):
        c.gravity()


def test_mixed_can_be_selected_on_start_disabled_cached_robot():
    c = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)), live=True)
    c.closed = True
    c.robot = SimpleNamespace()  # Startup cache has no hardware command API.
    c.select_controller('MIXED_J45')
    assert c.control_law == 'MIXED_J45'


def test_factory_assigns_four_position_speed_and_two_mit_arm_motors(monkeypatch):
    from i2rt.robots import get_robot as factory
    seen = {}
    def chain(motors, offsets, directions, channel, **kwargs):
        seen.update(kwargs)
        states = [SimpleNamespace(pos=0.) for _ in motors]
        return SimpleNamespace(motor_offset=np.zeros(7), commands=[SimpleNamespace(pos=0.) for _ in motors],
                               state=[SimpleNamespace(position=0.) for _ in motors],
                               position_speed_motor_ids={1, 2, 3, 6}, read_states=lambda: states,
                               _joint_position_real_to_sim_idx=lambda pos, idx: pos,
                               start_thread=lambda: None)
    monkeypatch.setattr(factory, 'DMChainCanInterface', chain)
    monkeypatch.setattr('i2rt.robots.position_speed_robot.PositionSpeedRobot', lambda **kwargs: kwargs)
    result = factory.get_yam_robot(arm_control_mode='MIXED_J45', gripper_limits_override=np.array([.2, -2.7]))
    assert seen['motor_modes'] == {1:'POS_VEL', 2:'POS_VEL', 3:'POS_VEL', 4:'MIT', 5:'MIT', 6:'POS_VEL', 7:'MIT'}
    assert result['mit_arm_joint_indices'] == (3, 4)
    assert not result['enable_gripper_calibration']
