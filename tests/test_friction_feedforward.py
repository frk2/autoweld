"""Friction calibration and continuous compensation, without hardware."""
import numpy as np
import pytest
from i2rt.robots.friction_feedforward import friction_torque
from test_default_profile import robot_at
from test_integral import sdk_stub

from autoweld.control import Controller


def test_zero_width_preserves_sign_and_positive_width_is_continuous():
    v = np.array([-.1, -.001, 0., .001, .1])
    old = friction_torque(v, np.full(5, .06), np.zeros(5))
    np.testing.assert_equal(old, .06*np.sign(v))
    smooth = friction_torque(v, np.full(5, .06), np.full(5, .02))
    assert np.all(np.diff(smooth) > 0)
    np.testing.assert_allclose(smooth, -smooth[::-1])
    assert abs(smooth[1]) < .0031 and smooth[2] == 0
    assert .059 < smooth[-1] <= .06


def test_sdk_disable_isolates_gravity_and_reenable_uses_smooth_compensation():
    r = sdk_stub()
    r._joint_state.vel[:] = .001
    r._compute_gravity_compensation = lambda _: np.full(7, 2.)
    r.configure_friction(enabled=False, transition_rad_s=np.full(6, .02))
    r.configure_gravity(np.full(6, 1.1))
    assert not r.use_coulomb_friction  # Changing gravity must not invalidate calibration.
    r.update()
    np.testing.assert_allclose(r.get_torque_breakdown()['friction_nm'], 0)
    np.testing.assert_allclose(r.sent[0][:6], 2.2)
    assert r._friction_transition_rad_s[6] == 0
    r.configure_friction(enabled=True)
    r.update()
    np.testing.assert_allclose(r.get_torque_breakdown()['friction_nm'][:6], .3*np.tanh(.05))
    assert r.use_gravity_comp


def test_friction_invalid_configuration_and_fault_do_not_mutate_settings():
    r = sdk_stub()
    for bad in [np.full(6, -1.), np.full(6, np.nan), np.zeros(5)]:
        with pytest.raises(ValueError):
            r.configure_friction(enabled=False, transition_rad_s=bad)
        assert r.use_coulomb_friction
        np.testing.assert_equal(r._friction_transition_rad_s, 0)
    r._motion_fault = {'reason': 'test'}
    with pytest.raises(RuntimeError):
        r.configure_friction(enabled=False)
    assert r.use_coulomb_friction


def test_disabled_friction_configuration_survives_enable():
    q = np.zeros(7)
    r = robot_at(q)
    c = Controller(r, np.tile([-3., 3.], (6, 1)), live=True)
    c.closed = True
    c.disabled_q = q.copy()
    c.configure_friction(False, np.full(6, .02))
    assert r.use_coulomb_friction
    fresh = robot_at(q)
    c.robot_factory = lambda *_, **_kw: fresh
    c.enable()
    assert not fresh.use_coulomb_friction
    np.testing.assert_equal(fresh._friction_transition_rad_s[:6], .02)
