"""Gravity geometry, calibration and torque accounting without hardware."""
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots.gravity_feedforward import LinearGravity
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.mujoco_utils import MuJoCoKDL
from test_integral import sdk_stub


def test_downward_load_linearization_and_pose_direction_independence():
    def gravity(q):
        return np.array([0, 4 * np.sin(q[1]) + 2 * np.sin(q[1] + q[2]), 2 * np.sin(q[1] + q[2])])
    ref = np.array([0, .2, .3])
    model = LinearGravity(gravity, ref)
    q = ref + [.02, .01, -.01]
    actual, mode = model.evaluate(q, gravity(q))
    assert mode == "linear"
    np.testing.assert_allclose(actual, gravity(q), atol=.001)
    assert model.jacobian[1, 2] != 0  # distal load also acts on proximal joint
    assert actual[0] == 0  # vertical yaw axis does not get a fabricated down torque
    np.testing.assert_equal(model.evaluate(q, gravity(q))[0], actual)
    far = ref + .3
    np.testing.assert_equal(model.evaluate(far, gravity(far))[0], gravity(far))
    assert model.evaluate(far, gravity(far))[1] == "sdk_fallback"


def test_sdk_yam_local_model_matches_static_gravity_at_small_test_poses():
    kdl = MuJoCoKDL(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
    np.testing.assert_equal(kdl.model.opt.gravity, [0, 0, -9.81])
    ref = np.array([.021, .006, .022, -.066, .002, -.004])
    def gravity(q):
        return kdl.compute_inverse_dynamics(q, np.zeros(6), np.zeros(6)).copy()
    model = LinearGravity(gravity, ref)
    for delta in [[.035,.025,.06,.035,-.025,.035],[-.035,.015,.035,-.035,.025,-.035]]:
        q = ref + delta
        approx, mode = model.evaluate(q, gravity(q))
        assert mode == "linear"
        np.testing.assert_allclose(approx, gravity(q), atol=.06)
    assert abs(model.base[0]) < 1e-6


def test_sdk_torque_total_includes_gravity_friction_i_and_native_pd():
    r = sdk_stub()
    r._compute_gravity_compensation = lambda _: np.arange(7, dtype=float) / 10
    r._joint_state.vel[:] = .02
    r._joint_state.eff[:] = .4
    r.gravity_comp_factor[:] = 1.1
    r.command_joint_pos_pid(np.full(7,.1), np.ones(7), np.ones(7))
    r.update()
    r._integral_time -= .004
    r.update()
    data = r.get_torque_breakdown()
    assert data['gravity_enabled'] and data['friction_enabled']
    np.testing.assert_allclose(data['gravity_nm'], np.arange(7)*.11)
    np.testing.assert_allclose(data['friction_nm'], .3)
    np.testing.assert_allclose(data['feedforward_sent_nm'], np.array(data['gravity_nm']) + data['friction_nm'] + data['i_nm'])
    np.testing.assert_allclose(data['total_estimate_nm'], np.array(data['feedforward_sent_nm']) + data['p_estimate_nm'] + data['d_estimate_nm'])
    np.testing.assert_allclose(data['motor_reported_nm'], .4)
    r.command_joint_pos(np.zeros(7))
    r.update()
    np.testing.assert_equal(r.get_integral_torque(), 0)
    np.testing.assert_allclose(r.get_torque_breakdown()['gravity_nm'], np.arange(7)*.11)


def test_gravity_config_atomic_preserves_gripper_and_enables_sdk_compensation():
    r = sdk_stub()
    r.kdl = SimpleNamespace(compute_inverse_dynamics=lambda q, v, a: np.sin(q),
                            model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0,0,-9.81]))))
    r.configure_gravity(np.full(6,1.2), model='linear', reference=np.zeros(6))
    assert r.gravity_comp_factor[6] == 1
    assert r.use_gravity_comp and r.use_coulomb_friction
    assert r.get_gravity_info()['model'] == 'linear'
    before = r.gravity_comp_factor.copy()
    for bad in [np.zeros(6), np.ones(5), np.full(6,np.nan)]:
        with pytest.raises(ValueError):
            r.configure_gravity(bad)
        np.testing.assert_equal(r.gravity_comp_factor, before)
    r._motion_fault = {'reason':'test'}
    with pytest.raises(RuntimeError):
        r.configure_gravity(np.ones(6))
