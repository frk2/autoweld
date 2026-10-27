"""Installed travel, startup gating and captured-rest safeguards; no hardware."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yourdfpy
from test_control import FakeRobot

from autoweld.control import Controller
from autoweld.joint_calibration import j6_operating_range, load_joint_calibration, verify_disabled_start
from autoweld.models import URDF, compiled_model


def test_model_and_visual_match_measured_endpoints_with_clearance():
    record = load_joint_calibration()
    bounds = j6_operating_range(record)
    physical = record['j6_physical_endpoints_rad']
    assert bounds[0] > physical[0] and bounds[1] < physical[1]
    assert np.rad2deg(bounds) == pytest.approx([-307.8276, -119.5848], abs=.001)
    np.testing.assert_allclose(compiled_model().jnt_range[5], bounds)
    joint = yourdfpy.URDF.load(str(URDF), load_meshes=False).joint_map['joint6']
    np.testing.assert_allclose([joint.limit.lower, joint.limit.upper], bounds)


@pytest.mark.parametrize('change', [
    {'j6_physical_endpoints_rad': [-2., -5.]}, {'clearance_rad': 0.},
    {'clearance_rad': 4.}, {'fixed_motor_offsets_rad': [0.]*5},
    {'fixed_motor_offsets_rad': [float('nan')]*6}, {'schema': 'unknown'},
])
def test_invalid_calibration_is_rejected(tmp_path, change):
    record = load_joint_calibration() | change
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_joint_calibration(path)


def test_captured_rest_cannot_expand_calibrated_j6_limits():
    robot = FakeRobot()
    controller = Controller(robot, np.tile([-3., 3.], (6, 1)))
    controller.strict_joint_limits = {5}
    controller.rest[5] = 3.1
    controller.rest[0] = 3.1  # Existing supported-rest exception for other axes.
    bounds = controller.effective_joint_bounds()
    assert bounds[0, 1] == 3.1 and bounds[5, 1] == 3.
    with pytest.raises(ValueError, match='J6'):
        controller.validate(controller.rest.copy())
    assert not robot.commands


@pytest.mark.parametrize('disabled,q6,native_delta,accepted', [
    (True, -3., 0., True), (True, -5.39, 0., False),
    (True, -2.07, 0., False), (False, -3., 0., False),
    (True, -3., .03, False),
])
def test_read_only_startup_checks_close_sdk_before_any_enable(monkeypatch, disabled, q6, native_delta, accepted):
    import i2rt.robots.encoder_teaching as sdk
    closed = []
    q = np.array([0., .3, .6, 0., 0., q6])
    sample = dict(disabled=disabled, sdk_feedback_position_rad=q,
                  native_p_m=[{'value': v+native_delta} for v in q], mapping={'directions': [1]*6})
    monkeypatch.setattr(sdk, 'ReadOnlyArmEncoder', lambda *a, **kw: SimpleNamespace(
        read=lambda **kw: sample, close=lambda: closed.append(True)))
    limits = np.tile([-6., 6.], (6, 1))
    limits[5] = j6_operating_range()
    if accepted:
        np.testing.assert_allclose(verify_disabled_start('can0', limits, np.zeros(6)), q)
    else:
        with pytest.raises(RuntimeError):
            verify_disabled_start('can0', limits, np.zeros(6))
    assert closed == [True]


@pytest.mark.parametrize('q6', [-5.37, -2.09])
def test_gravity_entry_rejects_endpoint_margin_before_mode_switch(q6):
    robot = FakeRobot()
    robot.q = np.array([0., .3, .6, 0., 0., q6])
    controller = Controller(robot, np.tile([-6., 6.], (6, 1)))
    controller.limits[5] = j6_operating_range()
    controller.strict_joint_limits = {5}
    with pytest.raises(RuntimeError, match='Move inward'):
        controller.gravity()
    assert controller.control_law == 'POS_VEL'
    assert not robot.commands and not controller.closed


def test_gravity_limit_guard_requests_hold_without_disabling(monkeypatch):
    robot = FakeRobot()
    robot.q = np.array([0., .3, .6, 0., 0., -2.1])
    controller = Controller(robot, np.tile([-6., 6.], (6, 1)))
    controller.limits[5] = j6_operating_range()
    controller.strict_joint_limits = {5}
    controller.mode = 'Gravity compensation'
    held = []
    monkeypatch.setattr(controller, 'hold', lambda: held.append(True))
    controller.tick()
    assert held == [True] and not controller.closed and not robot.closed
