"""Wider commissioning envelopes and offline preflight gates; no hardware I/O."""
import mujoco
import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService
from autoweld.control import Controller
from autoweld.joint_calibration import j6_operating_range
from autoweld.kinematics import shared_kinematics
from autoweld.models import compiled_model, tool_model
from autoweld.motion import MotionPlanner


def make_controller():
    robot = FakeRobot()
    robot.q = np.array([0., .35, .65, .05, -.1, -2.8])
    bounds = np.tile([-3., 3.], (6, 1))
    bounds[5] = j6_operating_range()
    controller = Controller(robot, bounds)
    controller.strict_joint_limits = {5}
    return controller


@pytest.mark.parametrize('limit', [.49, 1., 1.8])
def test_envelope_is_disabled_only_bounded_and_nonpersistent(limit):
    controller = make_controller()
    with pytest.raises(RuntimeError, match='motors disabled'):
        controller.configure_gravity_test_envelope(limit)
    assert controller.test_excursion == .49
    controller.closed = True
    controller.disabled_q = controller.rest.copy()
    controller.configure_gravity_test_envelope(limit)
    assert controller.test_excursion == limit
    assert make_controller().test_excursion == .49
    assert not controller.robot.commands and not controller.robot.closed


@pytest.mark.parametrize('limit', [0., .4899, 1.80001, -1., float('nan'), float('inf')])
def test_invalid_envelope_does_not_change_prior_configuration(limit):
    controller = make_controller()
    controller.closed = True
    controller.disabled_q = controller.rest.copy()
    controller.configure_gravity_test_envelope(1.2)
    with pytest.raises(ValueError, match=r'\[0.49, 1.8\]'):
        controller.configure_gravity_test_envelope(limit)
    assert controller.test_excursion == 1.2


def test_wider_envelope_keeps_strict_rest_current_and_prior_target_constraints():
    controller = make_controller()
    controller.closed = True
    controller.disabled_q = controller.rest.copy()
    controller.configure_gravity_test_envelope(1.2)
    controller.closed = False
    controller.live = True
    controller.health = lambda: None
    target = controller.rest.copy()
    target[0] += 1.199
    controller.validate(target, automatic=True)
    target[0] = controller.rest[0]+1.2
    with pytest.raises(ValueError, match='1.2'):
        controller.validate(target, automatic=True)
    target[0] = controller.rest[0]+.7
    controller.target[0] = controller.rest[0]-.6
    with pytest.raises(ValueError, match='prior target'):
        controller.validate(target, automatic=True)
    assert not controller.robot.commands


@pytest.fixture(scope='module')
def shared_model():
    return compiled_model()


@pytest.fixture
def service(shared_model):
    service = object.__new__(RobotService)
    service.ctrl = make_controller()
    service.ctrl.robot._model = shared_model
    service.ctrl.limits = shared_model.jnt_range[:6].copy()
    service.ctrl.disabled_q = service.ctrl.rest.copy()
    service.ctrl.closed = True
    service.ctrl.configure_gravity_test_envelope(1.2)
    service.planner = MotionPlanner(tool_model(), kin=shared_kinematics(service.ctrl.robot))
    service.ctrl.motion_planner = service.planner
    service.gravity_payload = None
    return service


def request(service, **updates):
    q = service.ctrl.rest[:6].tolist()
    return {'start_q': q, 'q': q, **updates}


@pytest.mark.parametrize('key', ['start_q', 'q'])
@pytest.mark.parametrize('bad', [[0.]*5, [0.]*7, [[0.]*6], [float('nan')]*6, [float('inf')]*6])
def test_preflight_rejects_bad_six_axis_inputs_before_planning(service, monkeypatch, key, bad):
    monkeypatch.setattr(service, 'planner_for_gravity_test', lambda: pytest.fail('Invalid input reached planner'))
    with pytest.raises(ValueError, match='finite six-axis'):
        service.gravity_test_preflight(request(service, **{key: bad}))
    assert not service.ctrl.robot.commands


def test_preflight_and_envelope_json_require_disabled_motors(service):
    result = service.dispatch({'action': 'gravity_test_envelope', 'excursion_rad': 1.3})
    assert result == {'excursion_rad': 1.3, 'persistent': False}
    service.ctrl.closed = False
    with pytest.raises(RuntimeError, match='motors disabled'):
        service.dispatch({'action': 'gravity_test_envelope', 'excursion_rad': 1.4})
    with pytest.raises(RuntimeError, match='motors disabled'):
        service.gravity_test_preflight(request(service))
    assert service.ctrl.test_excursion == 1.3
    assert not service.ctrl.robot.commands


def test_preflight_shares_sdk_model_and_reuses_scratch_without_device_calls(service, monkeypatch):
    import autoweld.gravity_calibration as calibration
    original = calibration.gravity_regressor
    observed = []
    def regressor(model, data, q):
        assert model is service.ctrl.robot._model is service.planner.model
        assert data.model is model and data is not service.planner.data
        observed.append(id(data))
        return original(model, data, q)
    monkeypatch.setattr(calibration, 'gravity_regressor', regressor)
    monkeypatch.setattr(service.ctrl.robot, 'get_joint_pos', lambda: pytest.fail('Offline preflight accessed robot'))
    monkeypatch.setattr(service.ctrl.robot, 'get_observations', lambda: pytest.fail('Offline preflight accessed robot'))
    assert service.planner_for_gravity_test().model is service.planner.model
    before_mass = service.planner.model.body_mass.copy()
    for _ in range(2):
        result = service.dispatch({'action': 'gravity_test_preflight', **request(service)})
        assert result['minimum_mount_height_m'] >= .05
        assert np.all(np.asarray(result['peak_gravity_nm']) <= [9., 9., 9., 3., 3., 3.])
        assert result['max_excursion_rad'] == 1.2
        assert result['duration_s'] > 0
    assert len(observed) == 162 and len(set(observed)) == 1
    assert service.gravity_data.model is service.planner.model
    np.testing.assert_array_equal(service.planner.model.body_mass, before_mass)
    assert not service.ctrl.robot.commands and not service.ctrl.robot.closed


def test_preflight_rejects_configured_envelope_and_joint_bounds(service):
    target = service.ctrl.rest[:6].copy()
    target[0] = service.ctrl.rest[0]+1.2
    with pytest.raises(ValueError, match='gravity test envelope'):
        service.gravity_test_preflight(request(service, q=target.tolist()))
    target = service.ctrl.rest[:6].copy()
    target[3] = service.ctrl.limits[3, 1]+.01
    service.ctrl.test_excursion = 1.8
    with pytest.raises(ValueError, match='joint bounds'):
        service.gravity_test_preflight(request(service, q=target.tolist()))
    assert not service.ctrl.robot.commands


def test_preflight_checks_interior_self_collision(service, monkeypatch):
    start = service.ctrl.rest[:6].copy()
    target = start.copy()
    target[0] += .1
    monkeypatch.setattr(service, 'has_self_collision', lambda q: start[0]+.04 < q[0] < start[0]+.06)
    with pytest.raises(ValueError, match='self-collision'):
        service.gravity_test_preflight(request(service, q=target.tolist()))
    assert not service.ctrl.robot.commands


def mount_corner_minimum(planner):
    """Independent box-corner geometry, rather than projected half extents."""
    heights = []
    for gid in range(planner.model.ngeom):
        name = mujoco.mj_id2name(planner.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('mount_'):
            size = planner.model.geom_size[gid]
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*size
            world = corners@planner.data.geom_xmat[gid].reshape(3, 3).T + planner.data.geom_xpos[gid]
            heights.append(world[:, 2].min())
    return min(heights)


@pytest.mark.parametrize('height, accepted', [(.0499, False), (.0501, True)])
def test_mount_clearance_uses_rotated_boxes_and_correct_gate(service, monkeypatch, height, accepted):
    original = service.planner.fk
    def adjusted(q):
        pose = original(q)
        shift = height-mount_corner_minimum(service.planner)
        service.planner.data.geom_xpos[:, 2] += shift
        return pose
    monkeypatch.setattr(service.planner, 'fk', adjusted)
    monkeypatch.setattr(service, 'has_self_collision', lambda q: False)
    if accepted:
        result = service.gravity_test_preflight(request(service))
        assert result['minimum_mount_height_m'] == pytest.approx(height)
    else:
        with pytest.raises(ValueError, match='Mount clearance.*below 0.05m'):
            service.gravity_test_preflight(request(service))
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('joint, limit', [(0, 9.), (1, 9.), (2, 9.), (3, 3.), (4, 3.), (5, 3.)])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_gravity_torque_gate_checks_every_axis_both_signs(service, monkeypatch, joint, limit, sign):
    import autoweld.gravity_calibration as calibration
    baseline = np.zeros(6)
    baseline[joint] = sign*(limit+.001)
    monkeypatch.setattr(calibration, 'gravity_regressor', lambda *args: (baseline, np.zeros((6, 4))))
    with pytest.raises(ValueError, match='commissioning torque gate'):
        service.gravity_test_preflight(request(service))
    assert not service.ctrl.robot.commands


def test_torque_gate_includes_configured_payload_and_joint_factors(service, monkeypatch):
    import autoweld.gravity_calibration as calibration
    service.gravity_payload = {'mass_kg': .5, 'com_flange_m': [.01, .02, .03]}
    service.ctrl.gravity_factors[5] = 1.1
    matrix = np.zeros((6, 4))
    matrix[5, 0] = 6.  # Unscaled load3Nm: factor1.1 must reject3.3Nm.
    monkeypatch.setattr(calibration, 'gravity_regressor', lambda *args: (np.zeros(6), matrix))
    with pytest.raises(ValueError, match='commissioning torque gate'):
        service.gravity_test_preflight(request(service))
    assert not service.ctrl.robot.commands


def test_preflight_torque_gate_catches_interior_peak_with_safe_endpoints(service, monkeypatch):
    import autoweld.gravity_calibration as calibration
    start = service.ctrl.rest[:6].copy()
    goal = start.copy()
    goal[0] += .1
    def load(model, data, q):
        baseline = np.zeros(6)
        if start[0]+.04 < q[0] < start[0]+.06:
            baseline[4] = 3.01
        return baseline, np.zeros((6, 4))
    monkeypatch.setattr(calibration, 'gravity_regressor', load)
    with pytest.raises(ValueError, match='commissioning torque gate'):
        service.gravity_test_preflight(request(service, q=goal.tolist()))
    assert not service.ctrl.robot.commands


def test_preflight_separates_path_floor_clearance_from_elevated_endpoint(service, monkeypatch):
    original = service.planner.fk
    start = service.ctrl.rest[:6].copy()
    goal = start.copy()
    goal[0] += .1
    def adjusted(q):
        pose = original(q)
        height = .06 + .6*(q[0]-start[0])
        service.planner.data.geom_xpos[:, 2] += height-mount_corner_minimum(service.planner)
        return pose
    monkeypatch.setattr(service.planner, 'fk', adjusted)
    monkeypatch.setattr(service, 'has_self_collision', lambda q: False)
    result = service.gravity_test_preflight(request(service, q=goal.tolist()))
    assert result['minimum_mount_height_m'] == pytest.approx(.06)
    assert result['endpoint_mount_height_m'] == pytest.approx(.12)
    assert not service.ctrl.robot.commands
