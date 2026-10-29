"""Read-only service rest inspection/capture without opening hardware."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService
from autoweld.control import Controller
from autoweld.disabled_rest import DisabledRestError, inspect_disabled_pose, publish_rest_records


@pytest.fixture
def reader_rig():
    clocks = [10., 1000.]
    q = np.array([.25, .2, .3, 0., 0., -3.])
    bounds = np.tile([-6., 6.], (6, 1))
    bounds[5] = [-5.37, -2.08]
    state = dict(sequence=0, closed=0, changed=None, reads=0)

    def read(**kwargs):
        assert kwargs == {'include_motor_position': True}
        clocks[0] += .15
        clocks[1] += .15
        state['sequence'] += 1
        state['reads'] += 1
        stamp = clocks[1]-.005
        result = dict(motor_ids=list(range(1, 7)), disabled=True, status=['0x0']*6,
            mapping=dict(offsets_rad=[0.]*6, directions=[1]*6, boot_normalization='none'),
            sdk_feedback_position_rad=q.copy(), joint_velocity_rad_s=np.zeros(6),
            status_timestamps=[stamp]*6, physical_timestamp=stamp, sequence=state['sequence'], session='owned-reader',
            native_p_m=[dict(motor_id=index, register='p_m', timestamp=stamp, value=float(value))
                        for index, value in enumerate(q, 1)],
            native_xout=[{'value': 99.}]*6, mapped_joint_rad=[99.]*6)
        if state['changed']:
            state['changed'](result)
        return result

    reader = SimpleNamespace(read=read, close=lambda: state.update(closed=state['closed']+1))

    def sleep(duration):
        clocks[0] += duration
        clocks[1] += duration

    def inspect(factory=lambda: reader, passed_bounds=None):
        return inspect_disabled_pose(factory, bounds if passed_bounds is None else passed_bounds,
            clock=lambda: clocks[0], wall_clock=lambda: clocks[1], sleep=sleep)

    return SimpleNamespace(inspect=inspect, reader=reader, clocks=clocks, state=state, q=q, bounds=bounds)


def test_readonly_quiet_survey_uses_command_branch_and_closes_single_sdk_reader(reader_rig):
    r = reader_rig
    result = r.inspect()
    assert result['quiet_seconds'] >= 1. and len(result['samples']) >= 3
    np.testing.assert_array_equal(result['joint_rad'], r.q)
    assert result['samples'][0]['sample']['mapped_joint_rad'] == [99.]*6
    assert r.state['closed'] == 1 and result['commands_sent'] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['enabled', 'fault', 'stale', 'future', 'native', 'mapping', 'bounds', 'moving',
                                     'motor7', 'cached', 'drift'])
def test_bad_feedback_cannot_establish_rest_and_reader_always_closes(reader_rig, failure):
    r = reader_rig

    def change(sample):
        if failure == 'enabled':
            sample['disabled'] = False
        elif failure == 'fault':
            sample['status'][2] = '0xd'
        elif failure == 'stale':
            sample['physical_timestamp'] -= 1.
        elif failure == 'future':
            sample['native_p_m'][0]['timestamp'] += 1.
        elif failure == 'native':
            sample['native_p_m'][0]['value'] += .03
        elif failure == 'mapping':
            sample['mapping']['offsets_rad'][5] = 2*np.pi
        elif failure == 'bounds':
            sample['sdk_feedback_position_rad'][5] = -5.5
        elif failure == 'moving':
            sample['joint_velocity_rad_s'][0] = .06
        elif failure == 'motor7':
            sample['motor_ids'][-1] = 7
        elif failure == 'cached':
            sample['sequence'] = 1
        elif failure == 'drift' and sample['sequence'] > 1:
            sample['sdk_feedback_position_rad'][0] += .01
            sample['native_p_m'][0]['value'] += .01

    r.state['changed'] = change
    with pytest.raises(DisabledRestError):
        r.inspect()
    assert r.state['closed'] == 1


@pytest.fixture
def service(tmp_path, monkeypatch, reader_rig):
    from autoweld import disabled_rest
    r = reader_rig
    s = object.__new__(RobotService)
    robot = FakeRobot()
    robot.q[5] = -3.
    s.ctrl = Controller(robot, r.bounds.copy())
    s.ctrl.closed = s.ctrl.cached_rest_display = True
    s.ctrl.disabled_q = s.ctrl.rest.copy()
    s.ctrl.mode = 'Disabled'
    s.runtime, s.saved_pose_path = tmp_path, tmp_path/'poses.json'
    s.saved_poses = {'Existing': {'q': [0.]*6}}
    s.model_fingerprint = 'corrected-mount-model'
    s.disabled_reader_factory = lambda: r.reader
    s.disabled_rest_inspection = None
    s.exit_after_disable, s.pending_pose = False, None
    s.calibration = s.laser = s.scan = s.weld = s.recovery = None
    s.message = 'previous'
    s.planner = SimpleNamespace(fk=lambda q: np.array([[1., 0, 0, q[0]], [0, 1., 0, q[1]],
                                                       [0, 0, 1., q[2]], [0, 0, 0, 1.]]))
    s.has_self_collision = lambda q: False
    s.host_guard = SimpleNamespace(check=lambda **_: pytest.fail('Read-only inspection must bypass host-pressure admission'))
    (tmp_path/'rest.json').write_text('{"prior_rest": true}\n')
    s.saved_pose_path.write_text(json.dumps(s.saved_poses))
    monkeypatch.setattr(disabled_rest, 'inspect_disabled_pose', lambda factory, bounds: r.inspect(factory, bounds))
    return s


def test_service_inspect_updates_disabled_display_without_changing_rest_or_files(service, reader_rig):
    s, r = service, reader_rig
    previous = s.ctrl.rest.copy()
    previous_files = [(path, path.read_bytes()) for path in (s.runtime/'rest.json', s.saved_pose_path)]
    result = s.dispatch({'action': 'rest_inspect'})
    np.testing.assert_array_equal(s.ctrl.positions()[:6], r.q)
    np.testing.assert_array_equal(s.ctrl.rest, previous)
    assert s.ctrl.closed and not s.ctrl.cached_rest_display
    assert result['rest_unchanged'] and result['captured'] is False
    assert not s.ctrl.robot.commands
    assert all(path.read_bytes() == value for path, value in previous_files)


def test_service_capture_persists_measured_rest_and_fk_under_current_model_without_enabling(service, reader_rig):
    s, r = service, reader_rig
    result = s.dispatch({'action': 'rest_capture'})
    np.testing.assert_array_equal(s.ctrl.rest[:6], r.q)
    assert s.ctrl.closed and s.ctrl.rest_verified and not s.ctrl.robot.commands
    saved = json.loads((s.runtime/'rest.json').read_text())
    assert saved['arm_rad'] == r.q.tolist() and saved['measured_disabled']['samples']
    pose = json.loads(s.saved_pose_path.read_text())['Captured rest']
    assert pose['q'] == r.q.tolist() and pose['position'] == r.q[:3].tolist()
    assert pose['model_fingerprint'] == 'corrected-mount-model'
    assert result['captured'] and result['rest_unchanged'] is False


@pytest.mark.parametrize('blocker', ['enabled', 'shutdown', 'busy', 'writer', 'chain', 'retained', 'pending'])
def test_service_owner_gates_prevent_a_second_sdk_reader(service, reader_rig, blocker):
    s = service
    if blocker == 'enabled':
        s.ctrl.closed = False
    elif blocker == 'shutdown':
        s.exit_after_disable = True
    elif blocker == 'busy':
        s.weld = SimpleNamespace(busy=True)
    elif blocker == 'writer':
        s.ctrl.robot._server_thread = SimpleNamespace(is_alive=lambda: True)
    elif blocker == 'chain':
        s.ctrl.robot.motor_chain = SimpleNamespace(running=True)
    elif blocker == 'retained':
        s.recovery = SimpleNamespace(state=lambda: {'active': False}, retained_chain=object(), retained_reader=None)
    else:
        s.pending_pose = ({}, 0)
    old = s.ctrl.rest.copy()
    with pytest.raises(RuntimeError):
        s.dispatch({'action': 'rest_capture'})
    assert reader_rig.state['reads'] == 0
    np.testing.assert_array_equal(s.ctrl.rest, old)
    assert not s.ctrl.robot.commands


def test_failure_keeps_rest_display_and_files_unchanged(service, reader_rig):
    reader_rig.state['changed'] = lambda sample: sample.update(disabled=False)
    old = service.ctrl.rest.copy()
    display = service.ctrl.positions()
    files = [(p, p.read_bytes()) for p in (service.runtime/'rest.json', service.saved_pose_path)]
    with pytest.raises(DisabledRestError):
        service.dispatch({'action': 'rest_capture'})
    np.testing.assert_array_equal(service.ctrl.rest, old)
    np.testing.assert_array_equal(service.ctrl.positions(), display)
    assert all(path.read_bytes() == value for path, value in files)


def test_publishing_both_rest_records_rolls_back_if_second_replace_fails(tmp_path, monkeypatch):
    rest, poses = tmp_path/'rest.json', tmp_path/'poses.json'
    rest.write_bytes(b'old rest bytes')
    poses.write_bytes(b'old poses bytes')
    original = Path.replace

    def replace(path, target):
        if path.name == 'rest.json.rest-capture.tmp':
            raise OSError('rest publication blocked')
        return original(path, target)

    monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises(OSError):
        publish_rest_records(rest, poses, {'arm_rad': [0.]*6}, {'Captured rest': {}})
    assert rest.read_bytes() == b'old rest bytes' and poses.read_bytes() == b'old poses bytes'


def test_matching_startup_rest_preserves_historical_proof_without_claiming_a_fresh_read():
    from autoweld.disabled_rest import preserve_rest_provenance
    current = dict(arm_rad=[0.]*6, gripper_normalized=0., gripper_limits_motor_rad=[0., 1.], live=True)
    previous = {**current, 'model_fingerprint': 'captured-model',
                'measured_disabled': {'physical_timestamp': 123., 'captured': True}}
    saved = preserve_rest_provenance(previous, current)
    assert saved['measured_disabled'] == previous['measured_disabled']
    assert saved['model_fingerprint'] == 'captured-model'
    assert saved['measured_disabled']['physical_timestamp'] == 123.  # Never refreshed by startup.


@pytest.mark.parametrize('field,value', [('arm_rad', [.001, 0, 0, 0, 0, 0]), ('gripper_normalized', .1),
                                        ('gripper_limits_motor_rad', [-1., 1.])])
def test_different_startup_rest_drops_proof_for_the_previous_pose(field, value):
    from autoweld.disabled_rest import preserve_rest_provenance
    current = dict(arm_rad=[0.]*6, gripper_normalized=0., gripper_limits_motor_rad=[0., 1.], live=True)
    previous = {**current, field: value, 'measured_disabled': {'physical_timestamp': 123.}}
    assert 'measured_disabled' not in preserve_rest_provenance(previous, current)
