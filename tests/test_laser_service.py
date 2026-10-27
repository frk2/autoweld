"""Laser requests share the existing service owner and preserve physical gates."""
import json
import queue
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService
from autoweld.control import Controller


@pytest.fixture
def service(tmp_path):
    s = object.__new__(RobotService)
    s.runtime = tmp_path
    (tmp_path/'requests').mkdir()
    s.events = queue.Queue()
    s.ctrl = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    s.camera = SimpleNamespace(state=lambda **kwargs: {})
    s.pending_pose = ({'action': 'pose'}, 0.)
    s.request_status = None
    s.exit_after_disable = False
    s.calibration = None
    s.running = True
    s.saved_poses = {'Calibration Pose': {'q': [.1, .2, .3, 0., 0., 0.]}}
    s.calls = []
    s.laser = SimpleNamespace(busy=False,
        dispatch=lambda req: s.calls.append(req) or {'status': 'accepted'},
        state=lambda **kwargs: {'status': 'idle', 'cloud': {'count': 0}},
        cancel=lambda: s.calls.append('cancel'),
        abort_for_shutdown=lambda: s.calls.append('shutdown'),
        abort_for_cooldown=lambda: s.calls.append('cooldown'),
        tick=lambda: s.calls.append('tick'))
    return s


def test_laser_json_actions_use_coordinator_and_never_construct_camera(service):
    result = service.handle_request({'action': 'laser_auto', 'go_to_calibration_pose': True, '_id': 'laser-test'})
    assert result['ok'] and service.pending_pose is None
    assert service.calls == [{'action': 'laser_auto', 'go_to_calibration_pose': True, '_id': 'laser-test'}]
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('reason', ['disabled', 'shutdown', 'cooldown', 'moving', 'missing_pose', 'camera_busy'])
def test_laser_start_rejects_unready_motion_ownership(service, reason):
    if reason == 'disabled':
        service.ctrl.closed = True
    elif reason == 'shutdown':
        service.exit_after_disable = True
    elif reason == 'cooldown':
        service.ctrl.cooldown_active = True
    elif reason == 'moving':
        service.ctrl.motion_plan = object()
    elif reason == 'missing_pose':
        service.saved_poses.clear()
    else:
        service.calibration = SimpleNamespace(future=SimpleNamespace(done=lambda: False))
    with pytest.raises(RuntimeError):
        service.can_start_laser()
    assert not service.ctrl.robot.commands


def test_laser_saved_pose_runs_through_controller_without_commissioning_or_speed_override(service):
    calls = []
    requested_pose = np.eye(4)
    requested_pose[0, 3] = .125
    service.planner = SimpleNamespace(fk=lambda q: requested_pose)
    service.ctrl.command = lambda *args, **kwargs: calls.append((args, kwargs))
    returned = service.move_laser_saved_pose('Calibration Pose')
    args, kwargs = calls[0]
    np.testing.assert_array_equal(args[0][:6], service.saved_poses['Calibration Pose']['q'])
    assert args[1] == 'Hold' and kwargs == {'automatic': False}
    assert service.ctrl.position_speed_limit == 2.
    np.testing.assert_array_equal(returned, requested_pose)
    assert returned is not requested_pose


def test_intrinsic_preemption_releases_sequence_without_extra_hold(service):
    service.calibration = SimpleNamespace(abort_for_shutdown=lambda: service.calls.append('intrinsic-abort'))
    service.preempt_for_laser()
    assert service.pending_pose is None and service.calls == ['intrinsic-abort']
    assert not service.ctrl.robot.commands


def test_operator_hold_preempts_laser_before_dispatching_control(service):
    service.laser.busy = True
    result = service.handle_request({'action': 'mode', 'mode': 'Hold'})
    assert result['ok'] and service.calls == ['cancel']
    assert len(service.ctrl.robot.commands) == 1


def test_shutdown_releases_laser_without_camera_cleanup_blocking_exit_intent(service):
    service.laser.abort_for_shutdown = lambda: (_ for _ in ()).throw(RuntimeError('Cleanup failed'))
    result = service.handle_request({'action': 'shutdown'})
    assert result['ok'] and service.ctrl.shutdown_requested and service.exit_after_disable
    assert not service.ctrl.robot.commands
    result = service.handle_request({'action': 'laser_auto'})
    assert not result['ok'] and 'Shutdown owns' in result['error']


def test_cooldown_aborts_laser_without_issuing_hold(service):
    service.ctrl.cooldown_active = True
    service.ctrl.tick = lambda: None
    service.laser.busy = True
    service.pending_pose = None
    service.tick()
    assert service.calls == ['cooldown', 'tick']
    assert not service.ctrl.robot.commands


def test_cloud_registration_uses_measured_pose_only_when_quiet_and_enabled(service, monkeypatch):
    inputs = []
    monkeypatch.setattr('autoweld.laser_calibration.transform_cloud',
        lambda cloud, pose, *a, **kw: inputs.append(pose) or {'points_world_m': [[1., 2., 3.]], 'count': 1})
    service.tcp_pose = lambda: np.eye(4)
    clock = [100.]
    monkeypatch.setattr('autoweld.backend.time.monotonic', lambda: clock[0])
    camera_state = {'laser': {'cloud': {'xyz_m': [[.1, .2, .3]], 'read_started': 99.9}}, 'handeye': {}}
    service.laser_state(camera_state)
    assert inputs.pop() is None  # A recent cloud may have been acquired during motion.
    clock[0] = 100.3
    camera_state['laser']['cloud']['read_started'] = 100.2
    for reason in ('quiet', 'moving', 'disabled', 'cached'):
        service.ctrl.closed = reason == 'disabled'
        service.ctrl.cached_rest_display = reason == 'cached'
        service.ctrl.motion_plan = object() if reason == 'moving' else None
        state = service.laser_state(camera_state)
        assert state['cloud']['points_base_m'] == [[1., 2., 3.]]
    assert np.array_equal(inputs[0], np.eye(4))
    assert inputs[1:] == [None, None, None]
    service.ctrl.cached_rest_display = False
    service.laser_state(camera_state)
    assert inputs[-1] is None  # Returning to quiet starts a new acquisition window.


def test_disk_telemetry_omits_dense_cloud_and_frame_without_mutating_snapshot():
    cloud = {'xyz_m': [[1, 2, 3]], 'points_base_m': [[1, 2, 3]], 'count': 1, 'sequence': 5}
    source = {'cloud': cloud, 'preview_jpeg_base64': 'large', 'quality': {'rms_mm': .2}}
    result = RobotService._laser_telemetry(source)
    assert result == {'cloud': {'count': 1, 'sequence': 5}, 'quality': {'rms_mm': .2}}
    assert 'xyz_m' in source['cloud'] and 'preview_jpeg_base64' in source
    json.dumps(result, allow_nan=False)
