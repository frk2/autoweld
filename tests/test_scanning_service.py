"""Scan JSON integration retains the service's motion, feedback and exit gates."""

import json
import queue
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService, recovery_cancels, recovery_discards
from autoweld.control import Controller
from autoweld.motion import MotionLimits


@pytest.fixture
def service(tmp_path):
    s = object.__new__(RobotService)
    s.runtime = tmp_path
    (tmp_path/'requests').mkdir()
    s.events = queue.Queue()
    s.ctrl = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    s.ctrl.mode = 'Hold'
    s.camera = SimpleNamespace(state=lambda **kwargs: {},
                               dispatch=lambda request: s.calls.append(('camera', request)))
    s.pending_pose, s.request_status = None, None
    s.exit_after_disable, s.running = False, True
    s.calibration = s.laser = None
    s.saved_poses = {'Tube Scanning': {'q': [.1, .2, .3, 0., 0., 0.]}}
    s.calls = []
    s.scan = SimpleNamespace(busy=False, status='idle', return_pose=np.eye(4), _handeye=np.eye(4),
        settings=SimpleNamespace(width_mm=100., height_mm=100.),
        dispatch=lambda request: s.calls.append(('scan', request)) or {'status': 'accepted'},
        cancel=lambda: s.calls.append('cancel'),
        abort_for_shutdown=lambda: s.calls.append('shutdown'),
        abort_for_cooldown=lambda: s.calls.append('cooldown'),
        tick=lambda: s.calls.append('tick'))
    return s


@pytest.mark.parametrize('action', ['scan_start', 'scan_resume', 'scan_state', 'scan_cancel', 'scan_clear', 'scan_export'])
def test_json_routes_to_shared_coordinator_without_robot_commands(service, action):
    request = {'action': action, '_id': 'scan-test'}
    response = service.handle_request(request)
    assert response['ok'] and response['result'] == {'status': 'accepted'}
    assert service.calls == [('scan', request)]
    assert not service.ctrl.robot.commands


def test_scan_start_requires_host_admission_but_cancel_remains_available(service):
    def reject(**kwargs):
        service.calls.append('host-rejected')
        raise RuntimeError('Host memory PSI unsafe')
    service.host_guard = SimpleNamespace(check=reject)
    response = service.handle_request({'action': 'scan_start'})
    assert not response['ok'] and 'Host memory PSI' in response['error']
    assert service.calls == ['host-rejected']
    assert service.handle_request({'action': 'scan_cancel'})['ok']
    assert service.calls[-1] == ('scan', {'action': 'scan_cancel'})


@pytest.mark.parametrize('reason', ['shutdown', 'cooldown', 'disabled', 'moving', 'gravity', 'queued',
                                     'intrinsics', 'laser', 'laser_future', 'host'])
def test_scan_readiness_rejects_other_motion_owners_and_pressure(service, reason):
    if reason == 'shutdown':
        service.ctrl.shutdown_requested = True
    elif reason == 'cooldown':
        service.ctrl.cooldown_active = True
    elif reason == 'disabled':
        service.ctrl.closed = True
    elif reason == 'moving':
        service.ctrl.motion_plan = object()
    elif reason == 'gravity':
        service.ctrl.mode = 'Gravity compensation'
    elif reason == 'queued':
        service.pending_pose = ({'action': 'pose'}, 0.)
    elif reason == 'intrinsics':
        service.calibration = SimpleNamespace(status='capturing', busy=True)
    elif reason == 'laser':
        service.laser = SimpleNamespace(status='capturing', busy=True)
    elif reason == 'laser_future':
        service.laser = SimpleNamespace(status='idle', busy=False, future=object())
    elif reason == 'host':
        service.host_guard = SimpleNamespace(check=lambda: (_ for _ in ()).throw(RuntimeError('Host unsafe')))
    with pytest.raises(RuntimeError):
        service.can_start_scan()
    assert not service.ctrl.robot.commands


def test_readiness_admits_healthy_shared_control_without_requiring_calibration_pose(service):
    assert 'Calibration Pose' not in service.saved_poses
    assert service.can_start_scan() is None
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('age, fresh', [(0., True), (.1, True), (.100001, False), (-.01, False),
                                       (float('inf'), False), (float('nan'), False)])
def test_scan_feedback_uses_actual_sdk_feedback_age(service, monkeypatch, age, fresh):
    service.ctrl.live = True
    service.ctrl.health = lambda: None
    service.ctrl.robot.get_feedback_diagnostics = lambda: {'age_s': age}
    monkeypatch.setattr('autoweld.backend.time.monotonic', lambda: 100.)
    result = service.scan_feedback()
    assert result['fresh'] is fresh
    if np.isfinite(age):
        assert result['measured_at'] == 100.-age
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('reason', ['motion', 'transition', 'disable', 'shutdown', 'cooldown', 'fault',
                                     'speed', 'nan_velocity'])
def test_scan_feedback_rejects_nonstationary_or_pending_control(service, reason):
    ctrl = service.ctrl
    ctrl.health = lambda: None
    if reason == 'motion':
        ctrl.motion_plan = object()
    elif reason == 'transition':
        ctrl.mode_transition = True
    elif reason == 'disable':
        ctrl.disable_pending = True
    elif reason == 'shutdown':
        ctrl.shutdown_requested = True
    elif reason == 'cooldown':
        ctrl.cooldown_active = True
    elif reason == 'fault':
        ctrl.motion_fault = lambda: {'message': 'real fault'}
    else:
        ctrl.robot.velocity = np.zeros(7)
        ctrl.robot.velocity[2] = .050001 if reason == 'speed' else float('nan')
    assert service.scan_feedback()['stationary'] is False
    assert not ctrl.robot.commands


@pytest.mark.parametrize('reason', ['disabled', 'cached'])
def test_disabled_or_cached_startup_is_not_live_scan_feedback(service, reason):
    service.ctrl.health = lambda: None
    service.ctrl.closed = reason == 'disabled'
    service.ctrl.cached_rest_display = reason == 'cached'
    with pytest.raises(RuntimeError, match='Enabled SDK'):
        service.scan_feedback()


def test_shutdown_retains_unconditional_path_after_scan_cleanup_error(service):
    service.scan.busy = True
    service.pending_pose = ({'action': 'pose'}, 0.)
    service.scan.abort_for_shutdown = lambda: (_ for _ in ()).throw(RuntimeError('scan disk failure'))
    response = service.handle_request({'action': 'shutdown'})
    assert response['ok'] and service.ctrl.shutdown_requested and service.exit_after_disable
    assert service.pending_pose is None and not service.ctrl.robot.commands
    response = service.handle_request({'action': 'scan_start'})
    assert not response['ok'] and 'Shutdown owns' in response['error']


def test_cooldown_releases_scan_without_an_extra_hold_and_drains_persistence(service):
    service.ctrl.cooldown_active = True
    service.ctrl.tick = lambda: None
    service.scan.busy = True
    service.tick()
    assert service.calls == ['cooldown', 'tick']
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('command', [
    {'action': 'mode', 'mode': 'Hold'}, {'action': 'joint', 'q': [0., .2, .3, 0., 0., 0.]},
    {'action': 'pose', 'position': [0., 0., 0.], 'rpy': [0., 0., 0.]},
    {'action': 'disable'}, {'action': 'camera_config', 'columns': 5},
    {'action': 'camera_analysis', 'mode': 'laser'},
    {'action': 'camera_stream', 'enabled': False},
])
def test_manual_or_camera_commands_cancel_scan_before_dispatch(service, command):
    service.scan.busy = True
    service.ctrl.hold = lambda **kwargs: service.calls.append('control-hold')
    service.ctrl.command = lambda *args, **kwargs: service.calls.append('control-joint')
    service.ctrl.request_disable = lambda **kwargs: service.calls.append('control-disable')
    service.move_pose = lambda *args: service.calls.append('control-pose')
    assert service.handle_request(command)['ok']
    assert service.calls[0] == 'cancel' and len(service.calls) >= 2


def test_recovery_accounts_for_scanning_and_releases_scan_without_replaying_targets(service):
    service.scan.busy = True
    assert service._recovery_busy()
    service.pending_pose = ({'action': 'pose'}, 0.)
    service.ctrl.motion_plan = object()
    service._release_for_recovery()
    assert service.calls == ['shutdown']
    assert service.pending_pose is None and service.ctrl.motion_plan is None
    assert not service.ctrl.robot.commands
    assert recovery_cancels({'action': 'scan_cancel'})
    assert recovery_discards({'action': 'scan_start'})
    assert recovery_discards({'action': 'scan_clear'})
    assert not recovery_discards({'action': 'scan_state'})
    assert not recovery_discards({'action': 'scan_export'})


def test_saved_interrupted_scan_does_not_claim_robot_motion_ownership(service):
    service.scan.status, service.scan.busy = 'interrupted', False
    assert not service._recovery_busy()


def test_pre_recovery_scan_start_cannot_enable_later(service):
    service._recovery_discard_before = 20.
    service._recovery_discarded = []
    response = service.handle_request({'action': 'scan_start', '_service_queued_monotonic': 19.})
    assert not response['ok'] and 'Discarded pre-recovery work' in response['error']
    assert not service.calls and not service.ctrl.robot.commands


def test_scan_waypoints_use_service_planner_and_configured_default_speeds(service):
    service._calibration_reference = None
    service.planner = SimpleNamespace(limits=MotionLimits())
    target = service.ctrl.positions()
    target[0] += .01
    service.validate_pose = lambda pose: target.copy()
    service.ctrl.command = lambda *args, **kwargs: service.calls.append((args, kwargs))
    requested = np.eye(4)
    requested[0, 3] = .001
    service.move_scan_pose(requested)
    args, kwargs = service.calls[0]
    assert args[1] == 'IK' and kwargs['automatic'] is False
    np.testing.assert_array_equal(args[0], target)
    np.testing.assert_array_equal(kwargs['cartesian'], requested)
    assert service.ctrl.position_speed_limit == 2.
    assert service.planner.limits.linear_velocity == 3. and service.planner.limits.joint_velocity == 2.
    with pytest.raises(ValueError, match='optical XY envelope'):
        service.move_scan_pose(np.array([[1., 0., 0., .081], [0., 1., 0., 0.],
                                        [0., 0., 1., 0.], [0., 0., 0., 1.]]))


def test_periodic_disk_telemetry_omits_dense_scan_cloud_and_browser_keeps_it(service):
    source = {'scan': {'status': 'complete', 'point_count': 1,
                       'cloud': {'points_base_m': [[1., 2., 3.]], 'points_world_m': [[1., 2., 3.]],
                                 'revision': 1, 'scan_id': 'session'}}}
    service.state = lambda: source.copy()
    service.snapshot_lock = Lock()
    service.publish()
    disk = json.loads((service.runtime/'status.json').read_text())
    assert disk['scan']['cloud'] == {'revision': 1, 'scan_id': 'session'}
    assert json.loads(service.snapshot_json)['scan']['cloud']['points_base_m'] == [[1., 2., 3.]]
    assert source['scan']['cloud']['points_base_m'] == [[1., 2., 3.]]


def test_workspace_probe_uses_installed_model_without_robot_targets_even_disabled(service, monkeypatch):
    service.camera.state = lambda **kwargs: {'handeye': {'tcp_from_camera': np.eye(4).tolist()}}
    service.planner = SimpleNamespace(kin=object(), _configuration=lambda q: q[:6].copy())
    service.has_self_collision = lambda q: False
    seed = service.ctrl.positions()[:6].copy()
    calls = []
    def solve(kin, pose, site, initial, **kwargs):
        calls.append(pose.copy())
        return initial
    monkeypatch.setattr('autoweld.backend.solve_pose', solve)
    service.ctrl.disabled_q = service.ctrl.positions()
    service.ctrl.closed = True
    result = service.probe_scan_pose({'baseline_pose': np.eye(4).tolist(),
        'offset_xy_mm': [50., -50.], 'depth_offset_mm': -23., 'optical_roll_deg': 15., 'seed_q': seed.tolist()})
    assert result['read_only'] and len(calls) == 1
    np.testing.assert_allclose(calls[0][:3, 3], [.05, -.05, -.023])
    np.testing.assert_allclose(calls[0][:3, 2], [0., 0., 1.])  # Optical viewing axis stays fixed.
    assert not service.ctrl.robot.commands


def test_workspace_probe_rejects_busy_motion_before_solving(service):
    service.ctrl.motion_plan = object()
    with pytest.raises(RuntimeError, match='Stop active motion'):
        service.probe_scan_pose({})
    assert not service.ctrl.robot.commands
