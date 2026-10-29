"""The seam panel cannot acquire a second robot owner or bypass motion gates."""

import queue
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService, recovery_cancels, recovery_discards
from autoweld.camera import CameraService
from autoweld.control import Controller


@pytest.fixture
def service(tmp_path):
    s = object.__new__(RobotService)
    s.runtime = tmp_path
    (tmp_path/'requests').mkdir()
    s.events = queue.Queue()
    s.ctrl = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    s.ctrl.mode = 'Hold'
    s.camera = SimpleNamespace(state=lambda **kwargs: {})
    s.pending_pose = s.request_status = None
    s.exit_after_disable, s.running = False, True
    s.calibration = s.laser = s.scan = None
    s.saved_poses = {'Weld Finder': {'q': [0.]*6}}
    s.calls = []
    s.weld = SimpleNamespace(busy=False, status='idle',
        dispatch=lambda request: s.calls.append(('weld', request)) or {'status': 'accepted'},
        cancel=lambda: s.calls.append('cancel'),
        abort_for_shutdown=lambda: s.calls.append('shutdown'),
        abort_for_cooldown=lambda: s.calls.append('cooldown'),
        tick=lambda: s.calls.append('tick'))
    return s


@pytest.mark.parametrize('action', ['weld_start', 'weld_replay', 'weld_state', 'weld_cancel',
                                     'weld_export', 'weld_clear'])
def test_shared_json_dispatch(service, action):
    request = {'action': action}
    result = service.handle_request(request)
    assert result['ok'] and service.calls == [('weld', request)]
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('reason', ['scan', 'disabled', 'moving', 'cooldown', 'pending', 'missing_pose'])
def test_start_preserves_physical_and_ownership_gates(service, reason):
    if reason == 'scan':
        service.scan = SimpleNamespace(busy=True)
    elif reason == 'disabled':
        service.ctrl.closed = True
    elif reason == 'moving':
        service.ctrl.motion_plan = object()
    elif reason == 'cooldown':
        service.ctrl.cooldown_active = True
    elif reason == 'pending':
        service.pending_pose = ({'action': 'pose'}, 0.)
    else:
        service.saved_poses.clear()
    with pytest.raises(RuntimeError):
        service.can_start_weld()
    assert not service.ctrl.robot.commands


def test_weld_owner_cannot_be_preempted_by_new_scan(service):
    service.weld.busy = True
    with pytest.raises(RuntimeError, match='Weld seam'):
        service.can_start_scan()
    # Its own readiness check can use the underlying feedback gates.
    service.can_start_weld()
    service.scan = SimpleNamespace(busy=False, dispatch=lambda request: pytest.fail('No scan dispatch'))
    result = service.handle_request({'action': 'scan_start'})
    assert not result['ok'] and 'Weld seam' in result['error']
    assert service.calls == []


@pytest.mark.parametrize('action', ['weld_start', 'weld_replay'])
def test_host_admission_and_unconditional_cancel(service, action):
    def reject(**kwargs):
        raise RuntimeError('Host memory unsafe')
    service.host_guard = SimpleNamespace(check=reject)
    assert not service.handle_request({'action': action})['ok']
    assert service.handle_request({'action': 'weld_cancel'})['ok']
    assert service.calls == [('weld', {'action': 'weld_cancel'})]


def test_operator_hold_cancels_before_new_target(service):
    service.weld.busy = True
    result = service.handle_request({'action': 'mode', 'mode': 'Hold'})
    assert result['ok'] and service.calls == ['cancel']
    assert len(service.ctrl.robot.commands) == 1


def test_shutdown_and_cooldown_do_not_leave_scan_motion_owner(service):
    service.weld.busy = True
    assert service.handle_request({'action': 'shutdown'})['ok']
    assert service.calls == ['shutdown'] and service.ctrl.shutdown_requested
    assert not service.handle_request({'action': 'weld_replay'})['ok']
    service.calls.clear()
    service.ctrl.cooldown_active = True
    service.ctrl.tick = lambda: None
    service.tick()
    assert service.calls == ['cooldown', 'tick']


def test_shutdown_cleanup_failure_still_requests_controller_shutdown(service):
    def fail():
        raise RuntimeError('Persistence failed')
    service.weld.abort_for_shutdown = fail
    assert service.handle_request({'action': 'shutdown'})['ok']
    assert service.ctrl.shutdown_requested and service.exit_after_disable


@pytest.mark.parametrize('action', ['weld_start', 'weld_replay', 'weld_clear'])
def test_pre_recovery_weld_intent_is_discarded(service, action):
    assert recovery_discards({'action': action})
    service._recovery_discard_before = 100.
    service._recovery_discarded = []
    result = service.handle_request({'action': action, '_service_queued_monotonic': 90.})
    assert not result['ok'] and 'pre-recovery' in result['error']
    assert service.calls == []


def test_weld_cancel_remains_recovery_cancellation():
    assert recovery_cancels({'action': 'weld_cancel'})
    assert not recovery_discards({'action': 'weld_cancel'})
    assert not recovery_discards({'action': 'weld_state'})


@pytest.mark.parametrize('status', ['insufficient_length', 'incomplete'])
def test_short_completed_measurement_does_not_keep_recovery_busy(service, status):
    service.weld.status = status
    assert not service._recovery_busy()


def test_camera_bridge_calls_pure_sdk_without_camera_owner(monkeypatch):
    native = {'units': 'mm', 'coordinate_frame': 'camera', 'xyz_mm': [[0., 0., 150.]]}
    calls = []
    api = SimpleNamespace(find_weld_seam=lambda cloud, **kwargs:
        calls.append((cloud, kwargs)) or {'found': False, 'status': 'no_seam'})
    monkeypatch.setattr('autoweld.camera._weldfinder_module', lambda name: api if name == 'seam' else None)
    result = CameraService.find_weld_seam(native, previous_point_mm=[0., 0., 150.])
    assert result['status'] == 'no_seam'
    assert calls == [(native, {'settings': None, 'previous_point_mm': [0., 0., 150.]})]
