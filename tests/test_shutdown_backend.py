"""Shutdown keeps motion ownership across JSON, calibration and service teardown."""

import json
import queue
from types import SimpleNamespace

import numpy as np
import pytest
from test_shutdown import Robot

from autoweld.backend import RobotService
from autoweld.calibration import CalibrationCoordinator
from autoweld.control import Controller


@pytest.fixture
def service(tmp_path):
    s = object.__new__(RobotService)
    s.runtime = tmp_path
    (tmp_path / 'requests').mkdir()
    s.events = queue.Queue()
    s.ctrl = Controller(Robot(), np.tile([-3., 3.], (6, 1)), live=True)
    s.pending_pose = ({'action': 'pose'}, 0.)
    s.request_status = None
    s.exit_after_disable = False
    s.calibration = None
    s.camera = None
    s.running = True
    s.published = []
    s.publish = lambda: s.published.append(s.ctrl.shutdown_status)
    return s


def test_filesystem_shutdown_preempts_stale_latch_and_calibration_failure(service):
    service.ctrl.robot.q[0] += 1.8
    service.ctrl.disable_pending = True
    service.ctrl.return_automatic = True
    service.ctrl.motion_plan = object()
    calls = []

    def failed_cleanup():
        calls.append('cancel')
        raise RuntimeError('Camera adapter failed')

    service.calibration = SimpleNamespace(abort_for_shutdown=failed_cleanup,
        tick=lambda: pytest.fail('Calibration must not reclaim shutdown motion'))
    request = service.runtime / 'requests' / 'shutdown-test.json'
    request.write_text(json.dumps({'action': 'shutdown'}))
    service.tick()
    response = json.loads((service.runtime / 'responses' / 'shutdown-test.json').read_text())
    assert response['ok'] and response['result']['requested']
    assert calls == ['cancel']
    assert service.pending_pose is None and service.exit_after_disable
    assert not service.ctrl.disable_pending and not service.ctrl.return_automatic
    assert service.ctrl.shutdown_phase == 'waiting'
    np.testing.assert_array_equal(service.ctrl.target[:6], service.ctrl.robot.q)
    assert not service.ctrl.closed and service.ctrl.robot.close_count == 0


@pytest.mark.parametrize('command', [
    {'action': 'pose', 'position': [0, 0, 0], 'rpy': [0, 0, 0]},
    {'action': 'joint', 'q': [0]*6},
    {'action': 'enable'}, {'action': 'rest'},
    {'action': 'mode', 'mode': 'Gravity compensation'},
    {'action': 'calibration_auto'}, {'action': 'acknowledge_fault'},
    {'action': 'gravity_payload'}, {'action': 'speed_limits'}, {'action': 'motion_limits'},
])
def test_shutdown_rejects_competing_commands_without_queueing(service, command):
    service.handle_request({'action': 'shutdown'})
    result = service.handle_request(command)
    assert not result['ok'] and 'Shutdown owns' in result['error']
    assert service.pending_pose is None
    assert service.ctrl.shutdown_requested and not service.ctrl.robot.commands


def test_hold_pauses_shutdown_then_explicit_json_retry_resumes(service):
    service.handle_request({'action': 'shutdown'})
    service.ctrl.tick()
    result = service.handle_request({'action': 'mode', 'mode': 'Hold'})
    assert result['ok'] and result['result']['retryable']
    assert service.ctrl.shutdown_phase == 'blocked'
    assert not service.ctrl.closed
    service.ctrl.tick()
    assert service.ctrl.shutdown_phase == 'blocked'
    result = service.handle_request({'action': 'shutdown', '_id': 'retry'})
    assert result['ok'] and service.ctrl.shutdown_phase == 'stopping'
    assert service.exit_after_disable


def test_disabled_shutdown_publishes_completion_and_exits_without_reconnecting(service):
    service.ctrl.closed = True
    service.ctrl.disabled_q = service.ctrl.rest.copy()
    service.events.put({'action': 'shutdown'})  # Same queue used by process signals.
    service.tick()
    assert not service.running and service.exit_after_disable
    assert service.published[-1]['phase'] == 'disabled'
    assert not service.ctrl.robot.commands


@pytest.mark.parametrize('reason', [
    'MemAvailable 800 MiB is below 1024 MiB reserve',
    'Memory PSI full avg10 1.41% reaches the 1% limit',
])
def test_host_guard_disables_at_verified_rest_without_exiting_and_requires_new_enable(service, monkeypatch, reason):
    clock = [100.]
    monkeypatch.setattr('autoweld.control.time.monotonic', lambda: clock[0])
    healthy = [False]
    checks = []

    def check(**kwargs):
        checks.append(kwargs)
        if not healthy[0]:
            raise RuntimeError(reason)

    service.host_guard = SimpleNamespace(sample=lambda: {'safe': healthy[0], 'reasons': [reason]}, check=check)
    owner = service.ctrl.robot
    service.tick()
    assert service.running and service.exit_after_disable and not service.ctrl.closed
    assert service.pending_pose is None and service.ctrl.shutdown_requested
    assert owner.close_count == 0
    service.tick()
    service.tick()
    assert not service.ctrl.closed  # Quiet rest must be sustained for a full second.
    clock[0] += 1.01
    service.tick()
    assert service.ctrl.closed and service.ctrl.rest_verified and owner.close_count == 1
    assert service.running and not service.exit_after_disable and not service.ctrl.shutdown_requested
    assert service.published[-1]['phase'] == 'disabled'
    assert 'service remains available' in service.message
    result = service.handle_request({'action': 'enable'})
    assert not result['ok'] and reason in result['error']
    assert checks == [{'force': True}] and owner.close_count == 1
    healthy[0] = True
    service.tick()
    assert service.running and service.ctrl.closed  # Recovery never automatically enables motors.
    enabled = []

    def enable():
        service.ctrl._guard_shutdown()  # A completed protective stop must release this latch too.
        enabled.append(True)

    service.ctrl.enable = enable
    assert service.handle_request({'action': 'enable'})['ok']
    assert enabled == [True] and checks[-1] == {'force': True}


def test_explicit_shutdown_during_host_guard_stop_keeps_progress_and_exits(service):
    service.host_guard = SimpleNamespace(sample=lambda: {'safe': False, 'reasons': ['Memory PSI']})
    service.tick()
    phase_started = service.ctrl._shutdown_phase_started
    result = service.handle_request({'action': 'shutdown'})
    assert result['ok'] and service.ctrl._shutdown_phase_started == phase_started
    assert not service._keep_service_after_disable
    service.ctrl.closed = True  # The controller's verified disable completion.
    service.tick()
    assert not service.running


def test_hold_pauses_host_guard_stop_without_automatic_retry_and_disable_keeps_server(service):
    service.host_guard = SimpleNamespace(sample=lambda: {'safe': False, 'reasons': ['Memory PSI']})
    service.tick()
    assert service.handle_request({'action': 'mode', 'mode': 'Hold'})['ok']
    assert service.ctrl.shutdown_phase == 'blocked'
    service.tick()
    assert service.ctrl.shutdown_phase == 'blocked' and service.running
    assert service.handle_request({'action': 'disable'})['ok']
    assert service.ctrl.shutdown_phase == 'stopping' and service._keep_service_after_disable
    service.ctrl.closed = True  # The controller's verified disable completion.
    service.tick()
    assert service.running and not service.exit_after_disable


def test_calibration_shutdown_abort_releases_hunt_without_calling_hold():
    coordinator = object.__new__(CalibrationCoordinator)
    coordinator._hunter = object()
    coordinator._hunt_moving = True
    coordinator.hunt = {'motion_active': True}
    coordinator.hold = lambda: pytest.fail('Calibration cannot replace the shutdown hold')
    coordinator.state = lambda: {'status': coordinator.status}
    assert coordinator.abort_for_shutdown()['status'] == 'cancelled'
    assert coordinator.cancelled
    assert coordinator._hunter is None and not coordinator._hunt_moving
    assert not coordinator.hunt['motion_active']
