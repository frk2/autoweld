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
