"""A stopped SDK cannot acknowledge a fault or certify cached hardware state."""

import json
import logging
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService
from autoweld.control import Controller
from autoweld.motion import MotionLimits


@pytest.fixture
def service(tmp_path):
    s = object.__new__(RobotService)
    s.runtime = tmp_path
    robot = FakeRobot()
    s.ctrl = Controller(robot, np.tile([-3., 3.], (6, 1)), live=True)
    robot.motor_chain = SimpleNamespace(running=False,
        _control_thread=SimpleNamespace(is_alive=lambda: False),
        get_control_failure=lambda: {'message': 'Motor error detected: motor id: 2, error: loss communication'})
    robot._server_thread = SimpleNamespace(is_alive=lambda: False)
    robot.get_feedback_diagnostics = lambda: {'timestamp': 100., 'age_s': 1200.}
    robot.acknowledge_motion_fault = lambda: pytest.fail('Dead SDK must not acknowledge or command')
    s.pending_pose = ({'action': 'pose'}, 0.)
    s.message, s.fault = '', None
    s.request_status, s.exit_after_disable = None, False
    s.camera, s.calibration, s.laser = None, None, None
    s.planner = SimpleNamespace(fk=lambda q: np.eye(4), limits=MotionLimits())
    s.saved_poses = {}
    return s


def test_dead_sdk_acknowledgement_is_rejected_without_changing_target(service):
    target = service.ctrl.target.copy()
    response = service.handle_request({'action': 'acknowledge_fault'})
    assert not response['ok'] and 'motor id: 2' in response['error']
    np.testing.assert_array_equal(service.ctrl.target, target)
    assert not service.ctrl.robot.commands


def test_stale_feedback_acknowledgement_is_rejected_before_native_call(service):
    robot = service.ctrl.robot
    robot.motor_chain.get_control_failure = lambda: None
    robot.motor_chain.running = True
    robot.motor_chain._control_thread.is_alive = lambda: True
    robot._server_thread.is_alive = lambda: True
    with pytest.raises(RuntimeError, match='Fresh SDK feedback required'):
        service.ctrl.acknowledge_fault()
    assert not robot.commands


def test_dead_motor_worker_is_detected_even_before_server_stops(service):
    robot = service.ctrl.robot
    robot.motor_chain.get_control_failure = lambda: None
    robot.motor_chain.running = True
    robot._server_thread.is_alive = lambda: True
    robot.get_feedback_diagnostics = lambda: {'timestamp': 100., 'age_s': 0.}
    assert not service.ctrl.sdk_status()['control_alive']
    with pytest.raises(RuntimeError, match='SDK control thread stopped'):
        service.ctrl.health()
    assert not service.ctrl.sdk_status()['feedback_fresh']


def test_snapshot_identifies_cached_codes_and_pose_even_after_sync(service):
    service.ctrl.robot.get_motor_power_state = lambda: {str(i): '0x1' for i in range(1, 7)}
    assert service.handle_request({'action': 'sync'})['ok']
    state = service.state()
    assert state['sdk']['feedback_age_s'] == 1200.
    assert state['sdk']['feedback_timestamp'] == 100.
    assert not state['sdk']['feedback_fresh']
    assert state['fault'] and 'unverified' in state['fault']
    assert 'cached SDK feedback' in state['feedback_source']
    assert state['motor_errors']['2'] == '0x1'  # explicitly cached, never current proof
    json.dumps(state, allow_nan=False)


def test_failure_snapshot_and_traceback_are_not_repeated_every_tick(service, caplog, monkeypatch):
    now = [100.]
    monkeypatch.setattr('autoweld.backend.time.monotonic', lambda: now[0])
    service.ctrl.motion_plan = object()
    service.ctrl.rest_verified = True
    with caplog.at_level(logging.ERROR):
        for _ in range(50):
            try:
                service.ctrl.health()
            except RuntimeError as error:
                service.report_tick_failure(error)
        first = (service.runtime / 'sdk-failure.json').read_text()
        now[0] += 5.
        try:
            service.ctrl.health()
        except RuntimeError as error:
            service.report_tick_failure(error)
    assert len(caplog.records) == 2
    assert caplog.records[0].exc_info and not caplog.records[1].exc_info
    assert 'power and physical pose are unverified' in caplog.records[0].message
    assert 'active hold' not in caplog.text
    assert (service.runtime / 'sdk-failure.json').read_text() == first
    assert len(list((service.runtime / 'evidence').glob('sdk-failure-*.json'))) == 1
    assert service.pending_pose is None and service.ctrl.motion_plan is None
    assert not service.ctrl.rest_verified and not service.ctrl.closed
    assert not service.ctrl.robot.commands and not service.ctrl.robot.closed


def test_shutdown_remains_accepted_with_specific_retryable_sdk_blocker(service):
    service.ctrl._ensure_position_speed = lambda: pytest.fail('No native mode change with dead SDK')
    result = service.handle_request({'action': 'shutdown'})
    assert result['ok'] and service.exit_after_disable
    service.ctrl.tick()
    state = service.ctrl.shutdown_status
    assert state['retryable'] and 'motor id: 2' in state['error']
    assert not service.ctrl.closed and not service.ctrl.robot.closed
    assert service.handle_request({'action': 'shutdown'})['ok']


def test_publish_archives_sdk_failure_when_shutdown_handles_tick_exception(service):
    service.snapshot_lock = threading.Lock()
    service.handle_request({'action': 'shutdown'})
    service.ctrl.tick()  # A blocked shutdown returns instead of raising.
    service.publish()
    recorded = json.loads((service.runtime / 'sdk-failure.json').read_text())
    assert 'motor id: 2' in recorded['sdk']['failure']['message']
    state = json.loads((service.runtime / 'status.json').read_text())
    assert state['fault'] and state['shutdown']['retryable']


@pytest.mark.parametrize('age', [float('nan'), float('inf'), -1.])
def test_invalid_feedback_age_is_never_reported_fresh(service, age):
    robot = service.ctrl.robot
    robot.motor_chain.get_control_failure = lambda: None
    robot.motor_chain.running = True
    robot.motor_chain._control_thread.is_alive = lambda: True
    robot._server_thread.is_alive = lambda: True
    robot.get_feedback_diagnostics = lambda: {'timestamp': float('nan'), 'age_s': age}
    status = service.ctrl.sdk_status()
    assert not status['feedback_fresh']
    json.dumps(status, allow_nan=False)


class UnsafeHost:
    def sample(self):
        return {'safe': False, 'reasons': ['I/O pressure above limit']}

    def check(self, **kwargs):
        raise RuntimeError('Unsafe host: I/O pressure above limit')


@pytest.mark.parametrize('payload', [
    {'action': 'enable'}, {'action': 'joint', 'q': [0.]*6},
    {'action': 'pose', 'position': [0., 0., 0.]},
    {'action': 'calibration_auto'}, {'action': 'camera_stream', 'enabled': True},
    {'action': 'mode', 'mode': 'Gravity compensation'},
])
def test_unsafe_host_rejects_new_work_before_sdk_or_queue(service, payload):
    service.host_guard = UnsafeHost()
    service.pending_pose = None
    result = service.handle_request(payload)
    assert not result['ok'] and 'I/O pressure' in result['error']
    assert service.pending_pose is None and not service.ctrl.robot.commands


def test_pressure_monitor_uses_dedicated_shutdown_once_and_keeps_hold_available(service):
    from test_shutdown import Robot
    service.ctrl = Controller(Robot(), np.tile([-3., 3.], (6, 1)), live=True)
    service.host_guard = UnsafeHost()
    service.events = queue.Queue()
    service.running = True
    (service.runtime / 'requests').mkdir()
    service.tick()
    assert service.exit_after_disable and service.ctrl.shutdown_requested
    assert service.pending_pose is None
    command_count = len(service.ctrl.robot.commands)
    service.tick()
    assert len(service.ctrl.robot.commands) == command_count
    assert service.handle_request({'action': 'mode', 'mode': 'Hold'})['ok']
    assert service.ctrl.shutdown_status['retryable']
    assert service.handle_request({'action': 'shutdown'})['ok']
    assert service.ctrl.robot.close_count == 0
