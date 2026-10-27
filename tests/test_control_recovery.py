"""Disabled recovery admission with fake SDK readers, clocks and owners only."""
import copy
import json
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.backend import JsonConnection, RobotService, stamp_admission
from autoweld.recovery import ControlRecovery, validate_disabled_sample


class Clock:
    def __init__(self):
        self.now = 100.
    def monotonic(self):
        return self.now
    def wall(self):
        return 1000.+self.now


def sample(clock, **changes):
    stamp = clock.wall()
    q = [0., 0., 0., 0., 0., -3.]
    def native(register, value):
        return [{'motor_id': mid, 'register': register, 'value': value if register != 'p_m' else q[mid-1],
                 'timestamp': stamp} for mid in range(1, 7)]
    result = {'motor_ids': list(range(1, 7)), 'joint_rad': q, 'sdk_feedback_position_rad': list(q),
        'velocity_rad_s': [0.] * 6, 'status': ['0x0', '0xd', '0x0', '0x0', '0x0', '0x0'],
        'physical_timestamp': stamp, 'timestamp': stamp,
        'mapping': {'offsets_rad': [0.] * 6, 'directions': [1] * 6, 'boot_normalization': 'none'},
        'temperatures': {'mos_c': [30.] * 6, 'rotor_c': [31.] * 6},
        'native_p_m': native('p_m', 0), 'native_timeout': native('TIMEOUT', 80000),
        'native_ctrl_mode': native('CTRL_MODE', 2)}
    result.update(changes)
    return result


class Guard:
    def __init__(self):
        self.calls = 0
        self.fail_at = None
    def check(self, *, force):
        assert force
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError('Host I/O PSI unsafe')
        return {'safe': True}


class Controller:
    def __init__(self, clock, events):
        self.clock, self.events = clock, events
        self.robot = SimpleNamespace(get_robot_info=lambda: {'sim': True})
        self.rest = np.array([0., 0., 0., 0., 0., -3., 0.])
        self.limits = np.array([[-2., 2.]] * 5+[[-5., -2.1]])
        self.closed, self.cached_rest_display = True, True
        self.mode, self.motion_plan, self.disable_pending = 'Disabled', None, False
        self.shutdown_requested, self.cooldown_active = False, False
        self.enabled = 0
        self.gripper_target = 0.
        self.robot_factory = self.factory
        self.healthy = False
    def sdk_status(self):
        return {'control_alive': self.healthy, 'feedback_fresh': self.healthy}
    def motion_fault(self):
        return None
    def factory(self, *args, **kwargs):
        self.events.append('factory')
        assert kwargs['recovery_near_rest_tolerance_rad'] == .15
        assert kwargs['enable_deadline'] == 110.
        kwargs['admission_guard']()
        robot = SimpleNamespace(name='new SDK owner')
        kwargs['on_created'](robot)
        return robot
    def enable(self):
        self.robot = self.robot_factory()
        self.events.append('enable')
        self.enabled += 1
        self.healthy, self.closed, self.mode = True, False, 'Hold'


class Reader:
    def __init__(self, clock, events):
        self.clock, self.events = clock, events
        self.change = None
        self.after_guard = None
    def survey(self):
        self.events.append('survey')
        value = sample(self.clock)
        if self.change:
            self.change(value)
        return value
    def clear_timeout_latches(self, **kwargs):
        self.events.append('quiet')
        before = self.survey()
        self.clock.now += 1.1
        kwargs['before_write']()
        self.events.append('clear disabled 0xd only')
        if self.after_guard:
            self.after_guard()
        after = sample(self.clock, status=['0x0'] * 6)
        return {'before': before, 'after': after, 'timeout_ms': 4000, 'timeout_register_ticks': 80000,
                'cleared_motor_ids': [2], 'enabled': False, 'flash_written': False}
    def close(self):
        self.events.append('reader_closed')


@pytest.fixture
def setup():
    clock, events = Clock(), []
    ctrl, guard = Controller(clock, events), Guard()
    reader = Reader(clock, events)
    def release():
        events.append('ownership_released')
        ctrl.shutdown_requested = ctrl.disable_pending = ctrl.cooldown_active = False
    recovery = ControlRecovery(ctrl, lambda: reader, guard, release,
        clock=clock.monotonic, wall_clock=clock.wall)
    return recovery, ctrl, guard, reader, clock, events


def test_full_disabled_proof_precedes_cleanup_enable_and_returns_watchdogs(setup):
    recovery, ctrl, guard, _, _, events = setup
    rest = ctrl.rest.copy()
    ctrl.shutdown_requested = ctrl.disable_pending = True
    result = recovery.recover(100.)
    assert result['phase'] == 'holding' and result['changed']
    assert ctrl.enabled == 1 and guard.calls == 0
    assert events.index('clear disabled 0xd only') < events.index('ownership_released')
    assert events.index('reader_closed') < events.index('factory') < events.index('enable')
    assert [row['value'] for row in result['timeout_readbacks']] == [80000] * 6
    np.testing.assert_array_equal(ctrl.rest, rest)
    assert not ctrl.shutdown_requested and not ctrl.disable_pending
    before = list(events)
    assert recovery.recover(clock_value := 101.1)['phase'] == 'already_controlled'
    assert clock_value < 110 and ctrl.enabled == 1 and events == before


@pytest.mark.parametrize('field,value,reason', [
    ('status', ['0x1'] * 6, 'disabled status'), ('status', ['0x8'] * 6, 'disabled status'),
    ('joint_rad', [.2, 0, 0, 0, 0, -3], 'fixed raw'),
    ('velocity_rad_s', [.051, 0, 0, 0, 0, 0], 'not quiet'),
    ('physical_timestamp', 1099., 'fresh physical'),
    ('motor_ids', [1, 2, 3, 4, 5, 7], 'exactly motor IDs'),
])
def test_bad_physical_feedback_never_clears_or_enables(setup, field, value, reason):
    recovery, ctrl, _, reader, _, events = setup
    reader.change = lambda row: row.update({field: value})
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and reason in result['error']
    assert ctrl.enabled == 0 and 'quiet' not in events and 'ownership_released' not in events


@pytest.mark.parametrize('modify,reason', [
    (lambda s: s['native_p_m'][5].update(value=-2.9), 'p_m disagree'),
    (lambda s: s['mapping']['offsets_rad'].__setitem__(5, 6.28), 'fixed zero-offset'),
    (lambda s: s['temperatures']['rotor_c'].__setitem__(3, 60.), 'below 60'),
    (lambda s: s['native_ctrl_mode'][4].update(value=4), 'modes'),
])
def test_native_encoder_mode_and_thermal_proof_is_mandatory(setup, modify, reason):
    recovery, ctrl, _, reader, _, events = setup
    reader.change = modify
    result = recovery.recover(100.)
    assert reason in result['error'] and ctrl.enabled == 0
    assert 'clear disabled 0xd only' not in events


def test_fixed_encoder_j6_bounds_and_near_rest_are_not_expanded(setup):
    recovery, ctrl, _, reader, _, _ = setup
    def outside(row):
        row['joint_rad'][5] = row['sdk_feedback_position_rad'][5] = -2.
        row['native_p_m'][5]['value'] = -2.
    reader.change = outside
    assert 'strict installed' in recovery.recover(100.)['error']
    ctrl.limits[5] = [-5., -1.]
    assert 'farther than 0.15' in recovery.recover(100.)['error']
    assert ctrl.enabled == 0


@pytest.mark.parametrize('temperature,allowed', [(54.9, True), (55., False), (59., False)])
def test_cooldown_retains_below_55_resume_hysteresis(setup, temperature, allowed):
    recovery, ctrl, _, reader, _, _ = setup
    ctrl.cooldown_active = True
    reader.change = lambda row: row['temperatures'].update(mos_c=[temperature] * 6)
    result = recovery.recover(100.)
    assert (result['phase'] == 'holding') is allowed
    assert ctrl.cooldown_active is not allowed


def test_explicit_recovery_bypasses_unsafe_host_pressure(setup):
    recovery, ctrl, guard, _, _, events = setup
    guard.fail_at = 1
    result = recovery.recover(100.)
    assert result['phase'] == 'holding' and result['host_pressure_bypassed']
    assert ctrl.enabled == 1 and guard.calls == 0
    assert 'clear disabled 0xd only' in events


def test_unsafe_host_exception_does_not_bypass_disabled_physical_proof(setup):
    recovery, ctrl, guard, reader, _, events = setup
    guard.fail_at = 1
    reader.change = lambda row: row.update(status=['0x1'] * 6)
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'disabled status' in result['error']
    assert ctrl.enabled == 0 and guard.calls == 0 and 'quiet' not in events


def test_expired_and_cancelled_intents_never_enable_or_retry_in_background(setup):
    recovery, ctrl, _, reader, clock, events = setup
    assert 'expired' in recovery.recover(89.)['error'] and not events
    recovery.cancelled = lambda intent: True
    assert 'cancelled' in recovery.recover(100.)['error'] and not events
    recovery.cancelled = lambda intent: False
    reader.after_guard = lambda: setattr(clock, 'now', 111.)
    assert 'expired' in recovery.recover(100.)['error']
    assert ctrl.enabled == 0 and 'ownership_released' not in events
    clock.now = 200.
    assert ctrl.enabled == 0  # There is no worker or deferred retry.


def test_failed_owner_writer_must_be_stopped_before_new_reader(setup):
    recovery, ctrl, _, _, _, events = setup
    alive = SimpleNamespace(is_alive=lambda: True)
    ctrl.robot = SimpleNamespace(motor_chain=SimpleNamespace(running=True, _control_thread=alive),
        _server_thread=alive, stop_for_disabled_recovery=lambda **kwargs: None)
    result = recovery.recover(100.)
    assert 'must be stopped' in result['error']
    assert not events and ctrl.enabled == 0


def test_busy_healthy_control_is_not_interrupted_or_retarged(setup):
    recovery, ctrl, _, _, _, events = setup
    ctrl.healthy, ctrl.mode, ctrl.motion_plan = True, 'IK', object()
    assert 'busy' in recovery.recover(100.)['error']
    assert ctrl.enabled == 0 and not events


def test_constructor_failure_retains_partial_owner_and_blocks_new_attempt(setup):
    recovery, ctrl, _, _, _, events = setup
    owner = object()
    def fail(*args, **kwargs):
        error = RuntimeError('Partial native startup failed')
        error.recovery_chain = owner
        error.recovery_cleanup_verified = False
        raise error
    ctrl.robot_factory = fail
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and result['partial_owner_retained']
    assert recovery.retained_chain is owner and not ctrl.closed
    events.clear()
    assert 'writer state is unknown' in recovery.recover(101.1)['error']
    assert not events


def test_post_enable_stopped_sdk_is_not_reported_as_success(setup):
    recovery, ctrl, _, _, _, _ = setup
    enable = ctrl.enable
    def fail_after():
        enable()
        ctrl.healthy = False
    ctrl.enable = fail_after
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'fresh, live' in result['error']
    assert ctrl.enabled == 1 and not ctrl.closed


def test_json_admission_replaces_forged_internal_timestamp_and_honors_public_age(monkeypatch):
    import autoweld.backend as backend
    monkeypatch.setattr(backend.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(backend.time, 'time', lambda: 1000.)
    service = SimpleNamespace(events=queue.Queue())
    JsonConnection(service).send(json.dumps({'action': 'recover_control', 'requested_at': 980.,
        '_recovery_requested_monotonic': 999999.}))
    assert service.events.get_nowait()['_recovery_requested_monotonic'] == 80.
    req = {'action': 'recover_control', 'requested_at': 999.}
    stamp_admission(req, file_mtime=980.)
    assert req['_recovery_requested_monotonic'] == 80.
    stamp_admission(req := {'action': 'recover_control', 'requested_at': 1001.})
    assert req['_recovery_requested_monotonic'] < 90.


def test_service_shutdown_cancellation_inspects_files_without_consuming(tmp_path):
    service = object.__new__(RobotService)
    service.runtime, service.events = tmp_path, queue.Queue()
    (tmp_path/'requests').mkdir()
    path = tmp_path/'requests'/'stop.json'
    path.write_text('{"action":"shutdown"}')
    assert service._recovery_cancelled(0.)
    assert path.exists()


def test_already_drained_old_motion_is_rejected_after_recovery_cutoff(tmp_path):
    service = object.__new__(RobotService)
    service.runtime = tmp_path
    service.pending_pose = None
    service._recovery_discard_before = 100.
    service._recovery_discarded = []
    sent = []
    service.dispatch = sent.append
    result = service.handle_request({'action': 'joint', 'q': [0.] * 6, '_service_queued_monotonic': 99.})
    assert not result['ok'] and not sent and 'Discarded pre-recovery' in result['error']
    result = service.handle_request({'action': 'shutdown', '_service_queued_monotonic': 99.})
    assert result['ok'] and sent[-1]['action'] == 'shutdown'
    assert service._recovery_discarded[0]['action'] == 'joint'


def test_validator_does_not_accept_nonfinite_feedback(setup):
    _, ctrl, _, _, clock, _ = setup
    data = sample(clock)
    data['velocity_rad_s'][2] = float('nan')
    with pytest.raises(RuntimeError, match='finite six-axis'):
        validate_disabled_sample(copy.deepcopy(data), ctrl.rest[:6], ctrl.limits, now=clock.wall())


def service_fixture(tmp_path, setup):
    recovery, ctrl, _, _, _, _ = setup
    service = object.__new__(RobotService)
    service.runtime = tmp_path
    for directory in ('requests', 'responses'):
        (tmp_path/directory).mkdir()
    service.ctrl, service.recovery = ctrl, recovery
    service.events = queue.Queue()
    service.pending_pose = None
    service.calibration = service.laser = service.host_guard = None
    service.exit_after_disable = False
    service._recovery_discarded = []
    service._recovery_discard_before = service._recovery_cancel_at = None
    recovery.clear_ownership = service._release_for_recovery
    recovery.cancelled = service._recovery_cancelled
    ctrl.tick = lambda: None
    return service


def test_actual_tick_drained_batch_never_replays_old_pose_after_recovery(tmp_path, setup):
    service = service_fixture(tmp_path, setup)
    service.events.put({'action': 'recover_control', '_id': 'recover', '_recovery_requested_monotonic': 100.})
    service.events.put({'action': 'joint', '_id': 'old-joint', 'q': [0.] * 6, '_service_queued_monotonic': 99.})
    service.tick()
    accepted = json.loads((tmp_path/'responses/recover.json').read_text())
    rejected = json.loads((tmp_path/'responses/old-joint.json').read_text())
    assert accepted['ok'] and accepted['result']['phase'] == 'holding'
    assert not rejected['ok'] and 'Discarded pre-recovery' in rejected['error']
    assert service.ctrl.enabled == 1 and service.ctrl.mode == 'Hold'


def test_old_filesystem_intent_is_not_renewed_when_tick_finally_reads_it(tmp_path, setup, monkeypatch):
    import os

    import autoweld.backend as backend
    service = service_fixture(tmp_path, setup)
    _, ctrl, _, _, clock, events = setup
    monkeypatch.setattr(backend.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(backend.time, 'time', clock.wall)
    path = tmp_path/'requests'/'aged.json'
    path.write_text(json.dumps({'action': 'recover_control', 'requested_at': clock.wall()-1}))
    os.utime(path, (clock.wall()-20, clock.wall()-20))
    service.tick()
    result = json.loads((tmp_path/'responses/aged.json').read_text())
    assert not result['ok'] and 'expired' in result['error']
    assert not events and ctrl.enabled == 0


def test_recovery_post_enable_expiry_keeps_new_owner_but_reports_blocker(setup):
    recovery, ctrl, _, _, clock, _ = setup
    enable = ctrl.enable
    def delayed_enable():
        enable()
        clock.now = 111.
    ctrl.enable = delayed_enable
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'expired' in result['error']
    assert ctrl.enabled == 1 and not ctrl.closed


def test_factory_guard_observes_late_shutdown_before_any_new_enable(setup):
    recovery, ctrl, _, _, _, _ = setup
    cancelled = [False]
    recovery.cancelled = lambda intent: cancelled[0]
    def factory(*args, **kwargs):
        cancelled[0] = True
        kwargs['admission_guard']()
        raise AssertionError('No enable may follow cancellation')
    ctrl.robot_factory = factory
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'cancelled' in result['error']
    assert ctrl.enabled == 0 and ctrl.closed


def test_signal_enqueue_cannot_deadlock_on_normal_queue_mutex():
    import threading
    service = object.__new__(RobotService)
    service.events = queue.Queue()
    service._urgent_events = queue.SimpleQueue()
    finished = threading.Event()
    def interrupt():
        service.enqueue_signal({'action': 'shutdown'})
        finished.set()
    thread = threading.Thread(target=interrupt, daemon=True)
    with service.events.mutex:
        thread.start()
        queued_while_locked = finished.wait(timeout=1.)
    thread.join(timeout=1.)
    assert queued_while_locked and not thread.is_alive()
    request = service._urgent_events.get_nowait()
    assert request['action'] == 'shutdown'
    assert service._recovery_cancel_at == request['_service_queued_monotonic']


def test_signal_shutdown_is_processed_before_older_normal_work(tmp_path, setup):
    service = service_fixture(tmp_path, setup)
    service._urgent_events = queue.SimpleQueue()
    service.events.put({'action': 'joint', 'q': [0.] * 6})
    service.enqueue_signal({'action': 'shutdown'})
    observed = []
    service.handle_request = lambda request: observed.append(request['action'])
    service.tick()
    assert observed == ['shutdown', 'joint']


class UnsafeHost:
    def __init__(self):
        self.checks = 0
    def sample(self):
        return {'safe': False, 'reasons': ['I/O PSI 95%']}
    def check(self, **kwargs):
        self.checks += 1
        raise RuntimeError('Host pressure blocks ordinary work')


def test_recovered_owner_host_monitor_warns_but_ordinary_motion_and_enable_stay_guarded(tmp_path, setup):
    service = service_fixture(tmp_path, setup)
    result = service.recover_control({'action': 'recover_control', '_recovery_requested_monotonic': 100.})
    assert result['phase'] == 'holding' and result['recovery_host_override']['active']
    host = UnsafeHost()
    service.host_guard = host
    shutdowns = []
    service.handle_request = lambda req: shutdowns.append(req)
    service.tick()
    assert not shutdowns
    assert service.recovery_host_override_status()['host_warning'] == 'I/O PSI 95%'
    for action in ('joint', 'enable'):
        with pytest.raises(RuntimeError, match='ordinary work'):
            service.dispatch({'action': action, 'q': [0.] * 6})
    assert host.checks == 2 and service.ctrl.enabled == 1
    assert service.recovery_host_override_status()['active']


def test_override_clears_after_disable_or_owner_replacement(tmp_path, setup):
    service = service_fixture(tmp_path, setup)
    service.recover_control({'action': 'recover_control', '_recovery_requested_monotonic': 100.})
    service.ctrl.closed = True
    assert not service.recovery_host_override_status()['active']
    service.ctrl.closed = False
    service._grant_recovery_host_override()
    service.ctrl.robot = SimpleNamespace(name='ordinary replacement')
    assert not service.recovery_host_override_status()['active']
    service.host_guard = UnsafeHost()
    requested = []
    service.handle_request = lambda req: requested.append(req)
    service.tick()
    assert requested[0]['action'] == 'shutdown'


def test_healthy_duplicate_does_not_grant_override_to_an_ordinary_owner(tmp_path, setup):
    service = service_fixture(tmp_path, setup)
    service.ctrl.healthy, service.ctrl.mode, service.ctrl.closed = True, 'Hold', False
    result = service.recover_control({'action': 'recover_control', '_recovery_requested_monotonic': 100.})
    assert result['phase'] == 'already_controlled'
    assert not result['recovery_host_override']['active']
    assert service.ctrl.enabled == 0


def test_retained_fresh_hold_after_request_expiry_receives_exact_owner_override(tmp_path, setup):
    service = service_fixture(tmp_path, setup)
    _, ctrl, _, _, clock, _ = setup
    enable = ctrl.enable
    def delayed():
        enable()
        clock.now = 111.
    ctrl.enable = delayed
    result = service.recover_control({'action': 'recover_control', '_recovery_requested_monotonic': 100.})
    assert result['phase'] == 'blocked' and 'expired' in result['error']
    assert result['recovery_host_override']['active']
    assert service.ctrl.mode == 'Hold' and not service.ctrl.closed
    service.host_guard = UnsafeHost()
    shutdowns = []
    service.handle_request = lambda req: shutdowns.append(req)
    service.tick()
    assert not shutdowns


def test_actual_connection_factory_skips_host_gate_only_for_explicit_guarded_recovery(tmp_path):
    import ast
    import hashlib

    import autoweld.backend as backend
    source = ast.parse(backend.__loader__.get_source(backend.__name__))
    run = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run_application')
    factory = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'connect_robot')
    model_path = tmp_path/'model.xml'
    model_path.write_text('<mujoco/>')
    bounds = np.array([[-2., 2.]] * 5+[[-5., -2.1]])
    host, calls = UnsafeHost(), []
    def sdk(**kwargs):
        kwargs['admission_guard']()
        calls.append(kwargs)
        return SimpleNamespace()
    namespace = {'check_connection_admission': backend.check_connection_admission,
        'host_guard': host, 'compiled_model': lambda: SimpleNamespace(jnt_range=bounds),
        'joint_calibration': {'fixed_motor_offsets_rad': [0.] * 6}, 'np': np,
        'live': True, 'channel': 'offline-test-only', 'verify_disabled_start': lambda *args: None,
        'get_yam_robot': sdk, 'ArmType': SimpleNamespace(YAM='yam'),
        'GripperType': SimpleNamespace(NO_GRIPPER='no-gripper'), 'model_path': str(model_path),
        'hashlib': hashlib, 'Path': type(model_path), 'gravity_record': lambda *args: None,
        'runtime': tmp_path, 'json': json}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), '<connection-factory>', 'exec'), namespace)
    connect = namespace['connect_robot']
    guards = []
    connect(admission_guard=lambda: guards.append('physical/deadline/cancel guard'), enable_deadline=110.,
        recovery_rest_q=np.array([0., 0., 0., 0., 0., -3.]), recovery_joint_limits=bounds)
    assert calls and len(guards) >= 3 and host.checks == 0
    assert calls[0]['recovery_near_rest_tolerance_rad'] == .15
    with pytest.raises(RuntimeError, match='ordinary work'):
        connect()
    assert host.checks == 1 and len(calls) == 1


def test_partial_chain_cleanup_failure_and_native_probe_are_exposed_without_new_sdk_io(setup):
    recovery, ctrl, _, _, clock, _ = setup
    probe = sample(clock, status=['0x0', '0xd', '0x0', '0x0', '0x0', '0x0'])
    calls = []
    chain = SimpleNamespace(running=False, powered=True, power_unknown=True,
        _startup_enabled_ids={1}, _startup_inflight_id=2,
        get_enable_failure_diagnostics=lambda: calls.append('cached accessor') or {
            'last_stage': 'qualify_before_fc', 'last_sample': probe,
            'enabled_motor_ids': [1], 'inflight_motor_id': 2})
    def fail(*args, **kwargs):
        error = RuntimeError('Explicit enable requires fresh clean-disabled feedback or its own enabled prefix')
        error.recovery_chain = chain
        error.recovery_cleanup_verified = False
        error.recovery_cleanup_error = 'Motor 2 remains communication-timeout disabled during cleanup proof'
        error.enable_failure = {'stage': 'before_fc', 'motor_id': 2}
        raise error
    ctrl.robot_factory = fail
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'own enabled prefix' in result['error']
    assert 'Motor 2 remains' in result['recovery_cleanup_error']
    assert result['recovery_cleanup_verified'] is False
    assert result['enable_failure'] == {'stage': 'before_fc', 'motor_id': 2}
    assert result['native_last_probe']['status'][1] == '0xd'
    assert result['partial_chain']['enabled_motor_ids'] == [1]
    assert result['partial_chain']['inflight_motor_id'] == 2
    assert result['partial_chain']['cached_only'] and not result['partial_chain']['current_hardware_verified']
    assert calls == ['cached accessor']
    assert ctrl.mode != 'Hold' and not ctrl.closed
    json.dumps(result, allow_nan=False)
    assert recovery.state()['recovery_cleanup_error'] == result['recovery_cleanup_error']


def test_diagnostic_accessor_failure_never_masks_original_startup_error(setup):
    recovery, ctrl, _, _, clock, _ = setup
    def unavailable():
        raise RuntimeError('Cached diagnostics unavailable')
    def fail(*args, **kwargs):
        error = RuntimeError('Original startup blocker')
        error.recovery_chain = SimpleNamespace(get_enable_failure_diagnostics=unavailable)
        error.recovery_cleanup_error = RuntimeError('Original cleanup blocker')
        error.native_sample = {**sample(clock), 'extra': np.array([float('nan'), 2.])}
        raise error
    ctrl.robot_factory = fail
    result = recovery.recover(100.)
    assert result['error'] == 'Original startup blocker'
    assert result['recovery_cleanup_error'] == 'Original cleanup blocker'
    assert result['diagnostic_error'] == 'Cached diagnostics unavailable'
    assert result['native_last_probe']['extra'] == [None, 2.]
    json.dumps(result, allow_nan=False)


class RetainedChain:
    def __init__(self, clock, events):
        self.running = False
        self._control_thread = None
        self.clock, self.events = clock, events
        self.change = None
        self.closed = False
    def release_disabled_recovery_owner(self, **kwargs):
        self.events.append('same retained interface survey')
        kwargs['check_intent']()
        data = sample(self.clock)
        if self.change:
            self.change(data)
        kwargs['before_release'](data)
        self.closed = True
        self.events.append('retained socket closed without motoroff')
        return {'released': True, 'socket_closed': True, 'can_worker_stopped': True,
                'motor_power_changed': False, 'sample': data}


def test_explicit_new_request_releases_fresh_disabled_retained_interface_then_reproves(setup):
    recovery, ctrl, _, _, clock, events = setup
    chain = RetainedChain(clock, events)
    recovery.retained_chain = chain
    ctrl.closed = False  # The preceding failed constructor could not verify power.
    result = recovery.recover(100.)
    assert result['phase'] == 'holding' and ctrl.enabled == 1
    assert recovery.retained_chain is None and chain.closed
    assert events.index('same retained interface survey') < events.index('retained socket closed without motoroff')
    assert events.index('retained socket closed without motoroff') < events.index('quiet') < events.index('factory')
    assert result['retained_owner_release']['motor_power_changed'] is False


@pytest.mark.parametrize('change,reason', [
    (lambda d: d.update(status=['0x1'] * 6), 'disabled status'),
    (lambda d: d.update(status=['0x8'] * 6), 'disabled status'),
    (lambda d: d.update(physical_timestamp=1000.), 'fresh physical'),
    (lambda d: d.update(velocity_rad_s=[.1] * 6), 'not quiet'),
    (lambda d: (d.update(joint_rad=[.2, 0, 0, 0, 0, -3.], sdk_feedback_position_rad=[.2, 0, 0, 0, 0, -3.]),
        d['native_p_m'][0].update(value=.2)), 'farther than 0.15'),
])
def test_retained_bad_physical_proof_never_closes_interface_or_opens_another_reader(setup, change, reason):
    recovery, ctrl, _, _, clock, events = setup
    chain = RetainedChain(clock, events)
    chain.change = change
    recovery.retained_chain = chain
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and reason in result['error']
    assert recovery.retained_chain is chain and not chain.closed
    assert result['partial_owner_retained']
    assert events == ['same retained interface survey'] and ctrl.enabled == 0


@pytest.mark.parametrize('writer', ['running', 'alive', 'unknown', 'robot_server'])
def test_retained_active_or_unknown_writer_never_releases_owner(setup, writer):
    recovery, ctrl, _, _, clock, events = setup
    chain = RetainedChain(clock, events)
    recovery.retained_chain = chain
    if writer == 'running':
        chain.running = True
    elif writer == 'alive':
        chain._control_thread = SimpleNamespace(is_alive=lambda: True)
    elif writer == 'unknown':
        chain.running = None
    else:
        ctrl.robot = SimpleNamespace(motor_chain=chain, _server_thread=SimpleNamespace(is_alive=lambda: True))
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'writer' in result['error']
    assert not events and not chain.closed and recovery.retained_chain is chain


def test_released_socket_is_not_stranded_if_deadline_expires_before_next_stage(setup):
    recovery, _, _, _, clock, events = setup
    chain = RetainedChain(clock, events)
    original = chain.release_disabled_recovery_owner
    def release(**kwargs):
        result = original(**kwargs)
        clock.now = 111.
        return result
    chain.release_disabled_recovery_owner = release
    recovery.retained_chain = chain
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'expired' in result['error']
    assert chain.closed and recovery.retained_chain is None
    assert 'factory' not in events and 'quiet' not in events


def test_raw_timeout_4000_is_200ms_and_cannot_claim_4000ms_proof(setup):
    recovery, ctrl, _, reader, _, _ = setup
    clear = reader.clear_timeout_latches
    def wrong_units(**kwargs):
        result = clear(**kwargs)
        for row in result['after']['native_timeout']:
            row['value'] = 4000
        return result
    reader.clear_timeout_latches = wrong_units
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'raw 80000' in result['error']
    assert ctrl.enabled == 0


def test_timeout_evidence_exposes_raw_ticks_and_converted_milliseconds(setup):
    result = setup[0].recover(100.)
    assert result['timeout_register_ticks'] == result['timeout_register_value'] == 80000
    assert result['timeout_register_tick_us'] == 50 and result['timeout_ms'] == 4000
    assert [row['value'] for row in result['timeout_readbacks']] == [80000] * 6
    assert [row['raw_value'] for row in result['timeout_readbacks_ms']] == [80000] * 6
    assert [row['timeout_ms'] for row in result['timeout_readbacks_ms']] == [4000.] * 6


def test_retained_robot_with_never_started_can_worker_and_stopped_server_can_retry(setup):
    recovery, ctrl, _, _, clock, events = setup
    chain = RetainedChain(clock, events)
    ctrl.robot = SimpleNamespace(motor_chain=chain, _server_thread=SimpleNamespace(is_alive=lambda: False))
    ctrl.closed = False
    recovery.retained_chain = chain
    result = recovery.recover(100.)
    assert result['phase'] == 'holding' and chain.closed and recovery.retained_chain is None


def test_retained_release_keeps_interface_during_cancellation_before_close(setup):
    recovery, ctrl, _, _, clock, events = setup
    chain = RetainedChain(clock, events)
    recovery.retained_chain = chain
    cancelled = [False]
    recovery.cancelled = lambda intent: cancelled[0]
    chain.change = lambda data: cancelled.__setitem__(0, True)
    result = recovery.recover(100.)
    assert result['phase'] == 'blocked' and 'cancelled' in result['error']
    assert not chain.closed and recovery.retained_chain is chain and ctrl.enabled == 0
