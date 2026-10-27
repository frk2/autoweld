"""Firmware timeout and disabled recovery, with no CAN sockets or SDK workers."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.motor_drivers import position_speed
from i2rt.motor_drivers.dm_driver import DMChainCanInterface
from i2rt.motor_drivers.utils import FeedbackFrameInfo
from i2rt.robots import disabled_recovery as recovery
from i2rt.robots.motor_chain_robot import MotorChainRobot


@pytest.fixture
def bench(monkeypatch):
    clock = [100.]
    q = np.array([0., .1, .2, 0., 0., -3.5])
    bounds = np.tile([-2., 2.], (6, 1))
    bounds[5] = [-5.37, -2.087]
    status = ['0xd']*6
    events = []
    options = dict(age=0., speed=0., temperature=32., pm_delta=0., foreign=False,
                   mode=2, bad_readback=False, freeze_timestamp=False)
    registers = {(mid, reg): (8000 if reg == 'TIMEOUT' else 2)
                 for mid in range(1, 7) for reg in ('TIMEOUT', 'CTRL_MODE')}

    def frame(mid, kind, **kwargs):
        return FeedbackFrameInfo(id=mid, error_code=status[mid-1], error_message='test',
            position=q[mid-1], velocity=options['speed'], torque=0.,
            temperature_mos=options['temperature'], temperature_rotor=32.,
            timestamp=100. if options['freeze_timestamp'] else clock[0]-options['age'])

    def sample(interface, mid, reg):
        value = q[mid-1]+options['pm_delta'] if reg == 'p_m' else registers[mid, reg]
        if reg == 'CTRL_MODE':
            value = options['mode']
        return dict(motor_id=7 if options['foreign'] else mid, register=reg, value=value,
                    timestamp=100. if options['freeze_timestamp'] else clock[0]-options['age'])

    def write(interface, mid, reg, value):
        events.append(('write', mid, reg, value, clock[0]))
        if not options['bad_readback']:
            registers[mid, reg] = value

    def clear(mid):
        events.append(('clear', mid, clock[0]))
        status[mid-1] = '0x0'

    interface = SimpleNamespace(read_motor_status=frame, clean_error=clear,
        close=lambda: events.append(('close',)))
    monkeypatch.setattr(recovery, 'read_register_sample', sample)
    monkeypatch.setattr(recovery, 'write_register', write)
    reader = recovery.DisabledArmRecovery(interface=interface, clock=lambda: clock[0],
        wall_clock=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0]+dt))
    return SimpleNamespace(reader=reader, clock=clock, q=q, bounds=bounds, status=status,
                           events=events, options=options, registers=registers)


def recover(b, **kwargs):
    return b.reader.clear_timeout_latches(rest_q=b.q.copy(), joint_limits=b.bounds, **kwargs)


def test_survey_is_read_only_json_and_preserves_negative_j6_branch(bench):
    sample = bench.reader.survey()
    assert sample['joint_rad'][5] == -3.5
    assert sample['status'] == ['0xd']*6
    assert sample['mapping']['boot_normalization'] == 'none'
    assert sample['native_timeout'][0]['value'] == 8000
    json.dumps(sample, allow_nan=False)
    assert not bench.events


def test_recovery_waits_quiet_and_writes_ram_4000_only_then_clears_only_timeout(bench):
    bench.status[2] = '0x0'
    checks = []
    result = recover(bench, before_write=lambda: checks.append(bench.clock[0]), deadline=110.)
    assert result['timeout_ms'] == 4000
    assert result['cleared_motor_ids'] == [1, 2, 4, 5, 6]
    assert result['after']['status'] == ['0x0']*6
    assert not result['enabled'] and not result['flash_written']
    assert len(checks) >= 11
    writes = [row for row in bench.events if row[0] == 'write']
    assert [(row[1], row[2], row[3]) for row in writes] == [(i, 'TIMEOUT', 80000) for i in range(1, 7)]
    assert writes[0][-1] >= 101.
    assert all(row[0] in ('write', 'clear') for row in bench.events)


@pytest.mark.parametrize('status', ['0x1', '0x8', '0x9', '0xa', '0xb', '0xc', '0xe'])
def test_enabled_or_other_fault_never_mutates(bench, status):
    bench.status[4] = status
    with pytest.raises(RuntimeError, match='only clean-disabled'):
        recover(bench)
    assert not bench.events


@pytest.mark.parametrize('option,value,pattern', [
    ('age', .501, 'Stale'), ('age', -.01, 'future'), ('speed', .051, 'stationary'),
    ('temperature', 60., 'below 60'), ('pm_delta', .021, 'disagree'),
    ('foreign', True, 'Foreign'), ('mode', 3, 'known native'),
    ('freeze_timestamp', True, 'advancing')])
def test_bad_feedback_never_mutates(bench, option, value, pattern):
    bench.options[option] = value
    with pytest.raises(RuntimeError, match=pattern):
        recover(bench)
    assert not bench.events


def test_rest_distance_and_j6_bounds_are_independent_guards(bench):
    rest = bench.q.copy()
    rest[0] += .151
    with pytest.raises(RuntimeError, match='near captured'):
        bench.reader.clear_timeout_latches(rest_q=rest, joint_limits=bench.bounds)
    bench.q[5] = -2.08
    rest[5] = -2.1
    with pytest.raises(RuntimeError, match='joint limits'):
        bench.reader.clear_timeout_latches(rest_q=rest, joint_limits=bench.bounds)
    assert not bench.events


def test_deadline_and_admission_failure_never_defer_or_enable(bench):
    with pytest.raises(TimeoutError, match='Original'):
        recover(bench, deadline=100.5)
    assert not bench.events
    def blocked():
        raise RuntimeError('host I/O pressure')
    with pytest.raises(RuntimeError, match='host I/O'):
        recover(bench, before_write=blocked)
    assert not bench.events


def test_host_guard_is_rechecked_before_each_mutation(bench):
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('host pressure changed')
    with pytest.raises(RuntimeError, match='host pressure'):
        recover(bench, before_write=guard)
    assert [row[:4] for row in bench.events] == [('write', 1, 'TIMEOUT', 80000)]
    assert bench.status == ['0xd']*6


def test_timeout_readback_failure_never_clears_or_enables(bench):
    bench.options['bad_readback'] = True
    with pytest.raises(RuntimeError, match='readback'):
        recover(bench)
    assert len(bench.events) == 1 and bench.events[0][:3] == ('write', 1, 'TIMEOUT')


def test_close_is_idempotent_socket_only(bench):
    bench.reader.close()
    bench.reader.close()
    assert bench.events == [('close',)]
    with pytest.raises(RuntimeError, match='closed'):
        bench.reader.survey()


def test_nonzero_offset_rejected_before_interface_open(monkeypatch):
    monkeypatch.setattr('i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface',
                        lambda **kw: pytest.fail('opened CAN'))
    with pytest.raises(ValueError, match='fixed zero'):
        recovery.DisabledArmRecovery(offsets_rad=np.ones(6))


def failed_robot(worker_alive=False, server_alive=False):
    events = []
    worker = SimpleNamespace(is_alive=lambda: worker_alive)
    server = SimpleNamespace(is_alive=lambda: server_alive, join=lambda **kw: events.append(('join', kw)))
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot._stop_event = threading.Event()
    robot._server_thread = server
    robot.motor_chain = SimpleNamespace(running=False, _control_thread=worker,
        _exchange_lock=threading.RLock(), motor_interface=SimpleNamespace(close=lambda: events.append(('close',))))
    return robot, events


def test_failed_owner_detach_stops_server_then_closes_socket_only():
    robot, events = failed_robot()
    result = robot.stop_for_disabled_recovery()
    assert robot._stop_event.is_set()
    assert events[0][0] == 'join' and 0 < events[0][1]['timeout'] <= 1.
    assert events[1:] == [('close',)]
    assert not result['motor_power_changed']


@pytest.mark.parametrize('worker_alive,server_alive', [(True, False), (False, True)])
def test_detach_refuses_any_live_writer(worker_alive, server_alive):
    robot, events = failed_robot(worker_alive, server_alive)
    with pytest.raises(RuntimeError, match='worker|writers'):
        robot.stop_for_disabled_recovery()
    assert ('close',) not in events


def test_startup_admission_preserves_original_deadline_and_checks_guard(monkeypatch):
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    calls = []
    chain.admission_guard = lambda: calls.append('guard')
    chain.enable_deadline = 10.
    monkeypatch.setattr('i2rt.motor_drivers.dm_driver.time.monotonic', lambda: 9.)
    chain._check_startup_admission()
    assert calls == ['guard']
    monkeypatch.setattr('i2rt.motor_drivers.dm_driver.time.monotonic', lambda: 11.)
    with pytest.raises(TimeoutError, match='Original'):
        chain._check_startup_admission()
    assert calls == ['guard']


def test_startup_never_implicitly_clears_0xd_even_legacy_allow_flag(monkeypatch):
    interface = SimpleNamespace(read_motor_status=lambda *a: SimpleNamespace(error_code='0xd', velocity=0.),
                                close=lambda: None)
    monkeypatch.setattr('i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface', lambda **kw: interface)
    monkeypatch.setattr(position_speed, 'read_register', lambda *a: pytest.fail('unexpected read after fault'))
    monkeypatch.setattr(position_speed, 'write_register', lambda *a: pytest.fail('wrote after latch'))
    with pytest.raises(RuntimeError, match='disabled'):
        position_speed.prepare_modes('unused', [(1, 'DM4340')], {1: 'POS_VEL'}, allow_timeout_recovery=True)


def test_detach_exchange_lock_has_bounded_wait():
    robot, events = failed_robot()
    class BusyLock:
        def acquire(self, **kwargs):
            assert 0 <= kwargs['timeout'] <= .01
            return False
        def release(self):
            pytest.fail('released unowned lock')
    robot.motor_chain._exchange_lock = BusyLock()
    with pytest.raises(TimeoutError, match='exchange lock'):
        robot.stop_for_disabled_recovery(timeout_s=.01)
    assert ('close',) not in events


@pytest.mark.parametrize('operation', ['fc', 'register', 'clear'])
def test_every_transport_mutation_retry_rechecks_original_admission(monkeypatch, operation):
    import can
    from i2rt.motor_config_tool import dm_motor_registers as registers
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
    from i2rt.motor_drivers.utils import ReceiveMode

    interface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    sent, checks = [], []
    clock = [1.]
    interface.name = 'offline'
    interface.receive_mode = ReceiveMode.p16
    interface.bus = SimpleNamespace(send=lambda packet: sent.append(packet), channel_info='mock')
    interface.try_receive_message = lambda *a, **kw: None
    def receive(*a, **kw):
        clock[0] += .02
        return None
    interface._receive_message = receive
    def guard():
        checks.append(1)
        if len(checks) >= 2:
            raise TimeoutError('Original intent expired')
    interface._mutation_admission_guard = guard
    monkeypatch.setattr('i2rt.motor_drivers.can_interface.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('i2rt.motor_drivers.can_interface.time.sleep', lambda dt: None)
    with pytest.raises(TimeoutError, match='Original'):
        if operation == 'fc':
            interface._send_message_get_response(1, 1, [255]*7+[0xFC])
        elif operation == 'register':
            registers.write_register(interface, 1, 'TIMEOUT', 80000)
        else:
            interface.clean_error(1)
    assert len(sent) == 1
    assert len(checks) == 2
    assert isinstance(sent[0], can.Message)


def test_recovery_removes_retry_guards_after_success_and_failure(bench):
    recover(bench)
    assert bench.reader.interface._mutation_admission_guard is None
    bench.options['bad_readback'] = True
    bench.registers[1, 'TIMEOUT'] = 8000
    with pytest.raises(RuntimeError, match='readback'):
        recover(bench)
    assert bench.reader.interface._mutation_admission_guard is None


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_guarded_factory_construction_failure_preserves_or_cleans_partial_owner(monkeypatch, cleanup_fails):
    from i2rt.robots import get_robot as factory
    from i2rt.robots.utils import GripperType

    events, captured = [], {}
    q = np.array([0., .1, .2, 0., 0., -3.5])
    class Chain:
        def __init__(self, motors, offsets, directions, *a, **kw):
            captured['chain'] = self
            captured['options'] = kw
            self.motor_offset = np.asarray(offsets)
            self.position_speed_motor_ids = set(range(1, 7))
            self.state = [SimpleNamespace(position=x) for x in q]
            self.commands = [SimpleNamespace(pos=0.) for _ in q]
        def read_states(self):
            return [SimpleNamespace(pos=x) for x in q]
        def _joint_position_real_to_sim_idx(self, value, index):
            return value
        def start_thread(self):
            events.append('worker started')
        def qualify_failed_startup_cleanup(self):
            return True
        def close(self, **kwargs):
            events.append('cleanup')
            if cleanup_fails:
                raise RuntimeError('disabled verification failed; retain owner')
    def fail_construct(**kwargs):
        assert kwargs['start_control_server'] is False
        raise ValueError('model constructor failed')
    monkeypatch.setattr(factory, 'DMChainCanInterface', Chain)
    monkeypatch.setattr('i2rt.robots.position_speed_robot.PositionSpeedRobot', fail_construct)
    limits = np.tile([-2., 2.], (6, 1))
    limits[5] = [-5.37, -2.087]
    with pytest.raises(ValueError, match='model constructor') as caught:
        factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='POS_VEL',
            xml_path_override='unused.xml', arm_joint_limits_override=limits,
            arm_motor_offsets_override=np.zeros(6), admission_guard=lambda: None, enable_deadline=10.)
    assert events == ['worker started', 'cleanup']
    assert caught.value.recovery_cleanup_verified is (not cleanup_fails)
    assert captured['options']['repair_startup'] is False
    if cleanup_fails:
        assert caught.value.recovery_chain is captured['chain']
    else:
        assert not hasattr(caught.value, 'recovery_chain')


def test_enable_feedback_accepts_only_its_own_prefix_not_faults(bench):
    bench.status[:] = ['0x0']*6
    bench.status[0] = '0x1'
    sample = bench.reader.qualify_enable_feedback(rest_q=bench.q, joint_limits=bench.bounds, enabled_motor_ids=[1])
    assert sample['status'][0] == '0x1'
    with pytest.raises(RuntimeError, match='enabled prefix='):
        bench.reader.qualify_enable_feedback(rest_q=bench.q, joint_limits=bench.bounds)
    bench.status[3] = '0xd'
    with pytest.raises(RuntimeError, match='enabled prefix='):
        bench.reader.qualify_enable_feedback(rest_q=bench.q, joint_limits=bench.bounds,
                                             enabled_motor_ids=[1], inflight_motor_id=4)
    assert not bench.events


@pytest.mark.parametrize('gap_or_drift', ['gap', 'drift'])
def test_quiet_proof_rejects_unsampled_time_or_pose_drift(bench, gap_or_drift):
    def sleep(dt):
        bench.clock[0] += .6 if gap_or_drift == 'gap' else dt
        if gap_or_drift == 'drift':
            bench.q[0] += .002
    bench.reader.sleep = sleep
    with pytest.raises(RuntimeError, match='continuous|pose changed'):
        recover(bench)
    assert not bench.events


@pytest.mark.parametrize('displaced', [False, True])
def test_guarded_startup_reproves_rest_after_setup_before_any_fc(bench, monkeypatch, displaced):
    from i2rt.motor_drivers import dm_driver
    bench.status[:] = ['0x0']*6
    rest = bench.q.copy()
    interface = bench.reader.interface
    fc = []
    interface._drain_bus = lambda **kw: bench.q.__setitem__(0, .2) if displaced else None
    interface.set_control = lambda *a, **kw: None
    def send(mid, motor_id, data):
        chain._check_startup_admission()
        fc.append(mid)
        bench.status[mid-1] = '0x1'
        return SimpleNamespace(error_code='0x1')
    interface._send_message_get_response = send
    interface.parse_recv_message = lambda row, *a, **kw: row
    monkeypatch.setattr(recovery, 'DisabledArmRecovery', lambda **kw: bench.reader)
    monkeypatch.setattr(dm_driver.time, 'monotonic', lambda: bench.clock[0])
    monkeypatch.setattr(dm_driver.time, 'sleep', lambda dt: bench.clock.__setitem__(0, bench.clock[0]+dt))
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_interface = interface
    chain.motor_list = bench.reader.motor_list
    chain.motor_offset = np.zeros(6)
    chain.recovery_rest_q = rest
    chain.recovery_joint_limits = bench.bounds
    chain.recovery_near_rest_tolerance_rad = .15
    chain.repair_startup = False
    chain.position_speed_motor_ids = set(range(1, 7))
    chain.admission_guard = lambda: None
    chain.enable_deadline = 110.
    chain._startup_feedback_guard = None
    chain._startup_enabled_ids = set()
    chain._startup_inflight_id = None
    chain._update_absolute_positions = lambda *a: None
    if displaced:
        with pytest.raises(RuntimeError, match='near captured'):
            chain._motor_on()
        assert not fc
    else:
        chain._motor_on()
        assert fc == list(range(1, 7))
        assert bench.clock[0] >= 101.
        assert chain.running


def test_factory_rejects_partial_or_mismatched_recovery_policy_before_can(monkeypatch):
    from i2rt.robots import get_robot as factory
    from i2rt.robots.utils import GripperType
    monkeypatch.setattr(factory, 'DMChainCanInterface', lambda *a, **kw: pytest.fail('opened CAN'))
    limits = np.tile([-2., 2.], (6, 1))
    limits[5] = [-5.37, -2.087]
    with pytest.raises(ValueError, match='Recovery startup'):
        factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='POS_VEL',
            xml_path_override='unused.xml', arm_joint_limits_override=limits,
            arm_motor_offsets_override=np.zeros(6), recovery_rest_q=[0, .1, .2, 0, 0, -3.5],
            recovery_joint_limits=limits+1, admission_guard=lambda: None, enable_deadline=110.)


def test_healthy_recovery_with_realistic_reply_cost_fits_original_deadline(bench, monkeypatch):
    # Live survey measured ~.18 s for 24 replies: charge every reply .0075 s.
    # Fake time advances; this test neither sleeps nor creates hardware/threads.
    original_frame = bench.reader.interface.read_motor_status
    original_sample = recovery.read_register_sample
    original_write = recovery.write_register
    def frame(*args, **kwargs):
        bench.clock[0] += .0075
        return original_frame(*args, **kwargs)
    def sample(*args, **kwargs):
        bench.clock[0] += .0075
        return original_sample(*args, **kwargs)
    def write(*args, **kwargs):
        bench.clock[0] += .0075
        return original_write(*args, **kwargs)
    bench.reader.interface.read_motor_status = frame
    monkeypatch.setattr(recovery, 'read_register_sample', sample)
    monkeypatch.setattr(recovery, 'write_register', write)
    start = bench.clock[0]
    result = recover(bench, deadline=start+10.)
    elapsed = bench.clock[0]-start
    assert 2. < elapsed < 4.
    assert result['after']['status'] == ['0x0']*6


def test_startup_callback_clears_after_success_and_never_becomes_runtime_timeout(monkeypatch):
    import time

    from i2rt.motor_drivers import dm_driver
    state = FeedbackFrameInfo(id=1, error_code='0x0', error_message='', position=.1,
        velocity=0., torque=0., temperature_mos=30., temperature_rotor=30., timestamp=time.time())
    interface = SimpleNamespace(_drain_bus=lambda **kw: None,
        read_motor_status=lambda *a, **kw: state, set_control=lambda *a, **kw: None,
        _send_message_get_response=lambda *a, **kw: state,
        parse_recv_message=lambda *a, **kw: FeedbackFrameInfo(**{**state.__dict__, 'error_code': '0x1'}),
        close=lambda: None)
    monkeypatch.setattr(dm_driver, 'DMSingleMotorCanInterface', lambda **kw: interface)
    monkeypatch.setattr(dm_driver, 'run_startup_checks', lambda *a, **kw: None)
    monkeypatch.setattr(position_speed, 'prepare_modes', lambda *a, **kw: {})
    chain = DMChainCanInterface([(1, 'DM4340')], np.zeros(1), np.ones(1), channel='can0',
        start_thread=False, motor_modes={1: 'POS_VEL'}, repair_startup=False,
        admission_guard=lambda: None, enable_deadline=time.monotonic()+10.)
    assert chain.motor_interface._mutation_admission_guard is None
    assert chain._startup_feedback_guard is None
    chain.enable_deadline = 0.  # An expired startup intent is not a runtime gate.
    assert chain.motor_interface._mutation_admission_guard is None


@pytest.mark.parametrize('disabled_verified', [False, True])
def test_chain_constructor_failure_keeps_owner_if_partial_disable_unverified(monkeypatch, disabled_verified):
    import time

    from i2rt.motor_drivers import dm_driver
    events = []
    state = SimpleNamespace(error_code='0x0' if disabled_verified else '0x1',
                            position=.1, velocity=0., timestamp=time.time())
    interface = SimpleNamespace(read_motor_status=lambda *a, **kw: state,
        motor_off=lambda mid: events.append(('off', mid)), close=lambda: events.append(('close',)))
    monkeypatch.setattr(dm_driver, 'DMSingleMotorCanInterface', lambda **kw: interface)
    monkeypatch.setattr(dm_driver, 'run_startup_checks', lambda *a, **kw: None)
    monkeypatch.setattr(position_speed, 'prepare_modes', lambda *a, **kw: {})
    def failed_enable(self):
        raise RuntimeError('FC reply lost after possible enable')
    monkeypatch.setattr(DMChainCanInterface, '_motor_on', failed_enable)
    with pytest.raises(RuntimeError, match='FC reply lost') as caught:
        DMChainCanInterface([(1, 'DM4340')], np.zeros(1), np.ones(1), channel='can0',
            start_thread=False, motor_modes={1: 'POS_VEL'}, repair_startup=False,
            admission_guard=lambda: None, enable_deadline=time.monotonic()+10.)
    if disabled_verified:
        assert events[-1] == ('close',)
        assert not hasattr(caught.value, 'recovery_chain')
    else:
        assert not events  # No rest proof: no FD, even for a possible enabled prefix.
        assert caught.value.recovery_chain.power_unknown
        assert caught.value.recovery_chain.motor_interface is interface
        assert not caught.value.recovery_chain.running


def test_failed_startup_rollback_refuses_poweroff_if_rest_proof_is_lost(bench, monkeypatch):
    bench.status[:] = ['0x1']+['0x0']*5
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_interface = bench.reader.interface
    chain.motor_list = bench.reader.motor_list
    chain.motor_offset = np.zeros(6)
    chain.recovery_rest_q = bench.q.copy()
    chain.recovery_joint_limits = bench.bounds
    chain.recovery_near_rest_tolerance_rad = .15
    chain._exchange_lock = threading.RLock()
    chain._control_thread = SimpleNamespace(join=lambda **kw: None, is_alive=lambda: False)
    chain.running = True
    monkeypatch.setattr(recovery, 'DisabledArmRecovery', lambda **kw: bench.reader)
    bench.q[0] += .2
    with pytest.raises(RuntimeError, match='near captured'):
        chain.qualify_failed_startup_cleanup()
    assert not chain.running
    assert not bench.events  # No FD, fault-clear, socket close or hidden writer.


def test_successful_guarded_factory_drops_one_shot_callback_and_deadline(monkeypatch):
    from i2rt.robots import get_robot as factory
    from i2rt.robots.utils import GripperType

    captured = {}
    q = [0., .1, .2, 0., 0., -3.5]
    class Chain:
        def __init__(self, motors, offsets, directions, *a, **kw):
            captured['chain'] = self
            self.motor_offset = np.asarray(offsets)
            self.position_speed_motor_ids = set(range(1, 7))
            self.state = [SimpleNamespace(position=x) for x in q]
            self.commands = [SimpleNamespace(pos=0.) for x in q]
            self.admission_guard = kw['admission_guard']
            self.enable_deadline = kw['enable_deadline']
        def read_states(self):
            return [SimpleNamespace(pos=x) for x in q]
        def _joint_position_real_to_sim_idx(self, value, index):
            return value
        def start_thread(self):
            pass
        def _check_startup_admission(self):
            self.admission_guard()
    monkeypatch.setattr(factory, 'DMChainCanInterface', Chain)
    monkeypatch.setattr('i2rt.robots.position_speed_robot.PositionSpeedRobot', lambda **kw:
        SimpleNamespace(_server_thread=SimpleNamespace(start=lambda: None)))
    limits = np.tile([-2., 2.], (6, 1))
    limits[5] = [-5.37, -2.087]
    calls = []
    factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='POS_VEL',
        xml_path_override='unused.xml', arm_joint_limits_override=limits,
        arm_motor_offsets_override=np.zeros(6), admission_guard=lambda: calls.append(1), enable_deadline=10.)
    assert calls == [1]
    assert captured['chain'].admission_guard is None
    assert captured['chain'].enable_deadline is None
    assert captured['chain'].startup_admission_deadline == 10.


def test_status_rejection_retains_exact_native_failure_and_prefix(bench):
    bench.status[:] = ['0x0']*6
    bench.status[1] = '0xd'
    with pytest.raises(RuntimeError, match='Motor 2:.*0xd.*inflight=None') as caught:
        bench.reader.qualify_enable_feedback(rest_q=bench.q, joint_limits=bench.bounds)
    detail = caught.value.enable_failure
    assert detail['motor_id'] == 2
    assert detail['status'] == '0xd'
    assert detail['sample']['status'] == ['0x0', '0xd', '0x0', '0x0', '0x0', '0x0']
    assert caught.value.native_sample is detail['sample']
    json.dumps(detail, allow_nan=False)


def test_enable_failure_accessor_returns_cached_copy_without_bus_reads(bench):
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain._startup_stage = 'motor_2_fc_send'
    chain._startup_enabled_ids = {1}
    chain._startup_inflight_id = 2
    chain._startup_last_sample = bench.reader.survey()
    chain._cache_enable_failure(RuntimeError('native transition rejected'))
    chain.motor_interface = SimpleNamespace(read_motor_status=lambda *a: pytest.fail('read SDK in cached accessor'))
    details = chain.get_enable_failure_diagnostics()
    assert details['failure']['stage'] == 'motor_2_fc_send'
    assert details['enabled_motor_ids'] == [1]
    assert details['inflight_motor_id'] == 2
    details['last_sample']['status'][0] = 'forged'
    assert chain.get_enable_failure_diagnostics()['last_sample']['status'][0] == '0xd'


@pytest.mark.parametrize('ticks,milliseconds', [(8000, 400), (4000, 200), (80000, 4000)])
def test_native_watchdog_units_are_50_microsecond_ticks(ticks, milliseconds):
    assert position_speed.timeout_register_ticks_to_ms(ticks) == milliseconds
    assert position_speed.timeout_ms_to_register_ticks(milliseconds) == ticks


def test_requested_watchdog_four_seconds_is_raw_eighty_thousand_ticks(bench):
    assert position_speed.MOTOR_COMMUNICATION_TIMEOUT_MS == 4000
    assert position_speed.MOTOR_COMMUNICATION_TIMEOUT_REGISTER_TICKS == 80000
    result = recover(bench)
    assert result['timeout_ms'] == 4000
    assert result['timeout_register_ticks'] == 80000
    assert [row['value'] for row in result['after']['native_timeout']] == [80000]*6
    assert [row['timeout_ms'] for row in result['after']['native_timeout']] == [4000]*6
    assert all(row['value_units'] == '50us_ticks' for row in result['after']['native_timeout'])


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), 2**32, .5])
def test_bad_native_timeout_ticks_rejected(value):
    with pytest.raises(ValueError, match='uint32'):
        position_speed.timeout_register_ticks_to_ms(value)


@pytest.mark.parametrize('statuses', [['0x0']*6, ['0xd']*6, ['0x0', '0xd']*3])
def test_retained_owner_release_uses_same_socket_and_never_mutates_motors(bench, monkeypatch, statuses):
    from i2rt.motor_drivers import dm_driver
    bench.status[:] = statuses
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_interface = bench.reader.interface
    chain.motor_offset = np.zeros(6)
    chain.running = False
    chain._exchange_lock = threading.RLock()
    chain._control_thread = None
    monkeypatch.setattr(recovery, 'DisabledArmRecovery', lambda **kw: bench.reader)
    monkeypatch.setattr(dm_driver.time, 'time', lambda: bench.clock[0])
    monkeypatch.setattr(dm_driver.time, 'monotonic', lambda: bench.clock[0])
    before = []
    report = chain.release_disabled_recovery_owner(rest_q=bench.q.copy(), joint_limits=bench.bounds,
        deadline=110., before_release=lambda sample: before.append(sample['status']))
    assert report['released'] and report['socket_closed'] and not report['motor_power_changed']
    assert before == [statuses]
    assert report['clean_disabled'] is all(status == '0x0' for status in statuses)
    assert bench.events == [('close',)]
    assert bench.status == statuses  # Native 0xD stays latched; release does not clear it.


def test_retained_owner_release_callback_rejection_preserves_owner(bench, monkeypatch):
    from i2rt.motor_drivers import dm_driver
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_interface = bench.reader.interface
    chain.motor_offset = np.zeros(6)
    chain.running = False
    chain._exchange_lock = threading.RLock()
    monkeypatch.setattr(recovery, 'DisabledArmRecovery', lambda **kw: bench.reader)
    monkeypatch.setattr(dm_driver.time, 'time', lambda: bench.clock[0])
    monkeypatch.setattr(dm_driver.time, 'monotonic', lambda: bench.clock[0])
    def blocked(sample):
        raise RuntimeError('collision admission refused')
    with pytest.raises(RuntimeError, match='collision admission'):
        chain.release_disabled_recovery_owner(rest_q=bench.q, joint_limits=bench.bounds,
            deadline=110., before_release=blocked)
    assert not bench.events


def test_failed_startup_already_timeout_disabled_does_not_require_fd(bench, monkeypatch):
    from i2rt.motor_drivers import dm_driver
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_interface = bench.reader.interface
    chain.motor_list = bench.reader.motor_list
    chain.motor_offset = np.zeros(6)
    chain.recovery_rest_q = bench.q.copy()
    chain.recovery_joint_limits = bench.bounds
    chain.recovery_near_rest_tolerance_rad = .15
    chain._exchange_lock = threading.RLock()
    chain.running = False
    monkeypatch.setattr(recovery, 'DisabledArmRecovery', lambda **kw: bench.reader)
    monkeypatch.setattr(dm_driver.time, 'time', lambda: bench.clock[0])
    assert chain.qualify_failed_startup_cleanup() is False
    assert not bench.events
    assert bench.status == ['0xd']*6


@pytest.mark.parametrize('native_ticks,starve_prefix', [(4000, False), (80000, False), (80000, True)])
def test_protocol_watchdog_units_and_prefix_expiry_are_not_masked(bench, monkeypatch, native_ticks, starve_prefix):
    from i2rt.motor_drivers import dm_driver
    bench.status[:] = ['0x0']*6
    interface = bench.reader.interface
    raw_frame = interface.read_motor_status
    last_command = [bench.clock[0]]*6
    timeout_s = position_speed.timeout_register_ticks_to_ms(native_ticks)/1000
    seed, fc = [], []
    def frame(mid, kind, **kwargs):
        if bench.clock[0]-last_command[mid-1] >= timeout_s:
            bench.status[mid-1] = '0xd'
        return raw_frame(mid, kind, **kwargs)
    interface.read_motor_status = frame
    interface._drain_bus = lambda **kw: None
    def set_control(mid, *args, **kwargs):
        chain._check_startup_admission()
        seed.append(mid)
        last_command[mid-1] = bench.clock[0]
        return frame(mid, None)
    def send(mid, *args):
        chain._check_startup_admission()
        fc.append(mid)
        last_command[mid-1] = bench.clock[0]
        bench.status[mid-1] = '0x1'
        response = SimpleNamespace(error_code='0x1')
        if starve_prefix and mid == 1:
            bench.clock[0] += 4.01
        return response
    interface.set_control = set_control
    interface._send_message_get_response = send
    interface.parse_recv_message = lambda row, *a, **kw: row
    monkeypatch.setattr(recovery, 'DisabledArmRecovery', lambda **kw: bench.reader)
    monkeypatch.setattr(dm_driver.time, 'monotonic', lambda: bench.clock[0])
    monkeypatch.setattr(dm_driver.time, 'sleep', lambda dt: bench.clock.__setitem__(0, bench.clock[0]+dt))
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_interface = interface
    chain.motor_list = bench.reader.motor_list
    chain.motor_offset = np.zeros(6)
    chain.recovery_rest_q = bench.q.copy()
    chain.recovery_joint_limits = bench.bounds
    chain.recovery_near_rest_tolerance_rad = .15
    chain.repair_startup = False
    chain.position_speed_motor_ids = set(range(1, 7))
    chain.admission_guard = lambda: None
    chain.enable_deadline = 110.
    chain._startup_feedback_guard = None
    chain._startup_enabled_ids = set()
    chain._startup_inflight_id = None
    chain._update_absolute_positions = lambda *a: None
    if native_ticks == 4000 or starve_prefix:
        with pytest.raises(RuntimeError, match='Motor 1:.*0xd'):
            chain._motor_on()
        diagnostics = chain.get_enable_failure_diagnostics()
        assert diagnostics['failure']['sample']['status'][0] == '0xd'
        assert diagnostics['enabled_motor_ids'] == ([1] if starve_prefix else [])
        assert fc == ([1] if starve_prefix else [])
        assert not bench.events  # No fault clear, FD or rollback behind the guard.
    else:
        chain._motor_on()
        assert seed == fc == list(range(1, 7))
        assert chain.running
