"""Native fault evidence and worker shutdown, using no robot or CAN socket."""

import json
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import can
import numpy as np
import pytest
from i2rt.motor_drivers.dm_driver import (
    DMChainCanInterface,
    DMSingleMotorCanInterface,
    MotorCmd,
    MotorFaultError,
)
from i2rt.motor_drivers.utils import ReceiveMode


def fault_packet():
    return can.Message(arbitration_id=0x12, data=[0xD2, 0x80, 0, 0x80, 0, 0, 35, 34],
                       timestamp=time.time(), is_extended_id=False)


def parser():
    interface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    interface.receive_mode = ReceiveMode.p16
    interface.name = 'fake'
    interface.bus = SimpleNamespace(channel_info='no hardware')
    return interface


def test_native_fault_keeps_received_status_and_frame_without_clearing():
    packet = fault_packet()
    with pytest.raises(MotorFaultError, match='motor id: 2, error: loss communication') as caught:
        parser().parse_recv_message(packet, 'DM4340')
    assert isinstance(caught.value, RuntimeError)
    detail = caught.value.diagnostic
    assert detail['feedback']['id'] == 2
    assert detail['feedback']['error_code'] == '0xd'
    assert detail['feedback']['temperature_mos'] == 35
    assert detail['feedback']['timestamp'] == packet.timestamp
    assert detail['frame']['data_hex'] == 'd280008000002322'
    assert detail['frame']['arbitration_id'] == 0x12
    json.dumps(detail, allow_nan=False)


def test_explicit_ignore_error_keeps_existing_status_read_behavior():
    result = parser().parse_recv_message(fault_packet(), 'DM4340', ignore_error=True)
    assert result.error_code == '0xd'


def chain():
    result = DMChainCanInterface.__new__(DMChainCanInterface)
    result.state_lock = threading.Lock()
    result.command_lock = threading.Lock()
    result._exchange_lock = threading.RLock()
    result._control_failure = None
    result._exchange_progress = None
    result._command_updated = time.monotonic()-.04
    result._timing = {'exchange_s': .003}
    result.mode_survey = {2: {'TIMEOUT': 8000, 'CTRL_MODE': 2}}
    result.enable_auto_recovery = False
    result.motor_list = [(1, 'DM4340'), (2, 'DM4340'), (3, 'DM4340')]
    result.channel = 'fake'
    result.motor_offset = np.zeros(3)
    result.motor_direction = np.ones(3)
    result.position_speed_motor_ids = {1, 2, 3}
    result.commands = [MotorCmd(vel=2.) for _ in range(3)]
    result._rate_recorder = nullcontext()
    result._report_interval = 30.
    result.same_bus_device_driver = None
    result.running = True
    return result


def test_worker_records_partial_exchange_and_stops_without_recovery_or_power_calls():
    result = chain()
    calls = []

    def exchange(**command):
        calls.append(command['motor_id'])
        if command['motor_id'] == 2:
            return parser().parse_recv_message(fault_packet(), 'DM4340')
        return SimpleNamespace(error_code='0x1', timestamp=time.time())

    # Deliberately supply no motor_on/off/clean_error methods: fault diagnostics
    # and the fatal path must never call them.
    result.motor_interface = SimpleNamespace(set_control=exchange, motor_modes={2: 'POS_VEL'})
    with pytest.raises(MotorFaultError):
        result._set_torques_and_update_state()
    assert not result.running
    assert calls == [1, 2]
    failure = result.get_control_failure()
    assert failure['exception_type'] == 'MotorFaultError'
    assert failure['native_fault']['feedback']['error_code'] == '0xd'
    assert failure['exchange']['motor_id'] == 2
    assert [row['motor_id'] for row in failure['exchange']['completed']] == [1]
    assert failure['mode_survey'][2]['TIMEOUT'] == 8000
    assert failure['command_age_s'] >= .04
    assert failure['cycle_period_s'] >= 0
    assert failure['exchange_elapsed_s'] >= 0
    assert not failure['automatic_recovery_enabled']
    json.dumps(failure, allow_nan=False)


def test_first_failure_is_retained_and_getter_does_not_expose_mutable_evidence():
    result = chain()
    result.motor_interface = SimpleNamespace(motor_modes={2: 'POS_VEL'})
    assert result.get_control_failure() is None
    now = time.monotonic()
    result._record_control_failure(RuntimeError('first'), cycle_started=now, last_cycle=now-.01)
    first = result.get_control_failure()
    result.mode_survey[2]['TIMEOUT'] = 0
    first['mode_survey'][2]['TIMEOUT'] = 123
    result._record_control_failure(RuntimeError('second'), cycle_started=now, last_cycle=now)
    retained = result.get_control_failure()
    assert retained['message'] == 'first'
    assert retained['mode_survey'][2]['TIMEOUT'] == 8000
    assert retained['cycle_period_s'] == pytest.approx(.01)


def test_stale_fault_frame_is_rejected_but_fresh_native_fault_remains_a_fault():
    interface = parser()
    packet = fault_packet()
    assert not interface._response_matches_request(packet, 2, packet.timestamp+.001)
    assert interface._response_matches_request(packet, 2, packet.timestamp-.001)
    with pytest.raises(MotorFaultError):
        interface.parse_recv_message(packet, 'DM4340')
