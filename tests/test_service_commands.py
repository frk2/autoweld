import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError

import pytest

from autoweld.service_commands import CommandRejected, ServiceAction, ServiceCommandBridge


def test_worker_waits_and_only_main_owner_executes_detached_payload():
    owner = threading.get_ident()
    executed = []
    bridge = ServiceCommandBridge(lambda action, payload: executed.append((threading.get_ident(), action, dict(payload))) or 'ok',
                                  lambda context, generation: context == 'scan' and generation == 4)
    payload = {'pose': [[1., 2., 3.]]}
    future = bridge.submit('move_pose', context='scan', generation=4, payload=payload)
    payload['pose'][0][0] = 999
    assert not executed and not future.done()
    with ThreadPoolExecutor(max_workers=1) as pool:
        error = pool.submit(bridge.drain).exception(timeout=1)
        assert isinstance(error, RuntimeError)
        waiter = pool.submit(future.result, 1.)
        assert bridge.drain() == 1
        assert waiter.result(timeout=1) == 'ok'
    assert executed == [(owner, ServiceAction.MOVE_POSE, {'pose': [[1., 2., 3.]]})]
    assert bridge.state()['executed'] == 1


def test_generation_and_cancel_guards_prevent_all_late_actions():
    executions = []
    generation = [2]
    bridge = ServiceCommandBridge(lambda *args: executions.append(args), lambda context, gen: gen == generation[0])
    changed = bridge.submit('start_path', context='scan', generation=2, payload={'plan': object()})
    generation[0] = 3
    invalidated = bridge.submit('move_saved_pose', context='scan', generation=3, payload={'name': 'Tube'})
    bridge.invalidate('scan', through_generation=3)
    cancelled = bridge.submit('hold', context='other', generation=3, cancelled=lambda: True)
    bridge.drain()
    with pytest.raises(CommandRejected):
        changed.result()
    with pytest.raises(CommandRejected):
        invalidated.result()
    with pytest.raises(CancelledError):
        cancelled.result()
    assert not executions


def test_expired_work_never_executes_and_queue_is_bounded(monkeypatch):
    clock = [10.]
    monkeypatch.setattr('autoweld.service_commands.time.monotonic', lambda: clock[0])
    executions = []
    bridge = ServiceCommandBridge(lambda *args: executions.append(args), lambda *args: True, max_pending=1)
    expired = bridge.submit('camera', context='scan', generation=0, timeout_s=.5)
    full = bridge.submit('hold', context='scan', generation=0)
    with pytest.raises(CommandRejected, match='full'):
        full.result()
    clock[0] += 1.
    bridge.drain()
    with pytest.raises(TimeoutError, match='expired'):
        expired.result()
    assert executions == []


def test_close_completes_pending_and_callbacks_can_inspect_bridge_without_deadlock():
    bridge = ServiceCommandBridge(lambda *args: None, lambda *args: True)
    future = bridge.submit('disable_at_rest', context='scan', generation=0)
    states = []
    future.add_done_callback(lambda f: states.append(bridge.state()))
    bridge.close()
    with pytest.raises(CommandRejected, match='closed'):
        future.result()
    assert states[0]['closed']
    rejected = bridge.submit('hold', context='scan', generation=0)
    with pytest.raises(CommandRejected, match='closed'):
        rejected.result()


def test_execution_exception_completes_future_and_later_commands_can_run():
    def execute(action, payload):
        if action is ServiceAction.MOVE_POSE:
            raise ValueError('installed collision')
        return 'held'

    bridge = ServiceCommandBridge(execute, lambda *args: True)
    bad = bridge.submit('move_pose', context='scan', generation=0)
    good = bridge.submit('hold', context='scan', generation=0)
    assert bridge.drain(max_commands=2) == 2
    with pytest.raises(ValueError, match='collision'):
        bad.result()
    assert good.result() == 'held'
    assert bridge.state()['rejected'] == 1


def test_worker_call_timeout_cancels_queued_action():
    executed = []
    bridge = ServiceCommandBridge(lambda *args: executed.append(args), lambda *args: True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(bridge.call, 'move_pose', context='scan', generation=0, timeout_s=.01)
        with pytest.raises(TimeoutError):
            future.result(timeout=1)
    bridge.drain()
    assert executed == []
    with pytest.raises(RuntimeError, match='never wait'):
        bridge.call('hold', context='scan', generation=0)


def test_start_path_keeps_artifact_and_state_never_exposes_dense_payload():
    artifact = object()
    seen = []
    bridge = ServiceCommandBridge(lambda action, payload: seen.append(payload['plan']), lambda *args: True)
    future = bridge.submit(ServiceAction.START_PATH, context='scan', generation=0, payload={'plan': artifact})
    assert set(bridge.state()) == {'pending', 'closed', 'executed', 'rejected', 'last_action', 'last_error'}
    bridge.drain()
    assert future.done() and seen == [artifact]


@pytest.mark.parametrize('kwargs', [dict(action='enable'), dict(action='hold', generation=-1),
                                    dict(action='hold', generation=True), dict(action='hold', context=''),
                                    dict(action='hold', timeout_s=float('nan')),
                                    dict(action='hold', timeout_s=121)])
def test_unknown_actions_and_invalid_context_cannot_enqueue(kwargs):
    bridge = ServiceCommandBridge(lambda *args: None, lambda *args: True)
    args = {'action': 'hold', 'context': 'scan', 'generation': 0}
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        bridge.submit(**args)
    assert bridge.state()['pending'] == 0
