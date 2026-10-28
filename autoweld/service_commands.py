"""Bounded worker-to-service commands; device calls execute on the main owner.

A worker waits on a Future or call(); it never invokes Controller or camera APIs.
Create this bridge on the service main thread and drain it there alongside tick.
The execute callback must be short: planning belongs in private-data workers.
"""

import copy
import queue
import threading
import time
from concurrent.futures import Future, InvalidStateError, TimeoutError
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Callable, Mapping


class ServiceAction(StrEnum):
    MOVE_SAVED_POSE = 'move_saved_pose'
    MOVE_POSE = 'move_pose'
    HOLD = 'hold'
    START_PATH = 'start_path'
    CAMERA = 'camera'
    DISABLE_AT_REST = 'disable_at_rest'


class CommandRejected(RuntimeError):
    """A queued action lost its context/admission before any execution."""


@dataclass(frozen=True)
class _Command:
    action: ServiceAction
    context: str
    generation: int
    payload: Mapping
    expires: float
    cancelled: Callable[[], bool] | None
    future: Future


class ServiceCommandBridge:
    """Closed action vocabulary, bounded queue, expiring context-bound Futures.

    execute(action,payload) and valid_context(context,generation) run only while
    drain() is executing on the creator/main thread. External shutdown/source
    changes should invalidate their context generation and signal cancellation;
    admission is checked immediately before executing each action. A running
    callback retains the Controller's own shutdown/health/guard checks.
    """

    def __init__(self, execute, valid_context, *, max_pending=32):
        if not callable(execute) or not callable(valid_context):
            raise TypeError('Execution and context validation callbacks are required')
        if type(max_pending) is not int or not 1 <= max_pending <= 256:
            raise ValueError('Queue capacity must be an integer from 1 to 256')
        self._execute = execute
        self._valid_context = valid_context
        self._owner = threading.get_ident()
        self._queue = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._closed = False
        self._invalid = {}
        self._executed = self._rejected = 0
        self._last_action = self._last_error = None

    def submit(self, action, *, context, generation, payload=None, cancelled=None, timeout_s=30.):
        """Queue without blocking. START_PATH carries a prevalidated plan object.

        Ordinary payloads are detached from caller mutation. A START_PATH artifact
        is retained by reference to avoid copying dense paths on worker threads;
        it must remain frozen and Controller rechecks its digest before cloning.
        No image/point payload is exposed by bridge state().
        """
        action = ServiceAction(action)
        if not isinstance(context, str) or not context or len(context) > 128:
            raise ValueError('A nonempty bounded operation context is required')
        if type(generation) is not int or generation < 0:
            raise ValueError('Context generation must be a nonnegative integer')
        if not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 120:
            raise ValueError('Command expiry must be in (0,120] seconds')
        if cancelled is not None and not callable(cancelled):
            raise TypeError('Cancellation guard must be callable')
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError('Command payload must be a mapping')
        values = dict(payload or {})
        if action is not ServiceAction.START_PATH:
            values = copy.deepcopy(values)
        command = _Command(action, context, generation, MappingProxyType(values),
                           time.monotonic() + timeout_s, cancelled, Future())
        rejection = None
        with self._lock:
            if self._closed:
                rejection = 'Service command bridge is closed'
            else:
                try:
                    self._queue.put_nowait(command)
                except queue.Full:
                    rejection = 'Service command queue is full'
        if rejection is not None:
            command.future.set_exception(CommandRejected(rejection))
        return command.future

    def call(self, action, *, timeout_s=30., **kwargs):
        """Worker convenience: bounded wait also cancels a queued late action."""
        if threading.get_ident() == self._owner:
            raise RuntimeError('The service owner must drain commands, never wait on its own bridge')
        future = self.submit(action, timeout_s=timeout_s, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError:
            future.cancel()
            raise

    def invalidate(self, context, *, through_generation):
        """Reject queued old generations without touching devices or active moves."""
        if not isinstance(context, str) or type(through_generation) is not int or through_generation < 0:
            raise ValueError('Invalid context generation')
        with self._lock:
            self._invalid[context] = max(self._invalid.get(context, -1), through_generation)

    def drain(self, *, max_commands=8, max_wall_s=.02):
        """Main owner only; budget excludes an unpreemptible individual callback."""
        if threading.get_ident() != self._owner:
            raise RuntimeError('Only the service main owner may execute queued commands')
        if type(max_commands) is not int or max_commands < 1:
            raise ValueError('Drain count must be a positive integer')
        if not isinstance(max_wall_s, (int, float)) or not 0 < max_wall_s <= 1:
            raise ValueError('Drain wall budget must be in (0,1] seconds')
        end = time.monotonic() + max_wall_s
        processed = 0
        while processed < max_commands:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            future = command.future
            if future.cancelled():
                continue
            try:
                with self._lock:
                    invalid = self._closed or command.generation <= self._invalid.get(command.context, -1)
                if invalid or not self._valid_context(command.context, command.generation):
                    raise CommandRejected('Operation context changed before command execution')
                if time.monotonic() >= command.expires:
                    raise TimeoutError('Queued service command expired; no action executed')
                if command.cancelled is not None and command.cancelled():
                    future.cancel()
                    with self._lock:
                        self._rejected += 1
                    continue
                if not future.set_running_or_notify_cancel():
                    continue
                result = self._execute(command.action, command.payload)
            except BaseException as error:
                # KeyboardInterrupt/SystemExit still reach the service's dedicated
                # shutdown path; do not strand the waiting worker's Future.
                if not future.done():
                    try:
                        future.set_exception(error)
                    except InvalidStateError:
                        if not future.cancelled():
                            raise
                with self._lock:
                    self._rejected += 1
                    self._last_action, self._last_error = command.action.value, str(error)
                if not isinstance(error, Exception):
                    raise
            else:
                future.set_result(result)
                with self._lock:
                    self._executed += 1
                    self._last_action, self._last_error = command.action.value, None
            if time.monotonic() >= end:
                break
        return processed

    def close(self):
        """Invalidate queued work; dedicated service shutdown remains its own path."""
        pending = []
        with self._lock:
            self._closed = True
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
        for command in pending:
            if not command.future.done():
                try:
                    command.future.set_exception(CommandRejected('Service command bridge closed before execution'))
                except InvalidStateError:
                    if not command.future.cancelled():
                        raise

    def state(self):
        with self._lock:
            return {'pending': self._queue.qsize(), 'closed': self._closed,
                    'executed': self._executed, 'rejected': self._rejected,
                    'last_action': self._last_action, 'last_error': self._last_error}
