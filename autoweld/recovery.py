"""Explicit, bounded recovery admission. SDK owns every hardware operation.

No worker, retry loop, model, socket, or CAN transport is created here. A failed
intent cannot later enable: the same deadline/cancellation/host guard is passed
to the disabled SDK helper and to the ordinary guarded connection factory.
"""
import threading
import time
from dataclasses import asdict, is_dataclass

import numpy as np

MOTORS = list(range(1, 7))
TIMEOUT_MS = 4000
TIMEOUT_TICK_US = 50
TIMEOUT_REGISTER_VALUE = TIMEOUT_MS * 1000 // TIMEOUT_TICK_US


def _diagnostic_json(value, depth=0):
    """Detach cached diagnostic data; never traverse SDK/device objects."""
    if depth > 12:
        return '<diagnostic depth limit>'
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _diagnostic_json(value.tolist(), depth+1)
    if isinstance(value, BaseException):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _diagnostic_json(asdict(value), depth+1)
    if isinstance(value, dict):
        return {str(key): _diagnostic_json(item, depth+1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_diagnostic_json(item, depth+1) for item in value]
    return f'<{type(value).__name__}>'


def enable_failure_diagnostics(error, chain=None):
    """Read exception/cache only. This does not survey or verify live hardware."""
    result = {'recovery_cleanup_error': _diagnostic_json(getattr(error, 'recovery_cleanup_error', None)),
        'recovery_cleanup_verified': getattr(error, 'recovery_cleanup_verified', None),
        'enable_failure': _diagnostic_json(getattr(error, 'enable_failure', None)),
        'native_last_probe': _diagnostic_json(getattr(error, 'native_sample', None))}
    if chain is None:
        return _diagnostic_json(result)
    cached = None
    accessor = getattr(chain, 'get_enable_failure_diagnostics', None)
    if callable(accessor):
        try:
            cached = _diagnostic_json(accessor())
        except Exception as diagnostic_error:
            result['diagnostic_error'] = str(diagnostic_error)
    result['partial_chain'] = {'cached_only': True, 'current_hardware_verified': False,
        'running': getattr(chain, 'running', None), 'powered': getattr(chain, 'powered', None),
        'power_unknown': getattr(chain, 'power_unknown', None),
        'enabled_motor_ids': _diagnostic_json(getattr(chain, '_startup_enabled_ids', None)),
        'inflight_motor_id': getattr(chain, '_startup_inflight_id', None),
        'enable_failure': cached}
    if result['native_last_probe'] is None and isinstance(cached, dict):
        result['native_last_probe'] = cached.get('owner_release_sample') or cached.get('last_sample')
    return _diagnostic_json(result)


class RecoveryBlocked(RuntimeError):
    pass


def validate_disabled_sample(sample, rest, bounds, *, now, allow_timeout=True):
    """Reject cached/normalized/enabled/foreign feedback before any mutation."""
    rest, bounds = np.asarray(rest, float), np.asarray(bounds, float)
    if (rest.shape != (6,) or bounds.shape != (6, 2) or not np.isfinite(np.r_[rest, bounds.ravel()]).all()
            or np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(rest < bounds[:, 0]) or np.any(rest > bounds[:, 1])):
        raise RecoveryBlocked('Captured rest and all six installed bounds must be valid; no bounds expansion is permitted')
    if sample.get('motor_ids') != MOTORS:
        raise RecoveryBlocked('Recovery needs exactly motor IDs 1–6')
    q, raw, velocity = (np.asarray(sample.get(key), float) for key in
                        ('joint_rad', 'sdk_feedback_position_rad', 'velocity_rad_s'))
    mapping = sample.get('mapping') or {}
    if (mapping.get('offsets_rad') != [0.] * 6 or mapping.get('directions') != [1] * 6
            or mapping.get('boot_normalization') != 'none'):
        raise RecoveryBlocked('Recovery requires the fixed zero-offset encoder branch without boot normalization')
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in (q, raw, velocity)):
        raise RecoveryBlocked('Recovery needs finite six-axis measured position and velocity')
    if np.max(abs(q-raw)) > 1e-12:
        raise RecoveryBlocked('Recovery joint coordinates differ from the fixed raw SDK encoder branch')
    statuses = sample.get('status')
    allowed = {'0x0', '0xd'} if allow_timeout else {'0x0'}
    if not isinstance(statuses, list) or len(statuses) != 6 or any(status not in allowed for status in statuses):
        raise RecoveryBlocked('Recovery permits only disabled status 0x0 or communication latch 0xd; enabled/other faults refused')
    stamp = sample.get('physical_timestamp')
    if stamp is None or not np.isfinite(stamp) or not 0 <= now-stamp <= .5:
        raise RecoveryBlocked('Recovery requires fresh physical SDK feedback, not a saved/cached display')
    for register, key in (('p_m', 'native_p_m'), ('TIMEOUT', 'native_timeout'), ('CTRL_MODE', 'native_ctrl_mode')):
        values = sample.get(key)
        if not isinstance(values, list) or len(values) != 6:
            raise RecoveryBlocked(f'Missing six-axis native {register} proof')
        for mid, item in zip(MOTORS, values, strict=True):
            if (item.get('motor_id') != mid or item.get('register') != register
                    or not np.isfinite([item.get('value', float('nan')), item.get('timestamp', float('nan'))]).all()
                    or not 0 <= now-item['timestamp'] <= .5):
                raise RecoveryBlocked(f'Foreign, stale or invalid native {register} proof')
        if register == 'p_m' and np.max(abs(raw-np.array([item['value'] for item in values]))) > .02:
            raise RecoveryBlocked('Packed SDK position and native p_m disagree by more than 0.02 rad')
        if register == 'CTRL_MODE' and any(item['value'] not in (1, 2) for item in values):
            raise RecoveryBlocked('Native motor modes are unknown')
    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        raise RecoveryBlocked('Measured position is outside strict installed joint bounds')
    if np.max(abs(q-rest)) > .15:
        raise RecoveryBlocked('Measured physical pose is farther than 0.15 rad per joint from captured supported rest')
    if np.max(abs(velocity)) > .05:
        raise RecoveryBlocked('Measured arm is not quiet: speed exceeds 0.05 rad/s')
    temperatures = sample.get('temperatures') or {}
    for key in ('mos_c', 'rotor_c'):
        values = np.asarray(temperatures.get(key), float)
        if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= 60):
            raise RecoveryBlocked('Recovery requires healthy six-axis temperatures below 60 C')
    return q.copy()


class ControlRecovery:
    def __init__(self, controller, reader_factory, host_guard, clear_ownership, *,
                 cancelled=lambda intent: False, busy=lambda: False, collision=None,
                 clock=time.monotonic, wall_clock=time.time):
        self.ctrl, self.reader_factory, self.host_guard = controller, reader_factory, host_guard
        self.clear_ownership, self.cancelled, self.busy = clear_ownership, cancelled, busy
        self.collision, self.clock, self.wall_clock = collision, clock, wall_clock
        self._lock = threading.Lock()
        self._state = {'phase': 'idle', 'active': False, 'retryable': False, 'error': None}
        self.retained_chain = None
        self.retained_reader = None

    def state(self):
        return dict(self._state)

    def _phase(self, phase, **fields):
        self._state.update(phase=phase, **fields)

    @staticmethod
    def _threads_stopped(robot):
        chain = getattr(robot, 'motor_chain', None)
        if chain is None:
            return True
        worker, server = getattr(chain, '_control_thread', None), getattr(robot, '_server_thread', None)
        return (getattr(chain, 'running', None) is False and (worker is None or not worker.is_alive())
                and server is not None and not server.is_alive())

    def _release_retained_owner(self, chain, guard, deadline):
        """A new explicit request may release only a freshly proven disabled owner."""
        def stopped():
            if getattr(chain, 'running', None) is not False:
                raise RecoveryBlocked('Retained CAN owner is running or its writer state is unknown')
            worker = getattr(chain, '_control_thread', None)
            if worker is not None and worker.is_alive():
                raise RecoveryBlocked('Retained CAN writer is still alive; owner cannot be released')
            robot_chain = getattr(self.ctrl.robot, 'motor_chain', None)
            if robot_chain is chain:
                server = getattr(self.ctrl.robot, '_server_thread', None)
                if server is None or server.is_alive():
                    raise RecoveryBlocked('Retained robot server writer is alive or unverified; owner cannot be released')
        stopped()
        release = getattr(chain, 'release_disabled_recovery_owner', None)
        if not callable(release):
            raise RecoveryBlocked('Retained partial SDK owner lacks verified same-interface disabled release support')

        approved = []

        def before_release(sample):
            guard()
            stopped()
            q = validate_disabled_sample(sample, self.ctrl.rest[:6], self.ctrl.limits, now=self.wall_clock())
            if self.ctrl.cooldown_active and max(sample['temperatures']['mos_c']+sample['temperatures']['rotor_c']) >= 55:
                raise RecoveryBlocked('Latched cooldown requires temperatures below 55 C before retained-owner release')
            if self.collision is not None and self.collision(np.r_[q, 0.]):
                raise RecoveryBlocked('Retained owner feedback fails installed collision checks')
            approved[:] = [_diagnostic_json(sample)]

        guard()
        proof = release(rest_q=self.ctrl.rest[:6].copy(), joint_limits=self.ctrl.limits.copy(),
            near_rest_tolerance_rad=.15, deadline=deadline, check_intent=guard, before_release=before_release)
        if (proof.get('released') is not True or proof.get('socket_closed') is not True
                or proof.get('can_worker_stopped') is not True or proof.get('motor_power_changed') is not False or not approved):
            raise RecoveryBlocked('Retained SDK owner did not provide verified write-free socket release')
        # The SDK has just closed the proven disabled socket. Clear its retained
        # reference even if cancellation/expiry arrives immediately afterward.
        self.retained_chain = None
        self._state['retained_owner_release'] = _diagnostic_json(proof)
        q = np.asarray(approved[0]['joint_rad'], float)
        self.ctrl.closed = True
        self.ctrl.disabled_q = np.r_[q, self.ctrl.gripper_target]
        self.ctrl.cached_rest_display = False
        self.ctrl.mode = 'Disabled; recovery qualification pending'

    def recover(self, requested_monotonic=None):
        if not self._lock.acquire(blocking=False):
            return {'phase': 'busy', 'active': True, 'retryable': True, 'error': 'A recovery request is already executing'}
        reader = candidate = None
        old_robot = self.ctrl.robot
        intent = self.clock() if requested_monotonic is None else requested_monotonic
        self._state = {'phase': 'admission', 'active': True, 'retryable': False, 'error': None,
                       'changed': False, 'requested_monotonic': intent, 'host_pressure_bypassed': True}
        old_stopped = False
        enable_started = False
        deadline = intent+10. if isinstance(intent, (int, float)) else float('nan')

        def check_intent():
            if type(intent) not in (int, float) or not np.isfinite([intent, deadline]).all() or intent > self.clock() or self.clock() >= deadline:
                raise RecoveryBlocked('Recovery intent expired or is invalid; submit a new explicit request')
            if self.cancelled(intent):
                raise RecoveryBlocked('Recovery cancelled by pending Shutdown, Hold or cancellation request')

        def guard():
            check_intent()
            # Explicit operator recovery bypasses host-pressure admission.
            # Physical qualification, original deadline, cancellation and
            # stopped-owner checks still apply at every SDK mutation.
            if old_stopped and not self._threads_stopped(old_robot):
                raise RecoveryBlocked('An old SDK writer resumed; recovery refused')

        def retain(robot):
            nonlocal candidate
            candidate = robot

        try:
            check_intent()
            if self.retained_reader is not None:
                raise RecoveryBlocked('Previous disabled SDK reader could not close; retain sole owner for diagnosis')
            retained_released = False
            if self.retained_chain is not None:
                self._phase('qualifying_retained_owner')
                self._release_retained_owner(self.retained_chain, guard, deadline)
                retained_released = True
            sdk = self.ctrl.sdk_status()
            if sdk.get('control_alive') and sdk.get('feedback_fresh'):
                if (self.ctrl.mode != 'Hold' or self.ctrl.motion_plan is not None or self.ctrl.disable_pending
                        or self.ctrl.shutdown_requested or self.ctrl.motion_fault() or self.busy()):
                    raise RecoveryBlocked('Healthy control is busy or outside idle Hold; recovery will not interrupt it')
                self._phase('already_controlled', active=False, retryable=False, error=None)
                return self.state()
            guard()
            self._phase('stopping_failed_owner')
            chain = getattr(old_robot, 'motor_chain', None)
            if retained_released:
                if not self._threads_stopped(old_robot):
                    raise RecoveryBlocked('Released SDK owner still has an active writer')
            elif chain is not None:
                stopper = getattr(old_robot, 'stop_for_disabled_recovery', None)
                if stopper is None:
                    raise RecoveryBlocked('SDK write-free failed-owner stop is unavailable')
                stopper(timeout_s=min(1., max(.001, deadline-self.clock())))
                if not self._threads_stopped(old_robot):
                    raise RecoveryBlocked('Both the old CAN worker and robot writer must be stopped')
            elif not self.ctrl.closed or not old_robot.get_robot_info().get('sim'):
                raise RecoveryBlocked('Missing proof of a stopped SDK owner or explicit disabled display')
            old_stopped = True
            guard()
            if self.reader_factory is None:
                raise RecoveryBlocked('Disabled SDK recovery reader is unavailable')
            self._phase('surveying_disabled')
            reader = self.reader_factory()
            first = reader.survey()
            q = validate_disabled_sample(first, self.ctrl.rest[:6], self.ctrl.limits, now=self.wall_clock())
            if self.collision is not None and self.collision(np.r_[q, 0.]):
                raise RecoveryBlocked('Measured near-rest configuration fails installed collision checks')
            if self.ctrl.cooldown_active and max(first['temperatures']['mos_c']+first['temperatures']['rotor_c']) >= 55:
                raise RecoveryBlocked('Latched cooldown requires fresh motor temperatures below 55 C before recovery')
            guard()
            self._phase('proving_and_clearing_timeout')
            proof = reader.clear_timeout_latches(rest_q=self.ctrl.rest[:6].copy(),
                joint_limits=self.ctrl.limits.copy(), near_rest_tolerance_rad=.15,
                before_write=guard, deadline=deadline)
            after = proof['after']
            q = validate_disabled_sample(after, self.ctrl.rest[:6], self.ctrl.limits,
                                        now=self.wall_clock(), allow_timeout=False)
            dwell = after['timestamp']-proof['before']['timestamp']
            if not np.isfinite(dwell) or dwell < 1.:
                raise RecoveryBlocked('SDK did not provide a full one-second quiet verification')
            readbacks = after.get('native_timeout')
            if (proof.get('timeout_ms') != TIMEOUT_MS or proof.get('timeout_register_ticks', proof.get('timeout_register_value')) != TIMEOUT_REGISTER_VALUE
                    or proof.get('enabled') is not False
                    or proof.get('flash_written') is not False
                    or any(row['value'] != TIMEOUT_REGISTER_VALUE for row in readbacks)):
                raise RecoveryBlocked('Recovery must verify all six TIMEOUT raw 80000 (50 µs ticks = 4000 ms), disabled, without Flash')
            if self.collision is not None and self.collision(np.r_[q, 0.]):
                raise RecoveryBlocked('Final disabled configuration fails installed collision checks')
            if self.ctrl.cooldown_active and max(after['temperatures']['mos_c']+after['temperatures']['rotor_c']) >= 55:
                raise RecoveryBlocked('Latched cooldown still requires temperatures below 55 C')
            converted = [{'motor_id': row['motor_id'], 'raw_value': row['value'],
                'timeout_ms': row['value']*TIMEOUT_TICK_US/1000., 'timestamp': row['timestamp']} for row in readbacks]
            self._state.update(proof=proof, timeout_readbacks=readbacks, timeout_readbacks_ms=converted,
                timeout_ms=TIMEOUT_MS, timeout_register_ticks=TIMEOUT_REGISTER_VALUE,
                timeout_register_value=TIMEOUT_REGISTER_VALUE, timeout_register_tick_us=TIMEOUT_TICK_US,
                measured_q_rad=q.tolist(), quiet_verified_s=dwell)
            reader.close()
            reader = None
            guard()
            self._phase('releasing_stale_ownership')
            self.clear_ownership()
            self._state['ownership_released'] = True
            # No hardware fault is acknowledged here. The old SDK instance stays
            # stopped; the normal factory creates a fresh measured Hold owner.
            self.ctrl.closed = True
            self.ctrl.disabled_q = np.r_[q, self.ctrl.gripper_target]
            self.ctrl.target = self.ctrl.disabled_q.copy()
            self.ctrl.cached_rest_display = False
            self.ctrl.mode = 'Disabled'
            self.ctrl.rest_verified = False
            self.ctrl.settled_since = None
            self.ctrl.feedback_warning = None
            self._phase('enabling_measured_hold')
            original_factory = self.ctrl.robot_factory
            if original_factory is None:
                raise RecoveryBlocked('Normal SDK reconnect factory is unavailable')

            def factory(*args, **kwargs):
                guard()
                return original_factory(*args, **kwargs, admission_guard=guard, enable_deadline=deadline,
                    on_created=retain, recovery_rest_q=self.ctrl.rest[:6].copy(),
                    recovery_joint_limits=self.ctrl.limits.copy(), recovery_near_rest_tolerance_rad=.15)

            self.ctrl.robot_factory = factory
            enable_started = True
            try:
                guard()
                self.ctrl.enable()
            finally:
                self.ctrl.robot_factory = original_factory
            guard()
            sdk = self.ctrl.sdk_status()
            if not sdk.get('control_alive') or not sdk.get('feedback_fresh') or self.ctrl.mode != 'Hold':
                raise RecoveryBlocked('New SDK did not establish fresh, live measured Hold; retained owner requires diagnosis')
            self._phase('holding', active=False, retryable=False, error=None, changed=True)
            return self.state()
        except Exception as error:
            candidate = getattr(error, 'recovery_robot', None) or candidate
            retained = getattr(error, 'recovery_chain', None)
            if retained is None:
                retained = self.retained_chain
            diagnostic_chain = retained if retained is not None else getattr(candidate, 'motor_chain', None)
            self._state.update(enable_failure_diagnostics(error, diagnostic_chain))
            if retained is not None and getattr(error, 'recovery_cleanup_verified', False) is not True:
                self.retained_chain = retained
                self.ctrl.closed = False
                self.ctrl.cached_rest_display = False
                self.ctrl.mode = 'Recovery blocked; partial SDK owner retained'
            if candidate is not None and self.ctrl.robot is not candidate:
                # A constructed owner can already be holding even if later
                # configuration failed. Retain it; never close or drop torque.
                self.ctrl.robot = candidate
                self.ctrl.closed = False
                self.ctrl.cached_rest_display = False
                self.ctrl.mode = 'Recovery blocked; SDK owner retained'
            self._phase('blocked', active=False, retryable=True, error=str(error),
                        blocked_phase=self._state['phase'], enable_attempted=enable_started,
                        partial_owner_retained=retained is not None or candidate is not None)
            return self.state()
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception as error:
                    self.retained_reader = reader
                    self._state['reader_close_error'] = str(error)
            self._lock.release()
