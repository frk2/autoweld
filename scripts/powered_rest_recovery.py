"""One-shot, operator-authorized powered recovery of a stuck AutoWeld owner.

Prepare powered-rest-path.npz / powered-rest-preflight.json first. No enable,
mode, zero, error-clear or register-write operation exists here. The old owner
is paused, never resumed after a control command, and killed only at verified
rest while this process continues SDK hold. An execution failure retains a
fixed powered hold instead of dropping the CAN socket away from rest.
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from i2rt.motor_config_tool.dm_motor_registers import read_register_sample
from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.robots.waypoint_motion import JointPath

from autoweld.powered_recovery import MOTORS, RecoveryController


def identity(pid):
    root = Path(f'/proc/{pid}')
    command = (root/'cmdline').read_bytes().split(b'\0')
    if not all(part in command for part in (b'autoweld.app', b'--live', b'8082', b'runtime/agents-live')):
        raise RuntimeError('PID is not the expected live AutoWeld owner')
    return (root/'stat').read_text().split()[21]


def stopped(pid):
    states = [p.read_text() for p in Path(f'/proc/{pid}/task').glob('*/status')]
    return bool(states) and all('State:\tT' in value for value in states)


def main(pid, evidence=Path('evidence/auto-calibration-live'), previous_guardian=None):
    evidence = Path(evidence)
    start_identity = identity(pid)
    if previous_guardian is not None:
        command = Path(f'/proc/{previous_guardian}/cmdline').read_bytes().split(b'\0')
        if b'scripts/powered_rest_recovery.py' not in command or str(pid).encode() not in command:
            raise RuntimeError('Previous guardian PID does not match this recovery')
        if not stopped(pid):
            raise RuntimeError('Original owner must remain stopped before replacing its guardian')
    pause_pid = previous_guardian if previous_guardian is not None else pid
    report = json.loads((evidence/'powered-rest-preflight.json').read_text())
    restfile = Path('runtime/agents-live/rest.json')
    if hashlib.sha256(restfile.read_bytes()).hexdigest() != report['rest_sha256']:
        raise RuntimeError('Captured rest changed after preflight')
    model = Path('runtime/models/receiver-camera.xml')
    if hashlib.sha256(model.read_bytes()).hexdigest() != report['model_sha256']:
        raise RuntimeError('Robot model changed after preflight')
    with np.load(evidence/'powered-rest-path.npz') as data:
        path = JointPath(data['knots'], data['positions'], float(data['duration']))
        bounds = data['bounds'].copy()
    if not np.allclose(path.positions[0, :6], report['start_rad'], atol=1e-12, rtol=0):
        raise RuntimeError('Preflight start does not match path')
    if not np.allclose(path.positions[-1, :6], report['rest_rad'], atol=1e-12, rtol=0):
        raise RuntimeError('Preflight rest does not match path')
    if report['collision_samples'] < 401 or report['planned_peaks']['joint_velocity'] > .12:
        raise RuntimeError('Missing conservative path preflight')
    policy = RecoveryController(path, started_at=time.monotonic(), speed_cap_rad_s=.12)
    interface = DMSingleMotorCanInterface(channel='can0', control_mode=ControlMode.POS_VEL,
                                        name='operator_powered_rest_recovery')
    lock = Path('/tmp/autoweld-can0.lock').open('a')
    paused = commands_started = False
    final_stage = None
    log = (evidence/'powered-recovery.jsonl').open('a', buffering=1)
    def emit(record):
        text = json.dumps(record, allow_nan=False)
        log.write(text+'\n')
        try:
            print(text, flush=True)
        except (BrokenPipeError, OSError):
            pass
    try:
        # Only read-only SDK requests occur while the old controller is active.
        survey = [dict(id=mid, mode=read_register_sample(interface, mid, 'CTRL_MODE')['value'],
                       timeout=read_register_sample(interface, mid, 'TIMEOUT')['value']) for mid, _ in MOTORS]
        if any(row['mode'] != 2 or row['timeout'] < 5000 for row in survey):
            raise RuntimeError('Need confirmed POS_VEL and at least 5 s native communication watchdog')
        frames = [interface.read_motor_status(mid, kind) for mid, kind in MOTORS]
        q = np.array([f.position for f in frames])
        if (any(f.error_code != '0x1' or not 0 <= time.time()-f.timestamp < .1 for f in frames)
                or max(abs(f.velocity) for f in frames) > .05
                or max(max(f.temperature_mos, f.temperature_rotor) for f in frames) >= 55
                or np.max(abs(q-policy.reference)) > .003
                or np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1])):
            raise RuntimeError('Fresh healthy quiet pose must match preflight before powered handoff')
        if identity(pid) != start_identity:
            raise RuntimeError('Old owner identity changed')
        emit(dict(event='preflight_passed', old_pid=pid, survey=survey, q_rad=q.tolist(),
                  duration_s=path.duration, motors_enabled=True, power_transition=False))
        pause_started = time.monotonic()
        os.kill(pause_pid, signal.SIGSTOP)
        paused = True
        while not stopped(pause_pid) or not stopped(pid):
            if time.monotonic()-pause_started > .2:
                raise RuntimeError('Old control threads did not all stop within 200 ms')
            time.sleep(.001)
        # No second active writer: every old thread is stopped before the first
        # powered command. Imports, path generation and native surveys are done.
        policy.last_now = time.monotonic()
        last_emit = last_stage = None
        first_exchange = True
        handled_resume = set()
        def interruption(signum, _frame):
            policy.fail(f'Operator signal {signum}; retaining fixed powered hold')
        signal.signal(signal.SIGINT, interruption)
        signal.signal(signal.SIGTERM, interruption)
        while True:
            cycle_started = time.monotonic()
            try:
                frames, failures = [], []
                commands_started = True
                for i, (mid, kind) in enumerate(MOTORS):
                    try:
                        frame = interface.set_control(mid, kind, float(policy.reference[i]), .12, 0., 0., 0.)
                        frames.append(frame)
                    except Exception as error:
                        failures.append(f'J{mid}: {error}')
                if first_exchange:
                    handoff_s = time.monotonic()-pause_started
                    emit(dict(event='powered_handoff_complete', gap_s=handoff_s, power_transition=False))
                    if handoff_s > .25:
                        policy.fail('Powered handoff exceeded 250 ms timing guard')
                    first_exchange = False
                if failures:
                    policy.fail('; '.join(failures))
                state = policy.step(time.monotonic(), frames, feedback_now=time.time())
                q = np.array([f.position for f in frames]) if len(frames) == 6 else None
                request_file = evidence/'powered-recovery-resume.json'
                if state['stage'] == 'failed_hold' and q is not None and request_file.exists():
                    request = json.loads(request_file.read_text())
                    request_id = request.get('id')
                    if request_id and request_id not in handled_resume:
                        handled_resume.add(request_id)
                        if request.get('action') != 'resume_preflight_rest':
                            raise ValueError('Unsupported recovery continuation action')
                        updated = json.loads((evidence/'powered-rest-preflight.json').read_text())
                        if (updated['rest_sha256'] != report['rest_sha256']
                                or updated['model_sha256'] != report['model_sha256']
                                or hashlib.sha256(restfile.read_bytes()).hexdigest() != updated['rest_sha256']
                                or hashlib.sha256(model.read_bytes()).hexdigest() != updated['model_sha256']
                                or updated['collision_samples'] < 401
                                or updated['planned_peaks']['joint_velocity'] > .12):
                            raise ValueError('Continuation must preserve rest, model and path guards')
                        with np.load(evidence/'powered-rest-path.npz') as data:
                            proposed = JointPath(data['knots'], data['positions'], float(data['duration']))
                            proposed_bounds = data['bounds'].copy()
                        if (not np.allclose(proposed.positions[-1, :6], policy.rest[:6], atol=1e-12, rtol=0)
                                or not np.allclose(proposed.positions[0, :6], updated['start_rad'], atol=1e-12, rtol=0)
                                or not np.allclose(proposed.positions[-1, :6], updated['rest_rad'], atol=1e-12, rtol=0)
                                or tuple(f.id for f in frames) != tuple(range(1, 7))
                                or not np.isfinite([[f.position, f.velocity, f.timestamp,
                                                    f.temperature_mos, f.temperature_rotor] for f in frames]).all()
                                or np.max(abs(proposed.positions[0, :6]-q)) > .003
                                or not np.array_equal(proposed_bounds, bounds)
                                or any(f.error_code != '0x1' or not 0 <= time.time()-f.timestamp < .1 for f in frames)
                                or max(abs(f.velocity) for f in frames) > .05
                                or max(max(f.temperature_mos, f.temperature_rotor) for f in frames) >= 55):
                            raise ValueError('Continuation needs fresh healthy quiet feedback matching its preflight')
                        policy = RecoveryController(proposed, started_at=time.monotonic(), speed_cap_rad_s=.12)
                        state = policy.snapshot()
                        emit(dict(event='explicit_preflight_continuation', request_id=request_id,
                                  duration_s=proposed.duration, power_transition=False))
                if q is not None and (np.any(q < bounds[:, 0]-.005) or np.any(q > bounds[:, 1]+.005)):
                    state = policy.fail('Measured joint position exceeded calibrated bounds')
                now = time.monotonic()
                if (state['stage'] != last_stage or last_emit is None or now-last_emit >= 2):
                    emit(dict(event='recovery_state', **state, q_rad=None if q is None else q.tolist(),
                              rest_error_deg=None if q is None else np.rad2deg(q-policy.rest[:6]).tolist(),
                              maximum_temperature_c=max((max(f.temperature_mos, f.temperature_rotor) for f in frames), default=None)))
                    last_emit, last_stage = now, state['stage']
                if state['rest_verified']:
                    if not stopped(pid) and final_stage is None:
                        policy.fail('Old process unexpectedly resumed; retain powered hold')
                    elif final_stage is None:
                        # The only process kill in this script is after fresh,
                        # stationary rest for 1 s, while this SDK still holds.
                        os.kill(pid, signal.SIGKILL)
                        if previous_guardian is not None:
                            os.kill(previous_guardian, signal.SIGKILL)
                        final_stage = 'awaiting_owner_lock'
                        emit(dict(event='old_owner_killed_at_verified_rest', q_rad=q.tolist()))
                    if final_stage == 'awaiting_owner_lock':
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            emit(dict(event='exclusive_owner_acquired_at_verified_rest'))
                            break
                        except BlockingIOError:
                            pass  # Continue sending the same powered rest hold.
            except BaseException as error:
                policy.fail(f'Recovery exception; retain fixed powered hold: {error}')
            time.sleep(max(0., .004-(time.monotonic()-cycle_started)))
        # No communication/motor OFF operation was used during transit.
        errors = []
        for mid, _ in MOTORS:
            try:
                interface.motor_off(mid)
            except Exception as error:
                errors.append(dict(id=mid, error=str(error)))
        samples = []
        for _ in range(3):
            rows = [asdict(interface.read_motor_status(mid, kind, quiet_disabled=True)) for mid, kind in MOTORS]
            if any(row['error_code'] != '0x0' or not 0 <= time.time()-row['timestamp'] <= .1 for row in rows):
                raise RuntimeError('Fresh all-six disabled state not confirmed; do not enable')
            samples.append(rows)
            time.sleep(.05)
        result = dict(event='disabled_at_verified_rest', all_six_disabled=True,
                      rest_rad=policy.rest[:6].tolist(), samples=samples, motor_off_reply_errors=errors)
        (evidence/'powered-recovery-result.json').write_text(json.dumps(result, indent=2))
        emit(result)
    finally:
        if commands_started and not policy.snapshot()['rest_verified']:
            # Defensive guard: never relinquish a live non-rest reference.
            # Normal failure handling stays in the loop above instead.
            raise RuntimeError('Cannot release powered recovery owner away from verified rest')
        if paused and not commands_started:
            os.kill(pause_pid, signal.SIGCONT)  # No new reference was ever sent.
        interface.close()
        lock.close()
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--previous-guardian', type=int)
    args = parser.parse_args()
    main(args.pid, previous_guardian=args.previous_guardian)
