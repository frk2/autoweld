"""One explicitly authorized mixed-wrist comparison condition via the SDK app.

Run one controller condition at a time. Motor disable between conditions releases
the grip; a retained board must be externally supported for each transition.
"""
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts.kp_path_repeat import check_gains
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/mixed-wrist')
TELEMETRY = ROOT / 'control-telemetry.jsonl'


def state():
    return json.loads((ROOT / 'status.json').read_text())


def healthy(controller=None):
    s = state()
    if time.time() - s['timestamp'] > 2 or s['fault'] or s['motion_fault']:
        raise RuntimeError(f"Controller unhealthy: {s['fault'] or s['motion_fault']}")
    if (not s['closed'] and not s['oscillation_guard_active']) or s['friction_enabled']:
        raise RuntimeError('Need active SDK guard and friction disabled')
    if s['automatic_test_excursion_rad'] != .49:
        raise RuntimeError('These tests require the original .49 rad bound')
    if s['tracking_position_tolerance_mm'] != 10 or s['tracking_deviation_action'] != 'record_only':
        raise RuntimeError('Unexpected path-deviation policy')
    if controller and s['controller'] != controller:
        raise RuntimeError('Controller changed during experiment')
    return s


def send(payload):
    start = time.time()
    request(ROOT, payload, timeout_s=45)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        s = state()
        # Planning temporarily pauses publication, but never the SDK loop.
        if (s.get('last_request') or {}).get('timestamp', 0) >= start:
            return healthy()
        time.sleep(.03)
    raise TimeoutError('No post-request status; do not resend uncertain command')


def observe(seconds, controller):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        healthy(controller)
        time.sleep(.05)


def settled(controller, timeout=35, path_start=False):
    deadline = time.monotonic() + timeout
    since = None
    while time.monotonic() < deadline:
        s = healthy(controller)
        error = abs(np.array(s['q_rad']) - s['target_rad'])
        tolerance = np.full(6, .0099) if path_start else np.deg2rad([.2]*3+[1.]*3)
        if not s['motion_active'] and np.all(error < tolerance):
            since = since or time.monotonic()
            if time.monotonic() - since >= 2:
                return s
        else:
            since = None
        time.sleep(.05)
    raise TimeoutError(f'Waypoint did not settle: errors {np.rad2deg(error).tolist()} degrees; '
                       f'tolerances {np.rad2deg(tolerance).tolist()} degrees')


def archive_tail(offset, path):
    with TELEMETRY.open('rb') as f:
        f.seek(offset)
        data = f.read()
    path.write_bytes(data[:data.rfind(b'\n') + 1])


def disable(name):
    # This path intentionally does not require the experiment's tighter settle.
    send({'action': 'disable'})
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        s = healthy()
        if s['closed']:
            if not s['rest_verified'] or not s['at_rest']:
                raise RuntimeError('Disable did not verify rest')
            send({'action': 'position_speed_limit', 'limit_rad_s': .15})
            (DEST / f'{name}-disabled.json').write_text(json.dumps(state(), indent=2))
            print('DISABLED AT VERIFIED REST', s['max_error_deg'], flush=True)
            return
        time.sleep(.05)
    raise TimeoutError('Rest verification incomplete; motors remain enabled')


def main(controller: str, board_state: str, repetitions: int = 2):
    if controller not in ('POS_VEL', 'MIXED_J45') or repetitions not in (1, 2):
        raise ValueError('Choose POS_VEL or MIXED_J45, one or two repetitions')
    if board_state not in ('removed', 'supported'):
        raise ValueError('Record the operator-confirmed board state first')
    name = f'{controller.lower()}-{time.time_ns()}'
    spec = json.loads((DEST / 'preflight.json').read_text())
    if not all(x['ok'] for x in spec['fixtures']):
        raise RuntimeError('Offline preflight failed')
    record = dict(controller=controller, board_state=board_state, name=name, runs=[])
    offset = TELEMETRY.stat().st_size
    with (ROOT / 'tuning.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            s = healthy()
            if not s['closed']:
                raise RuntimeError('Start each condition disabled at captured rest')
            send({'action': 'controller', 'controller': controller})
            send({'action': 'position_speed_limit', 'limit_rad_s': .25})
            record['enabled'] = send({'action': 'enable'})
            check_gains(record['enabled'])
            expected = ['POS_VEL'] * 6
            if controller == 'MIXED_J45':
                expected[3:5] = ['MIT', 'MIT']
                if record['enabled']['gravity']['active_arm_joints'] != [4, 5]:
                    raise RuntimeError('J4/J5 gravity not enabled')
            survey = record['enabled']['motor_mode_survey']
            if any(survey[str(j+1)]['selected_mode'] != (2 if mode == 'POS_VEL' else 1)
                   for j, mode in enumerate(expected)):
                raise RuntimeError('Independent actuator mode readback mismatch')
            # Startup holds freshly measured angles, which can differ from the
            # supported rest after gravity is engaged. Establish the same saved
            # rest in both conditions before applying the experiment settle test.
            observe(3, controller)
            send({'action': 'rest'})
            settled(controller)
            send({'action': 'gripper_close'})
            observe(3, controller)
            # Small commissioning move precedes the large paths in both modes.
            record['commission_begin'] = time.time()
            commission_offset = TELEMETRY.stat().st_size
            initial_q = np.array(healthy(controller)['q_rad'])
            send({'action': 'joint', 'q': spec['commissioning_q'][:6]})
            record['commission_settled'] = settled(controller)
            observe(4, controller)
            with TELEMETRY.open('rb') as f:
                f.seek(commission_offset)
                data = f.read()
            samples = [json.loads(x) for x in data[:data.rfind(b'\n')+1].splitlines()]
            target = np.array(spec['commissioning_q'][:6])
            direction = np.sign(target-initial_q)
            moved = abs(target-initial_q) > .002
            overshoot = np.maximum(0., np.max(
                np.rad2deg((np.array([r['q'] for r in samples])-target)*direction), axis=0))
            record['commission_overshoot_deg'] = overshoot.tolist()
            if np.any(overshoot[moved] >= np.array([.1]*3+[.5]*3)[moved]):
                raise RuntimeError(f'Commissioning overshoot exceeds standing limits: {overshoot.tolist()}')
            send({'action': 'rest'})
            settled(controller)
            record['commission_end'] = time.time()
            for repetition in range(repetitions):
                for fixture in spec['fixtures']:
                    trial_name = f"{name}-{fixture['name']}-r{repetition+1}"
                    print('TRIAL', trial_name, flush=True)
                    begin_offset = TELEMETRY.stat().st_size
                    send({'action': 'joint', 'q': fixture['seed_q'][:6]})
                    settled(controller)
                    send({'action': 'track_load', 'path': fixture['asset']})
                    payload = dict(edge=0, feed_m_s=fixture['feed_m_s'], orientation='Unconstrained',
                                   ramp_placement='Inside edge')
                    send({'action': 'track_approach', **payload})
                    before = settled(controller, path_start=True)
                    observe(4, controller)
                    running = send({'action': 'track_run', **payload})
                    deadline = time.monotonic() + running['motion_plan']['duration_s'] + 15
                    while time.monotonic() < deadline:
                        after = healthy(controller)
                        result = after.get('tracking') or {}
                        if not result.get('running', True) and result.get('trace'):
                            break
                        time.sleep(.05)
                    else:
                        raise TimeoutError('No finished tracking result')
                    (DEST / f'{trial_name}.jsonl').write_bytes(Path(result['trace']).read_bytes())
                    trial = dict(name=trial_name, fixture=fixture, before=before, after=after, result=result)
                    record['runs'].append(trial)
                    (DEST / f'{trial_name}.json').write_text(json.dumps(trial, indent=2))
                    if not result['completed']:
                        raise RuntimeError('Reference interrupted; inspect before another trial')
                    observe(5, controller)
                    archive_tail(begin_offset, DEST / f'{trial_name}-control.jsonl')
                    print('COMPLETED', trial_name, 'edge mm', result.get('max_cross_track_mm'), flush=True)
                    send({'action': 'rest'})
                    settled(controller)
            disable(name)
            record['final'] = state()
        except Exception as exc:
            record['error'] = str(exc)
            record['error_status'] = state()
            print('STOPPED:', str(exc), flush=True)
            s = state()
            if not s['closed'] and not s['fault'] and not s['motion_fault']:
                if s['motion_active']:
                    send({'action': 'mode', 'mode': 'Hold'})
                disable(name)
            raise
        finally:
            archive_tail(offset, DEST / f'{name}-session-control.jsonl')
            (DEST / f'{name}-session.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    tyro.cli(main)
