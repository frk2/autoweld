"""Repeat the reviewed large fixtures with the selected onboard gains, via SDK app."""
import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts.path_deviation_repeat import healthy, send, state
from scripts.stage3_trial import main as trial

DEST = Path('evidence/kp-path-repeat')
TELEMETRY = Path('runtime/live/control-telemetry.jsonl')


def check_fixture(item, s):
    if not item['ok'] or not s['closed']:
        raise ValueError('Need accepted preflight and disabled arm')
    if s['automatic_test_excursion_rad'] != item['automatic_limit_rad']:
        raise ValueError('Automatic excursion does not match fixture preflight')
    if item['automatic_limit_rad'] > .49:
        if (item['suite'] != 'lines' or item['index'] != 0
                or item['asset'] != 'evidence/long-cruise/assets/line-500mm.step'
                or item['automatic_limit_rad'] != 1.55):
            raise ValueError('Large excursion applies only to the reviewed 500 mm line')
    if not .15 <= item['cap_rad_s'] <= min(1., s['motion_limits']['joint_velocity']):
        raise ValueError('Invalid preflight speed ceiling')


def check_gains(s):
    expected = json.loads((DEST / 'selected-profile.json').read_text())['kp_apr']
    survey = s['motor_mode_survey']
    actual = [survey[str(i)]['KP_APR'] for i in range(1, 7)]
    if not np.allclose(actual, expected, rtol=1e-6, atol=0):
        raise RuntimeError(f'Onboard gains differ from selection: {actual}')
    for i in range(1, 8):
        if survey[str(i)]['status_before'] != '0x0':
            raise RuntimeError('All motors must have been disabled before enable')


def disable(name):
    """Use SDK idle-start and normal verified-rest checks, never experiment gates."""
    attempts = []
    for payload in ({'action': 'position_speed_limit', 'limit_rad_s': .15}, {'action': 'disable'}):
        for _ in range(10):
            healthy()
            try:
                send(payload)
                break
            except RuntimeError as exc:
                if 'Wait for the arm to stop' not in str(exc):
                    raise
                attempts.append(dict(action=payload['action'], reason=str(exc), timestamp=time.time()))
                time.sleep(1)
        else:
            raise RuntimeError('SDK idle-start continues to reject return; inspect')
    deadline = time.monotonic() + 80
    while time.monotonic() < deadline:
        s = state()
        if s['fault'] or s['motion_fault']:
            raise RuntimeError('SDK fault during return; inspect before recovery')
        if s['closed']:
            if not s['rest_verified'] or s['max_error_deg'] > .5:
                raise RuntimeError('Normal disable did not verify rest')
            (DEST / f'{name}-rest.json').write_text(json.dumps(s, indent=2))
            (DEST / 'final-rest-status.json').write_text(json.dumps(s, indent=2))
            (DEST / f'{name}-return-attempts.json').write_text(json.dumps(attempts, indent=2))
            print('DISABLED, verified rest', s['max_error_deg'], flush=True)
            return s
        time.sleep(.05)
    raise TimeoutError('Normal verified-rest disable timed out')


def collect_settling(name):
    """Archive a stationary-target tail without doing FK or plotting while enabled."""
    offset = TELEMETRY.stat().st_size
    begin = time.time()
    deadline = time.monotonic() + 30
    since = None
    settled = False
    target = healthy()['target_rad']
    while time.monotonic() < deadline:
        s = healthy()
        if s['target_rad'] != target or s['motion_active']:
            raise RuntimeError('Unexpected target/motion during endpoint observation')
        if np.max(abs(np.array(s['q_rad']) - target)) < .005:
            since = since or time.monotonic()
            if time.monotonic() - since >= 3:
                settled = True
                break
        else:
            since = None
        time.sleep(.05)
    with TELEMETRY.open('rb') as f:
        f.seek(offset)
        data = f.read()
        # An incomplete final append is not a valid telemetry row.
        data = data[:data.rfind(b'\n') + 1]
    (DEST / f'{name}-settling-control.jsonl').write_bytes(data)
    (DEST / f'{name}-settling-status.json').write_text(json.dumps(
        dict(begin=begin, end=time.time(), observed_settled=settled, status=s), indent=2))


def main(suite: str, start: int = 0, stop: int | None = None, attempt: int = 1):
    items = [x for x in json.loads((DEST / 'preflight.json').read_text()) if x['suite'] == suite]
    if not items or suite not in ('lines', 'circles'):
        raise ValueError('Choose lines or circles')
    chosen = items[start:stop]
    if not chosen:
        raise ValueError('No selected fixtures')
    try:
        for item in chosen:
            name = f"{item['name']}-attempt{attempt}"
            if not item['ok']:
                print('PREFLIGHT REJECTED', item['name'], item['error'], flush=True)
                continue
            if any(DEST.glob(name + '*.json')):
                raise FileExistsError(name)
            check_fixture(item, state())
            send({'action': 'controller', 'controller': 'POS_VEL'})
            send({'action': 'position_speed_limit', 'limit_rad_s': item['cap_rad_s']})
            try:
                enabled = send({'action': 'enable'})
                healthy()
                check_gains(enabled)
                (DEST / f'{name}-enable.json').write_text(json.dumps(enabled, indent=2))
                print('TRIAL', name, 'ceiling', item['cap_rad_s'], flush=True)
                trial(index=item['index'], feed=item['feed_mm_s'] / 1000, name=name,
                      suite=item['suite'], evidence_dir=str(DEST), step_path=item['asset'],
                      ramp_placement=item['ramp_placement'], seed_q=tuple(item['seed_q'][:6]),
                      settle_hold_s=2.)
                result = json.loads((DEST / f'{name}.json').read_text())['result']
                if not result['completed']:
                    raise RuntimeError('Reference interrupted; inspect before next trial')
                collect_settling(name)
            except Exception:
                s = state()
                if not s['closed'] and not s['fault'] and not s['motion_fault']:
                    if s['motion_active']:
                        send({'action': 'mode', 'mode': 'Hold'})
                    disable(name)
                raise
            disable(name)
    finally:
        s = state()
        if s['closed'] and s['automatic_test_excursion_rad'] > .49:
            s = send({'action': 'end_long_cruise_test'})
            (DEST / 'final-rest-status.json').write_text(json.dumps(s, indent=2))


if __name__ == '__main__':
    tyro.cli(main)
