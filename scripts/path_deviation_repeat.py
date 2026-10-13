"""One preflighted repeat; normal rest disable between every acquisition."""
import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts.stage3_trial import main as trial
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/path-deviation-repeat')


def state():
    s = json.loads((ROOT / 'status.json').read_text())
    if time.time() - s['timestamp'] > 1:
        raise RuntimeError('App status is stale')
    return s


def send(payload):
    before = time.time()
    request(ROOT, payload)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        # SDK reconnect can take >1 s while the UI publication is paused.
        # Wait for the new publication; do not mistake the old one for live state.
        s = json.loads((ROOT / 'status.json').read_text())
        if time.time() - s['timestamp'] <= 1 and (s.get('last_request') or {}).get('timestamp', 0) >= before:
            return s
        time.sleep(.03)
    raise RuntimeError('No post-request status; inspect before another command')


def healthy():
    s = state()
    if s['closed'] or s['fault'] or s['motion_fault'] or s['controller'] != 'POS_VEL':
        raise RuntimeError(f"SDK/controller state interrupted acquisition: {s['fault'] or s['mode']}")
    assert s['oscillation_guard_active'] and not s['friction_enabled']
    assert s['tracking_position_tolerance_mm'] == 10 and s['tracking_deviation_action'] == 'record_only'
    return s


def finish(name):
    deadline = time.monotonic() + 30
    since = None
    while time.monotonic() < deadline:
        s = healthy()
        if not s['motion_active'] and np.max(abs(np.array(s['q_rad']) - s['target_rad'])) < .005:
            since = since or time.monotonic()
            if time.monotonic() - since >= 2:
                break
        else:
            since = None
        time.sleep(.05)
    else:
        raise RuntimeError('Endpoint did not settle for return')
    # The SDK independently requires fresh feedback and low measured speed.
    attempts = []
    for payload in ({'action': 'position_speed_limit', 'limit_rad_s': .15}, {'action': 'disable'}):
        for _ in range(5):
            try:
                send(payload)
                break
            except RuntimeError as exc:
                if 'Wait for the arm to stop' not in str(exc):
                    raise
                attempts.append({'action': payload['action'], 'rejection': str(exc), 'time': time.time()})
                healthy()
                time.sleep(1.)
        else:
            raise RuntimeError('SDK idle-start check still rejects normal return')
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        s = state()
        if s['closed']:
            assert s['rest_verified'] and s['max_error_deg'] <= .5
            if s['automatic_test_excursion_rad'] > .49:
                s = send({'action': 'end_long_cruise_test'})
            assert s['automatic_test_excursion_rad'] == .49 and s['position_speed_limit_rad_s'] == .15
            (DEST / f'{name}-rest.json').write_text(json.dumps(s, indent=2))
            (DEST / 'final-rest-status.json').write_text(json.dumps(s, indent=2))
            (DEST / f'{name}-return-attempts.json').write_text(json.dumps(attempts, indent=2))
            print('DISABLED at verified rest', s['max_error_deg'], flush=True)
            return
        if s['fault'] or s['motion_fault']:
            raise RuntimeError('SDK fault during rest return; inspect before recovery')
        time.sleep(.05)
    raise TimeoutError('Normal verified-rest disable did not finish')


def main(index: int, attempt: int = 1, settle_hold_s: float = 2.):
    item = json.loads((DEST / 'preflight.json').read_text())[index]
    name = f"{item['name']}-attempt{attempt}"
    if not item['ok']:
        print('PREFLIGHT REJECTED:', item['error'], flush=True)
        return
    if any((DEST / f'{name}{suffix}').exists() for suffix in ('.json', '-error.json')):
        raise RuntimeError('Evidence already exists')
    s = state()
    assert s['closed'] and s['automatic_test_excursion_rad'] == item['automatic_limit_rad']
    if item['automatic_limit_rad'] > .49:
        assert item['name'] == 'repeat-line500-01-80mms'
        assert item['asset'] == 'evidence/long-cruise/assets/line-500mm.step'
    send({'action': 'controller', 'controller': 'POS_VEL'})
    try:
        send({'action': 'enable'})
        healthy()
        send({'action': 'position_speed_limit', 'limit_rad_s': item['cap_rad_s']})
        print('TRIAL', index, name, 'ceiling', item['cap_rad_s'], flush=True)
        trial(index=item['index'], feed=item['feed_mm_s'] / 1000, name=name,
              suite=item['suite'], evidence_dir=str(DEST), step_path=item['asset'],
              ramp_placement=item['ramp_placement'], seed_q=tuple(item['seed_q'][:6]), settle_hold_s=settle_hold_s)
    except Exception:
        s = state()
        if not s['closed'] and not s['fault'] and not s['motion_fault']:
            send({'action': 'mode', 'mode': 'Hold'})
            finish(name)
        raise
    finish(name)


if __name__ == '__main__':
    tyro.cli(main)
