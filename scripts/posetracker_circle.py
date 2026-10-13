"""One authorized 50 mm/s circle, powered closed grip and concurrent PoseTracker."""
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.kp_path_repeat import check_gains
from scripts.path_deviation_repeat import healthy, send, state
from scripts.stage3_trial import main as trial

DEST = Path('evidence/posetracker-circle50')
TELEMETRY = Path('runtime/live/control-telemetry.jsonl')


def event(name):
    with (DEST / 'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name, wall_time=time.time(), monotonic_ns=time.monotonic_ns()))+'\n')
    print(name, flush=True)


def dwell(seconds, camera=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        s = healthy()
        if not s['gripper_hold_closed'] or s['gripper_target_normalized'] != 0:
            raise RuntimeError('Closed-grip intent changed')
        if camera is not None and camera.poll() is not None:
            raise RuntimeError('PoseTracker stopped unexpectedly')
        time.sleep(.05)


def return_holding():
    s = healthy()
    if s['motion_active']:
        raise RuntimeError('Inspect active motion before return')
    for payload in ({'action': 'position_speed_limit', 'limit_rad_s': .15}, {'action': 'rest'}):
        for _ in range(10):
            try:
                send(payload)
                break
            except RuntimeError as exc:
                if 'Wait for the arm to stop' not in str(exc):
                    raise
                dwell(1)
        else:
            raise RuntimeError('SDK still rejects return from moving state')
    deadline = time.monotonic()+70
    while time.monotonic() < deadline:
        s = healthy()
        if not s['motion_active'] and s['rest_verified'] and s['at_rest']:
            (DEST / 'final-rest-holding.json').write_text(json.dumps(s, indent=2))
            event('verified arm rest, grip remains powered closed')
            return
        time.sleep(.05)
    raise RuntimeError('Return did not verify arm rest')


def main():
    if (DEST / 'poses.csv').exists():
        raise FileExistsError('Existing experiment; do not overwrite or repeat motion')
    item = next(x for x in json.loads(Path('evidence/kp-path-repeat/preflight.json').read_text())
                if x['name'] == 'circle60-50mms')
    s = state()
    assert s['automatic_test_excursion_rad'] == .49
    if not s['closed']:
        healthy()
        assert s['gripper_hold_closed'] and s['at_rest'] and not s['motion_active']
    assert item['ok'] and item['automatic_limit_rad'] == .49
    (DEST / 'preflight.json').write_text(json.dumps(item, indent=2))
    offset = TELEMETRY.stat().st_size
    camera = None
    log = None
    try:
        send({'action': 'controller', 'controller': 'POS_VEL'})
        send({'action': 'position_speed_limit', 'limit_rad_s': item['cap_rad_s']})
        enabled = send({'action': 'enable'}) if s['closed'] else healthy()
        check_gains(enabled)
        (DEST / 'enabled.json').write_text(json.dumps(enabled, indent=2))
        time.sleep(1)
        if not enabled['gripper_hold_closed']:
            send({'action': 'gripper_close'})
        dwell(3)
        event('closed grip established')
        log = (DEST / 'posetracker.log').open('w')
        camera = subprocess.Popen(['/home/faraz/Code/posetracker/.venv/bin/python',
                                   'scripts/record_board_pose.py', str(DEST)],
                                  stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        dwell(8, camera)
        with (DEST / 'poses.csv').open() as f:
            rows = list(csv.DictReader(f))
        latest = int(rows[-1]['host_monotonic_ns']) / 1e9
        recent = [r for r in rows if int(r['host_monotonic_ns']) / 1e9 >= latest - 2]
        if len(recent) < 100 or np.mean([r['valid'] == '1' for r in recent]) < .7:
            raise RuntimeError('Insufficient board detection; no circle sent')
        event('camera ready; approach and circle acquisition begins')
        trial(index=0, feed=.05, name='circle60-50mms', suite='circles',
              evidence_dir=str(DEST), step_path=item['asset'], ramp_placement='Inside edge',
              seed_q=tuple(item['seed_q'][:6]), settle_hold_s=8., pose_recording=str(DEST / 'poses.csv'))
        event('circle tracking finished')
        result = json.loads((DEST / 'circle60-50mms.json').read_text())['result']
        if not result['completed']:
            raise RuntimeError('Circle interrupted; inspect result')
        dwell(8, camera)
        event('post-circle dwell finished')
        return_holding()
        dwell(6, camera)
    finally:
        s = state()
        if not s['closed'] and not s['fault'] and not s['motion_fault']:
            if s['motion_active']:
                send({'action': 'mode', 'mode': 'Hold'})
            if s.get('gripper_hold_closed') and not (s['rest_verified'] and s['at_rest']):
                return_holding()
        if camera is not None and camera.poll() is None:
            camera.send_signal(signal.SIGINT)
            camera.wait(timeout=10)
        if log is not None:
            log.close()
        with TELEMETRY.open('rb') as f:
            f.seek(offset)
            data = f.read()
        (DEST / 'complete-control.jsonl').write_bytes(data[:data.rfind(b'\n')+1])
        (DEST / 'final-status.json').write_text(json.dumps(state(), indent=2))
        event('acquisition finished')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DEST = Path(sys.argv[1])
    main()
