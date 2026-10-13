"""Repeat the reviewed circle from its current endpoint, keeping the board clamped."""
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from scripts.kp_path_repeat import check_gains
from scripts.path_deviation_repeat import healthy
from scripts.tune_pid import request


def main():
    root = Path('runtime/live')
    with (root / 'tuning.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = healthy()
        if (before['motion_active'] or not before['gripper_hold_closed']
                or before['gripper_target_normalized'] != 0
                or before['automatic_test_excursion_rad'] != .49):
            raise RuntimeError('Need an idle, healthy arm with the board clamped and normal test bounds')
        check_gains(before)
        dest = Path(f'evidence/posetracker-circle50-capture-{time.time_ns()}')
        dest.mkdir()
        print('EVIDENCE', dest, flush=True)
        for name in ('calibration.json', 'board.json'):
            shutil.copyfile(Path('/home/faraz/Code/posetracker/data') / name, dest / name)
        (dest / 'before.json').write_text(json.dumps(before, indent=2))
        telemetry = root / 'control-telemetry.jsonl'
        telemetry_offset = telemetry.stat().st_size
        recorder = None

        def event(name, **data):
            with (dest / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event=name, wall_time=time.time(),
                                             monotonic_ns=time.monotonic_ns(), **data)) + '\n')
            print(name, data, flush=True)

        def send(payload):
            sent = time.time()
            request(root, payload, timeout_s=30)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                s = json.loads((root / 'status.json').read_text())
                if (time.time() - s['timestamp'] <= 1
                        and (s.get('last_request') or {}).get('timestamp', 0) >= sent):
                    return healthy()
                time.sleep(.03)
            raise RuntimeError('No fresh post-request status; do not retry motion')

        def dwell(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                healthy()
                if recorder is not None and recorder.poll() is not None:
                    raise RuntimeError('PoseTracker stopped unexpectedly')
                time.sleep(.1)

        try:
            send({'action': 'position_speed_limit', 'limit_rad_s': .25})
            send({'action': 'track_load', 'path': 'evidence/circle-speed/assets/circle.step'})
            payload = dict(edge=0, feed_m_s=.05, orientation='Unconstrained', ramp_placement='Inside edge')
            preview = send({'action': 'track_preview', **payload})
            (dest / 'preview.json').write_text(json.dumps(preview, indent=2))
            with (dest / 'posetracker.log').open('w') as log:
                recorder = subprocess.Popen(
                    ['/home/faraz/Code/posetracker/.venv/bin/python', 'scripts/record_board_pose.py',
                     str(dest), '--save-all'], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
            event('camera started')
            dwell(8)
            with (dest / 'poses.csv').open() as stream:
                poses = list(csv.DictReader(stream))
            latest = int(poses[-1]['host_monotonic_ns']) / 1e9
            recent = [r for r in poses if int(r['host_monotonic_ns']) / 1e9 >= latest - 2]
            fraction = sum(r['valid'] == '1' for r in recent) / len(recent)
            event('camera pre-motion coverage', valid_fraction=fraction, frames=len(recent))
            if time.monotonic() - latest > 2 or len(recent) < 60 or fraction < .8:
                raise RuntimeError('Camera coverage insufficient; no circle command sent')
            start = healthy()
            event('circle request')
            running = send({'action': 'track_run', **payload})
            previous_trace = (start.get('tracking') or {}).get('trace')
            deadline = time.monotonic() + running['motion_plan']['duration_s'] + 10
            while time.monotonic() < deadline:
                s = healthy()
                result = s.get('tracking') or {}
                if result.get('trace') and result['trace'] != previous_trace and not result.get('running'):
                    shutil.copyfile(result['trace'], dest / 'circle60-50mms.jsonl')
                    (dest / 'circle60-50mms.json').write_text(json.dumps(
                        dict(suite='circles', index=0, before=before, start=start, after=s, result=result), indent=2))
                    if not result.get('completed'):
                        raise RuntimeError(f'Circle did not complete: {result.get("reason")}')
                    event('circle complete', max_edge_mm=result['max_cross_track_mm'])
                    break
                time.sleep(.05)
            else:
                raise TimeoutError('Circle completion timeout')
            dwell(8)
            event('post-circle dwell complete')
        except Exception as exc:
            event('error', reason=str(exc))
            s = json.loads((root / 'status.json').read_text())
            if not s.get('closed') and not s.get('fault') and s.get('motion_active'):
                request(root, {'action': 'mode', 'mode': 'Hold'})
            raise
        finally:
            if recorder is not None and recorder.poll() is None:
                recorder.send_signal(signal.SIGINT)
                recorder.wait(timeout=20)
            s = healthy()
            if not s['motion_active']:
                send({'action': 'position_speed_limit', 'limit_rad_s': before['position_speed_limit_rad_s']})
            with telemetry.open('rb') as source:
                source.seek(telemetry_offset)
                data = source.read()
            (dest / 'complete-control.jsonl').write_bytes(data[:data.rfind(b'\n') + 1])
            (dest / 'final-status.json').write_text(json.dumps(healthy(), indent=2))
            event('recording stopped; holding endpoint with gripper closed')


if __name__ == '__main__':
    main()
