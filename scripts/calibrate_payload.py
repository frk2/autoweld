"""Bounded live payload identification through the running application's JSON API.

No CAN or SDK instance is opened here. Quiet position-speed holds measure effort;
rest/contact is excluded. Fitting never applies an unvalidated candidate.
"""
import json
import time
from pathlib import Path

import numpy as np
import tyro

from autoweld.gravity_calibration import fit_payload


def main(runtime: str, output: str = 'evidence/gravity-camera', poses: int = 24, resume: bool = False):
    root, dest = Path(runtime), Path(output)
    dest.mkdir(parents=True, exist_ok=True)

    def state():
        s = json.loads((root / 'status.json').read_text())
        if not s['live'] or s['closed'] or time.time()-s['timestamp'] > 1:
            raise RuntimeError('Need enabled fresh live service')
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError(f"SDK fault: {s.get('fault') or s.get('motion_fault')}")
        return s

    def request(payload):
        state()
        name = f'gravity-fit-{time.time_ns()}'
        p = root / 'requests' / f'{name}.json'
        temp = p.with_suffix('.tmp')
        submitted = time.time()
        temp.write_text(json.dumps(payload))
        temp.replace(p)
        response = root / 'responses' / p.name
        deadline = time.monotonic()+25
        while not response.exists():
            state()
            if time.monotonic()>deadline:
                raise TimeoutError(payload)
            time.sleep(.02)
        result = json.loads(response.read_text())
        if not result['ok']:
            raise RuntimeError(result)
        while (state().get('last_request') or {}).get('timestamp', 0) < submitted:
            if time.monotonic()>deadline:
                raise TimeoutError('No telemetry publication')
            time.sleep(.02)
        return result.get('result')

    rest = np.array(state()['rest_rad'])
    request({'action': 'mode', 'mode': 'Hold'})
    request({'action': 'gravity_tuning_status', 'status': 'sampling',
             'message': 'Measuring static motor effort at bounded free-space poses'})
    samples = json.loads((dest / 'samples.json').read_text()) if resume and (dest / 'samples.json').exists() else []
    samples = [x for x in samples if max(x['effort_std_nm']) <= .15]
    accepted = {x['pose_index'] for x in samples}
    rejected = []
    try:
        for i in range(poses):
            if i in accepted:
                continue
            q = rest+np.array([.12*np.sin(i*1.8), .21+.12*np.sin(i*.7), .22+.12*np.cos(i*.9),
                .18*np.sin(i*1.3), .19*np.sin(i*1.7), .17+.15*np.sin(i*1.1)])
            # Every candidate clears the contact rest by raising both main links.
            # Robot service additionally preflights full path bounds/collision.
            if np.max(np.abs(q-rest)) >= .4:
                raise ValueError('Calibration exceeds the bounded test envelope')
            request({'action': 'joint', 'q': q.tolist()})
            deadline = time.monotonic()+20
            stable = None
            while True:
                s = state()
                if not s['motion_active'] and np.max(np.abs(np.array(s['q_rad'])-q)) < .002:
                    stable = stable or time.monotonic()
                    if time.monotonic()-stable >= 2.:
                        break
                else:
                    stable = None
                if time.monotonic()>deadline:
                    raise TimeoutError('Calibration pose did not settle')
                time.sleep(.05)
            for attempt in range(5):
                rows, stamps = [], set()
                for _ in range(14):
                    row = request({'action': 'gravity_telemetry'})
                    stamp = row['feedback_timestamp']
                    if row['feedback_fresh'] and stamp not in stamps:
                        rows.append(row)
                        stamps.add(stamp)
                    time.sleep(.06)
                if len(rows) < 8:
                    raise RuntimeError('Insufficient fresh quiet samples')
                efforts = np.array([r['reported_effort_nm'] for r in rows])
                velocities = np.array([r['vel_rad_s'] for r in rows])
                qs = np.array([r['q_rad'] for r in rows])
                # Individual p16 velocity bins have a visible noise floor.
                # Require a quiet position window AND low mean velocity; never
                # hide position drift behind a smoothed velocity measurement.
                quiet = (np.max(np.abs(np.mean(velocities, axis=0))) <= .02
                         and np.max(np.ptp(qs, axis=0)) <= .0015
                         and np.max(np.std(efforts, axis=0)) <= .15)
                if quiet:
                    break
                rejected.append({'pose_index': i, 'attempt': attempt, 'raw_feedback': rows})
                (dest / 'rejected-windows.json').write_text(json.dumps(rejected, indent=2)+'\n')
                time.sleep(2.)
            if not quiet:
                raise RuntimeError('No stationary low-noise measurement window after five attempts')
            sample = {'q_rad': np.mean(qs, axis=0).tolist(),
                'effort_nm': np.mean(efforts, axis=0).tolist(),
                'effort_std_nm': np.std(efforts, axis=0).tolist(),
                'velocity_rad_s': np.mean(velocities, axis=0).tolist(),
                'baseline_nm': np.mean([r['baseline_nm'] for r in rows], axis=0).tolist(),
                'regressor': np.mean([r['regressor'] for r in rows], axis=0).tolist(),
                'split': 'validation' if i % 4 == 0 else 'train',
                'free_space': True, 'settled': True, 'contact': False, 'at_rest': False,
                'pose_index': i, 'raw_feedback': rows}
            samples.append(sample)
            (dest / 'samples.json').write_text(json.dumps(samples, indent=2)+'\n')
            print(json.dumps({'pose': i+1, 'count': poses,
                'torque_nm': sample['effort_nm'], 'std_nm': sample['effort_std_nm']}), flush=True)
        result = fit_payload(samples)
        (dest / 'payload-fit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        # Return is always the existing validated/planned SDK workflow.
        if not state().get('motion_active'):
            request({'action': 'rest'})


if __name__ == '__main__':
    tyro.cli(main)
