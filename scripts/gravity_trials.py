"""Bounded hands-off gravity tuning through the single live JSON service."""
import json
import time
from pathlib import Path

import numpy as np
import tyro


def main(runtime: str, output: str = 'evidence/gravity-camera', validation: bool = False, wrist: bool = False):
    root, dest = Path(runtime), Path(output)
    dest.mkdir(parents=True, exist_ok=True)

    def state():
        s = json.loads((root / 'status.json').read_text())
        if not s['live'] or s['closed'] or time.time()-s['timestamp'] > 1:
            raise RuntimeError('Fresh enabled live SDK service required')
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError(f"SDK fault: {s.get('fault') or s.get('motion_fault')}")
        if max(s['temperatures']['mos_c']+s['temperatures']['rotor_c']) >= 55:
            raise RuntimeError('Wait for motor cooldown before commissioning')
        return s

    def request(payload):
        state()
        name = f'gravity-probe-{time.time_ns()}'
        submitted = time.time()
        p = root / 'requests' / f'{name}.json'
        temp = p.with_suffix('.tmp')
        temp.write_text(json.dumps(payload))
        temp.replace(p)
        response = root / 'responses' / p.name
        deadline = time.monotonic()+30
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
                raise TimeoutError('Response not published')
            time.sleep(.02)
        return result.get('result')

    def move(q):
        request({'action': 'mode', 'mode': 'Hold'})
        request({'action': 'joint', 'q': np.asarray(q).tolist()})
        deadline = time.monotonic()+25
        stable = None
        while True:
            s = state()
            if not s['motion_active'] and max(abs(np.array(s['q_rad'])-q)) < .003:
                stable = stable or time.monotonic()
                if time.monotonic()-stable >= 1.:
                    break
            else:
                stable = None
            if time.monotonic()>deadline:
                raise TimeoutError('Probe pose did not settle')
            time.sleep(.05)

    profiles = [
        ('exact-unit', [1., 1., 1., 1., 1., 1.]),
        ('balanced-120', [1., 1.2, 1.2, 1., 1., 1.]),
        ('shoulder-140', [1., 1.4, 1.2, 1., 1., 1.]),
        ('torque-guided', [1., 1.25, 1.15, .98, 1., 1.]),
    ]
    if wrist:
        profiles = [('wrist-unit', [1.]*6), ('wrist-105', [1.,1.,1.,1.,1.,1.05]),
                    ('wrist-110', [1.,1.,1.,1.,1.,1.1]), ('wrist-095', [1.,1.,1.,1.,1.,.95])]
    if validation:
        selected = json.loads((dest / 'selected-profile.json').read_text())
        profiles = [(selected['name'], selected['factors'])]
    indices = [0] if wrist else [2, 7, 15] if validation else [0, 4, 8]
    rest = np.array(state()['rest_rad'])
    results = []
    try:
        for i in indices:
            q = rest+np.array([.12*np.sin(i*1.8), .21+.12*np.sin(i*.7), .22+.12*np.cos(i*.9),
                .18*np.sin(i*1.3), .19*np.sin(i*1.7), .17+.15*np.sin(i*1.1)])
            if wrist:
                q = rest+np.array([0., .2, .25, .05, .05, .15])
            for name, factors in profiles:
                move(q)
                request({'action': 'gravity_configure', 'factors': factors, 'model': 'sdk', 'persist': False})
                # The SDK performs the powered transition, starts a timed
                # gravity-only probe, and always restores MIT PD itself.
                request({'action': 'gravity_trial', 'duration_s': 1.5 if validation or wrist else .6,
                         'max_excursion_rad': .04, 'max_speed_rad_s': .3})
                deadline = time.monotonic()+3
                while True:
                    telemetry = request({'action': 'gravity_telemetry', 'include_regressor': False})
                    trial = telemetry['gravity_trial']
                    if trial and not trial['active']:
                        break
                    if time.monotonic()>deadline:
                        raise TimeoutError('SDK gravity probe did not restore hold')
                    time.sleep(.04)
                powers = state()['motor_errors']
                if any(powers.get(str(j)) != '0x1' for j in range(1, 7)):
                    raise RuntimeError(f'Probe did not preserve all six enabled motors: {powers}')
                request({'action': 'mode', 'mode': 'Hold'})
                result = {'name': name, 'factors': factors, 'pose_index': i,
                    'validation': validation, 'wrist': wrist, 'trial': trial, 'telemetry': telemetry, 'motor_errors': powers}
                results.append(result)
                filename = 'wrist-trials.json' if wrist else 'validation-trials.json' if validation else 'tuning-trials.json'
                (dest / filename).write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps({'name': name, 'pose': i, 'validation': validation,
                    'reason': trial['reason'], 'peak_deg': np.rad2deg(trial['peak_excursion_rad']),
                    'by_joint_deg': np.rad2deg(trial['peak_excursion_by_joint_rad']).tolist(),
                    'peak_speed': trial['peak_speed_rad_s']}), flush=True)
    finally:
        # A failure never disables away from rest. SDK trial expiry is independent
        # of this process and restores powered PD even if this client vanishes.
        request({'action': 'mode', 'mode': 'Hold'})
        request({'action': 'rest'})


if __name__ == '__main__':
    tyro.cli(main)
