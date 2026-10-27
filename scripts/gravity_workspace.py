"""Progressive wider-angle gravity commissioning through the existing JSON owner."""
import json
import time
from pathlib import Path

import numpy as np
import tyro

POSES = [
    ('baseline', [0., .3, .35, 0., 0., .25]),
    ('shoulder-35', [0., .6, .4, 0., 0., .4]),
    ('elbow-45', [0., .6, .8, 0., 0., .6]),
    ('shoulder-55', [0., .95, .8, 0., 0., .6]),
    ('elbow-70', [0., .95, 1.25, 0., 0., .6]),
    ('shoulder-90', [0., 1.55, 1.3, 0., 0., 1.]),
    ('elbow-90', [0., 1.2, 1.55, 0., 0., 1.]),
    ('wrist-roll-positive', [0., .9, 1.1, .6, 0., .9]),
    ('wrist-roll-negative', [0., .9, 1.1, -.7, 0., .9]),
    ('wrist-tilt-positive', [0., .9, 1.1, 0., .65, .9]),
    ('wrist-tilt-negative', [0., .9, 1.1, 0., -.65, .9]),
    ('tool-rotation', [0., .9, 1.1, .3, .3, 1.65]),
    ('base-positive', [.65, 1., 1.1, .4, -.4, 1.2]),
    ('base-negative', [-.65, 1.2, .9, -.4, .4, 1.2]),
    ('combined-positive', [.45, 1.4, 1.4, .6, .6, 1.55]),
    ('combined-negative', [-.45, 1.35, 1.55, -.6, -.6, 1.35]),
    ('combined-crossed', [.3, 1.1, 1.35, .8, -.55, 1.65]),
]


class Client:
    def __init__(self, runtime):
        self.root = Path(runtime)

    def state(self, enabled=False):
        s = json.loads((self.root/'status.json').read_text())
        if time.time()-s['timestamp'] > 2 or not s['live']:
            raise RuntimeError('Fresh live JSON owner required')
        if s.get('fault') or s.get('motion_fault') or s.get('mode_transition_error'):
            raise RuntimeError(s.get('fault') or s.get('motion_fault') or s.get('mode_transition_error'))
        if enabled:
            if s['closed'] or s.get('feedback_warning'):
                raise RuntimeError('Healthy fresh enabled SDK feedback required')
            if any(s['motor_errors'].get(str(j)) != '0x1' for j in range(1, 7)):
                raise RuntimeError(f"Motor power/status changed: {s['motor_errors']}")
            if max(s['temperatures']['mos_c']+s['temperatures']['rotor_c']) >= 55:
                raise RuntimeError('Commissioning temperature gate reached')
        return s

    def request(self, payload):
        # Recovery requests remain possible after a guard/error; SDK probe expiry
        # is independent of this client. Never resend a timed-out move blindly.
        name = f'gravity-workspace-{time.time_ns()}.json'
        submitted = time.time()
        path = self.root/'requests'/name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(payload))
        temp.replace(path)
        response = self.root/'responses'/name
        deadline = time.monotonic()+40
        while not response.exists():
            if time.monotonic()>deadline:
                raise TimeoutError(payload)
            time.sleep(.02)
        result = json.loads(response.read_text())
        if not result['ok']:
            raise RuntimeError(result)
        if payload['action'] == 'gravity_telemetry':
            return result.get('result')  # Matched SDK packet; no UI publish delay.
        while True:
            s = json.loads((self.root/'status.json').read_text())
            if (s.get('last_request') or {}).get('timestamp', 0) >= submitted:
                break
            if time.monotonic()>deadline:
                raise TimeoutError('Request response not yet published')
            time.sleep(.02)
        return result.get('result')

    def move(self, target):
        self.request({'action': 'mode', 'mode': 'Hold'})
        start = np.array(self.state(True)['q_rad'])
        # Wider pose coverage never permits a large individual reference step.
        steps = max(1, int(np.ceil(np.max(abs(target-start))/.18)))
        for goal in np.linspace(start, target, steps+1)[1:]:
            self.request({'action': 'joint', 'q': goal.tolist()})
            deadline, stable = time.monotonic()+25, None
            while True:
                s = self.state(True)
                quiet = (not s['motion_active'] and np.max(abs(np.array(s['q_rad'])-goal)) < .003)
                stable = (stable or time.monotonic()) if quiet else None
                if stable and time.monotonic()-stable >= .5:
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('Position-speed waypoint did not settle')
                time.sleep(.05)


def main(runtime: str = 'runtime/agents-live', output: str = 'evidence/gravity-workspace', execute: bool = False,
         j6_factor: float | None = None, j4_factor: float | None = None):
    client, dest = Client(runtime), Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    initial = client.state()
    if not initial['closed']:
        raise RuntimeError('Start with disabled motors so the entire wider sequence can be preflighted')
    rest = np.array(initial['rest_rad'])
    temporary_factors = j6_factor is not None or j4_factor is not None
    if temporary_factors:
        factors = initial['gravity_factors'].copy()
        if j6_factor is not None:
            factors[5] = j6_factor
        if j4_factor is not None:
            factors[3] = j4_factor
        client.request({'action': 'gravity_configure', 'factors': factors, 'persist': False})
    client.request({'action': 'gravity_test_envelope', 'excursion_rad': 1.8})
    accepted, rejected, results = [], [], []
    previous = rest.copy()
    try:
        for name, offset in POSES:
            target = rest+offset
            try:
                preview = client.request({'action': 'gravity_test_preflight',
                    'start_q': previous.tolist(), 'q': target.tolist()})
                # Extra endpoint clearance/margin for the bounded free-drive
                # neighborhood, beyond the full-path preflight gates.
                if preview['minimum_joint_margin_rad'] < .15 or preview['endpoint_mount_height_m'] < .10:
                    raise RuntimeError(f'Insufficient test neighborhood margin: {preview}')
            except RuntimeError as error:
                rejected.append({'name': name, 'q': target.tolist(), 'error': str(error)})
                continue
            accepted.append({'name': name, 'q': target.tolist(), 'offset_rad': offset, 'preflight': preview})
            previous = target
        client.request({'action': 'gravity_test_preflight', 'start_q': previous.tolist(), 'q': rest.tolist()})
        manifest = {'initial': initial, 'accepted': accepted, 'rejected': rejected,
            'test_factors': client.state()['gravity_factors'],
            'step_limit_rad': .18, 'gravity_excursion_rad': .04, 'gravity_speed_rad_s': .3,
            'progression_drift_limit_rad': .015}
        (dest/'preflight.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Preflight: {len(accepted)} accepted, {len(rejected)} rejected; no motors moved', flush=True)
        if not execute:
            return
        if not accepted:
            raise RuntimeError('No wider poses passed preflight')
        client.request({'action': 'enable'})
        client.state(True)
        for pose in accepted:
            target = np.array(pose['q'])
            for duration in (.5, 1.5):
                client.move(target)
                client.request({'action': 'gravity_trial', 'duration_s': duration,
                    'max_excursion_rad': .04, 'max_speed_rad_s': .3})
                samples, deadline = [], time.monotonic()+4
                while True:
                    t = client.request({'action': 'gravity_telemetry', 'include_regressor': False})
                    client.state(True)
                    if not t['feedback_fresh'] or t['arm_control_modes'] != ['MIT']*6:
                        raise RuntimeError('Probe mode/feedback check failed')
                    samples.append(t)
                    if not t['gravity_trial']['active']:
                        break
                    if time.monotonic()>deadline:
                        raise TimeoutError('SDK did not end bounded probe')
                    time.sleep(.04)
                trial = t['gravity_trial']
                client.request({'action': 'mode', 'mode': 'Hold'})
                s = client.state(True)
                result = {'name': pose['name'], 'target_q': pose['q'], 'duration_s': duration,
                    'trial': trial, 'samples': samples, 'after_hold': s}
                results.append(result)
                (dest/'trials.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps({'name': pose['name'], 'duration_s': duration,
                    'reason': trial['reason'], 'peak_deg': float(np.rad2deg(trial['peak_excursion_rad'])),
                    'by_joint_deg': np.rad2deg(trial['peak_excursion_by_joint_rad']).tolist(),
                    'max_temp_c': max(s['temperatures']['mos_c']+s['temperatures']['rotor_c'])}), flush=True)
                if trial['reason'] != 'duration complete' or trial['peak_excursion_rad'] > .015:
                    raise RuntimeError('Stop progression: bounded probe drift/guard acceptance failed')
    finally:
        s = json.loads((client.root/'status.json').read_text())
        if not s['closed']:
            client.request({'action': 'mode', 'mode': 'Hold'})
            client.request({'action': 'disable'})
            deadline = time.monotonic()+90
            while not (client.state()['closed'] and client.state()['rest_verified']):
                if time.monotonic()>deadline:
                    raise TimeoutError('Verified rest disable incomplete; inspect powered hold')
                time.sleep(.1)
            (dest/'final-status.json').write_text(json.dumps(client.state(), indent=2)+'\n')
        client.request({'action': 'gravity_test_envelope', 'excursion_rad': .49})
        if temporary_factors:
            client.request({'action': 'gravity_configure', 'factors': initial['gravity_factors'], 'persist': False})


if __name__ == '__main__':
    tyro.cli(main)
