"""Ten fixed, bounded POS_VEL waypoints and disabled-only offline preflight."""

import fcntl
import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts.pos_speed_kp_metrics import summarize
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/pos-speed-kp')


def state():
    s = json.loads((ROOT / 'status.json').read_text())
    if time.time() - s['timestamp'] > 1:
        raise RuntimeError('App status is stale')
    return s


def send(payload):
    before = time.time()
    request(ROOT, payload)
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        s = json.loads((ROOT / 'status.json').read_text())
        if time.time() - s['timestamp'] < 1 and (s.get('last_request') or {}).get('timestamp', 0) >= before:
            return s
        time.sleep(.03)
    raise RuntimeError('No fresh post-request status')


def healthy():
    s = state()
    if s['closed'] or s['fault'] or s['motion_fault'] or s['controller'] != 'POS_VEL':
        raise RuntimeError('SDK/controller state interrupted acquisition')
    if not s['oscillation_guard_active'] or s['friction_enabled']:
        raise RuntimeError('Required guard/configuration changed')
    if s['automatic_test_excursion_rad'] != .49 or s['position_speed_limit_rad_s'] != .15:
        raise RuntimeError('Automatic bounds or speed ceiling changed')
    return s


def wait_reference(previous_session, published, read=healthy):
    """An accepted app request can precede the SDK worker's first reference."""
    deadline = time.monotonic() + 1.
    while True:
        ref = published.get('motion_reference') or {}
        if ref.get('session') and ref['session'] != previous_session:
            return ref
        if time.monotonic() >= deadline:
            raise RuntimeError('Accepted waypoint has no new SDK reference after one second')
        time.sleep(.01)
        published = read()


def wait_idle():
    deadline, since = time.monotonic() + 25, None
    while time.monotonic() < deadline:
        s = healthy()
        if not s['motion_active'] and s['max_error_deg'] < .1:
            since = since or time.monotonic()
            if time.monotonic() - since >= 1.5:
                return s
        else:
            since = None
        time.sleep(.05)
    raise RuntimeError('Did not settle sufficiently for the next matched waypoint')


def disable(dest):
    if state()['closed']:
        return
    # The stricter matched-test starting tolerance must not prevent normal rest
    # return. The SDK still independently enforces fresh, low-speed feedback.
    deadline = time.monotonic() + 30
    while healthy()['motion_active']:
        if time.monotonic() >= deadline:
            raise RuntimeError('Reference did not finish before normal return')
        time.sleep(.05)
    for attempt in range(5):
        try:
            send({'action': 'disable'})
            break
        except RuntimeError as exc:
            if 'Wait for the arm to stop' not in str(exc) or attempt == 4:
                raise
            time.sleep(1.)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        s = state()
        if s['closed']:
            if not s['rest_verified'] or s['max_error_deg'] > .5:
                raise RuntimeError('Normal rest verification failed')
            (dest / 'rest-status.json').write_text(json.dumps(s, indent=2))
            print('DISABLED verified rest', s['max_error_deg'], flush=True)
            return
        if s['fault'] or s['motion_fault']:
            raise RuntimeError('SDK fault during return; inspect before recovery')
        time.sleep(.05)
    raise RuntimeError('Timed out waiting for normal rest disable')


def preflight():
    from dataclasses import replace

    import mujoco
    from i2rt.utils.viser_control_interface import ViserControlInterface

    from autoweld.app import tool_model
    from autoweld.motion import MotionLimits, MotionPlanner

    record = json.loads((ROOT / 'rest.json').read_text())
    rest = np.r_[record['arm_rad'], record['gripper_normalized']]
    center = rest[:6] + [0, .16, .24, .06, 0, 0]
    offsets = np.array([
        [.04, .04, .05, .10, .08, .09],
        [-.04, -.04, -.05, -.08, -.09, -.10],
        [.03, .05, .04, .09, .10, .08],
        [-.05, -.03, -.04, -.10, -.07, -.09],
        [.05, .03, .06, .08, .09, .11],
        [-.03, -.05, -.03, -.09, -.10, -.08],
        [.04, .06, .03, .11, .07, .10],
        [-.04, -.04, -.05, -.07, -.09, -.11],
        [.06, .04, .05, .09, .11, .07],
        [-.03, -.03, -.04, -.11, -.08, -.10],
    ])
    initial = np.r_[center-offsets[0], rest[6]]
    targets = [np.r_[center+x, rest[6]] for x in offsets]
    model = tool_model()
    try:
        p = MotionPlanner(model)
        p.limits = replace(MotionLimits(**json.loads((ROOT/'motion-limits.json').read_text())), joint_velocity=.15)
        checker = ViserControlInterface.__new__(ViserControlInterface)
        checker._model, checker._check_data = p.model, mujoco.MjData(p.model)
        bounds = p.model.jnt_range[:6].copy()
        bounds[:, 0], bounds[:, 1] = np.minimum(bounds[:, 0], rest[:6]), np.maximum(bounds[:, 1], rest[:6])

        def collision(q):
            return checker._has_self_collision(q, len(q))

        def plan(a, b):
            def validate(q):
                if max(np.max(abs(q[:6]-rest[:6])), np.max(abs(q[:6]-a[:6]))) >= .48:
                    raise ValueError('Preflight automatic excursion margin exceeded')
            return p.plan(a, b, bounds=bounds, collision=collision, validate=validate)

        result = dict(initial_q=initial.tolist(), rest=rest.tolist(), tests=[],
                      approach=plan(rest, initial).report, position_speed_ceiling_rad_s=.15,
                      criteria='max/min per-joint stable arrival <= 1.20; tolerances .2/.2/.2/1/1/1 degrees; '
                      'one second sustained in-band; directed overshoot < .1 degrees J1-J3, < .5 degrees J4-J6')
        previous = initial
        for i, goal in enumerate(targets, 1):
            motion = plan(previous, goal)
            result['tests'].append(dict(name=f'pose-{i:02d}', q=goal.tolist(),
                                        reference_start_q=previous.tolist(), plan=motion.report,
                                        return_plan=plan(goal, rest).report))
            previous = goal
        (DEST / 'preflight.json').write_text(json.dumps(result, indent=2))
        print('All ten paths, approach and returns passed offline preflight', flush=True)
    finally:
        Path(model).unlink()


def acquire(dest, name, target):
    before = wait_idle()
    log = ROOT / 'control-telemetry.jsonl'
    offset = log.stat().st_size
    start = time.time()
    record = dict(before=before, target=target, requested_at=start, telemetry_offset=offset)
    (dest / f'{name}.json').write_text(json.dumps(record, indent=2))
    try:
        after_send = send({'action': 'joint', 'q': target[:6]})
        reference = wait_reference((before.get('motion_reference') or {}).get('session'), after_send)
        session = reference['session']
        duration = reference['duration_s']
        deadline = time.monotonic() + duration + 6.
        while time.monotonic() < deadline:
            s = healthy()
            if (s.get('motion_reference') or {}).get('session') != session:
                raise RuntimeError('Waypoint was replaced during acquisition')
            time.sleep(.05)
        record['after'] = healthy()
    finally:
        with log.open() as f:
            f.seek(offset)
            rows = [json.loads(line) for line in f if line.endswith('\n')]
        (dest / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (dest / f'{name}.json').write_text(json.dumps(record, indent=2))
    result = summarize(rows, target[:6], before['q_rad'], session)
    record['metrics'] = result
    (dest / f'{name}.json').write_text(json.dumps(record, indent=2))
    print(name, 'settling', [round(j['settling_s'], 3) if j['settling_s'] else None for j in result['joints']],
          'ratio', result['arrival_ratio'], 'overshoot', max(j['overshoot_deg'] for j in result['joints']), flush=True)
    return result


def run(round_name: str, count: int = 10):
    if not 1 <= count <= 10:
        raise ValueError('Expected 1 to 10 tests')
    config = json.loads((DEST / 'preflight.json').read_text())
    dest = DEST / round_name
    dest.mkdir(exist_ok=False)
    with (ROOT / 'tuning.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not state()['closed']:
            raise RuntimeError('Start each round with motors disabled')
        send({'action': 'controller', 'controller': 'POS_VEL'})
        results = []
        try:
            send({'action': 'enable'})
            s = healthy()
            (dest / 'enabled-status.json').write_text(json.dumps(s, indent=2))
            # Enabling seeds the current encoder pose; permit it to become quiet.
            time.sleep(2.)
            send({'action': 'joint', 'q': config['initial_q'][:6]})
            wait_idle()
            for item in config['tests'][:count]:
                result = acquire(dest, item['name'], item['q'])
                results.append(dict(name=item['name'], **result))
                (dest / 'metrics.json').write_text(json.dumps(results, indent=2))
                if not result['overshoot_pass'] or not result['all_settled']:
                    raise RuntimeError('Candidate failed overshoot or sustained-settling criteria; stop round')
        except Exception as exc:
            (dest / 'error.json').write_text(json.dumps(dict(error=str(exc), state=state()), indent=2))
            raise
        finally:
            # A latched SDK fault must be inspected; never automatically clear it.
            s = state()
            if not s.get('fault') and not s.get('motion_fault'):
                disable(dest)
        return results


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'preflight': preflight, 'run': run})
