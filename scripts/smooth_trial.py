"""Small-angle tuning helpers; hardware communication is only through app/SDK."""
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from scripts.tune_pid import main as run_trial
from scripts.tune_pid import request

ROOT = Path('runtime/live')
OUT = Path('evidence/smooth')


def state():
    s = json.loads((ROOT/'status.json').read_text())
    if time.time()-s['timestamp'] > 1 or s.get('fault') or s.get('closed'):
        raise RuntimeError(f'Need fresh, healthy enabled app: {s.get("fault")}')
    return s


def configure_friction(enabled, width=0.):
    state()
    request(ROOT, {'action': 'friction_feedforward', 'enabled': enabled,
                   'transition_rad_s': [width]*6})
    time.sleep(.25)
    assert state()['friction']['enabled'] == enabled


def trial(name, delta, *, ki=None, kp=None, kd=None, gravity=None, duration=4.):
    s = state()
    name = 'smooth-'+name
    if Path(f'evidence/pid/{name}.json').exists():
        raise RuntimeError('Trial names must be unique')
    with contextlib.redirect_stdout(io.StringIO()):
        run_trial(name=name, delta=tuple(delta), duration=duration,
                  ki=tuple(s['ki'] if ki is None else ki), kp=kp, kd=kd,
                  gravity_factors=gravity, gravity_model=s['gravity_model'],
                  gravity_calibration=not s['friction']['enabled'] and not any(s['ki'] if ki is None else ki),
                  friction_compensation=s['friction']['enabled'])
    data = json.loads(Path(f'evidence/pid/{name}.json').read_text())
    print(json.dumps({'name': name, 'stable_s': data['settle_and_remain_s'],
                      'overshoot_deg': np.round(data['overshoot_deg'], 3).tolist(),
                      'tail_error_deg': np.round(data['final_2s_mean_error_deg'], 3).tolist(),
                      'tail_jitter_deg': np.round(data['final_2s_peak_to_peak_deg'], 3).tolist(),
                      'pass': data['speed_and_overshoot_pass']}), flush=True)
    return data
