"""Run one preflighted center-facing arc, then disable at verified rest."""
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import tyro
from i2rt.utils.viser_control_interface import ViserControlInterface

import scripts.stage3_weld_suite as rest_return
from autoweld.app import tool_model
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.tracking import load_tracks, plan_tracking
from scripts.pos_speed_suite import healthy
from scripts.stage3_trial import main as trial
from scripts.stage3_weld_suite import finish as rest_disable
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/small-oriented-arcs')


def finish(name):
    """Observe settling after a tracking Hold before requesting a return."""
    observed = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        s = healthy()
        observed.append(s)
        recent = [x for x in observed if s['timestamp'] - x['timestamp'] <= 2.]
        if (len(recent) > 10 and s['timestamp'] - recent[0]['timestamp'] >= 1.8
                and not s['motion_active']
                and np.max(np.ptp([x['q_rad'] for x in recent], axis=0)) < np.deg2rad(.1)
                and np.max(abs(np.array(s['q_rad']) - s['target_rad'])) < .005):
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Arm did not settle for normal return; left in Hold')
    (DEST / f'{name}-settle.json').write_text(json.dumps(observed, indent=2))
    # The app still independently checks raw speed, fresh feedback and rest.
    rest_disable()


def main(index: int = 3, suffix: str = 'trial1', feed_mm_s: float | None = None):
    item = json.loads((DEST / 'preflight.json').read_text())[index]
    before = json.loads((ROOT / 'status.json').read_text())
    assert before['closed'] and before['automatic_test_excursion_rad'] == .49
    if feed_mm_s is not None:
        model = tool_model()
        try:
            p = MotionPlanner(model)
            p.limits = MotionLimits(**before['motion_limits'])
            rest = np.array(before['rest_rad'])
            bounds = p.model.jnt_range[:6].copy()
            bounds[:, 0] = np.minimum(bounds[:, 0], rest)
            bounds[:, 1] = np.maximum(bounds[:, 1], rest)
            checker = ViserControlInterface.__new__(ViserControlInterface)
            checker._model = p.model
            checker._check_data = mujoco.MjData(p.model)

            def collision(q):
                return checker._has_self_collision(q, len(q))

            def validate(q):
                if np.max(abs(q[:6] - rest)) >= .475:
                    raise ValueError('Automatic rest margin exceeded')

            plan = plan_tracking(p, item['seed_q'], load_tracks(item['asset'])[0],
                                 feed_mm_s / 1000, 'Center facing', bounds=bounds,
                                 collision=collision, validate=validate, ramp_placement='Inside edge')
            dense = plan.motion.path.spline(np.linspace(0, 1, 1001))[:, :6]
            assert np.max(np.ptp(dense, axis=0)) < .475
            item = {**item, 'feed_mm_s': feed_mm_s, 'plan': plan.motion.report}
        finally:
            Path(model).unlink()
    name = f"{item['name']}-{item['feed_mm_s']:g}mms-{suffix}"
    if any((DEST / f'{name}{ext}').exists() for ext in ('.json', '-error.json')):
        raise ValueError('Evidence already exists; choose a new suffix')
    (DEST / f'{name}-preflight.json').write_text(json.dumps(item, indent=2))
    request(ROOT, {'action': 'controller', 'controller': 'POS_VEL'})
    request(ROOT, {'action': 'enable'})
    rest_return.DEST = DEST
    try:
        s = healthy()
        assert not s['friction_enabled'] and s['oscillation_guard_active']
        cap = max(.15, math.ceil(item['plan']['planned_peaks']['joint_velocity'] * 1.25 / .05) * .05)
        request(ROOT, {'action': 'position_speed_limit', 'limit_rad_s': cap})
        print('TRIAL', name, 'joint ceiling', cap, flush=True)
        trial(index=0, feed=item['feed_mm_s'] / 1000, name=name, suite='center-arcs',
              evidence_dir=str(DEST), step_path=item['asset'], ramp_placement='Inside edge',
              seed_q=tuple(item['seed_q'][:6]), settle_hold_s=2.)
    except Exception:
        s = json.loads((ROOT / 'status.json').read_text())
        if not s['closed'] and not s.get('fault') and not s.get('motion_fault'):
            request(ROOT, {'action': 'mode', 'mode': 'Hold'})
            finish(name)
        raise
    finish(name)
    (DEST / f'{name}-rest.json').write_text((DEST / 'final-rest-status.json').read_text())


if __name__ == '__main__':
    tyro.cli(main)
