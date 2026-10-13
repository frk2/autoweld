"""Authorized 20–100 mm/s sweep; SDK app only, with ramps inside finite edges."""

import json
import math
import time
from pathlib import Path

import tyro

from scripts.pos_speed_suite import healthy
from scripts.stage3_trial import main as trial
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/pos-speed-100')


def wait_idle(timeout=35):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        s = healthy()
        if not s['motion_active'] and not (s.get('tracking') or {}).get('running'):
            # The SDK also rejects ceiling changes until measured velocity is low.
            time.sleep(.3)
            return
        time.sleep(.05)
    raise TimeoutError('Motion did not finish')


def main(resume: bool = False):
    healthy()
    candidates = json.loads((DEST/'preflight.json').read_text())
    stopped = set()
    for item in candidates:
        index = item['index']
        feed = item['feed_mm_s']/1000
        name = f'line120-{index+1:02}-{feed*1000:g}mms'
        if resume and (DEST/f'{name}.json').exists():
            previous = json.loads((DEST/f'{name}.json').read_text())['result']
            if not previous['completed']:
                stopped.add(index)
            continue
        if not item['ok'] or index in stopped:
            (DEST/f'{name}-skipped.json').write_text(json.dumps({
                'preflight':item, 'reason':item.get('error') or 'Prior edge-distance stop in this direction'
            },indent=2))
            continue
        wait_idle()
        peak = item['plan']['planned_peaks']['joint_velocity']
        cap = max(.15, math.ceil(peak*1.25/.05)*.05)
        request(ROOT, {'action':'position_speed_limit','limit_rad_s':cap})
        print(f'TRIAL {name}; joint ceiling {cap:g}', flush=True)
        try:
            trial(index=index, feed=feed, name=name, evidence_dir=str(DEST),
                  step_path=str(DEST/'assets/lines-120mm.step'), ramp_placement='Inside edge')
            result = json.loads((DEST/f'{name}.json').read_text())['result']
            if not result['completed']:
                stopped.add(index)
                print(f'STOP higher feeds in direction {index+1}: {result["reason"]}', flush=True)
        except Exception:
            healthy()  # Hardware/guard faults end the sweep immediately.
            print(f'REJECTED {name}; see diagnostic', flush=True)
    wait_idle()
    request(ROOT, {'action':'position_speed_limit','limit_rad_s':.15})
    request(ROOT, {'action':'disable'})
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        s=json.loads((ROOT/'status.json').read_text())
        if s.get('closed'):
            assert s['rest_verified'] and s['max_error_deg']<=.5
            (DEST/'final-rest-status.json').write_text(json.dumps(s,indent=2))
            print(f'DISABLED AT VERIFIED REST {s["max_error_deg"]:.6f} deg; ceiling restored .15',flush=True)
            return
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError(f'Rest disable failed: {s.get("fault") or s.get("motion_fault")}')
        time.sleep(.05)
    raise TimeoutError('Rest disable did not finish')


if __name__ == '__main__':
    tyro.cli(main)
