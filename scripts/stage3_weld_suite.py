"""Authorized oriented Stage 3 trials at the user's selected safe feeds."""
import json
import math
import time
from pathlib import Path

from autoweld.tracking_limits import SELECTED_SAFE_FEED_M_S
from scripts.pos_speed_100 import wait_idle
from scripts.pos_speed_suite import healthy
from scripts.stage3_trial import main as trial
from scripts.tune_pid import request

ROOT=Path('runtime/live')
DEST=Path('evidence/stage3-welding')


def finish():
    wait_idle()
    request(ROOT,{'action':'position_speed_limit','limit_rad_s':.15})
    request(ROOT,{'action':'disable'})
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        s=json.loads((ROOT/'status.json').read_text())
        if s['closed']:
            assert s['rest_verified'] and s['max_error_deg']<=.5 and s['automatic_test_excursion_rad']==.49
            (DEST/'final-rest-status.json').write_text(json.dumps(s,indent=2))
            print(f"DISABLED AT VERIFIED REST {s['max_error_deg']:.6f} deg",flush=True)
            return
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError('Fault during rest disable')
        time.sleep(.05)
    raise TimeoutError('Rest verification did not finish')


def main():
    s=healthy()
    assert s['automatic_test_excursion_rad']==.49 and not s['friction_enabled'] and s['oscillation_guard_active']
    try:
        for item in json.loads((DEST/'preflight.json').read_text()):
            name=item['name']+f"-{item['feed_mm_s']:g}mms"
            if not item['ok']:
                print('SKIPPED',name,item['error'],flush=True)
                continue
            if (DEST/f'{name}.json').exists():
                print('RECORDED; not repeated:',name,flush=True)
                continue
            feed=item['feed_mm_s']/1000
            assert feed==SELECTED_SAFE_FEED_M_S['line' if item['suite']=='perpendicular-lines' else 'arc']
            wait_idle()
            cap=max(.15,math.ceil(item['plan']['planned_peaks']['joint_velocity']*1.25/.05)*.05)
            request(ROOT,{'action':'position_speed_limit','limit_rad_s':cap})
            print('TRIAL',name,'joint ceiling',cap,flush=True)
            trial(index=0,feed=feed,name=name,suite=item['suite'],evidence_dir=str(DEST),
                  step_path=item['asset'],ramp_placement='Inside edge',seed_q=tuple(item['seed_q'][:6]),settle_hold_s=2.)
            result=json.loads((DEST/f'{name}.json').read_text())['result']
            if not result['completed']:
                healthy()
                print('PATH STOPPED; next independently preflighted fixture:',result['reason'],flush=True)
    except Exception:
        s=json.loads((ROOT/'status.json').read_text())
        if not s['closed'] and not s.get('fault') and not s.get('motion_fault'):
            request(ROOT,{'action':'mode','mode':'Hold'})
            finish()
        raise
    finish()


if __name__=='__main__':
    main()
