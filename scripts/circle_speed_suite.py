"""User-authorized circle sweep through the SDK app, retaining original guards."""
import json
import math
import time
from pathlib import Path

from scripts.pos_speed_100 import wait_idle
from scripts.pos_speed_suite import healthy
from scripts.stage3_trial import main as trial
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/circle-speed')


def finish():
    wait_idle()
    request(ROOT, {'action':'position_speed_limit','limit_rad_s':.15})
    request(ROOT, {'action':'disable'})
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        s=json.loads((ROOT/'status.json').read_text())
        if s['closed']:
            assert s['rest_verified'] and s['max_error_deg']<=.5
            assert s['automatic_test_excursion_rad']==.49
            (DEST/'final-rest-status.json').write_text(json.dumps(s,indent=2))
            print(f"DISABLED AT VERIFIED REST {s['max_error_deg']:.6f} deg; ceiling restored",flush=True)
            return
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError('SDK fault during rest return; inspect before recovery')
        time.sleep(.05)
    raise TimeoutError('Verified rest disable did not finish')


def main():
    s=healthy()
    assert s['automatic_test_excursion_rad']==.49
    assert s['oscillation_guard_active'] and not s['friction_enabled']
    stopped=False
    try:
        for item in json.loads((DEST/'preflight.json').read_text()):
            feed=item['feed_mm_s']/1000
            name=f'circle60-{feed*1000:g}mms'
            if (DEST/f'{name}.json').exists():
                raise RuntimeError('Refusing to overwrite a recorded live trial')
            if not item['ok'] or stopped:
                reason=item.get('error') if not item['ok'] else 'Lower-speed circle exceeded the 5 mm edge guard'
                (DEST/f'{name}-skipped.json').write_text(json.dumps({'reason':reason,'preflight':item},indent=2))
                print(f'SKIP {name}: {reason}',flush=True)
                continue
            assert item['max_rest_excursion_rad']<.475 and item['max_joint_span_rad']<.475
            wait_idle()
            cap=max(.15,math.ceil(item['plan']['planned_peaks']['joint_velocity']*1.25/.05)*.05)
            request(ROOT,{'action':'position_speed_limit','limit_rad_s':cap})
            print(f'TRIAL {name}; joint ceiling {cap:g}',flush=True)
            trial(index=0,feed=feed,name=name,suite='circles',evidence_dir=str(DEST),
                  step_path=str(DEST/'assets/circle.step'),ramp_placement='Inside edge',
                  seed_q=tuple(item['seed_q'][:6]))
            result=json.loads((DEST/f'{name}.json').read_text())['result']
            if not result['completed']:
                stopped=True
                print(f"STOP HIGHER FEEDS: {result['reason']}",flush=True)
    except Exception:
        s=json.loads((ROOT/'status.json').read_text())
        if not s['closed'] and not s.get('fault') and not s.get('motion_fault'):
            request(ROOT,{'action':'mode','mode':'Hold'})
            finish()
        raise
    finish()


if __name__=='__main__':
    main()
