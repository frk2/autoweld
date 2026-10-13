"""The explicitly approved 500 mm sweep. Hardware I/O stays in the SDK app."""

import json
import math
import time
from pathlib import Path

from scripts.pos_speed_suite import healthy
from scripts.stage3_trial import main as trial
from scripts.tune_pid import request

ROOT = Path("runtime/live")
DEST = Path("evidence/long-cruise")


def idle(timeout=40):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        s = healthy()
        if not s["motion_active"] and not (s.get("tracking") or {}).get("running"):
            time.sleep(.3)
            return s
        time.sleep(.05)
    raise TimeoutError("Motion did not finish")


def finish():
    idle()
    request(ROOT, {"action":"position_speed_limit", "limit_rad_s":.15})
    request(ROOT, {"action":"disable"})
    deadline=time.monotonic()+50
    while time.monotonic()<deadline:
        s=json.loads((ROOT/"status.json").read_text())
        if s.get("closed"):
            assert s['rest_verified'] and s['max_error_deg']<=.5
            request(ROOT, {"action":"end_long_cruise_test"})
            for _ in range(30):
                s=json.loads((ROOT/"status.json").read_text())
                if s.get('automatic_test_excursion_rad')==.49:
                    (DEST/'final-rest-status.json').write_text(json.dumps(s,indent=2))
                    print(f"DISABLED AT VERIFIED REST {s['max_error_deg']:.6f} deg; limits restored",flush=True)
                    return
                time.sleep(.05)
            raise RuntimeError('Test excursion restoration not acknowledged')
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError(f"Rest disable fault: {s.get('fault') or s.get('motion_fault')}")
        time.sleep(.05)
    raise TimeoutError('Verified rest disable did not finish')


def main():
    s=healthy()
    assert s.get('automatic_test_excursion_rad')==1.55
    assert not s['friction_enabled'] and s['oscillation_guard_active']
    candidates=json.loads((DEST/'preflight.json').read_text())
    try:
        for item in candidates:
            assert item['ok'] and item['excursion_rad']<1.55
            feed=item['feed_mm_s']/1000
            name=f"line500-01-{feed*1000:g}mms"
            if (DEST/f'{name}.json').exists():
                raise RuntimeError('Refusing to overwrite an existing live trial')
            idle()
            cap=max(.15, math.ceil(item['plan']['planned_peaks']['joint_velocity']*1.25/.05)*.05)
            request(ROOT,{'action':'position_speed_limit','limit_rad_s':cap})
            print(f'TRIAL {name}; joint ceiling {cap:g}',flush=True)
            trial(index=0,feed=feed,name=name,evidence_dir=str(DEST),
                  step_path=str(DEST/'assets/line-500mm.step'),ramp_placement='Inside edge',
                  seed_q=tuple(item['seed_q'][:6]))
            result=json.loads((DEST/f'{name}.json').read_text())['result']
            if not result['completed']:
                print(f"STOP HIGHER FEEDS: {result['reason']}",flush=True)
                break
    except Exception:
        # A live SDK fault requires inspection/recovery, not an automatic retry.
        s=json.loads((ROOT/'status.json').read_text())
        if not s.get('closed') and not s.get('fault') and not s.get('motion_fault'):
            request(ROOT,{'action':'mode','mode':'Hold'})
            finish()
        raise
    finish()


if __name__=='__main__':
    main()
