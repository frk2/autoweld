"""Authorized matched 60 mm line tests via SDK app; no separate CAN connection."""

import json
import time
from pathlib import Path

from scripts.pos_speed_suite import healthy
from scripts.stage3_trial import main as trial
from scripts.tune_pid import request

ROOT = Path('runtime/live')
DEST = Path('evidence/pos-speed-faster')


def main():
    healthy()
    for feed in (.01, .02, .005):
        for index in (0, 2, 3, 4):
            healthy()
            name = f'line60-{index+1:02}-{feed*1000:g}mms'
            print(f'TRIAL {name}', flush=True)
            try:
                trial(index=index, feed=feed, name=name,
                      evidence_dir=str(DEST),
                      step_path=str(DEST/'assets/lines-60mm.step'))
            except Exception:
                # Preflight/start rejection is recorded; SDK faults abort immediately.
                healthy()
                print(f'REJECTED {name}; inspect recorded diagnostics', flush=True)
    healthy()
    request(ROOT, {'action': 'disable'})
    deadline = time.monotonic()+40
    while time.monotonic() < deadline:
        s = json.loads((ROOT/'status.json').read_text())
        if s.get('closed'):
            assert s['rest_verified'] and s['max_error_deg'] <= .5
            (DEST/'final-rest-status.json').write_text(json.dumps(s,indent=2))
            print(f'DISABLED AT VERIFIED REST: {s["max_error_deg"]:.6f} deg', flush=True)
            return
        if s.get('fault') or s.get('motion_fault'):
            raise RuntimeError(f'Rest disable failed: {s.get("fault") or s.get("motion_fault")}')
        time.sleep(.05)
    raise TimeoutError('Verified rest disable did not finish; inspect before continuing')


if __name__ == '__main__':
    main()
