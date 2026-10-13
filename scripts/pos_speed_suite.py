"""Bounded commissioning via the app inbox; never opens a second CAN connection."""

import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts.stage2_trial import main as waypoint_trial
from scripts.stage3_trial import main as tracking_trial

DEST = Path('evidence/pos-speed')


def healthy():
    s = json.loads(Path('runtime/live/status.json').read_text())
    # A rejected preflight can acknowledge just before the next UI publication.
    # Wait for a fresh status; never dispatch against a stale cached publication.
    deadline = time.monotonic()+3
    while time.time()-s['timestamp'] > 1 and time.monotonic() < deadline:
        time.sleep(.03)
        s = json.loads(Path('runtime/live/status.json').read_text())
    if (time.time()-s['timestamp'] > 1 or s.get('closed') or s.get('fault')
            or s.get('motion_fault') or s.get('controller') != 'POS_VEL'):
        raise RuntimeError(f'Position-speed commissioning stopped: {s.get("fault") or s.get("mode")}')
    return s


def main(stage: int, resume: bool = False):
    """1: free-space joint pairs; 2: Cartesian pairs; 3: all four STEP suites."""
    if stage not in (1, 2, 3):
        raise ValueError('Stage must be 1, 2 or 3')
    healthy()
    base = np.array([.03, .10, .20, .015, .015, .015])
    if stage == 1:
        waypoint_trial('s1-free-approach', tuple(base), evidence_dir=str(DEST))
        for joint in range(6):
            for direction, sign in [('out', 1), ('reverse', 0)]:
                healthy()
                target = base.copy()
                # Exceed the 1° wrist acceptance band so settling is nontrivial.
                target[joint] += (.015 if joint < 3 else .035)*sign
                name = f's1-j{joint+1}-{direction}'
                print(f'TRIAL {name}', flush=True)
                waypoint_trial(name, tuple(target), evidence_dir=str(DEST))
    elif stage == 2:
        for name, target, kind in [
            ('s2-joint-combined', base+np.array([.015,.02,.025,.01,.01,.01]), 'joint'),
            ('s2-cartesian-out', base+np.array([.02,.035,.055,.015,.01,.015]), 'pose'),
            ('s2-cartesian-reverse', base, 'pose')]:
            healthy()
            print(f'TRIAL {name}', flush=True)
            waypoint_trial(name, tuple(target), kind=kind, evidence_dir=str(DEST))
    else:
        dest = DEST/'stage3'
        dest.mkdir(exist_ok=True)
        for suite in ('lines', 'circles', 'perpendicular-lines', 'center-arcs'):
            for index in range(5):
                name = f'{suite}-{index+1:02}-5mms'
                if resume and any((dest/f'{name}{suffix}.json').exists() for suffix in ('','-error')):
                    continue
                healthy()
                print(f'TRIAL {suite} {index+1}', flush=True)
                try:
                    tracking_trial(index=index, suite=suite, evidence_dir=str(dest))
                except Exception as exc:
                    # Local IK/start failures are evidence; hardware faults stop the suite.
                    healthy()
                    print(f'Rejected: {exc}', flush=True)
    print(f'Stage {stage} commissioning run finished', flush=True)


if __name__ == '__main__':
    tyro.cli(main)
