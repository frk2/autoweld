"""Exercise the running offline JSON service. Never opens robot/camera hardware."""
import json
import time
from pathlib import Path

import numpy as np
import tyro


def main(runtime: str = 'runtime/sim'):
    root = Path(runtime)

    def state():
        result = json.loads((root / 'status.json').read_text())
        assert result.get('live') is False, 'Smoke test refuses live hardware'
        assert time.time() - result['timestamp'] < 2, 'Offline service is stale'
        return result

    def request(payload):
        state()
        submitted = time.time()
        name = f'smoke-{time.time_ns()}'
        path = root / 'requests' / f'{name}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload))
        temporary.replace(path)
        response = root / 'responses' / path.name
        deadline = time.monotonic() + 30
        while not response.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(payload)
            time.sleep(.05)
        result = json.loads(response.read_text())
        while (state().get('last_request') or {}).get('timestamp', 0) < submitted:
            if time.monotonic() > deadline:
                raise TimeoutError('Response was not published')
            time.sleep(.05)
        while state().get('motion_active') or state().get('disable_pending'):
            if time.monotonic() > deadline:
                raise TimeoutError('Motion/disable did not finish')
            time.sleep(.05)
        return result

    if state()['closed']:
        assert request({'action': 'enable'})['ok']
    baseline = state()['q_rad']
    assert request({'action': 'pose_save', 'name': 'Smoke baseline'})['ok']
    assert request({'action': 'mode', 'mode': 'IK'})['ok']
    target = baseline.copy()
    target[0] += .01
    assert request({'action': 'joint', 'q': target})['ok']
    np.testing.assert_allclose(state()['q_rad'], target, atol=1e-6)
    assert request({'action': 'pose_recall', 'name': 'Smoke baseline'})['ok']
    np.testing.assert_allclose(state()['q_rad'], baseline, atol=1e-6)
    before = state()['target_rad']
    assert not request({'action': 'pose', 'position': [10, 10, 10], 'rpy': [0, 0, 0]})['ok']
    np.testing.assert_array_equal(state()['target_rad'], before)
    assert not request({'action': 'mode', 'mode': 'Gravity compensation'})['ok']
    for _ in range(2):
        assert request({'action': 'disable'})['ok']
        assert state()['closed'] and state()['rest_verified']
        if _ == 0:
            assert request({'action': 'enable'})['ok']
    print('PASS: offline JSON motion, saved poses, invalid target, mode policy and verified-rest reconnect')


if __name__ == '__main__':
    tyro.cli(main)
