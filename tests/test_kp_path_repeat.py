import json

import pytest

from scripts import kp_path_repeat as repeat
from scripts import tune_pid


def test_large_exception_is_fixture_specific():
    item = dict(ok=True, automatic_limit_rad=1.55, suite='lines', index=0,
                asset='evidence/long-cruise/assets/line-500mm.step', cap_rad_s=.5)
    state = dict(closed=True, automatic_test_excursion_rad=1.55,
                 motion_limits={'joint_velocity': 1.})
    repeat.check_fixture(item, state)
    with pytest.raises(ValueError, match='reviewed 500 mm'):
        repeat.check_fixture({**item, 'asset': 'another.step'}, state)
    with pytest.raises(ValueError, match='disabled'):
        repeat.check_fixture(item, {**state, 'closed': False})


def test_rejected_preflight_cannot_run():
    with pytest.raises(ValueError, match='accepted preflight'):
        repeat.check_fixture({'ok': False}, {'closed': True})


def test_gain_reset_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(repeat, 'DEST', tmp_path)
    gains = [193.49, 193.49, 193.49, 54., 77.76, 64.8]
    (tmp_path / 'selected-profile.json').write_text(json.dumps({'kp_apr': gains}))
    survey = {str(i): {'KP_APR': gains[i-1] if i < 7 else 54., 'status_before': '0x0'}
              for i in range(1, 8)}
    repeat.check_gains({'motor_mode_survey': survey})
    survey['2']['KP_APR'] = 54.
    with pytest.raises(RuntimeError, match='differ from selection'):
        repeat.check_gains({'motor_mode_survey': survey})


def test_planning_reply_wait_does_not_resubmit(tmp_path, monkeypatch):
    (tmp_path / 'requests').mkdir()
    (tmp_path / 'responses').mkdir()
    clock = [0.]
    monkeypatch.setattr(tune_pid.time, 'monotonic', lambda: clock[0])

    def sleep(_):
        clock[0] += 1
        if clock[0] == 6:
            submitted = list((tmp_path / 'requests').glob('*.json'))
            assert len(submitted) == 1
            (tmp_path / 'responses' / submitted[0].name).write_text('{"ok": true}')

    monkeypatch.setattr(tune_pid.time, 'sleep', sleep)
    tune_pid.request(tmp_path, {'action': 'track_approach'}, timeout_s=30)
    assert clock[0] == 6


def test_unanswered_request_still_stops_without_retry(tmp_path, monkeypatch):
    (tmp_path / 'requests').mkdir()
    (tmp_path / 'responses').mkdir()
    clock = [0.]
    monkeypatch.setattr(tune_pid.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(tune_pid.time, 'sleep', lambda _: clock.__setitem__(0, clock[0]+1))
    with pytest.raises(TimeoutError, match='inspect state'):
        tune_pid.request(tmp_path, {'action': 'joint'})
    assert len(list((tmp_path / 'requests').glob('*.json'))) == 1
