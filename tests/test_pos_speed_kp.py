import numpy as np
import pytest

from scripts.pos_speed_kp_metrics import summarize
from scripts.pos_speed_kp_registers import check_step


@pytest.mark.parametrize('target', [[65]*6, [float('nan')]*6, [0]*6, [54]*5])
def test_reject_unsafe_gain_steps(target):
    with pytest.raises(ValueError):
        check_step([54]*6, target)


def test_accept_twenty_percent_and_bounded_restore():
    check_step([54]*6, [64.8]*3+[54]*3)
    check_step([64.8]*6, [54]*6)


def trace(leave_band_again=False, gap=False):
    rows = []
    for t in np.arange(.01, 5., .01):
        if gap and 2.0 < t < 2.2:
            continue
        # Six joints enter their individual bands at approximately 2 s.
        error = np.array([.2]*3+[1.]*3) * np.exp(2-t)
        if leave_band_again and 3. < t < 4.5:
            error *= 20
        q = -np.deg2rad(error)
        rows.append(dict(timestamp=1000+t, feedback_timestamps=[1000+t]*6,
                         q=q.tolist(), motion_reference=dict(session='test', elapsed_s=t, duration_s=1.)))
    return rows


def test_joint_settling_uses_feedback_and_sustained_window():
    result = summarize(trace(), [0]*6, [-.1]*6, 'test')
    assert result['synchronized'] and result['overshoot_pass']
    assert result['joints'][0]['settling_s'] == pytest.approx(2., abs=.02)
    assert result['joints'][0]['post_reference_settling_s'] == pytest.approx(1., abs=.02)
    assert not summarize(trace(leave_band_again=True), [0]*6, [-.1]*6, 'test')['all_settled']


def test_gap_cannot_qualify_settling():
    assert not summarize(trace(gap=True), [0]*6, [-.1]*6, 'test')['all_settled']


def test_revised_wrist_overshoot_does_not_relax_proximal_limit():
    rows = trace()
    # Current user limits: .1° on J1-J3 and .5° on J4-J6.
    for r in rows:
        if r['motion_reference']['elapsed_s'] > 3:
            r['q'][4] = float(np.deg2rad(.4))
    result = summarize(rows, [0]*6, [-.1]*6, 'test')
    assert result['overshoot_pass']
    assert result['overshoot_limits_deg'] == [.1, .1, .1, .5, .5, .5]
    rows[-1]['q'][0] = float(np.deg2rad(.11))
    assert not summarize(rows, [0]*6, [-.1]*6, 'test')['overshoot_pass']
    rows[-1]['q'][0] = 0.
    rows[-1]['q'][4] = float(np.deg2rad(.51))
    assert not summarize(rows, [0]*6, [-.1]*6, 'test')['overshoot_pass']


def test_enabled_motor_rejects_register_survey(monkeypatch):
    from types import SimpleNamespace

    import scripts.pos_speed_kp_registers as tuning

    interface = SimpleNamespace(read_motor_status=lambda *args: SimpleNamespace(error_code='0x1', velocity=0))
    monkeypatch.setattr(tuning, 'read_register', lambda *args: pytest.fail('No registers after enabled status'))
    with pytest.raises(RuntimeError, match='disabled'):
        tuning.survey(interface, [(1, 'DM4340')])


def test_failed_independent_readback_stops_further_writes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import scripts.pos_speed_kp_registers as tuning

    writes, closed = [], []
    interface = SimpleNamespace(
        read_motor_status=lambda *args: SimpleNamespace(error_code='0x0', velocity=0),
        close=lambda: closed.append(True))
    monkeypatch.setattr(tuning, 'DMSingleMotorCanInterface', lambda **kwargs: interface)
    monkeypatch.setattr(tuning, 'survey', lambda *args: {i: {'registers': {'KP_APR': 54}} for i in range(1, 8)})
    monkeypatch.setattr(tuning, 'write_register', lambda bus, mid, reg, value: writes.append((mid, reg, value)))
    monkeypatch.setattr(tuning, 'read_register', lambda *args: 54)  # Simulate ignored write.
    real_path = tuning.Path
    monkeypatch.setattr(tuning, 'Path', lambda p: tmp_path/'bus.lock' if p == '/tmp/autoweld-can0.lock' else real_path(p))
    output = tmp_path/'result.json'
    with pytest.raises(RuntimeError, match='read-back mismatch'):
        tuning.main((64.8,)*3+(54.,)*3, (54.,)*6, str(output))
    assert writes == [(1, 'KP_APR', 64.8)] and closed == [True]
    evidence = json.loads(output.read_text())
    assert len(evidence['writes']) == 1 and evidence['flash_written'] is False
    assert 'error' in evidence


def test_accepted_waypoint_waits_for_new_sdk_reference():
    from scripts.pos_speed_kp_trials import wait_reference

    states = iter([{'motion_reference': {'session': 'old'}},
                   {'motion_reference': {'session': 'new', 'duration_s': 3.}}])
    assert wait_reference('old', {'motion_reference': None}, read=lambda: next(states)) == {
        'session': 'new', 'duration_s': 3.}


def test_normal_disable_does_not_require_tighter_test_start_tolerance(tmp_path, monkeypatch):
    import scripts.pos_speed_kp_trials as trials

    sent = []
    states = iter([{'closed': False}, {'closed': True, 'rest_verified': True, 'max_error_deg': .2}])
    monkeypatch.setattr(trials, 'state', lambda: next(states))
    monkeypatch.setattr(trials, 'healthy', lambda: {'motion_active': False, 'max_error_deg': .2})
    monkeypatch.setattr(trials, 'wait_idle', lambda: pytest.fail('Must use SDK normal-return checks'))
    monkeypatch.setattr(trials, 'send', lambda payload: sent.append(payload))
    trials.disable(tmp_path)
    assert sent == [{'action': 'disable'}]
