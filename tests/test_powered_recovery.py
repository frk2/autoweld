"""Recovery policy is offline: no SDK construction, CAN socket, or process control."""

from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.powered_recovery import MOTORS, RecoveryController


class LinearPath:
    duration = .4

    def __init__(self):
        self.start = np.array([.1, -.2, .3, -.4, .5, -2.8, 0.])
        self.end = self.start+np.array([.02, -.01, 0., 0., 0., -.01, 0.])

    def sample(self, elapsed):
        fraction = np.clip(elapsed/self.duration, 0., 1.)
        velocity = (self.end-self.start)/self.duration if 0 <= elapsed < self.duration else np.zeros(7)
        return self.start+(self.end-self.start)*fraction, velocity, np.zeros(7), np.zeros(7)


def frames(q, stamp, **changes):
    result = [dict(id=i+1, position=float(q[i]), velocity=0., timestamp=stamp,
                   temperature_mos=32., temperature_rotor=34., error_code='0x1') for i in range(6)]
    result[0].update(changes)
    return result


@pytest.fixture
def c():
    return RecoveryController(LinearPath(), started_at=10.)


def update(c, now, **changes):
    return c.step(now, frames(c.reference, now, **changes))


def start_path(c):
    for i in range(51):
        result = update(c, 10.+.02*i)
    assert result['stage'] == 'path'
    assert result['path_elapsed_s'] == 0.
    return 11.


def finish_path(c):
    now = start_path(c)
    for i in range(1, 22):
        now = 11.+.02*i
        result = update(c, now)
        if result['stage'] == 'settle':
            return now
    raise AssertionError(result)


def test_initial_hold_and_rest_require_sustained_fresh_quiet_feedback(c):
    now = finish_path(c)
    np.testing.assert_allclose(c.reference, c.path.end[:6])
    for i in range(1, 50):
        result = update(c, now+.02*i)
        assert not result['rest_verified']
    result = update(c, now+1.)
    assert result['stage'] == 'verified_rest' and result['may_disable']
    assert not result['may_resume_old_owner']


def test_hold_quiet_timer_resets_on_motion(c):
    for i in range(26):
        update(c, 10.+i*.02)
    update(c, 10.52, velocity=.06)
    assert c.quiet_elapsed == 0.
    for i in range(1, 50):
        assert update(c, 10.52+i*.02)['stage'] == 'hold'
    assert update(c, 11.52)['stage'] == 'path'


def test_path_clock_caps_progress_instead_of_catching_up(c):
    now = start_path(c)
    result = update(c, now+.09)
    assert result['stage'] == 'path'
    assert result['path_elapsed_s'] == pytest.approx(.02)
    assert c.reference[0] == pytest.approx(c.start[0]+.001)


def test_gap_during_initial_hold_resets_quiet_without_starting_motion(c):
    update(c, 10.)
    update(c, 10.02)
    result = update(c, 11.)
    assert result['stage'] == 'hold'
    assert result['quiet_elapsed_s'] == 0.
    np.testing.assert_array_equal(result['reference_rad'], c.start[:6])


def test_timing_overrun_after_motion_latches_last_reference(c):
    now = start_path(c)
    update(c, now+.02)
    fixed = c.reference.copy()
    result = update(c, now+.121)
    assert result['stage'] == 'failed_hold' and 'timing' in result['error']
    np.testing.assert_array_equal(c.reference, fixed)
    assert update(c, now+10.) == result


@pytest.mark.parametrize(('change', 'message'), [
    ({'id': 7}, 'IDs1-6'),
    ({'error_code': '0x0'}, 'healthy'),
    ({'error_code': '0xd'}, 'healthy'),
    ({'error_code': '0x8'}, 'healthy'),
    ({'position': np.nan}, 'Nonfinite'),
    ({'velocity': np.inf}, 'Nonfinite'),
    ({'timestamp': np.nan}, 'Nonfinite'),
    ({'timestamp': 9.8}, 'Stale'),
    ({'timestamp': 10.01}, 'future'),
    ({'velocity': .35001}, 'speed'),
    ({'position': .14001}, 'tracking'),
])
def test_feedback_guards_latch_fixed_hold_without_power_action(c, change, message):
    fixed = c.reference.copy()
    result = c.step(10., frames(c.reference, 10., **change))
    assert result['stage'] == 'failed_hold' and message in result['error']
    assert not result['may_disable'] and not result['may_resume_old_owner']
    np.testing.assert_array_equal(c.reference, fixed)


@pytest.mark.parametrize('axis', range(6))
def test_guards_cover_every_motor(c, axis):
    feedback = frames(c.reference, 10.)
    feedback[axis]['velocity'] = .36
    assert 'speed' in c.step(10., feedback)['error']


@pytest.mark.parametrize('axis', range(6))
@pytest.mark.parametrize('sensor', ['temperature_mos', 'temperature_rotor'])
def test_hot_motor_requests_cooldown_but_continues_validated_rest_path(c, axis, sensor):
    now = start_path(c)
    feedback = frames(c.reference, now+.02)
    feedback[axis][sensor] = 60.
    result = c.step(now+.02, feedback)
    assert result['cooldown_requested'] and result['error'] is None
    assert result['stage'] == 'path' and result['path_elapsed_s'] == pytest.approx(.02)
    result = update(c, now+.04)
    assert result['cooldown_requested'] and result['stage'] == 'path'


def test_hot_motor_does_not_override_tracking_failure(c):
    now = start_path(c)
    result = update(c, now+.02, temperature_rotor=60., position=c.reference[0]+.041)
    assert result['cooldown_requested'] and result['stage'] == 'failed_hold'
    assert 'tracking' in result['error']


@pytest.mark.parametrize('feedback', [[], [dict(id=i) for i in range(1, 6)], [None]*6])
def test_incomplete_or_malformed_frames_fail_closed_to_powered_hold(c, feedback):
    assert c.step(10., feedback)['stage'] == 'failed_hold'


def test_wallclock_and_sdk_dataclass_frames_supported(c):
    feedback = [SimpleNamespace(**frame) for frame in frames(c.reference, 10000.)]
    result = c.step(10., feedback, feedback_now=10000.01)
    assert result['stage'] == 'hold' and result['error'] is None


def test_repeated_feedback_does_not_advance_path(c):
    now = start_path(c)
    feedback = frames(c.reference, now)
    for offset in (.02, .04, .06, .08):
        result = c.step(now+offset, feedback)
        assert result['path_elapsed_s'] == 0.
    assert c.step(now+.1, feedback)['stage'] == 'path'
    assert c.step(now+.12, feedback)['stage'] == 'failed_hold'


def test_regressing_timestamps_fail(c):
    update(c, 10.)
    result = c.step(10.02, frames(c.reference, 9.99))
    assert 'Regressing' in result['error']


def test_settling_resets_when_not_quiet_or_not_at_rest(c):
    now = finish_path(c)
    for i in range(1, 26):
        update(c, now+.02*i)
    update(c, now+.52, velocity=.051)
    assert c.quiet_elapsed == 0.
    moved = c.reference.copy()
    moved[0] += np.deg2rad(.6)
    result = c.step(now+.54, frames(moved, now+.54))
    assert result['stage'] == 'settle' and c.quiet_elapsed == 0.


def test_verified_rest_loses_authorization_on_new_guard_failure(c):
    now = finish_path(c)
    for i in range(1, 51):
        update(c, now+.02*i)
    assert c.stage == 'verified_rest'
    result = update(c, now+1.02, error_code='0xd')
    assert result['stage'] == 'failed_hold' and not result['may_disable']


def test_bad_preloaded_reference_never_replaces_last_safe_command(c):
    now = start_path(c)
    fixed = c.reference.copy()
    c.path.end[0] += 1.
    result = update(c, now+.02)
    assert result['stage'] == 'failed_hold' and 'speed' in result['error']
    np.testing.assert_array_equal(c.reference, fixed)


def test_abort_is_latched_without_recovery_or_resume(c):
    result = c.abort('Operator abort')
    assert result['stage'] == 'failed_hold' and result['error'] == 'Operator abort'
    assert update(c, 10.02) == result
    assert c.abort('New reason')['error'] == 'Operator abort'


def test_sdk_dispatch_only_calls_six_axis_positive_speed_set_control(c):
    calls = []
    interface = SimpleNamespace(set_control=lambda *args: calls.append(args))
    c.command(interface)
    assert len(calls) == 6
    for axis, ((mid, kind), args) in enumerate(zip(MOTORS, calls, strict=True)):
        assert args == (mid, kind, c.start[axis], .1, 0., 0., 0.)
    c.abort('Keep powered hold')
    c.command(interface)
    assert len(calls) == 12


def test_sdk_dispatch_failure_latches_reference_and_propagates(c):
    def broken(*_):
        raise RuntimeError('Lost reply')
    with pytest.raises(RuntimeError, match='Lost reply'):
        c.command(SimpleNamespace(set_control=broken))
    assert c.stage == 'failed_hold' and 'Lost reply' in c.error


@pytest.mark.parametrize('kwargs', [dict(max_gap_s=.11), dict(max_speed_rad_s=.36),
    dict(speed_cap_rad_s=0.), dict(tracking_error_rad=.041), dict(hold_s=.5),
    dict(settle_s=.5), dict(rest_tolerance_rad=.01), dict(max_frame_age_s=.2),
    dict(max_progress_dt_s=.2), dict(quiet_speed_rad_s=.06), dict(started_at=np.nan)])
def test_invalid_or_relaxed_limits_reject_before_dispatch(kwargs):
    with pytest.raises(ValueError):
        RecoveryController(LinearPath(), **(dict(started_at=10.) | kwargs))


def test_constructor_rejects_nonfixed_dummy_joint():
    path = LinearPath()
    path.end[6] = 1.
    with pytest.raises(ValueError, match='dummy'):
        RecoveryController(path, started_at=10.)


class CapturedRestPath:
    """Captured preflight's two-knot JointPath, without importing the SDK."""
    start = np.array([-1.9178683146410318, 1.095407034409094, .42324711985961727,
                      .5342565041580833, -.12493324177920151, -2.194056610971236, 0.])
    end = np.array([-.05283436331731117, .0036240177004653873, .014686808575570254,
                    -.06504158083466827, -.078774700541695, -2.2741664759288938, 0.])

    def __init__(self, scale):
        self.duration = 48.644584505681955*scale

    def sample(self, elapsed):
        u = min(1., max(0., elapsed/self.duration))
        s = 35*u**4-84*u**5+70*u**6-20*u**7
        ds = (140*u**3-420*u**4+420*u**5-140*u**6)/self.duration
        return self.start+(self.end-self.start)*s, (self.end-self.start)*ds


@pytest.mark.parametrize('lag_model', ['delay', 'first_order'])
@pytest.mark.parametrize(('scale', 'expected'), [(1., 'failed_hold'), (3., 'verified_rest')])
def test_captured_path_with_one_second_lag_requires_threefold_slowdown(scale, expected, lag_model):
    """Keep the tracking guard: slow references rather than accepting more lag."""
    from collections import deque

    path = CapturedRestPath(scale)
    controller = RecoveryController(path, started_at=10., speed_cap_rad_s=.12)
    dt, lag_s, bias = .02, 1., .0032  # Settled bias plus half a packed encoder bin.
    direction = np.sign(path.end[:6]-path.start[:6])
    measured = path.start[:6]-direction*bias
    history = deque([path.start[:6].copy() for _ in range(round(lag_s/dt))])
    peak_error = 0.
    for i in range(int((path.duration+15)/dt)+1):
        if lag_model == 'delay':
            history.append(controller.reference.copy())
            q = history.popleft()-direction*bias
        else:
            q = measured+(controller.reference-direction*bias-measured)*(1-np.exp(-dt/lag_s))
        velocity = (q-measured)/dt
        measured = q
        now = 10.+i*dt
        feedback = frames(q, now)
        for frame, speed in zip(feedback, velocity, strict=True):
            frame['velocity'] = float(speed)
        result = controller.step(now, feedback)
        peak_error = max(peak_error, float(np.max(abs(controller.reference-q))))
        if result['stage'] in ('verified_rest', 'failed_hold'):
            break
    assert result['stage'] == expected
    if scale == 3:
        assert peak_error < .032
        assert controller.tracking_error == .04
    else:
        assert 'tracking' in result['error']
