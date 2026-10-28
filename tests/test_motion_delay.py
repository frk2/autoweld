import numpy as np
import pytest

from autoweld.motion_delay import DelayCalibrationRejected, fit_motion_delay


def scene(delay=.037, noise=1e-5):
    def position(t):
        return .003*np.sin(2*np.pi*t/1.6)+.0006*np.sin(2*np.pi*t/.57)
    outputs = np.arange(0., 9., .003)
    frames = np.arange(.5, 8.5, .03)
    z = position(frames)
    depth = .140-z+5*z*z+np.random.default_rng(2).normal(0, noise, len(z))
    return frames, depth, np.floor(frames/2).astype(int), outputs, position(outputs-delay)


@pytest.mark.parametrize('delay', [.037, -.023])
def test_recovers_signed_unknown_delay_without_receipt_or_command_data(delay):
    fit = fit_motion_delay(*scene(delay))
    assert abs(fit.delay_s-delay) < .001
    assert fit.confidence_interval_s[0] <= delay <= fit.confidence_interval_s[1]
    assert fit.uncertainty_s <= .002
    assert fit.heldout_passed and fit.heldout_rms_m < .0001


def test_flat_unobservable_surface_response_cannot_qualify_a_clock():
    t, depth, cycles, outputs, z = scene()
    with pytest.raises(DelayCalibrationRejected, match='observable'):
        fit_motion_delay(t, np.full_like(depth, .14), cycles, outputs, z)


def test_changing_surface_fails_independent_cycle_validation():
    t, depth, cycles, outputs, z = scene()
    depth = depth + (cycles % 2)*.001
    with pytest.raises(DelayCalibrationRejected):
        fit_motion_delay(t, depth, cycles, outputs, z)


def test_missing_measured_brackets_and_one_way_probe_reject():
    t, depth, cycles, outputs, z = scene()
    with pytest.raises(DelayCalibrationRejected, match='bracket'):
        fit_motion_delay(t, depth, cycles, outputs[300:], z[300:])
    with pytest.raises(DelayCalibrationRejected, match='both directions'):
        fit_motion_delay(t, .14-.001*t, cycles, outputs, .001*outputs)


def test_cycle_correlated_latency_cannot_hide_behind_pixel_noise_confidence():
    t, depth, cycles, outputs, z = scene()
    varying_delay = np.where(cycles % 2, .012, -.012)
    depth = .140-np.interp(t+.037+varying_delay, outputs, z)
    with pytest.raises(DelayCalibrationRejected, match='confidence interval'):
        fit_motion_delay(t, depth, cycles, outputs, z)
