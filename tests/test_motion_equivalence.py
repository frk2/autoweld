"""Batched dynamic validation preserves the scalar sample sites and arithmetic."""

import numpy as np
import pytest

from autoweld.motion import JointPath, MotionPlanner


@pytest.mark.parametrize('duration,knots', [(.7, 3), (1., 41), (2.3, 17)])
def test_batched_limit_samples_match_scalar_reference_exactly(duration, knots):
    planner = object.__new__(MotionPlanner)
    calls = []
    def jacobian(q):
        calls.append(np.asarray(q)[:6].copy())
        return np.eye(6)+np.outer(np.asarray(q)[:6], np.arange(1., 7.))*.01
    planner.jacobian = jacobian
    phases = np.linspace(0., 1., knots)
    positions = np.zeros((knots, 7))
    for joint in range(6):
        positions[:, joint] = .1*np.sin(phases*(joint+1))+.03*phases
    path = JointPath(phases, positions, duration)
    sample_times = np.linspace(0., 1., 801)
    actual = planner.limit_samples(path, sample_times)
    actual_sites = np.asarray(calls)
    calls.clear()
    expected = planner.reference_limit_samples(path, sample_times)
    assert len(actual_sites) == len(calls) == 801*3
    np.testing.assert_array_equal(actual_sites, calls)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
    scalar = planner.derivatives(path, .3)
    assert np.asarray(scalar).shape == (3, 6)
    assert np.isfinite(scalar).all()
