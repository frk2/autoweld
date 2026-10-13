"""Check physical-time averaging independently of plot rendering."""

import numpy as np
import pytest

from scripts.plot_velocity_windows import integral_at, running_velocity


def test_time_weighted_average_handles_irregular_samples_and_partial_edges():
    t = np.array([0.0, 0.03, 0.13, 0.4, 0.45, 0.81, 1.0])
    # A linear velocity has an exact integral even on irregular sample intervals.
    y = 2 * t + 3
    at = np.array([0.0, 0.2, 0.55, 1.0])
    np.testing.assert_allclose(integral_at(t, y, at), at * at + 3 * at, atol=1e-12)
    x = 4 * t
    result = running_velocity(t, x, y, x, 0.5, ends=[0.7, 1.0])
    np.testing.assert_allclose(result["position_mm_s"], 4000)
    np.testing.assert_allclose(result["command_mm_s"], 4000)
    np.testing.assert_allclose(result["raw_mm_s"], (3 + 2 * result["center_s"]) * 1000)
    np.testing.assert_allclose(result["center_s"], [0.45, 0.75])


def test_window_does_not_invent_warmup_samples_or_bridge_bad_feedback():
    t = np.linspace(0, 2, 201)
    result = running_velocity(t, 3 * t, np.full(len(t), 3.0), 3 * t, 0.5, invalid=[(0.7, 0.8)])
    assert min(result["end_s"]) == pytest.approx(0.5)
    valid = ~((result["end_s"] - 0.5 < 0.8) & (result["end_s"] > 0.7))
    np.testing.assert_array_equal(result["valid"], valid)
    assert np.isnan(result["position_mm_s"][~valid]).all()
    with pytest.raises(ValueError):
        running_velocity(t, 3 * t, np.ones(len(t)), 3 * t, 1.0, ends=[0.5])


def test_average_reference_uses_identical_interval_through_ramp():
    t = np.linspace(0, 2, 2001)
    # Reference acceleration is 2m/s^2. Measured displacement follows at half scale.
    result = running_velocity(t, 0.5 * t * t, t, t * t, 1.0, ends=[1.5])
    assert result["command_mm_s"][0] == pytest.approx(2000)
    assert result["position_mm_s"][0] == pytest.approx(1000)
    # Comparing against instantaneous command at window END would incorrectly use 3000.
    assert result["center_s"][0] == pytest.approx(1.0)
