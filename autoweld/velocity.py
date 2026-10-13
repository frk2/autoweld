"""Timestamped encoder velocity diagnostics, independent of raw velocity codes."""

from collections import deque

import numpy as np


class PositionVelocity:
    """Local quadratic derivative over 120 ms, evaluated about 60 ms in the past.

    Fits each joint at its actual CAN receive timestamps, rejecting duplicates,
    stale data and discontinuities. Raw feedback remains the safety/qualification
    input. A position-quantization-only worst-case bound accompanies the estimate;
    it does not cover dynamics, calibration or external TCP errors.
    """

    def __init__(self, window=0.12, max_gap=0.05, max_skew=0.01):
        self.window = window
        self.max_gap = max_gap
        self.max_skew = max_skew
        self.rows = deque(maxlen=1024)

    def update(self, timestamps, q, now):
        ts, q = np.asarray(timestamps, float)[:6], np.asarray(q, float)[:6]
        if ts.shape != (6,) or q.shape != (6,) or not np.isfinite(np.r_[ts, q, now]).all():
            self.rows.clear()
            return None
        if now - min(ts) > self.max_gap or max(ts) > now + 0.005 or np.ptp(ts) > self.max_skew:
            self.rows.clear()
            return None
        if self.rows:
            delta = ts - self.rows[-1][0]
            if np.any(delta < 0) or np.max(delta) > self.max_gap:
                self.rows.clear()
            elif np.any(delta == 0):
                return None  # Never turn repeated cached positions into zero-speed samples.
        self.rows.append((ts.copy(), q.copy()))
        while len(self.rows) > 2 and min(ts - self.rows[0][0]) > self.window:
            self.rows.popleft()
        times = np.array([r[0] for r in self.rows])
        if len(times) < 5 or np.min(times[-1] - times[0]) < self.window * 0.7:
            return None
        # One common evaluation time for all six joints, within every fit window.
        # Subtract the epoch first to avoid ill-conditioned polynomial fits.
        epoch = float(np.min(times[-1]))
        relative = times - epoch
        center = float((np.max(relative[0]) + np.min(relative[-1])) / 2)
        u = (relative-center) / self.window
        design = np.stack((np.ones_like(u), u, u*u), axis=-1).transpose(1, 0, 2)
        inverse = np.linalg.pinv(design)
        weights = inverse[:, 1, :].T / self.window
        positions = np.array([r[1] for r in self.rows])
        velocity = (weights * (positions - positions[-1])).sum(axis=0)
        fitted_position = (inverse[:, 0, :].T * (positions-positions[-1])).sum(axis=0) + positions[-1]
        # All installed arm motors report position over +/-12.5 rad, 16 bits.
        bound = np.abs(weights).sum(axis=0) * (25 / 65535) / 2
        return {
            "joint_velocity_rad_s": velocity.tolist(),
            "joint_position_rad": fitted_position.tolist(),
            "joint_quantization_bound_rad_s": bound.tolist(),
            "window_s": float(np.min(times[-1] - times[0])),
            "timestamp": epoch + center,
            "delay_s": float(now - (epoch+center)),
            "method": "local quadratic at common window center",
            "samples": len(times),
        }
