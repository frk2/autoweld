"""Pure policy for an externally supervised, already-powered SDK handoff.

This module never opens CAN, manages processes, enables or disables motors. The
caller supplies a fully validated path and continues sending the returned fixed
reference even when the policy fails. No failure authorizes resuming an old owner.
"""

import math

import numpy as np

MOTORS = ((1, 'DM4340'), (2, 'DM4340'), (3, 'DM4340'),
          (4, 'DM4310'), (5, 'DM4310'), (6, 'DM4310'))


def _field(frame, name):
    return frame[name] if isinstance(frame, dict) else getattr(frame, name)


class RecoveryController:
    """Fixed hold, bounded path progress, then fresh sustained rest verification.

    ``now`` is monotonic. SDK frame timestamps use wall time, so callers must
    supply ``feedback_now=time.time()`` to step; no clocks are read internally.
    References contain six raw motor coordinates (fixed zero offsets/directions).
    The supplied path retains its logical seventh, stationary dummy coordinate.
    """

    def __init__(self, path, *, started_at, hold_s=1., settle_s=1.,
                 speed_cap_rad_s=.1, max_speed_rad_s=.35, tracking_error_rad=.04,
                 max_gap_s=.1, max_progress_dt_s=.02, max_frame_age_s=.1,
                 rest_tolerance_rad=math.radians(.5), quiet_speed_rad_s=.05):
        values = (started_at, hold_s, settle_s, speed_cap_rad_s, max_speed_rad_s,
                  tracking_error_rad, max_gap_s, max_progress_dt_s,
                  max_frame_age_s, rest_tolerance_rad, quiet_speed_rad_s)
        if not np.isfinite(values).all() or any(v <= 0 for v in values[1:]):
            raise ValueError('Finite start time and positive recovery limits required')
        if (hold_s < 1 or settle_s < 1 or speed_cap_rad_s > max_speed_rad_s
                or max_speed_rad_s > .35 or tracking_error_rad > .04
                or max_gap_s > .1 or max_progress_dt_s > max_gap_s
                or max_frame_age_s > .1 or rest_tolerance_rad > math.radians(.5)
                or quiet_speed_rad_s > .05):
            raise ValueError('Recovery limits must preserve conservative guard ceilings')
        if not np.isfinite(path.duration) or path.duration <= 0:
            raise ValueError('Finite positive prevalidated path duration required')
        self.path = path
        self.start = self._sample(0.)[0]
        self.rest = self._sample(path.duration)[0]
        if not np.isclose(self.start[6], self.rest[6], atol=1e-12, rtol=0):
            raise ValueError('Logical dummy joint must remain fixed')
        self.hold_s, self.settle_s = float(hold_s), float(settle_s)
        self.speed_cap = float(speed_cap_rad_s)
        self.max_speed = float(max_speed_rad_s)
        self.tracking_error = float(tracking_error_rad)
        self.max_gap, self.max_dt = float(max_gap_s), float(max_progress_dt_s)
        self.max_frame_age = float(max_frame_age_s)
        self.rest_tolerance, self.quiet_speed = float(rest_tolerance_rad), float(quiet_speed_rad_s)
        self.reference = self.start[:6].copy()
        self.stage = 'hold'
        self.error = None
        self.cooldown_requested = False
        self.path_elapsed = 0.
        self.quiet_elapsed = 0.
        self.last_now = float(started_at)
        self.last_stamps = None
        self.movement_started = False

    def _sample(self, elapsed):
        q, v = (np.asarray(x, float) for x in self.path.sample(elapsed)[:2])
        if q.shape != (7,) or v.shape != (7,) or not np.isfinite([q, v]).all():
            raise ValueError('Prevalidated path must return finite logical seven-axis position/velocity')
        return q, v

    def snapshot(self):
        return {'stage': self.stage, 'reference_rad': self.reference.tolist(),
                'error': self.error, 'path_elapsed_s': self.path_elapsed,
                'quiet_elapsed_s': self.quiet_elapsed,
                'rest_verified': self.stage == 'verified_rest',
                'cooldown_requested': self.cooldown_requested,
                'motor_ids': [mid for mid, _ in MOTORS],
                'speed_cap_rad_s': self.speed_cap,
                'response': 'Fixed powered hold; operator intervention required' if self.error else self.stage,
                'may_disable': self.stage == 'verified_rest', 'may_resume_old_owner': False}

    def fail(self, reason):
        """Latch the last safe command without any power or process operation."""
        if self.error is None:
            self.error = str(reason)
        self.stage = 'failed_hold'
        self.quiet_elapsed = 0.
        return self.snapshot()

    def abort(self, reason):
        return self.fail(reason)

    def step(self, now, frames, *, feedback_now=None):
        if self.error:
            return self.snapshot()
        try:
            now = float(now)
            feedback_now = now if feedback_now is None else float(feedback_now)
            if not np.isfinite([now, feedback_now]).all() or now < self.last_now:
                raise ValueError('Invalid or regressing recovery clock')
            gap = now-self.last_now
            if gap > self.max_gap and self.movement_started:
                raise ValueError('Recovery loop timing overrun')
            frames = list(frames)
            if len(frames) != 6 or tuple(_field(f, 'id') for f in frames) != tuple(range(1, 7)):
                raise ValueError('Exactly ordered motor IDs1-6 required')
            q, v, stamps, mos, rotor = (np.array([_field(f, key) for f in frames], float)
                for key in ('position', 'velocity', 'timestamp', 'temperature_mos', 'temperature_rotor'))
            if not np.isfinite([q, v, stamps, mos, rotor]).all():
                raise ValueError('Nonfinite SDK feedback')
            if any(str(_field(f, 'error_code')).lower() != '0x1' for f in frames):
                raise ValueError('Motor is not healthy and powered')
            age = feedback_now-stamps
            if np.any(stamps <= 0) or np.any(age < 0) or np.any(age > self.max_frame_age):
                raise ValueError('Stale or future SDK feedback')
            if np.max(np.r_[mos, rotor]) >= 60:
                # The only permitted trajectory already returns to supported
                # rest. Request cooldown there without stranding a hot arm in
                # a working pose; every independent motion guard still applies.
                self.cooldown_requested = True
            if np.max(np.abs(v)) > self.max_speed:
                raise ValueError('Measured joint speed exceeds recovery guard')
            if np.max(np.abs(q-self.reference)) > self.tracking_error:
                raise ValueError('Measured tracking error exceeds recovery guard')
            if self.last_stamps is not None and np.any(stamps < self.last_stamps):
                raise ValueError('Regressing SDK feedback timestamp')
            fresh = self.last_stamps is not None and np.all(stamps > self.last_stamps)
            self.last_stamps, self.last_now = stamps.copy(), now
            dt = min(gap, self.max_dt) if fresh else 0.
            if gap > self.max_gap:  # Initial hold may wait, but never counts an unobserved gap.
                self.quiet_elapsed = 0.
                dt = 0.
            quiet = np.max(np.abs(v)) <= self.quiet_speed
            if self.stage == 'hold':
                self.quiet_elapsed = self.quiet_elapsed+dt if quiet else 0.
                if self.quiet_elapsed >= self.hold_s-1e-12:
                    self.stage = 'path'
                    self.quiet_elapsed = 0.
                    self.movement_started = True
            elif self.stage == 'path':
                elapsed = min(self.path.duration, self.path_elapsed+dt)
                target, velocity = self._sample(elapsed)
                if (not np.isclose(target[6], self.start[6], atol=1e-12, rtol=0)
                        or np.max(np.abs(velocity[:6])) > self.speed_cap+1e-10
                        or np.max(np.abs(target[:6]-self.reference)) > self.speed_cap*dt+1e-10):
                    raise ValueError('Path reference exceeds speed or fixed dummy-joint guard')
                if np.max(np.abs(target[:6]-q)) > self.tracking_error:
                    raise ValueError('Proposed reference exceeds tracking error guard')
                self.reference = target[:6].copy()
                self.path_elapsed = elapsed
                if elapsed >= self.path.duration:
                    self.stage = 'settle'
                    self.quiet_elapsed = 0.
            else:  # Verified rest remains subject to the same live guards.
                at_rest = np.max(np.abs(q-self.rest[:6])) <= self.rest_tolerance
                if at_rest and quiet:
                    self.quiet_elapsed += dt
                    if self.quiet_elapsed >= self.settle_s-1e-12:
                        self.stage = 'verified_rest'
                else:
                    self.stage = 'settle'
                    self.quiet_elapsed = 0.
            return self.snapshot()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
            return self.fail(error)

    def command(self, interface):
        """Use only the supplied SDK interface; no startup or shutdown methods.

        Transport failures latch the policy but propagate to the caller, which
        owns continued hold transmission and supervision of individual axes.
        """
        try:
            return [interface.set_control(mid, kind, float(self.reference[i]), self.speed_cap, 0., 0., 0.)
                    for i, (mid, kind) in enumerate(MOTORS)]
        except Exception as error:
            self.fail(f'SDK command exchange failed: {error}')
            raise
