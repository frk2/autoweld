"""Encoder-based, per-joint settling metrics; no hardware access."""

import numpy as np

TOLERANCE_DEG = np.array([.2, .2, .2, 1., 1., 1.])
OVERSHOOT_LIMIT_DEG = np.array([.1, .1, .1, .5, .5, .5])


def summarize(rows, target, initial, session):
    rows = [r for r in rows if (r.get('motion_reference') or {}).get('session') == session]
    if len(rows) < 10:
        raise ValueError('Insufficient feedback for settling measurement')
    # The SDK reference uses monotonic elapsed time, while encoder timestamps use
    # wall time. The smallest publication-minus-elapsed is the least stale anchor.
    anchors = np.array([r['timestamp'] - r['motion_reference']['elapsed_s'] for r in rows])
    epoch = float(np.min(anchors))
    duration = rows[-1]['motion_reference']['duration_s']
    target, initial = np.asarray(target), np.asarray(initial)
    joints = []
    for j in range(6):
        stamps = np.array([r['feedback_timestamps'][j] for r in rows])
        stamps, ids = np.unique(stamps, return_index=True)
        q = np.array([r['q'][j] for r in rows])[ids]
        t = stamps - epoch
        keep = t >= 0
        t, q = t[keep], q[keep]
        if len(t) < 3:
            raise ValueError('Insufficient unique timestamped feedback')
        error = np.rad2deg(target[j] - q)
        bad = np.flatnonzero(abs(error) > TOLERANCE_DEG[j])
        entry = int(bad[-1] + 1) if len(bad) else 0
        valid = (entry < len(t) and t[-1] - t[entry] >= 1.
                 and t[0] < .1 and np.max(np.diff(t)) <= .1)
        settling = float(t[entry]) if valid else None
        direction = np.sign(target[j] - initial[j])
        overshoot = float(max(0., np.max(-direction * error)))
        joints.append(dict(joint=j+1, tolerance_deg=float(TOLERANCE_DEG[j]),
                           overshoot_limit_deg=float(OVERSHOOT_LIMIT_DEG[j]),
                           settling_s=settling,
                           post_reference_settling_s=max(0., settling-duration) if valid else None,
                           overshoot_deg=overshoot, final_error_deg=float(error[-1]),
                           max_sample_gap_s=float(np.max(np.diff(t))),
                           samples=len(t), stable_recorded_s=float(t[-1]-t[entry]) if valid else None))
    times = [j['settling_s'] for j in joints]
    ratio = max(times)/min(times) if all(t is not None and t > 0 for t in times) else None
    return dict(joints=joints, duration_s=duration, epoch_s=epoch,
                overshoot_limits_deg=OVERSHOOT_LIMIT_DEG.tolist(),
                anchor_spread_s=float(np.ptp(anchors)),
                arrival_ratio=ratio, synchronized=ratio is not None and ratio <= 1.2,
                overshoot_pass=all(j['overshoot_deg'] < j['overshoot_limit_deg'] for j in joints),
                all_settled=all(t is not None for t in times))
