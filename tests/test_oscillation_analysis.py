"""The reporting pipeline must preserve a known shake and exclude missing data."""
import numpy as np

from scripts.analyze_mixed_wrist import joint_signal, measure


def fixture(gap=False):
    rows = []
    for t in np.arange(-4., 13., .02):
        if gap and 3. < t < 3.2:
            continue
        reference = .01*np.clip(t, 0, 9)
        shake = np.deg2rad(.05)*np.sin(2*np.pi*7*t)
        rows.append(dict(timestamp=t, feedback_timestamps=[t]*6, q=[reference+shake]*6,
                         motion_reference=dict(session='test', q=[reference]*6)))
    return rows


def test_known_frequency_and_rms_recovered():
    segments, rate = joint_signal(fixture(), 3, 0., 9., 'test')
    metrics, _ = measure(segments, 0., 9., 20.)
    assert abs(rate-50) < 1e-8
    assert metrics['dominant_hz'] == 7.
    assert abs(metrics['rms_deg']-.05/np.sqrt(2)) < .0002
    assert metrics['coverage_fraction'] > .99


def test_gap_never_bridged_or_included_in_metrics():
    segments, _ = joint_signal(fixture(gap=True), 3, 0., 9., 'test')
    assert len(segments) == 2
    for segment in segments:
        t = segment['t'][segment['trusted']]
        assert not np.any((t > 2.) & (t < 4.2))
    metrics, _ = measure(segments, 0., 9., 20.)
    assert metrics['coverage_fraction'] < .8
    assert metrics['dominant_hz'] == 7.


def test_constant_encoder_does_not_invent_shaking_frequency():
    rows = fixture()
    for row in rows:
        row['q'] = [0.]*6
        row['motion_reference']['q'] = [1e-10*row['timestamp']]*6
    segments, _ = joint_signal(rows, 4, 0., 9., 'test')
    metrics, _ = measure(segments, 0., 9., 20.)
    assert metrics['encoder_position_constant']
    assert metrics['dominant_hz'] is None
