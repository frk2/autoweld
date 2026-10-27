"""Installed joint travel and fixed encoder mapping, independent of the UI."""
import json
from pathlib import Path

import numpy as np

CALIBRATION = Path(__file__).resolve().parents[1] / 'assets/camera_mount/joint-calibration.json'


def load_joint_calibration(path=CALIBRATION):
    record = json.loads(Path(path).read_text())
    endpoints = np.asarray(record['j6_physical_endpoints_rad'], float)
    offsets = np.asarray(record['fixed_motor_offsets_rad'], float)
    clearance = float(record['clearance_rad'])
    if (record.get('schema') != 'autoweld.installed-joints.v1'
            or endpoints.shape != (2,) or offsets.shape != (6,)
            or not np.isfinite(np.r_[endpoints, offsets, clearance]).all()
            or not 0 < endpoints[1]-endpoints[0] < 2*np.pi
            or not np.deg2rad(.5) <= clearance < (endpoints[1]-endpoints[0])/2):
        raise ValueError('Invalid installed joint calibration')
    return record


def j6_operating_range(record=None):
    record = load_joint_calibration() if record is None else record
    return np.asarray(record['j6_physical_endpoints_rad']) + np.array([1, -1])*record['clearance_rad']


def verify_disabled_start(channel, limits, offsets):
    """SDK read-only preflight: never energize a manually moved endpoint pose."""
    from i2rt.robots.encoder_teaching import ReadOnlyArmEncoder

    reader = ReadOnlyArmEncoder(channel, offsets_rad=offsets)
    try:
        sample = reader.read(include_motor_position=True)
    finally:
        reader.close()
    if not sample['disabled']:
        raise RuntimeError('Startup requires all six motors disabled; no enable was issued')
    raw = np.asarray(sample['sdk_feedback_position_rad'], float)
    native = np.array([item['value'] for item in sample['native_p_m']])
    if not np.isfinite(native).all() or np.max(abs(raw-native)) > .02:
        raise RuntimeError('SDK status and native encoder coordinates disagree; no enable was issued')
    q = (raw-np.asarray(offsets))*np.asarray(sample['mapping']['directions'])
    bounds = np.asarray(limits)
    # Preserve existing tolerance for other axes' captured supported rest. J6
    # must be strictly within its measured range with endpoint clearance.
    tolerance = np.full(6, .01)
    tolerance[5] = 0.
    outside = (q < bounds[:, 0]-tolerance) | (q > bounds[:, 1]+tolerance)
    if outside.any():
        joint = int(np.flatnonzero(outside)[0])+1
        raise RuntimeError(f'J{joint} is outside calibrated startup bounds; keep power disabled and move inward')
    return q
