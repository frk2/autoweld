"""Fresh disabled-arm inspection through a service-owned SDK read-only reader.

No model, writer, motor power transition, register write, zeroing or recovery is
performed. The command branch is packed SDK feedback verified against native
p_m, matching installed startup preflight. The separate xout81 evidence remains
available but never silently redefines the qualified motion coordinates.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np


class DisabledRestError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.diagnostic = {'operation': 'disabled_rest', 'error': message, 'evidence': evidence}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def validate_sample(sample, bounds, *, now):
    mapping = sample.get('mapping') or {}
    if (sample.get('motor_ids') != list(range(1, 7)) or sample.get('disabled') is not True
            or sample.get('status') != ['0x0']*6):
        raise DisabledRestError('Fresh clean-disabled feedback from motor IDs 1–6 is required', json_value(sample))
    if (mapping.get('offsets_rad') != [0.]*6 or mapping.get('directions') != [1]*6
            or mapping.get('boot_normalization') != 'none'):
        raise DisabledRestError('Installed fixed zero-offset encoder branch is required; no normalization is allowed')
    raw = np.asarray(sample.get('sdk_feedback_position_rad'), float)
    velocity = np.asarray(sample.get('joint_velocity_rad_s'), float)
    stamps = np.asarray(sample.get('status_timestamps'), float)
    if (any(value.shape != (6,) or not np.isfinite(value).all() for value in (raw, velocity, stamps))
            or np.any(now-stamps < 0) or np.any(now-stamps > .5)):
        raise DisabledRestError('Six fresh finite SDK position, velocity and status timestamps are required')
    physical = sample.get('physical_timestamp')
    if type(physical) not in (int, float) or not np.isfinite(physical) or not 0 <= now-physical <= .5:
        raise DisabledRestError('Disabled inspection refuses stale or future physical SDK feedback')
    native = sample.get('native_p_m')
    if not isinstance(native, list) or len(native) != 6:
        raise DisabledRestError('Six native p_m register replies are required')
    for motor, item in enumerate(native, 1):
        if (item.get('motor_id') != motor or item.get('register') != 'p_m'
                or not np.isfinite([item.get('value', np.nan), item.get('timestamp', np.nan)]).all()
                or not 0 <= now-item['timestamp'] <= .5):
            raise DisabledRestError('Native p_m replies must be fresh, finite and match motor IDs 1–6')
    if np.max(abs(raw-np.asarray([item['value'] for item in native]))) > .02:
        raise DisabledRestError('Packed SDK feedback and native p_m disagree by more than 0.02 rad')
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(raw < bounds[:, 0]) or np.any(raw > bounds[:, 1])):
        raise DisabledRestError('Measured disabled pose is outside strict installed joint bounds; rest cannot expand them')
    if np.max(abs(velocity)) > .05:
        raise DisabledRestError('Disabled arm is moving faster than the 0.05 rad/s quiet limit')
    return raw.copy()


def inspect_disabled_pose(reader_factory, bounds, *, clock=time.monotonic, wall_clock=time.time, sleep=time.sleep):
    """Return a full one-second quiet survey; always release the SDK socket."""
    reader = reader_factory()
    samples = []
    started = clock()
    try:
        first_q = first_at = last_stamp = last_native = session = sequence = None
        while True:
            sample = reader.read(include_motor_position=True)
            now = wall_clock()
            try:
                q = validate_sample(sample, bounds, now=now)
                observed = clock()
                if first_q is None:
                    first_q, first_at, session = q.copy(), observed, sample.get('session')
                    if not isinstance(session, str) or not session:
                        raise DisabledRestError('An owned read-only SDK acquisition session is required')
                if sample.get('session') != session or type(sample.get('sequence')) is not int:
                    raise DisabledRestError('Disabled SDK reader session or frame sequence changed')
                native_stamps = np.array([item['timestamp'] for item in sample['native_p_m']])
                if (last_stamp is not None and (sample['physical_timestamp'] <= last_stamp
                        or sample['sequence'] <= sequence or np.any(native_stamps <= last_native))):
                    raise DisabledRestError('Repeated/cached SDK frames cannot establish one second of quiet feedback')
                if np.max(abs(q-first_q)) > np.deg2rad(.5):
                    raise DisabledRestError('Disabled arm changed by more than 0.5 degrees during quiet verification')
                last_stamp, last_native, sequence = sample['physical_timestamp'], native_stamps, sample['sequence']
                samples.append(dict(observed_monotonic=observed, observed_wall_time=now, sample=json_value(sample)))
                if observed-first_at >= 1. and len(samples) >= 3:
                    return dict(schema='autoweld.disabled-rest-inspection.v1', joint_rad=q.tolist(),
                        motor_ids=list(range(1, 7)), motor_power_changed=False, enabled=False,
                        commands_sent=False, read_only=True, quiet_seconds=observed-first_at,
                        physical_timestamp=sample['physical_timestamp'], samples=samples,
                        coordinate_branch='fixed packed SDK feedback verified against native p_m; zero offsets; no normalization',
                        xout_role='retained native encoder evidence only; command mapping not substituted')
                if observed-started > 3.:
                    raise DisabledRestError('Disabled quiet survey exceeded its three-second deadline')
            except DisabledRestError as error:
                error.diagnostic['evidence'] = {'accepted_samples': samples, 'last_sample': json_value(sample)}
                raise
            sleep(.1)
    finally:
        reader.close()


def publish_rest_records(rest_path, poses_path, rest_record, poses_record):
    """Stage both JSON records, publish together, restore prior bytes on failure."""
    values = [(Path(rest_path), rest_record), (Path(poses_path), poses_record)]
    old = {path: path.read_bytes() if path.exists() else None for path, _ in values}
    staged = []
    published = []
    try:
        for path, value in values:
            temporary = path.with_suffix(path.suffix+'.rest-capture.tmp')
            temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False))
            staged.append((path, temporary))
        for path, temporary in reversed(staged):
            temporary.replace(path)
            published.append(path)
    except Exception:
        for path in reversed(published):
            previous = old[path]
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                rollback = path.with_suffix(path.suffix+'.rest-rollback.tmp')
                rollback.write_bytes(previous)
                rollback.replace(path)
        raise
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
    return copy.deepcopy(rest_record)


def preserve_rest_provenance(previous, current):
    """Retain historical capture evidence only for exactly the same rest data.

    This never creates a live feedback assertion. Disabled startup still uses a
    cached display until a new explicit read-only inspection succeeds.
    """
    try:
        matching = (np.array_equal(np.asarray(previous['arm_rad'], float), np.asarray(current['arm_rad'], float))
            and previous['gripper_normalized'] == current['gripper_normalized']
            and previous['gripper_limits_motor_rad'] == current['gripper_limits_motor_rad'])
    except (KeyError, TypeError, ValueError):
        matching = False
    return {**copy.deepcopy(previous), **current} if matching else current
