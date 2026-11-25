#!/usr/bin/env python3
"""Audit an archived weld path using files and NumPy only.

The recorded detector result is the measurement input, never recomputed or
replaced. This checks identity, timing, stationary-capture evidence and every
camera-to-world transform independently. It does not certify calibration or the
physical encoder sample time. Older archives without raw-file hashes or full
station feedback retain explicit evidence limitations in the report.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

IMMUTABLE_IDENTITY = ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id')
IDENTITY = (*IMMUTABLE_IDENTITY, 'generation')
MAX_FILE_BYTES = 16*1024*1024
MAX_OBSERVATIONS = 300


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f'{label}: finite number required')
    return float(value)


def array(value, shape, label):
    result = np.asarray(value, float)
    require(result.shape == shape and np.isfinite(result).all(), f'{label}: invalid/nonfinite shape')
    return result


def pose(value, label):
    result = array(value, (4, 4), label)
    rotation = result[:3, :3]
    require(np.allclose(result[3], [0., 0., 0., 1.], rtol=0, atol=1e-9)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
        and abs(np.linalg.det(rotation)-1.) <= 1e-6, f'{label}: proper rigid transform required')
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def load_json(path):
    require(path.is_file() and path.stat().st_size <= MAX_FILE_BYTES, f'{path}: missing/oversized JSON file')
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f'{path}: object required')
    return value


def resolve_native(value, directory):
    stored = Path(value)
    # A copied archive may retain its original absolute paths. Only the filename
    # is used for that relocation fallback; the file's schema/identity is checked.
    candidates = [directory/stored.name, stored] if stored.is_absolute() else [directory/stored, directory/stored.name, stored]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise AuditError(f'Missing native observation: {value}')


def pose_error(actual, expected):
    translation = float(np.linalg.norm(actual[:3, 3]-expected[:3, 3]))
    cosine = np.clip((np.trace(actual[:3, :3] @ expected[:3, :3].T)-1.)/2., -1., 1.)
    return translation, float(np.arccos(cosine))


def validate_endpoint_retry_bridge(retry, directory, prior_brackets):
    """Prove one identical-goal endpoint replan without renewing optical budget."""
    require(isinstance(retry, dict) and retry.get('status') == 'complete'
        and type(retry.get('attempt')) is int and type(retry.get('maximum_replans')) is int
        and retry.get('attempt') == retry.get('maximum_replans') == 1,
        'Exposure Hold bridge needs one completed bounded endpoint retry')
    require(directory is not None and isinstance(retry.get('diagnostic_path'), str),
        'Endpoint retry bridge lacks its persisted diagnostic file')
    diagnostic = resolve_native(retry['diagnostic_path'], directory)
    require(load_json(diagnostic) == retry, 'Endpoint retry diagnostic differs from manifest evidence')
    old = number(retry.get('prior_hold_monotonic'), 'Prior retry Hold')
    new = number(retry.get('replacement_hold_monotonic'), 'Replacement retry Hold')
    started = number(retry.get('started_monotonic'), 'Endpoint retry start')
    completed = number(retry.get('completed_monotonic'), 'Endpoint retry completion')
    deadline = number(retry.get('station_deadline'), 'Original station deadline')
    require(old < started <= new <= completed <= deadline <= old+30.,
        'Endpoint retry renewed or exceeded the original station deadline')
    count = retry.get('optical_recovery_attempts')
    require(type(count) is int and 1 <= count < 5 and len(prior_brackets) == count,
        'Endpoint retry does not preserve its preceding optical attempt count')
    require(prior_brackets and max(item['completed_monotonic'] for item in prior_brackets) <= started,
        'Endpoint retry overlaps a preceding exposure configuration')
    requested = pose(retry.get('requested_base_from_tcp'), 'Unchanged retry endpoint')

    def evidence(value, hold, attempt, qualified):
        require(isinstance(value, dict), 'Endpoint retry lacks measured before/after feedback')
        observed = number(value.get('observed_at'), 'Endpoint retry observation time')
        feedback = value.get('feedback') or {}
        stamp = number(feedback.get('measured_at'), 'Endpoint retry SDK feedback time')
        require(feedback.get('fresh') is True and feedback.get('stationary') is True
            and 0 <= observed-stamp <= .1 and stamp >= hold,
            'Endpoint retry before/after SDK feedback is not fresh and stationary')
        require(value.get('hold_monotonic') == hold and type(value.get('retry_attempt')) is int
            and value['retry_attempt'] == attempt
            and value.get('station_deadline') == deadline,
            'Endpoint retry before/after Hold, attempt or original deadline differs')
        goal = pose(value.get('requested_base_from_tcp'), 'Retry feedback requested endpoint')
        actual = pose(value.get('actual_base_from_tcp'), 'Retry measured endpoint')
        require(np.allclose(goal, requested, rtol=0, atol=1e-12),
            'Endpoint retry changed its requested endpoint')
        position, orientation = pose_error(actual, requested)
        require(abs(number(value.get('position_error_m'), 'Retry saved position error')-position) <= 1e-9
            and abs(number(value.get('orientation_error_rad'), 'Retry saved orientation error')-orientation) <= 1e-6,
            'Endpoint retry convergence errors differ from measured poses')
        if qualified:
            require(value.get('gate') == 'qualified' and position <= .0005+1e-12 and orientation <= .005+1e-10
                and number(value.get('quiet_seconds'), 'Retry quiet feedback duration') >= .4
                and observed-hold >= .4-1e-9,
                'Replacement retry Hold lacks quiet measured endpoint convergence')
        else:
            require((value.get('gate') == 'endpoint_position' and position > .0005)
                or (value.get('gate') == 'endpoint_orientation' and position <= .0005 and orientation > .005),
                'Endpoint retry was not preceded by a measured endpoint miss')
        return observed, actual

    before = retry.get('before_evidence')
    before_time, before_pose = evidence(before, old, 0, False)
    after_time, _ = evidence(retry.get('after_evidence'), new, 1, True)
    require(before_time <= started and abs(after_time-completed) <= 1e-9,
        'Endpoint retry before/after observations do not bound its replan')
    duration = number(before.get('endpoint_miss_seconds'), 'Sustained endpoint miss duration')
    samples = before.get('endpoint_miss_samples')
    require(duration >= .2 and type(samples) is int and samples >= 3,
        'Endpoint retry lacks a sustained measured Hold miss')
    history = retry.get('preceding_samples')
    require(isinstance(history, list) and len(history) <= 128
        and all(isinstance(value, dict) for value in history),
        'Endpoint retry lacks bounded preceding SDK evidence')
    sustained = [value for value in history
        if number(value.get('observed_at'), 'Retry preceding sample time') >= before_time-duration-.001]
    sustained.append(before)
    times = []
    for value in sustained:
        observed, actual = evidence(value, old, 0, False)
        position, orientation = pose_error(actual, before_pose)
        require(position <= .0003+1e-12 and orientation <= .003+1e-10,
            'Endpoint retry miss was not stationary at one measured Hold')
        times.append(observed)
    require(len(times) >= 3 and times[-1]-times[0] >= .2-.001
        and all(0 < second-first <= .1 for first, second in zip(times, times[1:])),
        'Endpoint retry lacks continuous fresh stationary miss evidence')
    return dict(prior_hold_monotonic=old, replacement_hold_monotonic=new,
        optical_recovery_attempts=count, completed_monotonic=completed,
        station_deadline=deadline, diagnostic_path=str(diagnostic))


def validate_generation_policy(saved, directory=None):
    binding = saved['binding']
    transitions = saved.get('generation_transitions', [])
    if 'generation_policy' not in binding:
        require(not transitions and not saved.get('exposure_brackets'),
            'Generation transitions or exposure brackets lack an explicit owned exposure policy')
        return None
    require(binding['generation_policy'] == 'owned_stripe_AE_per_station_requalification',
        'Unknown native generation policy')
    initial, final = binding.get('initial_generation'), binding.get('final_generation')
    require(type(initial) is int and initial >= 0 and type(final) is int and final >= initial
        and binding.get('generation') == final, 'Invalid initial/final native generation binding')
    require(isinstance(transitions, list) and len(transitions) <= 900, 'Invalid/unbounded generation transitions')
    brackets = {}
    attempts_by_hold = {}
    roi_by_hold = {}
    retries = saved.get('endpoint_retries', [])
    require(isinstance(retries, list) and len(retries) <= MAX_OBSERVATIONS
        and all(isinstance(retry, dict) for retry in retries), 'Invalid/unbounded endpoint retry evidence')
    retry_bridges, retired_holds = {}, set()
    previous_completion = -math.inf
    for bracket in saved.get('exposure_brackets', []):
        request_id = bracket.get('request_id')
        require(isinstance(request_id, str) and request_id and request_id not in brackets,
            'Owned exposure bracket request IDs must be unique')
        limits = bracket.get('limits') or {}
        minimum = number(limits.get('min_us'), 'Bracket sensor minimum')
        maximum = number(limits.get('max_us'), 'Bracket sensor maximum')
        cap = number(limits.get('fps_cap_us'), 'Bracket maximum-FPS exposure cap')
        requested = number(bracket.get('requested_us'), 'Bracket requested exposure')
        actual = number(bracket.get('actual_us'), 'Bracket actual exposure')
        readback = bracket.get('configuration_readback', bracket.get('readback')) or {}
        require(0 < minimum <= maximum <= cap and minimum <= requested <= maximum
            and minimum <= actual <= maximum
            and number(readback.get('exposure_us'), 'Bracket sensor readback') == actual
            and readback == bracket.get('readback') and readback.get('auto_exposure') is False,
            'Owned exposure bracket exceeds sensor/FPS bounds or readback differs')
        kind = bracket.get('kind', 'exposure')
        require(kind in ('exposure', 'gain', 'ae_lock', 'roi_transport'), 'Unknown owned exposure configuration kind')
        candidates = [max(minimum, maximum*factor) for factor in (1., .5, .25)]
        if kind == 'exposure':
            require(min(abs(requested-value) for value in candidates) <= 2., 'Owned bracket is outside maximum/half/quarter candidates')
        elif kind == 'gain':
            if (saved.get('settings') or {}).get('reflective_surface') is True:
                require(min(abs(requested-value) for value in candidates) <= 2.,
                    'Reflective gain recovery is outside maximum/half/quarter exposure candidates')
            else:
                require(requested == maximum, 'Gain recovery must use the bounded maximum-FPS exposure')
        if bracket.get('gain_bounds_db') is not None:
            gain_bounds = array(bracket['gain_bounds_db'], (2,), 'Sensor gain bounds')
            gain = number(readback.get('gain_db'), 'Actual sensor gain')
            requested_gain = number(bracket.get('requested_gain_db'), 'Requested sensor gain')
            require(gain_bounds[0] <= requested_gain <= gain_bounds[1]
                and gain_bounds[0] <= gain <= gain_bounds[1], 'Owned configuration exceeds actual sensor gain bounds')
            if kind == 'gain':
                require(bracket.get('gain_ceiling_db') == 24. and max(gain, requested_gain) <= 24.,
                    'Gain recovery exceeds the 24 dB ceiling')
        hold = number(bracket.get('measured_hold_monotonic'), 'Bracket measured Hold time')
        started = number(bracket.get('started_monotonic'), 'Bracket configuration start')
        completed = number(bracket.get('completed_monotonic'), 'Bracket configuration completion')
        require(hold <= started <= completed <= hold+30., 'Owned exposure bracket lacks bounded completion at measured Hold')
        require(started >= previous_completion, 'Owned exposure configuration order regressed or overlapped')
        previous_completion = completed
        attempt = bracket.get('attempt')
        if kind == 'roi_transport':
            require((saved.get('settings') or {}).get('reflective_surface') is True,
                'ROI transport requires explicit reflective-surface settings')
            require(type(attempt) is int and attempt == 0 and hold not in roi_by_hold
                and hold not in attempts_by_hold and hold not in retired_holds,
                'ROI transport must occur once before optical attempts at the same measured Hold')
            validate_roi_transport(bracket, hold)
            require(brackets and list(brackets.values())[-1]['readback'] == readback,
                'ROI-only transport changed exposure/gain or lacks preceding owned control readback')
            roi_by_hold[hold] = bracket
        elif kind == 'ae_lock':
            require(attempt == 0 and not any(item.get('kind') == 'ae_lock' for item in brackets.values()),
                'Initial native AE freeze must occur once and not consume recovery attempts')
            ae = bracket.get('auto_exposure') or {}
            require(ae.get('reliable') is True and ae.get('state') in ('stable', 'limits'),
                'Initial exposure freeze lacks reliable stable/limit native AE feedback')
            require(ae.get('source_id') == binding['source_id']
                and 0 <= started-number(ae.get('read_started'), 'Native AE feedback time') <= 1.,
                'Initial exposure freeze lacks fresh native AE feedback from the bound camera')
        else:
            require(type(attempt) is int and 1 <= attempt <= 5, 'Owned bracket exceeds five attempts at one measured Hold')
            require(hold not in retired_holds, 'Exposure configuration reused a replaced measured Hold')
            if hold not in attempts_by_hold:
                bridges = [retry for retry in retries if retry.get('replacement_hold_monotonic') == hold
                    and (retry.get('optical_recovery_attempts') != 0
                         or retry.get('prior_hold_monotonic') in attempts_by_hold)]
                require(len(bridges) <= 1, 'Measured Hold has competing endpoint retry bridges')
                if bridges:
                    retry = bridges[0]
                    old = retry.get('prior_hold_monotonic')
                    require(old in attempts_by_hold and old not in retry_bridges and old not in retired_holds,
                        'Endpoint retry bridge lacks one unreplaced preceding Hold budget')
                    prior = [item for item in brackets.values() if item['measured_hold_monotonic'] == old
                        and item.get('kind', 'exposure') in ('exposure', 'gain')]
                    proof = validate_endpoint_retry_bridge(retry, directory, prior)
                    require(all(item.get('direction') == bracket.get('direction')
                        and item.get('distance_mm') == bracket.get('distance_mm') for item in prior),
                        'Endpoint retry bridge changed the optical scan station')
                    require(len(attempts_by_hold[old]) == proof['optical_recovery_attempts'],
                        'Endpoint retry bridge reset or extended the optical attempt budget')
                    attempts_by_hold[hold] = attempts_by_hold[old]
                    retired_holds.add(old)
                    retry_bridges[hold] = proof
                else:
                    attempts_by_hold[hold] = []
            if hold in retry_bridges:
                proof = retry_bridges[hold]
                require(started >= proof['completed_monotonic'] and completed <= proof['station_deadline'],
                    'Bridged exposure configuration precedes settling or renewed the original deadline')
            prior_attempts = attempts_by_hold[hold]
            require(attempt == len(prior_attempts)+1, 'Owned bracket attempt sequence is not contiguous at measured Hold')
            prior_attempts.append(attempt)
        require(type(bracket.get('generation')) is int and bracket['generation'] >= initial,
            'Owned bracket lacks an actual camera generation')
        brackets[request_id] = bracket
    previous, observed = initial, -math.inf
    linked_brackets = set()
    for transition in transitions:
        current = transition.get('generation')
        require(transition.get('previous_generation') == previous and type(current) is int and current > previous,
            'Native generation transition chain is discontinuous or regressed')
        reason = transition.get('reason')
        require(reason in ('owned_native_stripe_auto_exposure', 'owned_exposure_bracket'),
            'Native generation transition is not an owned exposure update')
        stamp = number(transition.get('observed_at'), 'Generation transition time')
        require(stamp >= observed, 'Native generation transition timestamps regressed')
        controls = transition.get('controls')
        require(isinstance(controls, dict) and number(controls.get('exposure_us'), 'Transition exposure') > 0,
            'Native exposure transition lacks actual exposure controls')
        if reason == 'owned_exposure_bracket':
            request_id = transition.get('request_id')
            require(request_id in brackets and request_id not in linked_brackets,
                'Manual exposure generation transition lacks unique completed configuration proof')
            bracket = brackets[request_id]
            require(bracket['started_monotonic'] <= stamp <= bracket['completed_monotonic']
                and bracket['generation'] == current and transition.get('bracket_attempt') == bracket['attempt']
                and transition.get('requested_exposure_us') == bracket['requested_us']
                and controls == bracket.get('configuration_readback', bracket['readback'])
                and controls.get('auto_exposure') is False,
                'Manual exposure generation transition disagrees with its completed configuration')
            if 'bracket_kind' in transition:
                require(transition['bracket_kind'] == bracket.get('kind', 'exposure'),
                    'Generation transition changed the owned configuration kind')
            if 'requested_gain_db' in transition:
                require(transition['requested_gain_db'] == bracket.get('requested_gain_db'),
                    'Generation transition changed the requested sensor gain')
            if bracket.get('kind') == 'ae_lock':
                require(bracket['auto_exposure'].get('generation') == previous,
                    'Initial exposure freeze AE feedback belongs to another generation')
            linked_brackets.add(request_id)
        previous, observed = current, stamp
    require(previous == final, 'Final generation differs from the complete transition chain')
    require(linked_brackets == set(brackets), 'Completed manual exposure bracket lacks a bound generation transition')
    return dict(initial=initial, final=final, transitions=transitions, brackets=brackets,
                endpoint_retry_bridges=list(retry_bridges.values()), roi_transports=roi_by_hold)


def validate_roi_transport(bracket, hold):
    """A projected search hint is never optical recovery or a seam measurement."""
    transport = bracket.get('roi_transport') or {}
    roi = array(transport.get('roi'), (4,), 'Transported normalized ROI')
    require(np.all((roi >= 0.) & (roi <= 1.)) and np.all(roi[2:] > roi[:2]),
        'Transported ROI must be a finite normalized rectangle')
    require(transport.get('projection_only') is True and transport.get('basis') in (
        'fresh_measured_initial_root', 'previous_qualified_measured_root',
        'predicted_coped_view_target_not_a_measurement'),
        'ROI transport must identify an allowed projection-only root/view-target basis')
    require(number(transport.get('hold_monotonic'), 'ROI request measured Hold') == hold,
        'ROI transport request differs from its measured Hold')
    camera = pose(transport.get('measured_base_from_camera'), 'ROI projection measured camera')
    point = array(transport.get('point_base_m'), (3,), 'ROI projected world point')
    native = array(transport.get('point_camera_mm'), (3,), 'ROI projected camera point')
    require(native[2] > 0 and np.allclose(camera[:3, :3].T @ (point-camera[:3, 3])*1000.,
        native, rtol=0, atol=1e-7), 'ROI world/camera point projection algebra differs')
    anchor = transport.get('anchor') or {}
    extent = array(anchor.get('extent'), (2,), 'ROI anchor normalized extent')
    offset = array(anchor.get('offset'), (2,), 'ROI anchor normalized offset')
    require(np.all((extent > 0.) & (extent <= 1.))
        and np.allclose(roi[2:]-roi[:2], extent, atol=1e-9, rtol=0)
        and np.all(abs(offset) <= extent/2+1e-9), 'ROI transport changed or invalidated its measured anchor extent/offset')
    require(isinstance(anchor.get('native_path'), str) and bool(anchor['native_path'])
        and isinstance(anchor.get('acquisition_key'), list) and len(anchor['acquisition_key']) == 3,
        'ROI transport anchor lacks an original native acquisition')
    require(array(anchor.get('seam_point_camera_mm'), (3,), 'ROI anchor measured root')[2] > 0,
        'ROI anchor root is behind the camera')
    pose(anchor.get('measured_base_from_camera'), 'ROI anchor measured camera')


def validate_roi_transport_observations(policy, observations, directory, handeye):
    for hold, bracket in (policy or {}).get('roi_transports', {}).items():
        transport = bracket['roi_transport']
        anchor = transport['anchor']
        native = resolve_native(anchor['native_path'], directory)
        require(native in observations, 'ROI anchor references an unknown native acquisition')
        original = observations[native]
        result, measurement = original['result'], original['measurement']
        require(result.get('status') == 'found' and result.get('found') is True
            and (result.get('quality') or {}).get('good') is True
            and anchor['acquisition_key'] == measurement['acquisition_key']
            and number(measurement.get('captured_at'), 'Original ROI anchor capture time') <= bracket['started_monotonic']
            and np.allclose(array(anchor['seam_point_camera_mm'], (3,), 'ROI anchor root'),
                array(result['seam_point_mm'], (3,), 'Original measured ROI anchor root'), rtol=0, atol=1e-9)
            and np.array_equal(pose(anchor['measured_base_from_camera'], 'ROI anchor camera'), original['camera']),
            'ROI anchor differs from its original hashed measured seam acquisition')
        matching = [item for item in observations.values()
            if item['measurement'].get('hold_monotonic') == hold]
        require(matching, 'ROI transport lacks a native acquisition at its measured Hold')
        projection_camera = pose(transport['measured_base_from_camera'], 'ROI projection camera')
        for item in matching:
            native_measurement = item['measurement']
            require(native_measurement.get('seam_search_roi') == transport
                and native_measurement['captured_at'] >= bracket['completed_monotonic'],
                'Native ROI projection differs from its completed same-Hold request')
            target = pose(native_measurement['requested_base_from_tcp'], 'ROI station requested endpoint')
            position, orientation = pose_error(projection_camera @ np.linalg.inv(handeye), target)
            require(position <= .0005+1e-12 and orientation <= .005+1e-10,
                'ROI projection camera was not converged to the same native station endpoint')
        if transport['basis'] != 'predicted_coped_view_target_not_a_measurement':
            point = np.asarray(transport['point_base_m'])
            require(any(item['result'].get('found') is True
                and number(item['measurement'].get('captured_at'), 'Prior measured ROI root time') <= bracket['started_monotonic']
                and np.allclose(
                item['camera'][:3, :3] @ (np.asarray(item['result']['seam_point_mm'])/1000.)+item['camera'][:3, 3],
                point, rtol=0, atol=1e-9) for item in observations.values()),
                'ROI claimed a measured root without an actual native seam acquisition')


def validate_endpoint_retry_observations(saved, policy, observations):
    """Bind the retry sidecar to independently hashed native acquisition evidence."""
    for proof in (policy or {}).get('endpoint_retry_bridges', []):
        retry = next(value for value in saved['endpoint_retries']
            if value['prior_hold_monotonic'] == proof['prior_hold_monotonic']
            and value['replacement_hold_monotonic'] == proof['replacement_hold_monotonic'])
        requested = pose(retry['requested_base_from_tcp'], 'Retry native requested endpoint')
        for after, hold in ((False, proof['prior_hold_monotonic']), (True, proof['replacement_hold_monotonic'])):
            matching = [item['measurement'] for item in observations.values()
                if item['measurement'].get('hold_monotonic') == hold]
            require(matching, 'Endpoint retry Hold bridge lacks matching native acquisition evidence')
            for measurement in matching:
                require(np.allclose(pose(measurement.get('requested_base_from_tcp'), 'Native retry endpoint'),
                    requested, rtol=0, atol=1e-12), 'Native acquisition changed the endpoint across a retry Hold bridge')
                gate = measurement.get('station_gate') or {}
                require(gate.get('station_deadline') == proof['station_deadline']
                    and gate.get('hold_monotonic') == hold and gate.get('retry_attempt') == int(after),
                    'Native acquisition differs from the retry Hold, budget or original deadline')
                captured = number(measurement.get('captured_at'), 'Retry native capture time')
                if after:
                    require(measurement.get('station_retry') == retry and captured >= proof['completed_monotonic'],
                        'Replacement Hold native acquisition lacks its completed endpoint retry proof')
                else:
                    require(captured <= retry['started_monotonic'],
                        'Old Hold native acquisition occurred after its endpoint replan started')
        proof['native_observation_evidence_verified'] = True


def generation_at(policy, stamp):
    current = policy['initial']
    for transition in policy['transitions']:
        if transition['observed_at'] > stamp:
            break
        current = transition['generation']
    return current


def validate_exposure_control(measurement, policy):
    control = measurement.get('exposure_control')
    if control is None:
        if policy is not None and policy['brackets']:
            sampled = number(measurement.get('captured_at'), 'Capture exposure qualification time')
            require(not any(item['completed_monotonic'] <= sampled for item in policy['brackets'].values()),
                'Manual generation measurement omitted its native exposure configuration proof')
        return
    require(policy is not None and control.get('mode') == 'owned_manual_bracket', 'Unknown/unbound manual exposure control mode')
    request_id = control.get('request_id')
    require(request_id in policy['brackets'], 'Manual native measurement lacks completed owned configuration proof')
    bracket = policy['brackets'][request_id]
    sampled = measurement['captured_at']
    prior = [item for item in policy['brackets'].values() if item['completed_monotonic'] <= sampled]
    require(prior and max(prior, key=lambda item: item['completed_monotonic'])['request_id'] == request_id,
        'Manual native measurement uses an old or incomplete exposure configuration')
    actual = number(control.get('actual_us'), 'Manual capture exposure')
    readback = control.get('configuration_readback', control.get('readback')) or {}
    native_controls = measurement['cloud'].get('controls') or {}
    require(control.get('requested_us') == bracket['requested_us'] and actual == bracket['actual_us']
        and number(readback.get('exposure_us'), 'Manual capture readback') == actual
        and readback == control.get('readback') == bracket.get('configuration_readback', bracket['readback'])
        and native_controls == readback
        and readback.get('auto_exposure') is False and native_controls.get('auto_exposure') is False,
        'Native manual controls differ from the completed bracket sensor readback')
    require(control.get('limits') == bracket['limits'], 'Manual measurement sensor/FPS bounds differ from configuration proof')
    require(measurement['cloud']['generation'] == bracket['generation']
        and measurement['quiet_since'] >= bracket['completed_monotonic'],
        'Manual measurement lacks new-generation quiet qualification after configuration completion')
    ae = measurement.get('auto_exposure') or {}
    require(ae.get('state') == 'owned_manual_bracket' and ae.get('reliable') is None
        and ae.get('request_id') == request_id,
        'Manual bracket must not claim reliable auto-exposure qualification')
    timing = measurement['cloud']['timing']
    for key in ('camera_exposure_us', 'exposure_duration_us'):
        if key in timing:
            require(abs(number(timing[key], 'Native hardware exposure duration')-actual) <= 2.,
                'Native hardware exposure differs from manual sensor readback')


def validate_observation(payload, binding, handeye, limitations, generation_policy=None):
    require(payload.get('schema') == 'autoweld.weld-native.v1', 'Unsupported native observation schema')
    measurement, result = payload['measurement'], payload['result']
    cloud = measurement['cloud']
    native_result = result.get('original_result') if result.get('status') == 'reacquire' else result
    require(isinstance(native_result, dict), 'Reacquire archive lacks its original detector result')
    provenance = native_result['provenance']
    require(cloud.get('units') == native_result.get('units') == 'mm'
        and cloud.get('coordinate_frame') == native_result.get('coordinate_frame') == 'camera',
        'Native and detector result must use camera-frame millimetres')
    for key in IMMUTABLE_IDENTITY:
        require(cloud.get(key) == binding[key] == provenance.get(key), f'Native/result/binding {key} mismatch')
    native_generation = cloud.get('generation')
    require(type(native_generation) is int and native_generation >= 0 and provenance.get('generation') == native_generation,
        'Native/result generation mismatch')
    if generation_policy is None:
        require(native_generation == binding['generation'], 'Native/result/binding generation mismatch')
    for key in ('sequence', 'read_started', 'timing'):
        require(provenance.get(key) == cloud.get(key), f'Detector/native {key} provenance mismatch')
    sequence = cloud.get('sequence')
    require(type(sequence) is int and sequence >= 0, 'Native sequence must be a nonnegative integer')
    require(measurement['acquisition_key'] == [binding['acquisition_session_id'], native_generation, sequence],
        'Acquisition key differs from native identity')
    xyz = np.asarray(cloud['xyz_mm'], float)
    uv = np.asarray(cloud['uv'], float)
    if xyz.shape == uv.shape == (0,):
        xyz, uv = xyz.reshape(0, 3), uv.reshape(0, 2)
        require(cloud.get('count') == 0 and cloud.get('bounds_mm') is None,
            'Empty native acquisition must declare zero points and no fabricated bounds')
    require(xyz.ndim == 2 and xyz.shape[1:] == (3,) and 0 <= len(xyz) <= 20000
        and uv.shape == (len(xyz), 2) and np.isfinite(xyz).all() and np.isfinite(uv).all(),
        'Native XYZ/UV must contain aligned finite native stripe points')
    if 'count' in cloud:
        require(cloud['count'] == len(xyz), 'Native XYZ count mismatch')
    if 'bounds_mm' in cloud and len(xyz):
        require(np.allclose(array(cloud['bounds_mm'], (2, 3), 'Native bounds'),
            [xyz.min(axis=0), xyz.max(axis=0)], rtol=0, atol=1e-9), 'Native XYZ bounds mismatch')
    if 'xyz_m' in cloud:
        metres = np.asarray(cloud['xyz_m'], float)
        if not len(xyz) and metres.shape == (0,):
            metres = metres.reshape(0, 3)
        require(metres.shape == xyz.shape and np.isfinite(metres).all()
            and np.allclose(metres, xyz/1000., rtol=0, atol=1e-12),
            'Native mm/metre XYZ copies disagree')
    require((cloud.get('frame_product') or {}).get('kind') == 'single_frame', 'Native product is not a single exposure')
    timing = cloud['timing']
    require(timing.get('host_clock_domain') == 'host_monotonic'
        and timing.get('sensor_shutter_mode') == 'global' and timing.get('motion_pairing_supported') is True,
        'Native exposure clock/shutter provenance is unqualified')
    mapping = timing.get('clock_mapping') or {}
    require(mapping.get('status') == 'qualified' and bool(mapping.get('mapping_id')),
        'Native camera/host clock mapping is unqualified')
    device = timing.get('device_frame_id')
    require(type(device) is int and device >= 0
        and timing.get('chunk_frame_id') == timing.get('exposure_end_frame_id') == device,
        'Owned-image/chunk/exposure-event device frame IDs differ')
    start = number(timing.get('exposure_host_start_s'), 'Exposure start')
    end = number(timing.get('exposure_host_end_s'), 'Exposure end')
    uncertainty = number(timing.get('exposure_time_uncertainty_s'), 'Exposure uncertainty')
    require(0 <= end-start <= .1 and 0 <= uncertainty <= .01, 'Exposure duration/uncertainty outside capture gate')
    for key in ('camera_exposure_us', 'exposure_duration_us'):
        if key in timing:
            require(abs(number(timing[key], 'Native hardware exposure')-(end-start)*1e6) <= 2.,
                'Native hardware exposure differs from mapped exposure interval duration')
    require(np.allclose(array(measurement['exposure_interval_s'], (2,), 'Exposure interval'),
        [start, end], rtol=0, atol=1e-9)
        and abs(number(measurement['exposure_uncertainty_s'], 'Saved uncertainty')-uncertainty) <= 1e-12,
        'Saved/native exposure intervals differ')
    cutoff = number(measurement['capture_cutoff'], 'Capture cutoff')
    hold = number(measurement['hold_monotonic'], 'Measured Hold time')
    quiet_since = number(measurement['quiet_since'], 'Quiet feedback start')
    require(hold <= quiet_since <= cutoff and cutoff-quiet_since >= .4-1e-9,
        'Measured Hold lacks 400 ms continuous post-Hold quiet evidence')
    require(start-uncertainty >= cutoff, 'Exposure overlaps or precedes stable capture cutoff')
    feedback = measurement['feedback']
    stamp = number(feedback.get('measured_at'), 'SDK feedback timestamp')
    acquired = number(cloud.get('read_started'), 'Native acquisition start')
    require(feedback.get('fresh') is True and feedback.get('stationary') is True,
        'Capture feedback is not fresh and stationary')
    require(stamp >= max(acquired, end+uncertainty), 'SDK feedback does not bracket the complete exposure')
    actual = pose(measurement['base_from_tcp'], 'Measured TCP')
    camera = pose(measurement['base_from_camera'], 'Measured camera')
    require(np.allclose(camera, actual @ handeye, rtol=0, atol=1e-10), 'Measured camera transform differs from TCP × hand-eye')
    quiet_pose = pose(measurement['quiet_pose'], 'Quiet reference pose')
    p_error, r_error = pose_error(actual, quiet_pose)
    require(p_error <= .0003+1e-12 and r_error <= .003+1e-10, 'Capture drift exceeds stationary pose gate')
    sampled = measurement.get('captured_at', measurement.get('sampled_monotonic'))
    if sampled is not None:
        require(0 <= number(sampled, 'Capture sample time')-stamp <= .1, 'Capture SDK feedback age exceeds 100 ms')
    else:
        limitations.add('Capture freshness is recorded as a service assertion; sample-return time is absent.')
    if generation_policy is not None:
        sampled = number(sampled, 'Generation-qualified capture time')
        require(native_generation == generation_at(generation_policy, sampled),
            'Native generation differs from explicit owned exposure transition history')
        prefix = [transition for transition in generation_policy['transitions'] if transition['observed_at'] <= sampled]
        require(measurement.get('generation_transitions') == prefix,
            'Capture generation transition provenance differs from the manifest history')
        if prefix:
            require(quiet_since >= prefix[-1]['observed_at'],
                'Changed exposure generation did not restart quiet qualification at the same Hold')
    validate_exposure_control(measurement, generation_policy)
    goal = measurement.get('requested_base_from_tcp')
    if goal is not None:
        p_error, r_error = pose_error(actual, pose(goal, 'Requested TCP'))
        require(p_error <= .0005+1e-12 and r_error <= .005+1e-10, 'Measured endpoint exceeds 0.5 mm / 0.005 rad')
        if 'goal_error' in measurement:
            reported = measurement['goal_error']
            require(abs(number(reported['position_m'], 'Saved position error')-p_error) <= 1e-9
                and abs(number(reported['orientation_rad'], 'Saved orientation error')-r_error) <= 1e-6,
                'Saved endpoint errors disagree with measured/requested poses')
    else:
        limitations.add('Requested endpoint pose is absent; endpoint convergence cannot be independently recomputed.')
    post = payload.get('post_capture') or measurement.get('post_capture')
    generation_matches = True
    if post:
        post_feedback = post['feedback']
        post_stamp = number(post_feedback['measured_at'], 'Post-analysis SDK time')
        require(post_feedback.get('fresh') is True and post_feedback.get('stationary') is True
            and post_stamp >= end+uncertainty, 'Post-analysis feedback is not fresh, stationary and after exposure')
        p_error, r_error = pose_error(pose(post['base_from_tcp'], 'Post-analysis TCP'), actual)
        require(p_error <= .0003+1e-12 and r_error <= .003+1e-10, 'Arm drifted during seam analysis')
        if 'observed_at' in post:
            require(0 <= number(post['observed_at'], 'Post-analysis sample time')-post_stamp <= .1,
                'Post-analysis SDK feedback age exceeds 100 ms')
        if generation_policy is not None:
            post_generation = generation_at(generation_policy, number(post.get('observed_at'), 'Post-analysis time'))
            require(post.get('camera_generation') == post_generation,
                'Post-analysis camera generation differs from explicit transition history')
            generation_matches = post_generation == native_generation
            require(post.get('generation_matches') is generation_matches,
                'Post-analysis native generation match flag is incorrect')
    else:
        limitations.add('Post-analysis feedback/pose is absent; stationary continuity after capture cannot be independently recomputed.')
    if 'quiet_duration_s' in measurement:
        require(number(measurement['quiet_duration_s'], 'Quiet duration') >= .4,
            'Recorded continuous quiet duration is shorter than 400 ms')
    direction = measurement.get('direction')
    require(type(direction) is int and direction in (-1, 0, 1), 'Invalid bidirectional observation direction')
    distance = number(measurement['distance_mm'], 'Nominal station distance')
    require(distance >= 0 and abs(distance-round(distance)) <= 1e-8 and (direction != 0 or distance == 0),
        'Observation station is not on the one-millimetre bidirectional grid')
    return dict(payload=payload, measurement=measurement, result=result, camera=camera,
        sequence=sequence, device=device, direction=direction, distance=distance, start=start,
        generation_matches=generation_matches)


def validate_world_record(record, observation, *, allow_provisional_coping=False):
    measurement, result, camera = observation['measurement'], observation['result'], observation['camera']
    require(result.get('status') == 'found' and result.get('found') is True
        and (result.get('quality') or {}).get('good') is True, 'Accepted path point has no good found seam')
    require(observation['generation_matches'], 'A changed-generation acquisition cannot qualify a measured path point')
    require(len(measurement['cloud']['xyz_mm']) >= 48, 'Accepted path point has insufficient native stripe support')
    require(record.get('detector') == result, 'Path detector result differs from native archived result')
    require(record.get('acquisition_key') == measurement['acquisition_key']
        and record.get('direction') == observation['direction']
        and record.get('distance_mm') == observation['distance'], 'Path/native acquisition or station association differs')
    require(np.allclose(pose(record['measured_base_from_tcp'], 'Path measured TCP'),
        pose(measurement['base_from_tcp'], 'Native measured TCP'), rtol=0, atol=1e-12), 'Path/native measured TCP mismatch')
    seam = array(result['seam_point_mm'], (3,), 'Camera seam')/1000.
    flanks = array(result['flank_lines_mm'], (2, 2, 3), 'Camera scan lines')/1000.
    torch = pose(result['camera_from_torch'], 'Camera torch')
    torch[:3, 3] /= 1000.
    delta = seam-torch[:3, 3]
    require(abs(np.linalg.norm(delta)-.150) <= 1e-8 and np.linalg.norm(delta/.150-torch[:3, 2]) <= 1e-6,
        'Torch +Z must point toward the seam from 150 mm clearance')
    stripe = array(result['stripe_direction_camera'], (3,), 'Camera stripe direction')
    advance = array(result['scan_direction_camera'], (3,), 'Camera scan direction')
    require(abs(np.linalg.norm(stripe)-1.) <= 1e-5 and abs(np.linalg.norm(advance)-1.) <= 1e-5
        and abs(stripe @ advance) <= 1e-5, 'Scan must advance perpendicular to the measured laser stripe')
    expected = camera[:3, :3] @ seam+camera[:3, 3]
    require(np.allclose(array(record['seam_point_base_m'], (3,), 'World seam'), expected, rtol=0, atol=1e-10),
        'World seam XYZ differs from measured camera transform')
    require(np.allclose(array(record['scan_lines_base_m'], (2, 2, 3), 'World scan lines'),
        flanks @ camera[:3, :3].T+camera[:3, 3], rtol=0, atol=1e-10), 'World scan lines differ from camera geometry')
    provisional = record.get('provisional_seed_scout', False)
    require(type(provisional) is bool, 'Provisional scout flag must be an explicit boolean')
    if provisional:
        require(allow_provisional_coping and record.get('station_basis') == 'coping_seed_scout'
            and record.get('direction') == 0 and record.get('distance_mm') == 0.
            and record.get('station_index') == 0 and measurement.get('coping_phase') == 'seed'
            and record.get('torch_orientation_policy') != 'coped_centerline'
            and not record.get('coping_geometry'),
            'Provisional coping locator cannot claim an angular station or fitted inward reference')
    if record.get('torch_orientation_policy') == 'coped_centerline':
        validate_coping_reference(record, expected)
        require(abs(np.linalg.norm(expected-camera[:3, 3])-.150) <= .002+1e-12,
            'Accepted coping seam is outside 150 ± 2 mm sensing range')
    else:
        require(np.allclose(pose(record['torch_pose_base_m'], 'World torch'), camera @ torch, rtol=0, atol=1e-10),
            'World torch pose differs from camera geometry')
        if provisional:
            require(seam[2] > 0, 'Provisional coping locator is behind the camera')
        else:
            require(abs(seam[2]-.150) <= .001+1e-12, 'Accepted seam is outside 150 ± 1 mm optical depth')


def validate_coping_reference(record, seam_base):
    """Independently reconstruct inward aiming from the retained cylinder axis."""
    geometry = record.get('coping_geometry') or {}
    model = geometry.get('model_seed_camera') or {}
    require(model.get('units') == 'mm' and model.get('coordinate_frame') == 'camera',
        'Coping model must remain in the frozen camera frame in millimetres')
    frame = pose(geometry.get('base_from_seed_camera'), 'Frozen coping camera')
    point = frame[:3, :3].T @ (seam_base-frame[:3, 3])*1000.
    require(np.allclose(array(geometry.get('measured_point_seed_camera_mm'), (3,), 'Frozen measured seam'),
        point, rtol=0, atol=1e-7), 'Coping pose substituted a predicted seam for measured geometry')
    index = geometry.get('coping_axis')
    require(type(index) is int and index in (0, 1) and len(model.get('axes', [])) == 2,
        'Coping pose lacks an explicit selected cylinder axis')
    residuals, closest = [], None
    for i, axis in enumerate(model['axes']):
        origin = array(axis.get('origin_mm'), (3,), 'Cylinder axis origin')
        direction = array(axis.get('direction'), (3,), 'Cylinder axis direction')
        require(abs(np.linalg.norm(direction)-1) <= 1e-7, 'Cylinder axis is not a unit vector')
        radius = number(axis.get('radius_mm'), 'Cylinder radius')
        require(radius > 0, 'Cylinder radius must be positive')
        projected = origin+direction*np.dot(point-origin, direction)
        residuals.append(abs(np.linalg.norm(point-projected)-radius))
        if i == index:
            closest = projected
    require(max(residuals) <= 1.+1e-9, 'Measured seam is farther than 1 mm from the fitted tube surfaces')
    require(np.allclose(array(geometry.get('radial_residual_mm'), (2,), 'Coping radial errors'),
        [residuals[index], residuals[1-index]], rtol=0, atol=1e-7), 'Saved cylinder residuals differ from measured geometry')
    require(np.allclose(array(geometry.get('centerline_closest_point_seed_camera_mm'), (3,), 'Coped centerline point'),
        closest, rtol=0, atol=1e-7), 'Coping aim is not toward the nearest point on its selected tube centerline')
    inward = closest-point
    inward /= np.linalg.norm(inward)
    require(np.allclose(array(geometry.get('inward_direction_seed_camera'), (3,), 'Coping inward direction'),
        inward, rtol=0, atol=1e-7), 'Saved inward direction differs from the selected cylinder')
    torch = pose(record.get('torch_pose_base_m'), 'Inward world torch')
    world_inward = frame[:3, :3] @ inward
    require(np.allclose(torch[:3, 2], world_inward, rtol=0, atol=1e-7)
        and np.allclose(torch[:3, 3], seam_base-.150*world_inward, rtol=0, atol=1e-10),
        'Coping torch reference does not point inward from 150 mm at the measured seam')
    selected, other = model['axes'][index], model['axes'][1-index]
    a, b = np.asarray(selected['direction']), np.asarray(other['direction'])
    cosine = float(a @ b)
    sine = np.sqrt(1.-cosine**2)
    e = (b-cosine*a)/sine
    local = point-np.asarray(selected['origin_mm'])
    signed = float(sine*np.dot(local, a)-cosine*np.dot(local, e))
    root_sign = geometry.get('root_sign')
    band = number(geometry.get('finite_root_ambiguity_band_mm'), 'Finite-root uncertainty band')
    require(root_sign in (-1, 1) and .2 <= band <= 1.
        and signed*root_sign > band, 'Measured coping point switched roots or is ambiguous near a crown')
    require(abs(number(geometry.get('finite_root_signed_coordinate_mm'), 'Saved finite-root coordinate')-signed) <= 1e-7,
        'Saved finite-root coordinate differs from actual seam geometry')
    association = geometry.get('association') or {}
    prediction = array(association.get('predicted_point_mm'), (3,), 'Selected-root association prediction')
    require(association.get('associated') is True and not association.get('singularity')
        and association.get('prediction_is_measurement') is False
        and association.get('measured_geometry_preserved') is True,
        'Coping reference lacks a measured association to a unique selected root')
    require(np.allclose(array(association.get('measured_point_mm'), (3,), 'Associated measured seam'),
        point, rtol=0, atol=1e-7), 'Coping association replaced its measured input')
    distance = float(np.linalg.norm(prediction-point))
    require(distance <= 1.+1e-9 and abs(number(association.get('residual_mm'), 'Root association residual')-distance) <= 1e-7,
        'Coping root association exceeds 1 mm or reports a false residual')
    for axis in model['axes']:
        d = prediction-np.asarray(axis['origin_mm'])
        v = np.asarray(axis['direction'])
        require(abs(np.linalg.norm(d-v*np.dot(d, v))-axis['radius_mm']) <= 1e-5,
            'Coping association prediction is not on both fitted cylinders')
    p_local = prediction-np.asarray(selected['origin_mm'])
    require(root_sign*(sine*np.dot(p_local, a)-cosine*np.dot(p_local, e)) > 0,
        'Coping association projected onto the opposite finite root')


def validate_coping_surface_profiles(saved, observations, directory):
    """Reconstruct fitted samples from actual native indices, without the SDK."""
    frame = pose(saved['coping_seed_base_from_camera'], 'Coping surface registration')
    for profile in saved.get('coping_profiles', []):
        provenance = profile['provenance']
        native = resolve_native(provenance['native_path'], directory)
        require(native in observations, 'Coping surface references an unknown native acquisition')
        require(hashlib.sha256(native.read_bytes()).hexdigest() == provenance['native_sha256'],
            'Coping surface native SHA256 mismatch')
        observation = observations[native]
        require(np.array_equal(pose(provenance['base_from_camera'], 'Surface measured camera'), observation['camera']),
            'Coping surface substituted its measured camera registration')
        cloud = observation['measurement']['cloud']
        source_path = provenance.get('native_cloud_path', 'cloud')
        require(source_path in ('cloud', 'full_profile'), 'Unknown coping surface native cloud source')
        if source_path == 'full_profile':
            full = cloud.get('full_profile')
            require(isinstance(full, dict), 'Coping surface full-profile source is missing')
            require(all(full.get(key) == cloud.get(key) for key in
                ('source_id', 'calibration_id', 'optical_intrinsics_id', 'acquisition_session_id',
                 'generation', 'sequence', 'read_started', 'image_size')),
                'Coping surface full profile belongs to another acquisition')
            cloud = full
        points = np.asarray(cloud['xyz_mm'], float)
        indices = provenance.get('native_point_indices', list(range(len(points))))
        require(isinstance(indices, list) and all(type(index) is int for index in indices)
            and all(0 <= index < len(points) for index in indices)
            and all(b > a for a, b in zip(indices, indices[1:])), 'Invalid coping surface native indices')
        selection = provenance.get('surface_return_selection')
        prepared = provenance.get('surface_profile_selection')
        if source_path == 'full_profile':
            require(isinstance(prepared, dict) and prepared.get('status') == 'selected'
                and prepared.get('schema') == 'weldfinder.coped_surface_profile_selection.v1'
                and prepared.get('counts_as_measured_seam') is False
                and prepared.get('native_point_indices') == indices,
                'Invalid primary full-profile surface selection')
            diagnostic = prepared.get('diagnostics') or {}
            require(diagnostic.get('native_geometry_preserved') is True
                and diagnostic.get('junction_exclusion_mm') == .5,
                'Coping surface changed its junction exclusion rule')
            reflection = cloud.get('reflection') or {}
            require(reflection.get('status', (reflection.get('summary') or {}).get('status')) == 'resolved_optical',
                'Coping surface used optically ambiguous primary returns')
            pixels = np.asarray(cloud['uv'], float)
            require(pixels.shape == (len(points), 2) and np.isfinite(pixels).all()
                and np.all(np.diff(pixels[:, 0]) > 0), 'Coping primary surface pixels are not ordered native returns')
            breaks = np.flatnonzero((np.diff(pixels[:, 0]) > 3.) | (abs(np.diff(pixels[:, 1])) > 8.))+1
            ends = [0, *breaks.tolist(), len(points)]
            connected = np.asarray([index for first, stop in zip(ends, ends[1:]) if stop-first >= 12
                                    for index in range(first, stop)], int)
            edges = diagnostic.get('gap_boundary_input_indices')
            if edges is not None:
                require(isinstance(edges, list) and len(edges) == 2 and all(type(i) is int for i in edges)
                    and 0 <= edges[0] < edges[1] < len(points), 'Invalid native gap boundary indices')
                left = connected[(connected <= edges[0])
                    & (np.linalg.norm(points[connected]-points[edges[0]], axis=1) >= .5)]
                right = connected[(connected >= edges[1])
                    & (np.linalg.norm(points[connected]-points[edges[1]], axis=1) >= .5)]
            else:
                hint = np.asarray(diagnostic.get('split_hint_mm'), float)
                require(hint.shape == (3,) and np.isfinite(hint).all(), 'Missing native surface association hint')
                split = int(np.argmin(np.linalg.norm(points-hint, axis=1)))
                require(np.linalg.norm(points[split]-hint) <= 2., 'Coping surface hint lacks nearby actual support')
                away = np.linalg.norm(points[connected]-hint, axis=1) >= .5
                left, right = connected[(connected < split) & away], connected[(connected >= split) & away]
            expected = np.r_[left, right].tolist()
            require(indices == expected and min(len(left), len(right)) >= 12,
                'Coping full-profile selection differs from declared native support/exclusion')
            require(prepared.get('surface_labels') == [0]*len(left)+[1]*len(right)
                and profile.get('surface_labels') == prepared['surface_labels'],
                'Coping full-profile surface membership differs from native branches')
            require(provenance.get('raw_native_point_count') == len(points)
                and provenance.get('native_point_count') == len(indices), 'Coping full-profile count mismatch')
        if selection is not None:
            require(selection.get('parameters') == dict(min_run_points=12, max_scan_gap_px=3., max_jump_px=8.)
                and selection.get('native_geometry_preserved') is True,
                'Coping surface changed its declared image-space support rule')
            pixels = np.asarray(cloud['uv'], float)
            require(pixels.shape == (len(points), 2) and np.isfinite(pixels).all()
                and np.all(np.diff(pixels[:, 0]) > 0), 'Coping surface has invalid ordered native pixels')
            boundaries = np.flatnonzero((np.diff(pixels[:, 0]) > 3.)
                | (abs(np.diff(pixels[:, 1])) > 8.))+1
            ends = [0, *boundaries.tolist(), len(points)]
            expected = [index for first, stop in zip(ends, ends[1:]) if stop-first >= 12
                        for index in range(first, stop)]
            require(indices == expected, 'Coping surface selection differs from native connected runs')
            require(provenance.get('raw_native_point_count') == len(points)
                and provenance.get('native_point_count') == len(indices), 'Coping surface count provenance mismatch')
        frozen = np.linalg.inv(frame) @ observation['camera']
        actual = points[indices] @ frozen[:3, :3].T+frozen[:3, 3]*1000.
        reported = np.asarray(profile['xyz_mm'], float)
        require(reported.shape == actual.shape and np.allclose(reported, actual, atol=1e-8, rtol=0),
            'Coping fit points differ from actual selected native geometry')


def validate_coping_coverage(saved, path, observations, directory):
    """Audit angular coverage without treating predictions as acquisitions."""
    settings, coverage = saved['settings'], saved.get('coping_coverage') or {}
    if saved.get('coping_profiles'):
        validate_coping_surface_profiles(saved, observations, directory)
    require(saved.get('minimum_basis') == 'measured_coped_angular_span', 'Coping coverage has a linear minimum basis')
    requested = number(settings.get('target_arc_deg'), 'Requested coping arc')
    require(10 <= requested <= 180 and coverage.get('requested_arc_deg') == requested,
        'Coping requested angular coverage differs from joint settings')
    provisional = [record for record in path if record.get('provisional_seed_scout') is True]
    if provisional:
        require(len(provisional) == len(path) == 1, 'Provisional locator cannot bridge or mix with angular weld stations')
        require(saved.get('status') == 'incomplete' and saved.get('minimum_satisfied') is False
            and saved.get('coverage_complete') is False and coverage.get('coverage_complete') is False
            and saved.get('replay_scope') == 'measured_section_only',
            'Provisional-only coping result must remain incomplete and not qualify coverage')
        for key in ('planned_arc_deg', 'observed_arc_deg', 'positive_observed_deg', 'negative_observed_deg'):
            require(number(coverage.get(key), f'Provisional {key}') == 0.,
                'Provisional coping locator cannot establish planned or observed angular coverage')
        require(coverage.get('measured_station_count') == 0 and coverage.get('provisional_seed_scout_count') == 1
            and number(saved.get('seam_length_mm'), 'Provisional path length') == 0.,
            'Provisional coping locator cannot count as a measured orbit station or path length')
        return dict(requested_arc_deg=requested, planned_arc_deg=0., observed_arc_deg=0.,
            positive_observed_deg=0., negative_observed_deg=0., complete=False,
            measured_station_count=0, provisional_seed_scout_count=1)
    require((saved.get('joint_model') or {}).get('status') == 'qualified', 'Coping path lacks a qualified joint model')
    model = saved['joint_model']['model']
    axes = model.get('axes', [])
    require(len(axes) == 2, 'Coping path needs two fitted tubes')
    selected_axis = settings.get('coping_axis', 0)
    require(type(selected_axis) is int and selected_axis in (0, 1), 'Coping branch side must be explicit and valid')
    for index, axis in enumerate(axes):
        key = 'branch_diameter_mm' if index == selected_axis else 'main_diameter_mm'
        require(abs(number(axis['radius_mm'], 'Model radius')*2-number(settings[key], 'Selected tube diameter')) <= 1e-8,
            'Coping fit changed a selected tube diameter')
    angle = number(settings['joint_angle_deg'], 'Selected tube crossing angle')
    require(abs(abs(np.dot(array(axes[0]['direction'], (3,), 'First tube axis'),
        array(axes[1]['direction'], (3,), 'Second tube axis')))-abs(np.cos(np.deg2rad(angle)))) <= 1e-7,
        'Fitted tube axes violate the selected crossing angle')
    extents = {1: 0., -1: 0.}
    seed = next((record for record in path if record['direction'] == 0), None)
    require(seed is not None, 'Coping path has no measured seed station')
    for record in path:
        require(record.get('torch_orientation_policy') == 'coped_centerline'
            and record.get('station_basis') == 'coped_angular_stations'
            and record.get('station_index') == record['distance_mm'],
            'Coping path lacks explicit angular station and inward-pose provenance')
        geometry = record['coping_geometry']
        require(geometry['model_seed_camera'] == model, 'Coping station used a different fitted model')
        require(geometry.get('root_sign') == (saved.get('coping_curve') or {}).get('root_sign'),
            'Coping station changed the selected finite-assembly root')
        require(geometry['base_from_seed_camera'] == saved.get('coping_seed_base_from_camera'),
            'Coping station changed the frozen camera registration')
        axis = model['axes'][geometry['coping_axis']]
        origin, direction = np.asarray(axis['origin_mm']), np.asarray(axis['direction'])
        frame = pose(geometry['base_from_seed_camera'], 'Coping angle registration')
        baseline = frame[:3, :3].T @ (np.asarray(seed['seam_point_base_m'])-frame[:3, 3])*1000.-origin
        baseline -= direction*np.dot(baseline, direction)
        baseline /= np.linalg.norm(baseline)
        radial = np.asarray(geometry['measured_point_seed_camera_mm'])-origin
        radial -= direction*np.dot(radial, direction)
        radial /= np.linalg.norm(radial)
        measured = float(np.rad2deg(np.arctan2(radial @ np.cross(direction, baseline), radial @ baseline)))
        reported = number(record.get('angular_theta_deg'), 'Measured coping angle')
        # The original seed and its fresh post-fit recheck may differ slightly;
        # all angles must share the model's original seed, persisted separately.
        expected = measured+number(seed.get('angular_theta_deg'), 'Coping seed angular offset')
        require(abs(reported-expected) <= 1e-5, 'Reported coping angle differs from measured radial geometry')
        if record['direction']:
            extents[record['direction']] = max(extents[record['direction']], record['direction']*reported)
        native = observations[resolve_native(record['native_path'], directory)]
        require(native['measurement'].get('station_basis') == 'coped_angular_stations',
            'Coping native acquisition lacks angular station provenance')
    for direction, key in ((1, 'positive_observed_deg'), (-1, 'negative_observed_deg')):
        require(abs(number(coverage.get(key), key)-extents[direction]) <= 1e-6,
            'Observed coping bound includes unmeasured predictions')
    observed = sum(extents.values())
    require(abs(number(coverage.get('observed_arc_deg'), 'Measured coping arc')-observed) <= 1e-6,
        'Reported coping span differs from actual measured seam angles')
    planned = number(coverage.get('planned_arc_deg'), 'Preflight coping arc')
    require(0 <= planned <= requested+1e-6, 'Preflight coping span exceeds requested coverage')
    complete = all(value+1e-8 >= requested/2 for value in extents.values())
    require(coverage.get('measured_station_count') == len(path), 'Coping station count includes model predictions')
    for direction in (-1, 1):
        rows = sorted((row for row in path if row['direction'] == direction), key=lambda row: row['station_index'])
        previous_angle = previous_request = 0.
        for row in rows:
            angle = number(row['angular_theta_deg'], 'Observed angular progress')
            requested_angle = number(row.get('requested_angular_theta_deg'), 'Requested angular station')
            advance = direction*(angle-previous_angle)
            requested_advance = abs(requested_angle-previous_request)
            require(1e-6 < advance <= max(requested_advance*1.5+1., 3.),
                'Coping angular coverage bridges a duplicate, reversal or unmeasured gap')
            previous_angle, previous_request = angle, requested_angle
    require(coverage.get('coverage_complete') is complete and saved.get('coverage_complete') is complete
        and saved.get('minimum_satisfied') is complete, 'Coping completion flag disagrees with measured angular span')
    require(saved.get('status') == ('complete' if complete else 'incomplete'), 'Coping result status disagrees with coverage')
    return dict(requested_arc_deg=requested, planned_arc_deg=planned, observed_arc_deg=observed,
        positive_observed_deg=extents[1], negative_observed_deg=extents[-1], complete=complete)


def validate_search(saved, observations, directory, minimum):
    if saved.get('minimum_basis') != 'measured_scan_plane_extent':
        return None
    frame = saved.get('search_frame') or {}
    require(frame.get('units') == 'm' and frame.get('coordinate_frame') == 'robot_base', 'Search frame must use robot-base metres')
    origin = array(frame['center_camera_origin_base_m'], (3,), 'Search center camera origin')
    normal = array(frame['normal_base'], (3,), 'Search plane normal')
    require(abs(np.linalg.norm(normal)-1.) <= 1e-6, 'Search plane normal must be unit length')
    center_file = resolve_native(frame['center_native_path'], directory)
    require(center_file in observations, 'Search frame center lacks a native observation')
    center = observations[center_file]
    require(center['direction'] == 0 and center.get('disposition') == 'path_point'
        and np.allclose(origin, center['camera'][:3, 3], rtol=0, atol=1e-10),
        'Search center is not the accepted measured center acquisition')
    initial_direction = array(center['result']['scan_direction_camera'], (3,), 'Initial camera scan direction')
    require(np.allclose(normal, center['camera'][:3, :3] @ initial_direction, rtol=0, atol=1e-6),
        'Search plane normal differs from the initial measured scan direction')
    stations = saved.get('search_stations')
    require(isinstance(stations, list) and len(stations) <= MAX_OBSERVATIONS, 'Missing/unbounded measured search stations')
    used, extents = set(), {1: 0., -1: 0.}
    for station in stations:
        native = resolve_native(station['native_path'], directory)
        require(native in observations and native not in used, 'Measured search station lacks a unique native acquisition')
        used.add(native)
        observation = observations[native]
        require(observation.get('disposition') in ('centering', 'path_point', 'endpoint_observation')
            and observation['result'].get('status') in ('found', 'no_seam') and observation['generation_matches'],
            'Rejected/reacquired/poor observation cannot establish measured search extent')
        require(station.get('acquisition_key') == observation['measurement']['acquisition_key']
            and station.get('direction') == observation['direction']
            and station.get('nominal_distance_mm') == observation['distance'], 'Search station/native association differs')
        measured_origin = observation['camera'][:3, 3]
        require(np.allclose(array(station['camera_origin_base_m'], (3,), 'Search station camera origin'),
            measured_origin, rtol=0, atol=1e-10), 'Search station substituted a nominal camera origin')
        progress = float((measured_origin-origin) @ normal*1000.)
        require(abs(number(station['signed_progress_mm'], 'Search projection')-progress) <= 1e-7,
            'Search station projected progress differs from measured camera transform')
        direction = observation['direction']
        if direction:
            extents[direction] = max(extents[direction], direction*progress)
    expected = {native for native, observation in observations.items()
        if observation.get('disposition') in ('centering', 'path_point', 'endpoint_observation')}
    require(used == expected, 'Measured search station list omits or adds processed qualified observations')
    total = extents[1]+extents[-1]
    require(abs(number(saved['search_positive_mm'], 'Positive search extent')-extents[1]) <= 1e-7
        and abs(number(saved['search_negative_mm'], 'Negative search extent')-extents[-1]) <= 1e-7
        and abs(number(saved['search_distance_mm'], 'Total search span')-total) <= 1e-7,
        'Reported explored scan span differs from measured scan-plane extent')
    return dict(total_mm=total, positive_mm=extents[1], negative_mm=extents[-1],
        minimum_satisfied=all(value+1e-8 >= minimum/2. for value in extents.values()))


def audit_manifest(manifest):
    manifest = Path(manifest).resolve()
    saved = load_json(manifest)
    require(saved.get('schema') == 'autoweld.weld-path.v1', 'Unsupported weld path schema')
    require(saved.get('units') == 'm' and saved.get('coordinate_frame') == 'robot_base', 'World path frame/units mismatch')
    path = saved['path']
    require(isinstance(path, list) and len(path) <= MAX_OBSERVATIONS, 'Invalid/unbounded weld path')
    require(digest(path) == saved['path_sha256'], 'Path SHA256 integrity mismatch')
    binding, settings = saved['binding'], saved['settings']
    coping = settings.get('joint_kind') == 'coped_tube'
    for key in (*IDENTITY, 'model_fingerprint'):
        require(binding.get(key) is not None and binding[key] != '', f'Missing binding {key}')
    handeye = pose(binding['tcp_from_camera'], 'Bound measured hand-eye')
    generation_policy = validate_generation_policy(saved, manifest.parent)
    selected_start = settings.get('start_pose')
    require(isinstance(selected_start, str) and 1 <= len(selected_start) <= 80
        and bool(selected_start.strip()) and settings.get('step_mm') == 1.
        and settings.get('standoff_mm') == 150. and settings.get('end_confirmations') == 3,
        'Weld path settings differ from required start, interval, standoff or endpoint confirmations')
    minimum = number(settings['min_distance_mm'], 'Requested minimum distance')
    require(1 <= minimum <= 100, 'Invalid requested minimum distance')
    limitations = set()
    files = {}
    for record in path:
        native = resolve_native(record['native_path'], manifest.parent)
        files[native] = load_json(native)
    # Discover no-seam and preparation observations by schema, not a hard-coded
    # filename convention. They are required evidence for physical endpoints.
    candidates = list(manifest.parent.glob('*.json'))
    require(len(candidates) <= MAX_OBSERVATIONS+10, 'Unbounded observation directory')
    for candidate in candidates:
        if candidate.resolve() == manifest:
            continue
        value = load_json(candidate)
        if value.get('schema') == 'autoweld.weld-native.v1':
            files[candidate.resolve()] = value
    require(len(files) <= MAX_OBSERVATIONS, 'Too many weld observations')
    native_files = saved.get('native_files')
    hashes = saved.get('observations_sha256')
    if native_files is not None:
        require(isinstance(native_files, list) and len(native_files) <= MAX_OBSERVATIONS
            and digest(native_files) == saved.get('native_files_sha256'), 'Native file index SHA256 integrity mismatch')
        hashes = {}
        for index, entry in enumerate(native_files, 1):
            require(isinstance(entry, dict) and entry.get('path') not in hashes, 'Invalid/duplicate native hash index entry')
            require(entry.get('observation_index') == index, 'Native hash index observation order is not contiguous')
            hashes[entry['path']] = entry['sha256']
    if hashes is not None:
        require(isinstance(hashes, dict) and len(hashes) == len(files), 'Native hash index/count differs from archive')
        indexed = {}
        for filename, expected in hashes.items():
            native = resolve_native(filename, manifest.parent)
            require(native in files and native not in indexed, 'Native hash index references an unknown or duplicate observation')
            require(hashlib.sha256(native.read_bytes()).hexdigest() == expected, f'Native observation SHA256 mismatch: {native.name}')
            indexed[native] = expected
    else:
        limitations.add('Raw observation file hashes are absent; internal native XYZ edits cannot all be detected without rerunning the detector.')
    observations = {filename: validate_observation(value, binding, handeye, limitations, generation_policy)
        for filename, value in files.items()}
    if native_files is not None:
        for entry in native_files:
            native = resolve_native(entry['path'], manifest.parent)
            require(entry.get('status') == observations[native]['result'].get('status'),
                'Native hash index status differs from observation')
            disposition = entry.get('disposition')
            if generation_policy is not None:
                allowed = ('reacquire', 'rejected', 'centering', 'path_point', 'endpoint_observation')
                if coping:
                    allowed += ('coping_scout', 'coping_scout_no_root', 'coping_unassociated_root',
                                'coping_view_no_seam', 'coping_surface_profile', 'coping_surface_candidate')
                require(disposition in allowed,
                    'Native generation policy needs an explicit final observation disposition')
            observations[native]['disposition'] = disposition
            observations[native]['observation_index'] = entry.get('observation_index')
    validate_endpoint_retry_observations(saved, generation_policy, observations)
    validate_roi_transport_observations(generation_policy, observations, manifest.parent, handeye)
    ordered_observations = sorted(observations.values(), key=lambda item: item['sequence'])
    require(len({item['sequence'] for item in ordered_observations}) == len(observations)
        and len({item['device'] for item in ordered_observations}) == len(observations), 'Duplicate native exposure reused')
    require(all(b['start'] > a['start'] and b['device'] > a['device']
        for a, b in zip(ordered_observations, ordered_observations[1:])), 'Native sequence/device/exposure ordering regressed')
    path_keys, signed = set(), []
    for record in path:
        native = resolve_native(record['native_path'], manifest.parent)
        require(native not in path_keys, 'Native observation reused for multiple path points')
        path_keys.add(native)
        if 'native_sha256' in record:
            require(record['native_sha256'] == hashlib.sha256(native.read_bytes()).hexdigest(),
                'Path point native file SHA256 binding mismatch')
        if generation_policy is not None:
            disposition = 'coping_scout' if coping and record.get('provisional_seed_scout') is True else 'path_point'
            require(observations[native].get('disposition') == disposition,
                'Archived-only or reacquire observation was used as a measured path point')
        validate_world_record(record, observations[native], allow_provisional_coping=coping)
        signed.append(record['direction']*record['distance_mm'])
    require(all(b-a == 1. for a, b in zip(signed, signed[1:])), 'Ordered weld path skips or duplicates a one-millimetre station')
    if path:
        require(sum(record['direction'] == 0 for record in path) == 1, 'Weld path must have exactly one measured center station')
    if coping:
        angular = validate_coping_coverage(saved, path, observations, manifest.parent)
        angular_path = [record for record in path if record.get('provisional_seed_scout') is not True]
        points = np.asarray([record['seam_point_base_m'] for record in angular_path], float)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([])
        require(np.all((lengths > 1e-9) & (lengths <= .005+1e-12)),
            'Measured coping section has duplicate points or a greater-than-5 mm gap')
        length = float(lengths.sum()*1000.)
        require(abs(number(saved['seam_length_mm'], 'Measured coping length')-length) <= 1e-7,
            'Reported coping length includes predicted geometry')
        scope = 'full_observed_path' if angular['complete'] else 'measured_section_only'
        require(saved.get('replay_scope') == scope, 'Coping replay scope disagrees with measured coverage')
        return dict(schema='autoweld.weld-path-audit.v1', passed=True, manifest=str(manifest),
            session_id=saved.get('session_id'), status=saved['status'], complete=angular['complete'],
            joint_kind='coped_tube', coverage_complete=angular['complete'], replay_scope=scope,
            replay_eligible=len(angular_path) >= 2 and length > 1e-8,
            replay_eligibility_basis='continuous measured section; live full-pose motion proof remains required',
            native_observations=len(observations), measured_seam_points=len(angular_path), seam_length_mm=length,
            provisional_seed_scout_points=len(path)-len(angular_path),
            measured_angular_coverage=angular, path_sha256=saved['path_sha256'],
            all_world_seam_flank_and_torch_transforms_match=True,
            torch_orientation_policy='coped_centerline' if angular_path else 'provisional_local_bisector_only',
            detector_rerun=False, raw_observation_hashes_verified=hashes is not None,
            verified_endpoint_retry_bridges=[] if generation_policy is None else generation_policy['endpoint_retry_bridges'],
            evidence_limitations=sorted(limitations),
            physical_limit='File consistency does not establish absolute calibration accuracy or physical encoder sample time.')
    search = validate_search(saved, observations, manifest.parent, minimum)
    for direction in (-1, 1):
        all_side = [item for item in ordered_observations if item['direction'] == direction]
        require(not all_side or all_side[0]['distance'] == 1., 'Directional scan must start at the first one-millimetre station')
        require(all(b['distance']-a['distance'] in (0., 1.) for a, b in zip(all_side, all_side[1:])),
            'Directional observations skipped or reversed a one-millimetre station')
        side = [item for item in all_side if item.get('disposition') not in ('reacquire', 'rejected')
            and item['result'].get('status') != 'reacquire']
        last_found_distance = -1.
        first_absent_distance = None
        for item in side:
            status = item['result'].get('status')
            if status == 'no_seam':
                require(item['result'].get('found') is False and (item['result'].get('quality') or {}).get('good') is True,
                    'Poor/ambiguous laser return cannot confirm a seam endpoint')
                require(item['generation_matches'], 'Changed-generation acquisition cannot qualify a no-seam endpoint')
                require(len(item['measurement']['cloud']['xyz_mm']) >= 48,
                    'No-seam endpoint has insufficient native stripe support')
                if generation_policy is not None:
                    require(item.get('disposition') == 'endpoint_observation',
                        'Archived-only no-seam observation was used as an endpoint')
                if first_absent_distance is None:
                    first_absent_distance = item['distance']
            elif status == 'found':
                require(first_absent_distance is None, 'Interior no-seam gap is bridged by later found points')
                last_found_distance = item['distance']
            elif saved.get('status') in ('complete', 'insufficient_length'):
                raise AuditError('Complete weld discovery contains an invalid or ambiguous observation')
        name = 'positive' if direction > 0 else 'negative'
        endpoint = saved['endpoints'].get(name) or {}
        if saved.get('status') in ('complete', 'insufficient_length') or endpoint.get('confirmed') is True:
            require(endpoint.get('confirmed') is True and endpoint.get('no_seam_stations', 0) >= 3,
                f'{name}: physical endpoint is not confirmed')
            absent = [item for item in side if item['result'].get('status') == 'no_seam'
                and item['distance'] > last_found_distance]
            require(len(absent) >= 3 and all(b['distance']-a['distance'] == 1.
                for a, b in zip(absent, absent[1:])), f'{name}: lacks three fresh contiguous no-seam stations')
            require(endpoint.get('no_seam_stations') == len(absent)
                and endpoint.get('distance_mm') == absent[-1]['distance']
                and (search is not None or absent[-1]['distance'] >= minimum/2.),
                f'{name}: endpoint progress disagrees with observations')
            if search is not None:
                extent = search['positive_mm' if direction > 0 else 'negative_mm']
                require(abs(number(endpoint['measured_search_extent_mm'], f'{name} measured extent')-extent) <= 1e-7
                    and extent+1e-8 >= minimum/2., f'{name}: endpoint lacks minimum measured scan-plane search extent')
            if generation_policy is not None:
                used = endpoint.get('observations')
                require(isinstance(used, list) and len(used) == 3,
                    f'{name}: final endpoint lacks three explicitly bound observations')
                final_absent = absent[-3:]
                for link, expected in zip(used, final_absent):
                    native = resolve_native(link['native_path'], manifest.parent)
                    require(observations[native] is expected
                        and link.get('acquisition_key') == expected['measurement']['acquisition_key']
                        and link.get('distance_mm') == expected['distance']
                        and link.get('observation_index') == expected.get('observation_index'),
                        f'{name}: endpoint linked an archived-only or different acquisition')
            last_records = [record for record in path if record['direction'] == direction]
            last_record = max(last_records, key=lambda record: record['distance_mm']) if last_records else next(
                record for record in path if record['direction'] == 0)
            require(np.allclose(array(endpoint['last_seam_point_base_m'], (3,), f'{name} last seam'),
                last_record['seam_point_base_m'], rtol=0, atol=1e-10), f'{name}: endpoint seam position differs from path')
    points = np.asarray([record['seam_point_base_m'] for record in path], float)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([])
    require(np.all((segments > 1e-9) & (segments <= .005+1e-12)), 'Measured path contains repeated points or a greater-than-5 mm branch jump')
    length = float(segments.sum()*1000.)
    require(abs(number(saved['seam_length_mm'], 'Reported seam length')-length) <= 1e-7,
        'Reported seam length differs from all measured world points')
    minimum_satisfied = search['minimum_satisfied'] if search is not None else length+1e-8 >= minimum
    require(saved.get('minimum_satisfied') is minimum_satisfied, 'Minimum-length flag disagrees with measured length')
    if saved.get('status') == 'complete':
        require(minimum_satisfied and len(path) >= 2, 'Complete path fails actual measured minimum distance')
    if saved.get('status') == 'insufficient_length':
        require(not minimum_satisfied, 'Insufficient-length status disagrees with actual measured length')
    coverage = saved.get('coverage_complete', saved.get('status') == 'complete')
    scope = saved.get('replay_scope', 'full_observed_path' if saved.get('status') == 'complete' else None)
    if saved.get('status') == 'incomplete':
        require(coverage is False and scope == 'measured_section_only', 'Incomplete path must explicitly limit coverage and replay scope')
    if saved.get('status') == 'complete':
        require(coverage is True and scope == 'full_observed_path', 'Complete path lacks complete coverage scope')
    replayable_section = len(path) >= 2 and length > 1e-8 and saved.get('status') in ('complete', 'incomplete')
    return dict(schema='autoweld.weld-path-audit.v1', passed=True, manifest=str(manifest),
        session_id=saved.get('session_id'), status=saved.get('status'), complete=saved.get('status') == 'complete',
        coverage_complete=coverage, replay_scope=scope, replay_eligible=replayable_section,
        replay_eligibility_basis='archive provenance and continuous measured section; live motion preflight remains required',
        unconfirmed_ends=[name for name, endpoint in (saved.get('endpoints') or {}).items()
            if not endpoint or endpoint.get('confirmed') is not True], path_sha256=saved['path_sha256'],
        native_observations=len(observations), measured_seam_points=len(path), seam_length_mm=length,
        minimum_distance_mm=minimum, minimum_satisfied=minimum_satisfied,
        minimum_basis=saved.get('minimum_basis', 'legacy_measured_seam_length'),
        measured_search=search,
        all_world_seam_flank_and_torch_transforms_match=True, detector_rerun=False,
        verified_endpoint_retry_bridges=[] if generation_policy is None else generation_policy['endpoint_retry_bridges'],
        raw_observation_hashes_verified=hashes is not None, evidence_limitations=sorted(limitations),
        physical_limit='File consistency does not establish absolute calibration accuracy or firmware encoder sample age.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_manifest(args.manifest)
    except (AuditError, KeyError, ValueError, TypeError, OSError) as error:
        result = dict(schema='autoweld.weld-path-audit.v1', passed=False, manifest=str(args.manifest), error=str(error))
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload+'\n')
    print(payload)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
