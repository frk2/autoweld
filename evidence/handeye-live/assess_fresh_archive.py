"""Offline QA of the finished fresh hand-eye archive; never installs anything."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.calibration import solve_handeye, transform

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ACTIVE = ROOT / 'runtime/agents-live/handeye.json'
MODEL = ROOT / 'runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json'
OLD = ROOT / 'evidence/handeye-review/candidate.json'


def evaluate(samples, x, reference=None):
    boards = np.array([transform(s['base_from_tcp']) @ x @ transform(s['camera_from_target_m']) for s in samples])
    positions, rotations = boards[:, :3, 3], Rotation.from_matrix(boards[:, :3, :3])
    if reference is None:
        reference = np.eye(4)
        reference[:3, :3] = rotations.mean().as_matrix()
        reference[:3, 3] = positions.mean(0)
    distance = np.linalg.norm(positions-reference[:3, 3], axis=1)*1000
    angle = np.rad2deg((Rotation.from_matrix(reference[:3, :3]).inv()*rotations).magnitude())
    return dict(translation_rms_mm=float(np.sqrt(np.mean(distance**2))),
                translation_max_mm=float(distance.max()), rotation_rms_deg=float(np.sqrt(np.mean(angle**2))),
                rotation_max_deg=float(angle.max()), per_view_translation_mm=distance.tolist(),
                per_view_rotation_deg=angle.tolist(),
                gates_passed=bool(np.sqrt(np.mean(distance**2)) <= 3 and np.sqrt(np.mean(angle**2)) <= 1.5)), reference


def difference(a, b):
    return dict(translation_mm=float(np.linalg.norm(a[:3, 3]-b[:3, 3])*1000),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude())))


def assess():
    raw = ACTIVE.read_bytes()
    archive = json.loads(raw)
    assert ACTIVE.stat().st_mtime > 1792799122, 'Older archive'
    assert archive['schema_version'] == 2 and archive['views'] == 25
    samples = archive['paired_samples']
    assert len(samples) == 25
    model_raw = MODEL.read_bytes()
    model = json.loads(model_raw)
    intrinsic_identity = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert archive['source_id'] == model['source_id']
    assert archive['calibration_id'] == intrinsic_identity
    assert archive['units'] == 'm' and archive['coordinate_frame'] == 'tcp'
    freshness, reversals, sequences, starts, sessions = [], [], [], [], set()
    prepost = []
    ref_axis = None
    for i, s in enumerate(samples):
        assert s['source_id'] == model['source_id'] and s['calibration_id'] == intrinsic_identity
        assert s['native_validation'] is True and s['units'] == 'm'
        assert s['image_size'] == model['image_size'] and s['grid'] == model['board']
        assert isinstance(s['timing'], dict) and isinstance(s['controls'], dict)
        timing = s['timing']
        assert timing['acquisition_session_id'] == s['acquisition_session_id']
        assert timing['host_read_started_s'] == s['read_started']
        assert timing['host_read_completed_s'] == s['read_completed']
        assert timing['camera_frame_id'] == s['sequence']
        assert timing['camera_timestamp_ns'] > 0
        assert timing['metadata_errors'] == {} and not timing['reconnect_required']
        assert timing['duplicate_total'] == timing['regression_total'] == 0
        p = np.array(s['raw_corners_native_px'])
        normalized = np.array(s['corners_native_px'])
        assert p.shape == normalized.shape == (35, 2) and np.isfinite(p).all()
        assert np.all(p >= 0) and np.all(p < np.array(s['image_size']))
        raw_c, normalized_c = transform(s['raw_camera_from_target_m']), transform(s['camera_from_target_m'])
        n = s['normalization']
        assert n['index'] == i and n['reference_index'] == 0
        correction = transform(n['board180_transform_m'])
        expected = raw_c @ correction if n['reversed_180'] else raw_c
        np.testing.assert_allclose(normalized_c, expected, atol=1e-12)
        np.testing.assert_array_equal(normalized, p[::-1] if n['reversed_180'] else p)
        if n['reversed_180']:
            reversals.append(i)
            assert n['image_axis_dot'] < -np.cos(np.deg2rad(20))
            assert n['tcp_rotation_from_reference_deg'] < 20
        else:
            assert n['image_axis_dot'] > np.cos(np.deg2rad(20))
        axis = normalized[4]-normalized[0]
        axis /= np.linalg.norm(axis)
        if ref_axis is None:
            ref_axis = axis
        assert axis@ref_axis > np.cos(np.deg2rad(20))
        robot = s['robot_capture']
        pre, post = transform(robot['pre_base_from_tcp']), transform(robot['post_base_from_tcp'])
        np.testing.assert_array_equal(pre, s['base_from_tcp'])
        assert robot['post_ready'] is True
        assert robot['translation_delta_m'] <= .0005 and robot['rotation_delta_rad'] <= .005
        assert np.linalg.norm(pre[:3, 3]-post[:3, 3]) <= .0005
        assert Rotation.from_matrix(pre[:3, :3].T@post[:3, :3]).magnitude() <= .005
        assert s['read_started'] >= robot['not_before_monotonic'] >= robot['settled_at_monotonic']+.4-1e-9
        assert s['read_started'] <= s['read_completed'] <= robot['post_pose_monotonic']
        assert robot['pre_pose_monotonic'] <= robot['post_pose_monotonic']
        freshness.append(s['read_started']-robot['not_before_monotonic'])
        prepost.append(dict(index=i, translation_delta_m=robot['translation_delta_m'],
                            rotation_delta_rad=robot['rotation_delta_rad'],
                            processing_and_postcheck_s=robot['post_pose_monotonic']-s['read_started'],
                            acquisition_start_minus_pre_pose_s=s['read_started']-robot['pre_pose_monotonic']))
        sequences.append(s['sequence'])
        starts.append(s['read_started'])
        sessions.add((s['generation'], s['acquisition_session_id']))
    assert len(sessions) == 1
    assert all(b > a for a, b in zip(sequences, sequences[1:], strict=False))
    assert all(b > a for a, b in zip(starts, starts[1:], strict=False))
    fitted = solve_handeye(samples)
    x = transform(fitted['tcp_from_camera'])
    installed = transform(archive['tcp_from_camera'])
    np.testing.assert_allclose(x, installed, atol=1e-12)
    fresh_errors, fresh_reference = evaluate(samples, x)
    old = json.loads(OLD.read_text())
    assert old['source_id'] == archive['source_id'] and old['calibration_id'] == archive['calibration_id']
    old_x = transform(old['tcp_from_camera'])
    old_errors, old_reference = evaluate(samples, old_x)
    folds = []
    for fold in range(5):
        train = [s for i, s in enumerate(samples) if i % 5 != fold]
        heldout = [s for i, s in enumerate(samples) if i % 5 == fold]
        fit = solve_handeye(train)
        xf = transform(fit['tcp_from_camera'])
        _, reference = evaluate(train, xf)
        errors, _ = evaluate(heldout, xf, reference)
        folds.append(dict(fold=fold, training_views=20, heldout_views=5, result=fit,
                          heldout=errors, change_from_all25=difference(xf, x)))
    translations = np.concatenate([f['heldout']['per_view_translation_mm'] for f in folds])
    angles = np.concatenate([f['heldout']['per_view_rotation_deg'] for f in folds])
    result = dict(source_archive=str(ACTIVE.relative_to(ROOT)), source_archive_sha256=hashlib.sha256(raw).hexdigest(),
        archive_mtime_epoch=ACTIVE.stat().st_mtime, archive_schema=2, views=25,
        source_id=archive['source_id'], calibration_id=intrinsic_identity,
        intrinsic_file=str(MODEL.relative_to(ROOT)), intrinsic_file_sha256=hashlib.sha256(model_raw).hexdigest(),
        metadata_checks_passed=True, acquisition_sessions=[list(s) for s in sessions],
        timing_summary=dict(host_clock_domain=samples[0]['timing']['host_clock_domain'],
            camera_timestamp_semantics=samples[0]['timing']['camera_timestamp_semantics'],
            clock_mapping=samples[0]['timing']['clock_mapping'],
            motion_pairing_supported=samples[0]['timing']['motion_pairing_supported'],
            camera_frame_id_gap_total_max=max(s['timing']['camera_frame_id_gap_total'] for s in samples),
            duplicate_total_max=max(s['timing']['duplicate_total'] for s in samples),
            regression_total_max=max(s['timing']['regression_total'] for s in samples)),
        first_sequence=sequences[0], last_sequence=sequences[-1], first_read_started=starts[0], last_read_started=starts[-1],
        exposure_after_cutoff_min_s=min(freshness), exposure_after_cutoff_max_s=max(freshness),
        reversed_180_view_indices=reversals, prepost=prepost,
        fresh_result=fitted, fresh_stationary_board=fresh_errors, fresh_base_from_target_mean=fresh_reference.tolist(),
        old_candidate=str(OLD.relative_to(ROOT)), old_candidate_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(), change_from_old_candidate=difference(x, old_x),
        old_candidate_on_fresh_pairs=old_errors, old_candidate_fresh_base_from_target_mean=old_reference.tolist(),
        folds=folds, heldout_aggregate=dict(translation_rms_mm=float(np.sqrt(np.mean(translations**2))),
            translation_max_mm=float(translations.max()), rotation_rms_deg=float(np.sqrt(np.mean(angles**2))),
            rotation_max_deg=float(angles.max()), all_fold_gates_passed=all(f['heldout']['gates_passed'] for f in folds)),
        limitations=['Retained host acquisition timestamps and settled pre/post feedback are checked; no independent hardware exposure/CAN clock synchronization measurement.',
            'Fresh image/TCP pairs are independent of the original intrinsic run, but all views use the same fitted intrinsic model and robot kinematic model.',
            'This evaluates the recorded compact cone, not independent wider-workspace or dimensional accuracy.',
            'No hardware, service calls, active-file mutation, extra model or installation occurred.'])
    (OUT/'archive-snapshot.json').write_bytes(raw)
    (OUT/'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('metadata_checks_passed', 'reversed_180_view_indices', 'fresh_stationary_board', 'change_from_old_candidate', 'old_candidate_on_fresh_pairs', 'heldout_aggregate')}, indent=2))


if __name__ == '__main__':
    assess()
