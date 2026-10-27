"""Native-pair association and audit persistence; no camera or robot access."""

import copy
import json
from concurrent.futures import Future

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.calibration import CalibrationCoordinator, cone_plan, solve_handeye
from autoweld.handeye_observation import HandeyeAssociation


def observation(sequence=1, *, reverse=False, target=None):
    points = np.mgrid[:5, :7].T.reshape(-1, 2)*20.+[100, 100]
    t = np.eye(4) if target is None else target.copy()
    if target is None:
        t[2, 3] = .15
    board180 = np.diag([-1., -1., 1., 1.])
    board180[:3, 3] = [.016, .024, 0]
    if reverse:
        points = points[::-1].copy()
        t = t @ board180
    t[:3, 3] *= 1000
    return dict(camera_from_target=t.tolist(), corners_native_px=points.tolist(),
                grid=dict(columns=5, rows=7, square_mm=4.), image_size=[640, 480],
                source_id='fixture:camera', calibration_id='intrinsic:one', generation=1,
                acquisition_session_id='acquisition:one', sequence=sequence, read_started=float(sequence),
                read_completed=sequence+.1, timing={'exposure_us': 1000}, native_validation=True,
                units='mm', coordinate_frame='camera', reprojection_rms_px=.4, controls={'gain_db': 0.})


def test_reversal_association_recovers_known_handeye_before_any_fit():
    known = np.eye(4)
    known[:3, :3] = Rotation.from_euler('xyz', [.05, -.04, .03]).as_matrix()
    known[:3, 3] = [.01, -.02, .03]
    board = np.eye(4)
    board[:3, 3] = [.02, .01, .4]
    poses, _ = cone_plan(np.eye(4))
    association, samples = HandeyeAssociation(), []
    for i, g in enumerate(poses):
        target = np.linalg.inv(known) @ np.linalg.inv(g) @ board
        obs = observation(i+1, reverse=i in (3, 4, 5), target=target)
        original = copy.deepcopy(obs)
        sample = association.prepare(obs, g)
        assert obs == original
        np.testing.assert_allclose(sample['camera_from_target_m'], target, atol=1e-14)
        assert sample['normalization']['reversed_180'] is (i in (3, 4, 5))
        association.commit(sample)
        samples.append(sample)
    result = solve_handeye(samples)
    np.testing.assert_allclose(result['tcp_from_camera'], known, atol=1e-12)
    assert result['translation_rms_mm'] < 1e-8
    assert samples[3]['raw_corners_native_px'][0] == [180., 220.]
    assert samples[3]['corners_native_px'][0] == [100., 100.]


def test_unaccepted_prepare_does_not_anchor_or_reserve_frame():
    a = HandeyeAssociation()
    first = a.prepare(observation(), np.eye(4))
    assert a.reference is None and a.accepted == []
    assert a.prepare(observation(), np.eye(4)) == first
    a.commit(first)
    with pytest.raises(ValueError, match='Stale'):
        a.commit(first)
    with pytest.raises(ValueError, match='Duplicate'):
        a.prepare(observation(), np.eye(4))


@pytest.mark.parametrize('change', [
    lambda o: o.pop('corners_native_px'),
    lambda o: o.update(corners_native_px=o['corners_native_px'][:-1]),
    lambda o: o['corners_native_px'][0].__setitem__(0, float('nan')),
    lambda o: o['corners_native_px'][0].__setitem__(0, -1),
    lambda o: o['corners_native_px'].__setitem__(0, o['corners_native_px'][1]),
    lambda o: o.update(corners_native_px=[[i, i] for i in range(35)]),
    lambda o: o.update(native_validation=False),
    lambda o: o.update(units='m'),
    lambda o: o.update(timing=None),
    lambda o: o.pop('acquisition_session_id'),
    lambda o: o.update(read_started=float('inf')),
    lambda o: o.update(read_completed=0.),
    lambda o: o['grid'].update(square_mm=0),
    lambda o: o['grid'].update(columns=5.0),
])
def test_invalid_native_geometry_and_provenance_rejected(change):
    o = observation()
    change(o)
    with pytest.raises(ValueError):
        HandeyeAssociation().prepare(o, np.eye(4))


@pytest.mark.parametrize('key,value', [('source_id', 'other'), ('calibration_id', 'other'),
    ('generation', 2), ('acquisition_session_id', 'other'), ('image_size', [800, 600]),
    ('grid', dict(columns=5, rows=7, square_mm=5.))])
def test_configuration_identity_is_fixed_after_first_accepted_pair(key, value):
    a = HandeyeAssociation()
    a.commit(a.prepare(observation(), np.eye(4)))
    o = observation(2)
    o[key] = value
    with pytest.raises(ValueError, match='configuration changed'):
        a.prepare(o, np.eye(4))


def test_ambiguous_grid_axis_and_large_physical_rotation_are_rejected():
    a = HandeyeAssociation()
    a.commit(a.prepare(observation(), np.eye(4)))
    o = observation(2)
    p = np.asarray(o['corners_native_px'])
    o['corners_native_px'] = (p @ Rotation.from_euler('z', 45, degrees=True).as_matrix()[:2, :2].T+[200, 0]).tolist()
    with pytest.raises(ValueError, match='Ambiguous'):
        a.prepare(o, np.eye(4))
    rotated = np.eye(4)
    rotated[:3, :3] = Rotation.from_euler('z', 21, degrees=True).as_matrix()
    with pytest.raises(ValueError, match='large physical rotation'):
        a.prepare(observation(2, reverse=True), rotated)


class Camera:
    def __init__(self):
        self.observation = observation()
    def state(self, **kwargs):
        return dict(fresh=True, board={'found': True, 'settings': self.observation['grid']},
                    intrinsics={}, source_id=self.observation['source_id'], generation=1,
                    calibration_id='intrinsic:one', read_started=1.)
    def target_pose(self, **kwargs):
        return self.observation


def coordinator(tmp_path, camera=None, tcp=None):
    return CalibrationCoordinator(camera or Camera(), tcp or (lambda: np.eye(4)),
                                  lambda *a, **k: None, lambda: True, lambda: None, tmp_path)


def test_coordinator_only_commits_after_post_capture_check_and_archives_atomically(tmp_path):
    c = coordinator(tmp_path)
    try:
        c.kind = 'handeye'
        c.capture_pose = np.eye(4)
        c.capture_pose_monotonic = .1
        c.settled_at, c.capture_not_before = .2, .6
        result = c._capture()
        assert not c.handeye_association.accepted
        c.future, c.pending_kind = Future(), 'capture_manual'
        c.future.set_result(result)
        c.tick()
        assert c.status == 'idle'
        assert len(c.samples) == len(c.handeye_association.accepted) == 1
        assert c.samples[0]['robot_capture']['post_ready']
        c._save_handeye({'tcp_from_camera': np.eye(4).tolist(), 'views': 1})
        saved = json.loads((tmp_path/'handeye.json').read_text())
        assert saved['schema_version'] == 2
        assert saved['paired_samples'] == c.samples
        before = (tmp_path/'handeye.json').read_bytes()
        c.samples[0]['timing']['bad'] = float('nan')
        with pytest.raises(ValueError):
            c._save_handeye({})
        assert (tmp_path/'handeye.json').read_bytes() == before
        c.dispatch({'action': 'calibration_reset', 'kind': 'handeye'})
        assert c.handeye_association.reference is None and c.samples == []
    finally:
        c.close()


def test_robot_movement_rejects_pair_without_committing_reference(tmp_path):
    current = np.eye(4)
    c = coordinator(tmp_path, tcp=lambda: current)
    try:
        c.kind = 'handeye'
        c.capture_pose = np.eye(4)
        c.capture_not_before = .5
        result = c._capture()
        c.future, c.pending_kind = Future(), 'capture_manual'
        c.future.set_result(result)
        current[0, 3] = .001
        c.tick()
        assert c.status == 'error'
        assert c.samples == [] and c.handeye_association.reference is None
    finally:
        c.close()


@pytest.mark.parametrize('mismatch', ['grid', 'calibration_id', 'generation'])
def test_coordinator_checks_live_state_matches_native_target(tmp_path, mismatch):
    camera = Camera()
    original = camera.state()
    if mismatch == 'grid':
        original['board']['settings'] = {**original['board']['settings'], 'square_mm': 9.}
    else:
        original[mismatch] = 'other'
    camera.state = lambda **kwargs: original
    c = coordinator(tmp_path, camera)
    try:
        c.kind, c.capture_pose, c.capture_not_before = 'handeye', np.eye(4), .5
        with pytest.raises(ValueError, match='configuration|changed'):
            c._capture()
    finally:
        c.close()


def test_new_start_resets_association_and_does_not_reset_intrinsics(tmp_path):
    c = coordinator(tmp_path)
    try:
        c.handeye_association.commit(c.handeye_association.prepare(observation(), np.eye(4)))
        c.samples = [{'old': True}]
        phases = []
        c._begin_hunt = phases.append
        c.dispatch({'action': 'calibration_start', 'kind': 'handeye'})
        assert phases == ['baseline']
        assert c.handeye_association.reference is None and c.samples == []
    finally:
        c.close()


def test_acquisition_timestamp_must_advance_even_with_new_frame_sequence():
    a = HandeyeAssociation()
    a.commit(a.prepare(observation(), np.eye(4)))
    o = observation(2)
    o['read_started'] = 1.
    with pytest.raises(ValueError, match='out-of-order'):
        a.prepare(o, np.eye(4))


def test_failed_archive_flush_preserves_previous_result(tmp_path, monkeypatch):
    import autoweld.calibration as calibration
    c = coordinator(tmp_path)
    try:
        target = tmp_path/'handeye.json'
        target.write_text('{"previous":true}')
        def fail(_):
            raise OSError('disk flush failed')
        monkeypatch.setattr(calibration.os, 'fsync', fail)
        with pytest.raises(OSError, match='disk flush'):
            c._save_handeye({'views': 20})
        assert json.loads(target.read_text()) == {'previous': True}
        assert not hasattr(c.camera, '_handeye_result')
    finally:
        c.close()


def test_saved_result_validation_precedes_archive_replacement(tmp_path):
    camera = Camera()
    calls = []
    def validate(result):
        calls.append('validate')
        raise ValueError('intrinsic identity changed')
    camera.validate_handeye = validate
    camera.install_handeye = lambda result: calls.append('install')
    c = coordinator(tmp_path, camera)
    try:
        target = tmp_path/'handeye.json'
        target.write_text('{"previous":true}')
        with pytest.raises(ValueError, match='identity changed'):
            c._save_handeye({'views': 25})
        assert calls == ['validate']
        assert json.loads(target.read_text()) == {'previous': True}
    finally:
        c.close()
