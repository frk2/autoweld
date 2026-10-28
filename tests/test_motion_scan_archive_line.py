"""Measured line sampling is independent of stationary metadata and surface returns."""
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_motion_scan_archive import completion, frame, metadata, read_ply

from autoweld.motion_scan_archive import MotionScanArchive


def line_metadata(method='precision', length=80.):
    data = metadata()
    start = np.eye(4)
    start[:3, :3] = Rotation.from_euler('xyz', [10., 20., 30.], degrees=True).as_matrix()
    start[:3, 3] = [.2, -.3, .4]
    end = start.copy()
    end[2, 3] -= length / 1000.
    handeye = np.eye(4)
    handeye[:3, 3] = [.03, -.02, .06]
    data['handeye']['tcp_from_camera'] = handeye.tolist()
    digest = hashlib.sha256(json.dumps(handeye.tolist(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    data.update(scan_profile='vertical_down', scan_method=method,
                scan_protocol='precision_stationary' if method == 'precision' else 'fast_continuous',
                sweep_length_mm=length, direction_base=[0., 0., -1.],
                line_start_base_from_camera=start.tolist(), line_end_base_from_camera=end.tolist(),
                comparison_id='matched-line', comparison_binding=dict(
                    source_id='camera', optical_intrinsics_id='optics', calibration_id='laser',
                    sweep_length_mm=length, handeye_sha256=digest))
    return data


def observation(data, sequence, down_mm, *, phase='stationary_main', index=None, outcome='accepted'):
    cloud, reg = frame(sequence, [] if outcome == 'no_return' else None)
    midpoint = 10. + sequence * .01
    for key in ('exposure_host_mid_s', 'exposure_host_midpoint_s'):
        cloud['timing'][key] = midpoint
    cloud['timing']['exposure_host_start_s'] = midpoint - .0005
    cloud['timing']['exposure_host_end_s'] = midpoint + .0005
    reg['exposure_monotonic_s'] = midpoint
    reg['exposure_interval_s'] = [midpoint - .0005, midpoint + .0005]
    reg['robot_brackets_monotonic_s'] = np.tile([midpoint - .005, midpoint + .005], (6, 1))
    camera = np.asarray(data['line_start_base_from_camera']).copy()
    camera[2, 3] -= down_mm / 1000.
    handeye = np.asarray(data['handeye']['tcp_from_camera'])
    reg['base_from_tcp'] = camera @ np.linalg.inv(handeye)
    reg['points_world_m'] = np.asarray(cloud['xyz_m']).reshape(-1, 3) @ camera[:3, :3].T + camera[:3, 3]
    context = dict(acquisition_phase=phase)
    if phase != 'moving_main':
        if index is None:
            index = round(down_mm)
        requested = np.asarray(data['line_start_base_from_camera']).copy()
        requested[2, 3] -= index / 1000.
        requested = requested @ np.linalg.inv(handeye)
        context.update(station_index=index if phase == 'stationary_main' else None,
                       requested_base_from_tcp=requested.tolist(),
                       settle=dict(pre_hold_qualified_monotonic=midpoint - 1., hold_monotonic=midpoint - .9,
                           post_hold_stable_since=midpoint - .8, accepted_monotonic=midpoint - .1,
                           quiet_seconds=.4, requested_base_from_tcp=requested.tolist(),
                           measured_base_from_tcp=requested.tolist()),
                       stationary_feedback=dict(fresh=True, stationary=True, measured_at=midpoint + .001,
                                                observed_monotonic=midpoint + .002))
    return cloud, reg, context


def append(archive, data, sequence, down, *, outcome='accepted', **kwargs):
    cloud, reg, context = observation(data, sequence, down, outcome=outcome, **kwargs)
    return archive.append(cloud, reg, status=outcome, reason=None if outcome == 'accepted' else outcome,
                          acquisition_metadata=context)


def test_line_all_selected_native_and_world_exports_have_no_square_crop(tmp_path):
    archive = MotionScanArchive(tmp_path)
    data = line_metadata('fast')
    archive.begin(data)
    state = append(archive, data, 0, 0, phase='start_anchor')
    assert state['point_count'] == state['display_point_count'] == 2
    state = archive.finish('interrupted')
    assert state['roi_point_count'] == 2
    full, roi = read_ply(state['export_paths']['ply']), read_ply(state['export_paths']['roi_ply'])
    np.testing.assert_array_equal(full, roi)
    assert np.max(np.abs(full[:, :2])) > .1
    assert json.loads(open(state['export_paths']['roi_metadata']).read())['roi_mode'] == 'all_selected_line_points'


@pytest.mark.parametrize('tamper', ['endpoint', 'rotation', 'direction', 'handeye', 'length'])
def test_invalid_line_begin_does_not_change_old_geometry_or_create_new_attributes(tmp_path, tamper):
    archive = MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame())
    prior = archive.finish('interrupted')
    old = {key: value for key, value in archive.__dict__.items()}
    data = line_metadata()
    if tamper == 'endpoint':
        data['line_end_base_from_camera'][0][3] += .001
    elif tamper == 'rotation':
        data['line_end_base_from_camera'][:3] = Rotation.from_euler('z', .2).as_matrix().tolist()
    elif tamper == 'direction':
        data['direction_base'] = [0., 0., 1.]
    elif tamper == 'handeye':
        data['handeye']['tcp_from_camera'][0][3] += .001
    else:
        data['sweep_length_mm'] = 80.5
    with pytest.raises((ValueError, TypeError)):
        archive.begin(data)
    assert archive.state()['scan_id'] == prior['scan_id']
    assert archive.__dict__.keys() == old.keys()
    assert archive._line_profile is False
    np.testing.assert_array_equal(archive._handeye, old['_handeye'])


@pytest.mark.parametrize('tamper', ['missing_proof', 'pre_hold', 'not_quiet', 'old_feedback', 'old_frame',
                                  'station_index', 'actual_xy', 'actual_r', 'target'])
def test_stationary_metadata_alone_never_increments_or_persists_a_station(tmp_path, tamper):
    data = line_metadata()
    archive = MotionScanArchive(tmp_path)
    archive.begin(data)
    cloud, reg, context = observation(data, 0, 20.)
    if tamper == 'missing_proof':
        context.pop('stationary_feedback')
    elif tamper == 'pre_hold':
        context['settle']['accepted_monotonic'] = 10.
    elif tamper == 'not_quiet':
        context['stationary_feedback']['stationary'] = False
    elif tamper == 'old_feedback':
        context['stationary_feedback']['measured_at'] = 9.999
    elif tamper == 'old_frame':
        context['settle']['post_hold_stable_since'] = 9.8
    elif tamper == 'station_index':
        context['station_index'] = 21
    elif tamper == 'actual_xy':
        reg['base_from_tcp'][0, 3] += .0006
        reg['points_world_m'][:, 0] += .0006
    elif tamper == 'actual_r':
        pose = reg['base_from_tcp'] @ np.asarray(data['handeye']['tcp_from_camera'])
        pose[:3, :3] = Rotation.from_euler('z', .006).as_matrix() @ pose[:3, :3]
        reg['base_from_tcp'] = pose @ np.linalg.inv(data['handeye']['tcp_from_camera'])
        reg['points_world_m'] = np.asarray(cloud['xyz_m']) @ pose[:3, :3].T + pose[:3, 3]
    else:
        context['requested_base_from_tcp'][0][3] += .001
    with pytest.raises(ValueError):
        archive.append(cloud, reg, acquisition_metadata=context)
    state = archive.state()
    assert state['visited'] == state['stations_complete'] == 0
    assert not list(archive.session_directory.glob('frame-*.npz'))


def test_precision_81_stations_three_fresh_exposures_and_no_return_are_distinct(tmp_path):
    data = line_metadata()
    archive = MotionScanArchive(tmp_path)
    archive.begin(data)
    for index in range(81):
        for repeat in range(3):
            sequence = index * 3 + repeat
            outcome = 'no_return' if index == 40 else 'accepted'
            state = append(archive, data, sequence, index, outcome=outcome)
    assert state['stations_complete'] == state['station_total'] == 81
    assert state['station_capture_counts']['40'] == 3
    assert state['station_outcomes']['40'] == dict(accepted=0, no_return=3, rejected=0)
    assert state['line_start_observed'] and state['line_end_observed']
    finished = archive.finish(completion_evidence=dict(completion(), line_path_complete=True))
    restored = MotionScanArchive(tmp_path).state()
    assert restored['status'] == finished['status'] == 'complete'
    assert restored['station_capture_counts'] == state['station_capture_counts']
    assert restored['station_outcomes'] == state['station_outcomes']
    assert restored['line_exposure_bins_observed'] == 81


def test_rejected_and_repeated_or_wrong_station_exposures_cannot_complete_precision(tmp_path):
    data = line_metadata(length=1.)
    archive = MotionScanArchive(tmp_path)
    archive.begin(data)
    append(archive, data, 0, 0.)
    cloud, reg, context = observation(data, 0, 0.)
    with pytest.raises(ValueError, match='Duplicate'):
        archive.append(cloud, reg, acquisition_metadata=context)
    for sequence in (1, 2):
        append(archive, data, sequence, 0., outcome='rejected')
    for sequence in (3, 4, 5):
        state = append(archive, data, sequence, 1.)
    assert state['station_capture_counts']['0'] == 1
    assert state['station_outcomes']['0']['rejected'] == 2
    with pytest.raises(ValueError, match='three fresh'):
        archive.finish(completion_evidence=dict(completion(), line_path_complete=True))


def test_fast_bins_do_not_prove_endpoint_and_anchors_do_not_prove_main_sweep(tmp_path):
    data = line_metadata('fast')
    archive = MotionScanArchive(tmp_path)
    archive.begin(data)
    for sequence, down in enumerate((0., 40., 80.)):
        state = append(archive, data, sequence, down, phase='moving_main')
    assert state['line_samples_observed'] == 3 and state['moving_main_axial_span_mm'] == pytest.approx(80.)
    evidence = dict(completion(), line_path_complete=True)
    with pytest.raises(ValueError, match='stationary start and end'):
        archive.finish(completion_evidence=evidence)
    append(archive, data, 3, 0., phase='start_anchor')
    append(archive, data, 4, 80., phase='end_anchor')
    assert archive.finish(completion_evidence=evidence)['status'] == 'complete'

    archive = MotionScanArchive(tmp_path / 'anchors-only')
    archive.begin(data)
    append(archive, data, 0, 0., phase='start_anchor')
    append(archive, data, 1, 80., phase='end_anchor')
    with pytest.raises(ValueError, match='accepted moving'):
        archive.finish(completion_evidence=evidence)


def test_restore_rejects_fabricated_journal_station_and_complete_summary(tmp_path):
    data = line_metadata()
    archive = MotionScanArchive(tmp_path)
    archive.begin(data)
    append(archive, data, 0, 0.)
    archive.finish('interrupted')
    journal = archive.session_directory / 'frames.jsonl'
    record = json.loads(journal.read_text())
    record['scan_context']['station_index'] = 30
    journal.write_text(json.dumps(record) + '\n')
    restored = MotionScanArchive(tmp_path).state()
    assert restored['status'] == 'error' and 'native frame provenance' in restored['message']
    assert restored['stations_complete'] == 0
    journal.write_text(json.dumps({**record, 'scan_context': observation(data, 0, 0.)[2]}) + '\n')
    # The untouched raw journal, with a forged complete selector, is still not
    # enough to turn one captured station into81 complete stations.
    selector = archive.selector
    summary = json.loads(selector.read_text())
    summary.update(status='complete', completion_evidence=dict(completion(), line_path_complete=True))
    selector.write_text(json.dumps(summary))
    restored = MotionScanArchive(tmp_path).state()
    assert restored['status'] == 'error'
