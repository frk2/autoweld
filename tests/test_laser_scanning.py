"""Pure laser scan segmentation, optical geometry and durable capture gates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.laser_scanning import (
    ScanArchive,
    ScanSettings,
    scan_offsets,
    segment_object,
    tracking_target,
)


def cloud(count=60, depth=.140):
    x = np.linspace(-.015, .015, count)
    return {'coordinate_frame': 'camera', 'units': 'mm', 'xyz_m_units': 'm',
            'source_id': 'flir', 'calibration_id': 'laser-and-optical',
            'optical_intrinsics_id': 'optical', 'acquisition_session_id': 'session',
            'sequence': 5, 'generation': 2, 'read_started': 99.9, 'read_completed': 99.95,
            'image_size': [1000, 800], 'frame_product': {'kind': 'single_frame'},
            'xyz_m': np.column_stack((x, np.zeros(count), np.full(count, depth))).tolist(),
            'uv': np.column_stack((np.arange(count)+300, np.full(count, 250))).tolist(),
            'strength': np.full(count, 80.).tolist(), 'width_px': np.full(count, 3.).tolist(),
            'rejected': {'candidates': count, 'saturated': 0, 'ambiguous': 0}}


def feedback(**updates):
    return {'fresh': True, 'stationary': True, 'measured_at': 100., 'stable_since': 99., **updates}


def append(archive, c, **kwargs):
    return archive.append(c, np.eye(4), np.eye(4), segment_object(c), now=100.,
                          not_before_monotonic=99.8, feedback=feedback(), **kwargs)


def test_grid_has_one_mm_adjacency_fixed_plane_and_bounded_capture_count():
    points = np.asarray(scan_offsets(ScanSettings(width_mm=4, height_mm=2)))
    assert points.shape == (15, 2)
    np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), .001)
    np.testing.assert_allclose(points.min(axis=0), [-.002, -.001])
    np.testing.assert_allclose(points.max(axis=0), [.002, .001])
    assert len(set(map(tuple, points))) == len(points)
    assert scan_offsets(ScanSettings(width_mm=0, height_mm=0)) == [[0., 0.]]


@pytest.mark.parametrize('settings', [
    ScanSettings(width_mm=-1), ScanSettings(width_mm=101), ScanSettings(width_mm=100, height_mm=101),
    ScanSettings(width_mm=1.5), ScanSettings(step_mm=2), ScanSettings(standoff_mm=150),
    ScanSettings(max_depth_step_mm=6), ScanSettings(width_mm=True), ScanSettings(height_mm=np.nan),
    ScanSettings(max_points=300001),
])
def test_settings_reject_invalid_or_nonuniform_envelopes(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_segment_retains_native_measurements_and_excludes_disconnected_background():
    c = cloud(90)
    other = cloud(25, .2)
    for key in ('xyz_m', 'strength', 'width_px'):
        c[key] += other[key]
    c['uv'] += (np.asarray(other['uv'])+[300, 0]).tolist()
    c['rejected']['candidates'] = 115
    result = segment_object(c)
    assert result['quality']['good'] and result['quality']['count'] == 90
    assert result['quality']['components'] == 2
    np.testing.assert_array_equal(result['xyz_m'], c['xyz_m'][:90])
    assert result['indices'] == list(range(90))
    np.testing.assert_allclose(result['centroid_camera_m'], [0., 0., .14], atol=1e-15)


def test_competing_surfaces_require_unambiguous_previous_object_association():
    c, other = cloud(60), cloud(60, .17)
    for key in ('xyz_m', 'strength', 'width_px'):
        c[key] += other[key]
    c['uv'] += (np.asarray(other['uv'])+[300, 0]).tolist()
    c['rejected']['candidates'] = 120
    with pytest.raises(ValueError, match='ambiguous at scan start'):
        segment_object(c)
    assert segment_object(c, [0, 0, .14])['indices'] == list(range(60))
    with pytest.raises(ValueError, match='association is ambiguous'):
        segment_object(c, [0, 0, .155])
    with pytest.raises(ValueError, match='lost'):
        segment_object(c, [0, 0, .5])


@pytest.mark.parametrize('reason', ['few', 'weak', 'saturated', 'multiframe', 'nonfinite', 'wrong_units',
                                     'broad', 'no_frame_product', 'off_image'])
def test_bad_stripes_never_authorize_scanning(reason):
    c = cloud(10 if reason == 'few' else 60)
    if reason == 'weak':
        c['strength'] = [1.]*60
    elif reason == 'saturated':
        c['rejected'] = {'saturated': 90, 'candidates': 120}
    elif reason == 'multiframe':
        c['frame_product']['kind'] = 'background_subtracted'
    elif reason == 'nonfinite':
        c['xyz_m'][0][0] = float('nan')
    elif reason == 'wrong_units':
        del c['xyz_m_units']
    elif reason == 'broad':
        c['width_px'] = [45.]*60
    elif reason == 'no_frame_product':
        del c['frame_product']
    elif reason == 'off_image':
        c['uv'][0] = [-1., 0.]
    with pytest.raises(ValueError):
        segment_object(c)


def test_target_uses_measured_optical_frame_fixed_nominal_orientation_and_five_mm_depth_cap():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [15., 30., -10.], degrees=True).as_matrix()
    pose[:3, 3] = [.2, -.1, .4]
    optical = np.eye(4)
    optical[:3, :3] = Rotation.from_euler('xyz', [10., -20., 25.], degrees=True).as_matrix()
    optical[:3, 3] = [.01, .02, .03]
    segment = segment_object(cloud(depth=.180))
    target = tracking_target(pose, optical, segment, [.001, 0.])
    np.testing.assert_allclose(target[:3, :3], pose[:3, :3])
    np.testing.assert_allclose(target[:3, 3]-pose[:3, 3], (pose @ optical)[:3, :3] @ [.001, 0., .005])
    assert np.linalg.norm(target[:3, 3]-pose[:3, 3]) < .0051
    np.testing.assert_array_equal(pose[:3, 3], [.2, -.1, .4])
    with pytest.raises(ValueError, match='at most 1 mm'):
        tracking_target(pose, optical, segment, [.002, 0.])


def test_archive_persists_measured_geometry_raw_snapshots_provenance_and_ply(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin({'saved_pose': 'Tube Scanning'})
    c = cloud()
    pose, optical = np.eye(4), np.eye(4)
    pose[:3, 3] = [.3, .2, .1]
    optical[:3, 3] = [.01, 0, 0]
    record = archive.append(c, pose, optical, segment_object(c), now=100.,
                            not_before_monotonic=99.8, feedback=feedback())
    result = archive.finish()
    world = np.asarray(c['xyz_m'])+[.31, .2, .1]
    np.testing.assert_allclose(archive.state(True)['points_base_m'], world)
    assert result['captures'] == 1 and result['count'] == 60
    assert result['scan_id'] and result['cloud_revision'] == 1
    with np.load(record['raw_snapshot_path'], allow_pickle=False) as native:
        np.testing.assert_array_equal(native['xyz_camera_m'], c['xyz_m'])
        np.testing.assert_array_equal(native['uv'], c['uv'])
        np.testing.assert_array_equal(native['base_from_tcp'], pose)
    assert 'element vertex 60' in Path(result['export_paths']['ply']).read_text()
    saved = json.loads(Path(result['metadata_path']).read_text())
    assert saved['captures'][0]['feedback']['stationary'] is True
    np.testing.assert_allclose(ScanArchive(tmp_path).state(True)['points_world_m'], world)


@pytest.mark.parametrize('reason', ['stale', 'before_quiet', 'moving', 'old_sdk', 'before_acquisition',
                                     'no_sdk', 'identity', 'duplicate', 'depth'])
def test_archive_rejects_bad_associations_without_losing_saved_partial(tmp_path, reason):
    archive = ScanArchive(tmp_path)
    archive.begin()
    append(archive, cloud())
    c, fb = cloud(), feedback()
    c['sequence'] = 6
    if reason == 'stale':
        c['read_started'] = 98.
    elif reason == 'before_quiet':
        fb['stable_since'] = 99.95
    elif reason == 'moving':
        fb['stationary'] = False
    elif reason == 'old_sdk':
        fb['measured_at'] = 99.7
    elif reason == 'before_acquisition':
        fb['measured_at'] = 99.85
    elif reason == 'no_sdk':
        fb = None
    elif reason == 'identity':
        c['optical_intrinsics_id'] = 'different'
    elif reason == 'duplicate':
        c['sequence'] = 5
    elif reason == 'depth':
        c['xyz_m'] = (np.asarray(c['xyz_m'])+[0, 0, .002]).tolist()
    with pytest.raises(ValueError):
        archive.append(c, np.eye(4), np.eye(4), segment_object(c), now=100.,
                       not_before_monotonic=99.8, feedback=fb)
    assert archive.state()['count'] == 60
    assert ScanArchive(tmp_path).state()['count'] == 60
    assert ScanArchive(tmp_path).state()['status'] == 'interrupted'


def test_archive_preview_budget_never_discards_accepted_raw_points(tmp_path):
    archive = ScanArchive(tmp_path, max_points=1000)
    archive.begin()
    c = cloud(600)
    append(archive, c)
    next_cloud = copy.deepcopy(c)
    next_cloud['sequence'] += 1
    append(archive, next_cloud)
    assert archive.state()['count'] == 1200
    assert archive.state()['preview_count'] == 1000
    assert len(list((tmp_path/'laser-scans'/archive.state()['scan_id']).glob('*.npz'))) == 2
    result = archive.finish()
    assert 'element vertex 1200' in Path(result['export_paths']['ply']).read_text()
    archive.clear()
    assert archive.state()['count'] == 0 and ScanArchive(tmp_path).state()['count'] == 0


def test_new_scan_preserves_completed_session_points_and_per_view_manifest(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin({'saved_pose': 'Tube Scanning'})
    append(archive, cloud())
    state = archive.finish()
    old_paths = state['export_paths']
    archive.begin()
    append(archive, cloud())
    assert archive.state()['scan_id'] != state['scan_id']
    for key in ('manifest', 'points', 'ply'):
        assert Path(old_paths[key]).exists()
    manifest = json.loads(Path(old_paths['manifest']).read_text())
    assert len(manifest['captures']) == 1
    assert manifest['captures'][0]['base_from_tcp'] == np.eye(4).tolist()
    assert Path(manifest['captures'][0]['raw_snapshot_path']).exists()


def raster_metadata(width=2, height=0):
    from dataclasses import asdict
    return {'settings': asdict(ScanSettings(width_mm=width, height_mm=height)),
            'baseline_pose': np.eye(4).tolist(),
            'handeye': {'tcp_from_camera': np.eye(4).tolist()},
            'identity': ['flir', 'optical', 'laser-and-optical', 'session'],
            'model_fingerprint': 'installed-model'}


def cell_pose(offset):
    pose = np.eye(4)
    pose[:2, 3] = offset
    return pose


def test_full_100_mm_square_has_10201_unique_one_mm_grid_points():
    settings = ScanSettings(width_mm=100, height_mm=100).validate()
    offsets = np.asarray(scan_offsets(settings))
    assert offsets.shape == (10201, 2)
    assert len(set(map(tuple, offsets))) == 10201
    np.testing.assert_allclose(np.linalg.norm(np.diff(offsets, axis=0), axis=1), .001, atol=1e-15)
    np.testing.assert_allclose(offsets.min(0), [-.05, -.05])
    np.testing.assert_allclose(offsets.max(0), [.05, .05])
    assert ScanSettings().width_mm == ScanSettings().height_mm == 10


def test_empty_and_rejected_cells_record_fresh_native_coverage_without_adding_points(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin(raster_metadata())
    empty = cloud(0)
    blank = archive.record_cell('no_return', empty, cell_pose([-.001, 0]), np.eye(4), now=100.,
        not_before_monotonic=99.8, feedback=feedback(), cell_index=0, offset_xy_m=[-.001, 0],
        reason='No native stripe returns at settled grid point')
    c = cloud()
    c['sequence'] += 1
    archive.append(c, np.eye(4), np.eye(4), segment_object(c), now=100.,
        not_before_monotonic=99.8, feedback=feedback(), cell_index=1, offset_xy_m=[0, 0])
    bad = cloud()
    bad['sequence'] += 2
    bad['strength'] = [1.]*60
    rejected = archive.record_cell('rejected', bad, cell_pose([.001, 0]), np.eye(4), now=100.,
        not_before_monotonic=99.8, feedback=feedback(), cell_index=2, offset_xy_m=[.001, 0],
        reason='Weak stripe rejected by native quality gate')
    state = archive.finish()
    assert state['captures'] == 1 and state['points'] == 60
    assert state['cells_visited'] == 3 and state['cells_no_return'] == state['cells_rejected'] == 1
    assert state['next_index'] == state['grid_total'] == 3
    with np.load(blank['raw_snapshot_path']) as raw:
        assert raw['xyz_camera_m'].shape == (0, 3) and raw['xyz_world_m'].shape == (0, 3)
    with np.load(rejected['raw_snapshot_path']) as raw:
        assert raw['xyz_camera_m'].shape == (60, 3) and raw['selected_indices'].size == 0
    manifest = json.loads(Path(state['export_paths']['manifest']).read_text())
    assert [cell['status'] for cell in manifest['cells']] == ['no_return', 'captured', 'rejected']
    assert manifest['cells'][0]['feedback']['stationary'] is True
    assert len(manifest['captures']) == 1
    assert ScanArchive(tmp_path).state()['cells_visited'] == 3


@pytest.mark.parametrize('failure', ['stale', 'moving', 'wrong_grid', 'wrong_pose', 'duplicate', 'handeye'])
def test_empty_coverage_requires_fresh_measured_unique_grid_cell(tmp_path, failure):
    archive = ScanArchive(tmp_path)
    archive.begin(raster_metadata())
    c, fb = cloud(0), feedback()
    offset, pose, correction = [-.001, 0], cell_pose([-.001, 0]), np.eye(4)
    if failure == 'stale':
        c['read_started'] = 98.
    elif failure == 'moving':
        fb['stationary'] = False
    elif failure == 'wrong_grid':
        offset = [0, 0]
    elif failure == 'wrong_pose':
        pose[0, 3] = .001
    elif failure == 'handeye':
        correction[0, 3] = .001
    else:
        archive.record_cell('no_return', c, pose, correction, now=100.,
            not_before_monotonic=99.8, feedback=fb, cell_index=0, offset_xy_m=offset)
    with pytest.raises(ValueError):
        archive.record_cell('no_return', c, pose, correction, now=100.,
            not_before_monotonic=99.8, feedback=fb, cell_index=0, offset_xy_m=offset)
    assert archive.state()['cells_visited'] == (1 if failure == 'duplicate' else 0)


def test_large_archive_retains_full_native_world_data_and_checkpoints_only_bounded_preview(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin()
    first_checkpoint = archive.state()['export_paths']['points']
    for index in range(40):
        c = cloud(600)
        c['sequence'] = index
        pose = cell_pose([index*.001, 0])
        archive.append(c, pose, np.eye(4), segment_object(c), now=100.,
                       not_before_monotonic=99.8, feedback=feedback(), cell_index=index)
        if index < 31:
            assert archive.state()['export_paths']['points'] == first_checkpoint
    active = archive.state(True)
    assert active['points'] == 24000 and active['captures'] == active['cells_visited'] == 40
    assert len(active['points_base_m']) <= 10000
    assert 'captures' not in json.loads((tmp_path/'laser-scans/latest.json').read_text())
    journal = Path(active['export_paths']['journal'])
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(records) == 40 and sum(record['count'] for record in records) == 24000
    assert len({record['raw_snapshot_path'] for record in records}) == 40
    restored = ScanArchive(tmp_path)
    assert restored.state()['points'] == 24000 and restored.state()['next_index'] == 40
    result = restored.finish('cancelled')
    lines = Path(result['export_paths']['ply']).read_text().splitlines()
    assert 'element vertex 24000' in lines
    assert len(lines[lines.index('end_header')+1:]) == 24000
    manifest = json.loads(Path(result['export_paths']['manifest']).read_text())
    assert len(manifest['captures']) == 40 and len(manifest['cells']) == 40
    assert ScanArchive(tmp_path).state(True)['preview_count'] == 10000
    assert len(json.dumps(restored.state(False))) < 10000


def test_resume_keeps_coverage_and_requires_matching_calibration_and_baseline(tmp_path):
    archive = ScanArchive(tmp_path)
    metadata = raster_metadata()
    archive.begin(metadata)
    c = cloud()
    archive.append(c, cell_pose([-.001, 0]), np.eye(4), segment_object(c), now=100.,
        not_before_monotonic=99.8, feedback=feedback(), cell_index=0, offset_xy_m=[-.001, 0])
    first_session = archive.state()['scan_id']
    archive.finish('cancelled')
    restored = ScanArchive(tmp_path)
    with pytest.raises(ValueError, match='metadata changed'):
        restored.resume({'model_fingerprint': 'another-model'})
    changed_baseline = np.eye(4)
    changed_baseline[0, 3] = .01
    with pytest.raises(ValueError, match='metadata changed'):
        restored.resume({'baseline_pose': changed_baseline.tolist()})
    result = restored.resume(metadata, acquisition_session_id='new-camera-session')
    assert result['scan_id'] == first_session and result['next_index'] == 1
    c['acquisition_session_id'] = 'new-camera-session'
    archive_record = restored.append(c, np.eye(4), np.eye(4), segment_object(c), now=100.,
        not_before_monotonic=99.8, feedback=feedback(), cell_index=1, offset_xy_m=[0, 0])
    assert archive_record['sequence'] == 5  # Same sequence in a new explicit acquisition session is distinct.
    assert restored.state()['points'] == 120 and restored.state()['next_index'] == 2
    np.testing.assert_array_equal(restored.state()['last_cell']['base_from_tcp'], np.eye(4))
    with pytest.raises(ValueError, match='every configured grid cell'):
        restored.finish('complete')


def test_restart_replays_durable_view_if_summary_write_failed(tmp_path, monkeypatch):
    archive = ScanArchive(tmp_path)
    archive.begin()
    monkeypatch.setattr(archive, '_save', lambda: (_ for _ in ()).throw(OSError('summary disk error')))
    with pytest.raises(OSError):
        append(archive, cloud())
    restored = ScanArchive(tmp_path)
    assert restored.state()['points'] == 60 and restored.state()['next_index'] == 1
    restored.resume()
    other = cloud()
    other['sequence'] = 6
    other['optical_intrinsics_id'] = 'different-optics'
    with pytest.raises(ValueError, match='identity changed'):
        append(restored, other)


def test_torn_last_journal_line_does_not_create_coverage_and_resume_truncates_it(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin()
    append(archive, cloud())
    journal = Path(archive.state()['export_paths']['journal'])
    with journal.open('ab') as handle:
        handle.write(b'{"incomplete":')
    restored = ScanArchive(tmp_path)
    assert restored.state()['next_index'] == 1
    restored.resume()
    other = cloud()
    other['sequence'] = 6
    append(restored, other)
    assert len([json.loads(line) for line in journal.read_text().splitlines()]) == 2
    assert ScanArchive(tmp_path).state()['points'] == 120


def test_legacy_archive_restores_and_can_migrate_without_losing_measurements(tmp_path):
    directory = tmp_path/'laser-scans'
    raw_directory = directory/'legacy-session'
    raw_directory.mkdir(parents=True)
    c = cloud()
    native = np.asarray(c['xyz_m'])
    np.savez_compressed(directory/'legacy-1.npz', points_world_m=native)
    raw = raw_directory/'0000.npz'
    np.savez_compressed(raw, xyz_camera_m=native, uv=np.asarray(c['uv']),
                        strength=np.asarray(c['strength']), width_px=np.asarray(c['width_px']),
                        selected_indices=np.arange(60), base_from_tcp=np.eye(4), tcp_from_camera=np.eye(4))
    record = {key: copy.deepcopy(c[key]) for key in ('source_id', 'optical_intrinsics_id',
              'calibration_id', 'acquisition_session_id', 'sequence', 'generation', 'read_started',
              'read_completed', 'frame_product')}
    record.update(acquisition_identity=['flir', 'session', 2, 5], base_from_tcp=np.eye(4).tolist(),
                  tcp_from_camera=np.eye(4).tolist(), feedback=feedback(), count=60,
                  raw_snapshot_path=str(raw), quality=segment_object(c)['quality'], point_offset=0)
    (directory/'latest.json').write_text(json.dumps({'status': 'cancelled', 'session_id': 'legacy-session',
        'count': 60, 'points_file': 'legacy-1.npz', 'captures': [record],
        'identity': ['flir', 'optical', 'laser-and-optical', 'session'], 'metadata': {}}))
    restored = ScanArchive(tmp_path)
    np.testing.assert_allclose(restored.state(True)['points_base_m'], native)
    restored.resume()
    c['sequence'] = 6
    append(restored, c)
    final = restored.finish()
    assert final['points'] == 120 and final['captures'] == 2 and raw.exists()
    assert len(json.loads(Path(final['export_paths']['manifest']).read_text())['captures']) == 2
    np.testing.assert_allclose(ScanArchive(tmp_path).state(True)['points_base_m'], np.r_[native, native])


def test_preview_state_is_detached_from_internal_reservoir(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin()
    append(archive, cloud())
    preview = archive.state(True)['points_base_m']
    preview[0][0] = 999.
    assert archive.state(True)['points_base_m'][0][0] != 999.


def test_missing_committed_raw_view_blocks_resume_instead_of_losing_coverage(tmp_path):
    archive = ScanArchive(tmp_path)
    archive.begin()
    record = append(archive, cloud())
    Path(record['raw_snapshot_path']).unlink()
    restored = ScanArchive(tmp_path)
    assert restored.state()['status'] == 'error'
    with pytest.raises(ValueError, match='requires repair'):
        restored.resume()
