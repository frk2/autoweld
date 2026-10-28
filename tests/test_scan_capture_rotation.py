"""Optional fixed optical roll changes capture orientation, never Tube grid axes."""
import json
from dataclasses import asdict

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.laser_scanning import ScanArchive, ScanSettings, segment_object


def cloud():
    return {'coordinate_frame': 'camera', 'xyz_m_units': 'm', 'source_id': 'flir',
        'optical_intrinsics_id': 'optics', 'calibration_id': 'laser', 'acquisition_session_id': 'camera',
        'generation': 1, 'sequence': 1, 'read_started': 99.9, 'read_completed': 99.95,
        'image_size': [1000, 800], 'frame_product': {'kind': 'single_frame'},
        'xyz_m': np.column_stack((np.linspace(-.01, .01, 30), np.zeros(30), np.full(30, .14))).tolist(),
        'uv': np.column_stack((np.arange(30)+300, np.full(30, 200))).tolist(),
        'strength': [80.]*30, 'width_px': [3.]*30}


def fixture():
    baseline = np.eye(4)
    baseline[:3, :3] = Rotation.from_euler('xyz', [15., -10., 30.], degrees=True).as_matrix()
    baseline[:3, 3] = [.2, -.1, .3]
    handeye = np.eye(4)
    handeye[:3, :3] = Rotation.from_euler('xyz', [10., 5., -20.], degrees=True).as_matrix()
    handeye[:3, 3] = [.02, .01, -.01]
    optical = baseline @ handeye
    roll = np.eye(4)
    roll[:3, :3] = Rotation.from_euler('z', 90., degrees=True).as_matrix()
    view = optical @ roll @ np.linalg.inv(handeye)
    metadata = {'settings': asdict(ScanSettings(width_mm=2, height_mm=0)),
        'baseline_pose': baseline.tolist(), 'view_pose': view.tolist(),
        'capture_rotation': view[:3, :3].tolist(), 'handeye': {'tcp_from_camera': handeye.tolist()}}
    return baseline, handeye, view, metadata


def append(archive, pose, handeye, *, offset=(-.001, 0.), cell_index=0):
    c = cloud()
    return archive.append(c, pose, handeye, segment_object(c), now=100.,
        not_before_monotonic=99.8, feedback={'fresh': True, 'stationary': True,
        'stable_since': 99., 'measured_at': 100.}, offset_xy_m=offset, cell_index=cell_index)


def test_fixed_roll_accepts_actual_pose_in_original_tube_grid_and_preserves_world_geometry(tmp_path):
    baseline, handeye, view, metadata = fixture()
    original_optical = baseline @ handeye
    rolled_optical = view @ handeye
    np.testing.assert_allclose(original_optical[:3, 3], rolled_optical[:3, 3])
    # The chosen pose translates on original Tube optical X, irrespective of roll.
    actual = view.copy()
    actual[:3, 3] += original_optical[:3, :3] @ [-.001, 0., 0.]
    archive = ScanArchive(tmp_path)
    archive.begin(metadata)
    record = append(archive, actual, handeye)
    world_expected = np.asarray(cloud()['xyz_m']) @ (actual @ handeye)[:3, :3].T+(actual @ handeye)[:3, 3]
    np.testing.assert_allclose(archive.state(True)['points_base_m'], world_expected)
    np.testing.assert_allclose(record['base_from_tcp'], actual)
    result = archive.finish('cancelled')
    manifest = json.loads(open(result['export_paths']['manifest']).read())
    assert manifest['metadata']['capture_rotation'] == metadata['capture_rotation']
    np.testing.assert_allclose(manifest['metadata']['baseline_pose'], baseline)


def test_grid_does_not_rotate_with_selected_camera_roll(tmp_path):
    baseline, handeye, view, metadata = fixture()
    wrong = view.copy()
    wrong[:3, 3] += (view @ handeye)[:3, :3] @ [-.001, 0., 0.]
    archive = ScanArchive(tmp_path)
    archive.begin(metadata)
    with pytest.raises(ValueError, match='measured grid endpoint'):
        append(archive, wrong, handeye)
    assert archive.state()['captures'] == 0


def test_absent_override_keeps_existing_baseline_orientation_gate(tmp_path):
    baseline, handeye, view, metadata = fixture()
    metadata.pop('capture_rotation')
    view[:3, 3] += (baseline @ handeye)[:3, :3] @ [-.001, 0., 0.]
    archive = ScanArchive(tmp_path)
    archive.begin(metadata)
    with pytest.raises(ValueError, match='measured grid endpoint'):
        append(archive, view, handeye)


def test_capture_roll_error_still_rejects_above_point005_radians(tmp_path):
    baseline, handeye, view, metadata = fixture()
    optical = view @ handeye
    error = np.eye(4)
    error[:3, :3] = Rotation.from_euler('z', .006).as_matrix()
    wrong = optical @ error @ np.linalg.inv(handeye)
    wrong[:3, 3] += (baseline @ handeye)[:3, :3] @ [-.001, 0., 0.]
    archive = ScanArchive(tmp_path)
    archive.begin(metadata)
    with pytest.raises(ValueError, match='measured grid endpoint'):
        append(archive, wrong, handeye)


@pytest.mark.parametrize('rotation', [np.zeros((3, 3)), np.eye(4),
    np.diag([1., 1., -1.]), np.full((3, 3), np.nan), np.diag([1., 1., 1.01])])
def test_invalid_capture_rotations_rejected_before_archive_replacement(tmp_path, rotation):
    archive = ScanArchive(tmp_path)
    archive.begin({'unchanged': True})
    original = archive.state()['scan_id']
    with pytest.raises(ValueError):
        archive.begin({'capture_rotation': rotation.tolist()})
    assert archive.state()['scan_id'] == original


def test_resume_rejects_change_to_fixed_capture_orientation(tmp_path):
    _, _, _, metadata = fixture()
    archive = ScanArchive(tmp_path)
    archive.begin(metadata)
    archive.finish('cancelled')
    archive = ScanArchive(tmp_path)
    with pytest.raises(ValueError, match='metadata changed: capture_rotation'):
        archive.resume({'capture_rotation': np.eye(3).tolist()})
