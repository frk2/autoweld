"""Offline diagnostics recover known optics and retain independent holdout views."""

import json

import cv2
import numpy as np

from scripts.assess_intrinsics import _object_points, assess_checkpoint


def checkpoint(tmp_path, *, count=25, front_on=False, images=False):
    board = {'columns': 5, 'rows': 7, 'square_mm': 4.}
    objects = _object_points(board)
    k = np.array([[2200., 0., 720.], [0., 2180., 540.], [0., 0., 1.]])
    points = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        r = np.zeros(3) if front_on else np.array([.14 * np.cos(angle), .14 * np.sin(angle), .05 * np.sin(2 * angle)])
        t = np.array([-8. + 5 * np.sin(3 * angle), -12. + 3 * np.cos(2 * angle), 140. + 12 * np.sin(angle)])
        corners = cv2.projectPoints(objects, r, t, k, np.zeros(5))[0].astype(np.float32)
        points.append(corners.tolist())
    data = {'source_id': 'fixture:known-intrinsics', 'board': board, 'image_size': [1440, 1080],
            'camera_views': points, 'captures': {'camera': []}, 'images': {}}
    if images:
        import base64
        encoded = cv2.imencode('.png', np.full((1080, 1440), 63, np.uint8))[1].tobytes()
        data['captures']['camera'] = [{'id': 'native-fixture'}]
        data['images']['native-fixture'] = {'native': base64.b64encode(encoded).decode()}
    path = tmp_path / 'checkpoint.json'
    path.write_text(json.dumps(data))
    return path, k


def test_known_intrinsics_and_independent_twenty_view_cross_validation(tmp_path):
    path, k = checkpoint(tmp_path)
    original = path.read_bytes()
    report = assess_checkpoint(path)
    assert report['existing_quality_gates_passed']
    np.testing.assert_allclose(report['fit']['camera_matrix'], k, atol=.1)
    assert report['fit']['rms_px'] < 1e-3
    assert report['fit']['max_normal_separation_deg'] > 12
    assert report['fit']['normal_second_axis_spread_deg'] > 2
    validation = report['cross_validation']
    assert validation['worst_view_rms_px'] < 1e-3
    assert max(validation['focal_relative_range']) < .001
    covered = []
    for fold in validation['folds']:
        assert len(fold['training_indices']) == 20
        assert not set(fold['training_indices']) & set(fold['heldout_indices'])
        covered.extend(fold['heldout_indices'])
    assert sorted(covered) == list(range(25))
    assert path.read_bytes() == original
    assert not report['installed']


def test_one_view_cannot_bootstrap_focal_length_but_exports_exact_native_png(tmp_path):
    path, _ = checkpoint(tmp_path, count=1, images=True)
    output = tmp_path / 'analysis'
    report = assess_checkpoint(path, output_directory=output, export_native=True)
    assert not report['existing_quality_gates_passed']
    assert 'fit' not in report
    assert any('Fewer than three' in w for w in report['warnings'])
    assert len(list(output.glob('native-*.png'))) == 1
    assert (output / 'assessment.json').exists()
    assert report['image_diagnostics'][0]['brightness_percentiles'] == [63.] * 5
    assert report['image_diagnostics'][0]['saturation_fraction'] == 0.


def test_front_on_views_do_not_pass_tilt_gate_even_with_small_residual(tmp_path):
    path, k = checkpoint(tmp_path, front_on=True)
    initial = {'camera_matrix': k.tolist(), 'distortion': [0.] * 5}
    report = assess_checkpoint(path, initial=initial)
    assert report['fit']['rms_px'] < 1e-3
    assert not report['existing_quality_gates_passed']
    assert report['fit']['max_normal_separation_deg'] < 1
    assert any('Weak second tilt axis' in w for w in report['warnings'])
