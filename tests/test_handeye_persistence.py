"""Saved hand-eye identity/quality gates without importing vision or devices."""
import hashlib
import json
import math
import subprocess
import sys

import pytest

from autoweld.camera import CameraService

SERIAL = 'persistence-test-camera'
SOURCE = f'spinnaker:{SERIAL}'


def identity(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def records(workspace):
    intrinsic = {'schema_version': 1, 'source_id': SOURCE, 'image_size': [640, 480],
        'camera_matrix': [[900., 0., 320.], [0., 910., 240.], [0., 0., 1.]],
        'distortion': [0.] * 5, 'camera_rms_px': .4,
        'board': {'columns': 5, 'rows': 7, 'square_mm': 4.}, 'synthetic': False,
        'laser_plane': None, 'laser_quality': {}, 'reference_rvec': None, 'reference_tvec': None}
    c, s = math.cos(.21), math.sin(.21)
    result = {'schema_version': 2, 'source_id': SOURCE, 'calibration_id': identity(intrinsic),
        'tcp_from_camera': [[c, -s, 0., .01], [s, c, 0., -.02], [0., 0., 1., .03], [0., 0., 0., 1.]],
        'units': 'm', 'coordinate_frame': 'tcp', 'views': 25,
        'translation_rms_mm': 1., 'rotation_rms_deg': .3,
        'paired_samples': [{'native_corners': [[12., 14.], [16., 18.]]}]}
    key = hashlib.sha256(SOURCE.encode()).hexdigest()[:16]
    intrinsic_path = workspace / 'runtime' / 'flir' / 'data' / f'calibration-{key}.json'
    intrinsic_path.parent.mkdir(parents=True)
    intrinsic_path.write_text(json.dumps(intrinsic, indent=2))
    result_path = workspace / 'handeye.json'
    result_path.write_text(json.dumps(result, indent=2))
    return intrinsic, result, intrinsic_path, result_path


def forbidden_backend(*args, **kwargs):
    raise AssertionError('Persistence must not construct a camera backend')


def test_load_retains_verified_result_and_count_off_without_activation(tmp_path):
    _, result, _, archive = records(tmp_path)
    original_archive = archive.read_bytes()
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    try:
        state = service.state(include_frame=False)
        assert service._backend is None
        assert not state['enabled'] and not state['fresh']
        assert state['counts']['handeye'] == 25
        assert state['handeye'] == {key: value for key, value in result.items() if key != 'paired_samples'}
        assert 'paired_samples' not in state['handeye']
        assert archive.read_bytes() == original_archive
        assert service.dispatch({'action': 'camera_stream', 'enabled': False})['handeye'] == state['handeye']
    finally:
        service.close()


def test_constructor_imports_no_vision_sdk_or_model_even_with_saved_result(tmp_path):
    records(tmp_path)
    script = '''import builtins, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'cv2', 'i2rt', 'mujoco', 'weldfinder', 'PySpin', 'scipy'):
        raise AssertionError('Unexpected vision/device/model import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from autoweld.camera import CameraService
service = CameraService(sys.argv[1], serial=sys.argv[2], backend_factory=lambda *a, **k: 1/0)
try:
    assert service.state()['counts']['handeye'] == 25
    assert service._backend is None
finally:
    service.close()
'''
    completed = subprocess.run([sys.executable, '-c', script, str(tmp_path), SERIAL],
        capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr


def replace(record, path, value):
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize('path,value', [
    (['source_id'], 'spinnaker:another-camera'), (['calibration_id'], 'stale-intrinsics'),
    (['units'], 'mm'), (['coordinate_frame'], 'base'), (['views'], 19), (['views'], True),
    (['translation_rms_mm'], 3.001), (['translation_rms_mm'], -1.),
    (['rotation_rms_deg'], 1.501), (['rotation_rms_deg'], -1.),
    (['translation_rms_mm'], float('nan')), (['tcp_from_camera', 0, 3], float('inf')),
    (['tcp_from_camera', 3, 3], 2.), (['tcp_from_camera', 0, 0], 2.),
    (['tcp_from_camera'], [[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
    (['tcp_from_camera'], [[1., 0., 0.]]), (['resolution'], [1280, 960]),
])
def test_stale_malformed_or_invalid_handeye_warns_and_does_not_load(tmp_path, caplog, path, value):
    _, result, _, result_path = records(tmp_path)
    replace(result, path, value)
    result_path.write_text(json.dumps(result))
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    try:
        state = service.state()
        assert state['handeye'] is None and state['counts']['handeye'] == 0
        assert 'Ignoring saved hand-eye calibration' in caplog.text
        assert service._backend is None
    finally:
        service.close()


@pytest.mark.parametrize('path,value', [
    (['source_id'], 'spinnaker:another-camera'), (['synthetic'], True),
    (['schema_version'], 42), (['image_size'], [0, 480]), (['image_size'], [640.5, 480]),
    (['camera_matrix', 0, 0], -1.), (['camera_matrix', 2, 2], 0.),
    (['distortion'], [0.]), (['board', 'columns'], 2), (['board', 'square_mm'], 0.),
    (['camera_rms_px'], -1.),
])
def test_invalid_installed_intrinsic_model_rejects_matching_hash(tmp_path, caplog, path, value):
    intrinsic, result, intrinsic_path, result_path = records(tmp_path)
    replace(intrinsic, path, value)
    result['calibration_id'] = identity(intrinsic)
    intrinsic_path.write_text(json.dumps(intrinsic))
    result_path.write_text(json.dumps(result))
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    try:
        assert service.state()['handeye'] is None
        assert 'Ignoring saved hand-eye calibration' in caplog.text
    finally:
        service.close()


@pytest.mark.parametrize('which', ['handeye', 'intrinsic', 'missing-intrinsic'])
def test_unreadable_or_missing_model_file_is_nonfatal(tmp_path, caplog, which):
    _, _, intrinsic_path, result_path = records(tmp_path)
    if which == 'missing-intrinsic':
        intrinsic_path.unlink()
    else:
        (result_path if which == 'handeye' else intrinsic_path).write_text('{broken JSON')
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    try:
        assert service.state()['handeye'] is None
        assert 'Ignoring saved hand-eye calibration' in caplog.text
    finally:
        service.close()


def test_canonical_intrinsic_identity_ignores_file_whitespace_and_key_order(tmp_path):
    intrinsic, _, intrinsic_path, _ = records(tmp_path)
    intrinsic_path.write_text(json.dumps(dict(reversed(list(intrinsic.items()))), indent=4))
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    try:
        assert service.state()['handeye'] is not None
    finally:
        service.close()


class StateBackend:
    def __init__(self, state):
        self.value = state
    def state(self, **kwargs):
        return self.value
    def close(self):
        pass


@pytest.mark.parametrize('change', [
    {'calibration_id': 'new-intrinsics'}, {'calibration_id': None},
    {'source_id': 'spinnaker:another-camera'}, {'resolution': [1280, 960]},
])
def test_active_intrinsic_source_or_resolution_change_invalidates_result(tmp_path, caplog, change):
    _, result, _, _ = records(tmp_path)
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    active = {**CameraService._empty_state(), 'enabled': True, 'source_id': SOURCE,
        'calibration_id': result['calibration_id'], 'resolution': [640, 480]}
    backend = StateBackend(active)
    service._backend = backend
    try:
        assert service.state()['handeye'] is not None
        backend.value = {**active, **change}
        state = service.state()
        assert state['handeye'] is None and state['counts']['handeye'] == 0
        assert 'Invalidating hand-eye calibration' in caplog.text
        backend.value = active
        assert service.state()['handeye'] is None
    finally:
        service.close()


def test_disconnect_retains_valid_measurement_and_pure_validation_does_not_install(tmp_path):
    _, result, _, result_path = records(tmp_path)
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    service._backend = StateBackend(CameraService._empty_state())
    try:
        assert service.state()['handeye'] is not None
        assert service.state()['counts']['handeye'] == 25
    finally:
        service.close()
    result_path.unlink()
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=forbidden_backend)
    try:
        validated = service.validate_handeye(result)
        assert 'paired_samples' not in validated
        assert service.state()['handeye'] is None
        service.install_handeye(result)
        result['tcp_from_camera'][0][3] = 99.
        assert service.state()['handeye']['tcp_from_camera'][0][3] == .01
        assert not result_path.exists()
    finally:
        service.close()
