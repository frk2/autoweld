"""Laser bridge units and safe full-hash to optical-identity migration."""

import copy
import json

import pytest
from test_handeye_persistence import SERIAL, SOURCE, identity, records

from autoweld.camera import CameraService, _optical_intrinsics_id


class Backend:
    def __init__(self, model, path):
        self.model, self.path, self.calls = model, path, []
    def laser_command(self, request):
        self.calls.append(request)
        if request['action'] == 'laser_solve':
            self.model.update(laser_plane=[0., .8, -.6, 50.], laser_quality={'rms_mm': .08})
            self.path.write_text(json.dumps(self.model))
        return {'views': 25}
    def state(self, **kwargs):
        return {**CameraService._empty_state(), 'enabled': True, 'source_id': SOURCE,
                'calibration_id': identity(self.model), 'optical_intrinsics_id': _optical_intrinsics_id(self.model),
                'resolution': self.model['image_size']}
    def close(self):
        pass


def test_plane_save_binds_verified_optics_preserves_archive_and_survives_restart(tmp_path):
    model, original, path, archive_path = records(tmp_path)
    backend = Backend(model, path)
    service = CameraService(tmp_path, serial=SERIAL, backend_factory=lambda *a, **k: backend)
    try:
        service.laser_command({'action': 'laser_new'})
        bound = json.loads(archive_path.read_text())
        assert bound['calibration_id'] == original['calibration_id']
        assert bound['paired_samples'] == original['paired_samples']
        assert bound['optical_intrinsics_id'] == _optical_intrinsics_id(model)
        assert bound['optical_identity_binding']['verified_full_calibration_id'] == original['calibration_id']
        service.laser_command({'action': 'laser_solve'})
        assert identity(model) != original['calibration_id']
        assert service.state()['handeye']['calibration_id'] == original['calibration_id']
        assert service.state()['handeye']['optical_intrinsics_id'] == _optical_intrinsics_id(model)
        assert json.loads(archive_path.read_text()) == bound
    finally:
        service.close()
    restarted = CameraService(tmp_path, serial=SERIAL, backend_factory=lambda *a, **k: pytest.fail('No camera allowed'))
    try:
        assert restarted.state()['counts']['handeye'] == 25
        assert restarted.state()['handeye']['tcp_from_camera'] == original['tcp_from_camera']
    finally:
        restarted.close()


@pytest.mark.parametrize('change', [
    lambda m: m['camera_matrix'][0].__setitem__(0, 901.),
    lambda m: m['distortion'].__setitem__(0, .01),
    lambda m: m.update(image_size=[1280, 960]),
    lambda m: m.update(source_id='spinnaker:different'),
    lambda m: m['board'].update(square_mm=4.1),
])
def test_changed_optics_or_metric_board_invalidates_migrated_handeye(tmp_path, change):
    model, _, path, _ = records(tmp_path)
    service = CameraService(tmp_path, serial=SERIAL)
    service._bind_handeye_optical_identity()
    service.close()
    change(model)
    path.write_text(json.dumps(model))
    restarted = CameraService(tmp_path, serial=SERIAL)
    try:
        assert restarted.state()['handeye'] is None
    finally:
        restarted.close()


def test_optical_id_without_matching_optical_snapshot_is_rejected(tmp_path):
    model, result, _, archive = records(tmp_path)
    result['optical_intrinsics_id'] = _optical_intrinsics_id(model)
    result['calibration_id'] = 'some-other-full-hash'
    archive.write_text(json.dumps(result))
    service = CameraService(tmp_path, serial=SERIAL)
    try:
        assert service.state()['handeye'] is None
    finally:
        service.close()


def test_stale_legacy_hash_cannot_be_rebound_and_original_archive_is_unchanged(tmp_path):
    _, result, _, archive = records(tmp_path)
    result['calibration_id'] = 'stale'
    archive.write_text(json.dumps(result))
    before = archive.read_bytes()
    service = CameraService(tmp_path, serial=SERIAL)
    try:
        with pytest.raises(ValueError, match='installed intrinsics'):
            service._bind_handeye_optical_identity()
        assert archive.read_bytes() == before
    finally:
        service.close()


def test_cloud_exposes_exact_millimetre_to_metre_conversion_without_robot_transform(tmp_path):
    cloud = dict(xyz_mm=[[100., -20., 150.], [0., 0., 0.]], units='mm', coordinate_frame='camera', sequence=7)
    class CloudBackend:
        def state(self, **kwargs):
            return {**CameraService._empty_state(), 'laser': {'cloud': copy.deepcopy(cloud)}}
        def close(self):
            pass
    service = CameraService(tmp_path)
    service._backend = CloudBackend()
    try:
        result = service.state()['laser']['cloud']
        assert result['xyz_mm'] == cloud['xyz_mm']
        assert result['xyz_m'] == [[.1, -.02, .15], [0., 0., 0.]]
        assert result['xyz_m_units'] == 'm'
        cloud['units'] = 'm'
        with pytest.raises(ValueError, match='in mm'):
            service.state()
    finally:
        service.close()


def test_unknown_laser_action_does_not_construct_backend(tmp_path):
    service = CameraService(tmp_path, backend_factory=lambda *a, **k: pytest.fail('No backend'))
    try:
        with pytest.raises(ValueError, match='Unsupported'):
            service.laser_command({'action': 'laser_on'})
        assert service._backend is None
    finally:
        service.close()
