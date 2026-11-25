"""Weld preparation configures the existing owner and cannot outlive its lease."""

from threading import Event

import pytest

from autoweld.camera import CameraService


def request(token='weld-session'):
    return {'action': 'camera_laser_configure', 'request_id': token,
            'auto_exposure': {'enabled': True}}


def test_configuration_runs_once_on_existing_lifecycle_owner(tmp_path):
    camera = CameraService(tmp_path)
    calls = []
    camera.laser_command = lambda payload: calls.append(payload)
    try:
        camera.dispatch({**request(), '_id': 'filesystem-request', '_service_queued_monotonic': 42.})
        camera._operation.result(timeout=2)
        state = camera.state(include_frame=False)
        assert state['laser_configuration']['status'] == 'complete'
        camera.dispatch(request())
        assert calls == [{'action': 'laser_configure', 'auto_exposure': {'enabled': True}}]
        assert camera._backend is None  # The test and adapter never open hardware.
    finally:
        camera.close()


def test_running_cancel_preserves_ownership_until_config_finishes(tmp_path):
    camera, entered, release = CameraService(tmp_path), Event(), Event()
    def configure(_payload):
        entered.set()
        assert release.wait(2)
    camera.laser_command = configure
    try:
        camera.dispatch(request())
        assert entered.wait(2)
        result = camera.dispatch({'action': 'camera_laser_cancel', 'request_id': 'weld-session'})
        assert result['configuring']
        assert result['laser_configuration']['status'] == 'cancelling'
        with pytest.raises(RuntimeError, match='lifecycle'):
            camera.dispatch(request('later-session'))
        release.set()
        camera._operation.result(timeout=2)
        state = camera.state(include_frame=False)
        assert not state['configuring']
        assert state['laser_configuration']['status'] == 'cancelled'
        camera.dispatch(request('later-session'))
        camera._operation.result(timeout=2)
        assert camera.state()['laser_configuration']['status'] == 'complete'
    finally:
        release.set()
        camera.close()


def test_queued_cancel_never_calls_sdk_later(tmp_path):
    camera, release = CameraService(tmp_path), Event()
    blocker = camera._executor.submit(lambda: release.wait(2))
    calls = []
    camera.laser_command = lambda payload: calls.append(payload)
    try:
        camera.dispatch(request())
        camera.dispatch({'action': 'camera_laser_cancel', 'request_id': 'weld-session'})
        release.set()
        blocker.result(timeout=2)
        camera._executor.submit(lambda: None).result(timeout=2)
        assert camera.state()['laser_configuration']['status'] == 'cancelled'
        assert not calls
    finally:
        release.set()
        camera.close()


def test_configuration_error_is_explicit_and_new_request_can_retry(tmp_path):
    camera = CameraService(tmp_path)
    def fail(_payload):
        raise ValueError('Sensor exposure bounds conflict')
    camera.laser_command = fail
    try:
        camera.dispatch(request())
        with pytest.raises(ValueError):
            camera._operation.result(timeout=2)
        state = camera.state()
        assert state['laser_configuration']['status'] == 'error'
        assert 'exposure bounds' in state['laser_configuration']['error']
        camera.laser_command = lambda _: None
        camera.dispatch(request('retry-session'))
        camera._operation.result(timeout=2)
        assert camera.state()['laser_configuration']['status'] == 'complete'
    finally:
        camera.close()


def test_sensor_readback_preserves_firmware_quantization_instead_of_nominal_value(tmp_path):
    camera = CameraService(tmp_path)
    camera.laser_command = lambda _: {'controls': {'exposure_us': 4148., 'gain_db': 12., 'auto_exposure': False}}
    try:
        camera.dispatch({**request(), 'controls': {'exposure_us': 4140.74}})
        camera._operation.result(timeout=2)
        configuration = camera.state()['laser_configuration']
        assert configuration['requested_controls'] == {'exposure_us': 4140.74}
        assert configuration['readback']['exposure_us'] == 4148.
        assert configuration['status'] == 'complete'
    finally:
        camera.close()


@pytest.mark.parametrize('change', [{'request_id': ''}, {'request_id': 5}, {'unexpected': True}])
def test_invalid_configuration_is_rejected_before_worker_submission(tmp_path, change):
    camera = CameraService(tmp_path)
    try:
        with pytest.raises(ValueError):
            camera.dispatch({**request(), **change})
        assert camera._operation is None and camera._backend is None
    finally:
        camera.close()


def test_coping_geometry_wrappers_forward_to_pure_sdk_without_constructing_camera(monkeypatch):
    from types import SimpleNamespace

    from autoweld import camera as module

    calls = []
    names = ('fit_coped_joint', 'sample_coping_curve', 'project_coping_seam', 'coping_reference_pose',
             'select_coped_surface_returns')
    def method(name):
        def invoke(*args, **kwargs):
            calls.append((name, args, kwargs))
            return {'method': name}
        return invoke
    sdk = SimpleNamespace(**{name: method(name) for name in names})
    def load(name):
        assert name == 'coped_joint'
        return sdk
    monkeypatch.setattr(module, '_weldfinder_module', load)
    profiles, model, seed = [{'xyz_mm': [[0., 0., 150.]]}], {'units': 'mm'}, [0., 0., 150.]
    assert CameraService.fit_coped_joint(profiles, coordinate_frame='camera', radii_mm=(12.7, 12.7)) == {'method': names[0]}
    CameraService.sample_coping_curve(model, seed, span_deg=180.)
    CameraService.project_coping_seam(model, [seed], seed, coping_axis=0)
    CameraService.coping_reference_pose(model, seed, standoff_mm=150.)
    CameraService.select_coped_surface_returns({'xyz_mm': [seed]})
    assert [item[0] for item in calls] == list(names)
    assert calls[0][1][0] is profiles and calls[0][2]['coordinate_frame'] == 'camera'
    assert calls[-2][2]['standoff_mm'] == 150.


def test_cached_native_diagnostics_do_not_create_or_advance_camera_owner(tmp_path):
    from types import SimpleNamespace
    camera = CameraService(tmp_path, backend_factory=lambda: pytest.fail('Diagnostics opened a camera'))
    try:
        assert camera.dispatch({'action': 'camera_native_diagnostics'})['available'] is False
        assert camera._backend is None and camera._operation is None
        camera._backend = SimpleNamespace(native_diagnostics=lambda: {'raw_pointcloud': {'xyz_ms': 57.}},
                                          close=lambda: None)
        assert camera.dispatch({'action': 'camera_native_diagnostics'}) == {'raw_pointcloud': {'xyz_ms': 57.}}
        assert camera._operation is None
    finally:
        camera.close()
