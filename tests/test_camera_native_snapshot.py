"""Native diagnostics reuse the sole owner and preserve its acquisition payload."""

from types import SimpleNamespace

import pytest

from autoweld.camera import CameraService


def bridge(backend=None, operation=None):
    camera = object.__new__(CameraService)
    camera._backend, camera._operation = backend, operation
    return camera


@pytest.mark.parametrize('camera', [bridge(), bridge(SimpleNamespace(), object())])
def test_snapshot_cannot_construct_or_overlap_a_camera_lifecycle(camera):
    with pytest.raises(RuntimeError, match='active, idle camera lifecycle owner'):
        camera.dispatch({'action': 'camera_native_snapshot'})


def test_native_snapshot_delegates_same_owner_without_relabeling_or_copying_data():
    captured = []
    payload = dict(frame_png_base64='original', metadata={'sequence': 7}, cloud={'sequence': 7})
    def snapshot(**kwargs):
        captured.append(kwargs)
        return payload
    camera = bridge(SimpleNamespace(native_laser_snapshot=snapshot))
    result = camera.dispatch(dict(action='camera_native_snapshot', max_age_s=.4, not_before_monotonic=123.))
    assert result is payload
    assert captured == [dict(max_age_s=.4, not_before_monotonic=123.)]


def test_missing_native_api_reports_upgrade_without_opening_another_camera():
    with pytest.raises(RuntimeError, match='Update Weldfinder'):
        bridge(SimpleNamespace()).dispatch({'action': 'camera_native_snapshot'})
