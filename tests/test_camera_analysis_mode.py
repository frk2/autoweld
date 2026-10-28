"""Camera analysis intent stays lazy and off the robot service thread."""

import threading
import time

import pytest

from autoweld.camera import CameraService


class Backend:
    def __init__(self, *args, **kwargs):
        self.enabled = False
        self.mode = 'checkerboard'
        self.calls = []

    def set_analysis_mode(self, mode):
        self.calls.append(('mode', mode, threading.get_ident()))
        self.mode = mode

    def stream(self, enabled):
        self.calls.append(('stream', enabled, threading.get_ident()))
        self.enabled = enabled

    def state(self, **kwargs):
        return {'enabled': self.enabled, 'fresh': self.enabled, 'error': None,
                'counts': {'intrinsic': 0}, 'board': {'found': False},
                'analysis_mode': self.mode, 'board_detection_enabled': self.mode == 'checkerboard'}

    def close(self):
        self.enabled = False


def settled(service):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        state = service.state()
        if not any(state.get(key) for key in ('starting', 'stopping', 'configuring', 'analysis_pending')):
            return state
        time.sleep(.001)
    raise AssertionError('Camera operation did not complete')


def test_analysis_intent_with_camera_off_never_constructs_owner(tmp_path):
    constructed = []
    service = CameraService(tmp_path, backend_factory=lambda *a, **k: constructed.append(True))
    try:
        state = service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        assert state['analysis_mode'] == 'laser'
        assert state['analysis_mode_active'] is None
        assert not state['enabled'] and not state['configuring']
        assert not constructed and service._backend is None
        service.dispatch({'action': 'camera_analysis', 'mode': 'checkerboard'})
        assert not constructed
    finally:
        service.close()


@pytest.mark.parametrize('mode', [None, '', True, 1, [], {}, 'native', 'Laser'])
def test_invalid_analysis_intent_rejects_without_constructing_camera(tmp_path, mode):
    service = CameraService(tmp_path, backend_factory=Backend)
    try:
        with pytest.raises(ValueError, match='checkerboard or laser'):
            service.dispatch({'action': 'camera_analysis', 'mode': mode})
        assert service.state()['analysis_mode'] == 'checkerboard'
        assert service._backend is None
    finally:
        service.close()


def test_laser_intent_is_applied_before_stream_start_on_same_worker(tmp_path):
    service = CameraService(tmp_path, backend_factory=Backend)
    main_thread = threading.get_ident()
    try:
        service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        state = settled(service)
        assert state['analysis_mode'] == state['analysis_mode_active'] == 'laser'
        assert not state['board_detection_enabled'] and state['enabled']
        calls = service._backend.calls
        assert [entry[:2] for entry in calls] == [('mode', 'laser'), ('stream', True)]
        assert all(entry[2] != main_thread for entry in calls)
        assert len({entry[2] for entry in calls}) == 1
        service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        settled(service)
        assert service._backend.calls == calls
        assert len(calls) == 2
    finally:
        service.close()


def test_mode_configuration_never_blocks_state_or_camera_service_tick(tmp_path):
    started, release = threading.Event(), threading.Event()

    class BlockingMode(Backend):
        def set_analysis_mode(self, mode):
            if mode == 'laser':
                started.set()
                release.wait(timeout=3)
            super().set_analysis_mode(mode)

    service = CameraService(tmp_path, backend_factory=BlockingMode)
    try:
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        settled(service)
        state = service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        assert started.wait(timeout=1)
        assert state['configuring'] and state['analysis_pending']
        assert service.state()['configuring']
        assert not service.state()['fresh']
        release.set()
        state = settled(service)
        assert state['analysis_mode_active'] == 'laser' and state['fresh']
    finally:
        release.set()
        service.close()


def test_latest_analysis_intent_during_startup_is_coalesced_without_another_owner(tmp_path):
    started, release = threading.Event(), threading.Event()
    created = []

    class BlockingStart(Backend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def stream(self, enabled):
            if enabled:
                started.set()
                release.wait(timeout=3)
            super().stream(enabled)

    service = CameraService(tmp_path, backend_factory=BlockingStart)
    try:
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        assert started.wait(timeout=1)
        service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        service.dispatch({'action': 'camera_analysis', 'mode': 'checkerboard'})
        queued = service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        assert queued['starting'] and queued['analysis_mode'] == 'laser'
        assert queued['analysis_pending'] and not queued['fresh']
        release.set()
        state = settled(service)
        assert len(created) == 1
        assert state['enabled'] and state['analysis_mode_active'] == 'laser'
        assert [entry[:2] for entry in created[0].calls] == [
            ('mode', 'checkerboard'), ('stream', True), ('mode', 'laser')]
    finally:
        release.set()
        service.close()


def test_queued_stop_and_mode_change_do_not_reopen_camera(tmp_path):
    started, release = threading.Event(), threading.Event()

    class BlockingStart(Backend):
        def stream(self, enabled):
            if enabled:
                started.set()
                release.wait(timeout=3)
            super().stream(enabled)

    service = CameraService(tmp_path, backend_factory=BlockingStart)
    try:
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        assert started.wait(timeout=1)
        service.dispatch({'action': 'camera_stream', 'enabled': False})
        service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        release.set()
        state = settled(service)
        assert not state['enabled'] and state['analysis_mode_active'] == 'laser'
        assert [entry[:2] for entry in service._backend.calls] == [
            ('mode', 'checkerboard'), ('stream', True), ('stream', False), ('mode', 'laser')]
    finally:
        release.set()
        service.close()


def test_old_camera_backend_supports_default_but_explicit_laser_has_clear_error(tmp_path):
    class OldBackend(Backend):
        set_analysis_mode = None

    service = CameraService(tmp_path, backend_factory=OldBackend)
    try:
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        assert settled(service)['enabled']
        service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        state = settled(service)
        assert 'set_analysis_mode API' in state['error']
        assert state['analysis_mode'] == 'laser' and state['analysis_mode_active'] == 'checkerboard'
        assert [entry[:2] for entry in service._backend.calls] == [('stream', True)]
    finally:
        service.close()


def test_old_backend_cannot_open_stream_for_unsupported_laser_intent(tmp_path):
    class OldBackend(Backend):
        set_analysis_mode = None

    service = CameraService(tmp_path, backend_factory=OldBackend)
    try:
        service.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        state = settled(service)
        assert not state['enabled']
        assert 'set_analysis_mode API' in state['error']
        assert service._backend.calls == []
    finally:
        service.close()


def test_explicit_same_mode_restores_actual_board_state_changed_by_native_operation(tmp_path):
    class DriftBackend(Backend):
        board_enabled = True

        def set_analysis_mode(self, mode):
            super().set_analysis_mode(mode)
            self.board_enabled = mode == 'checkerboard'

        def state(self, **kwargs):
            return {**super().state(**kwargs), 'board_detection_enabled': self.board_enabled}

    service = CameraService(tmp_path, backend_factory=DriftBackend)
    try:
        service.dispatch({'action': 'camera_stream', 'enabled': True})
        settled(service)
        backend = service._backend
        backend.board_enabled = False  # Native pointcloud_configure selects discovery.
        assert not service.state()['board_detection_enabled']
        assert service.state()['analysis_mode_active'] == 'checkerboard'
        service.dispatch({'action': 'camera_analysis', 'mode': 'checkerboard'})
        assert settled(service)['board_detection_enabled']
        assert [entry[:2] for entry in backend.calls] == [
            ('mode', 'checkerboard'), ('stream', True), ('mode', 'checkerboard')]
        before = len(backend.calls)
        service.dispatch({'action': 'camera_analysis', 'mode': 'checkerboard'})
        settled(service)
        assert len(backend.calls) == before
    finally:
        service.close()
