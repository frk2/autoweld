"""JSON bridge preserves native capture acquisition cutoffs and older adapters."""

from autoweld.camera import CameraService


def test_capture_without_cutoff_preserves_legacy_backend_signature(tmp_path):
    calls = []
    class Backend:
        def __init__(self, *args, **kwargs):
            pass
        def capture(self, metadata):
            calls.append(metadata)
            return {'count': 1}
        def close(self):
            pass
    service = CameraService(tmp_path, backend_factory=Backend)
    try:
        metadata = {'base_from_tcp': 'fixture'}
        result = service.dispatch({'action': 'calibration_capture', 'metadata': metadata})
        assert result == {'count': 1}
        assert calls == [metadata]
    finally:
        service.close()


def test_capture_cutoff_is_forwarded_without_replacing_or_retiming_it(tmp_path):
    calls = []
    class Backend:
        def __init__(self, *args, **kwargs):
            pass
        def capture(self, metadata, *, not_before_monotonic):
            calls.append((metadata, not_before_monotonic))
            return {'count': 1}
        def close(self):
            pass
    service = CameraService(tmp_path, backend_factory=Backend)
    try:
        metadata = {'base_from_tcp': 'fixture'}
        result = service.dispatch({'action': 'calibration_capture', 'metadata': metadata,
                                   'not_before_monotonic': 12345.678})
        assert result == {'count': 1}
        assert calls == [(metadata, 12345.678)]
    finally:
        service.close()
