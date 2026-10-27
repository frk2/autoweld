"""Hardware-free tests for camera ownership and measured hand-eye geometry."""

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.calibration import CalibrationCoordinator, cone_plan, solve_handeye
from autoweld.camera import CameraService


def paired_samples():
    known = np.eye(4)
    known[:3, :3] = Rotation.from_euler('xyz', [.13, -.2, .07]).as_matrix()
    known[:3, 3] = [.015, -.022, .041]
    board = np.eye(4)
    board[:3, :3] = Rotation.from_euler('xyz', [.1, .1, -.2]).as_matrix()
    board[:3, 3] = [.25, .05, .4]
    poses, _ = cone_plan(np.eye(4))
    samples = [dict(base_from_tcp=g.tolist(),
                    camera_from_target_m=(np.linalg.inv(known) @ np.linalg.inv(g) @ board).tolist(),
                    source_id='fixture:camera', calibration_id='fixture:intrinsics') for g in poses]
    return known, samples


def test_handeye_recovers_known_metric_transform_and_rejects_bad_pairs():
    known, samples = paired_samples()
    result = solve_handeye(samples)
    assert np.allclose(result['tcp_from_camera'], known, atol=1e-9)
    assert result['translation_rms_mm'] < 1e-8
    assert result['rotation_rms_deg'] < 1e-8
    samples[-1]['camera_from_target_m'][0][3] += .05
    with pytest.raises(ValueError, match='residual'):
        solve_handeye(samples)


def test_handeye_rejects_insufficient_excitation_and_mixed_intrinsics():
    _, samples = paired_samples()
    with pytest.raises(ValueError, match='20 accepted'):
        solve_handeye(samples[:10])
    samples[-1]['calibration_id'] = 'other'
    with pytest.raises(ValueError, match='share one camera'):
        solve_handeye(samples)
    sample = samples[0]
    with pytest.raises(ValueError, match='two distinct axes'):
        solve_handeye([sample] * 20)


def test_cone_points_optical_axis_at_apex_with_configured_mount_offset():
    baseline = np.eye(4)
    baseline[:3, :3] = Rotation.from_euler('xyz', [.2, .3, .4]).as_matrix()
    baseline[:3, 3] = [.2, .1, .3]
    original = baseline.copy()
    offset = np.array([.01, .02, .03])
    poses, geometry = cone_plan(baseline, camera_offset_m=offset)
    apex = np.asarray(geometry['apex_m'])
    camera_origin = baseline[:3, 3] + baseline[:3, :3] @ offset
    expected_radius = .150*np.tan(np.deg2rad(7.5))
    np.testing.assert_allclose(apex-camera_origin, baseline[:3, 2]*.150, atol=1e-14)
    assert len(poses) == geometry['count'] == 25
    assert geometry['opening_angle_deg'] == 15.
    assert geometry['half_angle_deg'] == 7.5
    assert geometry['radius_m'] == pytest.approx(.019747875)
    assert not any(np.allclose(pose, baseline) for pose in poses)
    camera_positions = []
    for pose in poses:
        origin = pose[:3, 3] + pose[:3, :3] @ offset
        camera_positions.append(origin)
        axis = apex-origin
        axis /= np.linalg.norm(axis)
        np.testing.assert_allclose(axis, pose[:3, 2], atol=1e-14)
        relative = baseline[:3, :3].T@(origin-camera_origin)
        assert relative[2] == pytest.approx(0., abs=1e-14)
        assert np.linalg.norm(relative[:2]) == pytest.approx(expected_radius, abs=1e-14)
        assert np.rad2deg(np.arccos(np.clip(axis@baseline[:3, 2], -1., 1.))) == pytest.approx(7.5)
        # A minimum tilt has no baseline optical-axis roll component.
        tilt = Rotation.from_matrix(baseline[:3, :3].T@pose[:3, :3]).as_rotvec()
        assert tilt[2] == pytest.approx(0., abs=1e-14)
        assert np.linalg.norm(tilt) == pytest.approx(np.deg2rad(7.5))
    ring = np.asarray(geometry['cone_ring_m'])
    assert len(ring) == len(np.unique(ring, axis=0)) == 25
    np.testing.assert_allclose(ring, camera_positions, atol=1e-14)
    np.testing.assert_allclose(geometry['capture_tcp_origins_m'], np.array(poses)[:, :3, 3], atol=1e-14)
    local = (ring-camera_origin)@baseline[:3, :3]
    azimuths = np.mod(np.arctan2(local[:, 1], local[:, 0]), 2*np.pi)
    differences = np.mod(np.roll(azimuths, -1)-azimuths, 2*np.pi)
    np.testing.assert_allclose(differences, 2*np.pi/25, atol=1e-14)
    closed = np.asarray(geometry['cone_ring_closed_m'])
    assert len(closed) == 26
    np.testing.assert_array_equal(closed[:-1], ring)
    np.testing.assert_array_equal(closed[-1], ring[0])
    np.testing.assert_array_equal(baseline, original)
    json.dumps(geometry, allow_nan=False)


@pytest.mark.parametrize('count', [20, 25, 100])
def test_explicit_view_count_remains_unique_with_fixed_depth(count):
    poses, geometry = cone_plan(np.eye(4), count=count)
    assert len(poses) == len(geometry['cone_ring_m']) == count
    origins = np.array(poses)[:, :3, 3]
    assert len(np.unique(origins, axis=0)) == count
    np.testing.assert_allclose(origins[:, 2], 0., atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(origins[:, :2], axis=1), .150*np.tan(np.deg2rad(7.5)))


@pytest.mark.parametrize('count', [19, 101, 25.5, True])
def test_invalid_cone_view_count_is_rejected(count):
    with pytest.raises(ValueError, match='20–100'):
        cone_plan(np.eye(4), count=count)


def test_camera_singleton_is_lazy_and_browser_handles_share_it(tmp_path):
    created = []
    class Backend:
        def __init__(self, *args, **kwargs):
            created.append(self)
            self.enabled = False
        def stream(self, enabled):
            self.enabled = enabled
        def state(self, **kwargs):
            return {'enabled': self.enabled, 'counts': {'intrinsic': 0}}
        def close(self):
            pass
    old = CameraService._instance
    CameraService._instance = None
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            handles = list(pool.map(lambda _: CameraService.get_instance(tmp_path, backend_factory=Backend), range(16)))
        assert len({id(h) for h in handles}) == 1
        assert not created
        assert not handles[0].state()['enabled']
        handles[0].dispatch({'action': 'camera_stream', 'enabled': False})
        assert not created
        handles[0].dispatch({'action': 'camera_stream', 'enabled': True})
        # dispatch() can already consume a fast completed lifecycle future.
        # Wait on the public state, not a private future that may be cleared.
        deadline = time.monotonic()+1
        while not handles[0].state()['enabled'] and time.monotonic() < deadline:
            time.sleep(.001)
        assert len(created) == 1
        assert handles[1].state()['enabled']
        json.dumps(handles[0].state(), allow_nan=False)
    finally:
        handles[0].close()
        CameraService._instance = old


class FakeCamera:
    def state(self, **kwargs):
        return {'fresh': False, 'board': {'found': False}, 'intrinsics': None}


def test_auto_requires_a_good_baseline_before_motion_or_camera_reset(tmp_path):
    moves = []
    coordinator = CalibrationCoordinator(FakeCamera(), lambda: np.eye(4),
                                         lambda *args, **kwargs: moves.append(args),
                                         lambda: True, lambda: None, tmp_path)
    try:
        with pytest.raises(ValueError, match='Baseline'):
            coordinator.dispatch({'action': 'calibration_start'})
        assert not moves
        assert coordinator.geometry is None
    finally:
        coordinator.close()


def test_cancel_discards_background_capture_and_never_moves(tmp_path):
    started, release = threading.Event(), threading.Event()
    moves, holds = [], []
    class Camera(FakeCamera):
        def state(self, **kwargs):
            return {'fresh': True, 'board': {'found': True}, 'intrinsics': {}}
        def dispatch(self, request):
            started.set()
            release.wait(timeout=2)
            return {'count': 1}
    coordinator = CalibrationCoordinator(Camera(), lambda: np.eye(4),
                                         lambda *args, **kwargs: moves.append(args),
                                         lambda: True, lambda: holds.append(True), tmp_path)
    try:
        coordinator.dispatch({'action': 'calibration_capture'})
        assert started.wait(timeout=1)
        coordinator.dispatch({'action': 'calibration_cancel'})
        release.set()
        coordinator.future.result(timeout=1)
        coordinator.tick()
        assert not moves
        assert holds == [True]
        assert coordinator.state()['status'] == 'cancelled'
    finally:
        release.set()
        coordinator.close()


def test_camera_startup_and_state_never_wait_for_hardware(tmp_path):
    started, release = threading.Event(), threading.Event()
    class Backend:
        def __init__(self, *args, **kwargs):
            self.enabled = False
        def stream(self, enabled):
            if enabled:
                started.set()
                release.wait(timeout=2)
            self.enabled = enabled
        def state(self, **kwargs):
            return {'enabled': self.enabled, 'fresh': self.enabled, 'error': None,
                    'counts': {'intrinsic': 0}, 'board': {'found': False}}
        def close(self):
            self.enabled = False
    service = CameraService(tmp_path, backend_factory=Backend)
    try:
        result = service.dispatch({'action': 'camera_stream', 'enabled': True})
        assert result['starting']
        assert result['status'] == 'queued'
        assert started.wait(timeout=1)
        # This must return while the SDK startup is still blocked on release.
        assert service.state()['starting']
        assert not service.state()['fresh']
        service.dispatch({'action': 'camera_stream', 'enabled': False})
        release.set()
        service._operation.result(timeout=1)
        service.state()  # Schedules the latest requested off state.
        if service._operation is not None:
            service._operation.result(timeout=1)
        assert not service.state()['enabled']
    finally:
        release.set()
        service.close()


def test_cooldown_cancellation_never_overwrites_controller_rest_motion(tmp_path):
    moves, holds = [], []
    coordinator = CalibrationCoordinator(FakeCamera(), lambda: np.eye(4),
                                         lambda *a, **kw: moves.append(a), lambda: True,
                                         lambda: holds.append(True), tmp_path)
    try:
        coordinator.status = 'capturing'
        coordinator._hunter = object()
        coordinator._hunt_moving = True
        coordinator.hunt = {'phase': 'capture', 'motion_active': True}
        coordinator.future = Future()
        coordinator.pending_kind = 'capture'
        coordinator.future.set_result({'late': 'capture'})
        state = coordinator.abort_for_cooldown()
        coordinator.tick()
        assert state['status'] == coordinator.status == 'cancelled'
        assert 'returning to rest' in state['message']
        assert coordinator.cancelled
        assert coordinator._hunter is None and not coordinator._hunt_moving
        assert not coordinator.hunt['motion_active']
        assert coordinator.index == 0 and not coordinator.samples
        assert coordinator.future is None
        assert not holds and not moves
    finally:
        coordinator.close()


def test_auto_captures_twenty_five_edge_poses_and_returns_without_baseline_capture(tmp_path, monkeypatch):
    import autoweld.calibration as calibration
    now = [10.]
    monkeypatch.setattr(calibration.time, 'monotonic', lambda: now[0])
    current = np.eye(4)
    current[:3, 3] = [.2, .1, .3]
    baseline = current.copy()
    moves, captured, validated = [], [], []
    class Camera:
        def state(self, **kwargs):
            return {'fresh': True, 'enabled': True, 'error': None, 'intrinsics': {},
                    'source_id': 'fixture:camera', 'generation': 1, 'sequence': int(now[0]*100),
                    'read_started': now[0], 'resolution': [640, 480],
                    'board': {'found': True, 'corners': [[300., 200.], [320., 200.], [300., 220.], [320., 220.]]}}
        def dispatch(self, request):
            if request['action'] == 'calibration_reset':
                captured.clear()
                return {}
            if request['action'] == 'calibration_capture':
                captured.append(np.asarray(request['metadata']['base_from_tcp']))
                return {'count': len(captured)}
            assert request['action'] == 'calibration_solve'
            return {'views': len(captured)}
    def move(pose, **kwargs):
        moves.append(np.asarray(pose).copy())
        current[:] = pose
    coordinator = CalibrationCoordinator(Camera(), lambda: current.copy(), move, lambda: True,
                                         lambda: pytest.fail('Successful calibration held unexpectedly'), tmp_path,
                                         validate_pose=lambda pose: validated.append(np.asarray(pose).copy()))
    def submit(kind, function):
        coordinator.pending_kind = kind
        coordinator.future = Future()
        try:
            coordinator.future.set_result(function())
        except Exception as error:
            coordinator.future.set_exception(error)
    monkeypatch.setattr(coordinator, '_submit', submit)
    try:
        assert coordinator.view_count == 25
        coordinator.dispatch({'action': 'calibration_start', 'kind': 'intrinsic'})
        for _ in range(100):
            now[0] += .5
            coordinator.tick()
            if coordinator.status in ('complete', 'error'):
                break
        assert coordinator.status == 'complete', coordinator.state()
        assert coordinator.last_result == {'views': 25}
        assert coordinator.index == 25
        assert len(coordinator.poses) == len(captured) == 25
        assert len(validated) == 25
        np.testing.assert_allclose(captured, coordinator.poses, atol=1e-14)
        assert not any(np.allclose(pose, baseline) for pose in captured)
        np.testing.assert_allclose(moves[-1], baseline, atol=1e-14)
        assert len(moves) == 26
        np.testing.assert_allclose(coordinator.geometry['cone_ring_m'], np.array(captured)[:, :3, 3])
    finally:
        coordinator.close()
