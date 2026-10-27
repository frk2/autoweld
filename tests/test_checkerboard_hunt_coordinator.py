"""Frame/motion integration for automatic search, with no camera or CAN device."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.calibration import CalibrationCoordinator


class Camera:
    def __init__(self, clock):
        self.clock = clock
        self.sequence = 0
        self.enabled = self.fresh = True
        self.full = False
        self.corners = [[700., 200.], [740., 200.], [700., 240.], [740., 240.]]
        self.read_started = clock[0]
        self.processing_completed = self.read_started
        self.error = None
        self.captures = []
        self.resets = 0

    def frame(self, *, full=None, corners=None):
        self.sequence += 1
        self.read_started = self.clock[0]
        self.processing_completed = self.clock[0]
        if full is not None:
            self.full = full
        if corners is not None:
            self.corners = corners

    def state(self, **kwargs):
        return {'enabled': self.enabled, 'fresh': self.fresh, 'error': self.error,
            'source_id': 'synthetic', 'generation': self.resets, 'sequence': self.sequence,
            'read_started': self.read_started, 'timestamp_monotonic': self.processing_completed,
            'resolution': [1000, 800], 'intrinsics': {'calibration_id': 'known'},
            'board': {'found': self.full, 'corners': self.corners}}

    def dispatch(self, req):
        if req['action'] == 'calibration_reset':
            self.resets += 1
            return {}
        if req['action'] == 'calibration_capture':
            if not self.full:
                raise ValueError('Full checkerboard required')
            self.captures.append(copy.deepcopy(req))
            return {'count': len(self.captures)}
        if req['action'] == 'calibration_solve':
            return {'views': len(self.captures)}
        raise AssertionError(req)

    def target_pose(self, **kwargs):
        assert self.full
        return {'read_started': self.read_started, 'camera_from_target': np.diag([1., 1., 1., 1.]).tolist(),
            'source_id': 'synthetic', 'calibration_id': 'known', 'sequence': self.sequence,
            'generation': self.resets, 'reprojection_rms_px': .1}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('autoweld.calibration.time', SimpleNamespace(monotonic=lambda: clock[0]))
    camera = Camera(clock)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [.2, -.3, .8]).as_matrix()
    pose[:3, 3] = [.2, .1, .3]
    current = [pose.copy()]
    ready = [True]
    moves, validated, holds = [], [], []
    def move(target, **kwargs):
        assert kwargs == {'automatic': True}
        moves.append(target.copy())
        ready[0] = False
    def hold():
        holds.append(True)
        ready[0] = True
    c = CalibrationCoordinator(camera, lambda: current[0], move, lambda: ready[0], hold, tmp_path,
                               validate_pose=lambda p: validated.append(p.copy()))
    def fresh(dt=.5, **kwargs):
        clock[0] += dt
        camera.frame(**kwargs)
        c.tick()
    def settle():
        current[0] = moves[-1].copy()
        ready[0] = True
        c.tick()
    def start(kind='intrinsic'):
        c.dispatch({'action': 'calibration_start', 'kind': kind, 'views': 20})
        if c.future:
            c.future.result(timeout=1)
            c.tick()
        assert c.status == 'hunting'
    yield SimpleNamespace(c=c, camera=camera, clock=clock, current=current, ready=ready,
        initial=pose, moves=moves, validated=validated, holds=holds, fresh=fresh, settle=settle, start=start)
    c.close()


@pytest.mark.parametrize('kind', ['intrinsic', 'handeye'])
def test_partial_baseline_moves_exact_5mm_in_optical_frame_then_rebases_cone(rig, kind):
    rig.start(kind)
    assert rig.c.geometry is None  # No cone around an unframed board.
    rig.fresh()
    offset = np.mean(rig.camera.corners, axis=0)-[500, 400]
    expected = rig.initial[:3, :3]@np.r_[.005*offset/np.linalg.norm(offset), 0]
    np.testing.assert_allclose(rig.moves[0][:3, 3]-rig.initial[:3, 3], expected, atol=1e-12)
    np.testing.assert_allclose(rig.moves[0][:3, :3], rig.initial[:3, :3], atol=1e-12)
    assert len(rig.validated) == 1
    assert rig.c.state()['hunt']['corner_count'] == 4
    assert rig.c.state()['hunt']['motion_active']
    assert not rig.camera.captures
    rig.c.tick()
    assert len(rig.moves) == 1  # No new image/pose while moving.
    rig.settle()
    rig.fresh(full=True)
    assert rig.c.status == 'moving'
    np.testing.assert_allclose(rig.c.geometry['baseline_after_hunt'], rig.moves[0])
    np.testing.assert_allclose(rig.c.poses[0], rig.moves[-1])
    assert len(rig.moves) == 2 and len(rig.c.poses) == 20
    assert not rig.camera.captures
    np.testing.assert_allclose(rig.c.return_pose, rig.initial)
    assert len(rig.validated) == 21  # Step + all cone endpoints.
    json.dumps(rig.c.state(), allow_nan=False)


def test_full_baseline_has_no_hunt_motion(rig):
    rig.camera.full = True
    rig.start()
    rig.fresh()
    assert rig.c.status == 'moving'
    assert rig.c.hunt['steps'] == 0 and not rig.c.hunt['motion_active']
    assert len(rig.moves) == 1 and not rig.camera.captures
    np.testing.assert_allclose(rig.c.poses[0], rig.moves[0])
    assert not np.allclose(rig.c.poses[0], rig.initial)
    np.testing.assert_allclose(rig.c.return_pose, rig.initial)


def test_partial_capture_pose_hunts_and_records_actual_corrected_robot_pose(rig):
    rig.c.kind = 'intrinsic'
    rig.c.poses = [rig.initial.copy(), rig.initial.copy()]
    rig.c.status, rig.c.settled_at = 'settling', rig.clock[0]
    rig.fresh()
    assert rig.c.status == 'hunting'
    assert rig.c.hunt['phase'] == 'capture'
    rig.fresh()
    rig.settle()
    rig.fresh(full=True)
    corrected = rig.current[0].copy()
    rig.fresh()
    assert rig.c.status == 'capturing'
    rig.c.future.result(timeout=1)
    rig.c.tick()
    assert len(rig.camera.captures) == 1
    np.testing.assert_allclose(rig.camera.captures[0]['metadata']['base_from_tcp'], corrected)
    assert rig.c.index == 1


def test_hunt_ignores_pre_move_exposure_even_if_processing_completed_later(rig):
    rig.start()
    rig.fresh()
    rig.settle()
    rig.clock[0] += .6
    rig.camera.processing_completed = rig.clock[0]
    rig.c.tick()
    assert len(rig.moves) == 1
    rig.clock[0] += 1.5
    rig.c.tick()
    assert rig.c.status == 'error'
    assert 'after' in rig.c.last_error
    assert rig.holds


@pytest.mark.parametrize('failure', ['stream_off', 'camera_error', 'stale'])
def test_camera_failure_stops_hunting_without_another_move(rig, failure):
    rig.start()
    if failure == 'stream_off':
        rig.camera.enabled = False
    elif failure == 'camera_error':
        rig.camera.error = 'Camera disconnected'
    else:
        rig.camera.fresh = False
        rig.clock[0] += 2.1
    rig.c.tick()
    assert rig.c.status == 'error'
    assert rig.holds and not rig.moves


def test_cancel_while_hunting_holds_and_never_restarts_on_next_frame(rig):
    rig.start()
    rig.fresh()
    rig.c.dispatch({'action': 'calibration_cancel'})
    rig.fresh(full=True)
    assert rig.c.status == 'cancelled'
    assert rig.holds == [True]
    assert len(rig.moves) == 1
    assert not rig.c.hunt['motion_active']


def test_planner_rejection_never_executes_hunt_step(rig):
    rig.start()
    def reject(_):
        raise ValueError('Blocked path')
    rig.c.validate_pose = reject
    rig.fresh()
    assert rig.c.status == 'error'
    assert rig.c.last_error == 'Blocked path'
    assert not rig.moves and rig.holds


def test_blank_frames_use_bounded_5mm_reacquisition_without_fabricated_centroid(rig):
    rig.camera.corners = []
    rig.start()
    rig.fresh()
    assert rig.c.hunt['reacquiring']
    assert rig.c.hunt['centroid_px'] is None
    assert rig.c.hunt['offset_px'] is None
    assert np.linalg.norm(rig.moves[0][:3, 3]-rig.initial[:3, 3]) == pytest.approx(.005)
    rig.settle()
    rig.clock[0] += 60.
    rig.camera.frame()
    rig.c.tick()
    assert rig.c.status == 'error'
    assert 'timeout' in rig.c.last_error
    assert len(rig.moves) == 1


def test_partial_board_is_never_captured_and_duplicate_start_is_rejected(rig):
    rig.start()
    with pytest.raises(RuntimeError, match='Cancel'):
        rig.c.dispatch({'action': 'calibration_start'})
    rig.fresh()
    assert not rig.camera.captures
    assert rig.c.samples == []


@pytest.mark.parametrize('field', ['source', 'generation'])
def test_camera_identity_change_stops_an_existing_hunt(rig, field):
    rig.start()
    rig.fresh()
    rig.settle()
    if field == 'generation':
        rig.camera.resets += 1
    else:
        original = rig.camera.state
        rig.camera.state = lambda **kwargs: {**original(**kwargs), 'source_id': 'other'}
    rig.fresh()
    assert rig.c.status == 'error'
    assert 'changed' in rig.c.last_error
    assert len(rig.moves) == 1 and rig.holds


def test_partial_view_that_reappears_during_native_capture_reenters_hunt(rig):
    rig.camera.full = True
    rig.start()
    rig.fresh()
    rig.c._capture = lambda: (_ for _ in ()).throw(ValueError('Full board lost during capture'))
    rig.settle()  # The first edge must settle before its fresh capture can fail.
    rig.fresh()
    try:
        rig.c.future.result(timeout=1)
    except ValueError:
        pass
    rig.camera.full = False
    rig.c.tick()
    assert rig.c.status == 'hunting'
    assert rig.c.hunt['phase'] == 'capture'
    assert not rig.camera.captures


def test_return_after_hunted_baseline_uses_original_requested_pose(rig):
    rig.start()
    rig.fresh()
    rig.settle()
    rig.fresh(full=True)
    rig.settle()
    rig.c.index = len(rig.c.poses)-1
    rig.fresh()
    rig.c.future.result(timeout=1)
    rig.c.tick()
    assert rig.c.status == 'returning'
    np.testing.assert_allclose(rig.moves[-1], rig.initial)


def test_actual_opencv_partial_grid_drives_virtual_camera_into_full_view(rig):
    # No frame grabber/model: render the board as it moves oppositely to the
    # camera's measured image-plane translation, then run the real detector.
    from autoweld.camera import _backend_class
    try:
        _backend_class()
    except RuntimeError:
        pytest.skip('Weldfinder camera checkout unavailable')
    from weldfinder.camera_settings import CameraSettings
    from weldfinder.vision import board_observation

    settings = CameraSettings()
    history = []
    original_state = rig.camera.state
    def render_state(**kwargs):
        translation = rig.initial[:3, :3].T@(rig.current[0][:3, 3]-rig.initial[:3, 3])
        image = np.full((440, 360), 255, np.uint8)
        origin = np.rint([210, 60]-translation[:2]*10000).astype(int)
        for row in range(settings.rows+1):
            for col in range(settings.columns+1):
                if (row+col) % 2 == 0:
                    x, y = origin+[col*40, row*40]
                    if x < 360 and y < 440 and x+40 > 0 and y+40 > 0:
                        image[max(0, y):min(440, y+40), max(0, x):min(360, x+40)] = 0
        observed = board_observation(image, settings, search_scale=.5)
        corners = observed['corners']
        points = np.asarray(corners).reshape(-1, 2).tolist() if corners is not None else []
        history.append((observed['full'], len(points)))
        return {**original_state(**kwargs), 'resolution': [360, 440],
            'board': {'found': bool(observed['full']), 'corners': points}}
    rig.camera.state = render_state
    rig.start('handeye')
    assert history[0][0] is False and history[0][1] >= 9
    for _ in range(8):
        rig.fresh()
        if rig.c.status == 'moving' and rig.c.hunt.get('status') == 'complete':
            break
        assert rig.c.status == 'hunting'
        if rig.c.hunt['motion_active']:
            rig.settle()
    assert rig.c.status == 'moving'
    assert history[-1] == (True, settings.columns*settings.rows)
    search_moves = rig.moves[:-1]  # Last command is the first cone edge, not a hunt step.
    assert 1 <= len(search_moves) <= 4
    assert rig.c.hunt['steps'] == len(search_moves)
    np.testing.assert_allclose(rig.moves[-1], rig.c.poses[0])
    previous = rig.initial
    for target in search_moves:
        assert np.linalg.norm(target[:3, 3]-previous[:3, 3]) == pytest.approx(.005)
        np.testing.assert_allclose(target[:3, :3], rig.initial[:3, :3], atol=1e-12)
        previous = target
    assert not rig.camera.captures


def test_hunt_wall_time_limit_also_cancels_an_inflight_move(rig):
    rig.start()
    rig.fresh()
    assert not rig.ready[0]
    rig.clock[0] += 60
    rig.c.tick()
    assert rig.c.status == 'error'
    assert 'timeout' in rig.c.last_error
    assert rig.holds and not rig.c.hunt['motion_active']
    assert len(rig.moves) == 1


def test_post_move_frame_grace_starts_after_motion_settles(rig):
    rig.start()
    rig.fresh()
    rig.clock[0] += 4
    rig.camera.fresh = False
    rig.settle()
    assert rig.c.status == 'hunting' and not rig.holds
    rig.camera.fresh = True
    rig.fresh(full=True)
    assert rig.c.status == 'moving'
    assert len(rig.moves) == 2 and not rig.holds
    np.testing.assert_allclose(rig.moves[-1], rig.c.poses[0])
