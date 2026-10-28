"""Offline scan orchestration: measured endpoints, fresh captures and ownership."""
import copy
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_laser_scanning import cloud

from autoweld.scanning import LaserScanCoordinator


class ImmediateExecutor:
    """Deterministic persistence completion without worker or hardware threads."""

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, **_kwargs):
        pass


@pytest.fixture
def rig(tmp_path):
    clock = [100.]
    pose = np.eye(4)
    c = cloud(40)
    state = {'enabled': True, 'fresh': True, 'source_id': c['source_id'],
        'optical_intrinsics_id': c['optical_intrinsics_id'], 'calibration_id': c['calibration_id'],
        'acquisition_session_id': c['acquisition_session_id'], 'generation': c['generation'],
        'handeye': {'source_id': c['source_id'], 'optical_intrinsics_id': c['optical_intrinsics_id'],
                    'tcp_from_camera': np.eye(4).tolist()},
        'intrinsics': {'laser_plane': [0., 0., 1., -.14]}, 'laser': {'cloud': c}}
    events, measured = [], {'fresh': True, 'stationary': True}
    camera = SimpleNamespace(state=lambda **_kwargs: copy.deepcopy(state))
    ready = [True]

    def move(goal):
        events.append(('move', np.array(goal)))

    def hold():
        events.append(('hold', pose.copy()))

    def saved(name):
        goal = pose.copy()
        goal[0, 3] += .02
        events.append(('saved', name, goal))
        return goal

    coordinator = LaserScanCoordinator(camera, lambda: pose.copy(), move, lambda: ready[0], hold,
        saved, tmp_path, feedback=lambda: {**measured, 'measured_at': clock[0]}, can_start=lambda: None,
        validate_pose=lambda goal: events.append(('validate', np.array(goal))), clock=lambda: clock[0],
        executor=ImmediateExecutor())

    def advance(seconds=.5, *, fresh_cloud=True, follow=False):
        clock[0] += seconds
        if follow:
            targets = [event[-1] for event in events if event[0] in ('move', 'saved')]
            if targets:
                pose[:] = targets[-1]
        if fresh_cloud:
            c['read_started'], c['read_completed'] = clock[0], clock[0]
            c['sequence'] += 1
        coordinator.tick()

    result = SimpleNamespace(scan=coordinator, state=state, cloud=c, clock=clock, pose=pose,
        events=events, measured=measured, ready=ready, advance=advance, path=tmp_path)
    yield result
    coordinator.close()


def start(rig, horizontal=0, vertical=0, start_pose='current'):
    return rig.scan.start({'start_pose': start_pose, 'horizontal_mm': horizontal, 'vertical_mm': vertical})


def pump_until(rig, predicate, *, steps=100):
    for _ in range(steps):
        if predicate():
            return
        rig.advance(follow=True)
    raise AssertionError(f'Coordinator did not reach expected state: {rig.scan.state()}')


def capture_partial(rig):
    start(rig, horizontal=2)
    pump_until(rig, lambda: rig.scan._saved.get('captures', 0) >= 1)
    assert rig.scan._saved['points'] == 40


@pytest.mark.parametrize('failure', ['missing', 'weak', 'ambiguous', 'multiframe', 'laser', 'handeye', 'stale'])
def test_start_requires_good_native_stripe_and_calibrations_before_any_movement(rig, failure):
    if failure == 'missing':
        rig.state['laser'].pop('cloud')
    elif failure == 'weak':
        rig.cloud['strength'] = [1.]*40
    elif failure == 'ambiguous':
        other = cloud(40, .18)
        for key in ('xyz_m', 'strength', 'width_px'):
            rig.cloud[key] += other[key]
        rig.cloud['uv'] += (np.asarray(other['uv'])+[300, 0]).tolist()
    elif failure == 'multiframe':
        rig.cloud['frame_product']['kind'] = 'background_subtracted'
    elif failure == 'laser':
        rig.state['intrinsics'].pop('laser_plane')
    elif failure == 'handeye':
        rig.state['handeye']['optical_intrinsics_id'] = 'other'
    else:
        rig.cloud['read_started'] = 98.
    assert not rig.scan.state()['ready']
    with pytest.raises(ValueError):
        start(rig, start_pose='Tube Scanning')
    assert rig.events == []


def test_saved_pose_waits_for_measured_endpoint_and_quiet_before_one_hold(rig):
    start(rig, start_pose='Tube Scanning')
    assert [event[0] for event in rig.events] == ['saved']
    rig.ready[0] = True  # An elapsed planner reference alone cannot establish arrival.
    for _ in range(3):
        rig.advance()
    assert rig.scan.status == 'positioning'
    assert not rig.scan.endpoint_error['converged']
    assert not any(event[0] == 'hold' for event in rig.events)
    rig.pose[:] = rig.events[0][-1]
    rig.measured['stationary'] = False
    rig.advance()
    assert rig.scan.status == 'positioning'
    rig.measured['stationary'] = True
    rig.advance()
    assert rig.scan.status == 'baseline_settling'
    assert len([event for event in rig.events if event[0] == 'hold']) == 1
    rig.advance(.1)
    rig.advance(.1)
    assert len([event for event in rig.events if event[0] == 'hold']) == 1


def test_capture_requires_continuous_quiet_dwell_and_new_acquisition(rig):
    start(rig)
    rig.advance(.1)
    stable_since = rig.scan._stable_since
    rig.advance(.3, fresh_cloud=False)
    assert rig.scan.future is None  # Dwell has not elapsed.
    rig.advance(.11, fresh_cloud=False)
    assert rig.scan.future is None  # Old image remains inside freshness TTL but predates quiet dwell.
    assert rig.scan._stable_since == stable_since
    rig.advance(.01)
    assert rig.scan._pending_kind == 'begin'
    rig.advance(.01)
    rig.advance(.01)
    rig.measured['stationary'] = False
    rig.advance(.5)
    assert rig.scan._stable_since is None and rig.scan._saved['captures'] == 0
    rig.measured['stationary'] = True
    rig.advance(.1)
    rig.advance(.41)
    assert rig.scan._pending_kind == 'append'
    rig.advance(.01)
    rig.advance(.01)
    assert rig.scan.status == 'complete' and rig.scan._saved['captures'] == 1
    assert len([event for event in rig.events if event[0] == 'hold']) == 1


def test_pose_drift_resets_the_quiet_interval_without_retargeting_hold(rig):
    start(rig)
    rig.advance(.1)
    first = rig.scan._stable_since
    rig.pose[0, 3] += .00031
    rig.advance(.3)
    assert rig.scan._stable_since > first
    rig.advance(.2)
    assert rig.scan.future is None
    assert len([event for event in rig.events if event[0] == 'hold']) == 1


def test_raster_uses_one_mm_discrete_moves_and_measured_fixed_optical_orientation(rig):
    rotation = Rotation.from_euler('xyz', [20., -15., 30.], degrees=True).as_matrix()
    rig.pose[:3, :3] = rotation
    start(rig, horizontal=2)
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    moves = [event[1] for event in rig.events if event[0] == 'move']
    assert len(moves) == 3  # Initial centered-raster corner, then two adjacent snapshots.
    previous = np.zeros(3)
    for target in moves:
        np.testing.assert_allclose(target[:3, :3], rotation)
        assert np.linalg.norm(target[:3, 3]-previous) == pytest.approx(.001)
        previous = target[:3, 3]
    assert rig.scan._saved['captures'] == 3
    assert rig.scan._saved['points'] == 120
    assert len([event for event in rig.events if event[0] == 'validate']) == len(moves)+3  # Full grid preflight.


@pytest.mark.parametrize('depth,sign', [(.150, 1.), (.130, -1.)])
def test_depth_correction_moves_along_measured_optical_z_with_correct_sign(rig, depth, sign):
    optical_rotation = Rotation.from_euler('y', 30., degrees=True).as_matrix()
    handeye = np.eye(4)
    handeye[:3, :3] = optical_rotation
    rig.state['handeye']['tcp_from_camera'] = handeye.tolist()
    rig.cloud['xyz_m'] = (np.asarray(rig.cloud['xyz_m'])+[0, 0, depth-.14]).tolist()
    start(rig)
    pump_until(rig, lambda: any(event[0] == 'move' for event in rig.events))
    target = next(event[1] for event in rig.events if event[0] == 'move')
    np.testing.assert_allclose(target[:3, 3], optical_rotation @ [0., 0., sign*.005])
    np.testing.assert_array_equal(target[:3, :3], np.eye(3))
    assert rig.scan._saved['captures'] == 0


@pytest.mark.parametrize('failure', ['missing', 'calibration', 'handeye', 'stale_feedback'])
def test_observation_failure_stops_in_measured_hold_and_retains_partial(rig, failure):
    capture_partial(rig)
    holds_before = len([event for event in rig.events if event[0] == 'hold'])
    if failure == 'missing':
        rig.state['laser'].pop('cloud')
    elif failure == 'calibration':
        rig.state['optical_intrinsics_id'] = 'changed'
    elif failure == 'handeye':
        rig.state['handeye']['tcp_from_camera'][0][3] += .001
    else:
        rig.measured['fresh'] = False
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    assert rig.scan.error
    assert rig.scan._saved['captures'] == 1 and rig.scan._saved['points'] == 40
    assert len([event for event in rig.events if event[0] == 'hold']) == holds_before+1
    assert rig.scan.state()['cloud']['points_base_m']


@pytest.mark.parametrize('operation,hold_count', [('cancel', 1), ('abort_for_shutdown', 0), ('abort_for_cooldown', 0)])
def test_cancel_and_lifecycle_abort_preserve_partial_and_never_reclaim_motion(rig, operation, hold_count):
    capture_partial(rig)
    count = len(rig.events)
    getattr(rig.scan, operation)()
    pump_until(rig, lambda: not rig.scan.busy)
    for _ in range(3):
        rig.advance()
    assert [event[0] for event in rig.events[count:]] == ['hold']*hold_count
    assert rig.scan.status == 'cancelled' and rig.scan._saved['captures'] == 1


def test_archive_error_is_terminal_and_does_not_retry_writes_each_tick(rig, monkeypatch):
    capture_partial(rig)
    calls = []

    def unavailable(**_kwargs):
        calls.append('finish')
        raise OSError('Scan disk full')

    monkeypatch.setattr(rig.scan.archive, 'finish', unavailable)
    rig.scan.cancel()
    rig.advance()
    for _ in range(5):
        rig.advance()
    assert calls == ['finish']
    assert rig.scan.status == 'error' and not rig.scan.busy
    assert 'Scan disk full' in rig.scan.error
    assert rig.scan._saved['captures'] == 1


def test_failed_saved_pose_approach_does_not_relabel_preceding_retained_scan(rig, monkeypatch):
    start(rig)
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    previous = copy.deepcopy(rig.scan._saved)

    def reject(_name):
        raise ValueError('Saved pose path collision')

    monkeypatch.setattr(rig.scan, 'move_saved_pose', reject)
    with pytest.raises(ValueError, match='path collision'):
        start(rig, start_pose='Tube Scanning')
    assert rig.scan.status == 'error' and not rig.scan.busy
    assert rig.scan._saved == previous
    assert rig.scan.archive.state()['status'] == 'complete'
    assert rig.scan.state()['cloud']['points_base_m'] == previous['points_base_m']


def test_duplicate_cancel_preserves_pending_finish_and_does_not_retarget_hold(rig):
    capture_partial(rig)
    rig.scan.cancel()
    future = rig.scan.future
    count = len(rig.events)
    rig.scan.cancel()
    assert rig.scan.future is future
    assert rig.events[count:] == []
    rig.advance()
    rig.scan.cancel()
    assert rig.events[count:] == []
    assert rig.scan.status == 'cancelled' and not rig.scan.busy


def test_camera_generation_can_advance_for_exposure_changes_with_new_native_cloud(rig):
    start(rig)
    rig.state['generation'] += 1
    rig.cloud['generation'] = rig.state['generation']
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan.error is None and rig.scan._saved['captures'] == 1


def test_cloud_generation_must_match_current_camera_generation(rig):
    start(rig)
    rig.state['generation'] += 1
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    assert 'fresh stripe within 15 s' in rig.scan.error
    assert rig.scan._saved['captures'] == 0


def test_captures_reject_orientation_drift_outside_fixed_view_tolerance(rig):
    start(rig)
    pump_until(rig, lambda: rig.scan._saved.get('status') == 'scanning' and rig.scan.future is None)
    rig.pose[:3, :3] = Rotation.from_euler('z', .006).as_matrix()
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    assert 'fixed-view orientation' in rig.scan.error
    assert rig.scan._saved['captures'] == 0


def test_next_move_restores_fixed_baseline_orientation_despite_small_measured_drift(rig):
    start(rig, horizontal=2)
    pump_until(rig, lambda: rig.scan._saved.get('status') == 'scanning' and rig.scan.future is None)
    rig.pose[:3, :3] = Rotation.from_euler('z', .002).as_matrix()
    pump_until(rig, lambda: any(event[0] == 'move' for event in rig.events))
    target = next(event[1] for event in rig.events if event[0] == 'move')
    np.testing.assert_array_equal(target[:3, :3], np.eye(3))


@pytest.mark.parametrize('operation', ['cancel', 'abort_for_shutdown', 'abort_for_cooldown'])
def test_late_finish_error_never_reclaims_hold_after_cancel_or_lifecycle_handoff(rig, monkeypatch, operation):
    capture_partial(rig)

    def unavailable(**_kwargs):
        raise OSError('Archive unavailable after ownership transfer')

    monkeypatch.setattr(rig.scan.archive, 'finish', unavailable)
    getattr(rig.scan, operation)()
    count = len(rig.events)
    rig.advance()
    rig.advance()
    assert rig.events[count:] == []
    assert rig.scan.status == 'error' and not rig.scan.busy
    assert rig.scan._saved['captures'] == 1


def test_executor_submission_error_stops_once_without_finish_retry(rig, monkeypatch):
    calls = []

    def unavailable(*_args, **_kwargs):
        calls.append('submit')
        raise RuntimeError('Persistence executor stopped')

    monkeypatch.setattr(rig.scan.executor, 'submit', unavailable)
    start(rig)
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    count = len(rig.events)
    for _ in range(4):
        rig.advance()
    assert calls == ['submit'] and rig.events[count:] == []
    assert 'executor stopped' in rig.scan.error


def test_close_reports_pending_archive_failure_without_any_motor_command(rig, monkeypatch):
    capture_partial(rig)

    def unavailable(**_kwargs):
        raise OSError('Archive unavailable during close')

    monkeypatch.setattr(rig.scan.archive, 'finish', unavailable)
    rig.scan.abort_for_shutdown()
    count = len(rig.events)
    rig.scan.close()
    assert rig.events[count:] == []
    assert rig.scan.status == 'error' and not rig.scan.busy
    assert 'during close' in rig.scan.error


@pytest.mark.parametrize('transient', ['stale_native', 'missing_cloud', 'generation', 'exposure'])
def test_exposure_change_waits_for_fresh_matching_native_cloud_without_motion_or_capture(rig, transient):
    start(rig)
    rig.advance(.1)
    if transient == 'stale_native':
        rig.cloud['read_started'] = rig.clock[0]-.5
    elif transient == 'missing_cloud':
        rig.state['laser'].pop('cloud')
    elif transient == 'generation':
        rig.state['generation'] += 1
    else:
        rig.state['laser']['pointcloud_status'] = {'auto_exposure_status': {'state': 'settling'}}
        rig.cloud['strength'] = [1.]*40
    count = len(rig.events)
    rig.advance(.5, fresh_cloud=transient != 'stale_native')
    rig.advance(.5, fresh_cloud=transient != 'stale_native')
    assert rig.scan.status == 'baseline_settling' and rig.scan.future is None
    assert rig.events[count:] == [] and rig.scan._saved['captures'] == 0
    rig.state['fresh'] = True
    rig.state['laser']['cloud'] = rig.cloud
    rig.cloud['generation'] = rig.state['generation']
    rig.cloud['strength'] = [80.]*40
    rig.state['laser']['pointcloud_status'] = {'auto_exposure_status': {'state': 'stable'}}
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan.error is None and rig.scan._saved['captures'] == 1


def test_brief_stale_sdk_feedback_resets_quiet_window_without_motion_or_capture(rig):
    start(rig)
    rig.state['fresh'] = False
    rig.measured['fresh'] = False
    events = len(rig.events)
    rig.advance(.1)
    assert rig.scan.status == 'baseline_settling'
    assert rig.scan._stable_since is None
    assert len(rig.events) == events and rig.scan.future is None
    rig.measured['fresh'] = True
    rig.state['fresh'] = True
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['captures'] == 1


def test_sustained_stale_sdk_feedback_times_out_without_moving(rig):
    start(rig)
    rig.measured['fresh'] = False
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    assert not any(event[0] == 'move' for event in rig.events)
    assert rig.scan._saved['captures'] == 0


def test_grid_accepts_service_endpoint_tolerance_without_replanning_subtolerance_error(rig):
    start(rig)
    rig.pose[0, 3] += .0004
    from autoweld.laser_scanning import segment_object
    segment = segment_object(rig.cloud)
    before = len(rig.events)
    assert rig.scan._at_offset(segment)
    assert len(rig.events) == before


def test_xy_positioning_has_a_finite_move_budget(rig):
    start(rig, horizontal=2, vertical=2)
    rig.scan._alignment_moves = 240
    from autoweld.laser_scanning import segment_object
    with pytest.raises(ValueError, match='bounded positioning'):
        rig.scan._at_offset(segment_object(rig.cloud))


def test_native_stripe_freshness_is_independent_of_checkerboard_observation_freshness(rig):
    rig.state['fresh'] = False
    assert rig.scan.state()['ready']
    start(rig)
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['captures'] == 1 and rig.scan.error is None


def test_stale_native_cloud_never_qualifies_despite_fresh_checkerboard_observation(rig):
    rig.state['fresh'] = True
    rig.cloud['read_started'] = rig.clock[0]-2.
    assert not rig.scan.state()['ready']
    with pytest.raises(ValueError, match='acquisition is stale'):
        start(rig)
    assert not rig.events


def test_hundred_mm_grid_is_centered_and_entirely_preflighted_one_pose_per_tick_before_xy_motion(rig):
    start(rig, horizontal=100, vertical=100)
    pump_until(rig, lambda: rig.scan.status == 'preflighting')
    previous = len([event for event in rig.events if event[0] == 'validate'])
    while rig.scan.status == 'preflighting':
        rig.advance(.001)
        count = len([event for event in rig.events if event[0] == 'validate'])
        assert count-previous <= 1
        previous = count
    assert rig.scan.total == 10201 and rig.scan._preflight_complete
    assert not any(event[0] == 'move' for event in rig.events)
    poses = [event[1] for event in rig.events if event[0] == 'validate']
    assert len(poses) == 10201
    xy = np.array([pose[:2, 3] for pose in poses])
    np.testing.assert_allclose(xy.min(0), [-.05, -.05])
    np.testing.assert_allclose(xy.max(0), [.05, .05])
    assert len(set(map(tuple, xy))) == 10201
    assert all(np.array_equal(pose[:3, :3], np.eye(3)) for pose in poses)
    geometry = rig.scan.state()['geometry']
    assert np.asarray(geometry['outline_base_m']).shape == (5, 3)
    rig.scan.cancel()


def test_unreachable_corner_fails_before_xy_motion_or_replacing_saved_archive(rig, monkeypatch):
    start(rig)
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    prior = copy.deepcopy(rig.scan._saved)
    def reject(goal):
        rig.events.append(('validate', np.array(goal)))
        if goal[0, 3] > .04:
            raise ValueError('Installed J6 bound blocks square corner')
    monkeypatch.setattr(rig.scan, 'validate_pose', reject)
    start(rig, horizontal=100, vertical=100)
    count = len(rig.events)
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    assert not any(event[0] == 'move' for event in rig.events[count:])
    assert 'J6 bound' in rig.scan.error
    assert rig.scan._saved == prior
    assert rig.scan._preflight_index < 5


def test_center_depth_is_corrected_before_xy_preflight_without_redefining_grid_center(rig):
    native = np.asarray(rig.cloud['xyz_m']).copy()
    rig.cloud['xyz_m'] = (native+[0, 0, -.018]).tolist()
    start(rig, horizontal=100, vertical=100)
    original = rig.scan.return_pose.copy()
    for _ in range(80):
        if rig.scan.status == 'moving':
            rig.pose[:] = [event[1] for event in rig.events if event[0] == 'move'][-1]
        rig.cloud['xyz_m'] = (native+[0, 0, -.018-rig.pose[2, 3]]).tolist()
        rig.advance(.5)
        if rig.scan.status == 'preflighting':
            break
    else:
        raise AssertionError(rig.scan.state())
    moves = [event[1] for event in rig.events if event[0] == 'move']
    assert moves and all(np.linalg.norm(goal[:2, 3]) == 0 for goal in moves)
    assert rig.scan._safe_depth_offset == pytest.approx(-.018)
    np.testing.assert_array_equal(rig.scan.return_pose, original)
    np.testing.assert_array_equal(rig.scan._baseline_optical[:2, 3], [0., 0.])
    preflight = [event[1] for event in rig.events if event[0] == 'validate'][-1]
    assert preflight[2, 3] == pytest.approx(-.018)
    rig.scan.cancel()


@pytest.mark.parametrize('failure', ['blank', 'weak', 'ambiguous'])
def test_no_return_or_rejected_cells_are_durable_and_grid_continues_at_last_safe_depth(rig, failure):
    capture_partial(rig)
    accepted = copy.deepcopy(rig.scan._saved['points_base_m'])
    if failure == 'blank':
        for key in ('xyz_m', 'uv', 'strength', 'width_px'):
            rig.cloud[key] = []
        rig.cloud['rejected'] = {'candidates': 0}
    elif failure == 'weak':
        rig.cloud['strength'] = [1.]*40
    else:
        rig.cloud['rejected'] = {'candidates': 80, 'ambiguous': 70}
    count = len(rig.events)
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    state = rig.scan.state()
    assert state['visited'] == 3 and state['accepted'] == 1
    assert state['no_return'] == (2 if failure == 'blank' else 0)
    assert state['rejected'] == (0 if failure == 'blank' else 2)
    assert rig.scan._saved['points_base_m'] == accepted
    for event in rig.events[count:]:
        if event[0] == 'move':
            assert event[1][2, 3] == 0  # No depth direction invented from a missing/weak stripe.
    assert rig.scan.dispatch({'action': 'scan_export'})['journal']


def test_surface_patch_prediction_follows_image_plane_grid_across_blank_span(rig):
    start(rig, horizontal=100)
    from autoweld.laser_scanning import segment_object
    rig.scan._update_segment(segment_object(rig.cloud), np.eye(4))
    rig.pose[0, 3] = .04
    predicted = rig.scan._previous_centroid(rig.pose)
    assert predicted[0] == pytest.approx(0.)
    result = segment_object(rig.cloud, predicted)
    assert result['quality']['good']
    rig.scan.cancel()


def test_cancelled_scan_resumes_same_grid_and_retains_captures_with_explicit_new_camera_session(rig):
    capture_partial(rig)
    scan_id = rig.scan._saved['scan_id']
    baseline = rig.scan.return_pose.copy()
    rig.scan.cancel()
    pump_until(rig, lambda: not rig.scan.busy)
    assert rig.scan.state()['resume_available']
    rig.state['acquisition_session_id'] = rig.cloud['acquisition_session_id'] = 'explicit-resumed-session'
    rig.scan.dispatch({'action': 'scan_resume'})
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['scan_id'] == scan_id
    assert rig.scan._saved['cells_visited'] == 3 and rig.scan._saved['captures'] == 3
    assert rig.scan._saved['points'] == 120
    np.testing.assert_array_equal(rig.scan.return_pose, baseline)
    assert not rig.scan.state()['resume_available']


def test_resume_rejects_robot_outside_original_view_and_never_replays_target(rig):
    capture_partial(rig)
    rig.scan.abort_for_cooldown()
    pump_until(rig, lambda: not rig.scan.busy)
    rig.pose[:3, :3] = Rotation.from_euler('z', .1).as_matrix()
    count = len(rig.events)
    with pytest.raises(ValueError, match='Tube Scanning view'):
        rig.scan.resume()
    assert rig.events[count:] == []
    assert rig.scan._saved['captures'] == 1


def test_restarted_coordinator_restores_incomplete_grid_without_any_robot_command(rig):
    capture_partial(rig)
    rig.scan.cancel()
    pump_until(rig, lambda: not rig.scan.busy)
    calls = []
    restored = LaserScanCoordinator(rig.scan.camera, lambda: rig.pose.copy(),
        lambda *args: calls.append('move'), lambda: True, lambda: calls.append('hold'),
        lambda *args: calls.append('saved'), rig.path,
        feedback=lambda: {'fresh': True, 'stationary': True, 'measured_at': rig.clock[0]},
        can_start=lambda: None, executor=ImmediateExecutor(), clock=lambda: rig.clock[0])
    try:
        assert calls == []
        assert restored.index == 1 and restored.total == 3
        assert restored.state()['resume_available']
        assert restored._saved['points'] == 40
    finally:
        restored.close()


@pytest.mark.parametrize('outcome', ['accepted', 'no_return', 'rejected'])
def test_capture_waits_for_sdk_feedback_after_native_frame_without_extra_hold(rig, outcome):
    start(rig)
    pump_until(rig, lambda: rig.scan._saved.get('status') == 'scanning' and rig.scan.future is None)
    if outcome == 'no_return':
        for key in ('xyz_m', 'uv', 'strength', 'width_px'):
            rig.cloud[key] = []
        rig.cloud['rejected'] = {'candidates': 0}
    elif outcome == 'rejected':
        rig.cloud['strength'] = [1.]*40
    lag = [True]
    rig.scan.feedback = lambda: {'fresh': True, 'stationary': True,
        'measured_at': rig.clock[0]-(.08 if lag[0] else 0.)}
    holds = len([event for event in rig.events if event[0] == 'hold'])
    for _ in range(4):
        rig.advance(.5)
    assert rig.scan.future is None and rig.scan._saved['cells_visited'] == 0
    assert rig.scan.error is None
    assert 'after the native camera acquisition' in rig.scan.message
    lag[0] = False
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['cells_visited'] == 1
    assert len([event for event in rig.events if event[0] == 'hold']) == holds
    assert rig.scan._saved['captures'] == (1 if outcome == 'accepted' else 0)


def test_freshness_lost_after_quiet_check_resets_dwell_without_capture_or_extra_hold(rig):
    start(rig)
    pump_until(rig, lambda: rig.scan._saved.get('status') == 'scanning' and rig.scan.future is None)
    rig.advance(.1)
    holds = len([event for event in rig.events if event[0] == 'hold'])
    samples = iter([{'fresh': True, 'stationary': True, 'measured_at': rig.clock[0]+.5},
                    {'fresh': False, 'stationary': True, 'measured_at': rig.clock[0]+.5}])
    rig.scan.feedback = lambda: next(samples)
    rig.advance(.5)
    assert rig.scan.future is None and rig.scan._stable_since is None
    assert rig.scan._saved['cells_visited'] == 0
    assert len([event for event in rig.events if event[0] == 'hold']) == holds
    rig.scan.feedback = lambda: {'fresh': True, 'stationary': True, 'measured_at': rig.clock[0]}
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['captures'] == 1


@pytest.mark.parametrize('roll', [-180., -90., 37., 180.])
def test_fixed_optical_roll_preserves_camera_origin_axis_and_original_raster_axes(rig, roll):
    from autoweld.calibration import transform
    nominal = np.eye(4)
    nominal[:3, :3] = Rotation.from_euler('xyz', [20., -10., 35.], degrees=True).as_matrix()
    nominal[:3, 3] = [.1, -.2, .3]
    rig.pose[:] = nominal
    handeye = np.eye(4)
    handeye[:3, :3] = Rotation.from_euler('xyz', [5., -7., 9.], degrees=True).as_matrix()
    handeye[:3, 3] = [.025, .017, -.012]
    rig.state['handeye']['tcp_from_camera'] = handeye.tolist()
    original = nominal @ handeye
    rig.scan.start({'start_pose': 'current', 'horizontal_mm': 2, 'vertical_mm': 2, 'optical_roll_deg': roll})
    assert rig.scan.status == 'moving'
    target = [event[1] for event in rig.events if event[0] == 'move'][-1]
    rolled = target @ handeye
    np.testing.assert_allclose(rolled[:3, 3], original[:3, 3], atol=1e-14)
    np.testing.assert_allclose(rolled[:3, 2], original[:3, 2], atol=1e-14)
    expected = original[:3, :3] @ Rotation.from_euler('z', roll, degrees=True).as_matrix()
    np.testing.assert_allclose(rolled[:3, :3], expected, atol=1e-14)
    np.testing.assert_array_equal(rig.scan.return_pose, nominal)
    grid = transform(rig.scan._grid_pose([.001, -.001], depth=-.018)) @ handeye
    np.testing.assert_allclose(grid[:3, 3], original[:3, 3]+original[:3, :3] @ [.001, -.001, -.018])
    np.testing.assert_allclose(grid[:3, :3], expected)
    assert not rig.scan._saved.get('captures')
    rig.scan.cancel()


@pytest.mark.parametrize('roll', [True, None, np.nan, np.inf, 181, -181, '90'])
def test_invalid_roll_is_rejected_before_any_hold_or_motion(rig, roll):
    with pytest.raises(ValueError, match='optical roll'):
        rig.scan.start({'start_pose': 'current', 'optical_roll_deg': roll})
    assert not rig.events


def test_fixed_roll_is_preserved_across_all_grid_poses_archived_rotation_and_resume(rig):
    rig.scan.start({'start_pose': 'current', 'horizontal_mm': 2, 'vertical_mm': 0, 'optical_roll_deg': 90.})
    pump_until(rig, lambda: rig.scan._saved.get('captures', 0) >= 1)
    expected = Rotation.from_euler('z', 90., degrees=True).as_matrix()
    for event in rig.events:
        if event[0] in ('move', 'validate'):
            np.testing.assert_allclose(event[1][:3, :3], expected, atol=1e-14)
    np.testing.assert_allclose(rig.scan._saved['metadata']['capture_rotation'], expected, atol=1e-14)
    np.testing.assert_array_equal(rig.scan._saved['metadata']['baseline_pose'], np.eye(4))
    scan_id = rig.scan._saved['scan_id']
    rig.scan.cancel()
    pump_until(rig, lambda: not rig.scan.busy)
    rig.scan.resume()
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['scan_id'] == scan_id
    assert rig.scan._saved['cells_visited'] == 3
    assert rig.scan.optical_roll_deg == 90.
    for event in rig.events:
        if event[0] in ('move', 'validate'):
            np.testing.assert_allclose(event[1][:3, :3], expected, atol=1e-14)


@pytest.mark.parametrize('bad', ['saturated', 'blank', 'weak'])
def test_baseline_waits_for_active_smart_exposure_bad_frame_without_motion_or_capture(rig, bad):
    start(rig)
    if bad == 'saturated':
        rig.cloud['rejected'] = {'candidates': 80, 'saturated': 70}
    elif bad == 'blank':
        for key in ('xyz_m', 'uv', 'strength', 'width_px'):
            rig.cloud[key] = []
    else:
        rig.cloud['strength'] = [1.]*40
    rig.state['laser']['pointcloud_status'] = {'auto_exposure': {'enabled': True},
        'auto_exposure_status': {'state': 'searching'}}
    count = len(rig.events)
    for _ in range(4):
        rig.advance(.5)
    assert rig.scan.status == 'baseline_settling' and rig.scan.future is None
    assert rig.events[count:] == []
    assert rig.scan.error is None and rig.scan._saved['captures'] == 0
    assert 'smart stripe exposure' in rig.scan.message
    rig.cloud.update(cloud(40))
    rig.cloud['read_started'] = rig.clock[0]
    pump_until(rig, lambda: rig.scan.status == 'complete' and not rig.scan.busy)
    assert rig.scan._saved['captures'] == 1


def test_baseline_smart_exposure_wait_remains_bounded_without_new_hold(rig):
    start(rig)
    rig.cloud['rejected'] = {'candidates': 80, 'saturated': 70}
    rig.state['laser']['pointcloud_status'] = {'auto_exposure': {'enabled': True},
        'auto_exposure_status': {'state': 'searching'}}
    pump_until(rig, lambda: rig.scan.status == 'error' and not rig.scan.busy)
    assert 'within 15 s' in rig.scan.error
    assert not any(event[0] == 'move' for event in rig.events)
    assert len([event for event in rig.events if event[0] == 'hold']) == 2


def test_fixed_roll_resume_after_go_to_tube_reapplies_roll_at_fresh_camera_origin(rig):
    rig.scan.start({'start_pose': 'current', 'horizontal_mm': 2, 'optical_roll_deg': -15.})
    pump_until(rig, lambda: rig.scan._saved.get('captures', 0) >= 1)
    rig.scan.abort_for_cooldown()
    pump_until(rig, lambda: not rig.scan.busy)
    baseline = rig.scan.return_pose.copy()
    rig.pose[:] = baseline
    rig.pose[:3, 3] += [.0004, -.0002, -.003]
    measured_origin = (rig.pose @ rig.scan._handeye)[:3, 3].copy()
    count = len(rig.events)
    rig.scan.resume()
    assert rig.scan.status == 'moving'
    moves = [event[1] for event in rig.events[count:] if event[0] == 'move']
    assert len(moves) == 1
    np.testing.assert_allclose((moves[0] @ rig.scan._handeye)[:3, 3], measured_origin, atol=1e-14)
    np.testing.assert_allclose(moves[0][:3, :3], rig.scan.view_pose[:3, :3], atol=1e-14)
    np.testing.assert_array_equal(rig.scan.return_pose, baseline)
    assert rig.scan.index == 1
    rig.scan.cancel()


def test_fixed_roll_resume_does_not_reorient_an_already_correct_working_view(rig):
    rig.scan.start({'start_pose': 'current', 'horizontal_mm': 2, 'optical_roll_deg': -15.})
    pump_until(rig, lambda: rig.scan._saved.get('captures', 0) >= 1)
    rig.scan.cancel()
    pump_until(rig, lambda: not rig.scan.busy)
    count = len(rig.events)
    rig.scan.resume()
    assert rig.scan.status == 'baseline_settling'
    assert not any(event[0] == 'move' for event in rig.events[count:])
    assert len([event for event in rig.events[count:] if event[0] == 'hold']) == 1
    rig.scan.cancel()


@pytest.mark.parametrize('wrong', ['rotation', 'center', 'depth'])
def test_fixed_roll_resume_refuses_untrusted_original_pose_before_any_command(rig, wrong):
    rig.scan.start({'start_pose': 'current', 'horizontal_mm': 2, 'optical_roll_deg': -15.})
    pump_until(rig, lambda: rig.scan._saved.get('captures', 0) >= 1)
    rig.scan.abort_for_cooldown()
    pump_until(rig, lambda: not rig.scan.busy)
    rig.pose[:] = rig.scan.return_pose
    if wrong == 'rotation':
        rig.pose[:3, :3] = Rotation.from_euler('y', .01).as_matrix()
    elif wrong == 'center':
        rig.pose[0, 3] += .0011
    else:
        rig.pose[2, 3] += .061
    count = len(rig.events)
    with pytest.raises(ValueError, match='Tube Scanning view'):
        rig.scan.resume()
    assert rig.events[count:] == []


def test_next_xy_move_starts_without_pre_capture_dwell_but_arrival_capture_still_waits(rig):
    capture_partial(rig)
    assert rig.scan.index == 1 and rig.scan.status == 'settling'
    holds = len([event for event in rig.events if event[0] == 'hold'])
    rig.advance(.001)
    assert rig.scan.status == 'moving'
    assert len([event for event in rig.events if event[0] == 'hold']) == holds
    target = [event[1] for event in rig.events if event[0] == 'move'][-1]
    rig.pose[:] = target
    rig.advance(.001)
    assert rig.scan.status == 'settling'
    assert len([event for event in rig.events if event[0] == 'hold']) == holds+1
    rig.advance(.001)
    rig.advance(.2)
    assert rig.scan.future is None and rig.scan._saved['cells_visited'] == 1
    rig.advance(.201)
    assert rig.scan._pending_kind == 'append'
    assert len([event for event in rig.events if event[0] == 'hold']) == holds+1


@pytest.mark.parametrize('blocked', ['sdk_stale', 'sdk_moving', 'not_ready', 'camera_missing', 'generation'])
def test_fast_pre_move_waits_for_current_native_cloud_and_fresh_stationary_ready_sdk(rig, blocked):
    capture_partial(rig)
    count = len(rig.events)
    if blocked == 'sdk_stale':
        rig.measured['fresh'] = False
    elif blocked == 'sdk_moving':
        rig.measured['stationary'] = False
    elif blocked == 'not_ready':
        rig.ready[0] = False
    elif blocked == 'camera_missing':
        rig.state['laser'].pop('cloud')
    else:
        rig.state['generation'] += 1
    rig.advance(.001)
    assert rig.scan.status == 'settling' and rig.scan.future is None
    assert rig.events[count:] == []
    assert rig.scan._saved['cells_visited'] == 1


def test_fast_pre_move_rejects_stale_cloud_without_commanding_next_target(rig):
    capture_partial(rig)
    count = len(rig.events)
    rig.cloud['read_started'] = rig.clock[0]-1.01
    rig.advance(.001, fresh_cloud=False)
    assert rig.scan.status == 'error'
    assert not any(event[0] == 'move' for event in rig.events[count:])
    assert 'stale' in rig.scan.error
    pump_until(rig, lambda: not rig.scan.busy)
    assert rig.scan._saved['cells_visited'] == 1


def test_first_full_grid_corner_uses_one_normal_positioning_waypoint_after_preflight(rig):
    start(rig, horizontal=100, vertical=100)
    pump_until(rig, lambda: rig.scan.status == 'preflighting')
    while rig.scan.status == 'preflighting':
        rig.advance(.001)
    pump_until(rig, lambda: any(event[0] == 'move' for event in rig.events))
    moves = [event[1] for event in rig.events if event[0] == 'move']
    assert len(moves) == 1
    np.testing.assert_allclose(moves[0][:3, 3], [-.05, -.05, 0.])
    assert rig.scan._preflight_complete
    assert rig.scan._saved['cells_visited'] == 0
    assert rig.scan._grid_approach
    rig.scan.cancel()


def test_resume_first_remaining_cell_uses_positioning_waypoint_then_adjacent_cells_stay_one_mm(rig):
    start(rig, horizontal=10)
    pump_until(rig, lambda: rig.scan._saved.get('captures', 0) >= 1)
    rig.scan.cancel()
    pump_until(rig, lambda: not rig.scan.busy)
    rig.pose[:] = rig.scan.return_pose
    count = len(rig.events)
    rig.scan.resume()
    pump_until(rig, lambda: any(event[0] == 'move' for event in rig.events[count:]))
    target = [event[1] for event in rig.events[count:] if event[0] == 'move'][-1]
    np.testing.assert_allclose(target[:3, 3], [-.004, 0., 0.])
    assert rig.scan.index == 1 and rig.scan._grid_approach
    pump_until(rig, lambda: rig.scan._saved['cells_visited'] >= 2)
    assert not rig.scan._grid_approach
    position = rig.pose[:3, 3].copy()
    count = len(rig.events)
    rig.advance(.001)
    moves = [event[1] for event in rig.events[count:] if event[0] == 'move']
    assert len(moves) == 1
    assert np.linalg.norm(moves[0][:3, 3]-position) == pytest.approx(.001)
    rig.scan.cancel()


def test_next_nominal_cell_is_not_shortened_by_previous_measured_hold_error(rig):
    capture_partial(rig)
    rig.pose[:3, 3] = [-.0014, .0002, 0.]
    count = len(rig.events)
    rig.advance(.001)
    moves = [event[1] for event in rig.events[count:] if event[0] == 'move']
    assert len(moves) == 1
    np.testing.assert_allclose(moves[0][:3, 3], [0., 0., 0.], atol=1e-14)
    assert np.linalg.norm(moves[0][:3, 3]-rig.pose[:3, 3]) > .001
    rig.scan.cancel()


def test_one_nominal_profile_captures_with_existing_half_mm_measured_endpoint_error(rig):
    capture_partial(rig)
    rig.pose[:3, 3] = [-.0014, .0002, 0.]
    count = len(rig.events)
    rig.advance(.001)
    target = [event[1] for event in rig.events[count:] if event[0] == 'move'][-1]
    rig.pose[:] = target
    rig.pose[:3, 3] += [-.00035, .0002, 0.]
    measured_pose = rig.pose.copy()
    for _ in range(12):
        rig.advance(.1)
        if rig.scan._saved['cells_visited'] == 2:
            break
    else:
        raise AssertionError(rig.scan.state())
    assert len([event for event in rig.events[count:] if event[0] == 'move']) == 1
    assert rig.scan._saved['captures'] == 2
    np.testing.assert_array_equal(rig.scan._saved['last_cell']['base_from_tcp'], measured_pose)
    rig.scan.cancel()


def test_systematic_pose_lag_keeps_one_profile_per_nominal_cell_including_row_turns(rig):
    start(rig, horizontal=2, vertical=2)
    for _ in range(150):
        if rig.scan.status == 'moving':
            rig.pose[:] = [event[1] for event in rig.events if event[0] == 'move'][-1]
            rig.pose[:3, 3] += [-.0003, .0002, 0.]
        rig.advance(.5)
        if rig.scan.status == 'complete' and not rig.scan.busy:
            break
    else:
        raise AssertionError(rig.scan.state())
    moves = [event[1] for event in rig.events if event[0] == 'move']
    assert len(moves) == 9
    xy = np.array([pose[:2, 3] for pose in moves])
    np.testing.assert_allclose(xy, rig.scan._offsets, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(np.diff(xy, axis=0), axis=1), .001, atol=1e-14)
    assert rig.scan._saved['cells_visited'] == rig.scan._saved['captures'] == 9


def test_hundred_mm_corner_tracks_illuminated_patch_without_fixed_center_anchor_loss(rig):
    start(rig, horizontal=100, vertical=100)
    from autoweld.laser_scanning import segment_object
    measured = segment_object(rig.cloud)
    rig.scan._update_segment(measured, rig.pose.copy())
    original_world = rig.scan._object_world.copy()
    rig.pose[:3, 3] = [-.05, -.05, 0.]
    prediction = rig.scan._previous_centroid(rig.pose)
    np.testing.assert_allclose(prediction, measured['centroid_camera_m'], atol=1e-14)
    result = segment_object(rig.cloud, prediction)
    assert result['quality']['good']
    np.testing.assert_array_equal(rig.scan._object_world, original_world)  # Prediction is not fabricated measurement.
    rig.scan.cancel()


def test_surface_patch_prediction_excludes_depth_motion_and_preserves_native_depth_continuity(rig):
    start(rig, horizontal=100, vertical=100)
    from autoweld.laser_scanning import segment_object
    rig.scan._update_segment(segment_object(rig.cloud), rig.pose.copy())
    rig.pose[:3, 3] = [.04, -.05, -.006]
    predicted = rig.scan._previous_centroid(rig.pose)
    np.testing.assert_allclose(predicted, [0., 0., .146], atol=1e-14)
    returned = copy.deepcopy(rig.cloud)
    returned['xyz_m'] = (np.asarray(returned['xyz_m'])+[0., 0., .006]).tolist()
    assert segment_object(returned, predicted)['quality']['good']
    returned['xyz_m'] = (np.asarray(returned['xyz_m'])+[0., 0., .025]).tolist()
    with pytest.raises(ValueError, match='Tracked object stripe was lost'):
        segment_object(returned, predicted)
    rig.scan.cancel()


def test_surface_patch_prediction_uses_fixed_optical_plane_with_roll_and_measured_handeye(rig):
    from autoweld.laser_scanning import segment_object
    handeye = np.eye(4)
    handeye[:3, :3] = Rotation.from_euler('xyz', [10., -25., 30.], degrees=True).as_matrix()
    handeye[:3, 3] = [.025, -.01, .015]
    rig.state['handeye']['tcp_from_camera'] = handeye.tolist()
    rig.scan.start({'start_pose': 'current', 'horizontal_mm': 100, 'vertical_mm': 100, 'optical_roll_deg': -15.})
    rig.pose[:] = rig.scan.view_pose
    measured = segment_object(rig.cloud)
    rig.scan._update_segment(measured, rig.pose.copy())
    optical = rig.pose @ handeye
    original = optical.copy()
    optical[:3, 3] += rig.scan._baseline_optical[:3, :3] @ [.05, -.05, -.004]
    rig.pose[:] = optical @ np.linalg.inv(handeye)
    predicted = rig.scan._previous_centroid(rig.pose)
    expected = np.asarray(measured['centroid_camera_m'])+original[:3, :3].T @ (rig.scan._baseline_optical[:3, 2]*.004)
    np.testing.assert_allclose(predicted, expected, atol=1e-14)
    rig.scan.cancel()


def test_adjacent_patch_updates_and_blank_gap_do_not_reanchor_to_baseline_center(rig):
    start(rig, horizontal=100, vertical=100)
    from autoweld.laser_scanning import segment_object
    native = copy.deepcopy(rig.cloud)
    first = segment_object(native)
    rig.scan._update_segment(first, rig.pose.copy())
    for x in (-.05, -.049, -.048):
        rig.pose[:3, 3] = [x, -.05, 0.]
        native['xyz_m'] = (np.asarray(rig.cloud['xyz_m'])+[.0002*(x/.001+50), 0., 0.]).tolist()
        predicted = rig.scan._previous_centroid(rig.pose)
        segment = segment_object(native, predicted)
        rig.scan._update_segment(segment, rig.pose.copy())
    last_native = segment['centroid_camera_m']
    rig.pose[:3, 3] = [.045, -.049, 0.]
    predicted = rig.scan._previous_centroid(rig.pose)
    np.testing.assert_allclose(predicted, last_native, atol=1e-14)
    assert segment_object(native, predicted)['quality']['good']
    rig.scan.cancel()


def test_surface_patch_prediction_keeps_existing_ambiguity_and_native_quality_rejections(rig):
    start(rig, horizontal=100)
    from autoweld.laser_scanning import segment_object
    rig.scan._update_segment(segment_object(rig.cloud), rig.pose.copy())
    rig.pose[:3, 3] = [-.05, 0., 0.]
    predicted = rig.scan._previous_centroid(rig.pose)
    native = copy.deepcopy(rig.cloud)
    native['rejected'] = {'candidates': 80, 'ambiguous': 70}
    with pytest.raises(ValueError, match='saturated or ambiguous'):
        segment_object(native, predicted)
    native = copy.deepcopy(rig.cloud)
    other = cloud(40)
    other['xyz_m'] = (np.asarray(other['xyz_m'])+[.001, 0., 0.]).tolist()
    for key in ('xyz_m', 'strength', 'width_px'):
        native[key] += other[key]
    native['uv'] += (np.asarray(other['uv'])+[300., 0.]).tolist()
    with pytest.raises(ValueError, match='association is ambiguous'):
        segment_object(native, predicted)
    rig.scan.cancel()
