"""Offline laser coordination: mocked native vision, no SDK, camera or model."""

import copy
import json
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.laser_calibration import (
    LaserCalibrationCoordinator,
    camera_points_to_j6,
    j6_from_camera,
    transform_cloud,
)


def pose(xyz=(0., 0., 0.), degrees=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', degrees, degrees=True).as_matrix()
    result[:3, 3] = xyz
    return result


def handeye(matrix=None):
    return dict(tcp_from_camera=(np.eye(4) if matrix is None else matrix).tolist(),
                units='m', coordinate_frame='tcp', source_id='camera', calibration_id='optics')


class DeferredExecutor:
    def __init__(self):
        self.queue = []

    def submit(self, function, *args):
        future = Future()
        self.queue.append((future, function, args))
        return future

    def run(self):
        future, function, args = self.queue.pop(0)
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)

    def shutdown(self, **kwargs):
        while self.queue:
            self.run()


class Camera:
    def __init__(self, clock, correction):
        self.clock = clock
        self.correction = correction
        self.enabled, self.full, self.fresh = True, True, True
        self.source, self.optical, self.session = 'camera', 'optics', 'acquisition'
        self.sequence = self.generation = self.count = 0
        self.stamp = clock[0]
        self.corners = [[700., 300.], [740., 300.], [700., 340.], [740., 340.]]
        self.requests = []
        self.reject = set()
        self.stale_capture = False
        self.solve_error = None
        self.capture_errors = {}

    def frame(self):
        self.sequence += 1
        self.stamp = self.clock[0]

    def state(self, **kwargs):
        return dict(enabled=self.enabled, fresh=self.fresh, source_id=self.source,
            optical_intrinsics_id=self.optical, calibration_id='plane-dependent-identity',
            acquisition_session_id=self.session, generation=self.generation, sequence=self.sequence,
            read_started=self.stamp, resolution=[1000, 800], intrinsics={'installed': True},
            handeye=handeye(self.correction), board={'found': self.full, 'corners': self.corners},
            laser={'captures': self.count})

    def laser_command(self, request):
        self.requests.append(copy.deepcopy(request))
        action = request['action']
        if action == 'laser_new':
            self.count = 0
        elif action in ('laser_capture', 'laser_check'):
            errors = self.capture_errors.get(request.get('metadata', {}).get('view_index'), [])
            if errors:
                raise ValueError(errors.pop(0))
            if request.get('metadata', {}).get('view_index') in self.reject:
                raise ValueError('Native stripe or duplicate-view quality rejection')
            self.generation += 3  # Native board/stripe/verification exposure bracket.
            observation = self.state()
            if self.stale_capture:
                observation['read_started'] = request['not_before_monotonic']-.001
            if action == 'laser_capture':
                self.count += 1
            return {'count': self.count, 'observation': observation, 'diagnostic': {'line_rms_mm': .05}}
        elif action == 'laser_solve':
            if self.solve_error:
                raise ValueError(self.solve_error)
            return {'views': self.count, 'plane': [0., 1., 0., -10.],
                    'quality': {'rms_mm': .08, 'normal_span_deg': 14., 'depth_span_mm': 4.}}
        return {'ok': True}


@pytest.fixture
def rig():
    clock, current, quiet = [100.], [pose((.2, -.1, .3), (10, -20, 40))], [True]
    correction = pose((.026, .023, -.013), (4, 6, 120))
    camera, executor = Camera(clock, correction), DeferredExecutor()
    moves, holds, saved, validated, preemptions = [], [], [], [], []
    def move(target, **kwargs):
        assert kwargs == {'automatic': True}
        moves.append(np.asarray(target).copy())
        quiet[0] = False
    def go_saved(name):
        saved.append(name)
        quiet[0] = False
        return current[0].copy()
    def hold():
        holds.append(current[0].copy())
        quiet[0] = True
    c = LaserCalibrationCoordinator(camera, lambda: current[0], move, lambda: quiet[0], hold, go_saved,
        flange_from_nominal_tcp=pose((-.0695, .0321, -.0688), (180, 0, 90)),
        validate_pose=lambda p: validated.append(p.copy()), preempt=lambda: preemptions.append(True),
        clock=lambda: clock[0], executor=executor)
    def work():
        executor.run()
        c.tick()
    def fresh(dt=.5):
        clock[0] += dt
        camera.frame()
        c.tick()
    def settle():
        current[0] = moves[-1].copy()
        quiet[0] = True
        c.tick()
    def start():
        c.start(go_to_calibration_pose=False)
        assert c.status == 'preparing' and not camera.requests
        work()
        assert c.status == 'preparation_settling'
        fresh()
        assert c.status == 'hunting'
        fresh()
        assert c.status == 'moving' and len(moves) == 1
    yield SimpleNamespace(c=c, camera=camera, executor=executor, clock=clock, current=current, quiet=quiet,
        correction=correction, moves=moves, holds=holds, saved=saved, validated=validated,
        preemptions=preemptions, work=work, fresh=fresh, settle=settle, start=start)
    c.close()


def test_auto_goes_to_saved_pose_through_service_before_native_preparation(rig):
    rig.c.start()
    assert rig.saved == ['Calibration Pose'] and rig.preemptions == [True]
    assert not rig.camera.requests and not rig.executor.queue and rig.c.return_pose is None
    rig.quiet[0] = True
    rig.c.tick()
    assert rig.c.status == 'preparing' and len(rig.executor.queue) == 1
    np.testing.assert_allclose(rig.c.return_pose, rig.current[0])
    rig.work()
    assert rig.camera.requests[0]['action'] == 'laser_new'


def test_exposure_preparation_allows_final_robot_settling_then_freezes_actual_baseline(rig):
    rig.c.start(go_to_calibration_pose=False)
    rig.current[0][0, 3] += .001  # Setup takes seconds; this is not a paired measurement.
    rig.quiet[0] = False
    rig.work()
    assert rig.c.status == 'preparation_settling' and len(rig.holds) == 1
    rig.fresh()
    assert not rig.moves
    rig.quiet[0] = True
    rig.c.tick()
    rig.fresh(.3)
    assert rig.c.status == 'preparation_settling'
    rig.fresh(.3)
    assert rig.c.status == 'hunting'
    np.testing.assert_array_equal(rig.c.return_pose, rig.current[0])
    rig.fresh()
    assert rig.c.status == 'moving' and len(rig.moves) == 1


def test_capture_settling_waits_for_continuous_quiet_and_resets_acquisition_cutoff(rig):
    rig.start()
    rig.settle()
    old_cutoff = rig.c._settled_at
    rig.quiet[0] = False
    rig.fresh()
    assert rig.c.status == 'settling' and not rig.executor.queue and len(rig.holds) == 2
    rig.quiet[0] = True
    rig.c.tick()
    rig.fresh(.3)
    assert rig.c.status == 'settling'
    rig.fresh(.2)
    assert rig.c.status == 'capturing'
    assert rig.c._capture_cutoff > old_cutoff+.4


def test_completed_edge_holds_once_then_waits_without_retargeting_or_chasing_drift(rig):
    rig.start()
    assert len(rig.holds) == 1  # Initial calibration baseline.
    for _ in range(4):
        rig.c.tick()  # The service still owns an unfinished discrete move.
    assert len(rig.holds) == 1
    rig.settle()
    assert len(rig.holds) == 2
    held = rig.holds[-1].copy()
    rig.fresh(.2)
    assert rig.c.status == 'settling' and not rig.executor.queue
    rig.current[0][0, 3] += .0006  # Reset observation cutoff; do not chase it with Hold.
    rig.fresh(.1)
    assert rig.c.status == 'settling' and len(rig.holds) == 2
    rig.fresh(.5)
    assert rig.c.status == 'capturing' and len(rig.holds) == 2
    np.testing.assert_array_equal(rig.holds[-1], held)
    rig.work()
    capture = [r for r in rig.camera.requests if r['action'] == 'laser_capture'][-1]
    np.testing.assert_array_equal(capture['metadata']['base_from_tcp'], rig.current[0])
    assert len(rig.holds) == 2


def test_quiet_but_lagging_endpoint_is_not_frozen_before_arrival(rig):
    rig.start()
    goal = rig.moves[-1].copy()
    rig.quiet[0] = True  # A finished reference and low velocity alone are insufficient.
    rig.c.tick()
    assert rig.c.status == 'moving' and len(rig.holds) == 1
    diagnostic = rig.c.state()['endpoint_error']
    assert diagnostic['ready'] and not diagnostic['converged']
    assert diagnostic['position_mm'] > .5
    measured = goal.copy()
    measured[0, 3] += .0004
    measured[:3, :3] = measured[:3, :3] @ Rotation.from_rotvec([.004, 0., 0.]).as_matrix()
    rig.current[0] = measured
    rig.c.tick()
    assert rig.c.status == 'settling' and len(rig.holds) == 2
    assert rig.c.endpoint_error['converged']
    np.testing.assert_array_equal(rig.holds[-1], measured)
    rig.fresh()
    assert rig.c.status == 'capturing'
    rig.work()
    captured = [r for r in rig.camera.requests if r['action'] == 'laser_capture'][-1]
    np.testing.assert_array_equal(captured['metadata']['base_from_tcp'], measured)


def test_endpoint_timeout_preserves_diagnostics_after_safe_stop_hold(rig):
    rig.start()
    requested = rig.moves[-1].copy()
    rig.quiet[0] = True
    rig.fresh(30.1)
    assert rig.c.status == 'error' and 'endpoint did not converge' in rig.c.error
    assert len(rig.holds) == 2
    diagnostic = rig.c.state()['endpoint_error']
    assert not diagnostic['converged'] and diagnostic['position_mm'] > .5
    np.testing.assert_array_equal(diagnostic['requested_pose'], requested)


def test_endpoint_requires_orientation_convergence_and_quiet_even_when_position_matches(rig):
    rig.start()
    goal = rig.moves[-1].copy()
    rig.current[0] = goal.copy()
    rig.current[0][:3, :3] = goal[:3, :3] @ Rotation.from_rotvec([.006, 0., 0.]).as_matrix()
    rig.quiet[0] = True
    rig.c.tick()
    assert len(rig.holds) == 1 and not rig.c.endpoint_error['within_tolerance']
    rig.current[0] = goal.copy()
    rig.quiet[0] = False
    rig.c.tick()
    assert len(rig.holds) == 1 and rig.c.endpoint_error['within_tolerance']
    assert not rig.c.endpoint_error['ready']
    rig.quiet[0] = True
    rig.c.tick()
    assert len(rig.holds) == 2 and rig.c.status == 'settling'


def test_saved_pose_callback_must_supply_real_requested_goal_not_current_pose(rig):
    goal = pose((.3, -.1, .3), (10, -20, 40))
    rig.c.move_saved_pose = lambda name: goal.copy()
    rig.c.start()
    rig.c.tick()
    assert rig.c.status == 'going_to_calibration_pose' and not rig.holds and not rig.executor.queue
    assert not rig.c.endpoint_error['converged']
    rig.current[0] = goal
    rig.c.tick()
    assert rig.c.status == 'preparing' and len(rig.holds) == 1


def test_missing_saved_pose_goal_cannot_fall_back_to_measured_pose(rig):
    rig.c.move_saved_pose = lambda name: None
    with pytest.raises(ValueError, match='requested nominal TCP pose'):
        rig.c.start()
    assert rig.c.status == 'error' and rig.c._move_goal is None
    assert not rig.camera.requests and len(rig.holds) == 1


def test_completed_hunt_move_holds_once_before_using_new_settled_frame(rig):
    rig.camera.full = False
    rig.c.start(go_to_calibration_pose=False)
    rig.work()
    rig.fresh()
    rig.fresh()
    assert len(rig.moves) == 1 and len(rig.holds) == 1
    rig.quiet[0] = True  # Remains at the old point despite an idle/quiet report.
    rig.c.tick()
    assert rig.c.hunt['motion_active'] and len(rig.holds) == 1
    assert rig.c.endpoint_error['position_mm'] == pytest.approx(5.)
    rig.settle()
    assert len(rig.holds) == 2 and not rig.c.hunt['motion_active']
    rig.camera.full = True
    rig.fresh(.2)
    assert rig.c.status == 'hunting' and len(rig.moves) == 1
    rig.fresh(.3)
    assert rig.c.status == 'moving' and len(rig.moves) == 2
    assert len(rig.holds) == 2


def test_hunt_waits_through_post_profile_velocity_jitter_before_using_image(rig):
    rig.c.start(go_to_calibration_pose=False)
    rig.work()
    rig.fresh()
    assert rig.c.status == 'hunting'
    rig.quiet[0] = False
    rig.fresh()
    assert rig.c.status == 'hunting' and not rig.moves and len(rig.holds) == 1
    rig.quiet[0] = True
    rig.c.tick()
    rig.fresh(.3)
    assert not rig.moves
    rig.fresh(.2)
    assert rig.c.status == 'moving'


@pytest.mark.parametrize('phase', ['preparation', 'capture', 'hunt'])
def test_persistently_nonquiet_preobservation_state_has_bounded_specific_failure(rig, phase):
    if phase == 'capture':
        rig.start()
        rig.settle()
    else:
        rig.c.start(go_to_calibration_pose=False)
        rig.work()
        if phase == 'hunt':
            rig.fresh()
    rig.quiet[0] = False
    rig.fresh(15.1)
    assert rig.c.status == 'error' and 'quiet and pose-stable' in rig.c.error
    assert rig.holds


def test_actual_measured_camera_frame_drives_25_edge_poses_without_tcp_mutation(rig):
    initial = rig.current[0].copy()
    rig.start()
    geometry = rig.c.geometry
    base_from_camera = initial @ rig.correction
    np.testing.assert_allclose(geometry['axis_start_m'], base_from_camera[:3, 3])
    np.testing.assert_allclose(geometry['apex_m'], base_from_camera[:3, 3]+.15*base_from_camera[:3, 2])
    assert len(rig.c.poses) == len(geometry['cone_ring_m']) == 25
    actual_poses = [p @ rig.correction for p in rig.c.poses]
    np.testing.assert_allclose([p[:3, 3] for p in actual_poses], geometry['cone_ring_m'])
    for p in actual_poses:
        direction = np.asarray(geometry['apex_m'])-p[:3, 3]
        np.testing.assert_allclose(p[:3, 2], direction/np.linalg.norm(direction), atol=1e-12)
    np.testing.assert_array_equal(rig.c.return_pose, initial)
    np.testing.assert_array_equal(rig.camera.correction, rig.correction)
    assert len(rig.validated) == 26 and len(rig.holds) == 1


def test_centroid_hunt_uses_measured_image_axes_exactly_5mm(rig):
    rig.camera.full = False
    initial = rig.current[0].copy()
    rig.c.start(go_to_calibration_pose=False)
    rig.work()
    rig.fresh()
    rig.fresh()
    offset = np.mean(rig.camera.corners, axis=0)-[500, 400]
    expected = (initial @ rig.correction)[:3, :3] @ np.r_[.005*offset/np.linalg.norm(offset), 0.]
    np.testing.assert_allclose(rig.moves[-1][:3, 3]-initial[:3, 3], expected)
    np.testing.assert_allclose(rig.moves[-1][:3, :3], initial[:3, :3])
    assert rig.c.hunt['step_mm'] == 5. and rig.c.hunt['motion_active']


def run_auto(rig):
    rig.start()
    for _ in range(25):
        rig.settle()
        rig.fresh()
        assert rig.c.status == 'capturing'
        rig.work()
    assert rig.c.status == 'returning'
    rig.settle()
    assert rig.c.status == 'return_settling'
    rig.fresh()
    if rig.c.status == 'solving':
        rig.work()


def test_auto_captures_25_fresh_native_views_then_returns_and_uses_native_solver(rig):
    run_auto(rig)
    assert rig.c.status == 'complete' and rig.c.count == 25
    captures = [r for r in rig.camera.requests if r['action'] == 'laser_capture']
    assert len(captures) == 25
    assert [r['metadata']['view_index'] for r in captures] == list(range(25))
    assert all('not_before_monotonic' in r for r in captures)
    assert len(rig.moves) == 26
    np.testing.assert_allclose(rig.moves[-1], rig.c.return_pose)
    assert len(rig.holds) == 27  # Initial pose, 25 edge poses, baseline return.
    json.dumps(rig.c.state(rig.camera.state()), allow_nan=False)


@pytest.mark.parametrize('corner_count', [0, 9, 25])
def test_partial_cheap_preview_cannot_authorize_hunt_before_native_capture(rig, corner_count):
    rig.start()
    rig.camera.full = False
    rig.camera.corners = [[600.+i, 400.+i] for i in range(corner_count)]
    rig.settle()
    rig.fresh()
    assert rig.c.status == 'capturing' and len(rig.moves) == 1
    rig.work()
    assert rig.c.count == 1 and rig.c.index == 1 and len(rig.moves) == 2
    assert rig.c._capture_hunt_used == set()
    assert rig.c.hunt['phase'] == 'baseline'


def native_missing_board_retry(rig):
    rig.start()
    rig.camera.full = False
    rig.settle()
    rig.fresh()
    rig.work()
    assert rig.c.status == 'hunting' and rig.c.hunt['phase'] == 'capture'
    assert rig.c.index == 0 and rig.c.count == 0 and len(rig.moves) == 1
    rig.camera.full = True
    rig.fresh()
    assert rig.c.status == 'settling'
    rig.fresh()
    assert rig.c.status == 'capturing'
    rig.work()


def test_native_missing_board_allows_one_bounded_hunt_then_one_successful_capture(rig):
    rig.camera.capture_errors[0] = ['Native checkerboard not found, including exhaustive search']
    native_missing_board_retry(rig)
    assert rig.c.count == 1 and rig.c.index == 1 and len(rig.moves) == 2
    assert rig.c._capture_hunt_used == {0}
    assert len([r for r in rig.camera.requests if r['action'] == 'laser_capture']) == 2
    assert rig.camera.count == 1  # Rejected attempt never adds a successful view.


def test_second_native_missing_board_skips_view_without_unbounded_repeated_hunt(rig):
    rig.camera.capture_errors[0] = ['Native checkerboard not found']*2
    native_missing_board_retry(rig)
    assert rig.c.index == 1 and rig.c.count == 0 and len(rig.moves) == 2
    assert rig.c.status == 'moving' and rig.c._capture_hunt_used == {0}
    assert [r['reacquiring'] for r in rig.c.rejected] == [True, False]


@pytest.mark.parametrize('error', ['Board moved during exposure bracket (3.158px); capture rejected',
    'Too few unobscured checkerboard corners; reduce laser glare or narrow the ROI',
    'Laser stripe span is too short', 'Laser-ON board pose error exceeds 1.5px'])
def test_native_quality_errors_never_authorize_centroid_hunt(rig, error):
    rig.camera.capture_errors[0] = [error]
    rig.start()
    rig.camera.full = False
    rig.settle()
    rig.fresh()
    rig.work()
    assert rig.c.status == 'moving' and rig.c.index == 1 and rig.c.count == 0
    assert not rig.c._capture_hunt_used and not rig.c.rejected[-1]['reacquiring']


@pytest.mark.parametrize(('rejected', 'expected'), [(set(range(5)), 'complete'), (set(range(6)), 'error')])
def test_rejected_native_views_are_bounded_and_never_relax_20_view_gate(rig, rejected, expected):
    rig.camera.reject = rejected
    run_auto(rig)
    assert rig.c.status == expected and rig.c.count == 25-len(rejected)
    assert len(rig.c.rejected) == len(rejected)
    assert bool([r for r in rig.camera.requests if r['action'] == 'laser_solve']) == (expected == 'complete')


def test_native_diversity_quality_failure_is_reported_without_relaxation(rig):
    rig.camera.solve_error = 'Laser views lack independent normal/depth diversity'
    run_auto(rig)
    assert rig.c.status == 'error' and 'diversity' in rig.c.error
    assert rig.holds


def test_camera_generation_changes_from_bracket_are_allowed_but_session_changes_abort(rig):
    rig.start()
    rig.settle()
    rig.fresh()
    rig.work()
    assert rig.c.count == 1 and rig.camera.generation == 3
    rig.settle()
    rig.fresh()
    count = len(rig.moves)
    rig.camera.session = 'replacement-stream'
    rig.work()
    assert rig.c.status == 'error' and 'session changed' in rig.c.error
    assert len(rig.moves) == count and rig.holds


@pytest.mark.parametrize('failure', ['cutoff', 'moving', 'position', 'rotation', 'source', 'optics'])
def test_native_capture_must_match_cutoff_identity_and_stationary_robot(rig, failure):
    rig.start()
    rig.settle()
    rig.fresh()
    count = len(rig.moves)
    if failure == 'cutoff':
        rig.camera.stale_capture = True
    elif failure == 'moving':
        rig.quiet[0] = False
    elif failure == 'position':
        rig.current[0][0, 3] += .0006
    elif failure == 'rotation':
        rig.current[0][:3, :3] = rig.current[0][:3, :3] @ Rotation.from_rotvec([.006, 0, 0]).as_matrix()
    elif failure == 'source':
        rig.camera.source = 'another-camera'
    else:
        rig.camera.optical = 'other-intrinsics'
    rig.work()
    assert rig.c.status == 'error' and rig.c.count == 0 and len(rig.moves) == count


@pytest.mark.parametrize('reason,hold_count', [('cancel', 1), ('abort_for_cooldown', 0), ('abort_for_shutdown', 0)])
def test_cancellation_ignores_inflight_capture_and_never_overwrites_thermal_or_shutdown_motion(rig, reason, hold_count):
    rig.start()
    rig.settle()
    rig.fresh()
    before = len(rig.moves)
    prior_holds = len(rig.holds)
    getattr(rig.c, reason)()
    rig.work()  # Cancelled queued work cannot authorize another arm move.
    assert rig.c.status == 'cancelled' and len(rig.moves) == before
    assert len(rig.holds) == prior_holds+hold_count
    assert rig.executor.queue[-1][2][0] == {'action': 'laser_cancel'}


def test_stale_pre_move_frame_never_authorizes_capture(rig):
    rig.start()
    rig.settle()
    rig.clock[0] += .5
    rig.c.tick()
    assert rig.c.status == 'settling' and not rig.executor.queue
    rig.fresh()
    assert rig.c.status == 'capturing'


def test_missing_board_uses_bounded_reacquisition_instead_of_capture(rig):
    rig.camera.full, rig.camera.corners = False, []
    rig.c.start(go_to_calibration_pose=False)
    rig.work()
    rig.fresh()
    rig.fresh()
    assert rig.c.hunt['reacquiring'] and rig.c.hunt['steps'] == 1
    assert not [r for r in rig.camera.requests if r['action'] == 'laser_capture']
    rig.clock[0] += 60.
    rig.c.tick()
    assert rig.c.status == 'error' and rig.holds


def test_manual_dispatch_uses_worker_and_never_moves_robot(rig):
    for action in ('laser_config', 'laser_preview', 'laser_check', 'laser_capture', 'laser_solve', 'laser_reset'):
        if action == 'laser_solve':
            rig.camera.count = 20
        before = len(rig.camera.requests)
        rig.c.dispatch({'action': action, 'enabled': True})
        assert len(rig.camera.requests) == before
        rig.work()
    assert not rig.moves and not rig.saved and rig.c.count == 0


@pytest.mark.parametrize('action', ['laser_config', 'laser_preview', 'laser_reset'])
def test_service_transport_id_is_not_forwarded_as_a_camera_setting(rig, action):
    payload = {'enabled': True} if action == 'laser_preview' else {}
    rig.c.dispatch({'action': action, '_id': 'filesystem-request',
                    '_service_queued_monotonic': 123., '_blocked_during_recovery': False, **payload})
    rig.work()
    assert '_id' not in rig.camera.requests[-1]
    assert not any(key.startswith('_') for key in rig.camera.requests[-1])
    expected = {'laser_config': 'laser_configure', 'laser_preview': 'laser_preview',
                'laser_reset': 'laser_new'}
    assert rig.camera.requests[-1]['action'] == expected[action]


def test_camera_only_configuration_failure_does_not_steal_robot_motion(rig):
    def rejected(request):
        raise ValueError('Invalid laser threshold')
    rig.camera.laser_command = rejected
    rig.c.dispatch({'action': 'laser_config'})
    rig.quiet[0] = False
    rig.work()
    assert rig.c.status == 'error' and 'threshold' in rig.c.error
    assert not rig.holds and not rig.moves


@pytest.mark.parametrize('failure', ['disabled', 'not_ready', 'missing_handeye', 'identity', 'count'])
def test_start_rejects_before_motion_or_camera_work(rig, failure):
    if failure == 'disabled':
        rig.camera.enabled = False
    elif failure == 'not_ready':
        rig.quiet[0] = False
    elif failure == 'missing_handeye':
        original = rig.camera.state
        rig.camera.state = lambda **kw: original(**kw) | {'handeye': None}
    elif failure == 'identity':
        rig.camera.optical = 'other'
    with pytest.raises(ValueError):
        rig.c.start(count=20 if failure == 'count' else 25)
    assert not rig.moves and not rig.saved and not rig.executor.queue


def test_camera_operation_runs_off_main_thread_without_blocking_motion_tick(rig):
    started, release, finished = Event(), Event(), Event()
    main_thread = get_ident()
    def native(request):
        assert get_ident() != main_thread
        started.set()
        release.wait(2.)
        finished.set()
        return {}
    rig.c.executor = ThreadPoolExecutor(max_workers=1)
    rig.camera.laser_command = native
    try:
        rig.c.dispatch({'action': 'laser_config'})
        assert started.wait(1.) and not finished.is_set()
        rig.c.tick()
        assert rig.c.busy and not finished.is_set()
    finally:
        release.set()
        rig.c.future.result(timeout=1.)


def test_j6_and_world_composition_uses_nominal_frame_and_handeye_once():
    flange = pose((-.0695, .0321, -.0688), (180, 0, 90))
    correction = pose((.026, .023, -.013), (4, 6, 120))
    base = pose((.2, -.1, .3), (10, 20, 30))
    points = np.array([[0., 0., .15], [.01, -.02, .2]])
    cloud = dict(xyz_m=points.tolist(), units='m', coordinate_frame='camera', source_id='camera',
        optical_intrinsics_id='optics', calibration_id='full-plane-id', read_started=99.9, sequence=4)
    original = copy.deepcopy(cloud)
    result = transform_cloud(cloud, base, handeye(correction), flange, now=100.)
    expected_j6 = (flange @ correction @ np.c_[points, np.ones(2)].T).T[:, :3]
    expected_world = (base @ correction @ np.c_[points, np.ones(2)].T).T[:, :3]
    np.testing.assert_allclose(result['points_j6_m'], expected_j6)
    np.testing.assert_allclose(result['points_world_m'], expected_world)
    np.testing.assert_allclose(result['base_from_j6'], base @ np.linalg.inv(flange))
    assert result['coordinate_frame'] == 'j6' and result['world_frame'] == 'robot_base'
    assert cloud == original
    json.dumps(result, allow_nan=False)


def test_no_valid_sdk_world_pose_still_provides_rigid_tool_points_only():
    cloud = dict(xyz_m=[[0., 0., .15]], units='m', coordinate_frame='camera', source_id='camera', calibration_id='optics')
    result = transform_cloud(cloud, None, handeye(), np.eye(4))
    assert result['points_j6_m'] == [[0., 0., .15]]
    assert result['points_world_m'] == [] and result['base_from_j6'] is None
    assert result['status'] == 'j6_only'


def test_explicit_meter_product_coexists_with_original_native_mm_cloud():
    cloud = dict(xyz_m=[[0., 0., .15]], xyz_m_units='m', xyz_mm=[[0., 0., 150.]], units='mm',
                 coordinate_frame='camera', source_id='camera', calibration_id='optics')
    assert transform_cloud(cloud, None, handeye(), np.eye(4))['points_j6_m'] == [[0., 0., .15]]
    for declared in (None, 'mm'):
        invalid = dict(cloud)
        if declared is None:
            del invalid['xyz_m_units']
        else:
            invalid['xyz_m_units'] = declared
        with pytest.raises(ValueError, match='explicit meter units'):
            transform_cloud(invalid, None, handeye(), np.eye(4))


def test_optical_identity_survives_plane_addition_without_rewriting_handeye_provenance(rig):
    migrated = handeye(rig.correction) | {'calibration_id': 'original-full-model', 'optical_intrinsics_id': 'optics'}
    original_state = rig.camera.state
    rig.camera.state = lambda **kwargs: original_state(**kwargs) | {'handeye': migrated}
    rig.start()
    cloud = dict(xyz_m=[[0., 0., .15]], units='m', coordinate_frame='camera', source_id='camera',
                 calibration_id='new-model-with-plane', optical_intrinsics_id='optics')
    assert transform_cloud(cloud, None, migrated, np.eye(4))['count'] == 1
    assert migrated['calibration_id'] == 'original-full-model'


@pytest.mark.parametrize('change', [dict(coordinate_frame='robot_base'), dict(source_id='other'),
    dict(optical_intrinsics_id='other'), dict(xyz_m=[[float('nan'), 0., 0.]]), dict(read_started=98.),
    dict(read_started=101.), dict(xyz_m=[[1., 2.]]), dict(xyz_m=None)])
def test_cloud_rejects_frame_identity_units_shape_and_stale_data(change):
    cloud = dict(xyz_m=[[0., 0., .15]], units='m', coordinate_frame='camera', source_id='camera',
                 calibration_id='optics', read_started=99.9) | change
    with pytest.raises((ValueError, TypeError)):
        transform_cloud(cloud, np.eye(4), handeye(), np.eye(4), now=100.)


def test_cloud_downsampling_is_bounded_and_empty_cloud_is_valid():
    cloud = dict(xyz_m=np.ones((10001, 3)).tolist(), units='m', coordinate_frame='camera', source_id='camera',
                 calibration_id='optics')
    result = transform_cloud(cloud, np.eye(4), handeye(), np.eye(4), max_points=5000)
    assert result['count'] <= 5000 and result['total_count'] == 10001 and result['stride'] == 3
    cloud['xyz_m'] = []
    assert transform_cloud(cloud, None, handeye(), np.eye(4))['count'] == 0


def test_geometry_rejects_improper_rotations_and_non_meter_handeye():
    reflection = np.diag([-1., 1., 1., 1.])
    with pytest.raises(ValueError):
        j6_from_camera(reflection, np.eye(4))
    with pytest.raises(ValueError, match='meters'):
        camera_points_to_j6([[0., 0., .15]], np.eye(4), handeye() | {'units': 'mm'})
