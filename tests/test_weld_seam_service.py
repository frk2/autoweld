"""Hardware-free ownership, acquisition association and bidirectional path checks."""

import copy
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.motion import MotionLimits
from autoweld.weld_seam_service import WeldSeamService, WeldSettings


class ImmediateExecutor:
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
    clock, pose, goal = [100.], np.eye(4), np.eye(4)
    events, plans = [], []
    feedback = dict(fresh=True, stationary=True)
    state = dict(enabled=False, starting=False, stopping=False, configuring=False,
        source_id='camera', optical_intrinsics_id='optics', calibration_id='laser',
        acquisition_session_id='session', generation=1,
        handeye=dict(source_id='camera', optical_intrinsics_id='optics', tcp_from_camera=np.eye(4).tolist()),
        intrinsics=dict(laser_plane=[0, 1, 0, 0]), laser={})
    mode = dict(ends_mm=14., failure=None, exposure_delay=0., previous=[], detector_calls=0,
                configuration_pending=False, ae_reliable=True, ae_state='stable')
    state['laser']['pointcloud_status'] = dict(auto_exposure={'enabled': False}, auto_exposure_status={'state': 'off'})
    state['controls'] = {'exposure_us': 4000., 'gain_db': 0.}
    state['laser']['exposure_limits'] = dict(sensor_us=[100., 100000.], fps_cap_us=16000.)

    def camera_dispatch(request):
        events.append(('camera', request))
        if request['action'] == 'camera_stream':
            state['enabled'] = request['enabled']
        elif request['action'] == 'camera_laser_configure':
            assert type(request['auto_exposure']['enabled']) is bool
            state['laser_configuration'] = dict(request_id=request['request_id'],
                status='pending' if mode['configuration_pending'] else 'complete')
            state['configuring'] = mode['configuration_pending']
            if not mode['configuration_pending']:
                state['laser']['pointcloud_status']['auto_exposure']['enabled'] = request['auto_exposure']['enabled']
                if request.get('controls'):
                    state['controls'].update(request['controls'])
                    state['generation'] += 1
                state['laser_configuration']['readback'] = copy.deepcopy(state['controls'])
        elif request['action'] == 'camera_laser_cancel':
            assert request['request_id'] == state['laser_configuration']['request_id']
            state['laser_configuration']['status'] = 'cancelling' if mode['configuration_pending'] else 'cancelled'

    def find(cloud, *, settings=None, previous_point_mm=None):
        assert settings == {'torch_standoff_mm': 150.}
        mode['previous'].append(previous_point_mm)
        mode['detector_calls'] += 1
        if mode['failure']:
            return dict(status=mode['failure'], found=False, reason='poor stripe')
        if abs(pose[1, 3])*1000. > mode['ends_mm']:
            return dict(status='no_seam', found=False, quality={'good': True})
        torch = np.eye(4)
        return dict(status='found', found=True, seam_point_mm=[0., 0., 150.],
            camera_from_torch=torch.tolist(), scan_direction_camera=[0., 1., 0.],
            stripe_direction_camera=[1., 0., 0.], surface_normal_camera=[0., 0., -1.],
            flank_lines_mm=[[[-10., 0., 150.], [-2., 0., 150.]],
                            [[2., 0., 150.], [10., 0., 150.]]], quality={'good': True})

    camera = SimpleNamespace(state=lambda **_: copy.deepcopy(state), dispatch=camera_dispatch, find_weld_seam=find)

    def move(target, automatic=False):
        assert not automatic
        goal[:] = target
        events.append(('move', goal.copy()))

    def saved(name):
        assert name == 'Weld Finder'
        target = np.eye(4)
        events.append(('saved', name))
        goal[:] = target
        return target

    def hold(automatic=False):
        assert not automatic
        events.append(('hold', pose.copy()))

    def validate(target):
        result = np.zeros(7)
        result[:3] = target[:3, 3]
        return result

    def plan(start, target, **kwargs):
        plans.append((np.array(start), np.array(target), kwargs))
        if mode.get('plan_fail_at') == len(plans):
            raise ValueError('collision in replay preflight')
        assert callable(kwargs['validate']) and callable(kwargs['collision'])

    service = SimpleNamespace(camera=camera, runtime=tmp_path, model_fingerprint='model',
        saved_poses={'Weld Finder': {'q': [0.]*6}}, can_start_weld=lambda: None,
        tcp_pose=lambda: pose.copy(), move_pose=move, move_laser_saved_pose=saved,
        scan_feedback=lambda: {**feedback, 'measured_at': clock[0]}, validate_pose=validate,
        has_self_collision=lambda q: False, planner=SimpleNamespace(plan=plan, limits=MotionLimits()),
        ctrl=SimpleNamespace(ready_for_waypoint=lambda: True, hold=hold,
            positions=lambda: np.r_[pose[:3, 3], np.zeros(4)], validate=lambda *a, **k: None,
            effective_joint_bounds=lambda: np.tile([-10., 10.], (6, 1))))
    def preflight(source, start, poses, bounds, cancelled, progress):
        seed = start
        for index, replay_pose in enumerate(poses):
            assert not cancelled()
            target = validate(replay_pose)
            source.plan(seed, target, cartesian=replay_pose, bounds=bounds,
                        validate=lambda q: None, collision=lambda q: False)
            progress(index+1)
            seed = target
        return {'segment_count': len(poses)}

    weld = WeldSeamService(service, clock=lambda: clock[0], executor=ImmediateExecutor(),
                           replay_preflight=preflight)
    sequence = [0]

    def advance(dt=.1, follow=True, update_frame=True):
        clock[0] += dt
        if follow:
            pose[:] = goal
        if state['laser']['pointcloud_status']['auto_exposure']['enabled']:
            state['laser']['pointcloud_status']['auto_exposure_status'] = dict(state=mode['ae_state'],
                reliable=mode['ae_reliable'], source_id=state['source_id'], generation=state['generation'],
                read_started=clock[0]-.001, sequence=sequence[0])
        if update_frame:
            sequence[0] += 1
            delayed = clock[0]-mode['exposure_delay']
            xyz = np.column_stack((np.linspace(-10, 10, 101), np.zeros(101), np.full(101, 150.)))
            state['laser']['cloud'] = dict(xyz_mm=xyz.tolist(), controls=copy.deepcopy(state['controls']), uv=np.c_[np.arange(101), np.zeros(101)].tolist(),
                sequence=sequence[0], read_started=clock[0]-.0005, read_completed=clock[0],
                source_id='camera', optical_intrinsics_id='optics', calibration_id='laser',
                acquisition_session_id='session', generation=state['generation'], units='mm', coordinate_frame='camera',
                timing=dict(host_clock_domain='host_monotonic', motion_pairing_supported=True, clock_mapping={'status': 'qualified'},
                    exposure_host_start_s=delayed-.002, exposure_host_end_s=delayed-.001,
                    exposure_time_uncertainty_s=.0001))
        weld.tick()

    result = SimpleNamespace(weld=weld, service=service, clock=clock, state=state, feedback=feedback,
        pose=pose, goal=goal, events=events, mode=mode, plans=plans, advance=advance, path=tmp_path)
    yield result
    weld.close()


def pump(rig, predicate, maximum=3000):
    for _ in range(maximum):
        if predicate():
            return
        rig.advance()
    raise AssertionError(rig.weld.state(camera_state=rig.state))


def complete(rig, minimum=25.):
    rig.weld.start({'min_distance_mm': minimum})
    pump(rig, lambda: rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy)
    assert rig.weld.status == 'complete', rig.weld.error


@pytest.mark.parametrize('settings', [dict(min_distance_mm=0), dict(min_distance_mm=101),
    dict(min_distance_mm=True), dict(standoff_mm=140), dict(start_pose='Tube Scanning'),
    dict(step_mm=.5), dict(end_confirmations=1), dict(max_distance_each_way_mm=101)])
def test_setting_bounds_are_explicit(settings):
    with pytest.raises(ValueError):
        WeldSettings(**settings).validate()


def test_start_prepares_one_camera_and_uses_weld_finder_before_measuring(rig):
    rig.weld.start({})
    assert [event[1]['action'] for event in rig.events] == ['camera_analysis', 'camera_stream']
    pump(rig, lambda: rig.weld.status == 'moving')
    assert rig.events[-1] == ('saved', 'Weld Finder')
    assert rig.weld.status == 'moving'
    assert rig.mode['detector_calls'] == 0


def test_endpoint_and_continuous_quiet_are_required_before_one_hold(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'moving')
    rig.pose[0, 3] = .002
    for _ in range(7):
        rig.advance(follow=False)
    assert not any(event[0] == 'hold' for event in rig.events)
    rig.pose[:] = rig.goal
    rig.feedback['stationary'] = False
    for _ in range(7):
        rig.advance()
    assert not any(event[0] == 'hold' for event in rig.events)
    rig.feedback['stationary'] = True
    pump(rig, lambda: any(event[0] == 'hold' for event in rig.events))
    for _ in range(3):
        rig.advance()
    assert len([event for event in rig.events if event[0] == 'hold']) == 1
    assert rig.mode['detector_calls'] == 0


def test_old_or_straddling_exposure_waits_at_same_hold_without_retarget(rig):
    rig.mode['exposure_delay'] = 1.
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    for _ in range(5):
        rig.advance()
    assert rig.mode['detector_calls'] == 0
    assert len([event for event in rig.events if event[0] == 'hold']) == 1
    rig.mode['exposure_delay'] = 0.
    pump(rig, lambda: rig.mode['detector_calls'] == 1)


def test_bilateral_path_is_measured_ordered_and_persisted_with_confirmed_ends(rig):
    complete(rig)
    state = rig.weld.state(camera_state=rig.state)
    points = np.asarray(state['geometry']['seam_points_base_m'])
    assert len(points) > 20
    assert np.all(np.diff(points[:, 1]) > 0)
    assert points[0, 1] < 0 < points[-1, 1]
    assert state['endpoints']['positive']['no_seam_stations'] >= 3
    assert state['endpoints']['negative']['no_seam_stations'] >= 3
    assert state['result']['seam_length_mm'] >= 25.
    assert state['geometry']['units'] == 'm'
    assert state['geometry']['coordinate_frame'] == 'robot_base'
    assert state['replay_ready']
    native = list((rig.path/'weld-paths'/rig.weld.session_id).glob('observation-*.json'))
    assert len(native) == rig.weld._stations
    assert len(native) > len(rig.weld.records)
    assert (rig.path/'weld-paths'/rig.weld.session_id/'path.json').exists()
    assert rig.mode['previous'][0] is None
    assert all(previous is not None for previous in rig.mode['previous'][1:])


def test_early_valley_end_still_observes_requested_minimum_without_inventing_seam(rig):
    rig.mode['ends_mm'] = 2.
    rig.weld.start({'min_distance_mm': 25.})
    pump(rig, lambda: rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy)
    assert rig.weld.status == 'complete'
    assert rig.weld._search_extents[1] >= 12.5 and rig.weld._search_extents[-1] >= 12.5
    assert rig.weld._result['seam_length_mm'] < 5.
    assert rig.weld._result['search_distance_mm'] >= 25.
    assert rig.weld._result['minimum_basis'] == 'measured_scan_plane_extent'
    assert rig.weld.state(camera_state=rig.state)['replay_ready']


@pytest.mark.parametrize('failure', ['invalid', 'ambiguous'])
def test_bad_stripe_or_ambiguity_cannot_confirm_a_seam_end(rig, failure):
    rig.mode['failure'] = failure
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    assert rig.weld.endpoints == {'positive': None, 'negative': None}
    assert 'endpoint is unknown' in rig.weld.error
    assert not rig.weld.records


def test_no_seam_at_start_is_not_an_empty_complete_path(rig):
    rig.mode['ends_mm'] = -1.
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    assert 'no seam' in rig.weld.error
    assert rig.weld._result is None


def test_identity_change_interrupts_before_motion_from_new_source(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.mode['detector_calls'] == 1)
    rig.state['acquisition_session_id'] = 'replacement'
    rig.advance()
    assert rig.weld.status == 'error'
    assert not any(event[0] == 'move' for event in rig.events)


def test_disappearance_requires_good_lines_and_bounded_travel_cannot_claim_an_end(rig):
    rig.mode['ends_mm'] = 1000.
    rig.weld.start({'min_distance_mm': 1., 'max_distance_each_way_mm': 5.})
    pump(rig, lambda: rig.weld.status == 'error')
    assert 'endpoint is unconfirmed' in rig.weld.error
    assert rig.weld.endpoints['positive'] is None


def test_shutdown_cancels_future_generation_without_hold_or_later_motion(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.future is not None)
    before = len(rig.events)
    rig.weld.abort_for_shutdown()
    for _ in range(5):
        rig.advance()
    assert len(rig.events) == before
    assert rig.weld.status == 'interrupted'
    assert not rig.weld.records


def test_restore_geometry_never_resumes_robot_or_camera(rig):
    complete(rig)
    profiles = rig.weld.state(camera_state=rig.state)['geometry']['profile_points_base_m']
    assert len(profiles) == 101*len(rig.weld.records)
    before = len(rig.events)
    restored = WeldSeamService(rig.service, clock=lambda: rig.clock[0], executor=ImmediateExecutor())
    assert len(rig.events) == before
    assert restored.status == 'complete' and restored.records
    assert restored._goal is None
    assert restored.state(camera_state=rig.state)['geometry']['profile_points_base_m'] == profiles
    restored.close()


def test_full_profile_uses_measured_camera_transform_without_filling_missing_returns(rig):
    pose = np.eye(4)
    pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    pose[:3, 3] = [.1, .2, .3]
    record = {'native_path': 'profile'}
    measurement = dict(base_from_camera=pose.tolist(), cloud=dict(units='mm', coordinate_frame='camera',
        xyz_mm=[[-25., 0., 150.], [-24., 0., 150.], [25., 0., 155.]]))
    rig.weld._retain_profile(record, measurement)
    np.testing.assert_allclose(rig.weld._profiles['profile'],
        [[.1, .175, .45], [.1, .176, .45], [.1, .225, .455]])
    assert len(rig.weld._profiles['profile']) == 3  # No bridge through the missing interval.


def test_replay_preflights_every_segment_before_any_motion_and_uses_torch_reference(rig):
    complete(rig)
    before = len(rig.events)
    rig.weld.replay()
    total = len(rig.weld._replay)
    assert len(rig.events) == before
    assert len(rig.plans) == total  # Worker proof has completed, owner has not submitted.
    rig.advance()
    assert len(rig.plans) == total
    assert rig.events[-1][0] == 'move'
    expected = np.asarray(rig.weld._result['path'][0]['torch_pose_base_m'])
    np.testing.assert_allclose(rig.events[-1][1], expected)
    pump(rig, lambda: rig.weld.status == 'complete')
    assert rig.weld._replay_index == total


def test_late_replay_preflight_collision_rejects_before_any_move(rig):
    complete(rig)
    before = len(rig.events)
    rig.mode['plan_fail_at'] = 5
    rig.weld.replay()
    pump(rig, lambda: rig.weld.status == 'error')
    assert not any(event[0] == 'move' for event in rig.events[before:])
    assert 'collision' in rig.weld.error


@pytest.mark.parametrize('change', ['model', 'handeye', 'optics', 'laser', 'path'])
def test_replay_cannot_use_changed_calibration_model_or_tampered_path(rig, change):
    complete(rig)
    if change == 'model':
        rig.service.model_fingerprint = 'replacement'
    elif change == 'handeye':
        rig.state['handeye']['tcp_from_camera'][0][3] = .001
    elif change == 'optics':
        rig.state['handeye']['optical_intrinsics_id'] = 'replacement'
    elif change == 'laser':
        rig.state['calibration_id'] = 'replacement'
    else:
        rig.weld._result['path'][0]['seam_point_base_m'][0] += .001
    with pytest.raises(ValueError):
        rig.weld.replay()


def test_stale_post_analysis_feedback_waits_same_hold_without_discard_or_retarget(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld._future_kind == 'analyze')
    before_holds = len([event for event in rig.events if event[0] == 'hold'])
    rig.feedback['fresh'] = False
    for _ in range(4):
        rig.advance()
    assert rig.weld.status == 'validating'
    assert not rig.weld.records
    assert len([event for event in rig.events if event[0] == 'hold']) == before_holds
    rig.feedback['fresh'] = True
    pump(rig, lambda: len(rig.weld.records) == 1)
    assert rig.weld.status == 'moving'


def test_reappearing_seam_after_unobserved_station_cannot_bridge_into_replay(rig):
    original = rig.service.camera.find_weld_seam

    def with_gap(cloud, **kwargs):
        if .0035 < rig.pose[1, 3] < .0045:
            return dict(status='no_seam', found=False, quality={'good': True})
        return original(cloud, **kwargs)

    rig.service.camera.find_weld_seam = with_gap
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    assert 'unobserved interior gap' in rig.weld.error
    assert rig.weld._result['status'] == 'incomplete'
    assert rig.weld._result['replay_scope'] == 'measured_section_only'
    assert max(record['distance_mm'] for record in rig.weld.records) == 3.
    assert rig.weld.state(camera_state=rig.state)['replay_ready']
    assert len(list(rig.weld._directory().glob('observation-*.json'))) > len(rig.weld.records)


def test_rejected_native_frame_and_full_detector_result_are_archived(rig):
    rig.mode['failure'] = 'invalid'
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    import json
    raw_path = next(rig.weld._directory().glob('observation-*.json'))
    raw = json.loads(raw_path.read_text())
    assert raw['result']['status'] == 'invalid'
    assert len(raw['measurement']['cloud']['xyz_mm']) == 101
    assert raw['measurement']['feedback']['fresh']
    assert raw['measurement']['exposure_interval_s'][0] >= raw['measurement']['capture_cutoff']


def test_measured_search_minimum_does_not_accept_a_quarter_mm_shortfall(rig):
    complete(rig)
    requested = rig.weld._search_extents[1]+rig.weld._search_extents[-1]+.25
    rig.weld.settings = WeldSettings(min_distance_mm=requested)
    rig.weld._finish()
    rig.advance()
    assert rig.weld.status == 'insufficient_length'
    assert not rig.weld._result['minimum_satisfied']


def test_measured_handeye_is_applied_once_to_world_geometry_and_inverse_once_for_replay(rig):
    rig.state['handeye']['tcp_from_camera'][0][3] = .012
    complete(rig)
    points = np.asarray(rig.weld.state(camera_state=rig.state)['geometry']['seam_points_base_m'])
    np.testing.assert_allclose(points[:, 0], .012)
    rig.weld.replay()
    rig.advance()
    torch = np.asarray(rig.weld._result['path'][0]['torch_pose_base_m'])
    correction = np.asarray(rig.state['handeye']['tcp_from_camera'])
    np.testing.assert_allclose(rig.events[-1][1], torch @ np.linalg.inv(correction))
    assert abs(rig.events[-1][1][0, 3]) < 1e-12


def test_worker_replay_preflight_uses_private_data_and_snapshot_callbacks_only(monkeypatch):
    from autoweld import diagnostics, scan_plan_probe
    from autoweld.weld_seam_service import preflight_replay

    source = SimpleNamespace(model=object(), limits=MotionLimits())
    clones, checked, progress = [], [], []

    def clone(existing):
        clones.append(existing)

        def plan(start, target, **kwargs):
            kwargs['validate'](target)
            assert not kwargs['collision'](target)
            checked.append((start.copy(), target.copy()))
            return SimpleNamespace(report={'validated': True})

        return SimpleNamespace(kin='private-kin', model=existing.model, site_name='weld_tcp',
            _configuration=lambda q: q.copy(), plan=plan)

    def solve(kin, pose, site, seed, **kwargs):
        assert kin == 'private-kin' and site == 'weld_tcp'
        assert kwargs['stage'] == 'weld_replay_preflight'
        seed[:3] = pose[:3, 3]
        return seed

    monkeypatch.setattr(scan_plan_probe, 'private_planner', clone)
    monkeypatch.setattr(scan_plan_probe, 'private_self_collision', lambda planner, q: False)
    monkeypatch.setattr(diagnostics, 'solve_pose', solve)
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .001
    result = preflight_replay(source, np.zeros(7), poses, np.tile([-1., 1.], (6, 1)),
                              lambda: False, progress.append)
    assert clones == [source] and progress == [1, 2]
    assert len(checked) == 2 and result['segment_count'] == 2
    assert result['limits']['linear_velocity'] == 3.
    assert result['limits']['joint_velocity'] == 2.
    with pytest.raises(RuntimeError, match='cancelled'):
        preflight_replay(source, np.zeros(7), poses, np.tile([-1., 1.], (6, 1)), lambda: True, lambda _: None)


def test_autonomous_preparation_enables_native_stripe_ae_and_awaits_configuration_before_move(rig):
    rig.mode['configuration_pending'] = True
    rig.weld.start({})
    for _ in range(5):
        rig.advance()
    configured = [event for event in rig.events if event[0] == 'camera' and
                  event[1]['action'] == 'camera_laser_configure']
    assert len(configured) == 1
    assert configured[0][1]['auto_exposure'] == {'enabled': True}
    assert not any(event[0] in ('saved', 'move') for event in rig.events)
    rig.state['laser_configuration']['status'] = 'complete'
    rig.state['configuring'] = False
    rig.state['laser']['pointcloud_status']['auto_exposure']['enabled'] = True
    rig.advance()
    assert rig.events[-1] == ('saved', 'Weld Finder')


def test_cancelling_pending_native_configuration_retains_ownership_and_cannot_move_later(rig):
    rig.mode['configuration_pending'] = True
    rig.weld.start({})
    rig.advance()
    rig.weld.cancel()
    assert rig.weld.busy
    assert rig.events[-2][1]['action'] == 'camera_laser_cancel'
    for _ in range(3):
        rig.advance()
    assert rig.weld.busy
    rig.state['laser_configuration']['status'] = 'cancelled'
    rig.state['configuring'] = False
    rig.advance()
    assert not rig.weld.busy and rig.weld.status == 'cancelled'
    assert not any(event[0] in ('saved', 'move') for event in rig.events)


def test_native_ae_reliability_is_required_after_saved_arrival_and_bounded_at_same_hold(rig):
    rig.mode['ae_reliable'] = False
    rig.weld.start({})
    pump(rig, lambda: rig.weld._ae_wait_started is not None)
    assert any(event[0] == 'saved' for event in rig.events)
    assert rig.mode['detector_calls'] == 0
    holds = len([event for event in rig.events if event[0] == 'hold'])
    for _ in range(10):
        rig.advance()
    assert rig.mode['detector_calls'] == 0
    assert len([event for event in rig.events if event[0] == 'hold']) == holds
    pump(rig, lambda: rig.weld.status == 'error')
    assert 'five seconds' in rig.weld.error


def test_unowned_generation_change_after_manual_lock_cannot_accept_a_measurement(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld._future_kind == 'analyze')
    rig.state['generation'] += 1
    rig.state['controls'] = {'exposure_us': 3000., 'gain_db': 0.}
    rig.advance()
    assert rig.weld.status == 'error'
    assert 'Unexpected camera generation' in rig.weld.error
    assert not rig.weld.records


def test_all_native_files_are_byte_hashed_and_capture_pose_gates_are_independently_auditable(rig):
    import hashlib
    import json
    from pathlib import Path

    complete(rig)
    result = rig.weld._result
    files = result['native_files']
    assert len(files) == rig.weld._stations
    assert {Path(entry['path']) for entry in files} == set(rig.weld._directory().glob('observation-*.json'))
    for entry in files:
        path = Path(entry['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        measurement = json.loads(path.read_text())['measurement']
        assert measurement['quiet_duration_s'] >= .4
        assert 0 <= measurement['captured_at']-measurement['feedback']['measured_at'] <= .1
        assert measurement['goal_error']['position_m'] <= .0005
        assert measurement['goal_error']['orientation_rad'] <= .005
        assert measurement['post_capture']['feedback']['stationary']
        assert measurement['post_capture']['goal_error']['position_m'] <= .0005
        assert np.asarray(measurement['requested_base_from_tcp']).shape == (4, 4)
    manifest = json.loads((rig.weld._directory()/'observations.json').read_text())
    assert manifest['native_files'] == result['native_files']
    assert manifest['native_files_sha256'] == result['native_files_sha256']


@pytest.mark.parametrize('selected_us,expected_requests', [(16000., [16000.]), (8000., [16000., 8000.])])
def test_fixed_hold_bracket_recovers_dim_or_clipped_geometry_and_preserves_selected_manual_exposure(
        rig, selected_us, expected_requests):
    original = rig.service.camera.find_weld_seam

    def exposure_sensitive(cloud, **kwargs):
        if abs(cloud['controls']['exposure_us']-selected_us) > 1.:
            return dict(status='invalid', found=False, quality={'good': False, 'reason': 'root too dim or clipped'})
        return original(cloud, **kwargs)

    rig.service.camera.find_weld_seam = exposure_sensitive
    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 3)
    requests = [event[1] for event in rig.events if event[0] == 'camera' and
                event[1]['action'] == 'camera_laser_configure' and event[1].get('controls')
                and not event[1]['request_id'].endswith('-ae-lock')]
    assert [request['controls']['exposure_us'] for request in requests] == expected_requests
    assert all(request['auto_exposure']['enabled'] is False for request in requests)
    assert len({request['request_id'] for request in requests}) == len(requests)
    assert rig.weld._manual_exposure_us == selected_us
    assert not rig.state['laser']['pointcloud_status']['auto_exposure']['enabled']
    assert all(record['detector']['status'] == 'found' for record in rig.weld.records)
    assert any(entry['disposition'] == 'rejected' for entry in rig.weld._native_files)
    assert all(transition['reason'] == 'owned_exposure_bracket' for transition in rig.weld._generation_transitions)
    import json
    from pathlib import Path
    measured = json.loads(Path(rig.weld.records[0]['native_path']).read_text())['measurement']
    assert measured['exposure_control']['actual_us'] == selected_us
    assert measured['cloud']['controls']['exposure_us'] == selected_us
    # All optical retries precede the next physical station and retain the one
    # measured Hold established before their first native image.
    first_manual = next(index for index, event in enumerate(rig.events)
        if event[0] == 'camera' and event[1]['action'] == 'camera_laser_configure' and event[1].get('controls'))
    first_move = next(index for index, event in enumerate(rig.events) if event[0] == 'move')
    assert not any(event[0] in ('move', 'hold') for event in rig.events[first_manual:first_move])


def test_manual_optical_bracket_exhaustion_is_bounded_and_cannot_confirm_an_endpoint(rig):
    rig.state['controls']['exposure_us'] = 5000.
    rig.mode['failure'] = 'invalid'
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    requests = [event[1] for event in rig.events if event[0] == 'camera' and
                event[1]['action'] == 'camera_laser_configure' and event[1].get('controls')
                and not event[1]['request_id'].endswith('-ae-lock')]
    assert [request['controls']['exposure_us'] for request in requests] == [16000., 8000., 4000.]
    assert rig.weld.endpoints == {'positive': None, 'negative': None}
    assert not rig.weld.records
    assert 'exhausted its bounded exposure bracket' in rig.weld.error


def test_cancellation_during_manual_bracket_drains_configuration_without_late_motion(rig):
    rig.mode['failure'] = 'invalid'
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'analyzing')
    rig.mode['configuration_pending'] = True
    pump(rig, lambda: rig.weld.status == 'exposure_recovery')
    before = len(rig.events)
    rig.weld.cancel()
    assert rig.weld.busy
    assert rig.state['laser_configuration']['status'] == 'cancelling'
    rig.state['laser_configuration']['status'] = 'cancelled'
    rig.state['configuring'] = False
    rig.advance()
    assert not rig.weld.busy and rig.weld.status == 'cancelled'
    assert not any(event[0] == 'move' for event in rig.events[before:])
    assert not rig.weld.records


def test_transient_unqualified_native_timing_waits_next_frame_at_same_hold(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    before = len(rig.events)
    cutoff, hold = rig.weld._capture_cutoff, rig.weld._hold_at
    bad = rig.state['laser']['cloud']
    bad['timing']['clock_mapping']['status'] = 'unqualified'
    bad['read_started'] = rig.clock[0]
    rig.advance(update_frame=False)
    assert rig.weld.status == 'capturing'
    assert rig.weld._stations == 0 and rig.mode['detector_calls'] == 0
    assert rig.weld._acquisition_gap_count == 1
    assert rig.weld._acquisition_gaps[-1]['acquisition_key'][2] == bad['sequence']
    assert rig.weld._capture_cutoff == cutoff and rig.weld._hold_at == hold
    assert len(rig.events) == before
    rig.advance(update_frame=False)
    assert rig.weld._acquisition_gap_count == 1  # Same rejected image cannot accumulate gaps.
    rig.advance()
    assert rig.mode['detector_calls'] == 1
    assert len(rig.events) == before
    assert rig.weld._stations == 0  # No frame becomes a station until its result qualifies.


def test_persistent_unqualified_native_timing_retains_original_station_deadline(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    deadline = rig.weld._phase_started+rig.weld.STATION_TIMEOUT_S
    for _ in range(6):
        rig.state['laser']['cloud']['sequence'] += 1
        rig.state['laser']['cloud']['read_started'] = rig.clock[0]
        rig.state['laser']['cloud']['timing']['clock_mapping']['status'] = 'unqualified'
        rig.advance(update_frame=False)
    assert rig.weld._phase_started+rig.weld.STATION_TIMEOUT_S == deadline
    assert rig.weld._acquisition_gap_count == 6
    assert not rig.weld.records and rig.weld.endpoints == {'positive': None, 'negative': None}
    rig.advance(dt=deadline-rig.clock[0]+.01, update_frame=False)
    assert rig.weld.status == 'error'
    assert 'within 30 seconds' in rig.weld.error
    assert not any(event[0] == 'move' for event in rig.events)


def test_search_extent_uses_measured_plane_progress_instead_of_nominal_commands(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld._center is not None and bool(rig.weld.records))
    weld = rig.weld
    origin = np.asarray(weld._search_frame['center_camera_origin_base_m'])
    normal = np.asarray(weld._search_frame['normal_base'])
    measurement = copy.deepcopy(weld._pending_measurement)
    measurement['direction'], measurement['distance_mm'] = 1, 13.
    optical = np.eye(4)
    optical[:3, 3] = origin+normal*.01225+np.array([.04, 0., .03])
    measurement['base_from_camera'] = optical.tolist()
    weld._pending_native_path = weld._directory()/'measured-span-fixture.json'
    weld._record_search_station(measurement)
    weld._direction, weld._missing[1], weld._distance[1] = 1, 3, 13.
    weld._advance()
    assert weld.endpoints['positive'] is None
    assert weld._distance[1] == 14.
    assert weld._search_extents[1] == pytest.approx(12.25)
    optical[:3, 3] += normal*.00025
    measurement['base_from_camera'] = optical.tolist()
    measurement['distance_mm'] = 14.
    weld._record_search_station(measurement)
    weld._advance()
    assert weld.endpoints['positive']['confirmed']
    assert weld.endpoints['positive']['measured_search_extent_mm'] == pytest.approx(12.5)


def test_single_detected_seam_point_cannot_become_a_degenerate_replay_path(rig):
    rig.mode['ends_mm'] = .1
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    assert len(rig.weld.records) == 1
    assert 'two distinct measured seam points' in rig.weld.error
    assert not rig.weld.state(camera_state=rig.state)['replay_ready']


def test_quantized_manual_exposure_uses_verified_sensor_readback_without_requested_tolerance(rig):
    rig.state['controls']['exposure_us'] = 5000.
    dispatch, find = rig.service.camera.dispatch, rig.service.camera.find_weld_seam

    def quantized_dispatch(request):
        dispatch(request)
        if request.get('controls', {}).get('exposure_us') == 4000.:
            rig.state['controls']['exposure_us'] = 4007.25
            rig.state['laser_configuration']['readback'] = copy.deepcopy(rig.state['controls'])

    def exposure_sensitive(cloud, **kwargs):
        if abs(cloud['controls']['exposure_us']-4007.25) > 1e-8:
            return dict(status='invalid', found=False, quality={'good': False, 'reason': 'root too dim or clipped'})
        return find(cloud, **kwargs)

    rig.service.camera.dispatch, rig.service.camera.find_weld_seam = quantized_dispatch, exposure_sensitive
    rig.weld.start({})
    pump(rig, lambda: bool(rig.weld.records))
    assert rig.weld._manual_requested_us == 4000.
    assert rig.weld._manual_exposure_us == 4007.25
    assert rig.weld._bracket_history[-1]['configuration_readback']['exposure_us'] == 4007.25


def test_manual_bracket_waits_cached_controls_to_match_verified_configuration_at_same_hold(rig):
    rig.mode['failure'] = 'invalid'
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'exposure_recovery' and rig.weld._bracket_kind != 'ae_lock')
    expected = copy.deepcopy(rig.state['laser_configuration']['readback'])
    rig.state['controls']['exposure_us'] = 4000.
    before = len(rig.events)
    rig.advance()
    assert rig.weld.status == 'exposure_recovery'
    assert rig.weld._manual_exposure_us == 4000.  # Initial AE lock remains selected until new readback matches.
    assert len(rig.events) == before
    rig.state['controls'] = expected
    rig.advance()
    assert rig.weld.status == 'capturing'
    assert rig.weld._manual_exposure_us == expected['exposure_us']
    assert len(rig.events) == before


def test_bounded_gain_recovery_follows_exposures_and_preserves_actual_selected_controls(rig):
    rig.state['control_limits'] = {'gain_db': [0., 48.]}
    rig.state['controls']['gain_db'] = 6.
    original = rig.service.camera.find_weld_seam

    def gain_sensitive(cloud, **kwargs):
        if cloud['controls']['gain_db'] < 18. or cloud['controls']['exposure_us'] < 15000.:
            return dict(status='invalid', found=False, quality={'good': False, 'reason': 'weak seam root'})
        return original(cloud, **kwargs)

    rig.service.camera.find_weld_seam = gain_sensitive
    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 2)
    requests = [event[1]['controls'] for event in rig.events if event[0] == 'camera' and
                event[1]['action'] == 'camera_laser_configure' and event[1].get('controls')
                and not event[1]['request_id'].endswith('-ae-lock')]
    assert requests == [dict(exposure_us=16000.), dict(exposure_us=8000.),
                        dict(exposure_us=16000., gain_db=12.), dict(exposure_us=16000., gain_db=18.)]
    assert rig.weld._manual_configuration_readback == {'exposure_us': 16000., 'gain_db': 18.}
    assert rig.weld._bracket_history[-1]['kind'] == 'gain'
    assert rig.weld._bracket_history[-1]['gain_ceiling_db'] == 24.


@pytest.mark.parametrize('base_gain,sensor_max,expected_gain_trials', [(12., 48., [18., 24.]),
    (6., 10., [10.]), (30., 48., [])])
def test_gain_recovery_is_sensor_bounded_capped_at_24_and_deduplicated(rig, base_gain, sensor_max, expected_gain_trials):
    rig.state['control_limits'] = {'gain_db': [0., sensor_max]}
    rig.state['controls']['gain_db'] = base_gain
    rig.state['controls']['exposure_us'] = 5000.
    rig.mode['failure'] = 'invalid'
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'error')
    requests = [event[1]['controls'] for event in rig.events if event[0] == 'camera' and
                event[1]['action'] == 'camera_laser_configure' and event[1].get('controls')
                and not event[1]['request_id'].endswith('-ae-lock')]
    assert [request['gain_db'] for request in requests if 'gain_db' in request] == expected_gain_trials
    assert len(requests) <= 5
    assert all(request['gain_db'] <= 24. for request in requests if 'gain_db' in request)


def test_optical_exhaustion_on_positive_side_still_searches_negative_and_replays_only_known_section(rig):
    original = rig.service.camera.find_weld_seam
    rig.mode['ends_mm'] = 3.

    def unresolved_positive(cloud, **kwargs):
        if rig.pose[1, 3] > .0035:
            return dict(status='invalid', found=False, quality={'good': False, 'reason': 'unobservable root'})
        return original(cloud, **kwargs)

    rig.service.camera.find_weld_seam = unresolved_positive
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'incomplete' and not rig.weld.busy)
    assert rig.weld.endpoints['positive']['confirmed'] is False
    assert rig.weld.endpoints['positive']['reason'] == 'visibility_unresolved'
    assert rig.weld.endpoints['negative']['confirmed'] is True
    assert any(record['direction'] == -1 for record in rig.weld.records)
    state = rig.weld.state(camera_state=rig.state)
    assert state['coverage_complete'] is False
    assert state['replay_scope'] == 'measured_section_only'
    assert state['replay_ready']
    rig.weld.replay()
    pump(rig, lambda: rig.weld.status == 'incomplete' and not rig.weld.busy)
    assert rig.weld._replay_index == len(rig.weld.records)
    assert len(rig.plans) == len(rig.weld.records)
    assert 'Measured weld section replay completed' in rig.weld.message


def test_motion_failure_never_triggers_the_optical_other_branch_exception(rig):
    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 2)
    rig.service.move_pose = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError('actual IK path blocker'))
    pump(rig, lambda: rig.weld.status == 'error')
    assert rig.weld.endpoints['positive'] is None
    assert not any(record['direction'] == -1 for record in rig.weld.records)
    assert 'actual IK path blocker' in rig.weld.error


@pytest.mark.parametrize('failure_status', ['error', 'cancelled'])
def test_terminal_partial_points_export_without_motion_and_keep_finding_reason(rig, failure_status):
    import json
    from pathlib import Path

    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 3)
    rig.weld.error = 'retained finding diagnostic'
    rig.weld.cancel(status=failure_status, reason='retain known section')
    before = len(rig.events)
    rig.weld.dispatch({'action': 'weld_export'})
    rig.advance()
    assert len(rig.events) == before
    manifest = json.loads(Path(rig.weld._export_path).read_text())
    assert manifest['status'] == 'incomplete' and manifest['coverage_complete'] is False
    assert manifest['finding_status'] == failure_status
    assert manifest['finding_error'] == 'retained finding diagnostic'
    assert manifest['replay_scope'] == 'measured_section_only'
    assert len(manifest['path']) >= 3 and manifest['native_files']
    assert rig.weld.state(camera_state=rig.state)['replay_ready']


def test_one_point_partial_can_export_but_cannot_replay(rig):
    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) == 1)
    rig.weld.cancel()
    rig.weld.dispatch({'action': 'weld_export'})
    rig.advance()
    assert rig.weld._export_path
    assert len(rig.weld._result['path']) == 1
    assert not rig.weld.state(camera_state=rig.state)['replay_ready']
    with pytest.raises(ValueError, match='two distinct'):
        rig.weld.replay()


@pytest.mark.parametrize('failure', ['unpaired', 'zero_exposure'])
def test_mapped_clock_without_owned_event_pair_or_positive_exposure_only_skips_that_frame(rig, failure):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    timing = rig.state['laser']['cloud']['timing']
    if failure == 'unpaired':
        timing['motion_pairing_supported'] = False
    else:
        timing['exposure_host_start_s'] = timing['exposure_host_end_s']
    before = len(rig.events)
    rig.advance(update_frame=False)
    assert rig.weld.status == 'capturing' and rig.weld._acquisition_gap_count == 1
    assert rig.mode['detector_calls'] == 0 and len(rig.events) == before
    rig.advance()
    assert rig.mode['detector_calls'] == 1 and len(rig.events) == before


def test_replay_verifies_native_file_bytes_on_worker_before_any_motion(rig):
    from pathlib import Path

    complete(rig)
    raw = Path(rig.weld._result['native_files'][0]['path'])
    raw.write_bytes(raw.read_bytes()+b'\n')
    before = len(rig.events)
    rig.weld.replay()
    assert len(rig.events) == before
    rig.advance()
    assert rig.weld.status == 'error'
    assert 'bytes differ' in rig.weld.error
    assert not any(event[0] == 'move' for event in rig.events[before:])


def test_initial_ae_jitter_can_advance_more_than_three_generations_then_freezes_once_before_capture(rig):
    rig.mode['ae_reliable'] = False
    rig.weld.start({})
    pump(rig, lambda: rig.weld._hold_at is not None)
    hold = rig.weld._hold_at
    for index in range(5):
        rig.state['generation'] += 1
        rig.state['controls']['exposure_us'] = 4000.+index*20.
        rig.advance()
        assert rig.weld.status == 'settling'
        assert not rig.weld.records and rig.mode['detector_calls'] == 0
    assert len(rig.weld._generation_transitions) == 5
    rig.mode['ae_reliable'] = True
    rig.advance()
    assert rig.weld.status == 'exposure_recovery'
    assert rig.weld._bracket_kind == 'ae_lock'
    assert rig.weld._bracket_attempts == 0
    assert rig.mode['detector_calls'] == 0
    rig.advance()
    assert not rig.weld._ae_lock_required
    assert rig.weld._hold_at == hold
    completed = rig.weld._bracket_history[0]['completed_monotonic']
    pump(rig, lambda: rig.weld._future_kind == 'analyze')
    lock_requests = [event for event in rig.events if event[0] == 'camera' and
                     event[1].get('request_id', '').endswith('-ae-lock')]
    assert len(lock_requests) == 1
    measurement = rig.weld._pending_measurement
    assert measurement['capture_cutoff'] >= completed+.4-1e-9
    assert measurement['exposure_interval_s'][0]-measurement['exposure_uncertainty_s'] >= measurement['capture_cutoff']
    assert measurement['exposure_control']['bracket_kind'] == 'ae_lock'
    assert measurement['cloud']['controls'] == rig.weld._manual_configuration_readback
    assert len([event for event in rig.events if event[0] == 'hold']) == 1
    assert not any(event[0] == 'move' for event in rig.events)


def test_cancellation_while_initial_ae_lock_pending_drains_without_late_capture_or_movement(rig):
    rig.mode['ae_reliable'] = False
    rig.weld.start({})
    pump(rig, lambda: rig.weld._hold_at is not None)
    rig.mode['configuration_pending'], rig.mode['ae_reliable'] = True, True
    rig.advance()
    assert rig.weld._bracket_kind == 'ae_lock' and rig.weld._configuration_pending
    before = len(rig.events)
    rig.weld.cancel()
    assert rig.weld.busy and rig.state['laser_configuration']['status'] == 'cancelling'
    rig.state['laser_configuration']['status'] = 'cancelled'
    rig.state['configuring'] = False
    rig.advance()
    assert not rig.weld.busy and rig.weld.status == 'cancelled'
    assert rig.mode['detector_calls'] == 0 and not rig.weld.records
    assert not any(event[0] == 'move' for event in rig.events[before:])


def test_restored_weld_replay_does_not_warm_or_freeze_a_camera(rig):
    complete(rig)
    rig.state['enabled'] = False
    restored = WeldSeamService(rig.service, clock=lambda: rig.clock[0], executor=ImmediateExecutor(),
                               replay_preflight=rig.weld._replay_preflight)
    before = len(rig.events)
    restored.replay()
    for _ in range(1000):
        rig.clock[0] += .1
        rig.pose[:] = rig.goal
        restored.tick()
        if restored.status == 'complete' and not restored.busy:
            break
    assert restored.status == 'complete' and restored._replay_index == len(restored.records)
    assert not any(event[0] == 'camera' for event in rig.events[before:])
    restored.close()


def test_true_post_hold_endpoint_miss_replans_original_goal_once_and_keeps_deadline_and_optical_budget(rig):
    import json
    from pathlib import Path

    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    original_goal, started = rig.weld._goal.copy(), rig.weld._phase_started
    old_hold = rig.weld._hold_at
    rig.weld._bracket_attempts = 2
    rig.weld._bracket_candidates = [dict(kind='gain', exposure_us=16000., gain_db=12.)]
    candidates = copy.deepcopy(rig.weld._bracket_candidates)
    rig.pose[0, 3] += .0006
    for _ in range(4):
        rig.advance(follow=False)
    moves = [event[1] for event in rig.events if event[0] == 'move']
    assert len(moves) == 1
    np.testing.assert_array_equal(moves[0], original_goal)
    assert rig.weld._phase_started == started
    assert rig.weld._bracket_attempts == 2 and rig.weld._bracket_candidates == candidates
    assert rig.weld._station_retry_attempt == 1
    assert rig.weld._hold_at is None
    pump(rig, lambda: rig.weld._future_kind == 'analyze')
    measurement = rig.weld._pending_measurement
    assert rig.weld._hold_at > old_hold
    assert len([event for event in rig.events if event[0] == 'hold']) == 2
    assert measurement['station_retry']['status'] == 'complete'
    assert measurement['station_retry']['before_evidence']['position_error_m'] > .0005
    assert measurement['station_retry']['after_evidence']['position_error_m'] <= .0005
    assert measurement['exposure_interval_s'][0]-measurement['exposure_uncertainty_s'] >= measurement['capture_cutoff']
    saved = json.loads(Path(measurement['station_retry']['diagnostic_path']).read_text())
    assert saved['status'] == 'complete' and saved['station_deadline'] == started+30.


def test_stale_sdk_feedback_cannot_trigger_an_endpoint_replan(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    goal = rig.weld._goal.copy()
    rig.pose[0, 3] += .0006
    rig.feedback['fresh'] = False
    for _ in range(8):
        rig.advance(follow=False)
    assert not rig.weld._endpoint_retries
    assert not any(event[0] == 'move' for event in rig.events)
    assert rig.weld._station_gate['gate'] == 'fresh_stationary_feedback'
    rig.pose[:] = goal
    rig.feedback['fresh'] = True
    pump(rig, lambda: rig.weld._future_kind == 'analyze')
    assert not rig.weld._endpoint_retries


def test_persistent_miss_after_one_identical_replan_fails_original_deadline_without_another_replan(rig):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    deadline = rig.weld._phase_started+30.
    rig.pose[0, 3] += .0006
    for _ in range(4):
        rig.advance(follow=False)
    pump(rig, lambda: len([event for event in rig.events if event[0] == 'hold']) == 2)
    rig.pose[0, 3] += .0006
    for _ in range(8):
        rig.advance(follow=False)
    assert len([event for event in rig.events if event[0] == 'move']) == 1
    rig.advance(dt=deadline-rig.clock[0]+.01, follow=False)
    assert rig.weld.status == 'error'
    assert 'gate=endpoint_position' in rig.weld.error and 'endpoint_replans=1' in rig.weld.error
    assert len([event for event in rig.events if event[0] == 'move']) == 1
    assert rig.weld._station_retry['status'] == 'failed'


@pytest.mark.parametrize('blocker', ['mode', 'calibration', 'collision'])
def test_endpoint_retry_cannot_bypass_changed_mode_calibration_or_normal_planner_rejection(rig, blocker):
    rig.weld.start({})
    pump(rig, lambda: rig.weld.status == 'capturing' and rig.weld._capture_cutoff is not None)
    rig.pose[0, 3] += .0006
    if blocker == 'mode':
        rig.service.ctrl.mode = 'Gravity comp'
    elif blocker == 'calibration':
        rig.state['handeye']['optical_intrinsics_id'] = 'changed'
    else:
        rig.service.move_pose = lambda *_a, **_k: (_ for _ in ()).throw(ValueError('normal planner collision rejection'))
    for _ in range(8):
        rig.advance(follow=False)
        if rig.weld.status == 'error':
            break
    assert rig.weld.status == 'error'
    assert not any(event[0] == 'move' for event in rig.events)
    assert rig.weld.endpoints == {'positive': None, 'negative': None}
    if blocker == 'collision':
        assert rig.weld._station_retry['status'] == 'failed'
        assert 'collision' in rig.weld.error


@pytest.mark.parametrize('terminal_status', ['error', 'cancelled'])
def test_terminal_partial_is_automatically_durable_across_close_restore_without_movement(rig, terminal_status):
    import hashlib
    import json
    from pathlib import Path

    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 3)
    geometry = rig.weld.state(camera_state=rig.state)['geometry']['seam_points_base_m']
    raw_hashes = {entry['path']: hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()
                  for entry in rig.weld._native_files}
    rig.weld.error = 'actual terminal diagnostic'
    rig.weld.cancel(status=terminal_status)
    rig.service.ctrl.closed = True
    rig.weld.close()
    assert rig.weld._manifest_path().exists()
    saved = json.loads(rig.weld._manifest_path().read_text())
    assert saved['finding_status'] == terminal_status
    assert saved['finding_error'] == 'actual terminal diagnostic'
    before = len(rig.events)
    restored = WeldSeamService(rig.service, clock=lambda: rig.clock[0], executor=ImmediateExecutor())
    assert restored.state(camera_state=rig.state)['geometry']['seam_points_base_m'] == geometry
    assert len(rig.events) == before
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in raw_hashes.items())
    restored.close()
    assert len(rig.events) == before


def test_shutdown_drain_hashes_uncommitted_native_write_and_never_accepts_or_moves_after_cancel(rig):
    import json

    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 2 and rig.weld._future_kind == 'save_observation')
    points = len(rig.weld.records)
    uncommitted = rig.weld._pending_native_path
    assert uncommitted.exists()  # File worker finished; owner has not accepted it.
    rig.weld.abort_for_shutdown()
    before = len(rig.events)
    rig.service.ctrl.closed = True
    rig.weld.close()
    assert len(rig.weld.records) == points and len(rig.events) == before
    result = json.loads(rig.weld._manifest_path().read_text())
    entry = next(entry for entry in result['native_files'] if entry['path'] == str(uncommitted))
    assert entry['disposition'] == 'reacquire'
    assert result['finding_status'] == 'interrupted'
    assert len(result['path']) == points


def test_automatic_partial_save_failure_is_explicit_and_does_not_move_the_arm(rig):
    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 2)
    writer = rig.weld._save_result
    rig.weld._save_result = lambda *_: (_ for _ in ()).throw(OSError('disk unavailable'))
    rig.weld.cancel()
    before = len(rig.events)
    state = rig.weld.state(camera_state=rig.state)
    assert state['save_status'] == 'error' and 'disk unavailable' in state['save_error']
    assert len(rig.events) == before
    rig.weld._save_result = writer
    rig.weld.dispatch({'action': 'weld_export'})
    rig.advance()


@pytest.mark.parametrize('operation', ['start', 'clear'])
def test_completed_old_partial_save_cannot_publish_an_export_link_into_a_new_owner(rig, operation):
    rig.weld.start({})
    pump(rig, lambda: len(rig.weld.records) >= 2)
    rig.weld.cancel()
    # Simulate a file future completing between two service ticks.
    rig.weld._partial_future = rig.weld.executor.submit(rig.weld._save_result, copy.deepcopy(rig.weld._result))
    assert rig.weld._partial_future.done()
    if operation == 'start':
        rig.weld.start({})
    else:
        rig.weld.dispatch({'action': 'weld_clear'})
    rig.advance()
    assert rig.weld._export_path is None
    assert rig.weld._partial_future is None


def test_cancelling_replay_does_not_rewrite_the_successful_finding_as_partial(rig):
    import json
    from pathlib import Path

    complete(rig)
    manifest_path = Path(rig.weld._export_path)
    original = manifest_path.read_bytes()
    rig.weld.replay()
    rig.weld.cancel()
    rig.advance()
    assert manifest_path.read_bytes() == original
    assert rig.weld._result['status'] == 'complete'
    assert rig.weld._result['coverage_complete'] is True
    assert json.loads(original)['finding_status'] == 'complete'
