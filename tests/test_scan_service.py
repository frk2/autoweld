"""Bridge/collector integration on pure fake owners; no robot model or devices."""
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_motion_scan_archive import frame, metadata

from autoweld.continuous_scanning import ContinuousScanCoordinator
from autoweld.motion_scan_archive import MotionScanArchive
from autoweld.scan_service import ContinuousScanService, ScanPlanProbe
from autoweld.service_commands import ServiceCommandBridge


class Ring:
    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()
    def add(self, cloud):
        with self.lock:
            self.frames.append(cloud)
    def recent_laser_clouds(self, after_key=None):
        with self.lock:
            frames = tuple(c for c in self.frames if after_key is None or c['timing']['device_frame_id'] > after_key[1])
            newest = None if not self.frames else ('session',self.frames[-1]['timing']['device_frame_id'])
            return dict(clouds=frames,newest_key=newest,overrun=False,session_changed=False)
    def latest_laser_cloud(self):
        with self.lock:
            return self.frames[-1] if self.frames else None


def receive_registration(registered):
    result = dict(registered)
    result.update(registration_reference='measured_kernel_rx',registration_method='measured_output_receive_reference',
        reference_registered=True,reference_timing_qualified=True,timing_qualified=False,
        physical_sample_timing_qualified=False,firmware_sample_age_s=None,physical_spatial_error_bound_m=None,
        reference_timing_uncertainty_s=result['timing_uncertainty_s'],reference_clock_uncertainty_s=.0001,
        reference_spatial_error_bound_m=result['spatial_error_bound_m'],reference_joint_error_rad=result['joint_error_rad'],
        max_reference_timing_uncertainty_s=.002,max_reference_spatial_error_m=.001,max_reference_gap_s=.1,
        timing_uncertainty_s=None,spatial_error_bound_m=None,joint_error_rad=None,
        applied_delay_s=0.,applied_empirical_delay_s=0.,delay_qualification_id=None)
    return result


def native(index, *, x=0., empty=False, rejected=False):
    c, registered = frame(index)
    now = time.monotonic()
    t = c['timing']
    t.update(exposure_host_mid_s=now,exposure_host_midpoint_s=now,
        exposure_host_start_s=now-.002,exposure_host_end_s=now+.002,exposure_duration_us=4000.)
    count = 0 if empty else 30
    points = np.column_stack((np.linspace(-.005,.005,count),np.zeros(count),np.full(count,.14)))
    c.update(xyz_m=points.tolist(),uv=np.column_stack((np.arange(count)+100,np.full(count,200))).tolist(),
        strength=np.full(count,1. if rejected else 100.).tolist(),width_px=np.full(count,2.).tolist(),
        image_size=[1440,1080], read_started=now, pose_x=x)
    registered.update(exposure_monotonic_s=now,exposure_interval_s=[now-.002,now+.002],
        robot_brackets_monotonic_s=np.tile([now-.005,now+.005],(6,1)),robot_max_gap_s=.011)
    pose = np.eye(4)
    pose[0,3] = x
    registered['base_from_tcp'] = pose
    registered['points_world_m'] = points+np.array([x,0,0])
    registered = receive_registration(registered)
    c['_test_registration'] = registered
    return c


def harness(tmp_path):
    service = ContinuousScanService.__new__(ContinuousScanService)
    ring = Ring()
    service.service = SimpleNamespace(camera=ring)
    service._lock = threading.Lock()
    service.archive = MotionScanArchive(tmp_path)
    meta = metadata()
    meta.update(registration_reference='measured_kernel_rx',physical_sample_timing_qualified=False,
        firmware_sample_age_s=None,registration_reference_metadata=dict(history_id='history',clock_mapping_id='clock',
            source_id='camera',optical_intrinsics_id='optics',acquisition_session_id='session'))
    service.archive.begin(meta)
    service.history = SimpleNamespace(history_id='history',reference_clock='kernel_receive')
    service.registrar = SimpleNamespace(register_receive_reference=lambda c:dict(c['_test_registration']))
    service.coordinator = SimpleNamespace(settings=SimpleNamespace(width_mm=100.,height_mm=100.),
        _generation=1,busy=True,_cancel_event=threading.Event(),_started=time.monotonic())
    service._last_registered = service._last_registered_time = None
    service._last_key = service._previous_centroid = service._last_enqueued_key = service._last_ring_key = None
    service._pending_clouds = deque(maxlen=64)
    service._capture_cutoff = 0.
    service._area_camera_shift = np.zeros(2)
    service._area_plan_active = service._footprint_mode = False
    service._area_repair_pad_m = .002
    service._main_scan = True
    service._max_path_gap = 0.
    service._quality_gaps = []
    service._surface_views = deque(maxlen=2048)
    service._collector_state = service._collector_error = service._collector_future = None
    service._latest_archive_snapshot = service._latest_archive_snapshot_monotonic = None
    service._collector_executor = ThreadPoolExecutor(max_workers=1)
    service._collector_stop = threading.Event()
    service._reported_frames = 0
    service._native_intervals = deque(maxlen=2048)
    service._last_native_stamp = None
    service._selected_feed_mm_s = None
    service._active_plan = SimpleNamespace(path=SimpleNamespace(duration=1.))
    service._path_started = time.monotonic()
    service._max_fraction = 0.
    context = dict(settings=dict(width_mm=100.,height_mm=100.,max_duration_s=900.),
        data={'baseline':{'handeye':{'tcp_from_camera':np.eye(4).tolist()},'baseline_optical':np.eye(4).tolist()}},
        generation=1,cancel_event=service.coordinator._cancel_event)
    return service,ring,context


def clean(service):
    service._collector_stop.set()
    if service._collector_future:
        service._collector_future.result(timeout=2.)
    service._collector_executor.shutdown()


def test_independent_collector_consumes_native_ring_without_main_tick_sampling(tmp_path):
    service,ring,context = harness(tmp_path)
    try:
        service._start_collector(context)
        for index in range(12):
            ring.add(native(index,x=index*.0003))
            time.sleep(.006)  # Deliberately above a50Hz main tick's publication rate.
        deadline = time.monotonic()+2.
        while service.archive.state()['native_frame_count'] < 12 and time.monotonic() < deadline:
            time.sleep(.005)
        service._stop_collector()
        state = service.archive.state()
        assert state['native_frame_count'] == state['accepted'] == 12
        assert state['gaps'] == 0
        assert service._collector_error is None
        status = service._capture(context)
        assert status['qualified']
        assert not service._capture(context)['qualified']  # No stale qualified refresh.
    finally:
        clean(service)


def test_cancelled_collector_exits_and_cannot_append_late_frames(tmp_path):
    service,ring,context = harness(tmp_path)
    try:
        ring.add(native(0))
        service._start_collector(context)
        time.sleep(.03)
        context['cancel_event'].set()
        service._stop_collector()
        count = service.archive.state()['native_frame_count']
        ring.add(native(1))
        time.sleep(.03)
        assert service.archive.state()['native_frame_count'] == count
        assert service._collector_future.done()
    finally:
        clean(service)


def test_rejected_frame_does_not_reset_good_coverage_or_qualify_liveness(tmp_path):
    service,ring,context = harness(tmp_path)
    try:
        ring.add(native(0,x=0.))
        assert service._capture_one(context)['qualified']
        ring.add(native(1,x=.0004,rejected=True))
        assert not service._capture_one(context)['qualified']
        assert service._last_registered[0] == 0.
        ring.add(native(2,x=.0012))
        assert service._capture_one(context)['qualified']
        assert len(service._quality_gaps) == 1
        assert service.archive.state()['rejected'] == 1
    finally:
        clean(service)


def test_dense_good_neighbors_can_cover_one_bad_frame_and_no_return_reacquires(tmp_path):
    service,ring,context = harness(tmp_path)
    try:
        ring.add(native(0,x=0.))
        service._capture_one(context)
        ring.add(native(1,x=.0003,rejected=True))
        service._capture_one(context)
        ring.add(native(2,x=.0006))
        service._capture_one(context)
        assert not service._quality_gaps
        ring.add(native(3,x=.0008,empty=True))
        assert service._capture_one(context)['qualified']
        assert service._previous_centroid is None
        ring.add(native(4,x=.001))
        assert service._capture_one(context)['qualified']
        assert service.archive.state()['no_return'] == 1
    finally:
        clean(service)


def test_native_ring_overrun_is_explicit_instead_of_silent_loss(tmp_path):
    service,ring,context = harness(tmp_path)
    ring.recent_laser_clouds = lambda _key:dict(clouds=(),overrun=True)
    try:
        with pytest.raises(RuntimeError,match='overran'):
            service._capture_one(context)
    finally:
        clean(service)


def test_worker_commands_execute_only_when_main_owner_drains(tmp_path):
    service,_,context = harness(tmp_path)
    owner = threading.get_ident()
    calls = []
    service.service.ctrl = SimpleNamespace(hold=lambda **_kw:calls.append(threading.get_ident()))
    service.bridge = ServiceCommandBridge(service._execute,service._valid_context)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            request = executor.submit(service._command,'hold',{},context)
            deadline = time.monotonic()+1.
            while not service.bridge.state()['pending'] and time.monotonic() < deadline:
                time.sleep(.001)
            assert not calls
            service.bridge.drain()
            assert request.result(timeout=1.) == {'accepted':True}
        assert calls == [owner]
    finally:
        service.bridge.close()
        clean(service)


def test_prepare_uses_fresh_measured_start_without_historical_scan_dependency(tmp_path):
    service,_,context = harness(tmp_path)
    fresh = np.eye(4)
    fresh[0,3] = .123
    service._camera_request = lambda *_args:None
    service._command = lambda *_args:fresh.tolist()
    service._settle = lambda *_args:fresh.copy()
    service.service.runtime = tmp_path/'never-created'
    context['settings']['start_pose'] = 'Tube Scanning'
    try:
        result = service._prepare(context)
        np.testing.assert_array_equal(result['baseline_pose'],fresh)
        assert not service.service.runtime.exists()
    finally:
        clean(service)


def test_probe_rejects_enabled_owner_without_submitting_work():
    probe = ScanPlanProbe(SimpleNamespace(ctrl=SimpleNamespace(closed=False,motion_plan=None)))
    try:
        with pytest.raises(RuntimeError,match='disabled'):
            probe.dispatch({'operation':'submit'})
        assert probe.future is None
    finally:
        probe.close()


def test_measured_coverage_reports_bins_not_frame_count(tmp_path):
    service,_,_ = harness(tmp_path)
    try:
        state = service._display_saved(dict(visited=800,captures=800,
            coverage={'observed_camera_xy_cells_1mm':321}))
        assert state['visited'] == 321 and state['total'] == 10201
        assert state['captures'] == 800
    finally:
        clean(service)


def test_baseline_depth_goals_keep_frozen_orientation_despite_measured_rotation_drift(tmp_path):
    from scipy.spatial.transform import Rotation
    service,_,context = harness(tmp_path)
    handeye = dict(tcp_from_camera=np.eye(4).tolist(),units='m',coordinate_frame='tcp')
    context['data']['prepare'] = {'baseline_pose':np.eye(4).tolist()}
    context['settings']['optical_roll_deg'] = -45.
    service.camera_state = lambda:{'handeye':handeye}
    service._make_footprint = lambda *_args:{'reference_plane_z_m':.14}
    measured = np.eye(4)
    service.snapshot = lambda:{'tcp_pose':measured.tolist()}
    commands = []
    def command(_action,payload,_context):
        goal = np.asarray(payload['pose'])
        service._check_envelope(goal)
        commands.append(goal.copy())
    service._command = command
    def settle(_context,goal):
        measured[:] = goal
        # A valid small feedback error at each settled endpoint must never be
        # copied into the next command, even across several depth corrections.
        measured[:3,:3] = Rotation.from_euler('x',.004).as_matrix()@measured[:3,:3]
        return measured.copy()
    service._settle = settle
    depths = iter([.155,.150,.145,.140])
    def good():
        depth = next(depths)
        return dict(xyz_m=[[0.,0.,depth]],source_id='camera',optical_intrinsics_id='optics',
            calibration_id='laser',acquisition_session_id='session'),np.eye(4),dict(indices=[0],centroid_camera_m=[0.,0.,depth])
    service._good_observation = good
    try:
        result = service._baseline(context)
        assert len(commands) == 4
        fixed = Rotation.from_euler('z',-45.,degrees=True).as_matrix()
        for goal in commands:
            np.testing.assert_allclose(goal[:3,:3],fixed,atol=1e-14)
        np.testing.assert_allclose(np.diff([goal[:3,3] for goal in commands],axis=0),
            np.tile([0.,0.,.005],(3,1)),atol=1e-14)
        np.testing.assert_allclose(result['fixed_optical_rotation'],fixed,atol=1e-14)
    finally:
        clean(service)


def test_settle_failure_persists_bounded_target_miss_diagnostics_before_raise(tmp_path,monkeypatch):
    import json
    service,ring,context = harness(tmp_path)
    service.service.runtime = tmp_path
    clock = FastSettleClock(monkeypatch)
    context['phase'] = 'baseline'
    expected = np.eye(4)
    held = False
    commands = []
    def command(action,payload,_context):
        nonlocal held
        commands.append((action,payload))
        held = True
    service._command = command
    def snapshot():
        pose = expected.copy()
        if held:
            pose[0,3] = .0006
        now = time.monotonic()
        return dict(tcp_pose=pose.tolist(),q=[0.]*6,motion_active=False,
            feedback=dict(fresh=True,stationary=True,measured_at=now-.001))
    service.snapshot = snapshot
    ring.add(native(0))
    def bounded_wait(_context,predicate,_timeout,reason):
        for _ in range(200):
            clock.advance(.05)
            result = predicate(snapshot())
            if isinstance(result,np.ndarray) or result:
                return result
        raise TimeoutError(reason)
    service._wait = bounded_wait
    try:
        with pytest.raises(TimeoutError,match='diagnostics:'):
            service._settle(context,expected,_retry_state={'attempt':1})
        files = list((tmp_path/'evidence').glob('continuous-settle-*.json'))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record['phase'] == 'baseline'
        assert record['summary']['hold_count'] == 1
        assert record['summary']['poll_count'] > 200
        assert len(record['samples']) == 128
        assert all(s['gate'] == 'target_position' and not s['arrived'] for s in record['samples'])
        assert all(s['target_position_error_m'] == .0006 for s in record['samples'])
        assert all(s['feedback_stationary'] and s['feedback_fresh'] for s in record['samples'])
        assert record['samples'][-1]['native_sequence'] == 0
        assert 'feedback_minus_native_read_started_s' in record['samples'][-1]
        assert commands == [('hold',{})]
    finally:
        clean(service)


def test_settle_diagnostics_distinguish_quiet_jitter_without_relaxing_gates(tmp_path,monkeypatch):
    import json
    service,ring,context = harness(tmp_path)
    service.service.runtime = tmp_path
    clock = FastSettleClock(monkeypatch)
    context['phase'] = 'baseline'
    expected = np.eye(4)
    service._command = lambda *_args:None
    count = 0
    def snapshot():
        nonlocal count
        count += 1
        pose = expected.copy()
        pose[0,3] = .0002 if count%2 else -.0002
        return dict(tcp_pose=pose.tolist(),motion_active=False,
            feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()))
    service.snapshot = snapshot
    ring.add(native(0))
    def bounded_wait(_context,predicate,_timeout,reason):
        for _ in range(200):
            clock.advance(.05)
            result = predicate(snapshot())
            if isinstance(result,np.ndarray) or result:
                return result
        raise TimeoutError(reason)
    service._wait = bounded_wait
    try:
        with pytest.raises(TimeoutError):
            service._settle(context,expected)
        record = json.loads(next((tmp_path/'evidence').glob('continuous-settle-*.json')).read_text())
        assert record['summary']['pre_hold_quiet_resets'] > 0
        assert record['summary']['gates']['pre_hold_position_drift'] > 0
        assert all(s['arrived'] for s in record['samples'])
        assert record['limits']['quiet_position_m'] == .0003
        assert record['limits']['quiet_seconds'] == .4
    finally:
        clean(service)


class FastSettleClock:
    def __init__(self,monkeypatch):
        self.now = time.monotonic()
        monkeypatch.setattr('autoweld.scan_service.time',SimpleNamespace(
            monotonic=lambda:self.now,time_ns=time.time_ns))
    def advance(self,seconds):
        self.now += seconds


def settle_harness(tmp_path,monkeypatch):
    service,_,context = harness(tmp_path)
    service.service.runtime = tmp_path
    context['phase'] = 'baseline'
    clock = FastSettleClock(monkeypatch)
    start = clock.now
    expected = np.eye(4)
    expected[:3,3] = [.1,.2,.3]
    service.cloud = lambda:dict(read_started=clock.now-.001,sequence=round((clock.now-start)*100))
    def wait(_ctx,predicate,timeout,reason):
        deadline = clock.now+timeout
        while True:
            context['cancel_event'].is_set() and pytest.fail('Unexpected cancellation')
            clock.advance(.05)
            value = predicate(service.snapshot())
            if isinstance(value,np.ndarray) or value:
                return value
            if clock.now > deadline:
                raise TimeoutError(reason)
    service._wait = wait
    def snapshot(error):
        pose = expected.copy()
        pose[0,3] += error
        return dict(tcp_pose=pose.tolist(),motion_active=False,q=[0.]*6,
            feedback=dict(fresh=True,stationary=True,measured_at=clock.now))
    return service,context,clock,start,expected,snapshot


def test_pre_hold_requires_continuous_endpoint_stability_not_first_transient(tmp_path,monkeypatch):
    service,context,clock,start,expected,snapshot = settle_harness(tmp_path,monkeypatch)
    calls = []
    def error():
        elapsed = clock.now-start
        return .00045 if elapsed < .1 else .0007 if elapsed < .35 else .0002
    service.snapshot = lambda:snapshot(error())
    service._command = lambda action,payload,_ctx:calls.append((action,clock.now-start,payload))
    try:
        service._settle(context,expected)
        assert len(calls) == 1 and calls[0][0] == 'hold'
        assert calls[0][1] >= .75  # After transient miss and0.4 continuous quiet.
        assert not list((tmp_path/'evidence').glob('continuous-settle-*.json'))
    finally:
        clean(service)


def test_one_exact_replan_then_one_new_hold_recovers_persistent_post_hold_miss(tmp_path,monkeypatch):
    import json
    service,context,clock,start,expected,snapshot = settle_harness(tmp_path,monkeypatch)
    calls = []
    held = replanned = False
    def command(action,payload,_ctx):
        nonlocal held,replanned
        calls.append((action,payload,clock.now))
        if action == 'hold':
            held = True
        elif action == 'move_pose':
            np.testing.assert_array_equal(payload['pose'],expected)
            replanned = True
    service._command = command
    service.snapshot = lambda:snapshot(.0007 if held and not replanned else 0.)
    try:
        result = service._settle(context,expected)
        np.testing.assert_array_equal(result,expected)
        assert [call[0] for call in calls] == ['hold','move_pose','hold']
        record = json.loads(Path(service._last_settle_retry['initial_diagnostic_path']).read_text())
        assert record['summary']['hold_count'] == 1
        assert record['samples'][-1]['target_position_error_m'] == pytest.approx(.0007)
        assert record['samples'][-1]['post_hold_miss_seconds'] >= .2
        assert service._last_settle_retry['status'] == 'complete'
        assert service._last_settle_retry['attempt'] == 1
        assert clock.now-start < 3.
    finally:
        clean(service)


def test_replan_is_once_only_and_cannot_renew_original_post_hold_deadline(tmp_path,monkeypatch):
    import json
    service,context,clock,_,expected,snapshot = settle_harness(tmp_path,monkeypatch)
    calls = []
    held = False
    in_move = False
    def command(action,payload,_ctx):
        nonlocal held,in_move
        calls.append((action,payload,clock.now))
        if action == 'move_pose':
            in_move = True
        elif action == 'hold':
            held,in_move = True,False
    service._command = command
    service.snapshot = lambda:snapshot(.0007 if held and not in_move else 0.)
    try:
        with pytest.raises(TimeoutError):
            service._settle(context,expected)
        assert [call[0] for call in calls] == ['hold','move_pose','hold']
        assert clock.now-calls[0][2] <= 30.1
        assert service._last_settle_retry['status'] == 'failed'
        retry = json.loads(Path(service._last_settle_retry['status_path']).read_text())
        assert retry['maximum_replans'] == retry['attempt'] == 1
        assert retry['status'] == 'failed'
    finally:
        clean(service)


def test_prepare_retry_allows_only_its_exact_original_goal_before_envelope_exists(tmp_path):
    from autoweld.service_commands import ServiceAction
    service,_,_ = harness(tmp_path)
    calls = []
    service.service.ctrl = SimpleNamespace()
    service.service.move_pose = lambda pose,**_kw:calls.append(pose.copy())
    service.coordinator.status = 'prepare'
    service._prepare_goal = np.eye(4)
    service._envelope = None
    try:
        service._execute(ServiceAction.MOVE_POSE,{'pose':np.eye(4).tolist()})
        assert len(calls) == 1
        changed = np.eye(4)
        changed[0,3] = .0001
        with pytest.raises(ValueError,match='established original optical baseline'):
            service._execute(ServiceAction.MOVE_POSE,{'pose':changed.tolist()})
        assert len(calls) == 1
    finally:
        clean(service)


def test_future_measured_bracket_wait_retains_bounded_frame_beyond_300ms(tmp_path,monkeypatch):
    from autoweld.motion_registration import RegistrationRejected
    service,ring,context = harness(tmp_path)
    source = native(0)
    stamp = source['timing']['exposure_host_mid_s']
    clock = FastSettleClock(monkeypatch)
    clock.now = stamp+.35
    ring.add(source)
    attempts = 0
    def register(cloud):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RegistrationRejected('Exposure has no measured motor bracket; extrapolation is forbidden')
        return dict(cloud['_test_registration'])
    service.registrar = SimpleNamespace(register_receive_reference=register)
    try:
        assert service._capture_one(context) is None
        assert len(service._pending_clouds) == 1
        assert service._pending_clouds.maxlen == 64
        clock.now = stamp+.36
        assert service._capture_one(context)['qualified']
        assert len(service._pending_clouds) == 0
        assert service.archive.state()['accepted'] == 1
    finally:
        clean(service)


def test_default_reference_setup_never_runs_empirical_probe_or_fabricates_delay(tmp_path,monkeypatch):
    service,ring,context = harness(tmp_path)
    service.fk = SimpleNamespace()
    service.service.runtime = tmp_path
    baseline = context['data']['baseline']
    baseline['handeye'] = metadata()['handeye']
    baseline['identity'] = ['camera','optics','laser','session']
    source = native(0)
    ring.add(source)
    def forbidden_probe(*_args,**_kwargs):
        pytest.fail('Default scan must not run empirical timing experiment')
    monkeypatch.setattr('autoweld.scan_delay_phase.run_delay_experiment',forbidden_probe)
    try:
        result = service._delay_calibration(context)
        assert result['reference_registered'] and result['reference_timing_qualified']
        assert result['physical_sample_timing_qualified'] is False
        assert result['firmware_sample_age_s'] is None
        assert 'qualification' not in result
        assert result['empirical_timing_diagnostic'] is None
        reference = result['registration_reference_metadata']
        assert reference['history_id'] == 'history' and reference['clock_mapping_id'] == 'clock'
        assert reference['applied_empirical_delay_s'] == 0.
        assert service.registrar._delay_qualification is None
    finally:
        clean(service)


def test_native_cadence_uses_all_unique_exposures_before_registration_not_accepted_spacing(tmp_path,monkeypatch):
    from autoweld.motion_registration import RegistrationRejected
    service,ring,context = harness(tmp_path)
    clock = FastSettleClock(monkeypatch)
    start = clock.now
    def add(index):
        source = native(index)
        stamp = start+index*.02
        source['timing'].update(exposure_host_mid_s=stamp,exposure_host_midpoint_s=stamp,
            exposure_host_start_s=stamp-.002,exposure_host_end_s=stamp+.002)
        ring.add(source)
    def no_bracket(_cloud):
        raise RegistrationRejected('Exposure has no measured motor bracket; extrapolation is forbidden')
    service.registrar = SimpleNamespace(register_receive_reference=no_bracket)
    for index in range(3):
        add(index)
    clock.now = start+.1
    try:
        assert service._capture_one(context) is None
        assert len(service._pending_clouds) == 3
        np.testing.assert_allclose(service._native_intervals,[.02,.02],atol=1e-10)
        assert service._capture_one(context) is None
        assert len(service._native_intervals) == 2  # Retry ofsame exposure adds no sample.
        add(3)
        assert service._capture_one(context) is None
        np.testing.assert_allclose(service._native_intervals,[.02,.02,.02],atol=1e-10)
        assert service._choose_capture_feed(30.,np.quantile(service._native_intervals,.99)) == 30.
        assert service.archive.state()['accepted'] == 0  # Cadence still established.
    finally:
        clean(service)


@pytest.mark.parametrize('maximum,expected_available',[(900.,725.),(200.,25.)])
def test_builder_budget_uses_elapsed_workflow_and_binary_export_rest_reserve(tmp_path,monkeypatch,maximum,expected_available):
    from dataclasses import asdict

    from autoweld.motion import MotionLimits
    service,_,context = harness(tmp_path)
    clock = FastSettleClock(monkeypatch)
    context.update(started_monotonic=clock.now-120.)
    context['settings'].update(max_duration_s=maximum,feed_mm_s=30.,optical_roll_deg=-45.)
    context['data']['baseline'].update(baseline_pose=np.eye(4).tolist(),depth_offset_m=0.)
    service.service.planner = SimpleNamespace(limits=MotionLimits())
    service.service.model_fingerprint = 'same-existing-model'
    service.snapshot = lambda:dict(limits=asdict(MotionLimits()),q=[0.,0.,0.,0.,0.,-3.],
        bounds=np.tile([-6.,6.],(6,1)).tolist(),gripper=0.)
    captured = {}
    monkeypatch.setattr('autoweld.scan_plan_probe.private_planner',lambda _source:SimpleNamespace())
    def build(_worker,**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(report={})
    monkeypatch.setattr('autoweld.scan_plan_probe.build_scan_plan',build)
    try:
        plan = service._build(context,spacing_mm=1.)
        assert captured['max_total_s'] == expected_available
        assert captured['max_preflight_s'] == min(90.,expected_available)
        assert plan.report['workflow_budget']['completion_reserve_s'] == 45.
        assert plan.report['workflow_budget']['entry_reserve_s'] == 10.
        assert plan.report['workflow_budget']['elapsed_before_planning_s'] == 120.
        assert plan.report['workflow_budget']['builder_max_total_s'] == expected_available
    finally:
        clean(service)


def test_footprint_uses_active_calibration_plane_and_native_identity(tmp_path):
    from test_scan_footprint import calibration
    service,_,context = harness(tmp_path)
    active = calibration()
    service.camera_state = lambda:{'intrinsics':active}
    baseline = dict(baseline_optical=np.eye(4).tolist(),fixed_optical_rotation=np.eye(3).tolist(),depth_offset_m=.007)
    source = dict(image_size=[1440,1080],source_id='camera',optical_intrinsics_id='optics',calibration_id='active-plane')
    try:
        config = service._make_footprint(context,baseline,source)
        assert config['native_calibration_id'] == 'active-plane'
        assert config['laser_plane_native_mm'] == active['laser_plane']
        assert config['reference_plane_z_m'] == pytest.approx(.147)
        assert 6 <= config['lanes'] <= 8
        np.testing.assert_allclose(config['origin_compensation_xy_m'],[.008,0.],atol=1e-12)
    finally:
        clean(service)


def test_area_envelope_shifts_camera_origins_only_and_keeps_fixed_view(tmp_path):
    from scipy.spatial.transform import Rotation
    service,_,_ = harness(tmp_path)
    original = np.eye(4)
    service._envelope = (original.copy(),np.eye(4),np.eye(3),np.array([.05,.05]))
    service._area_camera_shift = np.array([-.005,0.])
    target = np.eye(4)
    target[0,3] = -.055
    try:
        with pytest.raises(ValueError,match='XY/depth envelope'):
            service._check_envelope(target)
        service._area_plan_active = True
        service._check_envelope(target)
        np.testing.assert_array_equal(service._envelope[0],original)
        rotated = target.copy()
        rotated[:3,:3] = Rotation.from_euler('z',90,degrees=True).as_matrix()
        with pytest.raises(ValueError,match='fixed viewing orientation'):
            service._check_envelope(rotated)
        target[0,3] = -.058
        with pytest.raises(ValueError,match='XY/depth envelope'):
            service._check_envelope(target)
    finally:
        clean(service)


@pytest.mark.parametrize('area_complete',[True,False])
def test_final_area_completion_uses_actual_footprint_not_camera_origin_gap_count(tmp_path,area_complete):
    service,_,context = harness(tmp_path)
    service._footprint_mode = True
    service._last_registered = np.zeros(3)
    service._quality_gaps = [dict(measured_gap_m=.003)]
    service._repairs = []
    service._stop_collector = lambda:None
    service._repair_footprint = lambda _ctx:None
    coverage = dict(footprint_complete=area_complete,footprint_total_cells_1mm=10201,
        footprint_observed_cells_1mm=10201 if area_complete else 10200,observed_camera_xy_cells_1mm=1)
    calls = []
    def finish(**kwargs):
        calls.append(kwargs)
        return dict(coverage=coverage,status=kwargs['status'])
    service.archive = SimpleNamespace(state=lambda **_kw:{'coverage':coverage},finish=finish,
        footprint_missing_cells=lambda:[] if area_complete else [1])
    try:
        result = service._finalize(context)
        assert result['completion_qualified'] is area_complete
        assert result['visited'] == coverage['footprint_observed_cells_1mm']
        assert calls[0]['status'] == ('complete' if area_complete else 'interrupted')
        assert calls[0]['completion_evidence']['actual_footprint_complete'] is area_complete
        assert calls[0]['completion_evidence']['temporal_origin_gaps_are_diagnostic_only'] is True
    finally:
        clean(service)


def test_predicted_repair_footprint_never_marks_unobserved_cells_as_covered(tmp_path,monkeypatch):
    service,_,context = harness(tmp_path)
    context['data']['baseline'].update(scan_footprint={'origin_compensation_xy_m':[0.,0.]},
        fixed_optical_rotation=np.eye(3).tolist())
    context['data']['scout'] = {'field':SimpleNamespace(depth=lambda _x,_y:0.)}
    service._repairs = []
    service._last_registered_pose = np.eye(4)
    service.archive = SimpleNamespace(footprint_missing_cells=lambda:[5100])
    service._settle = lambda *_args:np.eye(4)
    commands = []
    service._command = lambda action,payload,_ctx:commands.append((action,payload))
    def capture(_ctx):
        key = getattr(service,'_patch_test_sequence',0)+1
        service._patch_test_sequence = key
        service._last_key = ('session',key)
        return {'qualified':True}
    service._capture_one = capture
    service._wait = lambda _ctx,predicate,*_args:predicate({})
    monkeypatch.setattr(service,'_solve_footprint_target',lambda *_args,**_kwargs:dict(
        camera=np.eye(4),predicted_camera=np.eye(4),residual_m=0.,iterations=1,compensation_xy_m=np.zeros(2)))
    monkeypatch.setattr('autoweld.scan_footprint.footprint_cells',lambda *_args:{'cells':[5100],
        'segments_original_m':[[[-.02,0.,.14],[.02,0.,.14]]]})
    try:
        service._repair_footprint(context)
        assert service.archive.footprint_missing_cells() == [5100]
        assert len(service._repairs) == len(commands) == 4
        assert all(repair['observed_cells_added'] == 0 for repair in service._repairs)
        assert all(repair['predicted_cells_never_count_as_observed'] for repair in service._repairs)
    finally:
        clean(service)


def test_repair_append_refreshes_busy_progress_and_cloud_without_archive_render_lock(tmp_path):
    service,ring,context = harness(tmp_path)
    service._continuous_selected = True
    service._planning_progress = None
    service._max_path_gap = 0.
    service.bridge = SimpleNamespace(state=lambda:{'pending':0})
    service.coordinator._data = {}
    service.coordinator.status = 'finalize'
    service.coordinator.PUBLIC = ContinuousScanCoordinator.PUBLIC
    service.coordinator.state = lambda **_kwargs:dict(status='finalize',phase='finalize',busy=True,
        message='Completing measured footprint repairs',point_count=0,captures=0,visited=0,
        cloud={'points_base_m':[],'units':'m'})
    try:
        ring.add(native(0,x=0.))
        first = service._capture_one(context)
        ring.add(native(1,x=.001))
        latest = service._capture_one(context)
        assert latest['native_frame_count'] > first['native_frame_count']
        assert latest['point_count'] > first['point_count']
        # Simulate the export worker owning the archive lock: rendering must use
        # the published append snapshot, never query archive.state() while busy.
        service.archive.state = lambda **_kwargs:pytest.fail('UI waited on archive export lock')
        state = service.state()
        assert state['status'] == state['phase'] == 'finalize'
        assert state['message'] == 'Completing measured footprint repairs'
        assert state['native_frame_count'] == latest['native_frame_count']
        assert state['point_count'] == latest['point_count']
        assert state['captures'] == latest['captures']
        assert state['visited'] == latest['visited']
        assert state['cloud'] == latest['cloud']
        assert state['archive_snapshot_age_s'] < 1.
    finally:
        clean(service)


def test_depth_aware_solver_reaches_all54_recorded_patch_cells_in_finite_fov():
    import json

    from autoweld.scan_footprint import footprint_cells
    fixture = json.loads((Path(__file__).parent/'fixtures/footprint_repair_depth.json').read_text())
    config = fixture['config']
    assert len(fixture['cases']) == 54
    compensated = 0
    for case in fixture['cases']:
        point = np.array([case['cell']%101-50,case['cell']//101-50])/1000.
        field = SimpleNamespace(depth=lambda _x,_y,z=case['camera_z_m']:z)
        result = ContinuousScanService._solve_footprint_target(config,field,point)
        assert case['cell'] in footprint_cells(config,result['predicted_camera'])['cells']
        assert abs(result['residual_m']) <= 1e-8
        assert np.max(abs(result['compensation_xy_m'])) <= .018
        np.testing.assert_array_equal(result['camera'][:3,:3],config['fixed_optical_rotation'])
        compensated += np.linalg.norm(result['compensation_xy_m']) > .002
    assert compensated >= 15  # Old2mm guard skipped these real required changes.


def test_depth_solver_requeries_sloped_field_and_keeps_measured_bias_command_orientation():
    import json

    from scipy.spatial.transform import Rotation

    from autoweld.scan_footprint import footprint_cells
    fixture = json.loads((Path(__file__).parent/'fixtures/footprint_repair_depth.json').read_text())
    config = fixture['config']
    case = max(fixture['cases'],key=lambda value:value['old_measured_line_miss_m'])
    point = np.array([case['cell']%101-50,case['cell']//101-50])/1000.
    nominal = point+np.asarray(config['origin_compensation_xy_m'])
    calls = []
    def depth(x,y):
        calls.append((x,y))
        return case['camera_z_m']+.25*(y-nominal[1])
    field = SimpleNamespace(depth=depth)
    solved = ContinuousScanService._solve_footprint_target(config,field,point)
    assert len(calls) > 2
    baseline = np.asarray(config['baseline_optical'])
    measured = solved['camera'].copy()
    measured[:3,3] += baseline[:3,:3]@np.array([.0007,-.0004,.0002])
    measured[:3,:3] = measured[:3,:3]@Rotation.from_euler('x',.002).as_matrix()
    corrected = ContinuousScanService._solve_footprint_target(config,field,point,
        initial_camera=solved['camera'],measured_camera=measured,commanded_camera=solved['camera'])
    assert case['cell'] in footprint_cells(config,corrected['predicted_camera'])['cells']
    assert abs(corrected['residual_m']) <= 1e-8
    np.testing.assert_array_equal(corrected['camera'][:3,:3],config['fixed_optical_rotation'])
    assert np.linalg.norm(corrected['camera'][:3,3]-solved['camera'][:3,3]) < .010
