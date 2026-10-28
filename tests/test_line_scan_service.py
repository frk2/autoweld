"""World-line service protocol tests without a model, camera or SDK owner."""
import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_scan_service import clean, harness, native

from autoweld.continuous_scanning import ContinuousScanSettings
from autoweld.scan_service import ContinuousScanService


def line_context(method='precision'):
    camera = np.eye(4)
    camera[:3,3] = [.2,.1,.3]
    handeye = np.eye(4)
    handeye[:3,3] = [.03,-.01,.02]
    return dict(settings=dict(scan_profile='vertical_down',scan_method=method,sweep_length_mm=80.),
        data={'baseline':dict(line_start_base_from_camera=camera.tolist(),handeye={'tcp_from_camera':handeye.tolist()})},
        cancel_event=threading.Event(),started_monotonic=time.monotonic())


def test_precision_visits_all81_frozen_world_minus_z_stations_once():
    service = ContinuousScanService.__new__(ContinuousScanService)
    context = line_context()
    commands, settled, captures = [], [], []
    service._command = lambda action,payload,ctx:commands.append((action,payload))
    service._settle = lambda ctx,goal:settled.append(goal.copy())
    service._capture_line_station = lambda ctx,index,phase:captures.append((index,phase))
    result = service._line_start_motion(context)
    assert result['stationary_protocol_complete'] and result['stations_complete']==81
    assert captures == [(i,'stationary_main') for i in range(81)]
    assert len(commands)==len(settled)==80
    start = np.asarray(context['data']['baseline']['line_start_base_from_camera'])
    correction = np.asarray(context['data']['baseline']['handeye']['tcp_from_camera'])
    for i,(action,payload) in enumerate(commands,1):
        assert action=='move_pose'
        camera = np.asarray(payload['pose'])@correction
        np.testing.assert_allclose(camera[:3,:3],start[:3,:3])
        np.testing.assert_allclose(camera[:3,3],start[:3,3]+[0,0,-i/1000.])


def test_station_burst_requires_three_distinct_new_qualified_frames_without_reholding(tmp_path):
    service,ring,context = harness(tmp_path)
    context['settings'].update(scan_profile='vertical_down',scan_method='precision',sweep_length_mm=80.)
    context['data']['baseline']['line_start_base_from_camera'] = np.eye(4).tolist()
    service._last_settle_evidence = dict(accepted_monotonic=time.monotonic()-.05,
        post_hold_stable_since=time.monotonic()-.5,measured_base_from_tcp=np.eye(4).tolist())
    service.snapshot = lambda:dict(tcp_pose=np.eye(4).tolist(),feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()))
    commands = []
    service._command = lambda *args:commands.append(args)
    original = service._capture_one
    index = 0
    def capture(ctx):
        nonlocal index
        cloud = native(index)
        time.sleep(.005)  # Fresh feedback can genuinely follow exposure end.
        ring.add(cloud)
        index += 1
        return original(ctx)
    service._capture_one = capture
    try:
        assert service._capture_line_station(context,0,'stationary_main')==3
        assert service.archive.state()['native_frame_count']==3
        assert not commands
        assert service._capture_metadata is None
    finally:
        clean(service)


def test_station_burst_rejects_motion_after_verified_hold_before_archive(tmp_path):
    from autoweld.motion_registration import RegistrationRejected
    service,ring,context = harness(tmp_path)
    service._capture_metadata = dict(acquisition_phase='stationary_main',requested_base_from_tcp=np.eye(4).tolist(),
        settle=dict(accepted_monotonic=time.monotonic()-.1,measured_base_from_tcp=np.eye(4).tolist()))
    drifted = np.eye(4)
    drifted[0,3]=.0004  # Still within endpoint0.5mm but outside stableHold0.3mm.
    service.snapshot = lambda:dict(tcp_pose=drifted.tolist(),feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()))
    cloud = native(0)
    cloud['timing']['exposure_host_end_s'] -= .01
    ring.add(cloud)
    try:
        with pytest.raises(RegistrationRejected,match='quiet feedback'):
            service._capture_one(context)
        assert service.archive.state()['native_frame_count']==0
    finally:
        clean(service)


def test_precision_request_false_continuous_is_not_legacy_protocol():
    service = ContinuousScanService.__new__(ContinuousScanService)
    service.legacy=SimpleNamespace(busy=False,dispatch=lambda _r:pytest.fail('Legacy route used'))
    started=[]
    service.coordinator=SimpleNamespace(busy=False,start=lambda r:started.append(copy.deepcopy(r)))
    service._cleanup_future=None
    service._lock=threading.Lock()
    service._pending_clouds=[]
    service.state=lambda:dict(ok=True)
    request=dict(action='scan_start',scan_profile='vertical_down',scan_method='precision',continuous=False)
    assert service.dispatch(request)=={'ok':True}
    assert started==[request]
    settings=ContinuousScanSettings.from_request(request)
    assert settings.optical_roll_deg==0 and settings.feed_mm_s==10
    with pytest.raises(ValueError,match='fast area'):
        ContinuousScanSettings.from_request(dict(scan_profile='area',scan_method='precision'))


def test_line_envelope_preserves_world_axis_and_rejects_lateral_drift():
    service=ContinuousScanService.__new__(ContinuousScanService)
    service._lock=threading.Lock()
    start=np.eye(4)
    service._envelope=(start.copy(),start.copy(),np.eye(3),np.array([.05,.05]))
    service._line_scope=dict(start=start,length_m=.08)
    goal=np.eye(4)
    goal[2,3]=-.08
    service._check_envelope(goal)
    goal[0,3]=.002
    with pytest.raises(ValueError,match='world-minus-Z'):
        service._check_envelope(goal)


def test_precision_reference_binds_saved_tube_content_and_calibration(tmp_path):
    import json
    service=ContinuousScanService.__new__(ContinuousScanService)
    service.service=SimpleNamespace(runtime=tmp_path,saved_poses={'Tube Scanning':{'q':[0.,1.,2.,3.,4.,5.]}})
    handeye={'tcp_from_camera':np.eye(4).tolist()}
    service.camera_state=lambda:dict(handeye=handeye)
    service.cloud=lambda:dict(source_id='camera',optical_intrinsics_id='optics',calibration_id='laser')
    context=line_context()
    context['settings']['optical_roll_deg']=0.
    binding=service._line_binding(context,handeye)
    reference=dict(comparison_id='paired',binding=binding)
    (tmp_path/'vertical-line-reference.json').write_text(json.dumps(reference))
    assert service._load_line_reference(context)==reference
    service.service.saved_poses['Tube Scanning']['q'][0]=.01
    with pytest.raises(ValueError,match='Tube/length/roll changed'):
        service._load_line_reference(context)


def centering_harness(tmp_path):
    service=ContinuousScanService.__new__(ContinuousScanService)
    service.service=SimpleNamespace(runtime=tmp_path)
    service.snapshot=lambda:dict(tcp_pose=np.eye(4).tolist(),motion_active=False,measured_monotonic=time.monotonic(),
        feedback=dict(fresh=True,stationary=True),q=[0.]*6)
    service._settle=lambda *args:None
    return service,line_context('fast')


def pre_motion_rejection():
    from autoweld.diagnostics import MotionRejection
    return MotionRejection('path_ik','ik_nonconvergence','Unreachable optional centering',pre_motion_rejection=True)


def test_optional_centering_halves_only_rejected_never_submitted_step(tmp_path):
    import json
    service,context=centering_harness(tmp_path)
    goals=[]
    def command(action,payload,ctx):
        goals.append(np.asarray(payload['pose']))
        if len(goals)==1:
            raise pre_motion_rejection()
    service._command=command
    moved,change=service._optional_line_center(context,np.eye(4),np.eye(4),np.array([1.,0,0]),400.,{})
    assert moved
    assert [g[0,3] for g in goals]==[.005,.0025]
    evidence=json.loads(open(change['diagnostic_path']).read())
    assert len(evidence['attempts'])==2 and evidence['attempts'][0]['stage']=='command'
    assert evidence['attempts'][1]['stage']=='complete'


def test_all_optional_centering_plans_rejected_preserves_good_view_without_move(tmp_path):
    service,context=centering_harness(tmp_path)
    goals=[]
    def command(action,payload,ctx):
        goals.append(payload['pose'])
        raise pre_motion_rejection()
    service._command=command
    service._settle=lambda *a:pytest.fail('Rejected centering must not Hold or settle')
    moved,change=service._optional_line_center(context,np.eye(4),np.eye(4),np.array([1.,0,0]),400.,{})
    assert not moved and change['kind']=='laser_tangent_centering_skipped'
    assert [g[0][3] for g in goals]==[.005,.0025,.00125]


@pytest.mark.parametrize('after_submission',[False,True])
def test_optional_centering_never_swallows_fault_or_failed_settle(tmp_path,after_submission):
    service,context=centering_harness(tmp_path)
    commands=[]
    def command(*args):
        commands.append(args)
        if not after_submission:
            raise RuntimeError('Native hardware fault')
    service._command=command
    service._settle=lambda *args:(_ for _ in ()).throw(TimeoutError('Endpoint did not settle'))
    with pytest.raises((RuntimeError,TimeoutError)):
        service._optional_line_center(context,np.eye(4),np.eye(4),np.array([1.,0,0]),400.,{})
    assert len(commands)==1


def test_failed_new_preparation_never_overlays_previous_archive_cloud():
    service=ContinuousScanService.__new__(ContinuousScanService)
    service._continuous_selected=True
    service._run_has_archive=False
    service._lock=threading.Lock()
    service._latest_archive_snapshot=service._latest_archive_snapshot_monotonic=None
    service._planning_progress=None
    service._max_path_gap=0.
    service._selected_feed_mm_s=10.
    service.bridge=SimpleNamespace(state=lambda:{})
    service.archive=SimpleNamespace(state=lambda:pytest.fail('Old archive must remain hidden for failed new run'))
    service.coordinator=SimpleNamespace(busy=False,PUBLIC=frozenset(),status='error',_data={},
        settings=SimpleNamespace(scan_profile='vertical_down',scan_method='fast',sweep_length_mm=80.),
        state=lambda **kwargs:dict(status='error',scan_profile='vertical_down',error='Preparation failed'))
    result=service.state()
    assert result['status']=='error' and result['scan_id'] is None and result['cloud'] is None
    assert result['point_count']==0 and result['coverage']=={} and result['total']==81


def planning_harness(tmp_path,monkeypatch,method='fast'):
    from dataclasses import asdict

    import autoweld.scan_plan_probe as probes
    from autoweld.motion import MotionLimits
    service,context=centering_harness(tmp_path)
    context['settings'].update(scan_method=method,optical_roll_deg=0.,feed_mm_s=10.,max_duration_s=900.)
    context['data']['baseline'].update(baseline_pose=np.eye(4).tolist(),comparison_id='pair',
        comparison_binding={'identity':'bound'},preparation_changes=[])
    service.service.planner=SimpleNamespace(limits=MotionLimits())
    service.snapshot=lambda:dict(limits=asdict(MotionLimits()),q=[0.]*6,bounds=[[-3.,3.]]*6,gripper=0.)
    service._planning_progress=None
    service._lock=threading.Lock()
    service._envelope=(np.eye(4),np.eye(4),np.eye(3),np.array([.05,.05]))
    service.archive=SimpleNamespace(session_directory=tmp_path)
    begun=[]
    service._begin_archive=lambda ctx:begun.append(copy.deepcopy(ctx['data']['baseline']))
    monkeypatch.setattr(probes,'private_planner',lambda planner:planner)
    monkeypatch.setattr(probes,'compact_plan_summary',lambda plan:{'duration_s':plan.path.duration})
    return service,context,begun


def test_whole_line_preflight_selects_roll_only_after_full_valid_plan(tmp_path,monkeypatch):
    import json

    import autoweld.vertical_scan as vertical
    service,context,begun=planning_harness(tmp_path,monkeypatch)
    candidates=[]
    def build(planner,**kwargs):
        candidate=np.asarray(kwargs['start_base_from_camera']).copy()
        candidates.append(candidate)
        assert not begun and not (tmp_path/'vertical-line-reference.json').exists()
        if len(candidates)==1:
            raise pre_motion_rejection()
        return SimpleNamespace(path=SimpleNamespace(duration=17.5))
    monkeypatch.setattr(vertical,'build_vertical_plan',build)
    service._command=lambda *a:pytest.fail('Preflight must not command hardware')
    result=service._line_plan(context)
    assert result['selected_optical_roll_deg']==-15.
    assert len(candidates)==2 and len(begun)==1
    np.testing.assert_array_equal(candidates[0][:3,3],candidates[1][:3,3])
    np.testing.assert_allclose(candidates[0][:3,2],candidates[1][:3,2])
    np.testing.assert_allclose(service._line_scope['start'],candidates[1])
    reference=json.loads((tmp_path/'vertical-line-reference.json').read_text())
    assert reference['selected_optical_roll_deg']==-15.
    np.testing.assert_allclose(reference['line_start_base_from_camera'],candidates[1])
    diagnostic=json.loads(open(result['line_plan_diagnostic_path']).read())
    assert diagnostic['attempts'][0]['diagnostic']['code']=='ik_nonconvergence'
    assert diagnostic['attempts'][1]['status']=='validated'


@pytest.mark.parametrize('code',['self_collision','derivative_limit','ik_solver_error'])
def test_whole_line_roll_fallback_never_swallows_other_guards(tmp_path,monkeypatch,code):
    import autoweld.vertical_scan as vertical
    from autoweld.diagnostics import MotionRejection
    service,context,begun=planning_harness(tmp_path,monkeypatch)
    calls=[]
    def build(*args,**kwargs):
        calls.append(kwargs)
        raise MotionRejection('raster_preflight',code,'Guard rejected candidate')
    monkeypatch.setattr(vertical,'build_vertical_plan',build)
    with pytest.raises(MotionRejection,match='Guard rejected'):
        service._line_plan(context)
    assert len(calls)==1 and not begun
    assert not (tmp_path/'vertical-line-reference.json').exists()
    assert list((tmp_path/'evidence').glob('line-plan-*.json'))


def test_precision_preflight_never_reselects_paired_fast_rotation(tmp_path,monkeypatch):
    import autoweld.vertical_scan as vertical
    service,context,begun=planning_harness(tmp_path,monkeypatch,'precision')
    context['data']['baseline']['selected_optical_roll_deg']=-30.
    calls=[]
    def build(*args,**kwargs):
        calls.append(kwargs)
        raise pre_motion_rejection()
    monkeypatch.setattr(vertical,'build_vertical_plan',build)
    with pytest.raises(ValueError,match='No bounded wrist roll'):
        service._line_plan(context)
    assert len(calls)==1 and not begun


def test_stationary_midpoint_after_hold_but_exposure_start_before_is_skipped(tmp_path):
    service,ring,context=harness(tmp_path)
    boundary=native(0)
    accepted=boundary['timing']['exposure_host_mid_s']-.001
    assert boundary['timing']['exposure_host_start_s'] < accepted < boundary['timing']['exposure_host_mid_s']
    service._capture_cutoff=accepted
    service._capture_metadata=dict(acquisition_phase='stationary_main',station_index=0,
        requested_base_from_tcp=np.eye(4).tolist(),settle=dict(accepted_monotonic=accepted,
            measured_base_from_tcp=np.eye(4).tolist()))
    service.snapshot=lambda:dict(tcp_pose=np.eye(4).tolist(),
        feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()))
    ring.add(boundary)
    try:
        assert service._capture_one(context) is None
        assert service.archive.state()['native_frame_count']==0
        assert service._last_station_capture_gate['code']=='exposure_before_hold'
        time.sleep(.005)
        fresh=native(1)
        assert fresh['timing']['exposure_host_start_s'] >= accepted
        ring.add(fresh)
        time.sleep(.005)
        assert service._capture_one(context)['qualified']
        assert service.archive.state()['native_frame_count']==1
        assert not service._pending_clouds
    finally:
        clean(service)


@pytest.mark.parametrize('persistent',[False,True])
def test_precision_station_gate_has_at_most_one_same_goal_replan(tmp_path,persistent):
    import json

    from autoweld.motion_registration import RegistrationRejected
    service,context=centering_harness(tmp_path)
    commands=[]
    service._command=lambda action,payload,ctx:commands.append((action,payload))
    service._settle=lambda ctx,goal:service._command('hold',{},ctx)
    attempts=[]
    def capture(ctx,index,phase):
        attempts.append(copy.deepcopy(service._station_retry_context))
        if len(attempts)==1 or persistent:
            error=RegistrationRejected('Actual quiet gate failed')
            error.station_gate='hold_pose_drift'
            error.diagnostic={'code':'hold_pose_drift','hold_position_drift_m':.00031}
            raise error
        return 3
    service._capture_line_station=capture
    if persistent:
        with pytest.raises(RegistrationRejected,match='Actual quiet'):
            service._capture_precision_with_retry(context,5)
    else:
        assert service._capture_precision_with_retry(context,5)==3
    assert [a['attempt'] for a in attempts]==[0,1]
    assert [action for action,_ in commands]==['move_pose','hold']
    np.testing.assert_array_equal(commands[0][1]['pose'],service._line_goal(context,5))
    assert service._last_station_retry['status']==('failed' if persistent else 'complete')
    evidence=json.loads(open(service._last_station_retry['evidence_path']).read())
    assert evidence['maximum_replans']==1 and evidence['station_index']==5
    assert service._station_retry_context is None


def test_precision_identity_failure_never_retries_or_retargets(tmp_path):
    from autoweld.motion_registration import RegistrationRejected
    service,context=centering_harness(tmp_path)
    service._command=lambda *args:pytest.fail('Identity failure must not command a retry')
    service._capture_line_station=lambda *args:(_ for _ in ()).throw(RegistrationRejected('Camera session identity changed'))
    with pytest.raises(RegistrationRejected,match='session identity'):
        service._capture_precision_with_retry(context,5)
    assert not list(tmp_path.glob('evidence/precision-station-retry-*.json'))


def test_precision_fault_after_capture_gate_preempts_retry_move(tmp_path):
    from autoweld.motion_registration import RegistrationRejected
    service,context=centering_harness(tmp_path)
    snapshot=service.snapshot
    service.snapshot=lambda:{**snapshot(),'fault':'Native fault'}
    service._command=lambda *args:pytest.fail('Fault must preempt station recovery')
    def capture(*args):
        error=RegistrationRejected('Transient freshness failed')
        error.station_gate='feedback_freshness'
        error.diagnostic={'code':'feedback_freshness'}
        raise error
    service._capture_line_station=capture
    with pytest.raises(RuntimeError,match='Physical protection'):
        service._capture_precision_with_retry(context,5)
    assert service._last_station_retry['status']=='failed'


def stationary_pending_harness(tmp_path):
    service,ring,context=harness(tmp_path)
    cloud=native(0)
    time.sleep(.005)
    service._capture_metadata=dict(acquisition_phase='stationary_main',station_index=0,
        requested_base_from_tcp=np.eye(4).tolist(),settle=dict(accepted_monotonic=cloud['timing']['exposure_host_start_s']-.01,
            measured_base_from_tcp=np.eye(4).tolist()))
    calls=[]
    service.registrar=SimpleNamespace(register_receive_reference=lambda c:(calls.append(c['timing']['device_frame_id']) or dict(c['_test_registration'])))
    ring.add(cloud)
    return service,ring,context,calls


def test_stale_main_snapshot_waits_same_hold_and_reuses_pending_registration(tmp_path):
    service,_,context,calls=stationary_pending_harness(tmp_path)
    service._command=lambda *args:pytest.fail('An aged cache must never retarget a quiet station')
    service.snapshot=lambda:dict(tcp_pose=np.eye(4).tolist(),feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()-.15))
    try:
        for _ in range(4):
            assert service._capture_one(context) is None
        assert len(calls)==1 and service.archive.state()['native_frame_count']==0
        assert service._last_station_capture_gate['code']=='feedback_snapshot_not_fresh'
        service.snapshot=lambda:dict(tcp_pose=np.eye(4).tolist(),feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()))
        assert service._capture_one(context)['qualified']
        assert len(calls)==1 and service.archive.state()['native_frame_count']==1
        assert service._pending_registration is None
    finally:
        clean(service)


@pytest.mark.parametrize('fresh,age',[(True,.15),(False,.005)])
def test_persistent_stale_snapshot_never_archives_or_counts_station(tmp_path,fresh,age):
    service,_,context,calls=stationary_pending_harness(tmp_path)
    service.snapshot=lambda:dict(tcp_pose=np.eye(4).tolist(),feedback=dict(fresh=fresh,stationary=True,measured_at=time.monotonic()-age))
    try:
        for _ in range(8):
            assert service._capture_one(context) is None
        assert len(calls)==1 and service.archive.state()['native_frame_count']==0
        assert len(service._pending_clouds)==1
    finally:
        clean(service)


def test_pending_registration_invalidated_by_new_registrar_and_discard(tmp_path):
    service,_,context,calls=stationary_pending_harness(tmp_path)
    service.snapshot=lambda:dict(tcp_pose=np.eye(4).tolist(),feedback=dict(fresh=True,stationary=True,measured_at=time.monotonic()-.15))
    try:
        assert service._capture_one(context) is None
        service.registrar=SimpleNamespace(register_receive_reference=lambda c:(calls.append(c['timing']['device_frame_id']) or dict(c['_test_registration'])))
        assert service._capture_one(context) is None
        assert len(calls)==2
        service._discard_bad_exposure(context,'Explicit test discard')
        assert service._pending_registration is None and not service._pending_clouds
    finally:
        clean(service)


def test_refresh_samples_feedback_after_blocking_camera_state(tmp_path,monkeypatch):
    from autoweld.motion import MotionLimits
    service=ContinuousScanService.__new__(ContinuousScanService)
    now=[100.]
    order=[]
    def camera_state(**kwargs):
        order.append('camera')
        now[0]+=.15
        return {'enabled':True}
    def feedback():
        order.append('feedback')
        return dict(fresh=True,stationary=True,measured_at=now[0]-.005)
    ctrl=SimpleNamespace(positions=lambda:(order.append('q') or np.zeros(7)),sdk_status=lambda:{},
        motion_plan=None,ready_for_waypoint=lambda:True,closed=False,disable_pending=False,rest_verified=False,
        shutdown_status={},cooldown_active=False,motion_fault=lambda:None,effective_joint_bounds=lambda:np.tile([-3.,3.],(6,1)),
        gripper_target=0.,planner_for_controller=lambda:SimpleNamespace(limits=MotionLimits()),robot=object())
    service.service=SimpleNamespace(ctrl=ctrl,camera=SimpleNamespace(state=camera_state),
        planner=SimpleNamespace(fk=lambda q:np.eye(4)),scan_feedback=feedback,fault=None)
    service._robot=ctrl.robot
    service._lock=threading.Lock()
    monkeypatch.setattr('autoweld.scan_service.time.monotonic',lambda:now[0])
    service._refresh()
    assert order==['camera','q','feedback']
    assert service.snapshot()['measured_monotonic']-service.snapshot()['feedback']['measured_at']==pytest.approx(.005)
