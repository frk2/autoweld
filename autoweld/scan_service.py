"""Service-owned continuous scan integration; workers use the main command bridge."""
import copy
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from autoweld.line_scan_service import VerticalScanMixin


class _PersistentEndpointMiss(RuntimeError):
    """A fixed measured Hold settled outside its unchanged requested endpoint."""


class ScanPlanProbe:
    """Bounded disabled-only whole-raster diagnostic on the existing model."""
    def __init__(self, service):
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-scan-probe')
        self.future = None
        self.cancelled = threading.Event()
        self.result = {'status': 'idle', 'read_only': True}

    def dispatch(self, request):
        if request.get('operation', 'submit') == 'cancel':
            self.cancelled.set()
            return self.state()
        if request.get('operation', 'submit') == 'status':
            return self.state()
        service = self.service
        if not service.ctrl.closed or service.ctrl.motion_plan is not None:
            raise RuntimeError('Whole-raster read-only probe requires disabled motors')
        if self.future is not None and not self.future.done():
            raise RuntimeError('A whole-raster probe is already running')
        manifest = Path(request.get('metadata_path', service.runtime/'laser-scans'/'20261027-144530-3eaec594.json'))
        record = json.loads(manifest.read_text())
        metadata = record.get('metadata', record)
        saved = service.saved_poses[request.get('start_pose', 'Tube Scanning')]
        source = copy.copy(service.ctrl.planner_for_controller())
        source.limits = copy.copy(source.limits)
        kwargs = dict(baseline_pose=metadata['baseline_pose'], tcp_from_camera=metadata['handeye']['tcp_from_camera'],
            seed_q=np.asarray(saved['q'], float).copy(), bounds=service.ctrl.effective_joint_bounds().copy(),
            gripper=service.ctrl.gripper_target, optical_roll_deg=request.get('optical_roll_deg', -45.),
            depth_offset_m=metadata.get('safe_depth_offset_m', 0.), feed_mm_s=request.get('feed_mm_s', 20.),
            width_mm=request.get('horizontal_mm', 100.), height_mm=request.get('vertical_mm', 100.),
            max_preflight_s=min(float(request.get('max_preflight_s', 120.)), 120.), max_total_s=900.,
            model_fingerprint=service.model_fingerprint)
        self.cancelled = threading.Event()
        cancelled = self.cancelled
        self.result = {'status': 'planning', 'read_only': True, 'started_monotonic': time.monotonic()}
        def run():
            from autoweld.scan_plan_probe import build_scan_plan, compact_plan_summary, private_planner
            planner = private_planner(source)
            return compact_plan_summary(build_scan_plan(planner, **kwargs, cancelled=cancelled.is_set,
                progress=lambda state: self.result.update(progress=state)))
        self.future = self.executor.submit(run)
        return self.state()

    def state(self):
        if self.future is not None and self.future.done():
            try:
                self.result = {**self.future.result(), 'status': 'complete'}
            except Exception as error:
                self.result = {**self.result, 'status': 'error', 'error': str(error),
                    'diagnostic': getattr(error, 'diagnostic', None)}
            self.future = None
        return copy.deepcopy(self.result)

    def close(self):
        self.cancelled.set()
        self.executor.shutdown(wait=False, cancel_futures=True)


class ContinuousScanService(VerticalScanMixin):
    """Main-thread effects, immutable measured caches, one continuous worker."""

    COMPLETION_BUDGET_S = 45.
    ENTRY_BUDGET_S = 10.
    FOOTPRINT_REPAIR_BUDGET_S = 300.
    FOOTPRINT_REPAIR_ATTEMPTS = 64
    def __init__(self, service, legacy):
        from autoweld.continuous_scanning import ContinuousScanCoordinator
        from autoweld.motion_registration import SharedModelFK
        from autoweld.motion_scan_archive import MotionScanArchive
        from autoweld.service_commands import ServiceCommandBridge

        self.service, self.legacy = service, legacy
        self.archive = MotionScanArchive(service.runtime)
        self.fk = SharedModelFK(service.planner.model)
        self._lock = threading.Lock()
        self._cache = {}
        self._camera_state = {}
        self._robot = self.history = self._detach = None
        self.registrar = None
        self._envelope = None
        self._prepare_goal = None
        self._line_scope = None
        self._capture_metadata = None
        self._line_fraction = 0.
        self._last_settle_evidence = None
        self._area_camera_shift = np.zeros(2)
        self._area_plan_active = False
        self._area_repair_pad_m = .002
        self._footprint_mode = False
        self._last_key = self._previous_centroid = self._last_registered = None
        self._last_registered_time = None
        self._pending_clouds = deque(maxlen=64)
        self._pending_registration = None
        self._last_enqueued_key = None
        self._max_path_gap = 0.
        self._quality_gaps = []
        self._repairs = []
        self._archive_started = False
        self._run_has_archive = True
        self._active_plan = None
        self._path_started = None
        self._main_scan = False
        self._planning_progress = None
        self._cleanup_future = None
        self._collector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-scan-capture')
        self._collector_future = None
        self._collector_stop = threading.Event()
        self._collector_state = None
        self._latest_archive_snapshot = None
        self._latest_archive_snapshot_monotonic = None
        self._collector_error = None
        self._reported_frames = 0
        self._native_intervals = deque(maxlen=2048)
        self._last_native_stamp = None
        self._selected_feed_mm_s = None
        self._last_ring_key = None
        self._capture_cutoff = 0.
        self._max_fraction = 0.
        self._maintenance = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-scan-finish')
        self.bridge = ServiceCommandBridge(self._execute, self._valid_context)
        callbacks = {phase: getattr(self, '_'+phase) for phase in ContinuousScanCoordinator.STAGES}
        callbacks.update(capture=self._capture, motion_state=self._motion_state)
        self.coordinator = ContinuousScanCoordinator(callbacks, can_start=service.can_start_scan,
            safety_state=self.snapshot, hold=service.ctrl.hold, release=self._release,
            capture_interval_s=.01, deadlines={'finalize':360.})
        self._continuous_selected = True
        self._refresh()

    def __getattr__(self, name):
        # Legacy envelope/persistence callbacks remain usable for explicit old scans.
        legacy = self.__dict__.get('legacy')
        if legacy is None:
            raise AttributeError(name)
        return getattr(legacy, name)

    @property
    def busy(self):
        return self.coordinator.busy or self.legacy.busy

    def _valid_context(self, context, generation):
        return (context == 'scan' and generation == self.coordinator._generation
            and self.coordinator.busy and not self.coordinator._cancel_event.is_set())

    def _execute(self, action, payload):
        from autoweld.service_commands import ServiceAction
        service, ctrl = self.service, self.service.ctrl
        if action is ServiceAction.CAMERA:
            request = dict(payload)
            if request.get('action') in ('camera_analysis', 'camera_stream'):
                return service.camera.dispatch(request)
            camera = service.camera
            with camera._lock:
                if camera._operation is not None:
                    raise RuntimeError('Camera lifecycle operation is still active')
                camera._start_operation('configuring', lambda: camera.laser_command(request))
            return {'queued': True}
        if action is ServiceAction.MOVE_SAVED_POSE:
            goal = service.move_laser_saved_pose(payload['name'])
            self._prepare_goal = np.array(goal,float,copy=True)
            return self._prepare_goal.tolist()
        if action is ServiceAction.MOVE_POSE:
            pose = np.asarray(payload['pose'], float)
            exact_prepare_retry = (self.coordinator.status == 'prepare' and self._prepare_goal is not None
                and pose.shape == (4,4) and np.array_equal(pose,self._prepare_goal))
            if not exact_prepare_retry:
                self._check_envelope(pose)
            from autoweld.diagnostics import MotionRejection
            before_plan = getattr(ctrl,'motion_plan',None)
            before_started = getattr(ctrl,'motion_started',None)
            before_target = np.array(getattr(ctrl,'target',None),copy=True)
            try:
                service.move_pose(pose, automatic=False)
            except MotionRejection as error:
                # These typed exceptions originate in endpoint/path validation,
                # before the SDK command. Bind that fact to unchanged controller
                # submission state so optional preparation can safely skip it.
                error.diagnostic['pre_motion_rejection'] = bool(before_plan is None and ctrl.motion_plan is None
                    and getattr(ctrl,'motion_started',None) == before_started
                    and np.array_equal(before_target,np.asarray(getattr(ctrl,'target',None))))
                raise
            return {'accepted': True,'motion_active':getattr(ctrl,'motion_plan',None) is not None}
        if action is ServiceAction.HOLD:
            ctrl.hold(automatic=False)
            return {'accepted': True}
        if action is ServiceAction.START_PATH:
            ctrl.command_continuous_scan(payload['plan'])
            self._active_plan = payload['plan']
            self._path_started = ctrl.motion_started
            return {'motion_started': ctrl.motion_started}
        if action is ServiceAction.DISABLE_AT_REST:
            ctrl.request_disable(automatic=False)
            return {'accepted': True}
        raise ValueError('Unsupported continuous scan service command')

    def _command(self, action, payload, context):
        return self.bridge.call(action, payload=payload, context='scan', generation=context['generation'],
            cancelled=context['cancel_event'].is_set, timeout_s=30.)

    def _refresh(self):
        ctrl = self.service.ctrl
        # Vision state can wait on native publication/deepcopy locks. Resolve
        # it BEFORE sampling SDK feedback so that wait cannot age a snapshot
        # labelled fresh immediately before it reaches the capture worker.
        camera = self.service.camera.state(include_frame=False) if self.service.camera else {}
        q = ctrl.positions().copy()
        pose = self.service.planner.fk(q).copy()
        sdk = ctrl.sdk_status()
        state = dict(q=q[:6].tolist(), tcp_pose=pose.tolist(), motion_active=ctrl.motion_plan is not None,
            ready=ctrl.ready_for_waypoint() if not ctrl.closed else False,
            closed=ctrl.closed, disable_pending=ctrl.disable_pending, rest_verified=bool(ctrl.rest_verified and (not sdk.get('required') or sdk.get('feedback_fresh'))),
            shutdown=dict(ctrl.shutdown_status), cooldown_active=ctrl.cooldown_active,
            fault=self.service.fault, motion_fault=ctrl.motion_fault(), measured_monotonic=time.monotonic(),
            bounds=ctrl.effective_joint_bounds().tolist(), gripper=ctrl.gripper_target,
            limits=as_motion_limits(ctrl.planner_for_controller().limits))
        try:
            feedback = self.service.scan_feedback()
        except (ValueError, RuntimeError):
            feedback = dict(fresh=bool(sdk.get('feedback_fresh')), stationary=False, measured_at=None)
        state.update(feedback=feedback,measured_monotonic=time.monotonic())
        with self._lock:
            self._cache, self._camera_state = state, camera
        if not ctrl.closed and self._robot is not ctrl.robot:
            if self._detach is not None:
                self._detach()
            from autoweld.motion_registration import JointOutputHistory
            self.history = JointOutputHistory(ctrl.effective_joint_bounds(), retention_s=60., reference_clock='kernel_receive')
            if not hasattr(ctrl.robot, 'add_joint_state_listener'):
                raise RuntimeError('Installed SDK measured-output publication hook is unavailable')
            self._detach = ctrl.robot.add_joint_state_listener(self.history.observe)
            self._robot = ctrl.robot

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._cache)

    def camera_state(self):
        with self._lock:
            return self._camera_state

    def cloud(self):
        latest = getattr(self.service.camera, 'latest_laser_cloud', None)
        if callable(latest):
            return latest()
        state = self.camera_state()
        return (state.get('laser') or {}).get('cloud')

    @staticmethod
    def _check_cancel(context):
        if context['cancel_event'].is_set():
            raise RuntimeError('Continuous scan stage cancelled')

    def _wait(self, context, predicate, timeout_s, reason):
        started = time.monotonic()
        while True:
            self._check_cancel(context)
            state = self.snapshot()
            shutdown = state.get('shutdown') or {}
            if state.get('fault') or state.get('motion_fault') or state.get('cooldown_active') or shutdown.get('requested') or shutdown.get('active'):
                raise RuntimeError('Physical protection interrupted continuous preparation')
            value = predicate(state)
            if isinstance(value, (np.ndarray, dict)) or value:
                return value
            if time.monotonic()-started > timeout_s:
                raise TimeoutError(reason)
            context['cancel_event'].wait(.01)

    @staticmethod
    def _pose_errors(actual, expected):
        from scipy.spatial.transform import Rotation
        return (float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
            float(Rotation.from_matrix(actual[:3, :3]@expected[:3, :3].T).magnitude()))

    def _settle(self, context, expected, timeout_s=30., *, capture=True, _retry_state=None, _deadline=None):
        """Converge, capture one Hold, then preserve all original quiet gates.

        Only a bounded diagnostic tail is retained. Failure persistence runs on
        this stage worker, never the SDK producer or robot tick.
        """
        expected = np.array(expected, float, copy=True)
        attempt = 0 if _retry_state is None else _retry_state["attempt"]
        samples = deque(maxlen=128)
        summary = dict(started_monotonic=time.monotonic(), poll_count=0, gates={},
            quiet_resets=0, hold_count=0, hold_monotonic=None, attempt=attempt,
            pre_hold_quiet_resets=0, pre_hold_qualified_monotonic=None)
        quiet_pose = quiet_since = None
        before_pose = before_since = None
        miss_pose = miss_since = None
        miss_samples = 0

        def remaining():
            if _deadline is None:
                return timeout_s
            value = min(timeout_s, _deadline-time.monotonic())
            if value <= 0:
                raise TimeoutError('Measured settle exhausted its original bounded deadline')
            return value

        def inspect(stage, state):
            now = time.monotonic()
            pose = np.asarray(state['tcp_pose'])
            p, r = self._pose_errors(pose, expected)
            feedback = state.get('feedback') or {}
            native = self.cloud()
            native = native if isinstance(native, dict) else {}
            read_started = native.get('read_started')
            measured_at = feedback.get('measured_at')
            def finite(value):
                return isinstance(value, (int, float)) and np.isfinite(value)
            delta = measured_at-read_started if finite(measured_at) and finite(read_started) else None
            qp, qr = (None,None) if quiet_pose is None else self._pose_errors(pose, quiet_pose)
            item = dict(stage=stage, monotonic_s=now, snapshot_monotonic_s=state.get('measured_monotonic'),
                target_position_error_m=p, target_orientation_error_rad=r,
                motion_active=bool(state.get('motion_active')), feedback_fresh=bool(feedback.get('fresh')),
                feedback_stationary=bool(feedback.get('stationary')), feedback_measured_at=measured_at,
                quiet_position_drift_m=qp, quiet_orientation_drift_rad=qr,
                quiet_seconds=None if quiet_since is None else now-quiet_since,
                native_read_started=read_started, native_sequence=native.get('sequence'),
                native_device_frame_id=(native.get('timing') or {}).get('device_frame_id'),
                feedback_minus_native_read_started_s=delta,
                native_age_s=now-read_started if finite(read_started) else None,
                q=state.get('q'))
            if item['motion_active']:
                gate = 'motion_active'
            elif not item['feedback_fresh']:
                gate = 'feedback_not_fresh'
            elif not item['feedback_stationary']:
                gate = 'feedback_not_stationary'
            elif p > .0005:
                gate = 'target_position'
            elif r > .005:
                gate = 'target_orientation'
            else:
                gate = 'arrived'
            item['arrived'] = gate == 'arrived'
            item['gate'] = gate
            return item, pose, native

        def record(item, gate=None):
            if gate is not None:
                item['gate'] = gate
            summary['poll_count'] += 1
            summary['gates'][item['gate']] = summary['gates'].get(item['gate'],0)+1
            samples.append(item)

        def arrived(state):
            nonlocal before_pose, before_since
            item, pose, _ = inspect('before_hold', state)
            if not item['arrived']:
                if before_since is not None:
                    summary['pre_hold_quiet_resets'] += 1
                before_pose = before_since = None
                record(item)
                return False
            if before_pose is None:
                before_pose, before_since = pose.copy(), item['monotonic_s']
            p, r = self._pose_errors(pose,before_pose)
            item.update(quiet_position_drift_m=p,quiet_orientation_drift_rad=r,
                quiet_seconds=item['monotonic_s']-before_since)
            if p > .0003 or r > .003:
                summary['pre_hold_quiet_resets'] += 1
                record(item,'pre_hold_position_drift' if p > .0003 else 'pre_hold_orientation_drift')
                before_pose = before_since = None
                return False
            if item['quiet_seconds'] < .4:
                record(item,'pre_hold_quiet_dwell')
                return False
            summary['pre_hold_qualified_monotonic'] = item['monotonic_s']
            record(item,'pre_hold_qualified')
            return True

        def stable(state):
            nonlocal quiet_pose, quiet_since, miss_pose, miss_since, miss_samples
            item, pose, native = inspect('after_hold', state)
            if not item['arrived']:
                if quiet_since is not None:
                    summary['quiet_resets'] += 1
                quiet_pose = quiet_since = None
                if item['gate'] in ('target_position','target_orientation'):
                    if miss_pose is None:
                        miss_pose, miss_since, miss_samples = pose.copy(),item['monotonic_s'],0
                    drift_p,drift_r = self._pose_errors(pose,miss_pose)
                    if drift_p > .0003 or drift_r > .003:
                        miss_pose,miss_since,miss_samples = pose.copy(),item['monotonic_s'],0
                    miss_samples += 1
                    item.update(post_hold_miss_seconds=item['monotonic_s']-miss_since,
                        post_hold_miss_samples=miss_samples,post_hold_miss_position_drift_m=drift_p,
                        post_hold_miss_orientation_drift_rad=drift_r)
                    if attempt == 0 and miss_samples >= 3 and item['monotonic_s']-miss_since >= .2:
                        record(item,'persistent_post_hold_'+item['gate'])
                        raise _PersistentEndpointMiss('Fresh quiet measured Hold persistently missed the requested endpoint')
                else:
                    miss_pose = miss_since = None
                    miss_samples = 0
                record(item)
                return False
            miss_pose = miss_since = None
            miss_samples = 0
            if quiet_pose is None:
                quiet_pose, quiet_since = pose.copy(), item['monotonic_s']
                item.update(quiet_seconds=0.,quiet_position_drift_m=0.,quiet_orientation_drift_rad=0.)
            p, r = self._pose_errors(pose, quiet_pose)
            if p > .0003 or r > .003:
                summary['quiet_resets'] += 1
                record(item,'quiet_position_drift' if p > .0003 else 'quiet_orientation_drift')
                quiet_pose = quiet_since = None
                return False
            if item['monotonic_s']-quiet_since < .4:
                record(item,'quiet_dwell')
                return False
            if not capture:
                record(item,'accepted')
                return pose
            acquired = native.get('read_started')
            if not isinstance(acquired,(int,float)) or not np.isfinite(acquired):
                record(item,'native_missing')
                return False
            if acquired < quiet_since+.4:
                record(item,'native_before_quiet')
                return False
            stamp = item['feedback_measured_at']
            if not isinstance(stamp,(int,float)) or not np.isfinite(stamp):
                record(item,'feedback_timestamp_missing')
                return False
            if stamp < acquired:
                record(item,'feedback_before_native')
                return False
            record(item,'accepted')
            return pose

        try:
            self._wait(context, arrived, remaining(), 'Measured scan endpoint failed continuous fresh quiet 0.5 mm / 0.005 rad convergence')
            self._command('hold', {}, context)
            summary.update(hold_count=1,hold_monotonic=time.monotonic())
            if _deadline is None:
                _deadline = time.monotonic()+timeout_s
            measured = self._wait(context, stable, remaining(), 'Fresh stable acquisition did not follow the single measured Hold')
            self._last_settle_evidence = dict(requested_base_from_tcp=expected.tolist(),
                measured_base_from_tcp=measured.tolist(),pre_hold_qualified_monotonic=summary['pre_hold_qualified_monotonic'],
                hold_monotonic=summary['hold_monotonic'],post_hold_stable_since=quiet_since,
                accepted_monotonic=time.monotonic(),quiet_seconds=.4,target_position_m=.0005,target_orientation_rad=.005,
                attempt=attempt,feedback=copy.deepcopy(self.snapshot().get('feedback',{})))
            return measured
        except Exception as error:
            try:
                directory = self.service.runtime/'evidence'
                directory.mkdir(parents=True,exist_ok=True)
                path = directory/f'continuous-settle-{context.get("generation",0)}-{time.time_ns()}.json'
                record = dict(schema='autoweld.continuous-settle-diagnostics.v1',error=str(error),
                    phase=context.get('phase'),generation=context.get('generation'),
                    target_pose=expected.tolist(),capture_required=capture,timeout_s=timeout_s,
                    finished_monotonic=time.monotonic(),summary=summary,samples=list(samples),
                    limits=dict(target_position_m=.0005,target_orientation_rad=.005,
                        quiet_position_m=.0003,quiet_orientation_rad=.003,quiet_seconds=.4))
                path.write_text(json.dumps(record,allow_nan=False,indent=2))
                self._last_settle_diagnostic = str(path)
            except Exception as evidence_error:
                error.add_note(f'Settle diagnostic persistence failed: {evidence_error}')
                raise error
            if isinstance(error,_PersistentEndpointMiss) and attempt == 0:
                remaining()  # Never renew the original post-Hold timeout budget.
                retry = dict(attempt=1,maximum_replans=1,status='replanning',
                    reason=str(error),initial_diagnostic_path=str(path),
                    requested_goal=expected.tolist(),deadline_monotonic=_deadline,
                    started_monotonic=time.monotonic())
                status_path = path.with_name(path.stem+'-retry.json')
                retry['status_path'] = str(status_path)
                def publish_retry(status,detail=None):
                    retry.update(status=status,updated_monotonic=time.monotonic())
                    if detail is not None:
                        retry['detail'] = str(detail)
                    self._last_settle_retry = dict(retry)
                    status_path.write_text(json.dumps(retry,allow_nan=False,indent=2))
                publish_retry('replanning')
                try:
                    self._command('move_pose',{'pose':expected.tolist()},context)
                    publish_retry('settling_after_new_move')
                    measured = self._settle(context,expected,timeout_s,capture=capture,
                        _retry_state=retry,_deadline=_deadline)
                except Exception as retry_error:
                    publish_retry('failed',retry_error)
                    raise
                publish_retry('complete')
                return measured
            raise type(error)(f'{error}; diagnostics: {path}') from error

    def _camera_request(self, context, request):
        self._wait(context, lambda _s: not any(self.camera_state().get(k) for k in ('starting', 'stopping', 'configuring')),
            15., 'Camera owner configuration did not finish')
        self._command('camera', request, context)
        acknowledged = time.monotonic()
        self._wait(context, lambda state: state.get('measured_monotonic', 0) > acknowledged and not any(self.camera_state().get(k) for k in ('starting', 'stopping', 'configuring')),
            15., 'Camera owner configuration did not finish')
        error = self.camera_state().get('error')
        if error:
            raise ValueError(str(error))

    def _good_observation(self):
        from autoweld.laser_scanning import segment_object
        state = self.camera_state()
        native, correction = self.legacy._observation(state, initial=True)
        return native, correction, segment_object(native)

    def _check_envelope(self, pose):
        from scipy.spatial.transform import Rotation

        from autoweld.calibration import transform
        pose = transform(pose)
        with self._lock:
            envelope = self._envelope
        if envelope is None:
            raise ValueError('Continuous motion requires an established original optical baseline')
        original, handeye, rotation, half = envelope
        optical = pose@handeye
        scope = getattr(self,'_line_scope',None)
        if scope is not None:
            start = scope['start']
            down = start[2,3]-optical[2,3]
            if (np.linalg.norm(optical[:2,3]-start[:2,3]) > .001
                    or not -.001 <= down <= scope['length_m']+.001
                    or Rotation.from_matrix(optical[:3,:3]@start[:3,:3].T).magnitude() > .005):
                raise ValueError('Vertical scan pose exceeds its fixed world-minus-Z line')
            return
        local = original[:3, :3].T@(optical[:3, 3]-original[:3, 3])
        shift = self._area_camera_shift if self._area_plan_active else np.zeros(2)
        pad = min(.020,max(.002,getattr(self,'_area_repair_pad_m',.002))) if self._area_plan_active else .001
        if np.any(abs(local[:2]-shift) > half+pad) or abs(local[2]) > .060:
            raise ValueError('Continuous scan pose exceeds original optical XY/depth envelope')
        if Rotation.from_matrix(optical[:3, :3]@rotation.T).magnitude() > .005:
            raise ValueError('Continuous scan pose changes its fixed viewing orientation')

    def _prepare(self, context):
        self._camera_request(context, {'action': 'camera_analysis', 'mode': 'laser'})
        self._camera_request(context, {'action': 'camera_stream', 'enabled': True})
        self._camera_request(context, {'action': 'pointcloud_configure', 'enabled': True})
        self._camera_request(context, {'action': 'laser_configure', 'auto_exposure': {'enabled': False},
            'controls': {'exposure_us': 4000., 'gain_db': 12.}})
        if context['settings']['start_pose'] == 'Tube Scanning':
            goal = self._command('move_saved_pose', {'name': 'Tube Scanning'}, context)
            measured = self._settle(context, goal)
        else:
            self._prepare_goal = np.array(self.snapshot()['tcp_pose'],float,copy=True)
            measured = self._settle(context, self._prepare_goal)
        baseline = measured.copy()
        return dict(baseline_pose=baseline.tolist(), measured_prepare_pose=measured.tolist())

    def _baseline(self, context):
        from scipy.spatial.transform import Rotation
        if self._is_line(context):
            reference = self._load_line_reference(context)
            if reference is not None:
                return self._restore_line_baseline(context,reference)
        baseline = np.asarray(context['data']['prepare']['baseline_pose'])
        handeye = self.camera_state().get('handeye')
        if not isinstance(handeye, dict):
            raise ValueError('Installed measured hand-eye is required')
        correction = np.asarray(handeye['tcp_from_camera'], float)
        original = baseline@correction
        rotation = original[:3, :3]@Rotation.from_euler('z', context['settings']['optical_roll_deg'], degrees=True).as_matrix()
        half = np.array([context['settings']['width_mm'], context['settings']['height_mm']])/2000.
        with self._lock:
            self._envelope = (original.copy(), correction.copy(), rotation.copy(), half.copy())
        optical = np.asarray(self.snapshot()['tcp_pose'])@correction
        optical[:3, :3] = rotation
        goal = optical@np.linalg.inv(correction)
        self._command('move_pose', {'pose': goal.tolist()}, context)
        self._settle(context, goal)
        for index in range(13):
            try:
                native, _, segment = self._good_observation()
            except ValueError:
                if index:
                    raise
                self._camera_request(context, {'action': 'laser_configure', 'auto_exposure': {'enabled': False},
                    'controls': {'exposure_us': 8000., 'gain_db': 12.}})
                self._settle(context, goal)
                native, _, segment = self._good_observation()
            depth_error = float(segment['centroid_camera_m'][2])-.140
            if abs(depth_error) <= .001:
                break
            if index == 12:
                raise ValueError('Baseline depth did not converge within bounded 5 mm corrections')
            optical = np.asarray(self.snapshot()['tcp_pose'])@correction
            # Measured orientation is feedback, never the next commanded view.
            # Reusing its tracking error compounds roll/axis drift across moves.
            optical[:3, :3] = rotation
            optical[:3, 3] += rotation[:, 2]*np.clip(depth_error, -.005, .005)
            goal = optical@np.linalg.inv(correction)
            self._command('move_pose', {'pose': goal.tolist()}, context)
            self._settle(context, goal)
        measured = np.asarray(self.snapshot()['tcp_pose'])
        current = measured@correction
        quality = dict(stripe_points=len(native['xyz_m']), object_points=len(segment['indices']),
            standoff_mm=float(segment['centroid_camera_m'][2])*1000.)
        result = dict(baseline_pose=baseline.tolist(), handeye=copy.deepcopy(handeye),
            baseline_optical=original.tolist(), settled_optical=current.tolist(),
            fixed_optical_rotation=rotation.tolist(), depth_offset_m=float((original[:3, :3].T@(current[:3, 3]-original[:3, 3]))[2]),
            laser_offset_camera_m=segment['centroid_camera_m'], quality=quality,
            identity=[native[k] for k in ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id')])
        if self._is_line(context):
            return self._finish_line_baseline(context,result,native)
        result['scan_footprint'] = self._make_footprint(context,result,native)
        return result

    def _make_footprint(self, context, baseline, native):
        from autoweld.scan_footprint import plan_footprint
        # The active owner may have a newer laser plane than any ROOT file.
        calibration = copy.deepcopy(self.camera_state().get('intrinsics') or {})
        calibration['image_size'] = list(native['image_size'])
        calibration['source_id'] = native['source_id']
        config = plan_footprint(calibration,baseline['baseline_optical'],baseline['fixed_optical_rotation'],
            reference_plane_z_m=baseline['depth_offset_m']+.140,
            width_mm=context['settings']['width_mm'],height_mm=context['settings']['height_mm'])
        config.update(native_calibration_id=native['calibration_id'],
            optical_intrinsics_id=native['optical_intrinsics_id'],source_id=native['source_id'])
        return config

    def _clock_warm(self, context):
        if not callable(getattr(self.service.camera, 'recent_laser_clouds', None)):
            raise RuntimeError('Continuous capture requires the bounded native exposure ring')
        def qualified(_state):
            native = self.cloud() or {}
            timing = native.get('timing') or {}
            stamp = timing.get('exposure_host_mid_s', timing.get('exposure_host_midpoint_s'))
            if (timing.get('motion_pairing_supported') is True and timing.get('clock_mapping', {}).get('status') == 'qualified'
                    and stamp is not None and 0 <= time.monotonic()-stamp <= .3
                    and timing.get('exposure_time_uncertainty_s', 1.) < .002):
                return timing
            return False
        timing = self._wait(context, qualified, 40., 'Native global-shutter hardware exposure timing did not qualify')
        self._wait(context, lambda _s: self.history is not None and min(self.history.state()['samples_per_joint']) >= 20,
            3., 'Measured SDK output history did not warm')
        return dict(clock_qualified=True, timing=copy.deepcopy(timing),
            reference_timing_uncertainty_s=timing['exposure_time_uncertainty_s'])

    def _delay_calibration(self, context):
        """Configure the receive-reference tier; physical sample age stays unknown."""
        from autoweld.motion_registration import MovingCloudRegistrar
        baseline = context['data']['baseline']
        if self.history is None or self.history.reference_clock != 'kernel_receive':
            raise ValueError('Moving receive-reference registration requires measured kernel-RX history')
        native = self.cloud() or {}
        timing = native.get('timing') or {}
        mapping = timing.get('clock_mapping') or {}
        keys = ('source_id','optical_intrinsics_id','calibration_id','acquisition_session_id')
        if [native.get(key) for key in keys] != baseline['identity']:
            raise ValueError('Camera identity changed before receive-reference registration')
        if timing.get('motion_pairing_supported') is not True or mapping.get('status') != 'qualified' or not mapping.get('mapping_id'):
            raise ValueError('Hardware exposure timing is not qualified')
        reference = dict(history_id=self.history.history_id,clock_mapping_id=mapping['mapping_id'],
            source_id=native['source_id'],optical_intrinsics_id=native['optical_intrinsics_id'],
            acquisition_session_id=native['acquisition_session_id'],reference_clock='kernel_receive',
            physical_sample_timing_qualified=False,firmware_sample_age_s=None,applied_empirical_delay_s=0.)
        self.registrar = MovingCloudRegistrar(self.history,self.fk,baseline['handeye'])
        if not callable(getattr(self.registrar,'register_receive_reference',None)):
            raise RuntimeError('Receive-reference registrar is unavailable')
        diagnostic = None
        if context['settings'].get('empirical_timing_diagnostic'):
            from autoweld.scan_delay_phase import run_delay_experiment
            try:
                estimate = run_delay_experiment(context,command=self._command,snapshot=self.snapshot,
                    cloud=self.cloud,history=self.history,fk=self.fk,baseline_optical=baseline['settled_optical'],
                    handeye=baseline['handeye'],evidence_dir=self.service.runtime/'evidence'/'motion-delay')
                diagnostic = dict(status='available',result=estimate,applied_to_scan=False)
            except Exception as error:
                self._check_cancel(context)
                diagnostic = dict(status='unavailable',error=str(error),applied_to_scan=False)
        return dict(reference_registered=True,reference_timing_qualified=True,
            registration_reference='measured_kernel_rx',registration_reference_metadata=reference,
            physical_sample_timing_qualified=False,firmware_sample_age_s=None,
            empirical_timing_diagnostic=diagnostic,
            reference_timing_uncertainty_s=timing.get('exposure_time_uncertainty_s'),
            message='Hardware exposure + kernel-received measured angles; firmware sample age unknown')

    def _build(self, context, *, spacing_mm, depth=None, area_plan=False):
        from autoweld.scan_plan_probe import build_scan_plan, private_planner
        baseline = context['data']['baseline']
        snapshot = self.snapshot()
        elapsed = max(0.,time.monotonic()-context['started_monotonic'])
        maximum = context['settings']['max_duration_s']
        available = min(maximum,maximum-elapsed-self.ENTRY_BUDGET_S-self.COMPLETION_BUDGET_S)
        if available <= 0:
            raise ValueError('No workflow time remains for validated scanning after entry/export/rest reserves')
        budget = dict(total_max_s=maximum,elapsed_before_planning_s=elapsed,
            entry_reserve_s=self.ENTRY_BUDGET_S,completion_reserve_s=self.COMPLETION_BUDGET_S,
            builder_max_total_s=available,basis='remaining workflow minus entry and binary export/rest')
        source = copy.copy(self.service.planner)
        from autoweld.motion import MotionLimits
        source.limits = MotionLimits(**snapshot['limits'])
        worker = private_planner(source)
        footprint = baseline.get('scan_footprint') if area_plan else None
        area_options = {} if footprint is None else dict(
            camera_xy_shift_m=footprint['origin_compensation_xy_m'],sweep_axis=footprint['sweep_axis'],
            lane_spacing_mm=footprint['lane_spacing_m']*1000.)
        plan = build_scan_plan(worker, baseline_pose=baseline['baseline_pose'],
            tcp_from_camera=baseline['handeye']['tcp_from_camera'], seed_q=snapshot['q'], bounds=snapshot['bounds'],
            gripper=snapshot['gripper'], optical_roll_deg=context['settings']['optical_roll_deg'],
            depth_offset_m=baseline['depth_offset_m'], depth=depth, feed_mm_s=self._selected_feed_mm_s or context['settings']['feed_mm_s'],
            width_mm=context['settings']['width_mm'], height_mm=context['settings']['height_mm'], spacing_mm=spacing_mm,
            max_preflight_s=min(90.,available), max_total_s=available,
            model_fingerprint=self.service.model_fingerprint, cancelled=context['cancel_event'].is_set,
            progress=lambda state: setattr(self, '_planning_progress', state), **area_options)
        plan.report['workflow_budget'] = budget
        return plan

    def _begin_archive(self, context):
        baseline = context['data']['baseline']
        metadata = dict(settings=context['settings'], baseline_pose=baseline['baseline_pose'],
            handeye=baseline['handeye'], identity=baseline['identity'],
            registration_reference='measured_kernel_rx',physical_sample_timing_qualified=False,firmware_sample_age_s=None,
            registration_reference_metadata=context['data']['delay_calibration']['registration_reference_metadata'],
            model_fingerprint=self.service.model_fingerprint)
        if self._is_line(context):
            start = np.asarray(baseline['line_start_base_from_camera'])
            end = start.copy()
            end[2,3] -= context['settings']['sweep_length_mm']/1000.
            metadata.update(scan_profile='vertical_down',scan_method=context['settings']['scan_method'],
                scan_protocol='precision_stationary' if context['settings']['scan_method']=='precision' else 'fast_continuous',
                sweep_length_mm=context['settings']['sweep_length_mm'],direction_base=[0.,0.,-1.],
                line_start_base_from_camera=start.tolist(),line_end_base_from_camera=end.tolist(),
                comparison_id=baseline['comparison_id'],comparison_binding=baseline['comparison_binding'],
                selected_optical_roll_deg=baseline['selected_optical_roll_deg'],
                roll_candidate_failures=baseline.get('roll_candidate_failures',[]),
                line_plan_diagnostic_path=baseline.get('line_plan_diagnostic_path'),
                preparation_changes=baseline['preparation_changes'],
                coverage_semantics='measured world-minus-Z line positions; stationary stations and native frames separate')
        else:
            metadata.update(scan_footprint=baseline['scan_footprint'],
                coverage_semantics='actual measured-pose laser/reference-plane footprint; object XYZ kept separate')
        state = self.archive.begin(metadata)
        self._archive_started = True
        self._run_has_archive = True
        self._footprint_mode = not self._is_line(context)
        with self._lock:
            self._latest_archive_snapshot = self._display_saved(state)
            self._latest_archive_snapshot_monotonic = time.monotonic()
        return state

    def _scout(self, context):
        if self._is_line(context):
            return dict(line_profile_ready=True)
        from autoweld.scan_surface import SurfaceDepthField
        self._begin_archive(context)
        scout = self._build(context, spacing_mm=context['settings']['height_mm']/4.)
        self._command('move_pose', {'pose': scout.report['entry_pose']}, context)
        self._settle(context, scout.report['entry_pose'])
        self._previous_centroid = None
        self._command('start_path', {'plan': scout}, context)
        worlds = []
        self._start_collector(context, scout_worlds=worlds)
        started = time.monotonic()
        try:
            while True:
                self._check_cancel(context)
                self._check_collector()
                motion = self._motion_state(context)
                if motion.get('error'):
                    raise RuntimeError(motion['error'])
                if motion.get('complete'):
                    break
                if time.monotonic()-started > scout.path.duration+5.:
                    raise TimeoutError('Coarse surface scout exceeded validated duration')
                context['cancel_event'].wait(.01)
        finally:
            self._stop_collector()
        baseline = context['data']['baseline']
        field = SurfaceDepthField(np.concatenate(worlds) if worlds else np.empty((0,3)), baseline['baseline_optical'],
            fixed_optical_rotation=baseline['fixed_optical_rotation'],
            laser_offset_camera_m=baseline['laser_offset_camera_m'], last_safe_camera_z_m=baseline['depth_offset_m'],
            support_radius_m=.015, nearest_limit_m=.012)
        if not len(field.nodes):
            raise ValueError('Surface scout did not retain any measured support')
        self._surface_views = deque((points[::max(1,len(points)//32+1)].copy() for points in worlds), maxlen=2048)
        return {**self.archive.state(), 'field': field, 'support_nodes': len(field.nodes),
            'native_interval_p99_s': float(np.quantile(self._native_intervals,.99)) if self._native_intervals else None,
            'native_cadence_basis':'all_unique_enqueued_hardware_exposures',
            'message': 'Full-area measured scout frozen; unsupported regions retain travel-only depth prior'}

    @staticmethod
    def _choose_capture_feed(requested, observed_interval_s):
        if observed_interval_s is None or not np.isfinite(observed_interval_s) or observed_interval_s <= 0:
            raise ValueError('Scout must establish real native exposure cadence')
        # Process feed selection is explicit and reported; physical speed ceilings
        # remain unchanged. Leave0.2mm for measured sampling/encoder variation.
        supported = np.floor((.0008/observed_interval_s)*10000.)/10.
        return min(float(requested), float(supported))

    def _plan(self, context):
        if self._is_line(context):
            self._selected_feed_mm_s = context['settings']['feed_mm_s']
            return self._line_plan(context)
        from autoweld.scan_plan_probe import compact_plan_summary
        field = context['data']['scout']['field']
        interval = context['data']['scout'].get('native_interval_p99_s')
        self._selected_feed_mm_s = self._choose_capture_feed(context['settings']['feed_mm_s'],interval)
        plan = self._build(context, spacing_mm=1., depth=field.depth, area_plan=True)
        footprint = context['data']['baseline']['scan_footprint']
        with self._lock:
            self._area_camera_shift = np.asarray(footprint['origin_compensation_xy_m'],float).copy()
            self._area_plan_active = True
        report = compact_plan_summary(plan)
        path = self.archive.session_directory/'validated-main-plan.json'
        path.write_text(json.dumps(report, allow_nan=False, indent=2))
        return {**self.archive.state(), 'plan':plan, 'validated':True, 'duration_s':plan.path.duration,
            'entry_duration_s':self.ENTRY_BUDGET_S, 'completion_budget_s':self.COMPLETION_BUDGET_S, 'plan_report_path':str(path),
            'geometry':self._geometry(context)}

    def _geometry(self, context):
        baseline = np.asarray(context['data']['baseline']['baseline_optical'])
        x, y = context['settings']['width_mm']/2000., context['settings']['height_mm']/2000.
        z = context['data']['baseline'].get('scan_footprint',{}).get('reference_plane_z_m',0.)
        points = np.array([[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z],[-x,-y,z]])
        return dict(units='m', coordinate_frame='robot_base', outline_base_m=(points@baseline[:3,:3].T+baseline[:3,3]).tolist())

    def _move_entry(self, context):
        plan = context['data']['plan']['plan']
        self._command('move_pose', {'pose': plan.report['entry_pose']}, context)
        self._settle(context, plan.report['entry_pose'])
        if self._is_line(context):
            if context['settings']['scan_method'] == 'fast':
                self._capture_line_station(context,0,'start_anchor')
            return {'entry_verified':True}
        self._pending_clouds.clear()
        self._pending_registration = None
        self._last_enqueued_key = self._last_ring_key = None
        self._capture_cutoff = time.monotonic()-.05
        self._previous_centroid = None
        self._wait(context, lambda _state: bool(self._capture_one(context)), 2., 'No qualified measured entry exposure')
        self._entry_registered = self._last_registered.copy()
        self._entry_registered_time = self._last_registered_time
        return {'entry_verified': True}

    def _start_motion(self, context):
        if self._is_line(context):
            return self._line_start_motion(context)
        self._main_scan = True
        self._previous_centroid = None
        self._pending_clouds.clear()
        self._pending_registration = None
        self._last_enqueued_key = None
        self._last_registered = self._entry_registered.copy()
        self._last_registered_time = self._entry_registered_time
        self._max_path_gap = 0.
        self._quality_gaps = []
        self._command('start_path', {'plan': context['data']['plan']['plan']}, context)
        self._start_collector(context)
        return {'motion_started': True}

    def _motion_state(self, context):
        state = self.snapshot()
        if state.get('fault') or state.get('motion_fault') or state.get('cooldown_active') or state.get('shutdown', {}).get('requested'):
            return {'error': 'Physical protection interrupted the continuous path'}
        if self._active_plan is None or self._path_started is None:
            return {'active': False, 'complete': False}
        # The endpoint FK uses only worker-local data and measured quiet feedback.
        expected, _ = self.fk.pose_and_jacobian(self._active_plan.path.positions[-1, :6])
        p, r = self._pose_errors(np.asarray(state['tcp_pose']), expected)
        feedback = state.get('feedback') or {}
        elapsed = max(0., time.monotonic()-self._path_started)
        complete = (elapsed >= self._active_plan.path.duration and not state['motion_active']
            and feedback.get('fresh') and feedback.get('stationary') and p <= .0005 and r <= .005)
        return dict(active=bool(state['motion_active']), complete=bool(complete),
            path_fraction=min(1., elapsed/self._active_plan.path.duration), endpoint_error_m=p, endpoint_error_rad=r)

    def _start_collector(self, context, scout_worlds=None):
        if self._collector_future is not None and not self._collector_future.done():
            raise RuntimeError('Only one moving native-cloud collector may run')
        self._collector_stop = threading.Event()
        self._collector_error = None
        self._reported_frames = 0
        self._native_intervals.clear()
        self._last_native_stamp = None
        self._collector_state = None
        self._pending_clouds.clear()
        self._pending_registration = None
        self._last_ring_key = self._last_enqueued_key = None
        self._capture_cutoff = self._path_started-.02
        stop = self._collector_stop
        def collect():
            try:
                from autoweld.motion_registration import RegistrationRejected
                while not stop.is_set() and not context['cancel_event'].is_set():
                    try:
                        result = self._capture_one(context, scout_worlds=scout_worlds)
                    except RegistrationRejected as error:
                        if any(word in str(error).lower() for word in ('identity', 'different', 'session', 'clock mapping')):
                            raise
                        self._discard_bad_exposure(context, str(error))
                        result = None
                    if result is not None:
                        with self._lock:
                            self._collector_state = result
                    stop.wait(.002)
            except Exception as error:
                with self._lock:
                    self._collector_error = error
        self._collector_future = self._collector_executor.submit(collect)

    def _stop_collector(self):
        self._collector_stop.set()
        if self._collector_future is not None:
            self._collector_future.result(timeout=2.)
        self._check_collector()

    def _check_collector(self):
        with self._lock:
            error = self._collector_error
        if error is not None:
            raise error

    def _capture(self, context):
        self._check_cancel(context)
        self._check_collector()
        with self._lock:
            result = None if self._collector_state is None else dict(self._collector_state)
        if result is None:
            return None
        frames = result.get('native_frame_count', 0)
        result['qualified'] = bool(result.get('qualified') and frames > self._reported_frames)
        self._reported_frames = frames
        return result

    def _verify_end_coverage(self, context):
        if self._last_registered is None:
            raise ValueError('No qualified measured exposure reached the raster')
        if self._footprint_mode:
            return  # Area completion comes from actual measured line footprints.
        handeye = np.asarray(context['data']['baseline']['handeye']['tcp_from_camera'])
        optical = np.asarray(self.snapshot()['tcp_pose'])@handeye
        original = np.asarray(context['data']['baseline']['baseline_optical'])
        delta = original[:3,:3].T@(optical[:3,3]-self._last_registered)
        gap = float(np.linalg.norm(delta[:2]))
        self._max_path_gap = max(self._max_path_gap, gap)
        if gap > .001:
            self._quality_gaps.append(dict(start=self._last_registered_time, end=time.monotonic(), measured_gap_m=gap,
                row=round((original[:3,:3].T@(optical[:3,3]-original[:3,3]))[1]*1000+context['settings']['height_mm']/2)))
            self.archive.record_gap('Last measured exposure did not reach raster edge within1mm',
                start_monotonic_s=self._last_registered_time,end_monotonic_s=time.monotonic(),path_fraction=1.,
                evidence={'measured_gap_m':gap})

    def _display_saved(self, saved):
        result = dict(saved)
        coverage = result.get('coverage') or {}
        if 'footprint_observed_cells_1mm' in coverage:
            result['visited'] = coverage['footprint_observed_cells_1mm']
        elif 'observed_camera_xy_cells_1mm' in coverage:
            result['visited'] = coverage['observed_camera_xy_cells_1mm']
        settings = self.coordinator.settings
        if getattr(settings,'scan_profile','area') == 'vertical_down':
            result['visited'] = result.get('stations_complete',0) if settings.scan_method == 'precision' else result.get('line_samples_observed',0)
            result['total'] = round(settings.sweep_length_mm)+1
            return result
        result['total'] = (round(settings.width_mm)+1)*(round(settings.height_mm)+1)
        return result

    def _discard_bad_exposure(self, context, reason):
        if not self._pending_clouds:
            raise RuntimeError(reason)
        cloud = self._pending_clouds.popleft()
        self._pending_registration = None
        self._last_key = (cloud.get('acquisition_session_id'), cloud.get('timing',{}).get('device_frame_id'))
        stamp = cloud.get('timing',{}).get('exposure_host_mid_s', time.monotonic())
        start = stamp if self._last_registered_time is None else min(stamp, self._last_registered_time)
        self.archive.record_gap(reason, start_monotonic_s=start, end_monotonic_s=stamp,
            path_fraction=self._fraction(), evidence={'device_frame_id':self._last_key[1]})

    @staticmethod
    def _solve_footprint_target(config,field,point_xy,*,initial_camera=None,measured_camera=None,commanded_camera=None):
        """Solve line-through-cell placement at the actual planned camera depth.

        Only commands are predicted here. Accepted archive cells still come
        exclusively from the new exposure's measured pose.
        """
        from autoweld.scan_footprint import footprint_cells
        baseline = np.asarray(config['baseline_optical'],float)
        fixed = np.asarray(config['fixed_optical_rotation'],float)
        laser = np.asarray(config['laser_plane_camera_m'],float)
        point_xy = np.asarray(point_xy,float)
        nominal = point_xy+np.asarray(config['origin_compensation_xy_m'],float)
        camera = baseline.copy() if initial_camera is None else np.array(initial_camera,float,copy=True)
        camera[:3,:3] = fixed
        xy = nominal.copy() if initial_camera is None else (baseline[:3,:3].T@(camera[:3,3]-baseline[:3,3]))[:2]
        bias = np.zeros(3)
        actual_rotation = fixed
        if measured_camera is not None:
            measured = np.asarray(measured_camera,float)
            commanded = np.asarray(commanded_camera,float)
            bias = baseline[:3,:3].T@(measured[:3,3]-commanded[:3,3])
            actual_rotation = measured[:3,:3]
        normal = baseline[:3,:3].T@actual_rotation@laser[:3]
        denominator = float(normal[:2]@normal[:2])
        if denominator < 1e-8:
            raise ValueError('Footprint normal cannot steer the requested reference cell')
        desired = np.r_[point_xy,config['reference_plane_z_m']]
        for iteration in range(24):
            local = np.r_[xy,float(field.depth(*xy))]
            if not np.isfinite(local).all() or abs(local[2]) > .060:
                raise ValueError('Depth-aware footprint placement exceeds bounded application compensation')
            if np.max(abs(xy-nominal)) > .018:
                xy = nominal.copy()
                continue
            residual = float(normal@(desired-local-bias)+laser[3])
            if abs(residual) <= 1e-8:
                break
            delta = normal[:2]*residual/denominator
            xy += delta*min(1.,.010/max(np.linalg.norm(delta),1e-18))
        else:
            # Steep but bounded depth priors can defeat fixed-point iteration.
            # Fall back to an actual scalar sign bracket, never accept residual.
            from scipy.optimize import brentq
            direction = normal[:2]/np.sqrt(denominator)
            center = nominal.copy()
            def scalar(offset):
                candidate = center+direction*offset
                if np.max(abs(candidate-nominal)) > .018:
                    raise ValueError('Footprint root exceeds compensation bounds')
                depth = float(field.depth(*candidate))
                if not np.isfinite(depth) or abs(depth) > .060:
                    raise ValueError('Invalid bounded footprint depth')
                return float(normal@(desired-np.r_[candidate,depth]-bias)+laser[3])
            samples = []
            for offset in np.linspace(-.018,.018,73):
                try:
                    samples.append((float(offset),scalar(offset)))
                except ValueError:
                    pass
            brackets = [(a,b) for (a,fa),(b,fb) in zip(samples,samples[1:]) if fa*fb <= 0]
            if not brackets:
                raise ValueError('Depth-aware footprint placement has no bounded root')
            low,high = min(brackets,key=lambda ends:abs(sum(ends)))
            offset = low if scalar(low) == 0 else brentq(scalar,low,high,xtol=1e-11,maxiter=64)
            xy = center+direction*offset
            local = np.r_[xy,float(field.depth(*xy))]
            residual = float(normal@(desired-local-bias)+laser[3])
            iteration = 24
        if abs(residual) > 1e-8:
            raise ValueError('Depth-aware footprint placement residual remains unqualified')
        camera[:3,3] = baseline[:3,3]+baseline[:3,:3]@local
        predicted = camera.copy()
        predicted[:3,:3] = actual_rotation
        predicted[:3,3] += baseline[:3,:3]@bias
        width = round(config['width_mm'])
        cell = round((point_xy[1]+config['height_mm']/2000.)*1000)*(width+1)+round((point_xy[0]+config['width_mm']/2000.)*1000)
        if cell not in footprint_cells(config,predicted)['cells']:
            raise ValueError('Solved reference cell is outside the predicted finite sensor footprint')
        return dict(camera=camera,predicted_camera=predicted,residual_m=residual,iterations=iteration+1,
            compensation_xy_m=xy-nominal)

    def _repair_footprint(self, context):
        """Fill measured reference-plane holes with a few normal discrete moves.

        Predictions choose poses only. A cell is observed only after archive
        registration of a new exposure at its actual measured camera pose.
        """
        from autoweld.scan_footprint import footprint_cells
        config = context['data']['baseline']['scan_footprint']
        original = np.asarray(context['data']['baseline']['baseline_optical'])
        handeye = np.asarray(context['data']['baseline']['handeye']['tcp_from_camera'])
        shift = np.asarray(config['origin_compensation_xy_m'])
        field = context['data']['scout']['field']
        columns = round(context['settings']['width_mm'])+1
        half = np.array([context['settings']['width_mm'],context['settings']['height_mm']])/2000.
        deadline = min(time.monotonic()+self.FOOTPRINT_REPAIR_BUDGET_S,self.coordinator._started+context['settings']['max_duration_s']-self.COMPLETION_BUDGET_S)
        attempted = {}
        solutions = {}

        def target(cell):
            point = np.array([cell%columns/1000.,cell//columns/1000.])-half
            if cell not in solutions:
                solutions[cell] = self._solve_footprint_target(config,field,point)
            return point,solutions[cell]

        def acquire(camera,point):
            from autoweld.motion_registration import RegistrationRejected
            local = original[:3,:3].T@(camera[:3,3]-original[:3,3])
            needed_pad = float(np.max(abs(local[:2]-(point+shift))))+.002
            if needed_pad > .020+1e-12:
                raise ValueError('Footprint repair requires more than20mm application compensation')
            with self._lock:
                self._area_repair_pad_m = max(self._area_repair_pad_m,needed_pad)
            goal = camera@np.linalg.inv(handeye)
            self._command('move_pose',{'pose':goal.tolist()},context)
            self._settle(context,goal)
            self._previous_centroid = None
            self._pending_clouds.clear()
            self._pending_registration = None
            self._last_ring_key = self._last_enqueued_key = None
            self._capture_cutoff = time.monotonic()-.03
            started = time.monotonic()
            count = 0
            first_key = last_key = None
            # Retain one fixed measured Hold throughout. Only NEW qualified
            # exposures count; no repeated Hold or predicted coverage is used.
            while time.monotonic() < min(deadline,started+2.):
                self._check_cancel(context)
                try:
                    result = self._capture_one(context)
                except RegistrationRejected as error:
                    if any(word in str(error).lower() for word in ('identity','different','session','clock mapping')):
                        raise
                    self._discard_bad_exposure(context,str(error))
                    result = None
                if result and result.get('qualified') and self._last_key is not None and self._last_key != last_key:
                    count += 1
                    first_key = self._last_key if first_key is None else first_key
                    last_key = self._last_key
                elapsed = time.monotonic()-started
                if not self.archive.footprint_missing_cells():
                    break
                if count >= 10 and elapsed >= .3:
                    break
                if count and elapsed >= .6:
                    break
                context['cancel_event'].wait(.003)
            if not count:
                raise TimeoutError('No qualified fresh footprint repair exposures within bounded burst')
            return dict(new_qualified_frames=count,duration_s=time.monotonic()-started,
                first_acquisition=first_key,last_acquisition=last_key,one_fixed_hold=True)

        for attempt in range(self.FOOTPRINT_REPAIR_ATTEMPTS):
            self._check_cancel(context)
            missing = set(self.archive.footprint_missing_cells())
            if not missing or time.monotonic()+3. > deadline:
                break
            candidates = [cell for cell in sorted(missing) if attempted.get(cell,0) < 4]
            if not candidates:
                break
            if len(candidates) > 64:
                candidates = [candidates[index] for index in np.linspace(0,len(candidates)-1,64,dtype=int)]
            best = None
            for cell in candidates:
                try:
                    point,solution = target(cell)
                except ValueError:
                    attempted[cell] = 4
                    continue
                gain = len(missing.intersection(footprint_cells(config,solution['predicted_camera'])['cells']))
                if best is None or gain > best[0]:
                    best = (gain,cell,point,solution)
            if best is None:
                break
            _,cell,point,solution = best
            camera = solution['camera'].copy()
            attempted[cell] = attempted.get(cell,0)+1
            burst = acquire(camera,point)
            after = set(self.archive.footprint_missing_cells())
            correction = correction_burst = None
            if cell in after and time.monotonic()+3. <= deadline:
                try:
                    learned = self._solve_footprint_target(config,field,point,initial_camera=camera,
                        measured_camera=self._last_registered_pose,commanded_camera=camera)
                    correction = (original[:3,:3].T@(learned['camera'][:3,3]-camera[:3,3]))[:2]
                    if np.linalg.norm(correction) > .010:
                        raise ValueError('Measured footprint correction exceeds10mm')
                    solutions[cell] = learned  # Persist learned correction across attempts.
                    if np.linalg.norm(correction) > 1e-6:
                        camera = learned['camera'].copy()
                        correction_burst = acquire(camera,point)
                        after = set(self.archive.footprint_missing_cells())
                except ValueError as error:
                    solution['correction_error'] = str(error)
            self._repairs.append(dict(kind='reference_footprint_patch',attempt=attempt+1,target_cell=int(cell),
                missing_cells_before=len(missing),settled_burst=burst,correction_burst=correction_burst,
                placement_residual_m=solution['residual_m'],placement_iterations=solution['iterations'],
                compensation_xy_m=solution['compensation_xy_m'].tolist(),correction_error=solution.get('correction_error'),
                observed_cells_added=len(missing)-len(after),remaining_cells=len(after),
                correction_xy_m=None if correction is None else correction.tolist(),
                predicted_cells_never_count_as_observed=True))

    def _repair_gaps(self, context):
        """Bounded normal-path row resweeps; no relaxed quality or motion gates."""
        if not self._quality_gaps:
            return
        from autoweld.motion import MotionLimits
        from autoweld.scan_plan_probe import build_scan_plan, compact_plan_summary, private_planner
        from autoweld.scan_surface import SurfaceDepthField
        height = round(context['settings']['height_mm'])
        baseline = context['data']['baseline']
        original = np.asarray(baseline['baseline_optical'])
        pairs = []
        for row in sorted(set(int(gap['row']) for gap in self._quality_gaps if 'row' in gap)):
            low = max(0,min(height-1,row))
            if not any(row in (pair,pair+1) for pair in pairs):
                pairs.append(low)
        repair_started = time.monotonic()
        for low in pairs[:3]:
            self._check_cancel(context)
            remaining = self.coordinator._started+context['settings']['max_duration_s']-time.monotonic()-self.COMPLETION_BUDGET_S
            if remaining < 18. or time.monotonic()-repair_started > 35.:
                break
            points = np.concatenate(tuple(self._surface_views)) if self._surface_views else np.empty((0,3))
            field = SurfaceDepthField(points, original, fixed_optical_rotation=baseline['fixed_optical_rotation'],
                laser_offset_camera_m=baseline['laser_offset_camera_m'], last_safe_camera_z_m=baseline['depth_offset_m'],
                support_radius_m=.015, nearest_limit_m=.012)
            center_y = (low+.5)/1000.-context['settings']['height_mm']/2000.
            shifted = np.asarray(baseline['baseline_pose']).copy()
            shifted[:3,3] += original[:3,:3]@np.array([0.,center_y,0.])
            snapshot = self.snapshot()
            source = copy.copy(self.service.planner)
            source.limits = MotionLimits(**snapshot['limits'])
            worker = private_planner(source)
            plan = build_scan_plan(worker, baseline_pose=shifted, tcp_from_camera=baseline['handeye']['tcp_from_camera'],
                seed_q=snapshot['q'], bounds=snapshot['bounds'],gripper=snapshot['gripper'],
                optical_roll_deg=context['settings']['optical_roll_deg'], feed_mm_s=min(20.,self._selected_feed_mm_s or context['settings']['feed_mm_s']),
                width_mm=context['settings']['width_mm'],height_mm=1.,spacing_mm=1.,
                depth=lambda x,y:field.depth(x,y+center_y), max_preflight_s=min(15.,remaining-12.),
                max_total_s=remaining, cancelled=context['cancel_event'].is_set,
                model_fingerprint=self.service.model_fingerprint)
            if plan.path.duration+6. > self.coordinator._started+context['settings']['max_duration_s']-time.monotonic()-self.COMPLETION_BUDGET_S:
                break
            path = self.archive.session_directory/f'repair-plan-{len(self._repairs)}.json'
            path.write_text(json.dumps(compact_plan_summary(plan),allow_nan=False,indent=2))
            old_gaps = self._quality_gaps
            self._quality_gaps = []
            self._command('move_pose',{'pose':plan.report['entry_pose']},context)
            self._settle(context,plan.report['entry_pose'])
            self._last_registered = self._last_registered_time = self._previous_centroid = None
            self._pending_clouds.clear()
            self._pending_registration = None
            self._last_ring_key = self._last_enqueued_key = None
            self._capture_cutoff = time.monotonic()-.05
            self._wait(context,lambda _state:bool(self._capture_one(context)),2.,'Repair entry lacks measured exposure')
            self._command('start_path',{'plan':plan},context)
            self._start_collector(context)
            started = time.monotonic()
            try:
                while not self._motion_state(context).get('complete'):
                    self._check_cancel(context)
                    self._check_collector()
                    if time.monotonic()-started > plan.path.duration+5.:
                        raise TimeoutError('Coverage repair did not reach verified endpoint')
                    context['cancel_event'].wait(.01)
            finally:
                self._stop_collector()
            self._verify_end_coverage(context)
            new_gaps = self._quality_gaps
            self._quality_gaps = [gap for gap in old_gaps if gap.get('row') not in (low,low+1)]+new_gaps
            self._repairs.append(dict(rows=[low,low+1], plan_report_path=str(path),
                validated_duration_s=plan.path.duration, measured_endpoint_complete=True,
                newly_unresolved_gaps=len(new_gaps)))

    def _capture_one(self, context, scout_worlds=None):
        from autoweld.laser_scanning import segment_object
        from autoweld.motion_registration import RegistrationRejected
        self._check_cancel(context)
        getter = getattr(self.service.camera, 'recent_laser_clouds', None)
        if callable(getter):
            batch = getter(self._last_ring_key)
            if batch.get('overrun') or batch.get('session_changed'):
                raise RuntimeError('Native exposure ring overran or changed acquisition session')
            incoming = batch.get('clouds') or ()
            self._last_ring_key = batch.get('newest_key') or self._last_ring_key
        else:
            latest = self.cloud()
            incoming = () if latest is None else (latest,)
        for latest in incoming:
            latest_key = (latest.get('acquisition_session_id'), (latest.get('timing') or {}).get('device_frame_id'))
            latest_stamp = (latest.get('timing') or {}).get('exposure_host_mid_s', (latest.get('timing') or {}).get('exposure_host_midpoint_s'))
            if latest_stamp is None or latest_stamp < self._capture_cutoff:
                continue
            capture_metadata = getattr(self,'_capture_metadata',None) or {}
            exposure_start = (latest.get('timing') or {}).get('exposure_host_start_s')
            if (capture_metadata.get('acquisition_phase') in ('stationary_main','start_anchor','end_anchor')
                    and isinstance(exposure_start,(int,float)) and exposure_start < self._capture_cutoff):
                # A midpoint after the Hold gate can still belong to an exposure
                # that started before it. Ignore that immutable frame, retaining
                # the same verified Hold for the next wholly fresh exposure.
                self._last_station_capture_gate = dict(code='exposure_before_hold',
                    exposure_start_s=exposure_start,exposure_midpoint_s=latest_stamp,
                    cutoff_s=self._capture_cutoff,device_frame_id=latest_key[1])
                continue
            if latest_key != self._last_enqueued_key and latest_key != self._last_key:
                if len(self._pending_clouds) == self._pending_clouds.maxlen:
                    raise RuntimeError('Bounded exposure registration queue overran; refusing silent coverage loss')
                self._pending_clouds.append(latest)
                # Sensor cadence is independent of reference-registration gaps.
                # Rejected or pending brackets must not throttle the whole path.
                if np.isfinite(latest_stamp):
                    if self._last_native_stamp is not None and latest_stamp > self._last_native_stamp:
                        self._native_intervals.append(latest_stamp-self._last_native_stamp)
                    self._last_native_stamp = latest_stamp
                self._last_enqueued_key = latest_key
        if not self._pending_clouds:
            return None
        cloud = self._pending_clouds[0]
        key = (cloud.get('acquisition_session_id'), (cloud.get('timing') or {}).get('device_frame_id'))
        timing = cloud.get('timing') or {}
        stamp = timing.get('exposure_host_mid_s', timing.get('exposure_host_midpoint_s'))
        if stamp is None or not 0 <= time.monotonic()-stamp <= .5:
            raise RegistrationRejected('Moving native exposure is stale or in the future')
        history_state = getattr(self.history,'state',None)
        history_generation = history_state().get('generation') if callable(history_state) else None
        registration_key = (key,cloud.get('generation'),cloud.get('sequence'),id(self.registrar),
            self.history.history_id,history_generation,context.get('generation'))
        cached = getattr(self,'_pending_registration',None)
        if cached is not None and cached[0] == registration_key:
            registered = cached[1]
        else:
            try:
                registered = self.registrar.register_receive_reference(cloud)
            except RegistrationRejected as error:
                if 'no measured motor bracket' in str(error) and time.monotonic()-stamp < .45:
                    return None  # Wait for real future output samples; never extrapolate.
                raise
            self._pending_registration = (registration_key,registered)
        acquisition = copy.deepcopy(getattr(self,'_capture_metadata',None))
        if acquisition and acquisition.get('acquisition_phase') in ('stationary_main','start_anchor','end_anchor'):
            expected = np.asarray(acquisition['requested_base_from_tcp'])
            p,r = self._pose_errors(np.asarray(registered['base_from_tcp']),expected)
            snapshot = self.snapshot()
            feedback = snapshot.get('feedback') or {}
            settled = acquisition['settle']
            measured_at = feedback.get('measured_at')
            exposure_end = timing.get('exposure_host_end_s')
            now = time.monotonic()
            feedback_age = now-measured_at if isinstance(measured_at,(int,float)) and np.isfinite(measured_at) else None
            if not feedback.get('fresh') or feedback_age is None or not 0 <= feedback_age <= .1:
                self._last_station_capture_gate = dict(code='feedback_snapshot_not_fresh',
                    feedback=copy.deepcopy(feedback),feedback_age_s=feedback_age,
                    exposure_start_s=timing.get('exposure_host_start_s'),exposure_end_s=exposure_end,
                    exposure_midpoint_s=stamp,device_frame_id=key[1])
                return None  # Await a new main-owner snapshot at this SAME Hold.
            if exposure_end is not None and measured_at < exposure_end:
                return None  # Wait for real post-exposure quiet feedback.
            stable_p,stable_r = self._pose_errors(np.asarray(snapshot['tcp_pose']),np.asarray(settled['measured_base_from_tcp']))
            gate = None
            if timing.get('exposure_host_start_s',0) < settled['accepted_monotonic']:
                gate = 'exposure_before_hold'
            elif p > .0005 or r > .005:
                gate = 'requested_endpoint'
            elif stable_p > .0003 or stable_r > .003:
                gate = 'hold_pose_drift'
            elif not feedback.get('stationary'):
                gate = 'feedback_nonstationary'
            if gate:
                diagnostic = dict(code=gate,target_position_error_m=p,target_orientation_error_rad=r,
                    hold_position_drift_m=stable_p,hold_orientation_drift_rad=stable_r,
                    feedback=copy.deepcopy(feedback),feedback_age_s=None if measured_at is None else now-measured_at,
                    exposure_start_s=timing.get('exposure_host_start_s'),exposure_end_s=exposure_end,
                    exposure_midpoint_s=stamp,cutoff_s=settled['accepted_monotonic'],device_frame_id=key[1])
                self._last_station_capture_gate = diagnostic
                message = ('Stationary exposure started before accepted Hold' if gate == 'exposure_before_hold' else
                    f'Stationary endpoint or quiet feedback gate failed: {gate}')
                error = RegistrationRejected(message)
                error.diagnostic = diagnostic
                error.station_gate = gate
                raise error
            acquisition['stationary_feedback'] = dict(fresh=True,stationary=True,measured_at=measured_at,
                observed_monotonic=now,pose_position_drift_m=stable_p,pose_orientation_drift_rad=stable_r)
        self._pending_clouds.popleft()
        self._pending_registration = None
        registered['history_id'] = self.history.history_id
        self._last_key = key
        native = np.asarray(cloud.get('xyz_m'), float).reshape(-1, 3)
        pose = np.asarray(registered['base_from_tcp'])@np.asarray(context['data']['baseline']['handeye']['tcp_from_camera'])
        indices, status, reason = [], 'no_return', 'No native laser returns'
        if not len(native):
            self._previous_centroid = None
            self._centroid_origin = None
        if len(native):
            try:
                previous = self._previous_centroid
                if previous is not None and getattr(self, '_centroid_origin', None) is not None:
                    previous = previous + pose[:3,:3].T@(self._centroid_origin-pose[:3,3])
                segment = segment_object(cloud, previous)
                indices, status, reason = segment['indices'], 'accepted', None
                self._previous_centroid = np.asarray(segment['centroid_camera_m'])
                self._centroid_origin = pose[:3,3].copy()
            except ValueError as error:
                status, reason = 'rejected', str(error)
        if self._last_registered is not None and self._main_scan and status != 'rejected':
            original = np.asarray(context['data']['baseline']['baseline_optical'])
            delta = original[:3, :3].T@(pose[:3, 3]-self._last_registered)
            gap = float(np.linalg.norm(delta if self._is_line(context) else delta[:2]))
            self._max_path_gap = max(self._max_path_gap, gap)
            if gap > .001 and not self._is_line(context):
                self._quality_gaps.append(dict(start=self._last_registered_time, end=stamp, measured_gap_m=gap,
                    row=round((original[:3,:3].T@(pose[:3,3]-original[:3,3]))[1]*1000+context['settings']['height_mm']/2)))
                self.archive.record_gap('Measured exposure path gap exceeds 1 mm', start_monotonic_s=self._last_registered_time,
                    end_monotonic_s=stamp, path_fraction=self._fraction(), evidence={'measured_gap_m': gap})
        if status != 'rejected':
            self._last_registered, self._last_registered_time = pose[:3, 3].copy(), stamp
            self._last_registered_pose = pose.copy()
        saved = self.archive.append(cloud, registered, selected_indices=indices, status=status, reason=reason,
            path_fraction=self._fraction(),acquisition_metadata=acquisition)
        if status == 'accepted':
            selected = np.asarray(registered['points_world_m'])[indices]
            if scout_worlds is not None:
                scout_worlds.append(selected.copy())
            elif hasattr(self, '_surface_views'):
                self._surface_views.append(selected[::max(1,len(selected)//32+1)].copy())
        saved = self._display_saved(saved)
        saved['quality'] = dict(standoff_mm=float(np.median(native[indices,2]))*1000 if indices else None,
            stripe_points=len(native), object_points=len(indices))
        result = {**saved, 'qualified':status != 'rejected',
            'registration_reference':registered.get('registration_reference'),
            'reference_registered':registered['reference_registered'],
            'reference_timing_qualified':registered['reference_timing_qualified'],
            'physical_sample_timing_qualified':registered['physical_sample_timing_qualified'],
            'firmware_sample_age_s':registered.get('firmware_sample_age_s'),
            'reference_timing_uncertainty_s':registered['reference_timing_uncertainty_s'],
            'reference_spatial_error_bound_m':registered.get('reference_spatial_error_bound_m')}
        if acquisition is not None:
            result.update(acquisition_phase=acquisition.get('acquisition_phase'),station_index=acquisition.get('station_index'))
        # Both the continuous collector and direct repair bursts append here.
        # Publish the returned immutable snapshot; rendering must never wait on
        # the archive lock while final export owns it for streaming I/O.
        with self._lock:
            self._latest_archive_snapshot = result
            self._latest_archive_snapshot_monotonic = time.monotonic()
        return result


    def _fraction(self):
        if getattr(self,'_line_scope',None) is not None and self.coordinator.settings.scan_method == 'precision':
            return self._line_fraction
        if not self._main_scan or self._active_plan is None or self._path_started is None:
            return 0.
        self._max_fraction = max(self._max_fraction,
            min(1., max(0., (time.monotonic()-self._path_started)/self._active_plan.path.duration)))
        return self._max_fraction

    def _finalize(self, context):
        if self._is_line(context):
            return self._line_finalize(context)
        self._stop_collector()
        self._verify_end_coverage(context)
        if self._footprint_mode:
            self._repair_footprint(context)
            coverage = self.archive.state(include_points=False).get('coverage') or {}
            complete = coverage.get('footprint_complete') is True
        else:
            self._repair_gaps(context)
            coverage = self.archive.state(include_points=False).get('coverage') or {}
            complete = not self._quality_gaps
        missing = self.archive.footprint_missing_cells() if self._footprint_mode else []
        evidence = dict(measured_path_complete=True,path_fraction=1.,continuous_quality_coverage=complete,
            workflow_elapsed_before_finish_s=time.monotonic()-self.coordinator._started,
            remaining_footprint_cells=len(missing),unobserved_footprint_cell_indices=missing[:256],
            missing_cells_truncated=len(missing)>256,
            coverage_gaps_resolved=complete,actual_footprint_complete=complete if self._footprint_mode else None,
            footprint_observed_cells_1mm=coverage.get('footprint_observed_cells_1mm'),
            footprint_total_cells_1mm=coverage.get('footprint_total_cells_1mm'),
            maximum_measured_exposure_path_gap_m=self._max_path_gap,requested_maximum_path_gap_m=.001,
            temporal_origin_gaps_are_diagnostic_only=bool(self._footprint_mode),repairs=self._repairs)
        saved = self.archive.finish(status='complete' if complete else 'interrupted',
            message='Measured reference-plane area captured and exported' if complete else 'Captured points exported with explicit reference-plane coverage holes',
            completion_evidence=evidence)
        self._archive_started = False
        saved['completion_qualified'] = complete
        return self._display_saved(saved)

    def _return_rest(self, context):
        self._command('disable_at_rest', {}, context)
        self._wait(context, lambda state: bool(state.get('closed') and state.get('rest_verified')),
            55., 'Normal disable-at-rest did not obtain fresh verified rest')
        return {'rest_verified': True}

    def _release(self, context):
        self._collector_stop.set()
        self.bridge.invalidate('scan', through_generation=context['generation'])
        if self._archive_started and context['outcome'] != 'complete':
            self._archive_started = False
            self._cleanup_future = self._maintenance.submit(self.archive.finish,
                status='interrupted', message=context.get('error') or context['message'])

    def dispatch(self, request):
        action = request['action']
        if action == 'scan_start':
            if self.busy:
                raise RuntimeError('A scan already owns robot motion')
            if self._cleanup_future is not None and not self._cleanup_future.done():
                raise RuntimeError('Previous moving archive is still finishing')
            self._continuous_selected = request.get('continuous', True) or request.get('scan_profile') in ('vertical_down','downward_line','line_down','singleline')
            if not self._continuous_selected:
                return self.legacy.dispatch(request)
            self._last_key = self._previous_centroid = self._last_registered = None
            with self._lock:
                self._latest_archive_snapshot = None
                self._latest_archive_snapshot_monotonic = None
            self._main_scan = False
            self._run_has_archive = False
            self._max_path_gap = 0.
            self._planning_progress = None
            self._selected_feed_mm_s = None
            self._prepare_goal = None
            self._line_scope = self._capture_metadata = self._last_settle_evidence = None
            self._line_fraction = 0.
            self._collector_error = None
            self._area_camera_shift = np.zeros(2)
            self._area_plan_active = self._footprint_mode = False
            self._area_repair_pad_m = .002
            self._last_settle_retry = None
            self._last_settle_diagnostic = None
            self._last_line_preparation_diagnostic = None
            self._last_station_capture_diagnostic = None
            self._last_station_capture_gate = None
            self._last_station_retry = self._station_retry_context = None
            self._repairs = []
            self._max_fraction = 0.
            self._last_ring_key = None
            self._pending_clouds.clear()
            self._pending_registration = None
            self._last_enqueued_key = None
            self._active_plan = self._path_started = None
            self.coordinator.start(request)
            return self.state()
        if action == 'scan_resume':
            if self._continuous_selected:
                raise ValueError('Continuous scans require a new Start; legacy progress remains archived')
            return self.legacy.dispatch(request)
        if action == 'scan_cancel':
            self.cancel()
            return self.state()
        if action == 'scan_state':
            return self.state()
        if self.busy:
            raise RuntimeError('Wait for the scan before clearing or exporting')
        if not self._continuous_selected:
            return self.legacy.dispatch(request)
        if action == 'scan_clear':
            self._cleanup_future = self._maintenance.submit(self.archive.clear)
            return self.state()
        if action == 'scan_export':
            return self.archive.state()['export_paths']
        raise ValueError('Unknown scan action')

    def state(self, camera_state=None):
        if not self._continuous_selected:
            return self.legacy.state(camera_state=camera_state)
        current = self.coordinator.state(camera_state=camera_state)
        with self._lock:
            saved = self._latest_archive_snapshot
            snapshot_at = self._latest_archive_snapshot_monotonic
        if not self.coordinator.busy and getattr(self,'_run_has_archive',True):
            saved = self.archive.state()
            current = {**saved, **current}
        if not getattr(self,'_run_has_archive',True):
            saved = None
            current.update(scan_id=None,cloud=None,coverage={},export_paths={},point_count=0,
                display_point_count=0,native_frame_count=0,captures=0,accepted=0,no_return=0,rejected=0,visited=0)
        if saved is not None:
            for key in self.coordinator.PUBLIC | {'native_frame_count'}:
                if key in saved:
                    current[key] = saved[key]
        if snapshot_at is not None:
            current['archive_snapshot_age_s'] = max(0.,time.monotonic()-snapshot_at)
        if self._planning_progress is not None and self.coordinator.status in ('scout', 'plan'):
            current['planning_progress'] = self._planning_progress
        current = self._display_saved(current)
        current['command_bridge'] = self.bridge.state()
        current['maximum_measured_exposure_path_gap_m'] = self._max_path_gap
        current['selected_feed_mm_s'] = self._selected_feed_mm_s
        baseline = self.coordinator._data.get('baseline') or {}
        footprint = baseline.get('scan_footprint')
        if footprint is not None:
            current['footprint_plan'] = {key:footprint.get(key) for key in
                ('sweep_axis','lane_spacing_mm','lanes','origin_compensation_xy_m','projected_cross_lane_span_m','reference_plane_z_m')}
            current['coverage_basis'] = 'measured_reference_plane_laser_footprint'
        current['settle_retry'] = copy.deepcopy(getattr(self,'_last_settle_retry',None))
        current['settle_diagnostic_path'] = getattr(self,'_last_settle_diagnostic',None)
        current['preparation_diagnostic_path'] = getattr(self,'_last_line_preparation_diagnostic',None)
        current['station_capture_diagnostic_path'] = getattr(self,'_last_station_capture_diagnostic',None)
        current['station_retry'] = copy.deepcopy(getattr(self,'_last_station_retry',None))
        return current

    def tick(self, camera_state=None):
        self._refresh()
        self.bridge.drain(max_commands=8, max_wall_s=.02)
        if self.legacy.busy:
            self.legacy.tick(camera_state=camera_state)
        self.coordinator.tick(camera_state=camera_state)
        self.bridge.drain(max_commands=8, max_wall_s=.02)

    def cancel(self, *, hold=True, reason='Scan cancelled; points retained'):
        self.coordinator.cancel(hold=hold, reason=reason)
        self.legacy.cancel(hold=hold, reason=reason)

    def abort_for_shutdown(self):
        self.coordinator.abort_for_shutdown()
        self.legacy.abort_for_shutdown()

    def abort_for_cooldown(self):
        self.coordinator.abort_for_cooldown()
        self.legacy.abort_for_cooldown()

    def close(self):
        self.coordinator.close()
        self.bridge.close()
        if self._detach is not None:
            self._detach()
        self.legacy.close()
        self._maintenance.shutdown(wait=False, cancel_futures=False)
        self._collector_executor.shutdown(wait=False, cancel_futures=True)


def as_motion_limits(limits):
    from dataclasses import asdict
    return asdict(limits)
