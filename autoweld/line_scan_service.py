"""Matched continuous/stationary line protocols through the service command bridge."""
import copy
import hashlib
import json
import time
import uuid

import numpy as np


class VerticalScanMixin:
    def _is_line(self,context=None):
        settings = context['settings'] if context is not None else self.coordinator.settings.__dict__
        return settings.get('scan_profile') == 'vertical_down'

    def _line_binding(self,context,handeye):
        state = self.camera_state()
        native = self.cloud() or {}
        saved = copy.deepcopy(self.service.saved_poses.get('Tube Scanning'))
        def digest(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return dict(source_id=native.get('source_id',state.get('source_id')),optical_intrinsics_id=native.get('optical_intrinsics_id',state.get('optical_intrinsics_id')),
            calibration_id=native.get('calibration_id',state.get('calibration_id')),handeye_sha256=digest(handeye['tcp_from_camera']),
            saved_tube_sha256=digest(saved),sweep_length_mm=context['settings']['sweep_length_mm'],
            optical_roll_deg=context['settings']['optical_roll_deg'])

    def _load_line_reference(self,context):
        if context['settings']['scan_method'] != 'precision':
            return None
        path = self.service.runtime/'vertical-line-reference.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        handeye = self.camera_state().get('handeye') or {}
        if record.get('binding') != self._line_binding(context,handeye):
            raise ValueError('Matched Fast reference calibration/Tube/length/roll changed; start a new Fast sweep')
        return record

    def _restore_line_baseline(self,context,record):
        from autoweld.calibration import transform
        handeye = copy.deepcopy(self.camera_state()['handeye'])
        correction = transform(handeye['tcp_from_camera'])
        original = transform(record['baseline_pose'])@correction
        start = transform(record['line_start_base_from_camera'])
        self._envelope = (original,correction,start[:3,:3].copy(),np.array([.05,.05]))
        self._line_scope = dict(start=start.copy(),length_m=context['settings']['sweep_length_mm']/1000.)
        goal = start@np.linalg.inv(correction)
        self._command('move_pose',{'pose':goal.tolist()},context)
        self._settle(context,goal)
        native,_,segment = self._good_observation()
        return dict(baseline_pose=record['baseline_pose'],handeye=handeye,baseline_optical=original.tolist(),
            settled_optical=(np.asarray(self.snapshot()['tcp_pose'])@correction).tolist(),
            fixed_optical_rotation=start[:3,:3].tolist(),line_start_base_from_camera=start.tolist(),
            depth_offset_m=0.,selected_optical_roll_deg=record.get('selected_optical_roll_deg',context['settings']['optical_roll_deg']),comparison_id=record['comparison_id'],comparison_binding=record['binding'],
            preparation_changes=[dict(kind='matched_fast_start_recall',reference_id=record['comparison_id'])],
            identity=[native[k] for k in ('source_id','optical_intrinsics_id','calibration_id','acquisition_session_id')],
            quality=dict(stripe_points=len(native['xyz_m']),object_points=len(segment['indices']),
                standoff_mm=segment['centroid_camera_m'][2]*1000.))

    def _finish_line_baseline(self,context,result,native):
        handeye = np.asarray(result['handeye']['tcp_from_camera'])
        rotation = np.asarray(result['fixed_optical_rotation'])
        calibration = self.camera_state()['intrinsics']
        normal = np.asarray(calibration['laser_plane'],float)[:3]
        tangent = np.cross(normal,[0.,0.,1.])
        if np.linalg.norm(tangent[:2]) < 1e-8:
            raise ValueError('Laser plane has no cameraXY tangent for bounded centering')
        tangent /= np.linalg.norm(tangent)
        if tangent[0] < 0:
            tangent *= -1
        changes = []
        for _ in range(5):
            native,_,segment = self._good_observation()
            uv = np.asarray(native['uv'])[segment['indices']]
            offset = float(np.mean(uv[:,0])-native['image_size'][0]/2)
            if abs(offset) <= native['image_size'][0]*.10:
                break
            camera = np.asarray(self.snapshot()['tcp_pose'])@handeye
            camera[:3,:3] = rotation
            moved,change = self._optional_line_center(context,camera,handeye,tangent,offset,native)
            changes.append(change)
            if not moved:
                break  # A usable stripe remains; the full line still preflights.
        # Recheck target standoff after the full native arc becomes visible.
        for _ in range(6):
            native,_,segment = self._good_observation()
            error = float(segment['centroid_camera_m'][2])-.140
            if abs(error) <= .001:
                break
            camera = np.asarray(self.snapshot()['tcp_pose'])@handeye
            camera[:3,:3] = rotation
            step = float(np.clip(error,-.005,.005))
            camera[:3,3] += rotation[:,2]*step
            goal = camera@np.linalg.inv(handeye)
            self._command('move_pose',{'pose':goal.tolist()},context)
            self._settle(context,goal)
            changes.append(dict(kind='initial_standoff',optical_z_delta_m=step))
        native,_,segment = self._good_observation()
        if abs(float(segment['centroid_camera_m'][2])-.140) > .001:
            raise ValueError('Prepared line standoff did not converge within bounded 5 mm corrections')
        start = np.asarray(self.snapshot()['tcp_pose'])@handeye
        start[:3,:3] = rotation
        result.update(line_start_base_from_camera=start.tolist(),preparation_changes=changes,
            comparison_id=uuid.uuid4().hex,comparison_binding=self._line_binding(context,result['handeye']),
            quality=dict(stripe_points=len(native['xyz_m']),object_points=len(segment['indices']),
                standoff_mm=segment['centroid_camera_m'][2]*1000.))
        self._line_scope = dict(start=start.copy(),length_m=context['settings']['sweep_length_mm']/1000.)
        return result

    def _save_line_reference(self,context,baseline):
        if context['settings']['scan_method'] != 'fast':
            return
        path = self.service.runtime/'vertical-line-reference.json'
        record = dict(schema='autoweld.matched-vertical-line.v1',comparison_id=baseline['comparison_id'],
            binding=baseline['comparison_binding'],baseline_pose=baseline['baseline_pose'],
            line_start_base_from_camera=baseline['line_start_base_from_camera'],
            preparation_changes=baseline['preparation_changes'],
            selected_optical_roll_deg=baseline['selected_optical_roll_deg'])
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,allow_nan=False,indent=2))
        temporary.replace(path)

    def _optional_line_center(self,context,camera,handeye,tangent,offset,native):
        """Only retry rejected, never-submitted optional centering translations."""
        from autoweld.diagnostics import MotionRejection
        allowed = {'ik_nonconvergence','joint_step','self_collision','joint_bounds','tcp_residual','command_constraint'}
        attempts = []
        def persist():
            directory = self.service.runtime/'evidence'
            directory.mkdir(parents=True,exist_ok=True)
            path = directory/f'line-centering-{context.get("generation",0)}-{time.time_ns()}.json'
            path.write_text(json.dumps(dict(schema='autoweld.line-centering.v1',phase=context.get('phase'),
                u_offset_px=float(offset),native_device_frame_id=(native.get('timing') or {}).get('device_frame_id'),
                native_identity={k:native.get(k) for k in ('source_id','optical_intrinsics_id','calibration_id','acquisition_session_id')},
                attempts=attempts),allow_nan=False,indent=2))
            self._last_line_preparation_diagnostic = str(path)
            return str(path)
        for step in (.005,.0025,.00125):
            self._check_cancel(context)
            candidate = camera.copy()
            delta = tangent*step*np.sign(offset)
            candidate[:3,3] += camera[:3,:3]@delta
            goal = candidate@np.linalg.inv(handeye)
            before = self.snapshot()
            attempt = dict(step_m=step,camera_delta_m=delta.tolist(),requested_base_from_tcp=goal.tolist(),
                before=before,stage='command')
            attempts.append(attempt)
            try:
                self._command('move_pose',{'pose':goal.tolist()},context)
            except Exception as error:
                attempt.update(error=str(error),diagnostic=copy.deepcopy(getattr(error,'diagnostic',None)),after=self.snapshot())
                path = persist()
                if (not isinstance(error,MotionRejection) or error.diagnostic.get('code') not in allowed
                        or error.diagnostic.get('pre_motion_rejection') is not True):
                    error.add_note(f'Line preparation command diagnostics: {path}')
                    raise
                rejected_at = time.monotonic()
                def quiet_after_rejection(state):
                    feedback = state.get('feedback') or {}
                    return (state.get('measured_monotonic',0) > rejected_at and not state.get('motion_active')
                        and feedback.get('fresh') and feedback.get('stationary'))
                self._wait(context,quiet_after_rejection,.5,'Optional centering rejection did not retain fresh quiet Hold')
                continue
            attempt['stage'] = 'settle'
            try:
                self._settle(context,goal)
            except Exception as error:
                attempt.update(error=str(error),after=self.snapshot())
                path = persist()
                error.add_note(f'Line preparation settle diagnostics: {path}')
                raise  # Never halve or skip a command that was actually submitted.
            attempt.update(stage='complete',after=self.snapshot())
            path = persist()
            return True,dict(kind='laser_tangent_centering',camera_delta_m=delta.tolist(),u_offset_px=float(offset),diagnostic_path=path)
        return False,dict(kind='laser_tangent_centering_skipped',reason='All bounded optional centering steps rejected before motion',
            u_offset_px=float(offset),diagnostic_path=persist())

    def _line_goal(self,context,index):
        baseline = context['data']['baseline']
        camera = np.asarray(baseline['line_start_base_from_camera']).copy()
        camera[:3,3] += [0.,0.,-index/1000.]
        return camera@np.linalg.inv(np.asarray(baseline['handeye']['tcp_from_camera']))

    def _line_plan(self,context):

        from autoweld.motion import MotionLimits
        from autoweld.scan_plan_probe import compact_plan_summary, private_planner
        from autoweld.vertical_scan import build_vertical_plan
        baseline = context['data']['baseline']
        snapshot = self.snapshot()
        source = copy.copy(self.service.planner)
        source.limits = MotionLimits(**snapshot['limits'])
        available = context['settings']['max_duration_s']-(time.monotonic()-context['started_monotonic'])-55.
        if available <= 0:
            raise ValueError('No line workflow budget remains')
        from scipy.spatial.transform import Rotation

        from autoweld.diagnostics import MotionRejection
        start = np.asarray(baseline['line_start_base_from_camera']).copy()
        requested = float(context['settings']['optical_roll_deg'])
        selected = float(baseline.get('selected_optical_roll_deg',requested))
        offsets = (0.,-15.,-30.,-45.,15.,30.) if context['settings']['scan_method'] == 'fast' else (0.,)
        failures = []
        planner = private_planner(source)  # Private DATA only; same installed MODEL.
        started = time.monotonic()
        directory = self.service.runtime/'evidence'
        directory.mkdir(parents=True,exist_ok=True)
        diagnostic_path = directory/f'line-plan-{context.get("generation",0)}-{time.time_ns()}.json'
        def persist(status):
            diagnostic_path.write_text(json.dumps(dict(schema='autoweld.line-plan-candidates.v1',status=status,
                requested_optical_roll_deg=requested,selected_optical_roll_deg=selected if status=='selected' else None,
                original_start_base_from_camera=start.tolist(),direction_base=[0.,0.,-1.],
                tcp_from_camera=baseline['handeye']['tcp_from_camera'],identity=baseline.get('identity'),
                sweep_length_mm=context['settings']['sweep_length_mm'],snapshot=snapshot,
                attempts=failures,planning_progress=self._planning_progress),allow_nan=False,indent=2))
            self._last_line_preparation_diagnostic = str(diagnostic_path)
        for offset in offsets:
            self._check_cancel(context)
            if not -180. <= selected+offset <= 180.:
                continue
            if time.monotonic()-started > 175.:
                raise TimeoutError('Bounded whole-line wrist-roll candidate preflight exhausted175 seconds')
            candidate = start.copy()
            candidate[:3,:3] = start[:3,:3]@Rotation.from_euler('z',offset,degrees=True).as_matrix()
            remaining = context['settings']['max_duration_s']-(time.monotonic()-context['started_monotonic'])-55.
            if remaining <= 0:
                persist('budget_exhausted')
                raise TimeoutError('No whole-line motion/export/rest budget remains')
            attempt = dict(optical_roll_deg=selected+offset,offset_deg=offset,
                start_base_from_camera=candidate.tolist(),seed_q=snapshot['q'],status='planning')
            failures.append(attempt)
            try:
                plan = build_vertical_plan(planner,start_base_from_camera=candidate,
                    tcp_from_camera=baseline['handeye']['tcp_from_camera'],seed_q=snapshot['q'],bounds=snapshot['bounds'],
                    gripper=snapshot['gripper'],length_mm=context['settings']['sweep_length_mm'],
                    feed_mm_s=context['settings']['feed_mm_s'],max_total_s=remaining,
                    cancelled=lambda:context['cancel_event'].is_set() or time.monotonic()-started > 175.,
                    progress=lambda state:setattr(self,'_planning_progress',state))
            except Exception as error:
                attempt.update(status='rejected',error=str(error),diagnostic=copy.deepcopy(getattr(error,'diagnostic',None)))
                fraction = (getattr(error,'diagnostic',None) or {}).get('path_fraction')
                if isinstance(fraction,(int,float)) and np.isfinite(fraction) and 0 <= fraction <= 1:
                    u = float(fraction)
                    distance = context['settings']['sweep_length_mm']/1000.*(35*u**4-84*u**5+70*u**6-20*u**7)
                    target = candidate.copy()
                    target[2,3] -= distance
                    attempt['failed_requested_base_from_tcp'] = (target@np.linalg.inv(np.asarray(baseline['handeye']['tcp_from_camera']))).tolist()
                persist('preflighting')
                if not isinstance(error,MotionRejection) or error.diagnostic.get('code') not in ('ik_nonconvergence','joint_bounds'):
                    error.add_note(f'Whole-line preflight diagnostics: {diagnostic_path}')
                    raise
                continue
            selected += offset
            attempt['status'] = 'validated'
            baseline.update(line_start_base_from_camera=candidate.tolist(),fixed_optical_rotation=candidate[:3,:3].tolist(),
                selected_optical_roll_deg=selected,roll_candidate_failures=copy.deepcopy(failures[:-1]),
                line_plan_diagnostic_path=str(diagnostic_path))
            self._line_scope = dict(start=candidate.copy(),length_m=context['settings']['sweep_length_mm']/1000.)
            with self._lock:
                original,correction,_,half = self._envelope
                self._envelope = (original,correction,candidate[:3,:3].copy(),half)
            self._save_line_reference(context,baseline)
            persist('selected')
            break
        else:
            persist('exhausted')
            raise ValueError(f'No bounded wrist roll produced a fully valid80mm world-minus-Z line; diagnostics: {diagnostic_path}')
        self._begin_archive(context)
        path = self.archive.session_directory/'validated-line-plan.json'
        path.write_text(json.dumps(compact_plan_summary(plan),allow_nan=False,indent=2))
        length = round(context['settings']['sweep_length_mm'])
        duration = plan.path.duration if context['settings']['scan_method'] == 'fast' else (length+1)*4.
        return dict(plan=plan,validated=True,duration_s=duration,entry_duration_s=5.,completion_budget_s=45.,
            line_start_base_from_camera=baseline['line_start_base_from_camera'],plan_report_path=str(path),
            geometry=self._line_geometry(context),station_total=length+1,selected_optical_roll_deg=selected,
            roll_candidate_failures=copy.deepcopy(failures[:-1]),line_plan_diagnostic_path=str(diagnostic_path))

    def _line_geometry(self,context):
        start = np.asarray(context['data']['baseline']['line_start_base_from_camera'])
        end = start[:3,3]+[0.,0.,-context['settings']['sweep_length_mm']/1000.]
        # Existing scene API accepts a closed four-segment outline; repeated
        # endpoints draw precisely the same one-dimensional line.
        return dict(units='m',coordinate_frame='robot_base',outline_base_m=[start[:3,3].tolist(),end.tolist(),end.tolist(),start[:3,3].tolist(),start[:3,3].tolist()])

    def _capture_line_station(self,context,index,phase,count=3):
        from autoweld.motion_registration import RegistrationRejected
        evidence = copy.deepcopy(getattr(self,'_last_settle_evidence',None))
        if not evidence:
            raise ValueError('Stationary capture requires measured settle evidence')
        self._capture_metadata = dict(scan_protocol='precision_stationary' if phase == 'stationary_main' else 'fast_continuous',
            acquisition_phase=phase,station_index=index if phase == 'stationary_main' else None,
            down_mm=float(index),requested_base_from_tcp=self._line_goal(context,index).tolist(),settle=evidence,
            station_retry=copy.deepcopy(getattr(self,'_station_retry_context',None)))
        self._pending_clouds.clear()
        self._pending_registration = None
        self._last_ring_key = self._last_enqueued_key = None
        self._capture_cutoff = evidence['accepted_monotonic']
        started = time.monotonic()
        captured = 0
        try:
            while captured < count:
                self._check_cancel(context)
                if time.monotonic()-started > 3.:
                    last_gate = (getattr(self,'_last_station_capture_gate',None) or {}).get('code','none')
                    raise TimeoutError(f'Station did not obtain three fresh qualified exposures within3 seconds; last gate: {last_gate}')
                try:
                    result = self._capture_one(context)
                except RegistrationRejected as error:
                    physical_gate = getattr(error,'station_gate',None) in ('requested_endpoint','hold_pose_drift','feedback_freshness','feedback_nonstationary')
                    if physical_gate:
                        directory = self.service.runtime/'evidence'
                        directory.mkdir(parents=True,exist_ok=True)
                        path = directory/f'station-capture-{context.get("generation",0)}-{time.time_ns()}.json'
                        path.write_text(json.dumps(dict(station_index=index,acquisition_phase=phase,
                            settle=evidence,error=str(error),diagnostic=error.diagnostic),allow_nan=False,indent=2))
                        self._last_station_capture_diagnostic = str(path)
                    if physical_gate or any(word in str(error).lower() for word in ('identity','different','session','clock mapping')):
                        raise
                    self._discard_bad_exposure(context,str(error))
                    result = None
                if result and result.get('qualified'):
                    captured += 1
                context['cancel_event'].wait(.003)
        finally:
            self._capture_metadata = None
        return captured

    def _capture_precision_with_retry(self,context,index):
        """One explicit same-goal normal move may renew a failed station gate."""
        from autoweld.motion_registration import RegistrationRejected
        allowed = {'requested_endpoint','hold_pose_drift','feedback_freshness','feedback_nonstationary'}
        self._station_retry_context = dict(attempt=0,maximum_replans=1,station_index=index)
        try:
            return self._capture_line_station(context,index,'stationary_main')
        except RegistrationRejected as error:
            if getattr(error,'station_gate',None) not in allowed:
                raise
            self._check_cancel(context)
            goal = self._line_goal(context,index)
            record = dict(schema='autoweld.precision-station-retry.v1',station_index=index,attempt=1,maximum_replans=1,
                reason=str(error),initial_gate=copy.deepcopy(getattr(error,'diagnostic',None)),
                initial_capture_diagnostic_path=getattr(self,'_last_station_capture_diagnostic',None),
                requested_base_from_tcp=goal.tolist(),started_monotonic=time.monotonic())
            directory = self.service.runtime/'evidence'
            directory.mkdir(parents=True,exist_ok=True)
            path = directory/f'precision-station-retry-{context.get("generation",0)}-{time.time_ns()}.json'
            record['evidence_path'] = str(path)
            def publish(status):
                record.update(status=status,updated_monotonic=time.monotonic())
                self._last_station_retry = copy.deepcopy(record)
                path.write_text(json.dumps(record,allow_nan=False,indent=2))
            publish('awaiting_fresh_quiet_feedback')
            try:
                rejected_at = time.monotonic()
                self._wait(context,lambda state:bool(state.get('measured_monotonic',0) > rejected_at
                    and not state.get('closed') and not state.get('disable_pending') and not state.get('motion_active')
                    and (state.get('feedback') or {}).get('fresh') and (state.get('feedback') or {}).get('stationary')),
                    .5,'Station retry did not retain fresh healthy quiet SDK control')
                self._check_cancel(context)
                publish('normal_same_goal_move')
                admission = self._command('move_pose',{'pose':goal.tolist()},context)
                record['normal_motion_active_on_accept'] = admission.get('motion_active') if isinstance(admission,dict) else None
                publish('settling_after_new_move')
                self._settle(context,goal)
                publish('capturing_fresh_retry_exposures')
                self._station_retry_context = copy.deepcopy(record)
                result = self._capture_line_station(context,index,'stationary_main')
            except Exception as retry_error:
                record['error'] = str(retry_error)
                record['retry_gate'] = copy.deepcopy(getattr(retry_error,'diagnostic',None))
                publish('failed')
                raise
            publish('complete')
            return result
        finally:
            self._station_retry_context = None

    def _line_start_motion(self,context):
        length = round(context['settings']['sweep_length_mm'])
        if context['settings']['scan_method'] == 'precision':
            for index in range(length+1):
                self._check_cancel(context)
                self._line_fraction = index/max(length,1)
                if index:
                    goal = self._line_goal(context,index)
                    self._command('move_pose',{'pose':goal.tolist()},context)
                    self._settle(context,goal)
                self._capture_precision_with_retry(context,index)
            return dict(stationary_protocol_complete=True,stations_complete=length+1,station_total=length+1)
        self._main_scan = True
        self._capture_metadata = dict(scan_protocol='fast_continuous',acquisition_phase='moving_main')
        self._command('start_path',{'plan':context['data']['plan']['plan']},context)
        self._start_collector(context)
        return dict(motion_started=True)

    def _line_finalize(self,context):
        self._stop_collector()
        length = round(context['settings']['sweep_length_mm'])
        if context['settings']['scan_method'] == 'fast':
            self._settle(context,self._line_goal(context,length))
            self._capture_line_station(context,length,'end_anchor')
        state = self.archive.state(include_points=False)
        precise = context['settings']['scan_method'] == 'precision'
        moving_count = 0
        moving_down = []
        start = np.asarray(context['data']['baseline']['line_start_base_from_camera'])
        handeye = np.asarray(context['data']['baseline']['handeye']['tcp_from_camera'])
        with (self.archive.session_directory/'frames.jsonl').open() as handle:
            for row in handle:
                record = json.loads(row)
                if (record.get('kind') == 'frame' and record.get('status') == 'accepted'
                        and (record.get('scan_context') or {}).get('acquisition_phase') == 'moving_main'):
                    camera = np.asarray(record['registration']['base_from_tcp'])@handeye
                    moving_down.append(float(start[2,3]-camera[2,3])*1000.)
                    moving_count += 1
        span = max(moving_down)-min(moving_down) if moving_down else 0.
        complete = state.get('stations_complete') == length+1 if precise else moving_count >= 3 and span >= length-1.
        if not precise and not complete:
            self.archive.record_gap('Fast line lacks accepted moving captures across its full axial span',
                start_monotonic_s=context['started_monotonic'],end_monotonic_s=time.monotonic(),path_fraction=1.,
                evidence=dict(accepted_moving_frames=moving_count,accepted_axial_span_mm=span))
        evidence = dict(measured_path_complete=True,line_path_complete=True,path_fraction=1.,
            continuous_quality_coverage=complete,coverage_gaps_resolved=True,
            stationary_stations_complete=complete if precise else None,
            sampled_line_bins=state.get('line_samples_observed'),station_total=length+1,
            accepted_moving_frames=moving_count,accepted_moving_axial_span_mm=span,
            workflow_elapsed_before_finish_s=time.monotonic()-context['started_monotonic'],
            maximum_measured_exposure_path_gap_m=self._max_path_gap,
            scan_profile='vertical_down',scan_method=context['settings']['scan_method'])
        saved = self.archive.finish(status='complete' if complete else 'interrupted',completion_evidence=evidence)
        self._archive_started = False
        saved['completion_qualified'] = complete
        return self._display_saved(saved)
