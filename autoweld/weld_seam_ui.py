"""Render registered weld geometry and submit JSON service requests only."""

import hashlib

import numpy as np


class WeldSeamUI:
    """The service owns discovery, measured poses, persistence and replay."""

    ACTIVE_CALIBRATION = frozenset(('starting', 'moving', 'settling', 'capturing', 'returning',
        'checking', 'solving', 'hunting', 'hunt_moving', 'hunt_settling', 'running', 'capturing_manual'))
    MAX_POINTS = 2000
    MAX_LINES = 4000
    MAX_PROFILE_POINTS = 500000
    MAX_DISPLAY_POSES = 25
    MAX_PREDICTED_POINTS = 10000
    JOINT_KINDS = {'Coped tube': 'coped_tube', 'Flat lap': 'flat_lap', 'Flat butt': 'flat_butt'}

    def __init__(self, gui, send):
        self.send = send
        self._stream_enabled = None
        self._handles = []
        self._signature = None
        self._profile_cache_key = self._profile_cache = self._profile_signature = None
        self._geometry_cache_key = self._geometry_cache = None
        self._start_pose_names = None
        self.folder = gui.add_folder('Weld seam finding')
        with self.folder:
            gui.add_markdown('Select a saved start pose with a clean laser line crossing the seam. '
                'The service follows the valley or discontinuity in **both directions**, perpendicular '
                'to the laser line, and keeps the seam near the center at about **150 mm standoff**. '
                'A missing or poor laser return does not establish an end of the seam.')
            self.stream = gui.add_checkbox('Stream camera for weld finding', initial_value=False)
            self.stream.on_update(self._stream_changed)
            self.start_pose = gui.add_dropdown('Weld start pose', options=['Weld Finder'], initial_value='Weld Finder')
            self.reflective = gui.add_checkbox('Reflective surface (low gain and glare rejection)', initial_value=True)
            with gui.add_folder('Seam search region (optional)'):
                self.use_roi = gui.add_checkbox('Restrict seam search region', initial_value=False)
                self.roi_left = gui.add_number('Seam ROI left (%)', 0., min=0., max=100., step=.1)
                self.roi_top = gui.add_number('Seam ROI top (%)', 0., min=0., max=100., step=.1)
                self.roi_right = gui.add_number('Seam ROI right (%)', 100., min=0., max=100., step=.1)
                self.roi_bottom = gui.add_number('Seam ROI bottom (%)', 100., min=0., max=100., step=.1)
                gui.add_markdown('Optional search hint for the starting image. The full measured laser profile '
                    'remains visible in red; the search region does not create seam measurements.')
                self.use_roi.on_update(lambda _: self._roi_changed())
                self._roi_changed()
            self.joint_kind = gui.add_dropdown('Joint type', options=list(self.JOINT_KINDS), initial_value='Coped tube')
            self.joint_kind.on_update(self._joint_changed)
            self.coping_axis = gui.add_dropdown('Coped branch in starting laser profile',
                options=['Left side', 'Right side'], initial_value='Left side')
            self.branch_hint_md = gui.add_markdown('Left/right refers to the starting native laser profile and image. '
                'The selected branch and main-tube labels are frozen at the initial fit; '
                'equal tube diameters alone cannot identify the coped branch.')
            self.branch_diameter = gui.add_number('Branch tube diameter (mm)', 25.4, min=2., max=500., step=.1)
            self.main_diameter = gui.add_number('Main tube diameter (mm)', 25.4, min=2., max=500., step=.1)
            self.joint_angle = gui.add_number('Tube joint angle (deg)', 90., min=10., max=170., step=1.)
            self.target_arc = gui.add_number('Requested cope arc (deg)', 180., min=10., max=180., step=5.)
            self.max_profile_gap = gui.add_number('Maximum missing profile gap (mm)', 5., min=0., max=10., step=.1)
            self.gap_help_md = gui.add_markdown('Missing returns between clean tube flanks may support an '
                '**inferred gap center**, with a reported uncertainty. This bound does not loosen surface-fit '
                'accuracy or make an unobserved gap into a measured seam or replayable path.')
            self.plate_thickness = gui.add_number('Plate thickness (mm)', 2., min=.1, max=100., step=.1)
            self.minimum = gui.add_number('Minimum local scan distance (mm)', 25., min=1., max=100., step=1.)
            self.joint_help_md = gui.add_markdown('')
            self.start_button = gui.add_button('Find weld seam')
            self.start_button.on_click(self._start)
            self.cancel_button = gui.add_button('Cancel weld operation and hold')
            self.cancel_button.on_click(lambda _: self.send('weld_cancel'))
            self.status_md = gui.add_markdown('Enable the healthy arm, then find the seam from the selected saved pose.')
            self.quality_md = gui.add_markdown('No weld observations yet.')
            self.model_md = gui.add_markdown('No fitted joint model yet.')
            self.coverage_md = gui.add_markdown('')
            self.show_seam = gui.add_checkbox('Show weld seam in green', initial_value=True)
            self.show_lines = gui.add_checkbox('Show full laser profiles in red', initial_value=True)
            self.show_poses = gui.add_checkbox('Show inward sensing pose axes', initial_value=False)
            self.show_predicted = gui.add_checkbox('Show predicted cope curve in cyan (unmeasured)', initial_value=False)
            self.geometry_md = gui.add_markdown('No captured weld path.')
            self.pose_md = gui.add_markdown('')
            gui.add_markdown('The **150 mm sensing pose reference** points toward the measured weld. '
                'A physical welding torch requires a calibrated tool offset. '
                'Replay follows the measured poses through the robot service. Unconfirmed endpoints are reported '
                'when only a measured section is available. Replay does not turn on a welding arc.')
            self.replay_button = gui.add_button('Replay weld path')
            self.replay_button.on_click(lambda _: self.send('weld_replay'))
            self.export_button = gui.add_button('Export weld path')
            self.export_button.on_click(lambda _: self.send('weld_export'))
            self.export_md = gui.add_markdown('')
            self.clear_button = gui.add_button('Clear weld path')
            self.clear_button.on_click(lambda _: self.send('weld_clear'))
        self._joint_changed()

    def _joint_changed(self, _event=None):
        coping = self.JOINT_KINDS.get(self.joint_kind.value) == 'coped_tube'
        self.coping_axis.visible = self.branch_hint_md.visible = coping
        for handle in (self.branch_diameter, self.main_diameter, self.joint_angle, self.target_arc,
                       self.max_profile_gap, self.gap_help_md):
            handle.visible = coping
        self.plate_thickness.visible = self.minimum.visible = not coping
        self.joint_help_md.content = ('Fit the two tube surfaces from measured profiles, then inspect the requested '
            'cope arc in both directions. Angular coverage counts fresh measured stations only. '
            'The 25 mm flat-joint search minimum does not establish a closed cope or its endpoints.' if coping else
            'Follow a local seam in both directions. The minimum is measured search span, not physical weld length. '
            'The service must confirm each visible end with good laser lines; no tube orbit is assumed.')

    def _stream_changed(self, event):
        if event.client_id is not None:
            self.send('camera_stream', enabled=self.stream.value)

    def _roi_changed(self):
        for handle in (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom):
            handle.disabled = not self.use_roi.value

    def _start(self, _event=None):
        kind = self.JOINT_KINDS.get(self.joint_kind.value)
        parameters = dict(min_distance_mm=25. if kind == 'coped_tube' else self.minimum.value,
            start_pose=self.start_pose.value, joint_kind=kind, reflective_surface=self.reflective.value)
        if self.reflective.value and self.use_roi.value:
            roi = [handle.value for handle in (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)]
            if (any(type(value) not in (int, float) for value in roi) or not np.isfinite(roi).all()
                    or not all(0 <= value <= 100 for value in roi) or not roi[0] < roi[2] or not roi[1] < roi[3]):
                self.status_md.content = 'Enter a seam search region with left < right and top < bottom within 0–100%.'
                return
            parameters['reflective_roi'] = [value/100. for value in roi]
        fields = ((('branch_diameter_mm', self.branch_diameter, 2., 500.),
            ('main_diameter_mm', self.main_diameter, 2., 500.),
            ('joint_angle_deg', self.joint_angle, 10., 170.),
            ('target_arc_deg', self.target_arc, 10., 180.),
            ('max_profile_gap_mm', self.max_profile_gap, 0., 10.)) if kind == 'coped_tube' else
            (('plate_thickness_mm', self.plate_thickness, .1, 100.),
             ('min_distance_mm', self.minimum, 1., 100.)))
        if kind is None:
            self.status_md.content = 'Choose a supported joint type before finding the seam.'
            return
        if kind == 'coped_tube':
            if self.coping_axis.value not in ('Left side', 'Right side'):
                self.status_md.content = 'Choose the coped branch in the starting native laser profile.'
                return
            parameters['coping_axis'] = 0 if self.coping_axis.value == 'Left side' else 1
        for key, widget, minimum, maximum in fields:
            value = widget.value
            if type(value) not in (int, float) or not np.isfinite(value) or not minimum <= value <= maximum:
                self.status_md.content = f'**Invalid {widget.label}:** enter a finite value from {minimum:g} to {maximum:g}.'
                return
            parameters[key] = value
        self.send('weld_start', **parameters)

    @staticmethod
    def _array(value, tail, limit, *, dtype=float):
        try:
            array = np.asarray(value, dtype=dtype)
            if (array.ndim != len(tail)+1 or array.shape[1:] != tail
                    or len(array) > limit or not np.isfinite(array).all()):
                return None
            return array
        except (ValueError, TypeError):
            return None

    def _profiles(self, geometry, weld):
        """Convert a changed native preview once; keep no second byte copy."""
        raw = geometry.get('profile_points_base_m')
        revision = geometry.get('profile_revision', geometry.get('revision', weld.get('revision')))
        length = len(raw) if isinstance(raw, (list, tuple, np.ndarray)) else None
        key = (weld.get('session_id'), revision, length) if revision is not None else None
        if key is not None and key == self._profile_cache_key:
            return self._profile_cache, self._profile_signature
        profiles = (None if length is not None and length > self.MAX_PROFILE_POINTS else
            self._array(raw, (3,), self.MAX_PROFILE_POINTS, dtype=np.float32))
        signature = None
        if profiles is not None:
            profiles = np.ascontiguousarray(profiles)
            fingerprint = hashlib.blake2b(memoryview(profiles).cast('B'), digest_size=16).digest() if profiles.size else b''
            signature = (profiles.shape, fingerprint)
        self._profile_cache_key, self._profile_cache, self._profile_signature = key, profiles, signature
        return profiles, signature

    def _clear_geometry(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._signature = None

    def _published_geometry(self, weld):
        """Reuse an immutable full publication only for its exact session/revision."""
        geometry = weld.get('geometry') or {}
        session, revision = weld.get('session_id'), geometry.get('revision')
        key = ((session, revision) if isinstance(session, str) and session
            and type(revision) is int and revision >= 0 else None)
        registered = geometry.get('units') == 'm' and geometry.get('coordinate_frame') == 'robot_base'
        if geometry.get('unchanged') is True:
            if registered and key is not None and key == self._geometry_cache_key:
                return self._geometry_cache
            self._geometry_cache_key = self._geometry_cache = None
            self._profile_cache_key = self._profile_cache = self._profile_signature = None
            return {}
        if not registered:
            self._geometry_cache_key = self._geometry_cache = None
            self._profile_cache_key = self._profile_cache = self._profile_signature = None
            return geometry
        if key != self._geometry_cache_key or 'profile_points_base_m' not in geometry:
            self._profile_cache_key = self._profile_cache = self._profile_signature = None
        self._geometry_cache_key = key
        self._geometry_cache = geometry if key is not None else None
        return geometry

    def _render_geometry(self, weld, scene):
        geometry = self._published_geometry(weld)
        if geometry.get('units') != 'm' or geometry.get('coordinate_frame') != 'robot_base':
            self._clear_geometry()
            self.geometry_md.content = 'No registered weld geometry.'
            self.pose_md.content = ''
            return
        points = self._array(geometry.get('seam_points_base_m', []), (3,), self.MAX_POINTS)
        coping = (weld.get('settings') or {}).get('joint_kind') == 'coped_tube'
        explicit_segments = 'measured_seam_segments_base_m' in geometry
        seam_segments = self._array(geometry.get('measured_seam_segments_base_m', []), (2, 3), self.MAX_LINES)
        has_profiles = 'profile_points_base_m' in geometry
        profiles, profile_signature = self._profiles(geometry, weld) if has_profiles else (None, None)
        lines = None if has_profiles else self._array(geometry.get('scan_lines_base_m', []), (2, 3), self.MAX_LINES)
        predicted = self._array(geometry.get('predicted_seam_points_base_m', []), (3,), self.MAX_PREDICTED_POINTS)
        poses = self._array(geometry.get('torch_poses_base_m', []), (4, 4), self.MAX_POINTS)
        if poses is not None and len(poses):
            rotations = poses[:, :3, :3]
            if (not np.allclose(poses[:, 3], [0., 0., 0., 1.], rtol=0, atol=1e-6)
                    or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-5)
                    or not np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-5)):
                poses = None
        signature = (tuple(array.tobytes() if array is not None else None for array in (points, lines, poses)),
            has_profiles, profile_signature, predicted.tobytes() if predicted is not None else None,
            coping, explicit_segments, seam_segments.tobytes() if seam_segments is not None else None,
            self.show_seam.value, self.show_lines.value, self.show_poses.value, self.show_predicted.value)
        if signature != self._signature:
            self._clear_geometry()
            if self.show_seam.value and points is not None and len(points):
                self._handles.append(scene.add_point_cloud('/weld/seam', points=points,
                    colors=(30, 235, 80), point_size=.002, point_shape='circle',
                    point_shading='flat', precision='float32'))
                if explicit_segments and seam_segments is not None and len(seam_segments):
                    self._handles.append(scene.add_line_segments('/weld/path',
                        points=seam_segments, colors=(30, 235, 80), thickness=2., thickness_units='screen'))
                elif not explicit_segments and not coping and len(points) > 1:
                    self._handles.append(scene.add_line_segments('/weld/path',
                        points=np.stack((points[:-1], points[1:]), axis=1), colors=(30, 235, 80),
                        thickness=2., thickness_units='screen'))
            if self.show_lines.value and profiles is not None and len(profiles):
                self._handles.append(scene.add_point_cloud('/weld/scan-profiles', points=profiles,
                    colors=(245, 55, 55), point_size=.0005, point_shape='circle',
                    point_shading='flat', precision='float32'))
            elif self.show_lines.value and not has_profiles and lines is not None and len(lines):
                self._handles.append(scene.add_line_segments('/weld/scan-lines', points=lines,
                    colors=(245, 55, 55), thickness=1., thickness_units='screen'))
            if self.show_predicted.value and predicted is not None and len(predicted):
                self._handles.append(scene.add_point_cloud('/weld/predicted-cope', points=predicted,
                    colors=(50, 210, 240), point_size=.0008, point_shape='circle',
                    point_shading='flat', precision='float32'))
            if self.show_poses.value and poses is not None and len(poses):
                indices = np.unique(np.linspace(0, len(poses)-1,
                    min(len(poses), self.MAX_DISPLAY_POSES), dtype=int))
                displayed = poses[indices]
                origins = displayed[:, :3, 3]
                for axis, color in enumerate(((235, 75, 75), (75, 235, 75), (75, 140, 255))):
                    ends = origins + .010*displayed[:, :3, axis]
                    self._handles.append(scene.add_line_segments(f'/weld/torch-axis-{axis}',
                        points=np.stack((origins, ends), axis=1), colors=color,
                        thickness=2., thickness_units='screen'))
            self._signature = signature
        count = len(points) if points is not None else 0
        self.geometry_md.content = (f'**{weld.get("point_count", count):,}** measured seam points · '
            f'green seam / red {"full laser profiles" if has_profiles else "scan lines"} · '
            'robot base · retained after the camera stops'
            + (' · cyan is an unmeasured prediction' if self.show_predicted.value and predicted is not None and len(predicted) else ''))
        pose_text = []
        if points is not None and len(points):
            seam = points[-1]*1000.
            label = 'Observed cope point' if coping else 'Path endpoint seam'
            pose_text.append(f'{label} (mm): **{seam[0]:.2f}, {seam[1]:.2f}, {seam[2]:.2f}**')
        if poses is not None and len(poses):
            position = poses[-1, :3, 3]*1000.
            aim = poses[-1, :3, 2]
            label = '150 mm sensing reference position' if coping else 'Torch reference position'
            pose_text.append(f'{label} (mm): **{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}**')
            label = 'Sensing +Z inward toward the cope' if coping else 'Torch +Z toward weld'
            pose_text.append(f'{label}: **{aim[0]:.3f}, {aim[1]:.3f}, {aim[2]:.3f}**')
        self.pose_md.content = '\n\n'.join(pose_text)

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and np.isfinite(value)

    def _render_joint_status(self, weld):
        fitted = weld.get('joint_model') or {}
        quality = fitted.get('quality') or {}
        if not isinstance(quality, dict):
            quality = {}
        model = fitted.get('model') or {}
        status = fitted.get('status')
        rows = [f'Joint model: **{str(status).replace("_", " ")}**'] if status else ['No fitted joint model yet.']
        labels = {'rms_mm': 'Fit RMS', 'residual_rms_mm': 'Profile residual RMS',
            'branch_rms_mm': 'Branch surface RMS', 'main_rms_mm': 'Main surface RMS'}
        for key in quality:
            if isinstance(key, str) and key.endswith('_rms_mm') and key not in labels:
                labels[key] = key[:-7].replace('_', ' ').capitalize()+' RMS'
        for key, label in labels.items():
            value = quality.get(key)
            if self._finite(value):
                rows.append(f'{label}: **{value:g} mm**')
        reason = fitted.get('reason') or quality.get('reason') or quality.get('message')
        if reason:
            rows.append(str(reason))
        if quality.get('branch_role_ambiguous') is True or quality.get('role_ambiguous') is True:
            rows.append('Branch and main tube role assignment is ambiguous.')
        if isinstance(model, dict) and model.get('kind'):
            rows.append(f'Model: {str(model["kind"]).replace("_", " ")}')
        self.model_md.content = '\n\n'.join(rows)
        coverage = weld.get('coping_coverage') or {}
        rows = []
        if isinstance(coverage, dict) and coverage:
            for key, label in (('requested_arc_deg', 'Requested cope arc'), ('planned_arc_deg', 'Reachable planned arc'),
                    ('observed_arc_deg', 'Actually observed arc'), ('positive_observed_deg', 'Observed positive arc'),
                    ('negative_observed_deg', 'Observed negative arc')):
                value = coverage.get(key)
                if self._finite(value):
                    rows.append(f'{label}: **{value:g}°**')
            count = coverage.get('measured_station_count')
            if type(count) is int and count >= 0:
                rows.append(f'Measured angular stations: **{count:,}**')
            if coverage.get('coverage_complete') is False:
                rows.append('Requested cope coverage is incomplete; predicted geometry does not count as measured.')
            elif coverage.get('coverage_complete') is True:
                rows.append('Requested cope arc is fully observed.')
            boundaries = coverage.get('boundaries') or {}
            if isinstance(boundaries, dict):
                for direction, boundary in boundaries.items():
                    if isinstance(boundary, dict):
                        text = boundary.get('reason') or boundary.get('status') or 'unconfirmed'
                        angle = boundary.get('angle_deg')
                        if self._finite(angle):
                            text = f'{text} at {angle:g}°'
                    else:
                        text = str(boundary)
                    rows.append(f'{str(direction).capitalize()} coverage limit: **{text}**')
            elif isinstance(boundaries, (list, tuple)):
                rows.extend(str(boundary.get('reason', boundary)) if isinstance(boundary, dict) else str(boundary)
                            for boundary in boundaries)
            preflight = coverage.get('preflight') or weld.get('coping_preflight') or {}
            if isinstance(preflight, dict):
                if preflight.get('status'):
                    rows.append(f'Arc preflight: **{str(preflight["status"]).replace("_", " ")}**')
                if preflight.get('sensing_view_family'):
                    rows.append(f'Camera view: **{str(preflight["sensing_view_family"]).replace("_", " ")}**')
                if self._finite(preflight.get('base_view_tilt_deg')):
                    rows.append(f'Camera viewing tilt: **{preflight["base_view_tilt_deg"]:g}°**')
                for key, label in (('positive_bound_deg', 'Positive reachable bound'),
                        ('negative_bound_deg', 'Negative reachable bound')):
                    value = preflight.get(key)
                    if self._finite(value):
                        rows.append(f'{label}: **{value:g}°**')
                if preflight.get('reason'):
                    rows.append(str(preflight['reason']))
            gaps = coverage.get('gaps') or []
            if isinstance(gaps, (list, tuple)) and gaps:
                rows.append(f'Unobserved angular gaps: **{len(gaps):,}**')
                for gap in gaps[:8]:
                    if isinstance(gap, dict):
                        start, end = gap.get('start_deg'), gap.get('end_deg')
                        text = f'{start:g}° to {end:g}°' if self._finite(start) and self._finite(end) else 'unobserved'
                        if gap.get('reason'):
                            text += f': {gap["reason"]}'
                        rows.append(text)
                    else:
                        rows.append(str(gap))
        self.coverage_md.content = '\n\n'.join(rows)

    def render(self, state, scene):
        camera, weld = state.get('camera') or {}, state.get('weld') or {}
        saved = state.get('saved_poses')
        if isinstance(saved, dict):
            names = tuple(name for name in saved if isinstance(name, str) and name.strip() and len(name) <= 200)
            if names != self._start_pose_names:
                selected = self.start_pose.value
                first = self._start_pose_names is None
                self._start_pose_names = names
                self.start_pose.options = names or ('Weld Finder',)
                if first or selected not in names:
                    self.start_pose.value = ('Weld Finder 2' if 'Weld Finder 2' in names else
                        'Weld Finder' if 'Weld Finder' in names else names[0] if names else 'Weld Finder')
        enabled = bool(camera.get('enabled'))
        if enabled != self._stream_enabled:
            self.stream.value = enabled
            self._stream_enabled = enabled
        active = bool(weld.get('busy'))
        calibration, laser = state.get('calibration') or {}, state.get('laser') or {}
        other_active = bool((state.get('scan') or {}).get('busy') or laser.get('busy')
            or laser.get('status') in self.ACTIVE_CALIBRATION
            or calibration.get('status') in self.ACTIVE_CALIBRATION)
        shutdown = state.get('shutdown') or {}
        blocked = bool(state.get('closed', True) or state.get('disable_pending') or state.get('motion_active')
            or state.get('fault') or state.get('motion_fault') or state.get('mode_transition')
            or state.get('cooldown_active') or shutdown.get('requested') or shutdown.get('active')
            or camera.get('starting') or camera.get('stopping') or camera.get('configuring') or other_active)
        self.start_button.disabled = active or blocked or not weld.get('start_ready', False)
        self.replay_button.disabled = active or blocked or not weld.get('replay_ready', False)
        for handle in (self.start_pose, self.reflective, self.minimum, self.joint_kind, self.coping_axis, self.branch_diameter, self.main_diameter,
                self.joint_angle, self.target_arc, self.max_profile_gap, self.plate_thickness):
            handle.disabled = active or other_active
        self.use_roi.disabled = active or other_active or not self.reflective.value
        for handle in (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom):
            handle.disabled = active or other_active or not self.reflective.value or not self.use_roi.value
        self.stream.disabled = active or other_active
        self.cancel_button.disabled = False
        self.clear_button.disabled = active
        self.export_button.disabled = active or not weld.get('point_count', 0)
        phase = weld.get('status', 'idle')
        reason = weld.get('error') or (weld.get('blocker') if not active else None)
        message = weld.get('message', 'Waiting for weld finding readiness')
        model, coverage = weld.get('joint_model') or {}, weld.get('coping_coverage') or {}
        provisional, measured = coverage.get('provisional_seed_scout_count'), coverage.get('measured_station_count')
        if ((weld.get('settings') or {}).get('joint_kind', weld.get('joint_kind')) == 'coped_tube'
                and model.get('status') in ('unobservable', 'ambiguous', 'role_ambiguous', 'error')
                and type(provisional) is int and provisional > 0 and type(measured) is int and measured == 0
                and not state.get('fault') and not state.get('motion_fault')):
            message = 'Local seam locator detected; coped-tube tracing is not qualified.'
            fit_quality = model.get('quality')
            fit_reason = fit_quality.get('reason') if isinstance(fit_quality, dict) else None
            reason = (fit_reason.strip()[:280] + ('…' if len(fit_reason.strip()) > 280 else '')
                      if isinstance(fit_reason, str) and fit_reason.strip() else None)
        self.status_md.content = (f'**{phase}** · {message}'
            + (f'\n\n**Reason:** {reason}' if reason else ''))
        settings = weld.get('settings') or {}
        distance = weld.get('distance_mm')
        start_pose = settings.get('start_pose', self.start_pose.value) if weld.get('session_id') else self.start_pose.value
        details = [f'Start: **{start_pose}** · sensing reference standoff: **{settings.get("standoff_mm", 150.):g} mm**']
        if settings.get('reflective_surface'):
            details.append('Reflective surface: native ridge continuity and competing-return checks; '
                'uncertain reflections remain unresolved.')
            if settings.get('joint_kind') == 'coped_tube':
                details.append(f'Maximum missing profile gap: **{settings.get("max_profile_gap_mm", 5.):g} mm**. '
                    'Gap tolerance does not relax tube-fit error limits.')
        gap = weld.get('fitup_gap') or {}
        if gap.get('status') == 'inferred_gap':
            separation, uncertainty = gap.get('observed_return_separation_mm'), gap.get('uncertainty_mm')
            if (type(separation) in (int, float) and type(uncertainty) in (int, float)
                    and np.isfinite([separation, uncertainty]).all()):
                details.append(f'Missing-return span: **{separation:g} mm** · inferred center uncertainty: '
                    f'**{uncertainty:g} mm**. This is not a measured seam or replay station.')
        elif gap.get('status') == 'ambiguous':
            details.append('Competing optical returns: the primary stripe and gap center remain unresolved.')
        if settings.get('joint_kind') == 'coped_tube':
            details.append('Coped tube: angular coverage is measured around the joint; no physical seam end is inferred from the requested arc.')
        else:
            details.append(f'Minimum total local scan: **{settings.get("min_distance_mm", weld.get("min_distance_mm", 25.)):g} mm**')
        if isinstance(distance, (int, float)) and np.isfinite(distance):
            details.append(f'Measured weld length: **{distance:g} mm**')
        search = weld.get('search_distance_mm')
        if isinstance(search, (int, float)) and np.isfinite(search):
            details.append(f'Explored scan span: **{search:g} mm**')
        if weld.get('replay_scope') == 'measured_section_only':
            details.append('Replay covers the measured section only; seam endpoints remain unconfirmed.')
        endpoints = weld.get('endpoints') or {}
        if isinstance(endpoints, dict):
            for key, endpoint in endpoints.items():
                label = 'Positive direction' if key in ('positive', '+', '1') else (
                    'Negative direction' if key in ('negative', '-', '-1') else str(key))
                if isinstance(endpoint, dict):
                    text = endpoint.get('reason') or endpoint.get('status') or 'searching'
                    if endpoint.get('confirmed'):
                        text = f'confirmed · {text}'
                else:
                    text = 'searching' if endpoint is None else str(endpoint)
                details.append(f'{label}: **{text}**')
        quality = weld.get('quality') or {}
        if quality.get('message'):
            details.append(str(quality['message']))
        if weld.get('replay_blocker'):
            details.append(f'Replay: {weld["replay_blocker"]}')
        self.quality_md.content = '\n\n'.join(details)
        self._render_joint_status(weld)
        paths = weld.get('export_paths') or {}
        if isinstance(paths, dict):
            paths = paths.values()
        self.export_md.content = '\n\n'.join(f'Saved weld path: `{path}`' for path in paths if path)
        self._render_geometry(weld, scene)
