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

    def __init__(self, gui, send):
        self.send = send
        self._stream_enabled = None
        self._handles = []
        self._signature = None
        self._profile_cache_key = self._profile_cache = self._profile_signature = None
        self.folder = gui.add_folder('Weld seam finding')
        with self.folder:
            gui.add_markdown('Start at **Weld Finder** with a clean laser line crossing the seam. '
                'The service follows the valley or discontinuity in **both directions**, perpendicular '
                'to the laser line, and keeps the seam near the center at about **150 mm standoff**. '
                'A missing or poor laser return does not establish an end of the seam.')
            self.stream = gui.add_checkbox('Stream camera for weld finding', initial_value=False)
            self.stream.on_update(self._stream_changed)
            self.minimum = gui.add_number('Minimum total scan distance (mm)', 25., min=1., max=100., step=1.)
            self.start_button = gui.add_button('Find weld seam')
            self.start_button.on_click(self._start)
            self.cancel_button = gui.add_button('Cancel weld operation and hold')
            self.cancel_button.on_click(lambda _: self.send('weld_cancel'))
            self.status_md = gui.add_markdown('Enable the healthy arm, then find the seam from Weld Finder.')
            self.quality_md = gui.add_markdown('No weld observations yet.')
            self.show_seam = gui.add_checkbox('Show weld seam in green', initial_value=True)
            self.show_lines = gui.add_checkbox('Show full laser profiles in red', initial_value=True)
            self.show_poses = gui.add_checkbox('Show torch pose axes', initial_value=False)
            self.geometry_md = gui.add_markdown('No captured weld path.')
            self.pose_md = gui.add_markdown('')
            gui.add_markdown('Torch poses lie midway between the fitted sides and point toward the weld. '
                'Replay follows the measured poses through the robot service. Unconfirmed endpoints are reported '
                'when only a measured section is available. Replay does not turn on a welding arc.')
            self.replay_button = gui.add_button('Replay weld path')
            self.replay_button.on_click(lambda _: self.send('weld_replay'))
            self.export_button = gui.add_button('Export weld path')
            self.export_button.on_click(lambda _: self.send('weld_export'))
            self.export_md = gui.add_markdown('')
            self.clear_button = gui.add_button('Clear weld path')
            self.clear_button.on_click(lambda _: self.send('weld_clear'))

    def _stream_changed(self, event):
        if event.client_id is not None:
            self.send('camera_stream', enabled=self.stream.value)

    def _start(self, _event=None):
        self.send('weld_start', min_distance_mm=self.minimum.value, start_pose='Weld Finder')

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

    def _render_geometry(self, weld, scene):
        geometry = weld.get('geometry') or {}
        if geometry.get('units') != 'm' or geometry.get('coordinate_frame') != 'robot_base':
            self._clear_geometry()
            self.geometry_md.content = 'No registered weld geometry.'
            self.pose_md.content = ''
            return
        points = self._array(geometry.get('seam_points_base_m', []), (3,), self.MAX_POINTS)
        has_profiles = 'profile_points_base_m' in geometry
        profiles, profile_signature = self._profiles(geometry, weld) if has_profiles else (None, None)
        lines = None if has_profiles else self._array(geometry.get('scan_lines_base_m', []), (2, 3), self.MAX_LINES)
        poses = self._array(geometry.get('torch_poses_base_m', []), (4, 4), self.MAX_POINTS)
        if poses is not None and len(poses):
            rotations = poses[:, :3, :3]
            if (not np.allclose(poses[:, 3], [0., 0., 0., 1.], rtol=0, atol=1e-6)
                    or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-5)
                    or not np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-5)):
                poses = None
        signature = (tuple(array.tobytes() if array is not None else None for array in (points, lines, poses)),
            has_profiles, profile_signature, self.show_seam.value, self.show_lines.value, self.show_poses.value)
        if signature != self._signature:
            self._clear_geometry()
            if self.show_seam.value and points is not None and len(points):
                self._handles.append(scene.add_point_cloud('/weld/seam', points=points,
                    colors=(30, 235, 80), point_size=.002, point_shape='circle',
                    point_shading='flat', precision='float32'))
                if len(points) > 1:
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
            'robot base · retained after the camera stops')
        pose_text = []
        if points is not None and len(points):
            seam = points[-1]*1000.
            pose_text.append(f'Path endpoint seam (mm): **{seam[0]:.2f}, {seam[1]:.2f}, {seam[2]:.2f}**')
        if poses is not None and len(poses):
            position = poses[-1, :3, 3]*1000.
            aim = poses[-1, :3, 2]
            pose_text.append(f'Torch reference position (mm): **{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}**')
            pose_text.append(f'Torch +Z toward weld: **{aim[0]:.3f}, {aim[1]:.3f}, {aim[2]:.3f}**')
        self.pose_md.content = '\n\n'.join(pose_text)

    def render(self, state, scene):
        camera, weld = state.get('camera') or {}, state.get('weld') or {}
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
        self.minimum.disabled = active
        self.stream.disabled = active or other_active
        self.cancel_button.disabled = False
        self.clear_button.disabled = active
        self.export_button.disabled = active or not weld.get('point_count', 0)
        phase = weld.get('status', 'idle')
        reason = weld.get('error') or (weld.get('blocker') if not active else None)
        self.status_md.content = (f'**{phase}** · {weld.get("message", "Waiting for weld finding readiness")}'
            + (f'\n\n**Reason:** {reason}' if reason else ''))
        settings = weld.get('settings') or {}
        distance = weld.get('distance_mm')
        details = [f'Start: **Weld Finder** · target standoff: **{settings.get("standoff_mm", 150.):g} mm**',
            f'Minimum total scan: **{settings.get("min_distance_mm", weld.get("min_distance_mm", 25.)):g} mm**']
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
        paths = weld.get('export_paths') or {}
        if isinstance(paths, dict):
            paths = paths.values()
        self.export_md.content = '\n\n'.join(f'Saved weld path: `{path}`' for path in paths if path)
        self._render_geometry(weld, scene)
