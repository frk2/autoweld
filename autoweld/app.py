"""Viser frontend: render JSON snapshots and emit JSON operator commands.

No hardware, camera acquisition, image processing, IK, or robot control lives here.
A single App is shared by every browser client; connecting a browser only sets its
viewpoint. The application service owns all models and devices.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

# Asset path helpers retained for consumers; no kinematics model is loaded.
from autoweld.models import URDF, tool_model

__all__ = ['App', 'LaserPlaneCalibrationUI', 'LaserScanningUI', 'URDF', 'tool_model', 'main', 'cli']


class LaserPlaneCalibrationUI:
    """Laser panel and preview only; every operation is a service JSON request."""

    ACTIVE = {'starting', 'moving', 'settling', 'capturing', 'returning', 'checking',
              'solving', 'hunting', 'hunt_moving', 'hunt_settling', 'running', 'capturing_manual'}
    MAX_PREVIEW_POINTS = 20000

    def __init__(self, gui, send):
        self.send = send
        self._stream_enabled = None
        self._settings_seen = {}
        self._control_readback = {}
        self._control_limits = {}
        self._cloud_handle = None
        self._cloud_signature = None
        self.folder = gui.add_folder('Laser plane calibration')
        with self.folder:
            gui.add_markdown('The laser is assumed on. Auto calibration moves through the service to the saved '
                '**Calibration Pose**, then captures the target at multiple poses. The camera preview above is shared.')
            self.stream = gui.add_checkbox('Stream camera for laser', initial_value=False)
            self.stream.on_update(self._stream_changed)
            self.status_md = gui.add_markdown('Turn on the camera to inspect the laser stripe.')
            self.auto_button = self._button(gui, 'Auto laser calibration', 'laser_auto',
                lambda: {'go_to_calibration_pose': True})
            self.check_button = self._button(gui, 'Check stripe and target', 'laser_check')
            self.capture_button = self._button(gui, 'Capture laser view', 'laser_capture')
            self.solve_button = self._button(gui, 'Solve laser plane', 'laser_solve')
            self.reset_button = self._button(gui, 'Reset laser captures', 'laser_reset')
            self.cancel_button = self._button(gui, 'Cancel laser calibration and hold', 'laser_cancel')
            self.quality_md = gui.add_markdown('No accepted laser plane.')
            self.exposure_md = gui.add_markdown('')
            self.preview = gui.add_checkbox('Show clean point cloud', initial_value=False)
            self.preview.on_update(self._preview_changed)
            self.cloud_md = gui.add_markdown('Point cloud preview off')
            with gui.add_folder('Laser line finder settings'):
                self.smart_exposure = gui.add_checkbox('Smart stripe exposure', initial_value=True)
                self.exposure_min = gui.add_number('Exposure minimum (µs)', 100., min=1., max=1000000., step=100.)
                self.exposure_max = gui.add_number('Exposure maximum (µs)', 10000., min=1., max=1000000., step=100.)
                self.manual_exposure = gui.add_number('Manual exposure (µs)', 100., min=1., max=1000000., step=100.)
                self.gain = gui.add_number('Camera gain (dB)', 0., step=.1)
                self.manual_exposure.disabled = self.gain.disabled = True
                self.control_readback_md = gui.add_markdown('Waiting for camera exposure and gain limits.')
                self.preview_gamma = gui.add_number('Preview gamma', .5, min=.1, max=3., step=.05)
                self.preview_brightness = gui.add_number('Preview brightness', 1., min=.1, max=4., step=.1)
                gui.add_markdown('Disable smart stripe exposure to hold a manual exposure. '
                    'Gamma and brightness change the displayed image only.')
                self.threshold = gui.add_number('Minimum stripe brightness', 30., min=0., max=255., step=1.)
                self.relative_threshold = gui.add_number('Relative stripe threshold', .4, min=.01, max=1., step=.05)
                self.min_width = gui.add_number('Minimum stripe width (px)', 2, min=1, max=100, step=1)
                self.max_width = gui.add_number('Maximum stripe width (px)', 24, min=1, max=100, step=1)
                self.noise_enabled = gui.add_checkbox('Reject isolated returns', initial_value=True)
                self.min_neighbors = gui.add_number('Minimum neighboring returns', 2, min=1, max=20, step=1)
                self.max_scan_gap = gui.add_number('Maximum neighbor gap (px)', 3, min=1, max=100, step=1)
                self.max_jump = gui.add_number('Maximum stripe jump (px)', 8., min=0., max=1000., step=1.)
                self.min_depth = gui.add_number('Minimum depth (mm)', 1., min=0., step=1.)
                self.max_depth = gui.add_number('Maximum depth (mm)', 2000., min=1., step=10.)
                self.configure_button = gui.add_button('Apply stripe and filter settings')
                self.configure_button.on_click(self._configure)

    def _button(self, gui, label, action, payload=None):
        handle = gui.add_button(label)
        handle.on_click(lambda _: self.send(action, **(payload() if payload else {})))
        return handle

    def _stream_changed(self, event):
        if event.client_id is not None:
            self.send('camera_stream', enabled=self.stream.value)

    def _preview_changed(self, event):
        if event.client_id is not None:
            self.send('laser_preview', enabled=self.preview.value)

    def _configure(self, _event=None):
        controls = {}
        for key, handle in (('exposure_us', self.manual_exposure), ('gain_db', self.gain)):
            if self._control_readback.get(key) is not None and self._control_limits.get(key) is not None:
                controls[key] = handle.value
        self.send('laser_config', controls=controls,
            preview={'gamma': self.preview_gamma.value, 'brightness': self.preview_brightness.value}, detection={
            'threshold': self.threshold.value, 'relative_threshold': self.relative_threshold.value,
            'min_width': int(self.min_width.value), 'max_width': int(self.max_width.value)},
            auto_exposure={'enabled': self.smart_exposure.value, 'min_us': self.exposure_min.value,
                'max_us': self.exposure_max.value}, noise={
                'enabled': self.noise_enabled.value, 'min_neighbors': int(self.min_neighbors.value),
                'max_scan_gap_px': int(self.max_scan_gap.value), 'max_jump_px': self.max_jump.value},
            depth_range_mm=[self.min_depth.value, self.max_depth.value])

    def _sync_settings(self, laser, camera=None):
        status = laser.get('pointcloud_status') or {}
        detection = status.get('detection') or laser.get('detection') or {}
        noise = status.get('noise') or laser.get('noise') or {}
        exposure = status.get('auto_exposure') or laser.get('auto_exposure') or {}
        depth = status.get('depth_range_mm')
        self._control_readback = laser.get('controls') or (camera or {}).get('controls') or {}
        self._control_limits = laser.get('control_limits') or {}
        requested_controls = laser.get('controls_requested') or {}
        exposure_limits = laser.get('exposure_limits') or {}
        requested_exposure = exposure_limits.get('requested_us')
        preview = laser.get('preview') or status.get('preview') or {}
        values = [(self.threshold, detection.get('threshold')), (self.relative_threshold, detection.get('relative_threshold')),
            (self.min_width, detection.get('min_width')), (self.max_width, detection.get('max_width')),
            (self.smart_exposure, exposure.get('enabled')), (self.noise_enabled, noise.get('enabled')),
            (self.min_neighbors, noise.get('min_neighbors')), (self.max_scan_gap, noise.get('max_scan_gap_px')),
            (self.max_jump, noise.get('max_jump_px')), (self.preview, laser.get('preview_enabled')),
            (self.exposure_min, requested_exposure[0] if requested_exposure else exposure.get('min_us')),
            (self.exposure_max, requested_exposure[1] if requested_exposure else exposure.get('max_us')),
            (self.manual_exposure, requested_controls.get('exposure_us', self._control_readback.get('exposure_us'))),
            (self.gain, requested_controls.get('gain_db', self._control_readback.get('gain_db'))),
            (self.preview_gamma, preview.get('gamma')), (self.preview_brightness, preview.get('brightness'))]
        if depth is not None and len(depth) == 2:
            values.extend(((self.min_depth, depth[0]), (self.max_depth, depth[1])))
        for handle, value in values:
            if value is None:
                continue
            key = id(handle)
            if key not in self._settings_seen or handle.value == self._settings_seen[key]:
                handle.value = value
            self._settings_seen[key] = value
        for key, handle in (('exposure_us', self.manual_exposure), ('gain_db', self.gain)):
            limits = self._control_limits.get(key)
            if limits is not None and len(limits) == 2:
                handle.min, handle.max = limits

    def _render_camera_controls(self, laser):
        lines = []
        for key, label, unit in (('exposure_us', 'Actual exposure', 'µs'), ('gain_db', 'Actual gain', 'dB')):
            value = self._control_readback.get(key)
            limits = self._control_limits.get(key)
            if value is not None:
                lines.append(f'{label}: **{value:g} {unit}**')
            if limits is not None:
                lines.append(f'Sensor {label.removeprefix("Actual ")} limits: {limits[0]:g}–{limits[1]:g} {unit}')
        limits = laser.get('exposure_limits') or {}
        for key, label in (('requested_us', 'Requested auto exposure'), ('effective_us', 'Effective auto exposure')):
            values = limits.get(key)
            if values is not None:
                lines.append(f'{label}: {values[0]:g}–{values[1]:g} µs')
        if limits.get('fps_cap_us') is not None:
            lines.append(f'Frame-rate exposure cap: {limits["fps_cap_us"]:g} µs')
        self.control_readback_md.content = '\n\n'.join(lines) or 'Waiting for camera exposure and gain limits.'

    def _clear_cloud(self):
        if self._cloud_handle is not None:
            self._cloud_handle.remove()
        self._cloud_handle = None
        self._cloud_signature = None

    def _render_cloud(self, cloud, scene, visible, hidden_reason=None):
        reason = cloud.get('error') or cloud.get('message') or cloud.get('status') or 'Waiting for a fresh point cloud'
        if not visible:
            self._clear_cloud()
            self.cloud_md.content = hidden_reason or 'Point cloud preview off'
            return
        raw = cloud.get('points_base_m', cloud.get('points_world_m'))
        try:
            points = np.asarray(raw, dtype=float)
            valid = (cloud.get('units') == 'm' and cloud.get('fresh') is not False and points.ndim == 2 and points.shape[1] == 3
                     and 0 < len(points) <= self.MAX_PREVIEW_POINTS and np.isfinite(points).all())
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self._clear_cloud()
            self.cloud_md.content = f'Point cloud unavailable · {reason}'
            return
        signature = (cloud.get('source_id'), cloud.get('calibration_id'), cloud.get('sequence'), points.tobytes())
        if signature != self._cloud_signature:
            self._clear_cloud()
            self._cloud_handle = scene.add_point_cloud('/laser/pointcloud', points=points,
                colors=(255, 85, 55), point_size=.002, point_shape='circle',
                point_shading='flat', precision='float32')
            self._cloud_signature = signature
        self.cloud_md.content = f'{len(points):,} clean points · J6 tool coordinates in meters · displayed in robot base'

    def render(self, state, scene):
        camera, coordinator = state.get('camera') or {}, state.get('laser') or {}
        laser = camera.get('laser') or {}
        enabled = camera.get('enabled', False)
        if enabled != self._stream_enabled or camera.get('error'):
            self.stream.value = enabled
            self._stream_enabled = enabled
        self._sync_settings(laser, camera)
        self._render_camera_controls(laser)
        phase = coordinator.get('status', 'idle')
        active = bool(coordinator.get('busy') or phase in self.ACTIVE)
        calibration = state.get('calibration') or {}
        other_active = calibration.get('status') in self.ACTIVE or bool((state.get('scan') or {}).get('busy'))
        shutdown = state.get('shutdown') or {}
        unsafe = bool(state.get('closed', True) or state.get('disable_pending') or state.get('motion_active')
            or state.get('fault') or state.get('motion_fault') or state.get('mode_transition')
            or state.get('cooldown_active') or shutdown.get('requested') or shutdown.get('active'))
        lifecycle = camera.get('starting') or camera.get('stopping') or camera.get('configuring')
        blocked = unsafe or active or other_active or lifecycle
        self.auto_button.disabled = not enabled or blocked
        self.capture_button.disabled = not camera.get('fresh') or blocked
        self.check_button.disabled = not camera.get('fresh') or blocked
        self.solve_button.disabled = not enabled or blocked
        self.reset_button.disabled = blocked
        configuration_blocked = (active or other_active or lifecycle or state.get('disable_pending')
            or state.get('motion_active') or state.get('fault') or state.get('motion_fault')
            or state.get('mode_transition') or state.get('cooldown_active')
            or shutdown.get('requested') or shutdown.get('active'))
        self.configure_button.disabled = not enabled or bool(configuration_blocked)
        for key, handle in (('exposure_us', self.manual_exposure), ('gain_db', self.gain)):
            handle.disabled = (self.configure_button.disabled or self._control_readback.get(key) is None
                               or self._control_limits.get(key) is None)
        self.preview.disabled = not enabled or lifecycle
        self.cancel_button.disabled = False
        error = coordinator.get('error') or laser.get('error')
        self.status_md.content = (f"**{phase}** · {coordinator.get('message', '')}"
            + (f"\n\nView {coordinator.get('index', 0)} / {coordinator.get('total', 0)}" if active else '')
            + (f'\n\n**Reason:** {error}' if error else ''))
        quality = laser.get('quality') or (coordinator.get('result') or {}).get('quality') or {}
        views = quality.get('views', laser.get('captures', coordinator.get('count', 0)))
        lines = [f'Accepted views: **{views}**']
        if quality.get('rms_mm') is not None:
            lines.append(f"Plane RMS: **{quality['rms_mm']:.3f} mm**")
        if quality.get('span_ratio') is not None:
            lines.append(f"Coverage ratio: {quality['span_ratio']:.3f}")
        if quality.get('inliers') is not None:
            lines.append(f"Accepted points: {quality['inliers']} / {quality.get('total', '—')}")
        attempt = laser.get('last_attempt') or {}
        failed_reason = attempt.get('error') or attempt.get('reason')
        if failed_reason:
            lines.append(f'Last capture: {failed_reason}')
        self.quality_md.content = '\n\n'.join(lines)
        status = laser.get('pointcloud_status') or {}
        exposure = status.get('auto_exposure_status') or {}
        contrast = exposure.get('stripe_contrast', exposure.get('contrast', exposure.get('peak_p90')))
        saturation = exposure.get('saturated_fraction')
        self.exposure_md.content = (f"Stripe exposure: **{exposure.get('state', 'waiting')}** · {exposure.get('message', '')}"
            + (f'\n\nStripe contrast / signal: {contrast:.1f}' if contrast is not None else '')
            + (f' · saturated {saturation:.1%}' if saturation is not None else ''))
        cloud = coordinator.get('cloud') or {}
        visible = bool(self.preview.value and enabled and camera.get('fresh') and not unsafe)
        hidden_reason = None
        if self.preview.value:
            hidden_reason = ('Camera is off' if not enabled else 'Waiting for a fresh camera frame'
                             if not camera.get('fresh') else 'Waiting for a stationary, healthy robot' if unsafe else None)
        self._render_cloud(cloud, scene, visible, hidden_reason)


class LaserScanningUI:
    """JSON-only scan controls and persistent, already registered scan geometry."""

    MAX_DISPLAY_POINTS = 10000

    def __init__(self, gui, send):
        self.send = send
        self._cloud_handle = None
        self._cloud_signature = None
        self._stream_enabled = None
        self._geometry_handles = []
        self._geometry_signature = None
        self.folder = gui.add_folder('Laser scanning')
        with self.folder:
            gui.add_markdown('Scan the visible side at **140 mm camera standoff**. **Fast scan** captures '
                'while moving. **Precision laser scan** stops and captures after each **1 mm** move. '
                'Each button starts the whole service-owned sequence: camera preparation, selected starting '
                'pose, path validation, capture, export and return to verified rest. Enable the healthy arm first.')
            gui.add_markdown('Fast scan uses hardware exposure + kernel-received measured angles; firmware sample age unknown.')
            self.stream = gui.add_checkbox('Stream camera for scanning', initial_value=False)
            self.stream.on_update(self._stream_changed)
            self.start_pose = gui.add_dropdown('Scan starting pose', options=['Tube Scanning', 'Current pose'], initial_value='Tube Scanning')
            self.pattern = gui.add_dropdown('Scan pattern', options=['Vertical down sweep', 'Area'],
                initial_value='Vertical down sweep')
            self.pattern.on_update(self._pattern_changed)
            self.sweep_length = gui.add_number('Downward sweep length (mm)', 80., min=1., max=100., step=1.)
            self.horizontal = gui.add_number('Horizontal scan extent (mm)', 100., min=1., max=100., step=1.)
            self.vertical = gui.add_number('Vertical scan extent (mm)', 100., min=1., max=100., step=1.)
            self.optical_roll = gui.add_number('Fixed optical roll (deg)', 0., min=-180., max=180., step=1.)
            gui.add_markdown('A fixed optical roll can provide wrist-limit clearance. It rotates about the '
                'same viewing axis, preserving the camera origin and the original scan-square center. '
                'It adds no orbit or additional viewing perspective.')
            self.area_md = gui.add_markdown('The optional **100 × 100 mm** area uses the calibrated line width to overlap a few lanes, '
                'with a **15 minute** maximum. The service reports the actual validated duration before scanning. '
                'Fresh snapshots track **1 mm sampling** along each sweep. Measured-pose laser footprints cover '
                'the original **10,201 reference-plane cells**. Surface points, empty observations and rejected frames remain separate.')
            self.fast_button = gui.add_button('Fast scan')
            self.fast_button.on_click(self._start)
            self.start_button = self.fast_button  # Retained for existing UI integration consumers.
            self.precision_button = gui.add_button('Precision laser scan')
            self.precision_button.on_click(lambda event: self._start(event, method='precision'))
            self.precision_md = gui.add_markdown('Precision uses stationary native captures at **1 mm intervals** '
                'on the vertical down sweep. Choose Fast scan for an area scan.')
            self.resume_button = gui.add_button('Resume legacy step scan')
            self.resume_button.visible = False
            self.resume_button.on_click(lambda _: self.send('scan_resume'))
            self.cancel_button = gui.add_button('Cancel scan and hold')
            self.cancel_button.on_click(lambda _: self.send('scan_cancel'))
            self.status_md = gui.add_markdown('Enable the healthy arm, then start. Camera preparation is automatic.')
            self.quality_md = gui.add_markdown('No scan observations yet.')
            self.show_cloud = gui.add_checkbox('Show captured scan', initial_value=True)
            self.show_envelope = gui.add_checkbox('Show scan envelope', initial_value=True)
            self.cloud_md = gui.add_markdown('No captured scan.')
            self.export_button = gui.add_button('Export captured scan')
            self.export_button.on_click(lambda _: self.send('scan_export'))
            self.export_md = gui.add_markdown('')
            self.clear_button = gui.add_button('Clear captured scan')
            self.clear_button.on_click(lambda _: self.send('scan_clear'))
        self._pattern_changed()

    @staticmethod
    def _duration_text(value):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            return 'being estimated'
        seconds = int(np.ceil(value))
        return f'{seconds//60}m {seconds%60:02d}s'

    def _stream_changed(self, event):
        if event.client_id is not None:
            self.send('camera_stream', enabled=self.stream.value)

    def _pattern_changed(self, _event=None):
        area = self.pattern.value == 'Area'
        self.horizontal.visible = self.vertical.visible = self.area_md.visible = area
        self.sweep_length.visible = not area
        self.precision_button.disabled = area or getattr(self.fast_button, 'disabled', False)
        default_roll = -45. if area else 0.
        if self.optical_roll.value == getattr(self, '_pattern_roll_default', 0.):
            self.optical_roll.value = default_roll
        self._pattern_roll_default = default_roll

    def _start(self, _event=None, *, method='fast'):
        self.send('scan_start', scan_method=method,
            scan_profile='area' if self.pattern.value == 'Area' else 'vertical_down',
            sweep_length_mm=self.sweep_length.value, continuous=method == 'fast',
            max_duration_s=900., feed_mm_s=30. if self.pattern.value == 'Area' else 10.,
            start_pose='Tube Scanning' if self.start_pose.value == 'Tube Scanning' else 'current',
            horizontal_mm=self.horizontal.value, vertical_mm=self.vertical.value,
            optical_roll_deg=self.optical_roll.value)

    def _clear_cloud(self):
        if self._cloud_handle is not None:
            self._cloud_handle.remove()
        self._cloud_handle = None
        self._cloud_signature = None

    def _render_cloud(self, scan, scene):
        cloud = scan.get('cloud') or {}
        raw = cloud.get('points_base_m')
        try:
            points = np.asarray(raw, dtype=float)
            valid = (cloud.get('units') == 'm' and points.ndim == 2 and points.shape[1] == 3
                and 0 < len(points) <= self.MAX_DISPLAY_POINTS and np.isfinite(points).all())
        except (TypeError, ValueError):
            valid = False
        if not self.show_cloud.value or not valid:
            self._clear_cloud()
            self.cloud_md.content = ('Captured scan hidden' if not self.show_cloud.value else
                cloud.get('message') or ('Coverage recorded; no accepted surface points.'
                    if scan.get('visited', 0) and not scan.get('point_count', 0) else 'No captured scan.'))
            return
        signature = (cloud.get('scan_id', scan.get('scan_id')), cloud.get('revision', cloud.get('sequence')),
            points.tobytes())
        if signature != self._cloud_signature:
            self._clear_cloud()
            self._cloud_handle = scene.add_point_cloud('/scan/pointcloud', points=points,
                colors=(70, 210, 235), point_size=.0015, point_shape='circle',
                point_shading='flat', precision='float32')
            self._cloud_signature = signature
        stored = scan.get('point_count', len(points))
        self.cloud_md.content = (f'{len(points):,} preview points displayed · {stored:,} stored surface points '
            'in robot base · retained after motion and camera stop')

    def _clear_geometry(self):
        for handle in self._geometry_handles:
            handle.remove()
        self._geometry_handles = []
        self._geometry_signature = None

    def _render_geometry(self, scan, scene):
        """Draw service-supplied coverage geometry; never infer a motion pose."""
        geometry = scan.get('geometry') or {}
        try:
            outline = np.asarray(geometry.get('outline_base_m'), float)
            current = np.asarray(geometry.get('current_base_m'), float)
            valid = (geometry.get('units') == 'm' and geometry.get('coordinate_frame') == 'robot_base'
                and outline.shape == (5, 3) and np.isfinite(outline).all()
                and np.allclose(outline[0], outline[-1], rtol=0, atol=1e-9))
            current_valid = current.shape == (3,) and np.isfinite(current).all()
        except (TypeError, ValueError):
            valid = current_valid = False
        if not self.show_envelope.value or not valid:
            self._clear_geometry()
            return
        signature = (outline.tobytes(), current.tobytes() if current_valid else None)
        if signature == self._geometry_signature:
            return
        self._clear_geometry()
        self._geometry_handles.append(scene.add_line_segments('/scan/envelope',
            points=np.stack((outline[:-1], outline[1:]), axis=1), colors=(245, 190, 80),
            thickness=1.5, thickness_units='screen'))
        if current_valid:
            self._geometry_handles.append(scene.add_point_cloud('/scan/grid-position',
                points=current.reshape(1, 3), colors=(255, 180, 60), point_size=.003,
                point_shape='circle', point_shading='flat', precision='float32'))
        self._geometry_signature = signature

    def render(self, state, scene):
        camera, scan = state.get('camera') or {}, state.get('scan') or {}
        enabled = bool(camera.get('enabled'))
        if enabled != self._stream_enabled:
            self.stream.value = enabled
            self._stream_enabled = enabled
        phase = scan.get('phase', scan.get('status', 'idle'))
        continuous = scan.get('continuous', False)
        active = bool(scan.get('busy'))
        laser, calibration = state.get('laser') or {}, state.get('calibration') or {}
        shutdown = state.get('shutdown') or {}
        blocked = bool(state.get('closed', True) or state.get('disable_pending') or state.get('motion_active')
            or state.get('fault') or state.get('motion_fault') or state.get('mode_transition')
            or state.get('cooldown_active') or shutdown.get('requested') or shutdown.get('active')
            or camera.get('starting') or camera.get('stopping') or camera.get('configuring')
            or laser.get('busy') or laser.get('status') in LaserPlaneCalibrationUI.ACTIVE
            or calibration.get('status') in LaserPlaneCalibrationUI.ACTIVE)
        self.start_button.disabled = blocked or active or not scan.get('start_ready', scan.get('ready', False))
        self.precision_button.disabled = self.start_button.disabled or self.pattern.value == 'Area'
        self.resume_button.visible = not continuous and bool(scan.get('resume_available'))
        self.resume_button.disabled = continuous or self.start_button.disabled or not scan.get('resume_available', False)
        self.stream.disabled = active
        self.start_pose.disabled = self.horizontal.disabled = self.vertical.disabled = active
        self.pattern.disabled = self.sweep_length.disabled = active
        self.optical_roll.disabled = active
        self.cancel_button.disabled = False
        self.clear_button.disabled = active
        count = scan.get('point_count', 0)
        visited = scan.get('visited', scan.get('index', 0))
        total = scan.get('total', 0)
        accepted = scan.get('accepted', scan.get('captures', 0))
        no_return = scan.get('no_return', 0)
        rejected = scan.get('rejected', 0)
        self.export_button.disabled = active or not (count or visited)
        reason = scan.get('error') or (scan.get('blocker') if not active else None)
        progress = 100*min(visited/max(total, 1), 1)
        preflight = scan.get('preflight') or {}
        terminal = phase in ('idle', 'complete', 'completed', 'cancelled', 'error', 'interrupted', 'disabled') and not active
        timing_text = ''
        if continuous:
            if scan.get('elapsed_s') is not None:
                timing_text = f'\n\nElapsed: **{self._duration_text(scan.get("elapsed_s"))}**'
            if not terminal:
                timing_text += f' · Remaining: **{self._duration_text(scan.get("eta_s"))}**'
        self.status_md.content = (f'**{phase}** · {scan.get("message", "Waiting for scan readiness")}'
            + (f'\n\nPreflight: **{preflight.get("index", 0):,} / {preflight.get("total", 0):,}**'
               if preflight else '')
            + f'\n\n{"Reference-plane footprint coverage" if scan.get("coverage_basis") == "measured_reference_plane_laser_footprint" else "Measured 1 mm coverage" if continuous else "Grid visited"}: **{visited:,} / {total:,}** · **{progress:.1f}%**'
            + f'\n\nAccepted: **{accepted:,}** · No return: **{no_return:,}** · Rejected: **{rejected:,}**'
            + f'\n\n{"Captured exposures" if continuous else "Captured snapshots"}: **{scan.get("captures", 0):,}**'
            + (f' · Timing/coverage gaps: **{scan.get("gaps", 0):,}**' if continuous else '')
            + timing_text
            + (f'\n\nValidated motion duration: **{self._duration_text(scan.get("validated_duration_s"))}**' if continuous and scan.get('validated_duration_s') is not None else '')
            + (f'\n\n**Reason:** {reason}' if reason else ''))
        quality = scan.get('quality') or {}
        lines = ['Coverage bins: **1 mm** · camera standoff: **140 mm** · fixed viewing orientation']
        if scan.get('selected_optical_roll_deg') is not None:
            lines.append(f'Selected optical roll: **{scan["selected_optical_roll_deg"]:g}°** · same viewing axis')
        footprint = scan.get('footprint_plan') or {}
        if footprint:
            lines.append(f'Overlapping laser lanes: **{footprint.get("lanes",0):g}** · lane pitch: **{footprint.get("lane_spacing_mm",0):g} mm**')
        receive_reference = scan.get('registration_reference') == 'measured_kernel_rx'
        timing_uncertainty = scan.get('reference_timing_uncertainty_s') if receive_reference else scan.get('timing_uncertainty_s', (scan.get('timing') or {}).get('uncertainty_s'))
        if receive_reference:
            lines.append('Hardware exposure + kernel-received measured angles; firmware sample age unknown.')
        if scan.get('selected_feed_mm_s') is not None:
            lines.append(f'Selected scan feed: **{scan["selected_feed_mm_s"]:g} mm/s**')
        if isinstance(timing_uncertainty, (int, float)) and np.isfinite(timing_uncertainty):
            lines.append(f'{"Receive-reference timing uncertainty" if receive_reference else "Exposure pose timing uncertainty"}: **{timing_uncertainty*1000:g} ms**')
        for key, label, unit in (('stripe_points', 'Decoded stripe', ' points'),
                ('standoff_mm', 'Measured standoff', ' mm'), ('object_points', 'Segmented object', ' points')):
            value = quality.get(key)
            if value is not None:
                lines.append(f'{label}: **{value:g}{unit}**')
        if quality.get('message'):
            lines.append(str(quality['message']))
        self.quality_md.content = '\n\n'.join(lines)
        paths = scan.get('export_paths') or ([scan['export_path']] if scan.get('export_path') else [])
        if isinstance(paths, dict):
            paths = list(paths.values())
        self.export_md.content = '\n\n'.join(f'Saved scan: `{path}`' for path in paths if path)
        self._render_cloud(scan, scene)
        self._render_geometry(scan, scene)


class App:
    def __init__(self, connection, urdf_path, *, port=8080):
        self.connection = connection
        self.server = viser.ViserServer(host='127.0.0.1', port=port, label='AutoWeld')
        self.server.gui.configure_theme(dark_mode=True)
        self.server.scene.add_grid('/floor', width=2, height=2, cell_size=.1, section_size=.5)
        self.visual = ViserUrdf(self.server, Path(urdf_path), root_node_name='/robot')
        self.ee_frame = self.server.scene.add_frame('/tcp', axes_length=.06, axes_radius=.002)
        self.target_handle = self.server.scene.add_transform_controls('/target', scale=.12, visible=False)
        self.state = {}
        self._synced = False
        self._saved_names = None
        self._camera_sequence = None
        self._camera_enabled = None
        self._board_settings_signature = None
        self._gravity_config_signature = None
        self._speed_limits_seen = None
        self._cone_handles = []
        self._cone_signature = None
        self._build_gui()
        self.target_handle.on_update(self._target_changed)

        @self.server.on_client_connect
        def connected(client):
            client.camera.position = (.9, .9, .7)
            client.camera.look_at = (0, 0, .25)

    def send(self, action, **payload):
        self.connection.send(json.dumps({'action': action, **payload}, allow_nan=False))

    async def _target_changed(self, event):
        # Gestures only become waypoints on release. Backend coalesces/queues
        # released targets, preventing old drags from causing delayed sweeps.
        if (event.phase == 'end' and self.state.get('mode') == 'IK'
                and not self.state.get('closed') and not self.state.get('disable_pending')):
            self.send('pose', position=list(event.target.position), wxyz=list(event.target.wxyz))

    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown('# AutoWeld\nYAM Standard · receiver and camera mount')
        self.status_md = gui.add_markdown('Waiting for service…')
        self.telemetry_md = gui.add_markdown('')

        def button(label, action, payload=None):
            handle = gui.add_button(label)
            handle.on_click(lambda _: self.send(action, **(payload() if payload else {})))
            return handle

        with gui.add_folder('Robot control'):
            self.enable_button = button('Enable motors', 'enable')
            self.disable_button = button('Disable motors at rest', 'disable')
            self.shutdown_button = gui.add_button('Shutdown service at rest')
            self.shutdown_button.on_click(self._request_shutdown)
            self.shutdown_md = gui.add_markdown('')
            self.rest_button = button('Return to captured rest', 'rest')
            self.hold_button = button('Stop and hold current pose', 'mode', lambda: {'mode': 'Hold'})
            self.mode_select = gui.add_dropdown('Mode', ['Hold', 'Gravity compensation', 'IK'])
            self.motor_mode_md = gui.add_markdown('Motor mode: waiting for SDK feedback')
            gui.add_markdown('Position–speed motion · TCP speed default 3 m/s; joint speed default 2 rad/s. '
                'Motion also respects joint speed, acceleration and jerk limits. '
                'Disable returns to captured rest and verifies it before removing motor power. '
                'Shutdown service performs the same sequence and then exits. '
                'At 60 °C the service returns to rest for cooldown.\n\n'
                'Gravity compensation automatically switches the same SDK connection to torque mode. '
                'Hold, IK, return to rest and disable automatically restore position–speed mode.')

            @self.mode_select.on_update
            def changed(event):
                if event.client_id is not None:
                    self.send('mode', mode=self.mode_select.value)

            self.ack_button = button('Acknowledge motion fault', 'acknowledge_fault')
        with gui.add_folder('Gravity model and tuning'):
            self.gravity_status_md = gui.add_markdown('Exact SDK gravity model')
            self.gravity_model_select = gui.add_dropdown('Gravity model', ['sdk', 'linear'])
            self.gravity_inputs = [gui.add_number(f'Gravity factor J{i+1}', 1., min=.01, max=5., step=.01) for i in range(6)]
            self.gravity_button = button('Apply gravity factors', 'gravity_configure', lambda: {
                'factors': [h.value for h in self.gravity_inputs], 'model': self.gravity_model_select.value})
            gui.add_markdown('Factors adjust the SDK gravity prediction. Saved settings are tied to this arm and camera mount model. '
                'In position–speed mode, settings are cached for the next gravity compensation entry.')
            self.gravity_torque_md = gui.add_markdown('')
        with gui.add_folder('TCP target · robot base frame'):
            self.xyz = gui.add_vector3('Position (m)', (0., 0., 0.), step=.001)
            self.rpy = gui.add_vector3('Roll / pitch / yaw (rad)', (0., 0., 0.), step=.01)
            self.pose_button = button('Send TCP pose', 'pose', lambda: {
                'position': list(self.xyz.value), 'rpy': list(self.rpy.value)})
            self.sync_button = gui.add_button('Snap target to measured TCP')
            self.sync_button.on_click(lambda _: self.sync_target())
            self.pose_status_md = gui.add_markdown('Select IK, then drag and release the TCP gizmo or send a numeric pose.')
        with gui.add_folder('Joint target · radians'):
            self.joint_inputs = [gui.add_number(f'Joint {i+1} (rad)', 0., step=.001) for i in range(6)]
            self.joint_button = button('Send joint waypoint', 'joint', lambda: {'q': [h.value for h in self.joint_inputs]})
        with gui.add_folder('Saved poses'):
            self.pose_name = gui.add_text('Pose name', initial_value='Rest view')
            self.save_button = button('Save current pose', 'pose_save', lambda: {'name': self.pose_name.value})
            self.saved_folder = gui.add_folder('Click a pose to move')
            self.saved_buttons = []
        with gui.add_folder('Motion limits'):
            self.speed = gui.add_number('TCP speed (m/s)', 3., min=.001, step=.1)
            self.joint_speed = gui.add_number('Joint speed (rad/s)', 2., min=.01, max=3., step=.01)
            self.limit_button = gui.add_button('Apply speed limits')

            self.limit_button.on_click(self._apply_speed_limits)

            self.motion_md = gui.add_markdown('')
        with gui.add_folder('Camera preview'):
            self.camera_on = gui.add_checkbox('Stream camera', initial_value=False)

            @self.camera_on.on_update
            def stream_changed(event):
                if event.client_id is not None:
                    self.send('camera_stream', enabled=self.camera_on.value)

            gui.add_markdown('Streaming starts only when enabled here. Automatic exposure and checkerboard processing run in the camera service.')
            self.camera_image = gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label='Camera is off')
            self.camera_md = gui.add_markdown('Camera off')
        self.calibration_panel_select = gui.add_dropdown('Calibration panel',
            options=['Camera calibration', 'Laser plane calibration', 'Laser scanning'], initial_value='Camera calibration')
        self.camera_calibration_folder = gui.add_folder('Camera calibration')
        with self.camera_calibration_folder:
            self.calibration_kind = gui.add_dropdown('Calibration', ['Intrinsics', 'Hand eye'])
            self.board_columns = gui.add_number('Inner corner columns', 5, min=3, max=20, step=1)
            self.board_rows = gui.add_number('Inner corner rows', 7, min=3, max=20, step=1)
            self.square_mm = gui.add_number('Square size (mm)', 4., min=1., step=.1)
            button('Apply checkerboard settings', 'camera_config', lambda: {
                'columns': int(self.board_columns.value), 'rows': int(self.board_rows.value), 'square_mm': self.square_mm.value})
            def kind():
                return {'kind': 'intrinsic' if self.calibration_kind.value == 'Intrinsics' else 'handeye'}
            self.auto_button = button('Auto calibration', 'calibration_start', kind)
            self.capture_button = button('Capture checkerboard', 'calibration_capture', kind)
            self.solve_button = button('Solve calibration', 'calibration_solve', kind)
            self.cancel_calibration_button = button('Cancel calibration and hold', 'calibration_cancel')
            button('Reset captures', 'calibration_reset', kind)
            self.calibration_md = gui.add_markdown('Auto calibration hunts for a full checkerboard at the baseline and capture poses in 5 mm steps. It then samples 25 edge poses on a cone with a 15° full opening and 150 mm height along the camera axis. Use Cancel to stop the search and hold.')
            self.hunt_md = gui.add_markdown('')
        self.laser_ui = LaserPlaneCalibrationUI(gui, self.send)
        self.laser_ui.folder.visible = False
        self.scan_ui = LaserScanningUI(gui, self.send)
        self.scan_ui.folder.visible = False
        self.calibration_panel_select.on_update(self._select_calibration_panel)

    def _select_calibration_panel(self, _event=None):
        laser_selected = self.calibration_panel_select.value == 'Laser plane calibration'
        self.camera_calibration_folder.visible = self.calibration_panel_select.value == 'Camera calibration'
        self.laser_ui.folder.visible = laser_selected
        self.scan_ui.folder.visible = self.calibration_panel_select.value == 'Laser scanning'
        if _event is not None and _event.client_id is not None:
            # Select camera-owned work before the shared stream starts. Native
            # laser calibration explicitly owns any board search it needs.
            mode = 'checkerboard' if self.camera_calibration_folder.visible else 'laser'
            self.send('camera_analysis', mode=mode)

    def _request_shutdown(self, _event=None):
        self.send('shutdown')

    def _render_shutdown(self):
        shutdown = self.state.get('shutdown') or {}
        phase = shutdown.get('phase', 'idle')
        self.shutdown_button.disabled = False
        self.shutdown_button.label = ('Retry shutdown at rest'
            if shutdown.get('retryable') or phase == 'blocked' else 'Shutdown service at rest')
        descriptions = {
            'stopping': 'Stopping motion before returning to captured rest.',
            'waiting': 'Waiting for fresh feedback and a quiet arm.',
            'returning': 'Returning to captured rest.',
            'settling': 'Verifying captured rest before disabling motors.',
            'blocked': 'Resolve the reported error, then retry shutdown.',
            'disabled': ('Motors disabled at verified rest; stopping the service.'
                         if shutdown.get('rest_verified') else 'Motors disabled; stopping the service.'),
        }
        visible = shutdown.get('requested') or shutdown.get('active') or phase != 'idle' or shutdown.get('error')
        self.shutdown_md.content = (f"**Shutdown: {phase}** · {descriptions.get(phase, '')}" if visible else '')
        if shutdown.get('error'):
            self.shutdown_md.content += f"\n\n**Shutdown error:** {shutdown['error']}"

    def _apply_speed_limits(self, _event=None):
        self.send('speed_limits', linear_velocity=self.speed.value, joint_velocity=self.joint_speed.value)

    def _sync_speed_limits(self):
        """Refresh confirmed values while preserving operator edits awaiting Apply."""
        incoming = (self.state['motion_limits']['linear_velocity'], self.state['position_speed_limit_rad_s'])
        previous = getattr(self, '_speed_limits_seen', None)
        for index, handle in enumerate((self.speed, self.joint_speed)):
            if previous is None or handle.value == previous[index]:
                handle.value = incoming[index]
        self._speed_limits_seen = incoming

    def sync_target(self):
        state = self.state
        if 'tcp_position_m' not in state:
            return
        self.target_handle.position = state['tcp_position_m']
        self.target_handle.wxyz = state['tcp_wxyz']
        self.xyz.value = tuple(state['tcp_position_m'])
        self.rpy.value = tuple(state['tcp_rpy_rad'])
        for handle, q in zip(self.joint_inputs, state['q_rad']):
            handle.value = q

    def _render_saved(self):
        names = tuple(self.state.get('saved_poses', {}))
        if names == self._saved_names:
            return
        for handle in self.saved_buttons:
            handle.remove()
        self.saved_buttons = []
        with self.saved_folder:
            for name in names:
                handle = self.server.gui.add_button(name)
                handle.on_click(lambda _, name=name: self.send('pose_recall', name=name))
                self.saved_buttons.append(handle)
        self._saved_names = names

    def _sync_board_settings(self, settings):
        signature = tuple(settings.get(key) for key in ('columns', 'rows', 'square_mm'))
        if any(value is None for value in signature) or signature == getattr(self, '_board_settings_signature', None):
            return
        self.board_columns.value, self.board_rows.value, self.square_mm.value = signature
        self._board_settings_signature = signature

    def _render_gravity(self):
        state = self.state
        modes = state.get('arm_control_modes') or [state.get('controller', 'unknown')] * 6
        transition = state.get('mode_transition')
        mode_text = modes[0] if len(set(modes)) == 1 else ', '.join(f'J{i+1}: {value}' for i, value in enumerate(modes))
        self.motor_mode_md.content = f"**Motor mode:** {mode_text}" + (' · switching' if transition else '')
        if state.get('mode_transition_error'):
            self.motor_mode_md.content += f"\n\n**Mode transition error:** {state['mode_transition_error']}"
        factors = state.get('gravity_factors') or [1.] * 6
        model = state.get('gravity_model', 'sdk')
        signature = (model, *factors)
        if signature != self._gravity_config_signature:
            for handle, factor in zip(self.gravity_inputs, factors):
                handle.value = factor
            self.gravity_model_select.value = model
            self._gravity_config_signature = signature
        tuning = state.get('gravity_tuning') or {}
        payload = state.get('gravity_payload') or {}
        self.gravity_status_md.content = (f"**{tuning.get('status', 'Default model')}** · {tuning.get('message', '')}"
            + (f"\n\nPayload {payload['mass_kg']:.3f} kg · COM in flange frame (m): {payload['com_flange_m']}" if payload else ''))
        torque = state.get('torque') or {}
        predicted = torque.get('gravity_nm') or []
        reported = state.get('reported_effort_nm') or torque.get('motor_reported_nm') or []
        if predicted or reported:
            def quantity(values, index):
                return f'{values[index]:+.3f}' if index < len(values) else '—'
            self.gravity_torque_md.content = ('| Joint | Gravity prediction Nm | Measured torque Nm |\n|---|---:|---:|\n'
                + '\n'.join(f'| J{i+1} | {quantity(predicted, i)} | {quantity(reported, i)} |' for i in range(6)))
        self.gravity_button.disabled = bool(state.get('motion_active') or state.get('disable_pending')
            or state.get('motion_fault') or state.get('cooldown_active') or transition)

    @staticmethod
    def _board_detection_text(board):
        """Present detector JSON; centroid and offset are computed by the backend."""
        count = board.get('corner_count')
        if count is None:
            corners = board.get('corners')
            count = len(corners) if corners is not None else None
        if board.get('found'):
            label = 'full checkerboard'
        elif board.get('partial_found') or count:
            label = 'partial checkerboard'
        else:
            label = 'no checkerboard corners'
        if count:
            label += f' · {count} corners'
        offset = board.get('offset_px')
        if offset is not None:
            label += f' · offset ({offset[0]:+.1f}, {offset[1]:+.1f}) px'
        return label

    @staticmethod
    def _hunt_progress_text(calibration):
        if calibration.get('status') not in ('hunting', 'hunt_moving', 'hunt_settling'):
            return ''
        hunt = calibration.get('hunt') or {}
        phase = 'Baseline' if hunt.get('phase') == 'baseline' else 'Capture pose'
        progress = (f"**{phase} checkerboard search** · step {hunt.get('steps', 0)} / "
                    f"{hunt.get('max_steps', '—')} · {hunt.get('step_mm', 5):g} mm increments")
        if hunt.get('motion_active'):
            progress += ' · moving; next observation after settling'
        count = hunt.get('corner_count', 0)
        centroid, offset = hunt.get('centroid_px'), hunt.get('offset_px')
        if count and centroid is not None and offset is not None:
            progress += (f"\n\n{count} corners · centroid ({centroid[0]:.1f}, {centroid[1]:.1f}) px"
                         f" · offset from image center ({offset[0]:+.1f}, {offset[1]:+.1f}) px")
        else:
            progress += '\n\nCentroid direction unavailable'
            progress += ' · bounded reacquisition search' if hunt.get('reacquiring') else ' · waiting for fresh corners'
        if hunt.get('message'):
            progress += f"\n\n{hunt['message']}"
        return progress

    def _render_cone(self, geometry):
        """Draw backend geometry in metres, with screen-sized wire thickness."""
        geometry = geometry or {}
        start = np.asarray(geometry.get('axis_start_m'), float)
        apex = np.asarray(geometry.get('apex_m'), float)
        ring = np.asarray(geometry.get('cone_ring_m'), float)
        valid = (start.shape == (3,) and apex.shape == (3,) and ring.ndim == 2
                 and ring.shape[1] == 3 and len(ring) >= 3
                 and all(np.isfinite(array).all() for array in (start, apex, ring)))
        if valid and np.array_equal(ring[0], ring[-1]):
            # Older geometry may repeat the first vertex to close the ring.
            # A closing vertex is not an additional capture-point marker.
            ring = ring[:-1]
            valid = len(ring) >= 3
        signature = (start.tobytes(), apex.tobytes(), ring.tobytes()) if valid else None
        if signature == getattr(self, '_cone_signature', None):
            return
        for handle in reversed(getattr(self, '_cone_handles', [])):
            handle.remove()
        self._cone_handles = []
        self._cone_signature = signature
        if not valid:
            return
        axis = np.stack((start, apex))[None]
        spokes = np.stack((np.broadcast_to(apex, ring.shape), ring), axis=1)
        edges = np.stack((ring, np.roll(ring, -1, axis=0)), axis=1)
        scene = self.server.scene
        self._cone_handles = [
            scene.add_line_segments('/calibration/cone/axis', points=axis,
                colors=(90, 190, 255), thickness=1.5, thickness_units='screen'),
            scene.add_line_segments('/calibration/cone/wire', points=np.concatenate((spokes, edges)),
                colors=(255, 180, 50), thickness=1., thickness_units='screen'),
            scene.add_point_cloud('/calibration/cone/edge-points', points=ring,
                colors=(255, 220, 120), point_size=.003, point_shape='circle',
                point_shading='flat', precision='float32'),
        ]

    def _render_camera(self):
        camera = self.state.get('camera') or {}
        enabled = camera.get('enabled', False)
        if enabled != self._camera_enabled or camera.get('error'):
            self.camera_on.value = enabled
            self._camera_enabled = enabled
        sequence = ('camera', camera.get('sequence'))
        encoded = camera.get('frame_jpeg_base64')
        selector = getattr(self, 'calibration_panel_select', None)
        if selector is not None and selector.value in ('Laser plane calibration', 'Laser scanning'):
            laser = camera.get('laser') or {}
            if laser.get('frame_jpeg_base64'):
                encoded = laser['frame_jpeg_base64']
                preview = laser.get('preview_frame') or {}
                sequence = ('laser', preview.get('generation'), preview.get('sequence', camera.get('sequence')), encoded)
        if encoded and sequence != self._camera_sequence:
            # Decode a transport JPEG for display only. All image acquisition,
            # checkerboard detection and annotations belong to the backend.
            from PIL import Image
            self.camera_image.image = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))
            self._camera_sequence = sequence
        if not camera.get('enabled'):
            if self._camera_sequence is not None:
                self.camera_image.image = np.zeros((240, 320, 3), dtype=np.uint8)
                self._camera_sequence = None
        board = camera.get('board') or {}
        self._sync_board_settings(board.get('settings') or {})
        counts = camera.get('counts') or {}
        self.camera_md.content = (f"**{'Streaming' if camera.get('enabled') else 'Off'}** · "
            f"{(camera.get('fps') or 0):.1f} fps · {self._board_detection_text(board)}\n\n"
            f"Captures: intrinsics {counts.get('intrinsic', 0)} · hand eye {counts.get('handeye', 0)}"
            + (f"\n\n{camera['error']}" if camera.get('error') else ''))
        calibration = self.state.get('calibration') or {}
        self.hunt_md.content = self._hunt_progress_text(calibration)
        laser_selected = selector is not None and selector.value == 'Laser plane calibration'
        self._render_cone(None if selector is not None and selector.value == 'Laser scanning' else
            ((self.state.get('laser') or {}) if laser_selected else calibration).get('geometry'))
        if calibration:
            self.calibration_md.content = (f"**{calibration.get('status', 'Idle')}** · "
                f"{calibration.get('message', '')}\n\n{calibration.get('index', 0)} / {calibration.get('total', 0)} poses")
            geometry = calibration.get('geometry') or {}
            if 'radius_m' in geometry and 'opening_angle_deg' in geometry:
                self.calibration_md.content += (f"\n\nCone: {len(geometry['cone_ring_m'])} edge points · "
                    f"{geometry['distance_m']*1000:g} mm height · {geometry['radius_m']*1000:.2f} mm radius · "
                    f"{geometry['opening_angle_deg']:g}° full opening")
        laser = self.state.get('laser') or {}
        laser_active = laser.get('busy') or laser.get('status') in LaserPlaneCalibrationUI.ACTIVE
        active = bool((self.state.get('scan') or {}).get('busy')) or laser_active or calibration.get('status') in ('starting', 'moving', 'settling', 'capturing', 'returning', 'capturing_manual', 'solving',
            'hunting', 'hunt_moving', 'hunt_settling')
        self.auto_button.disabled = not camera.get('enabled') or active or self.state.get('closed', True)
        self.capture_button.disabled = not camera.get('fresh') or active
        self.solve_button.disabled = active
        self.cancel_calibration_button.disabled = False

    def render(self):
        state = json.loads(self.connection.snapshot())
        if not state:
            return
        previous_mode = self.state.get('mode')
        self.state = state
        q = state['q_rad']
        self.visual.update_cfg(np.asarray(q))
        self.ee_frame.position = state['tcp_position_m']
        self.ee_frame.wxyz = state['tcp_wxyz']
        closed, pending = state['closed'], state['disable_pending']
        fault = state.get('motion_fault') or state.get('fault')
        busy = closed or pending or bool(fault) or state.get('motion_active') or state.get('cooldown_active')
        self.enable_button.disabled = not closed
        self.disable_button.disabled = closed or pending
        self._render_shutdown()
        self.hold_button.disabled = closed or pending
        self.rest_button.disabled = busy
        self.mode_select.disabled = closed or pending or bool(fault)
        if (state['mode'] in self.mode_select.options
                and (not self._synced or previous_mode != state['mode']
                     or (state.get('last_request') or {}).get('status') == 'rejected')):
            self.mode_select.value = state['mode']
        sdk_stopped = state.get('sdk', {}).get('control_alive') is False
        self.ack_button.disabled = closed or not state.get('motion_fault') or sdk_stopped
        self.joint_button.disabled = busy
        self.pose_button.disabled = closed or pending or bool(fault)
        self.save_button.disabled = closed or pending
        self.limit_button.disabled = busy or state.get('controller') != 'POS_VEL' or bool(state.get('mode_transition'))
        self.target_handle.visible = state['mode'] == 'IK' and not closed and not pending
        for handle in self.saved_buttons:
            handle.disabled = busy
        if not self._synced or previous_mode != state['mode']:
            self.sync_target()
            self._synced = True
        self._sync_speed_limits()
        self.status_md.content = (f"**{'LIVE HARDWARE' if state['live'] else 'OFFLINE SIMULATION'} · {state['mode']}**\n\n"
            + state.get('message', '') + (f"\n\n**Fault:** {fault}" if fault else '')
            + (f"\n\n**Feedback warning:** {state['feedback_warning']}" if state.get('feedback_warning') else '')
            + ('\n\n**Cooldown: returning to captured rest**' if state.get('cooldown_active') else ''))
        host = state.get('host_health')
        if host and not host['safe']:
            self.status_md.content += '\n\n**Host pressure:** ' + '; '.join(host['reasons'])
        temperatures = state.get('temperatures') or {}
        mos, rotor = temperatures.get('mos_c', []), temperatures.get('rotor_c', [])
        motor_errors = state.get('motor_errors') or {}
        rows = []
        for i in range(6):
            temp = max(mos[i] if i < len(mos) else 0, rotor[i] if i < len(rotor) else 0)
            rows.append(f"| J{i+1} | {q[i]:.4f} | {state['target_rad'][i]:.4f} | {state['error_deg'][i]:+.3f} | "
                f"{f'{temp:.1f}' if mos or rotor else '—'} | {motor_errors.get(str(i+1), '—')} |")
        cached = state.get('sdk', {}).get('required') and not state.get('sdk', {}).get('feedback_fresh')
        self.telemetry_md.content = (f"| Joint | {'Cached' if cached else 'Actual'} rad | Target rad | Error ° | Temp °C | Motor code |\n"
            '|---|---:|---:|---:|---:|---:|\n' + '\n'.join(rows)
            + f"\n\nFeedback: {state['feedback_source']} · Rest {'verified' if state['rest_verified'] else 'not verified'} "
            f"(±{state['rest_tolerance_deg']:g}° for one second)")
        if cached:
            self.telemetry_md.content += '\n\n**Cached readings: motor codes and temperatures do not verify current power or hardware health.**'
        request = state.get('last_request') or {}
        if request:
            diagnostic = request.get('diagnostic') or {}
            stage = diagnostic.get('stage', '').replace('_', ' ')
            self.pose_status_md.content = f"**{request['action']}: {request['status']}** {stage}\n\n" + request.get('error', request.get('message', ''))
        plan = state.get('motion_plan') or {}
        if plan:
            self.motion_md.content = f"**{'Moving' if state.get('motion_active') else 'Reference complete'}** · {plan.get('duration_s', 0):.2f} s"
        self._render_saved()
        self._render_gravity()
        self._render_camera()
        self.laser_ui.render(state, self.server.scene)
        self.scan_ui.render(state, self.server.scene)

    def close(self):
        self.server.stop()


def main(live: bool = False, port: int = 8080, channel: str = 'can0',
         camera: str = '', runtime: str | None = None, step: str | None = None,
         start_disabled: bool = False):
    """Serve the singleton application; --live explicitly connects the robot SDK."""
    from autoweld.backend import run_application
    run_application(live=live, port=port, channel=channel, camera_source=camera,
                    runtime=runtime, step=step, start_disabled=start_disabled)


def cli():
    import tyro
    tyro.cli(main)


if __name__ == '__main__':
    cli()
