"""Tick-driven laser scanning through the existing service owners only."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.calibration import transform
from autoweld.laser_scanning import ScanArchive, ScanSettings, scan_offsets, segment_object, tracking_target


class LaserScanCoordinator:
    terminal = frozenset(('idle', 'complete', 'cancelled', 'error', 'interrupted'))

    def __init__(self, camera, tcp_pose, move_pose, ready, hold, move_saved_pose, runtime, *,
                 feedback, can_start, validate_pose=None, model_fingerprint=None, clock=time.monotonic,
                 executor=None, move_is_validated=False):
        self.camera, self.tcp_pose, self.move_pose = camera, tcp_pose, move_pose
        self.ready, self.hold, self.move_saved_pose = ready, hold, move_saved_pose
        self.feedback, self.can_start, self.validate_pose = feedback, can_start, validate_pose
        self.move_is_validated = bool(move_is_validated)
        self.clock, self.model_fingerprint = clock, model_fingerprint
        self.archive = ScanArchive(runtime)
        self._saved = self.archive.state(include_points=True)
        self.status = self._saved.get('status', 'idle')
        self.message = self._saved.get('message', 'Ready for a calibrated stripe')
        self.error = None
        self.settings = ScanSettings()
        self.index = self.total = 0
        self.return_pose = None
        self.view_pose = None
        self.optical_roll_deg = 0.
        self.endpoint_error = self.quality = None
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-scan-save')
        self.future = self._pending_kind = None
        self._finish_pending = None
        self._identity = self._quiet_pose = self._stable_since = None
        self._goal = self._baseline_optical = self._handeye = None
        self._offsets = []
        self._segment = self._last_acquisition = None
        self._depth_moves = 0
        self._alignment_moves = 0
        self._grid_approach = False
        self._safe_depth_offset = 0.
        self._object_world = None
        self._object_camera_origin = None
        self._preflight_order = []
        self._preflight_index = 0
        self._preflight_complete = False
        self._preflight_last_progress = None
        self._after_move_status = 'settling'
        self._capture_arrival = False
        self._grid_miss_since = self._grid_miss_checked_at = None
        self._grid_miss_samples = 0
        self._resuming = False
        self._adjusted_baseline_pose = None
        self._geometry = None
        self._closed = False
        self._archive_started = False
        self._stop_hold_done = False
        self._restore_geometry()

    @property
    def busy(self):
        return self.status not in self.terminal or self.future is not None or self._finish_pending is not None

    @staticmethod
    def _camera_identity(state):
        return tuple(state.get(key) for key in
                     ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id'))

    def _observation(self, state, *, initial=False):
        if not state.get('enabled') or state.get('error'):
            raise ValueError(state.get('error') or 'Turn on a fresh camera stream before scanning')
        if state.get('starting') or state.get('stopping') or state.get('configuring'):
            raise ValueError('Wait for camera lifecycle operation before scanning')
        handeye = state.get('handeye')
        identity = self._camera_identity(state)
        if any(value is None or value == '' for value in identity):
            raise ValueError('Scanning needs installed optical and laser calibration identities')
        if state.get('generation') is None:
            raise ValueError('Scanning needs the current camera acquisition generation')
        if not initial and identity != self._identity:
            raise ValueError('Camera or calibration identity changed during scanning')
        if (not isinstance(handeye, dict) or handeye.get('source_id') != identity[0]
                or handeye.get('optical_intrinsics_id', handeye.get('calibration_id')) != identity[1]):
            raise ValueError('Scanning requires measured hand-eye matching the camera optics')
        correction = transform(handeye['tcp_from_camera'])
        if not initial and not np.allclose(correction, self._handeye, atol=1e-10, rtol=0):
            raise ValueError('Measured hand-eye changed during scanning')
        intrinsics = state.get('intrinsics') or {}
        plane = np.asarray(intrinsics.get('laser_plane'), float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
            raise ValueError('A calibrated laser plane is required before scanning')
        cloud = (state.get('laser') or {}).get('cloud')
        if not isinstance(cloud, dict):
            reason = ((state.get('laser') or {}).get('pointcloud_status') or {}).get('error')
            raise ValueError(reason or 'Enable clean point-cloud decoding and wait for a good stripe')
        acquired = cloud.get('read_started')
        if type(acquired) not in (int, float) or not np.isfinite(acquired) or not 0 <= self.clock()-acquired <= 1.:
            raise ValueError('Decoded stripe acquisition is stale or in the future')
        for key, expected in zip(('source_id', 'optical_intrinsics_id', 'calibration_id',
                                  'acquisition_session_id', 'generation'), (*identity, state['generation'])):
            if cloud.get(key) != expected:
                raise ValueError('Decoded stripe and current camera calibration identities differ')
        return cloud, correction

    def _readiness(self, camera_state):
        try:
            self.can_start()
            cloud, _ = self._observation(camera_state, initial=True)
            segment = segment_object(cloud)
            quality = self._quality(segment)
            return True, None, quality
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            return False, str(error), self.quality

    @staticmethod
    def _quality(segment):
        return {**segment['quality'], 'stripe_points': segment['quality']['input_count'],
                'object_points': segment['quality']['count'],
                'standoff_mm': segment['centroid_camera_m'][2]*1000.}

    def _restore_geometry(self):
        """Restore the fixed grid, never an old motion target or hardware owner."""
        metadata = self._saved.get('metadata') or {}
        if not metadata.get('baseline_pose') or not metadata.get('handeye'):
            return False
        try:
            self.settings = ScanSettings(**metadata['settings']).validate()
            self.return_pose = transform(metadata['baseline_pose'])
            self.view_pose = transform(metadata.get('view_pose', self.return_pose))
            self.optical_roll_deg = float(metadata.get('optical_roll_deg', 0.))
            self._handeye = transform(metadata['handeye']['tcp_from_camera'])
            self._baseline_optical = self.return_pose @ self._handeye
            self._offsets = scan_offsets(self.settings)
            self.total = len(self._offsets)
            self.index = int(self._saved.get('next_index', self._saved.get('captures', 0)))
            self._safe_depth_offset = float(metadata.get('safe_depth_offset_m', 0.))
            last_pose = (self._saved.get('last_cell') or {}).get('base_from_tcp')
            if last_pose is not None:
                self._safe_depth_offset = self._optical_offset(transform(last_pose))[2]
            self._adjusted_baseline_pose = transform(metadata.get('adjusted_baseline_pose', self.return_pose))
            self._identity = tuple((metadata.get('identity') or [])[:4])
            self._update_geometry()
            return True
        except (ValueError, KeyError, TypeError):
            return False

    def _resume_available(self):
        metadata = self._saved.get('metadata') or {}
        settings = metadata.get('settings') or {}
        total = (round(settings.get('width_mm', 0))+1)*(round(settings.get('height_mm', 0))+1)
        return bool(self._saved.get('scan_id') and metadata.get('baseline_pose') and metadata.get('handeye')
                    and self.status in ('cancelled', 'error', 'interrupted')
                    and self._saved.get('status') != 'complete'
                    and 0 <= self._saved.get('next_index', self.index) < total)

    def _optical_offset(self, pose):
        optical = transform(pose) @ self._handeye
        return self._baseline_optical[:3, :3].T @ (optical[:3, 3]-self._baseline_optical[:3, 3])

    def _grid_pose(self, xy, depth=None):
        optical = self._baseline_optical.copy()
        optical[:3, 3] += optical[:3, :3] @ np.r_[xy, self._safe_depth_offset if depth is None else depth]
        # Raster coordinates stay in the ORIGINAL Tube image plane. A chosen
        # fixed optical roll changes the viewing basis, never the grid center,
        # translation axes, optical origin or forward viewing axis.
        optical[:3, :3] = (self.view_pose @ self._handeye)[:3, :3]
        return optical @ np.linalg.inv(self._handeye)

    def _update_geometry(self):
        if self._baseline_optical is None:
            self._geometry = None
            return
        x, y = self.settings.width_mm/2000., self.settings.height_mm/2000.
        outline = np.array([[-x, -y, 0.], [x, -y, 0.], [x, y, 0.], [-x, y, 0.], [-x, -y, 0.]])
        self._geometry = {'units': 'm', 'coordinate_frame': 'robot_base',
                          'outline_base_m': (outline @ self._baseline_optical[:3, :3].T
                                             + self._baseline_optical[:3, 3]).tolist()}

    def state(self, camera_state=None):
        camera_state = self.camera.state(include_frame=False) if camera_state is None else camera_state
        ready, blocker, quality = (False, 'Scan is active', self.quality) if self.busy else self._readiness(camera_state)
        saved = self._saved
        cloud = {'units': 'm', 'coordinate_frame': 'robot_base', 'scan_id': saved.get('scan_id'),
                 'revision': saved.get('cloud_revision', 0), 'points_base_m': saved.get('points_base_m', [])}
        geometry = None if self._geometry is None else dict(self._geometry)
        if geometry is not None and self.index < len(self._offsets):
            point = np.r_[self._offsets[self.index], self._safe_depth_offset]
            geometry['current_base_m'] = (self._baseline_optical[:3, :3] @ point
                                           + self._baseline_optical[:3, 3]).tolist()
        return {'status': self.status, 'busy': self.busy, 'ready': ready, 'blocker': blocker,
                'message': self.message, 'error': self.error, 'index': self.index, 'total': self.total,
                'captures': saved.get('captures', 0), 'point_count': saved.get('points', 0),
                'scan_id': saved.get('scan_id'), 'cloud': cloud, 'quality': quality,
                'spacing_mm': 1., 'standoff_mm': 140., 'settings': asdict(self.settings),
                'optical_roll_deg': self.optical_roll_deg,
                'endpoint_error': self.endpoint_error, 'export_paths': saved.get('export_paths', {}),
                'visited': saved.get('cells_visited', saved.get('captures', 0)),
                'accepted': saved.get('captures', 0), 'no_return': saved.get('cells_no_return', 0),
                'rejected': saved.get('cells_rejected', 0), 'display_point_count': saved.get('preview_count',
                    len(saved.get('points_base_m', []))), 'resume_available': self._resume_available(),
                'preflight': {'index': self._preflight_index, 'total': len(self._preflight_order)},
                'geometry': geometry}

    def _submit(self, kind, function, *args, **kwargs):
        if self.future is not None:
            raise RuntimeError('Scan persistence is already active')
        def save():
            function(*args, **kwargs)
            return self.archive.state(include_points=True)
        self._pending_kind = kind
        try:
            self.future = self.executor.submit(save)
        except Exception as error:
            self._persistence_failed(error)
            return False
        self._operation_started = self.clock()
        return True

    def start(self, request):
        if self.busy:
            raise RuntimeError('Wait for the scan or cancellation persistence to finish')
        settings = ScanSettings(width_mm=request.get('horizontal_mm', 10.),
                                height_mm=request.get('vertical_mm', 10.)).validate()
        start_pose = request.get('start_pose', 'Tube Scanning')
        if start_pose not in ('Tube Scanning', 'current'):
            raise ValueError('Scan start must be Tube Scanning or current')
        roll = request.get('optical_roll_deg', 0.)
        if type(roll) not in (int, float) or not np.isfinite(roll) or not -180. <= roll <= 180.:
            raise ValueError('Fixed scan optical roll must be finite and between -180 and 180 degrees')
        self.can_start()
        state = self.camera.state(include_frame=False)
        cloud, handeye = self._observation(state, initial=True)
        segment = segment_object(cloud)  # No motion may begin without a usable stripe.
        self.settings, self._identity, self._handeye = settings, self._camera_identity(state), handeye
        self.optical_roll_deg = float(roll)
        self.quality, self.error, self.endpoint_error = self._quality(segment), None, None
        self._segment = self._last_acquisition = None
        self._offsets = scan_offsets(settings)
        self.index, self.total, self._depth_moves = 0, len(self._offsets), 0
        self._alignment_moves = 0
        self._safe_depth_offset = 0.
        self._object_world = None
        self._object_camera_origin = None
        self._preflight_order = []
        self._preflight_index = 0
        self._preflight_complete = False
        self._resuming = False
        self._geometry = None
        self._finish_pending = None
        self._archive_started = False
        self._stop_hold_done = False
        self.return_pose = None
        self.view_pose = None
        self.status, self.message = 'starting', 'Checking the single-perspective scan baseline'
        try:
            if start_pose == 'Tube Scanning':
                self._goal = transform(self.move_saved_pose(start_pose))
                self._move_started = self.clock()
                self.status, self.message = 'positioning', 'Moving to saved Tube Scanning pose'
            else:
                self._begin_baseline()
        except Exception as error:
            self._fail(error)
            raise
        return self.state(camera_state=state)

    def _begin_baseline(self):
        self.hold()  # One measured Hold, never retargeted during observation.
        self.return_pose = transform(self.tcp_pose())
        self._baseline_optical = self.return_pose @ self._handeye
        optical_view = self._baseline_optical.copy()
        optical_view[:3, :3] = optical_view[:3, :3] @ Rotation.from_euler('z', self.optical_roll_deg, degrees=True).as_matrix()
        self.view_pose = (self.return_pose.copy() if abs(self.optical_roll_deg) <= 1e-10
                          else optical_view @ np.linalg.inv(self._handeye))
        self._update_geometry()
        if abs(self.optical_roll_deg) > 1e-10:
            self._move(self.view_pose, after='baseline_settling')
            self.message = f'Applying fixed optical roll {self.optical_roll_deg:g} degrees at the original camera origin'
            return
        self._begin_settle('baseline_settling')

    def resume(self):
        if self.busy:
            raise RuntimeError('Wait for the current scan or persistence before resuming')
        if not self._resume_available() or not self._restore_geometry():
            raise ValueError('No incomplete scan grid is available to resume')
        self.can_start()
        state = self.camera.state(include_frame=False)
        cloud, handeye = self._observation(state, initial=True)
        segment = segment_object(cloud)
        metadata = self._saved['metadata']
        expected = tuple(metadata['identity'][:3])
        if self._camera_identity(state)[:3] != expected or not np.allclose(handeye, self._handeye, atol=1e-10, rtol=0):
            raise ValueError('Resume requires the original camera optics, laser and measured hand-eye calibration')
        if metadata.get('model_fingerprint') != self.model_fingerprint:
            raise ValueError('Resume requires the original installed robot model')
        current = transform(self.tcp_pose())
        offset = self._optical_offset(current)
        angle = Rotation.from_matrix(current[:3, :3] @ self.view_pose[:3, :3].T).magnitude()
        reapply_view = None
        if angle > .005:
            original_angle = Rotation.from_matrix(current[:3, :3] @ self.return_pose[:3, :3].T).magnitude()
            if (abs(self.optical_roll_deg) > 1e-10 and original_angle <= .005
                    and np.linalg.norm(offset[:2]) <= .001):
                # Explicit Resume after a controlled cooldown/Go-to-Tube may
                # reapply this session's fixed roll at the NEW measured camera
                # origin. It does not replay a saved working target.
                optical = current @ self._handeye
                optical[:3, :3] = (self.view_pose @ self._handeye)[:3, :3]
                reapply_view = optical @ np.linalg.inv(self._handeye)
            else:
                raise ValueError('Move back to the fixed Tube Scanning view before resuming the retained grid')
        if (abs(offset[0]) > self.settings.width_mm/2000.+.001
                or abs(offset[1]) > self.settings.height_mm/2000.+.001 or abs(offset[2]) > .06):
            raise ValueError('Move back to the fixed Tube Scanning view before resuming the retained grid')
        if self.validate_pose:
            self.validate_pose(current if reapply_view is None else reapply_view)
        self._identity = self._camera_identity(state)
        self._segment, self.quality = segment, self._quality(segment)
        self._object_world = None
        self._object_camera_origin = None
        self._preflight_complete = False
        self._preflight_order = []
        self._preflight_index = 0
        self._resuming = True
        self._archive_started = False
        self._stop_hold_done = False
        self._finish_pending = None
        self._depth_moves = self._alignment_moves = 0
        self._last_acquisition = None
        self._goal = None
        self.error = self.endpoint_error = None
        self.hold()
        self.message = 'Resuming the retained grid after fresh measured Hold and full remaining-grid preflight'
        if reapply_view is not None:
            self._move(reapply_view, after='baseline_settling')
            self.message = f'Reapplying fixed optical roll {self.optical_roll_deg:g} degrees at the fresh measured Tube origin'
        else:
            self._begin_settle('baseline_settling')
        return self.state(camera_state=state)

    def _begin_settle(self, status='settling', *, capture_arrival=False):
        self.status, self._quiet_pose, self._stable_since = status, None, None
        self._capture_arrival = capture_arrival
        self._clear_grid_miss()
        self._wait_started = self.clock()

    def _clear_grid_miss(self):
        self._grid_miss_since = self._grid_miss_checked_at = None
        self._grid_miss_samples = 0

    def _capture_grid_aligned(self, pose=None, *, correct=True):
        """Keep the exact grid gate while filtering isolated post-Hold jitter."""
        pose = self.tcp_pose() if pose is None else pose
        error = np.linalg.norm(self._optical_offset(pose)[:2]-self._offsets[self.index])
        if error <= .0005:
            self._clear_grid_miss()
            return True
        if self._alignment_moves >= 240:
            raise ValueError('Scan XY alignment did not converge within its bounded positioning moves')
        now = self.clock()
        if self._grid_miss_since is None:
            self._grid_miss_since = now
        if self._grid_miss_checked_at != now:
            self._grid_miss_samples += 1
            self._grid_miss_checked_at = now
        self.message = f'Waiting for fresh in-tolerance optical XY before capture ({error*1000.:.3f} mm)'
        if (correct and now-self._grid_miss_since >= .2 and self._grid_miss_samples >= 2
                and self._stable_since is not None and now-self._stable_since >= .4):
            self._position_xy()
        return False

    def _quiet(self):
        now = self.clock()
        feedback = self.feedback()
        pose = transform(self.tcp_pose())
        if not feedback.get('fresh'):
            self._quiet_pose = self._stable_since = None
            self._clear_grid_miss()
            self.message = 'Waiting for fresh SDK feedback before scan observation'
            return False
        if not feedback.get('stationary') or not self.ready():
            self._quiet_pose = self._stable_since = None
            self._clear_grid_miss()
            return False
        if (self._quiet_pose is None or np.linalg.norm(pose[:3, 3]-self._quiet_pose[:3, 3]) > .0003
                or Rotation.from_matrix(pose[:3, :3] @ self._quiet_pose[:3, :3].T).magnitude() > .003):
            self._quiet_pose, self._stable_since = pose, now
            self._clear_grid_miss()
        return now-self._stable_since >= .4

    def _endpoint(self):
        pose, now = transform(self.tcp_pose()), self.clock()
        position = float(np.linalg.norm(pose[:3, 3]-self._goal[:3, 3]))
        angle = float(Rotation.from_matrix(pose[:3, :3] @ self._goal[:3, :3].T).magnitude())
        feedback = self.feedback()
        optical_xy = None
        if self.status == 'moving' and self._after_move_status == 'settling' and self._handeye is not None:
            observed, requested = pose @ self._handeye, self._goal @ self._handeye
            delta = requested[:3, :3].T @ (observed[:3, 3]-requested[:3, 3])
            optical_xy = float(np.linalg.norm(delta[:2]))
        converged = (feedback.get('fresh') and position <= .0005 and angle <= .005
                     and (optical_xy is None or optical_xy <= .0005)
                     and feedback.get('stationary') and self.ready())
        self.endpoint_error = {'position_mm': position*1000., 'rotation_rad': angle,
                               'converged': bool(converged), 'position_tolerance_mm': .5,
                               'rotation_tolerance_rad': .005,
                               'optical_xy_mm': None if optical_xy is None else optical_xy*1000.,
                               'optical_xy_tolerance_mm': .5}
        if not converged and now-self._move_started > (300. if self.status == 'positioning' else 30.):
            detail = '' if optical_xy is None else f', optical XY {optical_xy*1000.:.3f} mm'
            raise ValueError(f'Scan endpoint did not converge: {position*1000.:.3f} mm / {angle:.6f} rad{detail}')
        return converged

    def _move(self, goal, *, after='settling'):
        goal = transform(goal)
        if self.view_pose is not None:
            goal[:3, :3] = self.view_pose[:3, :3]
        if self.validate_pose and not self.move_is_validated:
            self.validate_pose(goal)
        self.move_pose(goal)
        self._goal, self._move_started, self.status = goal, self.clock(), 'moving'
        self._after_move_status = after
        self._capture_arrival = False
        self._clear_grid_miss()

    def _position_xy(self):
        """Follow the preflighted grid at the last measured safe optical depth."""
        actual = self._optical_offset(self.tcp_pose())
        remaining = np.asarray(self._offsets[self.index])-actual[:2]
        if np.linalg.norm(remaining) <= .0005:
            return True
        self._alignment_moves += 1
        # The sample targets are the fixed nominal 1 mm raster. Prior measured
        # Hold may be up to 0.5 mm from its target; clamping the next measured
        # start-to-target displacement to 1 mm invents a short intermediate
        # target, then freezes its own tolerated lag and requires another move.
        # Plan directly to this cell through the normal full-path planner. Its
        # speed, acceleration, jerk, joint and collision checks are unchanged.
        if self._alignment_moves > 240:
            raise ValueError('Scan XY alignment did not converge within its bounded positioning moves')
        self._move(self._grid_pose(self._offsets[self.index]))
        self.message = f'Following fixed grid XY at cell {self.index+1}/{self.total}'
        return False

    def _prepare_preflight(self):
        if self._preflight_order:
            return
        # Reject an unreachable corner quickly, then validate every remaining
        # cell. Each tick performs at most one IK/installed-bound/collision check.
        corners = []
        if self._offsets:
            width = round(self.settings.width_mm)+1
            height = round(self.settings.height_mm)+1
            corners = [0, width-1, (height-1)*width, height*width-1]
        ordered = dict.fromkeys([i for i in corners if i >= self.index]
                                + list(range(self.index, len(self._offsets))))
        self._preflight_order = list(ordered)
        self._preflight_index = 0
        self._preflight_last_progress = self.clock()

    def _preflight_tick(self):
        self._prepare_preflight()
        if self._preflight_index >= len(self._preflight_order):
            self._preflight_complete = True
            return True
        feedback = self.feedback()
        if not feedback.get('fresh') or not feedback.get('stationary') or not self.ready():
            if self.clock()-self._preflight_last_progress > 15.:
                raise ValueError('Full-grid preflight requires fresh stationary SDK feedback')
            self.message = 'Waiting for fresh stationary SDK feedback during preflight'
            return False
        index = self._preflight_order[self._preflight_index]
        if self.validate_pose:
            self.validate_pose(self._grid_pose(self._offsets[index]))
        self._preflight_index += 1
        self._preflight_last_progress = self.clock()
        self.message = f'Preflighting fixed grid {self._preflight_index}/{len(self._preflight_order)} before XY motion'
        if self._preflight_index == len(self._preflight_order):
            self._preflight_complete = True
            return True
        return False

    def _update_segment(self, segment, pose):
        self._segment, self.quality = segment, self._quality(segment)
        optical = pose @ self._handeye
        self._object_world = (optical[:3, :3] @ np.asarray(segment['centroid_camera_m']) + optical[:3, 3])
        self._object_camera_origin = optical[:3, 3].copy()
        self._safe_depth_offset = float(self._optical_offset(pose)[2])

    def _previous_centroid(self, pose):
        if self._object_world is None:
            return None
        optical = pose @ self._handeye
        predicted_patch = self._object_world.copy()
        if self._object_camera_origin is not None:
            # The stripe samples a local surface patch as the fixed-view grid
            # moves. Predict that patch's image-plane shift, not a permanently
            # anchored world point left behind at the center of the scan.
            # This prediction selects native components only; it never changes
            # measured XYZ or invents a depth correction/capture coordinate.
            delta = optical[:3, 3]-self._object_camera_origin
            forward = self._baseline_optical[:3, 2]
            predicted_patch += delta-forward*np.dot(forward, delta)
        return optical[:3, :3].T @ (predicted_patch-optical[:3, 3])

    def _at_offset(self, segment):
        """Fixed baseline XY; only measured depth changes along optical +Z."""
        pose = transform(self.tcp_pose())
        # Use the same 0.5 mm measured endpoint tolerance as the service. A
        # tighter grid gate after measured Hold otherwise replans sub-tolerance
        # errors forever. Nominal raster targets remain exactly 1 mm apart;
        # every cloud stores the actual measured pose, rather than the target.
        if not self._capture_grid_aligned(pose):
            return False
        if abs(segment['centroid_camera_m'][2]-.140) > .001:
            self._depth_moves += 1
            if self._depth_moves > 12:
                raise ValueError('Object depth did not converge to 140 mm within 12 bounded corrections')
            self._move(tracking_target(pose, self._handeye, segment, settings=self.settings))
            self.message = 'Adjusting optical depth toward 140 mm'
            return False
        return True

    @staticmethod
    def _quality_rejection(error):
        return str(error).startswith(('A good decoded stripe', 'Laser stripe is dominated',
            'Laser stripe is too', 'No continuous object stripe', 'Multiple disconnected object stripes',
            'Tracked object stripe was lost', 'Tracked stripe association', 'Object stripe spans'))

    def _save_rejected_cell(self, cloud, pose, reason, *, outcome='rejected'):
        measured = self._capture_measurement(cloud)
        if measured is None:
            return
        pose, feedback = measured
        if self._submit('record_cell', self.archive.record_cell, outcome, copy.deepcopy(cloud), pose, self._handeye,
                        now=self.clock(), not_before_monotonic=self._stable_since+.4, feedback=feedback,
                        cell_index=self.index, offset_xy_m=self._offsets[self.index], reason=str(reason)):
            self.status, self.message = 'saving', f'Recording {outcome} coverage at cell {self.index+1}/{self.total}'

    def _capture_measurement(self, cloud):
        """Pair this acquisition with a fresh SDK sample taken after its read.

        Instantaneous feedback age can qualify as fresh while still preceding a
        newer camera frame. Wait for its matching SDK sample, preserving the
        acquisition identity and quiet window instead of rejecting persistence.
        """
        feedback = self.feedback()
        if not feedback.get('fresh') or not feedback.get('stationary') or not self.ready():
            self._quiet_pose = self._stable_since = None
            self._last_acquisition = None
            self.message = 'Waiting for fresh stationary SDK feedback before recording the native acquisition'
            return None
        stamp = feedback.get('measured_at')
        if stamp is None or not np.isfinite(stamp):
            raise ValueError('Scan capture requires a finite SDK feedback timestamp')
        if stamp < cloud['read_started']:
            self._last_acquisition = None
            self.message = 'Waiting for measured SDK feedback after the native camera acquisition'
            return None
        pose = transform(self.tcp_pose())
        if (self._quiet_pose is None or self._stable_since is None
                or np.linalg.norm(pose[:3, 3]-self._quiet_pose[:3, 3]) > .0003
                or Rotation.from_matrix(pose[:3, :3] @ self._quiet_pose[:3, :3].T).magnitude() > .003):
            self._quiet_pose = self._stable_since = None
            self._last_acquisition = None
            self.message = 'Waiting for continuous measured pose stability before recording'
            return None
        if not self._capture_grid_aligned(pose, correct=False):
            self._last_acquisition = None
            return None
        if Rotation.from_matrix(pose[:3, :3] @ self.view_pose[:3, :3].T).magnitude() > .005:
            raise ValueError('Scan capture exceeded the 0.005 rad fixed-view orientation tolerance')
        return pose, {**feedback, 'stable_since': self._stable_since}

    def _queue_finish(self, status, message):
        self.status, self.message = status, message
        if not self._archive_started:
            return  # A failed approach must not relabel the preceding retained scan.
        self._finish_pending = (status, message)
        if self.future is None:
            self._start_finish()

    def _start_finish(self):
        status, message = self._finish_pending
        self._finish_pending = None
        self._submit('finish', self.archive.finish, status=status, message=message)

    def cancel(self, *, hold=True, reason='Scan cancelled; accepted points retained'):
        was_active = self.busy and self.status not in self.terminal
        if self.busy:
            # Finishing a complete scan also releases motion ownership. A late
            # export failure must never interrupt a subsequent cooldown/return.
            self._stop_hold_done = True
        if was_active:
            self._queue_finish('cancelled', reason)
            if hold:
                try:
                    self.hold()
                except Exception as error:
                    self.error = f'{reason}; measured Hold unavailable: {error}'
        return self.state()

    def abort_for_shutdown(self):
        return self.cancel(hold=False, reason='Scan cancelled; shutdown owns return to rest')

    def abort_for_cooldown(self):
        return self.cancel(hold=False, reason='Scan cancelled; cooldown owns return to rest')

    def _fail(self, error):
        self.error = str(error)
        self._queue_finish('error', str(error))
        self._failure_hold()

    def _failure_hold(self):
        if self._stop_hold_done:
            return
        self._stop_hold_done = True
        try:
            self.hold()
        except Exception as hold_error:
            self.error += f'; measured Hold unavailable: {hold_error}'

    def _persistence_failed(self, error, *, hold=True):
        """A failed disk operation is terminal; never recursively retry finishing."""
        self.future = self._pending_kind = self._finish_pending = None
        self.status = 'error'
        self.error = self.message = f'Scan persistence failed: {error}'
        if hold:
            self._failure_hold()

    def dispatch(self, request):
        action = request['action']
        if action == 'scan_start':
            return self.start(request)
        if action == 'scan_resume':
            return self.resume()
        if action == 'scan_cancel':
            return self.cancel()
        if action == 'scan_state':
            return self.state()
        if self.busy:
            raise RuntimeError('Wait for scan persistence before clearing or exporting')
        if action == 'scan_clear':
            self._submit('clear', self.archive.clear)
        elif action == 'scan_export':
            if not self._saved.get('points') and not self._saved.get('cells_visited'):
                raise ValueError('No captured scan to export')
            return self._saved.get('export_paths', {})
        else:
            raise ValueError('Unknown scan action')
        return self.state()

    def tick(self, camera_state=None):
        if self._closed:
            return
        try:
            if self.future is not None:
                if not self.future.done():
                    return  # Filesystem work never blocks the service/motor tick.
                kind, future = self._pending_kind, self.future
                self.future = self._pending_kind = None
                try:
                    self._saved = future.result()
                except Exception as error:
                    self._persistence_failed(error)
                    return
                if self._finish_pending is not None:
                    self._start_finish()
                    return
                if kind in ('finish', 'clear'):
                    self._archive_started = False
                    if kind == 'clear':
                        self.status, self.message, self.error = 'idle', 'Scan display cleared; archived sessions retained', None
                    return
                if kind in ('append', 'record_cell'):
                    self._grid_approach = False
                    self.index = int(self._saved.get('next_index', self.index+1))
                    self._depth_moves = 0
                    self._alignment_moves = 0
                    if self.index >= self.total:
                        self._queue_finish('complete', 'Full grid complete; accepted points and rejected/no-return coverage retained')
                        return
                    self._begin_settle()
                elif kind in ('begin', 'resume'):
                    self._grid_approach = True
                    self.index = int(self._saved.get('next_index', self.index))
                    self._resuming = False
                    if self.index >= self.total:
                        self._queue_finish('complete', 'Full grid is durable; completing retained exports')
                        return
                    self._begin_settle()
                return
            if self.status in self.terminal:
                return
            state = camera_state if camera_state is not None else self.camera.state(include_frame=False)
            if self._camera_identity(state) != self._identity or not state.get('enabled') or state.get('error'):
                raise ValueError(state.get('error') or 'Camera stopped or calibration identity changed during scan')
            if self.status == 'preflighting':
                if self._preflight_tick():
                    self._begin_settle('baseline_settling')
                return
            if self.status in ('moving', 'positioning'):
                if self._endpoint():
                    if self.status == 'positioning':
                        self._begin_baseline()
                    else:
                        self.hold()
                        self._begin_settle(self._after_move_status, capture_arrival=True)
                return
            if self.clock()-self._wait_started > 15.:
                raise ValueError('Scan requires quiet, pose-stable feedback and a fresh stripe within 15 s')
            if self.status == 'settling' and not self._capture_arrival:
                # Quiet dwell qualifies a CAPTURE after arrival. It does not
                # need to be repeated before leaving a measured stationary
                # Hold for the next already-preflighted grid waypoint.
                actual = self._optical_offset(self.tcp_pose())
                if np.linalg.norm(np.asarray(self._offsets[self.index])-actual[:2]) > .0005:
                    cloud = (state.get('laser') or {}).get('cloud')
                    exposure = ((state.get('laser') or {}).get('pointcloud_status') or {}).get('auto_exposure_status') or {}
                    if (not isinstance(cloud, dict) or cloud.get('generation') != state.get('generation')
                            or exposure.get('state') == 'settling'):
                        self.message = 'Waiting for a fresh native stripe before the next planned XY waypoint'
                        return
                    self._observation(state)  # Never authorize motion from stale/mismatched cloud data.
                    feedback = self.feedback()
                    if not feedback.get('fresh') or not feedback.get('stationary') or not self.ready():
                        self._quiet_pose = self._stable_since = None
                        self.message = 'Waiting for fresh stationary SDK feedback before the next XY waypoint'
                        return
                    self._position_xy()
                    return
            if not self._quiet():
                return
            cloud = (state.get('laser') or {}).get('cloud')
            exposure = ((state.get('laser') or {}).get('pointcloud_status') or {}).get('auto_exposure_status') or {}
            if (not isinstance(cloud, dict)
                    or cloud.get('generation') != state.get('generation')
                    or exposure.get('state') == 'settling'):
                self.message = 'Waiting for a fresh native stripe after camera controls settle'
                return
            acquired = cloud.get('read_started')
            if type(acquired) in (int, float) and acquired < self._stable_since+.4:
                return
            # A newly acquired native cloud is mandatory after the quiet dwell.
            cloud, _ = self._observation(state)
            if cloud['read_started'] < self._stable_since+.4:
                return
            acquisition = (cloud.get('generation'), cloud.get('sequence'))
            if acquisition == self._last_acquisition:
                return
            self._last_acquisition = acquisition
            pose = transform(self.tcp_pose())
            orientation_error = Rotation.from_matrix(pose[:3, :3] @ self.view_pose[:3, :3].T).magnitude()
            if orientation_error > .005:
                raise ValueError('Scan capture exceeded the 0.005 rad fixed-view orientation tolerance')
            if self.status != 'baseline_settling' and not self._capture_grid_aligned(pose):
                return
            native = np.asarray(cloud.get('xyz_m'), float)
            if native.size == 0 and self.status != 'baseline_settling':
                self.quality = {'good': False, 'stripe_points': 0, 'object_points': 0, 'message': 'No native laser returns'}
                self._save_rejected_cell(cloud, pose, 'No native laser returns', outcome='no_return')
                return
            try:
                segment = segment_object(cloud, self._previous_centroid(pose))
            except ValueError as error:
                automatic = ((state.get('laser') or {}).get('pointcloud_status') or {}).get('auto_exposure') or {}
                if (self.status == 'baseline_settling' and automatic.get('enabled') is True
                        and (native.size == 0 or self._quality_rejection(error))):
                    self.quality = {'good': False, 'stripe_points': len(native), 'message': str(error)}
                    self.message = f'Waiting for smart stripe exposure at the scan baseline: {error}'
                    return
                if self.status == 'baseline_settling' or not self._quality_rejection(error):
                    raise
                self.quality = {'good': False, 'stripe_points': len(native), 'message': str(error)}
                self._save_rejected_cell(cloud, pose, error)
                return
            self._update_segment(segment, pose)
            if self.status == 'baseline_settling':
                # The grid center stays at the original measured Tube optical
                # XY. Correct only Z while the object is actually visible at
                # that center, before going to a potentially empty corner.
                if abs(segment['centroid_camera_m'][2]-.140) > .001:
                    self._depth_moves += 1
                    if self._depth_moves > 12:
                        raise ValueError('Baseline object depth did not converge to 140 mm within bounded corrections')
                    self._move(tracking_target(pose, self._handeye, segment, settings=self.settings),
                               after='baseline_settling')
                    self.message = 'Aligning measured center depth to 140 mm before full-grid preflight'
                    return
                if not self._preflight_complete:
                    if not self._preflight_tick():
                        self.status = 'preflighting'
                        return
                self._adjusted_baseline_pose = pose.copy()
                metadata = {'settings': asdict(self.settings), 'baseline_pose': self.return_pose.tolist(),
                            'view_pose': self.view_pose.tolist(), 'optical_roll_deg': self.optical_roll_deg,
                            'capture_rotation': self.view_pose[:3, :3].tolist(),
                            'handeye': state['handeye'], 'identity': list(self._identity),
                            'model_fingerprint': self.model_fingerprint,
                            'adjusted_baseline_pose': self._adjusted_baseline_pose.tolist(),
                            'safe_depth_offset_m': float(self._safe_depth_offset)}
                if self._resuming:
                    original = self._saved['metadata']
                    checks = {key: original[key] for key in ('settings', 'baseline_pose', 'handeye', 'model_fingerprint')}
                    if not self._submit('resume', self.archive.resume, checks,
                                        acquisition_session_id=state['acquisition_session_id']):
                        return
                elif not self._submit('begin', self.archive.begin, metadata):
                    return
                self._archive_started = True
                self.status = 'saving'
                return
            if not self._at_offset(segment):
                return
            measured = self._capture_measurement(cloud)
            if measured is None:
                return
            pose, feedback = measured
            self._update_segment(segment, pose)
            if not self._submit('append', self.archive.append, copy.deepcopy(cloud), pose, self._handeye,
                                copy.deepcopy(segment), now=self.clock(), not_before_monotonic=self._stable_since+.4,
                                feedback=feedback, cell_index=self.index, offset_xy_m=self._offsets[self.index]):
                return
            self.status, self.message = 'saving', f'Saving stable snapshot {self.index+1}/{self.total}'
        except Exception as error:
            self._fail(error)

    def close(self):
        self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=False)
        if self.future is not None:
            try:
                self._saved = self.future.result()
            except Exception as error:
                self._persistence_failed(error, hold=False)
                return
            self.future = self._pending_kind = None
        if self._finish_pending:
            status, message = self._finish_pending
            self._finish_pending = None
            try:
                self.archive.finish(status=status, message=message)
                self._saved = self.archive.state(include_points=True)
            except Exception as error:
                self._persistence_failed(error, hold=False)
