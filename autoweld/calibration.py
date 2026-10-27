"""Robot calibration geometry/orchestration; vision remains in Weldfinder.

Transforms use column vectors and meters. Camera target observations supplied by
Weldfinder explicitly use millimeters and are converted only at this boundary.
"""

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.checkerboard_hunt import CheckerboardHunt
from autoweld.handeye_observation import HandeyeAssociation


def transform(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Expected a finite rigid 4x4 transform')
    return value.copy()


def cone_plan(baseline, *, count=25, camera_offset_m=(0, 0, 0)):
    """Uniform edge views on a compact 15 degree full-opening, 150 mm cone.

    Camera origins stay in the baseline image plane. Every optical +Z points
    toward one fixed apex, using only the minimum tilt from the baseline frame.
    The baseline is an aiming/return pose, never one of the edge captures.
    """
    baseline = transform(baseline)
    if not isinstance(count, int) or count < 20 or count > 100:
        raise ValueError('Calibration requires 20–100 planned views')
    offset = np.asarray(camera_offset_m, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Camera offset must be three finite meters')
    rotation, start = baseline[:3, :3], baseline[:3, 3]
    camera_start = start + rotation @ offset
    distance = .150
    half_angle = np.deg2rad(7.5)
    radius = distance * np.tan(half_angle)
    apex = camera_start + rotation[:, 2] * distance
    poses, ring = [], []
    for azimuth in np.linspace(0., 2*np.pi, count, endpoint=False):
        camera_position = camera_start + rotation @ np.array([radius*np.cos(azimuth),
                                                              radius*np.sin(azimuth), 0.])
        # The tilt axis is perpendicular to baseline +Z and the radial offset.
        # No extra roll, depth or radius modulation is applied.
        tilt = Rotation.from_rotvec(half_angle*np.array([np.sin(azimuth), -np.cos(azimuth), 0.]))
        orientation = rotation @ tilt.as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = orientation
        pose[:3, 3] = camera_position - orientation @ offset
        poses.append(pose)
        ring.append(camera_position)
    ring = np.asarray(ring)
    geometry = {'axis_start_m': camera_start.tolist(), 'apex_m': apex.tolist(),
                'cone_ring_m': ring.tolist(),
                'cone_ring_closed_m': np.vstack((ring, ring[0])).tolist(),
                'capture_tcp_origins_m': [pose[:3, 3].tolist() for pose in poses],
                'opening_angle_deg': 15., 'half_angle_deg': 7.5, 'distance_m': distance,
                'radius_m': float(radius), 'count': count,
                'camera_offset_m': offset.tolist(), 'coordinate_frame': 'robot_base',
                'camera_origin_assumption': 'TCP origin' if np.allclose(offset, 0) else 'configured mount offset'}
    return poses, geometry


def solve_handeye(samples, *, minimum_views=20):
    """Eye-in-hand AX=XB solve with excitation and stationary-board checks.

    Returns tcp_from_camera in meters, never nominal CAD-derived extrinsics.
    No camera SDK, OpenCV, robot connection or model is constructed.
    """
    if len(samples) < minimum_views:
        raise ValueError(f'Hand-eye calibration requires {minimum_views} accepted paired views')
    identities = {(s['source_id'], s['calibration_id']) for s in samples}
    if len(identities) != 1:
        raise ValueError('Hand-eye samples must share one camera and intrinsic calibration')
    gs = [transform(s['base_from_tcp']) for s in samples]
    cs = [transform(s['camera_from_target_m']) for s in samples]
    pairs = [(np.linalg.inv(gs[j]) @ gs[i], cs[j] @ np.linalg.inv(cs[i]))
             for i in range(len(samples)) for j in range(i)]
    va = np.array([Rotation.from_matrix(a[:3, :3]).as_rotvec() for a, b in pairs])
    vb = np.array([Rotation.from_matrix(b[:3, :3]).as_rotvec() for a, b in pairs])
    excitation = np.linalg.svd(va, compute_uv=False)
    if excitation[1] / np.sqrt(len(pairs)) < np.deg2rad(2):
        raise ValueError('Hand-eye poses need rotations around at least two distinct axes')
    rotation, _ = Rotation.align_vectors(va, vb)
    rx = rotation.as_matrix()
    matrix = np.vstack([a[:3, :3] - np.eye(3) for a, b in pairs])
    rhs = np.concatenate([rx @ b[:3, 3] - a[:3, 3] for a, b in pairs])
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('Hand-eye translation is underconstrained')
    translation = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = rx, translation
    boards = [g @ x @ c for g, c in zip(gs, cs)]
    positions = np.array([b[:3, 3] for b in boards])
    mean_rotation = Rotation.from_matrix(np.array([b[:3, :3] for b in boards])).mean()
    translation_rms = float(np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1))))
    rotation_rms = float(np.sqrt(np.mean((mean_rotation.inv() * Rotation.from_matrix(
        np.array([b[:3, :3] for b in boards]))).magnitude() ** 2)))
    if translation_rms > .003 or rotation_rms > np.deg2rad(1.5):
        raise ValueError(f'Hand-eye stationary-board residual too high: {translation_rms*1000:.2f}mm, '
                         f'{np.rad2deg(rotation_rms):.2f}deg; check pose timing and board corner orientation')
    return {'tcp_from_camera': x.tolist(), 'units': 'm', 'coordinate_frame': 'tcp',
            'views': len(samples), 'source_id': samples[0]['source_id'],
            'calibration_id': samples[0]['calibration_id'],
            'translation_rms_mm': translation_rms * 1000,
            'rotation_rms_deg': float(np.rad2deg(rotation_rms)),
            'board_origin_warning': 'Use the same physical origin and corner orientation in every image'}


class CalibrationCoordinator:
    """Tick-driven capture sequence. Camera work never blocks the motor tick."""

    def __init__(self, camera, tcp_pose, move_pose, ready, hold, workspace, *,
                 validate_pose=None, camera_offset_m=(0, 0, 0)):
        self.camera, self.tcp_pose, self.move_pose = camera, tcp_pose, move_pose
        self.ready, self.hold = ready, hold
        self.workspace = Path(workspace)
        self.validate_pose = validate_pose
        self.camera_offset_m = camera_offset_m
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-calibration')
        self.future = None
        self.pending_kind = None
        self.kind = 'intrinsic'
        self.status, self.message = 'idle', 'Turn on the camera; auto calibration hunts for a full checkerboard in 5 mm steps'
        self.poses, self.geometry = [], None
        self.index = 0
        self.samples = []
        self.handeye_association = HandeyeAssociation()
        self.settled_at = None
        self.motion_started = None
        self.capture_pose = None
        self.capture_pose_monotonic = None
        self.cancelled = False
        self.last_error = None
        self.last_result = None
        self.hunt = None
        self._hunter = None
        self._hunt_moving = False
        self._hunt_last_frame = None
        self._hunt_last_fresh = None
        self.return_pose = None
        self.view_count = 25
        self.session_identity = None
        self.capture_not_before = None

    def state(self):
        return copy.deepcopy(dict(status=self.status, message=self.message, kind=self.kind,
                                  index=self.index, total=len(self.poses), geometry=self.geometry,
                                  accepted_handeye=len(self.samples), result=self.last_result,
                                  error=self.last_error, hunt=self.hunt))

    def abort_for_cooldown(self):
        """Discard calibration work while the controller owns thermal stopping.

        Calling the ordinary Hold callback here would replace the controller's
        return-to-rest path, so thermal cancellation must not command the arm.
        """
        self.cancelled = True
        self._hunter, self._hunt_moving = None, False
        if self.hunt:
            self.hunt['motion_active'] = False
        self.status = 'cancelled'
        self.message = 'Calibration cancelled for motor cooldown; controller is returning to rest'
        return self.state()

    def abort_for_shutdown(self):
        """Release sequence ownership without replacing the shutdown stop."""
        self.cancelled = True
        self._hunter, self._hunt_moving = None, False
        if self.hunt:
            self.hunt['motion_active'] = False
        self.status, self.message = 'cancelled', 'Calibration cancelled; shutdown owns return to rest'
        return self.state()

    @staticmethod
    def _frame_time(state):
        # Acquisition start, not processing completion: a slow detector must
        # never make an image from the preceding robot pose appear current.
        stamp = state.get('read_started')
        if stamp is None:
            stamp = state.get('timestamp_monotonic')
        return float(stamp) if stamp is not None and np.isfinite(stamp) else None

    def _fresh_settled_frame(self, state, *, delay=.4):
        stamp = self._frame_time(state)
        return bool(state.get('fresh') and stamp is not None
                    and 0 <= time.monotonic()-stamp <= 1.
                    and self.settled_at is not None and stamp >= self.settled_at+delay)

    def _check_session_frame(self, state):
        identity = (state.get('source_id'), state.get('generation'))
        if any(value is None for value in identity):
            raise ValueError('Checkerboard frame needs a camera source and acquisition generation')
        if self.session_identity is None:
            self.session_identity = identity
        elif identity != self.session_identity:
            raise ValueError('Camera source or acquisition generation changed during calibration')

    def _begin_hunt(self, phase):
        now = time.monotonic()
        self._hunter = CheckerboardHunt(transform(self.tcp_pose()), started_at=now)
        self._hunt_moving, self._hunt_last_frame = False, None
        self._hunt_last_fresh = now
        self.hunt = {'phase': phase, 'steps': 0, 'max_steps': 40, 'step_mm': 5.,
                     'corner_count': 0, 'centroid_px': None, 'offset_px': None,
                     'reacquiring': False, 'motion_active': False}
        self.settled_at = now
        self.status, self.message = 'hunting', f'Finding a full checkerboard at the {phase} pose'

    def _tick_hunt(self):
        now = time.monotonic()
        if now-self._hunter.started_at >= self._hunter.timeout_s:
            raise ValueError('Checkerboard search exhausted: timeout')
        if self._hunt_moving:
            if not self.ready():
                if now-self.motion_started > 30:
                    raise ValueError('Checkerboard hunt motion timed out')
                return
            self._hunt_moving = False
            self.hunt['motion_active'] = False
            self.settled_at = now
            self._hunt_last_fresh = now
        elif not self.ready():
            raise ValueError('Robot moved unexpectedly during checkerboard search')
        state = self.camera.state(include_frame=False)
        if state.get('enabled') is False or state.get('error'):
            raise ValueError(state.get('error') or 'Camera stopped during checkerboard search')
        if not state.get('fresh'):
            if now-self._hunt_last_fresh > 2:
                raise ValueError('No fresh camera frames during checkerboard search')
            return
        self._hunt_last_fresh = now
        if not self._fresh_settled_frame(state):
            # An old acquisition can remain "fresh" for a second. It cannot
            # authorize another move after the arm changed position.
            if now-self.settled_at > 2:
                raise ValueError('No new image acquired after the checkerboard hunt move')
            return
        stamp = self._frame_time(state)
        key = (state.get('source_id'), state.get('generation'), state.get('sequence'), stamp)
        if key == self._hunt_last_frame:
            if now-self.settled_at > 2:
                raise ValueError('Checkerboard search camera frame stopped advancing')
            return
        self._hunt_last_frame = key
        self._check_session_frame(state)
        board = state.get('board') or {}
        proposal = self._hunter.next_step(transform(self.tcp_pose()), board.get('corners', []),
            state.get('resolution'), full_board=bool(board.get('found')), now=now)
        phase = self.hunt['phase']
        self.hunt.update(proposal, phase=phase, motion_active=False)
        if proposal['status'] == 'exhausted':
            raise ValueError(f"Checkerboard search exhausted: {proposal['reason']}")
        if proposal['status'] == 'complete':
            self._hunter = None
            if phase == 'baseline':
                baseline = transform(self.tcp_pose())
                self.poses, self.geometry = cone_plan(baseline, count=self.view_count,
                    camera_offset_m=self.camera_offset_m)
                self.geometry['requested_baseline'] = self.return_pose.tolist()
                self.geometry['baseline_after_hunt'] = baseline.tolist()
                if self.validate_pose:
                    for pose in self.poses:
                        self.validate_pose(pose)
                self.move_pose(self.poses[0], automatic=True)
                self.status, self.message = 'moving', 'Moving to calibration edge view 1'
                self.motion_started = now
                self.settled_at = None
                return
            self.status, self.message = 'settling', 'Full checkerboard found; waiting for a fresh capture'
            self.settled_at = now
            return
        target = transform(proposal['target_pose'])
        if self.validate_pose:
            self.validate_pose(target)
        # The regular JSON service motion callback preflights the entire path
        # through the sole SDK controller; hunting does not bypass its guards.
        self.move_pose(target, automatic=True)
        self._hunt_moving = True
        self.hunt['motion_active'] = True
        self.motion_started = time.monotonic()
        self.message = f"Checkerboard hunt: 5 mm step {proposal['steps']} ({proposal['reason']})"

    def _submit(self, kind, function):
        if self.future is not None:
            raise RuntimeError('Wait for the current calibration operation')
        self.pending_kind = kind
        self.future = self.executor.submit(function)

    def _capture(self):
        state = self.camera.state(include_frame=False)
        if not state['fresh'] or not state['board']['found']:
            raise ValueError('A fresh full checkerboard is required')
        stamp = self._frame_time(state)
        if stamp is not None and self.capture_not_before is not None and stamp < self.capture_not_before:
            raise ValueError('Wait for an image acquired after the arm settled')
        if self.kind == 'intrinsic':
            return self.camera.dispatch({'action': 'calibration_capture', 'kind': 'intrinsic',
                                         'not_before_monotonic': self.capture_not_before,
                                         'metadata': {'base_from_tcp': self.capture_pose.tolist()}})
        pose = self.camera.target_pose(max_age_s=1.)
        if self.capture_not_before is not None and pose['read_started'] < self.capture_not_before:
            raise ValueError('Wait for an image acquired after the arm settled')
        settings = state['board'].get('settings', {})
        grid = pose.get('grid', {})
        if any(grid.get(key) != settings.get(key) for key in ('columns', 'rows', 'square_mm')):
            raise ValueError('Native hand-eye grid does not match current checkerboard configuration')
        if any(pose.get(key) != state.get(key) for key in ('source_id', 'generation', 'calibration_id')):
            raise ValueError('Native hand-eye source/calibration/configuration changed during capture')
        sample = self.handeye_association.prepare(pose, self.capture_pose)
        sample['robot_capture'] = dict(pre_base_from_tcp=self.capture_pose.tolist(),
                                      pre_pose_monotonic=self.capture_pose_monotonic,
                                      not_before_monotonic=self.capture_not_before,
                                      settled_at_monotonic=self.settled_at)
        return sample

    def dispatch(self, request):
        action = request['action']
        kind = request.get('kind', self.kind)
        if kind not in ('intrinsic', 'handeye'):
            raise ValueError('Calibration kind must be intrinsic or handeye')
        if action == 'calibration_state':
            return self.state()
        if action == 'calibration_cancel':
            self.cancelled = True
            self.hold()
            self._hunter, self._hunt_moving = None, False
            if self.hunt:
                self.hunt['motion_active'] = False
            self.status, self.message = 'cancelled', 'Calibration cancelled; holding current pose'
            return self.state()
        if self.future is not None or self.status in ('starting', 'hunting', 'moving', 'settling',
                'capturing', 'returning', 'capturing_manual', 'solving'):
            raise RuntimeError('Cancel the active calibration before starting another operation')
        self.kind, self.cancelled = kind, False
        self.last_error = None
        if action in ('calibration_start', 'calibration_auto'):
            if not self.ready():
                raise ValueError('Stop the robot before starting calibration')
            camera = self.camera.state(include_frame=False)
            if not camera['fresh'] or camera.get('enabled') is False:
                raise ValueError('Baseline needs a fresh camera stream before checkerboard hunting')
            if kind == 'handeye' and camera['intrinsics'] is None:
                raise ValueError('Solve matching camera intrinsics before hand-eye calibration')
            self.return_pose = transform(self.tcp_pose())
            self.view_count = int(request.get('views', 25))
            # Validate view count/geometry before changing the camera session,
            # but center the actual cone only after baseline acquisition.
            cone_plan(self.return_pose, count=self.view_count, camera_offset_m=self.camera_offset_m)
            self.poses, self.geometry = [], None
            self.index, self.samples = 0, []
            self.handeye_association = HandeyeAssociation()
            self.session_identity = None
            self.hunt, self._hunter = None, None
            self.camera._handeye_count = 0
            self.status, self.message = 'starting', 'Preparing calibration session before checkerboard search'
            if kind == 'intrinsic':
                self._submit('reset', lambda: self.camera.dispatch({'action': 'calibration_reset'}))
            else:
                self._begin_hunt('baseline')
        elif action == 'calibration_reset':
            self.samples = []
            self.handeye_association = HandeyeAssociation()
            self.camera._handeye_count = 0
            self.poses, self.geometry, self.index = [], None, 0
            self.hunt, self._hunter = None, None
            self.session_identity = None
            self.status, self.message = 'idle', 'Calibration captures cleared'
            if kind == 'intrinsic':
                self._submit('reset_manual', lambda: self.camera.dispatch({'action': 'calibration_reset'}))
        elif action == 'calibration_capture':
            if not self.ready():
                raise ValueError('Stop the robot before capturing a calibration image')
            self.capture_pose = transform(self.tcp_pose())
            self.capture_pose_monotonic = time.monotonic()
            self.settled_at = time.monotonic() - .5
            self.capture_not_before = self.settled_at
            self.status, self.message = 'capturing_manual', 'Capturing a native checkerboard image'
            self._submit('capture_manual', self._capture)
        elif action == 'calibration_solve':
            self._solve()
        else:
            raise ValueError(f'Unknown calibration action: {action}')
        return self.state()

    def _solve(self):
        self.status, self.message = 'solving', 'Fitting calibration and checking measured residuals'
        samples = copy.deepcopy(self.samples)
        self._submit('solve', (lambda: self.camera.dispatch({'action': 'calibration_solve'}))
                     if self.kind == 'intrinsic' else lambda: solve_handeye(samples))

    def _save_handeye(self, result):
        validator = getattr(self.camera, 'validate_handeye', None)
        if validator is not None:
            result = validator(result)
        target = self.workspace / 'handeye.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        # Result and the observations supporting it are one atomic document.
        # A failed write leaves the previous installed archive intact.
        payload = dict(result, schema_version=2, paired_samples=copy.deepcopy(self.samples))
        encoded = json.dumps(payload, indent=2, allow_nan=False)
        with temporary.open('w') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        installer = getattr(self.camera, 'install_handeye', None)
        if installer is not None:
            installer(result)
        else:
            self.camera._handeye_result = result

    def tick(self):
        try:
            self._tick()
        except Exception as exc:
            self.last_error = str(exc)
            self.status, self.message = 'error', str(exc)
            self._hunter, self._hunt_moving = None, False
            if self.hunt:
                self.hunt['motion_active'] = False
            self.hold()

    def _tick(self):
        if self.future is not None:
            if not self.future.done():
                return
            future, pending = self.future, self.pending_kind
            self.future = self.pending_kind = None
            if self.cancelled:
                return
            try:
                result = future.result()
            except (ValueError, RuntimeError):
                state = self.camera.state(include_frame=False)
                if (pending == 'capture' and not state.get('board', {}).get('found')
                        and state.get('enabled') is not False and not state.get('error') and self.ready()):
                    self._begin_hunt('capture')
                    return
                raise
            if pending == 'solve':
                self.last_result = result
                if self.kind == 'handeye':
                    self._save_handeye(result)
                self.status, self.message = 'complete', 'Calibration saved after quality checks'
                return
            if pending in ('reset', 'reset_manual'):
                if pending == 'reset_manual':
                    self.status = 'idle'
                    return
                self._begin_hunt('baseline')
            elif pending in ('capture', 'capture_manual'):
                after = transform(self.tcp_pose())
                if (not self.ready() or np.linalg.norm(after[:3, 3] - self.capture_pose[:3, 3]) > .0005
                        or Rotation.from_matrix(after[:3, :3] @ self.capture_pose[:3, :3].T).magnitude() > .005):
                    raise ValueError('Robot moved while the checkerboard image was being measured')
                if self.kind == 'handeye':
                    result['robot_capture'].update(post_base_from_tcp=after.tolist(),
                        post_pose_monotonic=time.monotonic(), post_ready=True,
                        translation_delta_m=float(np.linalg.norm(after[:3, 3]-self.capture_pose[:3, 3])),
                        rotation_delta_rad=float(Rotation.from_matrix(after[:3, :3] @ self.capture_pose[:3, :3].T).magnitude()))
                    self.handeye_association.commit(result)
                    self.samples.append(result)
                    self.camera._handeye_count = len(self.samples)
                if pending == 'capture_manual':
                    self.status, self.message = 'idle', 'Native calibration capture accepted'
                    return
                self.index += 1
                if self.index >= len(self.poses):
                    self.move_pose(self.return_pose if self.return_pose is not None else self.poses[0], automatic=True)
                    self.status, self.message = 'returning', 'Returning to the calibration baseline'
                    self.motion_started = time.monotonic()
                else:
                    self.move_pose(self.poses[self.index], automatic=True)
                    self.status, self.message = 'moving', f'Moving to calibration view {self.index + 1}'
                    self.motion_started = time.monotonic()
                self.settled_at = None
        if self.cancelled:
            return
        if self.status in ('moving', 'returning'):
            if self.ready():
                if self.status == 'returning':
                    self._solve()
                    return
                self.status = 'settling'
                self.settled_at = time.monotonic()
            elif time.monotonic() - self.motion_started > 30:
                raise ValueError('Calibration motion timed out; holding current pose')
        if self.status == 'hunting':
            self._tick_hunt()
            return
        if self.status == 'settling':
            if not self.ready():
                self.settled_at = time.monotonic()
                return
            state = self.camera.state(include_frame=False)
            if time.monotonic() - self.settled_at > 12:
                raise ValueError('No fresh checkerboard observation at calibration pose')
            if self._fresh_settled_frame(state):
                self._check_session_frame(state)
                if not state['board']['found']:
                    self._begin_hunt('capture')
                    return
                self.capture_pose = transform(self.tcp_pose())
                self.capture_pose_monotonic = time.monotonic()
                self.capture_not_before = self.settled_at+.4
                self.status, self.message = 'capturing', f'Capturing view {self.index + 1}'
                self._submit('capture', self._capture)

    def close(self):
        self.cancelled = True
        self.executor.shutdown(wait=False, cancel_futures=True)
