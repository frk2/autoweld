"""Service-owned laser sequence and camera-to-J6 geometry, without device I/O.

Weldfinder owns detection, smart exposure, native captures, plane fitting and
noise filtering. Only the supplied service callbacks may request robot motion.
The installed nominal motion TCP and robot model are never changed here.
"""

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.calibration import cone_plan, transform
from autoweld.checkerboard_hunt import CheckerboardHunt


def j6_from_camera(flange_from_nominal_tcp, tcp_from_camera):
    """Compose nominal camera TCP with measured hand-eye exactly once."""
    if isinstance(tcp_from_camera, dict):
        if tcp_from_camera.get('units') != 'm' or tcp_from_camera.get('coordinate_frame') != 'tcp':
            raise ValueError('Measured hand-eye must use meters in the nominal TCP frame')
        tcp_from_camera = tcp_from_camera['tcp_from_camera']
    return transform(flange_from_nominal_tcp) @ transform(tcp_from_camera)


def camera_points_to_j6(points_camera_m, flange_from_nominal_tcp, tcp_from_camera):
    points = np.asarray(points_camera_m, float)
    if points.size == 0:
        points = np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Camera points must be finite Nx3 meters')
    pose = j6_from_camera(flange_from_nominal_tcp, tcp_from_camera)
    return points @ pose[:3, :3].T + pose[:3, 3]


def transform_cloud(cloud, base_from_tcp, handeye, flange_from_nominal_tcp, *,
                    now=None, max_age_s=1., max_points=5000):
    """JSON camera XYZ meters → J6-center and world points, with provenance.

    ``base_from_tcp`` refers to the unchanged nominal camera motion TCP.
    Consequently base_from_J6 = base_from_tcp @ inv(flange_from_nominal_tcp).
    The caller authorizes association with fresh, idle SDK feedback; this pure
    helper validates acquisition identity/age and never fetches robot feedback.
    """
    if not isinstance(cloud, dict) or cloud.get('coordinate_frame') != 'camera':
        raise ValueError('Expected a calibrated camera-frame pointcloud')
    if cloud.get('xyz_m_units', cloud.get('units')) != 'm':
        raise ValueError('Pointcloud xyz_m requires explicit meter units')
    if not isinstance(handeye, dict):
        raise ValueError('An installed measured hand-eye result is required')
    optical_id = cloud.get('optical_intrinsics_id', cloud.get('calibration_id'))
    handeye_optical = handeye.get('optical_intrinsics_id', handeye.get('calibration_id'))
    if cloud.get('source_id') != handeye.get('source_id') or optical_id != handeye_optical:
        raise ValueError('Pointcloud and hand-eye optical identities differ')
    if not optical_id or not cloud.get('source_id'):
        raise ValueError('Pointcloud requires camera and optical intrinsic identity')
    if isinstance(max_points, bool) or not isinstance(max_points, int) or not 1 <= max_points <= 20000:
        raise ValueError('Pointcloud limit must be an integer in [1,20000]')
    if not np.isfinite(max_age_s) or not 0 < max_age_s <= 1.:
        raise ValueError('Pointcloud maximum age must be in (0,1] seconds')
    if now is not None:
        stamp = cloud.get('read_started')
        if stamp is None or not np.isfinite([now, stamp]).all() or not 0 <= now-stamp <= max_age_s:
            raise ValueError('Pointcloud acquisition is stale or in the future')
    points = np.asarray(cloud['xyz_m'], float)
    if points.size == 0:
        points = np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Pointcloud XYZ must be finite Nx3 meters')
    total = len(points)
    stride = max(1, int(np.ceil(total/max_points)))
    points = points[::stride]
    flange_from_tcp = transform(flange_from_nominal_tcp)
    points_j6 = camera_points_to_j6(points, flange_from_tcp, handeye)
    base_from_j6 = None if base_from_tcp is None else transform(base_from_tcp) @ np.linalg.inv(flange_from_tcp)
    points_world = (np.empty((0, 3)) if base_from_j6 is None else
                    points_j6 @ base_from_j6[:3, :3].T + base_from_j6[:3, 3])
    provenance = {key: cloud.get(key) for key in ('source_id', 'calibration_id', 'optical_intrinsics_id',
        'acquisition_session_id', 'sequence', 'generation', 'read_started', 'read_completed')}
    return {**provenance, 'coordinate_frame': 'j6', 'frame_label': 'J6 tool center', 'units': 'm',
            'world_frame': 'robot_base', 'points_j6_m': points_j6.tolist(),
            'points_world_m': points_world.tolist(), 'count': len(points), 'total_count': total,
            'stride': stride, 'noise_filter_owner': 'Weldfinder',
            'status': 'j6_only' if base_from_j6 is None else 'world_registered',
            'base_from_j6': None if base_from_j6 is None else base_from_j6.tolist()}


class LaserCalibrationCoordinator:
    """Tick-driven autonomous discrete poses; camera commands run on one worker."""
    terminal = frozenset(('idle', 'complete', 'cancelled', 'error'))

    def __init__(self, camera, tcp_pose, move_pose, ready, hold, move_saved_pose, *,
                 flange_from_nominal_tcp, preempt=None, can_start=None, validate_pose=None,
                 clock=time.monotonic, executor=None):
        self.camera, self.tcp_pose, self.move_pose = camera, tcp_pose, move_pose
        self.ready, self.hold, self.move_saved_pose = ready, hold, move_saved_pose
        self.flange_from_nominal_tcp = transform(flange_from_nominal_tcp)
        self.preempt, self.can_start, self.validate_pose = preempt, can_start, validate_pose
        self.clock = clock
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-laser')
        self.future = None
        self.pending_kind = None
        self._cleanup = None
        self.status, self.message, self.error = 'idle', 'Laser assumed ON; ready for calibration', None
        self.return_pose, self.geometry, self.hunt = None, None, None
        self.poses, self.rejected = [], []
        self.count = self.index = 0
        self.total = 25
        self.result = None
        self.cancelled = False
        self._identity = None
        self._hunter = None
        self._hunt_moving = False
        self._last_frame = None
        self._settled_at = self._motion_started = None
        self._quiet_pose = self._quiet_wait_started = None
        self._capture_pose = self._capture_cutoff = None
        self._camera_options = {}
        self._tcp_from_camera = np.eye(4)
        self._automatic = False
        self._closed = False
        self._capture_hunt_used = set()
        self._move_goal = None
        self.endpoint_error = None

    @property
    def busy(self):
        return self.status not in self.terminal

    def state(self, camera_state=None):
        laser = (camera_state or {}).get('laser') or {}
        accepted = laser.get('captures', self.count)
        return copy.deepcopy({'status': self.status, 'phase': self.status, 'busy': self.busy,
            'message': self.message, 'error': self.error, 'index': self.index, 'total': self.total,
            'count': accepted, 'captures': accepted, 'progress': min(1., self.index/max(1, self.total)),
            'result': self.result, 'geometry': self.geometry, 'hunt': self.hunt, 'rejected': self.rejected,
            'capture_reacquisition_views': sorted(self._capture_hunt_used),
            'endpoint_error': self.endpoint_error,
            'plane': laser.get('plane'), 'quality': laser.get('quality'),
            'last_attempt': laser.get('last_attempt'), 'detection': laser.get('detection'),
            'auto_exposure': laser.get('auto_exposure'), 'noise': laser.get('noise'),
            'exposure_bracket': laser.get('exposure_bracket'),
            'pointcloud_status': laser.get('pointcloud_status'),
            'cloud': {'coordinate_frame': 'j6', 'units': 'm', 'count': 0,
                      'status': 'Waiting for a fresh camera cloud and quiet SDK pose'}})

    @staticmethod
    def _frame_identity(state):
        optical = state.get('optical_intrinsics_id', state.get('calibration_id'))
        identity = (state.get('source_id'), optical, state.get('acquisition_session_id'))
        if any(value is None or value == '' for value in identity):
            raise ValueError('Laser frame requires source, optical intrinsics and acquisition session identity')
        size = state.get('resolution', state.get('image_size'))
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('Laser frame requires a native image resolution')
        return (*identity, tuple(size))

    def _check_identity(self, state):
        identity = self._frame_identity(state)
        if self._identity is None:
            self._identity = identity
        elif identity != self._identity:
            raise ValueError('Laser camera source, optical intrinsics or acquisition session changed')

    def _stationary(self, pose):
        after = transform(self.tcp_pose())
        if (not self.ready() or np.linalg.norm(after[:3, 3]-pose[:3, 3]) > .0005
                or Rotation.from_matrix(after[:3, :3] @ pose[:3, :3].T).magnitude() > .005):
            raise ValueError('Robot moved during native laser measurement')

    def _endpoint_converged(self, *, timeout_s=30., label='Laser calibration endpoint'):
        """A finished reference and a quiet sample do not prove arrival.

        Keep the SDK's requested endpoint until measured FK converges. The
        resulting diagnostics survive measured Hold, cancellation and errors.
        """
        if self._move_goal is None:
            raise ValueError('Requested nominal TCP endpoint is missing')
        measured = transform(self.tcp_pose())
        position = float(np.linalg.norm(measured[:3, 3]-self._move_goal[:3, 3]))
        rotation = float(Rotation.from_matrix(measured[:3, :3] @ self._move_goal[:3, :3].T).magnitude())
        ready = bool(self.ready())
        elapsed = float(self.clock()-self._motion_started)
        within = position <= .0005 and rotation <= .005
        self.endpoint_error = {'position_mm': position*1000., 'rotation_rad': rotation,
            'rotation_deg': float(np.rad2deg(rotation)), 'position_tolerance_mm': .5,
            'rotation_tolerance_rad': .005, 'within_tolerance': within, 'ready': ready,
            'converged': within and ready, 'elapsed_s': elapsed, 'timeout_s': timeout_s,
            'target_frame': 'nominal_camera_tcp', 'requested_pose': self._move_goal.tolist()}
        if within and ready:
            return True
        if elapsed > timeout_s:
            raise ValueError(f'{label} did not converge within {timeout_s:g} s: '
                f'position error {position*1000.:.3f} mm (limit 0.5 mm), '
                f'rotation error {rotation:.6f} rad (limit 0.005 rad), quiet/idle={ready}')
        return False

    def _begin_quiet_wait(self):
        self._quiet_wait_started = self.clock()
        self._quiet_pose = self._settled_at = None
        self._update_quiet_wait(dwell=0.)

    def _update_quiet_wait(self, *, dwell=.4):
        """Wait before an observation, without treating setup as measurement.

        Instantaneous SDK readiness can oscillate as the final commanded pose
        settles. Reset the acquisition cutoff on any such observation or pose
        drift; only an actual capture uses the strict before/after assertion.
        """
        now = self.clock()
        if now-self._quiet_wait_started > 15.:
            raise ValueError('Robot did not remain quiet and pose-stable before laser observation within 15 s')
        if not self.ready():
            self._quiet_pose = self._settled_at = None
            return False
        current = transform(self.tcp_pose())
        if (self._quiet_pose is None
                or np.linalg.norm(current[:3, 3]-self._quiet_pose[:3, 3]) > .0005
                or Rotation.from_matrix(current[:3, :3] @ self._quiet_pose[:3, :3].T).magnitude() > .005):
            self._quiet_pose, self._settled_at = current, now
        return now-self._settled_at >= dwell

    def _submit(self, kind, request):
        if self.future is not None or self._cleanup is not None and not self._cleanup.done():
            raise RuntimeError('Wait for the previous laser camera operation')
        self.pending_kind = kind
        self.future = self.executor.submit(self.camera.laser_command, copy.deepcopy(request))
        self._operation_started = self.clock()

    def _cleanup_camera(self):
        # Queued after an in-flight capture, so its exposure profile is restored
        # once. The motion intent is cancelled immediately on the main thread.
        if self._cleanup is None or self._cleanup.done():
            self._cleanup = self.executor.submit(self.camera.laser_command, {'action': 'laser_cancel'})

    def _abort(self, reason, *, hold):
        self.cancelled = True
        self._hunter, self._hunt_moving = None, False
        if self.hunt:
            self.hunt['motion_active'] = False
        self.status, self.message = 'cancelled', reason
        if self.future is not None:
            self.future.cancel()
        self._cleanup_camera()
        if hold:
            try:
                self.hold()
            except Exception as error:
                self.error = f'{reason}; measured Hold unavailable: {error}'
                self.message = self.error
        return self.state()

    def cancel(self):
        return self._abort('Laser calibration cancelled; measured Hold requested', hold=True)

    def abort_for_cooldown(self):
        return self._abort('Laser calibration cancelled for cooldown; controller owns return to rest', hold=False)

    def abort_for_shutdown(self):
        return self._abort('Laser calibration cancelled; shutdown owns return to rest', hold=False)

    def _fail(self, error):
        owns_motion = self._automatic or self.status in ('laser_capture', 'laser_check')
        self._abort(str(error), hold=owns_motion)
        self.status, self.error = 'error', self.error or str(error)

    def start(self, *, go_to_calibration_pose=True, count=25, options=None):
        if self.busy or self.future is not None or self._cleanup is not None and not self._cleanup.done():
            raise RuntimeError('Wait for the active laser operation or cancellation cleanup')
        if count != 25:
            raise ValueError('Automatic laser calibration uses exactly 25 cone edge poses')
        state = self.camera.state(include_frame=False)
        if not state.get('enabled') or state.get('error'):
            raise ValueError('Turn on a healthy camera before laser calibration')
        if not state.get('intrinsics'):
            raise ValueError('Installed camera intrinsics required for laser calibration')
        handeye = state.get('handeye')
        if not isinstance(handeye, dict):
            raise ValueError('Installed measured hand-eye required for calibrated laser camera motion')
        if (handeye.get('source_id') != state.get('source_id')
                or handeye.get('optical_intrinsics_id', handeye.get('calibration_id'))
                != state.get('optical_intrinsics_id', state.get('calibration_id'))):
            raise ValueError('Hand-eye and active camera optical identities differ')
        # Validate and freeze the measured camera correction without changing
        # the SDK motion TCP, geometry model, payload, or saved poses.
        j6_from_camera(self.flange_from_nominal_tcp, handeye)
        self._tcp_from_camera = transform(handeye['tcp_from_camera'])
        identity = self._frame_identity(state)
        if self.can_start is not None:
            if self.can_start() is False:
                raise ValueError('Enabled healthy idle motors required for laser calibration')
        elif not self.ready():
            raise ValueError('Enabled healthy idle motors required for laser calibration')
        if self.preempt:
            self.preempt()
        self.cancelled, self._automatic = False, True
        self.count = self.index = 0
        self.result = self.error = self.geometry = self.hunt = self._identity = None
        self._identity = identity
        self.return_pose, self.poses, self.rejected = None, [], []
        self._capture_hunt_used.clear()
        self._move_goal = self.endpoint_error = None
        self._camera_options = dict(options or {})
        self._motion_started = self.clock()
        if go_to_calibration_pose:
            try:
                requested = self.move_saved_pose('Calibration Pose')
                if requested is None:
                    raise ValueError('Saved Calibration Pose callback must return its requested nominal TCP pose')
                self._move_goal = transform(requested)
            except Exception as error:
                self._fail(error)
                raise
            self.status, self.message = 'going_to_calibration_pose', 'Moving to saved Calibration Pose'
        else:
            self._prepare()
        return self.state()

    def _prepare(self):
        if not self.ready():
            raise ValueError('Wait for quiet motion before laser preparation')
        # Finish the discrete service move with one fixed measured reference.
        # Never keep retargeting during settling or a native exposure bracket.
        self.hold()
        self.return_pose = transform(self.tcp_pose())
        self._capture_pose = self.return_pose.copy()
        self._submit('prepare', {**self._camera_options, 'action': 'laser_new'})
        self.status, self.message = 'preparing', 'Preparing laser-ON board capture and smart exposure'

    def dispatch(self, request):
        action = request['action']
        if action == 'laser_state':
            return self.state()
        if action == 'laser_cancel':
            return self.cancel()
        if action == 'laser_auto':
            return self.start(go_to_calibration_pose=request.get('go_to_calibration_pose', True))
        if self.busy or self.future is not None:
            raise RuntimeError('Wait for laser operation or select Cancel')
        actions = {'laser_config': 'laser_configure', 'laser_reset': 'laser_new',
                   'laser_capture': 'laser_capture', 'laser_check': 'laser_check',
                   'laser_solve': 'laser_solve', 'laser_preview': 'laser_preview'}
        if action not in actions:
            raise ValueError('Unknown laser calibration action')
        self.cancelled, self._automatic, self.error = False, False, None
        # Filesystem request IDs belong to the service transport, not camera settings.
        payload = {key: value for key, value in request.items() if not key.startswith('_')}
        payload['action'] = actions[action]
        if action in ('laser_capture', 'laser_check'):
            if not self.ready():
                raise ValueError('Native laser capture requires enabled quiet motors')
            self._capture_pose = transform(self.tcp_pose())
            self._capture_cutoff = self.clock()
            payload.update(not_before_monotonic=self._capture_cutoff,
                metadata={'base_from_tcp': self._capture_pose.tolist(), 'motion_tcp': 'nominal_camera_tcp'})
            self._identity = self._frame_identity(self.camera.state(include_frame=False))
        self._submit(action, payload)
        self.status, self.message = action, 'Weldfinder laser camera operation in progress'
        return self.state()

    def _move(self, pose):
        pose = transform(pose)
        if self.validate_pose:
            self.validate_pose(pose)
        self.move_pose(pose, automatic=True)
        self._move_goal = pose.copy()
        self.endpoint_error = None
        self._motion_started = self.clock()

    def _camera_pose(self):
        return transform(self.tcp_pose()) @ self._tcp_from_camera

    def _nominal_pose(self, camera_pose):
        return transform(camera_pose) @ np.linalg.inv(self._tcp_from_camera)

    def _begin_hunt(self, phase):
        now = self.clock()
        self._hunter = CheckerboardHunt(self._camera_pose(), started_at=now)
        self._hunt_moving, self._last_frame = False, None
        self._begin_quiet_wait()
        self.hunt = {'phase': phase, 'steps': 0, 'max_steps': 40, 'step_mm': 5.,
                     'corner_count': 0, 'reacquiring': False, 'motion_active': False}
        self.status, self.message = 'hunting', f'Hunting a full laser-ON board at {phase}'

    def _fresh_frame(self, state):
        stamp = state.get('read_started')
        now = self.clock()
        return bool(state.get('fresh') and stamp is not None and np.isfinite(stamp)
                    and 0 <= now-stamp <= 1. and stamp >= self._settled_at+.4)

    def _tick_hunt(self, state):
        now = self.clock()
        if now-self._hunter.started_at >= self._hunter.timeout_s:
            raise ValueError('Laser checkerboard hunt exhausted its 60 s bound')
        if self._hunt_moving:
            if not self._endpoint_converged(label='Laser hunt endpoint'):
                return
            self._hunt_moving = False
            self.hunt['motion_active'] = False
            self.hold()
            self._begin_quiet_wait()
        if not self._update_quiet_wait():
            return
        if not self._fresh_frame(state):
            if now-self._settled_at > 2:
                raise ValueError('No fresh native board image after laser hunt motion')
            return
        self._check_identity(state)
        key = (state.get('sequence'), state.get('generation'), state['read_started'])
        if key == self._last_frame:
            return
        self._last_frame = key
        board = state.get('board') or {}
        proposal = self._hunter.next_step(self._camera_pose(), board.get('corners', []),
            state.get('resolution'), full_board=bool(board.get('found')), now=now)
        self.hunt.update(proposal, motion_active=False)
        if proposal['status'] == 'exhausted':
            raise ValueError(f"Laser checkerboard search exhausted: {proposal['reason']}")
        if proposal['status'] == 'complete':
            self._hunter = None
            if self.hunt['phase'] == 'baseline':
                baseline = self._camera_pose()
                camera_poses, self.geometry = cone_plan(baseline, count=25)
                self.poses = [self._nominal_pose(pose) for pose in camera_poses]
                self.geometry['requested_baseline'] = self.return_pose.tolist()
                self.geometry['capture_nominal_tcp_origins_m'] = [pose[:3, 3].tolist() for pose in self.poses]
                self.geometry['camera_frame'] = 'measured_optical_camera'
                if self.validate_pose:
                    for pose in self.poses:
                        self.validate_pose(pose)
                self._move(self.poses[0])
                self.status = 'moving'
            else:
                self.status = 'settling'
                self._begin_quiet_wait()
            return
        self._move(self._nominal_pose(proposal['target_pose']))
        self._hunt_moving = True
        self.hunt['motion_active'] = True

    def _validate_observation(self, result):
        observation = result.get('observation') or {}
        self._check_identity(observation)
        stamp = observation.get('read_started')
        if stamp is None or not np.isfinite(stamp) or not self._capture_cutoff <= stamp <= self.clock():
            raise ValueError('Native laser acquisition predates settled capture cutoff')
        if self.clock()-stamp > 3.:
            raise ValueError('Native laser acquisition expired before acceptance')

    def _advance(self):
        self.index += 1
        if self.index == self.total:
            self._move(self.return_pose)
            self.status, self.message = 'returning', 'Returning to laser calibration baseline'
        else:
            self._move(self.poses[self.index])
            self.status, self.message = 'moving', f'Moving to laser view {self.index+1} of 25'
        self._settled_at = None

    def _finish_future(self):
        future, kind = self.future, self.pending_kind
        self.future = self.pending_kind = None
        if self.cancelled:
            return
        if kind in ('prepare', 'capture', 'laser_capture', 'laser_check'):
            self._check_identity(self.camera.state(include_frame=False))
        if kind in ('capture', 'laser_capture', 'laser_check'):
            self._stationary(self._capture_pose)
        try:
            result = future.result()
        except Exception as error:
            if kind == 'capture':
                # The cheap live detector can miss a complete laser-ON board.
                # Only the native exhaustive board search may authorize hunt
                # motion, and only one bounded hunt/retry is allowed per view.
                missing_board = str(error).startswith(('Native checkerboard not found', 'Checkerboard not found'))
                reacquire = missing_board and self.index not in self._capture_hunt_used
                self.rejected.append({'index': self.index, 'error': str(error), 'reacquiring': reacquire})
                if reacquire:
                    self._capture_hunt_used.add(self.index)
                    self._begin_hunt('capture')
                    return
                self._advance()
                return
            raise
        if kind == 'prepare':
            # Exposure discovery is configuration, not a paired measurement.
            # Establish the actual settled baseline after it completes.
            self.status, self.message = 'preparation_settling', 'Waiting for quiet settled laser baseline'
            self._begin_quiet_wait()
        elif kind in ('capture', 'laser_capture', 'laser_check'):
            self._validate_observation(result)
            if kind != 'laser_check':
                count = result.get('count')
                if type(count) is not int or count <= self.count:
                    raise ValueError('Weldfinder did not accept a new native laser view')
                if kind == 'capture' and count != self.count+1:
                    raise ValueError('Laser capture count changed outside this calibration owner')
                self.count = count
            self.result = result
            if kind == 'capture':
                self._advance()
            else:
                self.status = 'idle'
        elif kind in ('solve', 'laser_solve'):
            if type(result.get('views')) is not int or result['views'] < 20 or not result.get('quality'):
                raise ValueError('Laser solve needs 20 accepted views and native quality gates')
            plane = np.asarray(result.get('plane'), float)
            if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
                raise ValueError('Weldfinder returned an invalid laser plane')
            self.result, self.status = result, 'complete'
            self.message = 'Laser plane calibrated by Weldfinder'
        else:
            if kind == 'laser_reset':
                self.count = 0
            self.result, self.status = result, 'idle'

    def tick(self, camera_state=None):
        if self._closed:
            return
        try:
            if self.future is not None:
                if self.future.done():
                    self._finish_future()
                elif not self.cancelled and self.clock()-self._operation_started > 30:
                    raise ValueError('Native laser operation timed out')
                return
            if self.cancelled or not self.busy:
                return
            state = camera_state if camera_state is not None else self.camera.state(include_frame=False)
            if not state.get('enabled') or state.get('error'):
                raise ValueError(state.get('error') or 'Camera stopped during laser calibration')
            if self.status == 'going_to_calibration_pose':
                if self._endpoint_converged(timeout_s=300., label='Saved Calibration Pose endpoint'):
                    self._prepare()
                return
            self._check_identity(state)
            if self.status == 'preparation_settling':
                if self._update_quiet_wait(dwell=.5):
                    self.return_pose = transform(self.tcp_pose())
                    self._begin_hunt('baseline')
            elif self.status == 'hunting':
                self._tick_hunt(state)
            elif self.status in ('moving', 'returning'):
                if not self._endpoint_converged():
                    return
                returning = self.status == 'returning'
                self.hold()
                self.status = 'return_settling' if returning else 'settling'
                self._begin_quiet_wait()
            elif self.status == 'return_settling':
                if not self._update_quiet_wait():
                    return
                if self.count < 20:
                    raise ValueError(f'Only {self.count} accepted laser views; need 20 with normal/depth diversity')
                self._submit('solve', {'action': 'laser_solve'})
                self.status = 'solving'
            elif self.status == 'settling':
                if not self._update_quiet_wait():
                    return
                if not self._fresh_frame(state):
                    if self.clock()-self._settled_at > 3:
                        raise ValueError('No fresh native board frame after settled laser pose')
                    return
                # Native laser-ON board/stripe validation is authoritative.
                # A partial cheap preview alone must never move the arm.
                self._capture_pose = transform(self.tcp_pose())
                self._capture_cutoff = self._settled_at+.4
                self._submit('capture', {'action': 'laser_capture',
                    'not_before_monotonic': self._capture_cutoff,
                    'metadata': {'base_from_tcp': self._capture_pose.tolist(),
                                 'motion_tcp': 'nominal_camera_tcp', 'view_index': self.index}})
                self.status = 'capturing'
        except Exception as error:
            self._fail(error)

    def close(self):
        self.cancelled, self._closed = True, True
        self._cleanup_camera()
        self.executor.shutdown(wait=False, cancel_futures=False)
