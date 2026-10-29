"""Single-owner robot application service. All frontend traffic is serialized JSON.

The SDK thread continues independently of UI rendering and JSON telemetry. Browser
connections share this service, its Controller, and its single kinematic model.
"""
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.control import ANGLE_TOL, REST_TOL, Controller
from autoweld.diagnostics import MotionRejection, solve_pose
from autoweld.motion import MotionLimits, MotionPlanner


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


_TELEMETRY_IMAGE_FIELDS = frozenset(('frame_jpeg_base64', 'preview_jpeg_base64',
                                    'frame_png_base64', 'image_base64'))
_TELEMETRY_ARRAY_FIELDS = frozenset(('xyz_mm', 'xyz_m', 'uv', 'strength', 'width_px',
    'points_j6_m', 'points_world_m', 'points_base_m', 'points_camera_m', 'points_camera_mm',
    'corners', 'corners_px', 'corners_native_px', 'raw_corners_native_px',
    'pixels', 'pixels_px', 'frame', 'image', 'mask', 'preview'))


def compact_telemetry(value, *, _geometry=False):
    """Detach status metadata while pruning acquisition payloads before traversal.

    One-time API responses retain their full data. Cached request status and
    periodic telemetry keep counts, calibration/acquisition IDs, diagnostics,
    quality and explicitly named geometry/transforms without retaining another
    image or registered point cloud for every future render/tick.
    """
    if isinstance(value, dict):
        return {key: compact_telemetry(item, _geometry=_geometry or key == 'geometry')
                for key, item in value.items()
                if key not in _TELEMETRY_IMAGE_FIELDS and not (
                    key in _TELEMETRY_ARRAY_FIELDS and isinstance(item, (list, tuple, np.ndarray))
                    and not (_geometry and key in ('corners', 'frame')))}
    if isinstance(value, (list, tuple)):
        return [compact_telemetry(item, _geometry=_geometry) for item in value]
    return value


def gravity_record(runtime, name, model_fingerprint):
    """Prefer a matching local override, then this installed tool's live profile."""
    from autoweld.models import MOUNT

    for path in (Path(runtime) / name, MOUNT / 'gravity' / name):
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get('model_fingerprint') == model_fingerprint:
            return record
        logging.warning('Ignoring gravity record for another robot model: %s', path)
    return None


RECOVERY_WINDOW_S = 10.


def recovery_cancels(request):
    action = request.get('action')
    return (action == 'shutdown' or action == 'mode' and request.get('mode') == 'Hold'
            or action in ('calibration_cancel', 'laser_cancel', 'scan_cancel', 'weld_cancel'))


def recovery_discards(request):
    action = request.get('action', '')
    if recovery_cancels(request):
        return False
    return (action in ('mode', 'rest', 'disable', 'enable', 'joint', 'pose', 'pose_recall', 'acknowledge_fault',
                       'gravity_trial', 'gravity_configure', 'gravity_feedforward', 'gravity_payload',
                       'position_speed_limit', 'speed_limits', 'motion_limits')
            or action.startswith('calibration_') and action != 'calibration_state'
            or action.startswith('laser_') and action != 'laser_state'
            or action.startswith('scan_') and action not in ('scan_state', 'scan_export')
            or action.startswith('weld_') and action not in ('weld_state', 'weld_export'))


def stamp_admission(request, *, file_mtime=None):
    """Oldest admission time wins; public timestamps can never renew intent."""
    now, wall = time.monotonic(), time.time()
    ages = [0.]
    stamps = [] if file_mtime is None else [file_mtime]
    if request.get('action') == 'recover_control' and 'requested_at' in request:
        stamps.append(request['requested_at'])
    for stamp in stamps:
        if type(stamp) not in (int, float) or not np.isfinite(stamp) or stamp > wall:
            ages.append(RECOVERY_WINDOW_S+1.)
        else:
            ages.append(wall-stamp)
    request['_service_queued_monotonic'] = now-max(ages)
    if request.get('action') == 'recover_control':
        request['_recovery_requested_monotonic'] = request['_service_queued_monotonic']
    return request


def check_connection_admission(host_guard, *, admission_guard=None, recovery_rest_q=None,
                               recovery_joint_limits=None):
    """Only the explicit, physically qualified recovery factory skips pressure."""
    if admission_guard is not None:
        admission_guard()
    recovery = (admission_guard is not None and recovery_rest_q is not None
                and recovery_joint_limits is not None)
    if host_guard is not None and not recovery:
        host_guard.check(force=True)
    return recovery


class JsonConnection:
    """Queue boundary: UI never receives a controller, robot, camera, or model."""
    def __init__(self, service):
        self._service = service

    def send(self, message):
        request = json.loads(message)
        if not isinstance(request, dict) or not isinstance(request.get('action'), str):
            raise ValueError('Expected a JSON object with an action')
        stamp_admission(request)
        if hasattr(self._service, 'notice_recovery_cancellation'):
            self._service.notice_recovery_cancellation(request)
        self._service.events.put(request)

    def snapshot(self):
        with self._service.snapshot_lock:
            return self._service.snapshot_json


class RobotService:
    def __init__(self, robot, model_path, runtime, *, live=False, robot_factory=None,
                 start_disabled=False, camera=None, host_guard=None, recovery_factory=None,
                 disabled_reader_factory=None):
        self.runtime = Path(runtime)
        self.model_fingerprint = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        self.runtime.mkdir(parents=True, exist_ok=True)
        for folder in ('requests', 'responses'):
            (self.runtime / folder).mkdir(exist_ok=True)
        from autoweld.kinematics import shared_kinematics
        self.planner = MotionPlanner(model_path, kin=shared_kinematics(robot))
        self.ctrl = Controller.get_instance(lambda: robot, self.planner.model.jnt_range[:6],
            live=live, reconnect_factory=robot_factory)
        self.ctrl.strict_joint_limits = {5}
        if not self.ctrl.limits[5, 0] <= self.ctrl.rest[5] <= self.ctrl.limits[5, 1]:
            raise ValueError('Captured J6 rest is outside calibrated travel; position and capture an interior rest')
        self.ctrl.motion_planner = self.planner
        self.ctrl.motion_collision = self.has_self_collision
        if live and host_guard is None:
            from autoweld.host_health import HostHealthGuard
            host_guard = HostHealthGuard()
        self.host_guard = host_guard if live else None
        if start_disabled:
            self.ctrl.disabled_q = self.ctrl.rest.copy()
            self.ctrl.closed = True
            self.ctrl.cached_rest_display = True
            self.ctrl.mode = 'Disabled'
        else:
            self.ctrl.start_hold()
        self.camera = camera
        self.calibration = None
        self.laser = None
        self.scan = None
        self.weld = None
        self.scan_plan_probe = None
        self._calibration_reference = None
        self.events = queue.Queue()
        self._urgent_events = queue.SimpleQueue()
        self.recovery_factory = recovery_factory
        self.disabled_reader_factory = disabled_reader_factory
        self.disabled_rest_inspection = None
        self.recovery = None
        self._recovery_host_owner = None
        self._recovery_host_override = {'active': False}
        self._last_recovery_host_warning = 0.
        self._recovery_cancel_at = None
        self._recovery_discard_before = None
        self._recovery_discarded = []
        self.snapshot_lock = threading.Lock()
        self.snapshot_json = '{}'
        self.pending_pose = None
        self.request_status = None
        self.exit_after_disable = False
        self.running = True
        self.message = 'Holding captured rest' if not start_disabled else 'Motors disabled; showing captured rest'
        self.fault = None
        self.gravity_tuning = {'status': 'default', 'message': 'Exact SDK model; factors have not been fitted this session'}
        self.gravity_payload = None
        for name in ('gravity-payload.json', 'gravity-tuning.json'):
            record = gravity_record(self.runtime, name, self.model_fingerprint)
            if record is None:
                continue
            if name == 'gravity-payload.json':
                # The sole connection factory restores this profile before
                # enabling gravity; saved-display startup has no SDK device.
                self.gravity_payload = record
            else:
                self.gravity_tuning = record
        profile = gravity_record(self.runtime, 'gravity-config.json', self.model_fingerprint)
        if profile is not None:
            self.ctrl.configure_gravity(profile['factors'], model=profile.get('model', 'sdk'))
            if self.gravity_tuning['status'] == 'default':
                self.gravity_tuning = {'status': 'loaded', 'message': 'Saved gravity profile loaded', 'profile': profile}
        self.saved_pose_path = self.runtime / 'poses.json'
        self.saved_poses = json.loads(self.saved_pose_path.read_text()) if self.saved_pose_path.exists() else {}
        rest_record = {
            'arm_rad': self.ctrl.rest[:6].tolist(), 'gripper_normalized': float(self.ctrl.rest[6]),
            'gripper_limits_motor_rad': self.ctrl.gripper_limits.tolist(), 'live': live}
        rest_path = self.runtime/'rest.json'
        if rest_path.exists():
            from autoweld.disabled_rest import preserve_rest_provenance
            try:
                previous_rest = json.loads(rest_path.read_text())
            except (OSError, ValueError):
                logging.warning('Previous rest provenance could not be read; retaining only the current rest fields')
            else:
                rest_record = preserve_rest_provenance(previous_rest, rest_record)
        atomic_json(rest_path, rest_record)
        self.publish()

    def has_self_collision(self, q):
        # Same adjacent-link and floor exclusions as the i2rt Viser example.
        self.planner.fk(q)
        model, data = self.planner.model, self.planner.data
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist >= -.001:
                continue
            if any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in (contact.geom1, contact.geom2)):
                continue
            b1, b2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            if model.body_parentid[b1] != b2 and model.body_parentid[b2] != b1:
                return True
        return False

    def tcp_pose(self):
        return self.planner.fk(self.ctrl.positions()).copy()

    def validate_pose(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Expected a finite TCP transform')
        seed = self.planner._configuration(self.ctrl.positions())
        q = solve_pose(self.planner.kin, pose, 'weld_tcp', seed, stage='endpoint_ik')
        target = np.r_[q[:6], self.ctrl.gripper_target]
        if self.has_self_collision(target):
            raise MotionRejection('endpoint_validation', 'self_collision', 'TCP endpoint has robot self-collision')
        self.ctrl.validate(target)
        return target

    def move_pose(self, pose, automatic=False):
        target = self.validate_pose(pose)
        self.ctrl.command(target, 'IK', automatic=automatic, cartesian=np.asarray(pose))

    def move_calibration_pose(self, pose, automatic=True, *, owner=None):
        """Bound a calibration around its own baseline, retaining normal planning.

        A user-selected calibration pose can be far from supported rest. The
        commissioning test's rest-relative envelope is not its motion origin.
        """
        if self.ctrl.cooldown_active:
            raise RuntimeError('Calibration paused for thermal cooldown at rest')
        owner = self.calibration if owner is None else owner
        baseline = owner.return_pose
        if baseline is None:
            raise RuntimeError('Calibration motion requires a captured baseline')
        reference = self._calibration_reference
        if reference is None or reference[0] is not baseline:
            reference = (baseline, self.ctrl.positions()[:6].copy())
            self._calibration_reference = reference
        pose = np.asarray(pose, float)
        if np.linalg.norm(pose[:3, 3]-baseline[:3, 3]) > .08:
            raise ValueError('Calibration translation exceeds its 80 mm baseline envelope')
        angle = Rotation.from_matrix(pose[:3, :3] @ baseline[:3, :3].T).magnitude()
        if angle > np.deg2rad(20):
            raise ValueError('Calibration orientation exceeds its 20 degree baseline envelope')
        target = self.validate_pose(pose)
        if np.max(abs(target[:6]-reference[1])) >= .49:
            raise ValueError('Calibration joint excursion exceeds 0.49 rad from its baseline')
        self.ctrl.command(target, 'IK', automatic=False, cartesian=pose)

    def can_start_laser(self):
        if getattr(getattr(self, 'weld', None), 'busy', False):
            raise RuntimeError('Weld seam finding owns robot motion')
        self.ctrl.health()
        if not self.ctrl._feedback_health():
            raise RuntimeError('Fresh SDK feedback is required before laser calibration')
        if self.exit_after_disable or self.ctrl.shutdown_requested:
            raise RuntimeError('Shutdown owns robot motion')
        if self.ctrl.disable_pending or self.ctrl.cooldown_active:
            raise RuntimeError('Wait for rest return and thermal cooldown before laser calibration')
        if not self.ctrl.ready_for_waypoint():
            raise RuntimeError('Laser calibration requires enabled, healthy stationary robot control')
        if 'Calibration Pose' not in self.saved_poses:
            raise RuntimeError('Save a Calibration Pose before automatic laser calibration')
        calibration = self.calibration
        if calibration is not None and calibration.future is not None and not calibration.future.done():
            raise RuntimeError('Wait for the current camera operation before laser calibration')

    def preempt_for_laser(self):
        self.pending_pose = None
        if self.calibration is not None:
            self.calibration.abort_for_shutdown()
            self.calibration.message = 'Camera calibration released motion to laser calibration'

    def scan_feedback(self):
        """Certify measured SDK feedback without a second model/device owner."""
        ctrl = self.ctrl
        ctrl.health()
        if ctrl.closed or ctrl.cached_rest_display:
            raise RuntimeError('Enabled SDK control is required for scanning')
        read_started = time.monotonic()
        age = ctrl.robot.get_feedback_diagnostics()['age_s'] if ctrl.live else 0.
        velocity = np.asarray(ctrl.robot.get_observations()['joint_vel'][:6], float)
        stationary = bool(ctrl.motion_plan is None and not ctrl.mode_transition
            and not ctrl.disable_pending and not ctrl.shutdown_requested and not ctrl.cooldown_active
            and not ctrl.motion_fault() and np.isfinite(velocity).all() and np.max(abs(velocity)) <= .05)
        # Conservatively include all SDK-read latency after the clock anchor;
        # expensive health checks before the actual read do not age this sample.
        returned_age = age+max(0.,time.monotonic()-read_started)
        return {'fresh': bool(np.isfinite(age) and 0 <= age and returned_age <= .1),
                'stationary': stationary,'measured_at': float(read_started-age)}

    def can_start_scan(self, *, wait_for_fresh=False, _owner=None):
        weld = getattr(self, 'weld', None)
        if weld is not _owner and getattr(weld, 'busy', False):
            raise RuntimeError('Weld seam finding owns robot motion')
        if self.exit_after_disable or self.ctrl.shutdown_requested:
            raise RuntimeError('Shutdown owns robot motion')
        if self.ctrl.disable_pending or self.ctrl.cooldown_active:
            raise RuntimeError('Wait for rest return and thermal cooldown before scanning')
        if wait_for_fresh:
            # Explicit Start alone may await a new quiet exchange. Readiness
            # polls remain immediate and never advance SDK or camera lifecycle.
            if (self.ctrl.mode not in ('Hold','IK','Joint') or self.ctrl.closed
                    or self.ctrl.motion_plan is not None or self.ctrl.motion_fault()):
                raise RuntimeError('Scanning requires healthy stationary position-speed Hold or IK')
            self.ctrl._wait_for_fresh_motion_start()
        if self.ctrl.mode not in ('Hold', 'IK', 'Joint') or not self.ctrl.ready_for_waypoint():
            raise RuntimeError('Scanning requires healthy stationary position-speed Hold or IK')
        feedback = self.scan_feedback()
        if not feedback['fresh'] or not feedback['stationary']:
            raise RuntimeError('Fresh stationary SDK feedback is required before scanning')
        guard = getattr(self, 'host_guard', None)
        if guard is not None:
            guard.check()
        if self.pending_pose is not None:
            raise RuntimeError('Wait for the queued TCP target before scanning')
        for owner in (self.calibration, self.laser):
            if owner is not None and (getattr(owner, 'busy', False)
                    or getattr(owner, 'status', 'idle') not in ('idle', 'complete', 'cancelled', 'error')
                    or getattr(owner, 'future', None) is not None
                    or getattr(owner, '_cleanup', None) is not None and not owner._cleanup.done()):
                raise RuntimeError('Wait for camera calibration or laser operations before scanning')

    def can_start_weld(self, *, wait_for_fresh=False):
        if getattr(getattr(self, 'scan', None), 'busy', False):
            raise RuntimeError('Laser scanning owns robot motion')
        self.can_start_scan(wait_for_fresh=wait_for_fresh, _owner=getattr(self, 'weld', None))
        if 'Weld Finder' not in self.saved_poses:
            raise RuntimeError('Save a Weld Finder pose before finding a weld seam')

    def validate_scan_pose(self, pose):
        """Constrain the requested square in its frozen measured optical frame."""
        from autoweld.calibration import transform
        scan = self.scan
        if scan.return_pose is None or scan._handeye is None:
            raise RuntimeError('Scanning requires a measured baseline and matching hand-eye')
        baseline, pose = transform(scan.return_pose), transform(pose)
        camera = transform(scan._handeye)
        optical = baseline @ camera
        requested = pose @ camera
        local = optical[:3, :3].T @ (requested[:3, 3]-optical[:3, 3])
        half = np.array([scan.settings.width_mm, scan.settings.height_mm])/2000.
        if np.any(abs(local[:2]) > half+.001):
            raise ValueError('Scan target exceeds its centered optical XY envelope')
        if abs(local[2]) > .060:
            raise ValueError('Scan depth exceeds its 60 mm measured-baseline envelope')
        view = getattr(scan, 'view_pose', None)
        expected_rotation = baseline[:3, :3] if view is None else transform(view)[:3, :3]
        if Rotation.from_matrix(pose[:3, :3] @ expected_rotation.T).magnitude() > .005:
            raise ValueError('Scan target changes the fixed viewing orientation')
        return self.validate_pose(pose)

    def move_scan_pose(self, pose):
        """Normal discrete service planner; installed bounds/collisions remain enforced."""
        if self.ctrl.cooldown_active:
            raise RuntimeError('Scanning paused for thermal cooldown at rest')
        target = self.validate_scan_pose(pose)
        self.ctrl.command(target, 'IK', automatic=False, cartesian=np.asarray(pose))

    def probe_scan_pose(self, request):
        """Read-only installed-model IK diagnostics; never submit a robot target."""
        from autoweld.calibration import transform
        if self.ctrl.motion_plan is not None or self.exit_after_disable or self.scan.busy:
            raise RuntimeError('Stop active motion before read-only scan workspace diagnostics')
        baseline = transform(request['baseline_pose'])
        handeye = (self.camera.state(include_frame=False).get('handeye') or {}).get('tcp_from_camera')
        if handeye is None:
            raise ValueError('Workspace diagnostics require the installed measured hand-eye')
        correction = transform(handeye)
        xy = np.asarray(request['offset_xy_mm'], float)
        depth, roll = float(request['depth_offset_mm']), float(request.get('optical_roll_deg', 0.))
        if (xy.shape != (2,) or not np.isfinite(np.r_[xy, depth, roll]).all()
                or np.max(abs(xy)) > 50. or abs(depth) > 60. or abs(roll) > 180.):
            raise ValueError('Expected bounded finite square offset, depth and optical roll')
        optical = baseline @ correction
        requested = optical.copy()
        requested[:3, 3] += optical[:3, :3] @ np.r_[xy/1000., depth/1000.]
        requested[:3, :3] = optical[:3, :3] @ Rotation.from_euler('z', roll, degrees=True).as_matrix()
        pose = requested @ np.linalg.inv(correction)
        seed = np.asarray(request.get('seed_q', self.ctrl.positions()[:6]), float)
        bounds = self.ctrl.effective_joint_bounds()
        if (seed.shape != (6,) or not np.isfinite(seed).all()
                or np.any(seed < bounds[:, 0]) or np.any(seed > bounds[:, 1])):
            raise ValueError('Diagnostic seed is outside installed joint bounds')
        try:
            q = solve_pose(self.planner.kin, pose, 'weld_tcp',
                           self.planner._configuration(np.r_[seed, self.ctrl.gripper_target]), stage='scan_preflight')
        except MotionRejection as error:
            error.diagnostic.update(joint_bounds_rad=bounds.tolist(), seed_q=seed.tolist(),
                                    requested_pose=pose.tolist(), optical_roll_deg=roll,
                                    offset_xy_mm=xy.tolist(), depth_offset_mm=depth)
            raise
        if np.any(q[:6] < bounds[:, 0]) or np.any(q[:6] > bounds[:, 1]):
            raise ValueError('Diagnostic result exceeds installed joint bounds')
        if self.has_self_collision(np.r_[q[:6], self.ctrl.gripper_target]):
            raise MotionRejection('scan_preflight', 'self_collision', 'Diagnostic scan endpoint has self-collision')
        result = {'read_only': True, 'q': q[:6].tolist(), 'requested_pose': pose.tolist(),
                  'joint_margin_rad': np.minimum(q[:6]-bounds[:, 0], bounds[:, 1]-q[:6]).tolist()}
        if request.get('motion_compare') is True:
            import copy
            start_request = {**request, 'motion_compare': False,
                             'offset_xy_mm': request['start_offset_xy_mm']}
            start = self.probe_scan_pose(start_request)['q']
            planner = copy.copy(self.planner)  # Same installed model; no SDK/device or model construction.
            fast = planner.jacobian
            reference = getattr(planner, 'reference_jacobian', fast)
            fast_samples = getattr(planner, 'limit_samples', None)
            reference_samples = getattr(planner, 'reference_limit_samples', fast_samples)
            reports = {}
            try:
                for name, jacobian in (('reference', reference), ('optimized', fast)):
                    planner.jacobian = jacobian
                    if fast_samples is not None:
                        planner.limit_samples = reference_samples if name == 'reference' else fast_samples
                    began = time.monotonic()
                    planned = planner.plan(np.r_[start, self.ctrl.gripper_target],
                        np.r_[q[:6], self.ctrl.gripper_target], cartesian=pose,
                        bounds=bounds, collision=self.has_self_collision)
                    reports[name] = {'planning_wall_s': time.monotonic()-began, 'report': planned.report}
            finally:
                planner.jacobian = fast
                if fast_samples is not None:
                    planner.limit_samples = fast_samples
            result['motion_comparison'] = reports
        return result

    def move_laser_saved_pose(self, name):
        """An explicitly requested laser sequence uses the normal service planner."""
        saved = self.saved_poses[name]
        arm = np.asarray(saved['q'], float)
        if arm.shape != (6,) or not np.isfinite(arm).all():
            raise ValueError('Saved calibration pose must contain six finite joint angles')
        target = np.r_[arm, self.ctrl.gripper_target]
        requested_pose = self.planner.fk(target).copy()
        self.ctrl.command(target, 'Hold', automatic=False)
        return requested_pose

    def laser_state(self, camera_state=None):
        laser = getattr(self, 'laser', None)
        if laser is None:
            return None
        if camera_state is None:
            camera_state = self.camera.state(include_frame=False) if self.camera else {}
        state = laser.state(camera_state=camera_state)
        raw = (camera_state.get('laser') or {}).get('cloud')
        if raw is None:
            self._laser_registration_window = None
            return state
        from autoweld.laser_calibration import transform_cloud
        from autoweld.models import camera_from_flange

        # A rigid camera→J6 conversion never needs joint positions. Registration
        # into the base frame additionally requires fresh stationary SDK data.
        measured = None
        try:
            ctrl = self.ctrl
            now = time.monotonic()
            eligible = False
            if (not ctrl.closed and not ctrl.cached_rest_display and ctrl.motion_plan is None
                    and not ctrl.motion_fault() and not ctrl.mode_transition):
                velocity = np.asarray(ctrl.robot.get_observations()['joint_vel'][:6], float)
                age = ctrl.robot.get_feedback_diagnostics()['age_s'] if ctrl.live else 0.
                fresh = np.isfinite(age) and 0 <= age <= .1
                if fresh and np.isfinite(velocity).all() and np.max(abs(velocity)) <= .05:
                    pose = self.tcp_pose()
                    eligible = True
                    window = getattr(self, '_laser_registration_window', None)
                    if (window is None or np.linalg.norm(pose[:3, 3]-window[1][:3, 3]) > .0005
                            or Rotation.from_matrix(pose[:3, :3] @ window[1][:3, :3].T).magnitude() > .005):
                        window = self._laser_registration_window = (now, pose.copy())
                    acquired = raw.get('read_started')
                    if (acquired is not None and np.isfinite(acquired)
                            and acquired >= window[0] and now-window[0] >= .2):
                        measured = pose
            if not eligible:
                self._laser_registration_window = None
            cloud = transform_cloud(raw, measured, camera_state.get('handeye'), camera_from_flange(),
                                    now=now, max_points=5000)
            cloud['points_base_m'] = cloud.get('points_world_m', [])
            cloud['fresh'] = True
            state['cloud'] = cloud
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            state['cloud'] = {'coordinate_frame': 'j6', 'units': 'm', 'count': 0,
                              'fresh': False, 'status': str(error), 'points_base_m': [], 'points_j6_m': []}
        return state

    def gravity_telemetry(self, *, include_regressor=False):
        """Read cached SDK feedback and model output without another robot/model."""
        robot = self.ctrl.robot
        observations = robot.get_observations() if not self.ctrl.closed and hasattr(robot, 'get_observations') else {}
        def values(key):
            value = observations.get(key)
            return np.asarray(value, float)[:6].tolist() if value is not None else None
        measured_q = values('joint_pos') or self.ctrl.positions()[:6].tolist()
        timestamp = observations.get('feedback_timestamp')
        timestamp = float(timestamp) if timestamp is not None else None
        now = time.time()
        joint_timestamps = values('joint_timestamps')
        oldest = min(joint_timestamps) if joint_timestamps else timestamp
        age = max(0., now-oldest) if oldest is not None else None
        result = {'q_rad': measured_q, 'vel_rad_s': values('joint_vel'),
            'timestamp': now, 'feedback_timestamp': timestamp, 'joint_timestamps': joint_timestamps,
            'feedback_age_s': age, 'feedback_fresh': bool(age is not None and age <= .1 and not self.ctrl.closed),
            'feedback_source': 'saved display' if self.ctrl.closed else 'SDK motor feedback' if self.ctrl.live else 'ideal simulation',
            'reported_effort_nm': values('joint_eff'),
            'gravity': robot.get_gravity_info() if hasattr(robot, 'get_gravity_info') else None,
            'torque': robot.get_torque_breakdown() if hasattr(robot, 'get_torque_breakdown') else None,
            'gravity_factors': self.ctrl.gravity_factors.tolist(), 'gravity_model': self.ctrl.gravity_model,
            'arm_control_modes': (robot.get_arm_control_modes() if hasattr(robot, 'get_arm_control_modes')
                                  else [self.ctrl.control_law] * 6),
            'gravity_trial': getattr(robot, '_gravity_trial', None),
            'mode_transition': getattr(self.ctrl, 'mode_transition', False),
            'mode_transition_error': getattr(self.ctrl, 'mode_transition_error', None),
            'tuning': getattr(self, 'gravity_tuning', None),
            'payload': getattr(self, 'gravity_payload', None),
            'model_fingerprint': getattr(self, 'model_fingerprint', None)}
        if include_regressor:
            from autoweld.gravity_calibration import gravity_regressor
            if getattr(self, 'gravity_data', None) is None:
                self.gravity_data = mujoco.MjData(self.planner.model)
            baseline, matrix = gravity_regressor(self.planner.model, self.gravity_data, measured_q)
            result.update(baseline_nm=baseline.tolist(), regressor=matrix.tolist())
        return result

    def gravity_test_preflight(self, req):
        """Preview wider test paths on the existing model without motor access."""
        if not self.ctrl.closed:
            raise RuntimeError('Preflight wider gravity tests with motors disabled')
        start, goal = (np.asarray(req[key], float) for key in ('start_q', 'q'))
        rest, bounds = self.ctrl.rest[:6], self.ctrl.limits
        def validate(q):
            if np.max(abs(q[:6]-rest)) >= self.ctrl.test_excursion:
                raise ValueError('Path exceeds configured gravity test envelope')
        if any(q.shape != (6,) or not np.isfinite(q).all() for q in (start, goal)):
            raise ValueError('Expected finite six-axis start and goal')
        plan = self.planner_for_gravity_test().plan(np.r_[start, 0.], np.r_[goal, 0.],
            bounds=bounds, validate=validate, collision=self.has_self_collision)
        # Check installed box envelopes above the work surface and exact load
        # torque over the path, using scratch data but never another model.
        from autoweld.gravity_calibration import gravity_regressor
        if getattr(self, 'gravity_data', None) is None:
            self.gravity_data = mujoco.MjData(self.planner.model)
        payload = self.gravity_payload
        mass = payload['mass_kg'] if payload else 0.
        params = np.r_[mass, mass*np.asarray(payload['com_flange_m'] if payload else [0.]*3)]
        minimum_height, peak = float('inf'), np.zeros(6)
        for fraction in np.linspace(0, 1, 81):
            q = plan.path.spline(fraction)[:6]
            baseline, matrix = gravity_regressor(self.planner.model, self.gravity_data, q)
            peak = np.maximum(peak, abs((baseline+matrix@params)*self.ctrl.gravity_factors))
            self.planner.fk(q)
            endpoint_height = float('inf')
            for gid in range(self.planner.model.ngeom):
                name = mujoco.mj_id2name(self.planner.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
                if name.startswith('mount_'):
                    rotation = self.planner.data.geom_xmat[gid].reshape(3, 3)
                    half_height = abs(rotation[2])@self.planner.model.geom_size[gid]
                    height = self.planner.data.geom_xpos[gid, 2]-half_height
                    minimum_height = min(minimum_height, height)
                    endpoint_height = min(endpoint_height, height)
        if minimum_height < .05:
            raise ValueError(f'Mount clearance {minimum_height:.3f}m below 0.05m test gate')
        # Deliberately below motor encoding ranges (28/10 Nm). The first three
        # joints share a conservative 9 Nm commissioning load ceiling.
        if np.any(peak > [9., 9., 9., 3., 3., 3.]):
            raise ValueError(f'Predicted gravity exceeds commissioning torque gate: {peak.tolist()}')
        self.planner.fk(goal)
        return {'start_q': start.tolist(), 'q': goal.tolist(),
            'tcp_position_m': self.planner.data.site_xpos[self.planner.site].tolist(),
            'peak_gravity_nm': peak.tolist(), 'minimum_mount_height_m': float(minimum_height),
            'endpoint_mount_height_m': float(endpoint_height),
            'duration_s': plan.path.duration, 'max_excursion_rad': self.ctrl.test_excursion,
            'minimum_joint_margin_rad': float(np.min(np.minimum(goal-bounds[:, 0], bounds[:, 1]-goal)))}

    def planner_for_gravity_test(self):
        return self.ctrl.planner_for_controller()

    def configure_speed_limits(self, linear_velocity, joint_velocity):
        """Apply the UI's joint speed to planner and firmware as one command."""
        previous_limits = self.planner.limits
        previous_ceiling = self.ctrl.position_speed_limit
        limits = replace(previous_limits, linear_velocity=float(linear_velocity),
                         joint_velocity=float(joint_velocity))
        # Controller validation retains idle, mode, health and the SDK's 3 rad/s cap.
        # Publish/persist only after both the planner and SDK setting succeed.
        self.planner.limits = limits
        try:
            self.ctrl.configure_position_speed_limit(limits.joint_velocity)
            atomic_json(self.runtime / 'motion-limits.json', asdict(limits))
        except Exception:
            self.planner.limits = previous_limits
            if self.ctrl.position_speed_limit != previous_ceiling:
                self.ctrl.position_speed_limit = previous_ceiling
                self.ctrl._apply_position_speed_limit()
            raise
        self.message, self.fault = 'Speed limits applied', None
        return {'motion_limits': asdict(limits), 'position_speed_limit_rad_s': self.ctrl.position_speed_limit}

    def notice_recovery_cancellation(self, request):
        recovery = getattr(self, 'recovery', None)
        if recovery is not None and recovery.state().get('active') and recovery_discards(request):
            request['_blocked_during_recovery'] = True
        if recovery_cancels(request):
            self._recovery_cancel_at = time.monotonic()

    def enqueue(self, request):
        stamp_admission(request)
        self.notice_recovery_cancellation(request)
        self.events.put(request)

    def enqueue_signal(self, request):
        # CPython SimpleQueue.put is reentrant: a signal can interrupt the
        # normal queue while its non-reentrant mutex is held. No SDK calls.
        stamp = time.monotonic()
        request['_service_queued_monotonic'] = stamp
        self._recovery_cancel_at = stamp
        self._urgent_events.put(request)

    def _recovery_cancelled(self, intent):
        stamp = getattr(self, '_recovery_cancel_at', None)
        if stamp is not None and stamp >= intent:
            return True
        # Files may arrive while synchronous SDK recovery temporarily owns the
        # service tick. Inspect only; normal ordering/response handling remains.
        for path in (self.runtime / 'requests').glob('*.json'):
            try:
                if path.stat().st_size <= 65536 and recovery_cancels(json.loads(path.read_text())):
                    return True
            except (OSError, ValueError, TypeError):
                continue
        with self.events.mutex:
            return any(recovery_cancels(request) for request in self.events.queue)

    def _recovery_busy(self):
        for owner in (getattr(self, 'calibration', None), getattr(self, 'laser', None),
                      getattr(self, 'scan', None), getattr(self, 'weld', None)):
            if owner is not None and (getattr(owner, 'busy', False)
                    or getattr(owner, 'status', 'idle') not in ('idle', 'complete', 'cancelled', 'error', 'interrupted',
                                                              'paused', 'paused_cooldown', 'insufficient_length',
                                                              'incomplete')):
                return True
        return False

    def _release_for_recovery(self):
        """Only invoked after fresh disabled near-rest proof; no SDK calls."""
        self.pending_pose = None
        for owner in (getattr(self, 'calibration', None), getattr(self, 'laser', None),
                      getattr(self, 'scan', None), getattr(self, 'weld', None)):
            if owner is not None:
                owner.abort_for_shutdown()  # Explicitly releases ownership without Hold.
                owner.message = 'Calibration cancelled; control recovery established a new measured Hold owner'
        ctrl = self.ctrl
        ctrl.motion_plan = None
        ctrl.motion_started = None
        ctrl.motion_automatic = False
        ctrl.disable_pending = False
        ctrl.disable_warning = None
        ctrl.return_automatic = False
        ctrl.return_step_started = None
        ctrl.shutdown_requested = False
        ctrl.shutdown_phase = 'idle'
        ctrl.shutdown_error = None
        ctrl._shutdown_phase_started = None
        ctrl.mode_transition = False
        ctrl.mode_transition_error = None
        ctrl.cooldown_active = False
        self.exit_after_disable = False
        self._calibration_reference = None
        self._recovery_discard_before = time.monotonic()
        # Do not delete requests: each gets its ordinary journal/response with
        # a specific stale-work rejection, including the tick's local batch.

    def recovery_host_override_status(self):
        record = getattr(self, '_recovery_host_override', {'active': False})
        if record.get('active') and (self.ctrl.closed or self.ctrl.robot is not getattr(self, '_recovery_host_owner', None)):
            self._recovery_host_owner = None
            self._recovery_host_override = {'active': False, 'reason': 'Recovered owner disabled or replaced'}
        return dict(getattr(self, '_recovery_host_override', {'active': False}))

    def _grant_recovery_host_override(self):
        self._recovery_host_owner = self.ctrl.robot
        self._recovery_host_override = {'active': True, 'scope': 'recovered SDK owner only',
            'reason': 'Explicit operator recovery bypasses host pressure; normal new-work admission remains guarded',
            'since': time.time(), 'monitor_policy': 'warning_only', 'host_warning': None}

    def recover_control(self, request):
        from autoweld.recovery import ControlRecovery
        if getattr(self, 'recovery', None) is None:
            self.recovery = ControlRecovery(self.ctrl, getattr(self, 'recovery_factory', None),
                getattr(self, 'host_guard', None), self._release_for_recovery,
                cancelled=self._recovery_cancelled, busy=self._recovery_busy, collision=self.has_self_collision)
        intent = request.get('_recovery_requested_monotonic')
        if intent is None:
            stamp_admission(request)
            intent = request['_recovery_requested_monotonic']
        started = time.monotonic()
        result = self.recovery.recover(intent)
        finished = time.monotonic()
        self._recovery_attempt_window = (started, finished)
        sdk = self.ctrl.sdk_status()
        if ((result.get('changed') or result.get('partial_owner_retained'))
                and not self.ctrl.closed and self.ctrl.mode == 'Hold'
                and sdk.get('control_alive') and sdk.get('feedback_fresh')):
            self._grant_recovery_host_override()
        result['recovery_host_override'] = self.recovery_host_override_status()
        if result.get('ownership_released'):
            self._recovery_discard_before = finished
        if result.get('changed'):
            self.fault = None
            self._sdk_failure_persisted = False
            self._tick_failure_signature = None
        self.message = ('Recovery: measured Hold established' if result['phase'] == 'holding'
                        else 'Recovery: '+str(result.get('error') or result['phase']))
        atomic_json(self.runtime / 'recovery.json', result)
        return result

    def shutdown(self):
        """Always accept shutdown ownership, independently of command transport."""
        self.exit_after_disable = True
        self.pending_pose = None
        weld = getattr(self, 'weld', None)
        if weld is not None:
            try:
                weld.abort_for_shutdown()
            except Exception:
                logging.exception('Weld path cleanup failed; dedicated shutdown retains ownership')
        scan = getattr(self, 'scan', None)
        if scan is not None:
            try:
                scan.abort_for_shutdown()
            except Exception:
                logging.exception('Scan cleanup failed; dedicated shutdown retains ownership')
        calibration = getattr(self, 'calibration', None)
        if calibration is not None:
            try:
                calibration.abort_for_shutdown()
            except Exception:
                # Calibration ticks are suppressed after exit intent, so even
                # an adapter failure cannot retain motion ownership here.
                logging.exception('Calibration cleanup failed; dedicated shutdown retains ownership')
        laser = getattr(self, 'laser', None)
        if laser is not None:
            try:
                laser.abort_for_shutdown()
            except Exception:
                logging.exception('Laser cleanup failed; dedicated shutdown retains ownership')
        result = self.ctrl.shutdown(retry=True)
        self.message, self.fault = 'Shutdown: controlled return to captured rest', None
        return result

    def _disabled_rest_gate(self):
        """No read-only SDK socket may coexist with an active/retained writer."""
        ctrl = self.ctrl
        if (not ctrl.closed or self.exit_after_disable or ctrl.shutdown_requested or ctrl.disable_pending
                or ctrl.motion_plan is not None or ctrl.mode_transition or self.pending_pose is not None):
            raise RuntimeError('Rest inspection/capture requires idle disabled control with no shutdown or pending target')
        if self._recovery_busy():
            raise RuntimeError('An active calibration, scan or weld operation owns the service')
        recovery = getattr(self, 'recovery', None)
        if recovery is not None and (recovery.state().get('active') or recovery.retained_chain is not None
                                     or recovery.retained_reader is not None):
            raise RuntimeError('A recovery owner remains active or retained; disabled inspection cannot open another owner')
        chain = getattr(ctrl.robot, 'motor_chain', None)
        for owner, attribute in ((ctrl.robot, '_server_thread'), (chain, '_control_thread')):
            worker = getattr(owner, attribute, None)
            if worker is not None and worker.is_alive():
                raise RuntimeError('An SDK writer thread is still alive; disabled inspection was not opened')
        if chain is not None and getattr(chain, 'running', None) is not False:
            raise RuntimeError('The prior SDK motor-chain writer is not proven stopped')
        if not callable(getattr(self, 'disabled_reader_factory', None)):
            raise RuntimeError('This service has no configured read-only disabled encoder factory')

    def inspect_or_capture_rest(self, *, capture=False):
        from autoweld.disabled_rest import inspect_disabled_pose, publish_rest_records

        self._disabled_rest_gate()
        proof = inspect_disabled_pose(self.disabled_reader_factory, self.ctrl.limits.copy())
        self._disabled_rest_gate()  # Recheck ownership after the bounded read-only survey.
        q = np.r_[proof['joint_rad'], self.ctrl.gripper_target]
        pose = self.planner.fk(q).copy()
        proof.update(captured=bool(capture), model_fingerprint=self.model_fingerprint,
                     base_from_tcp=pose.tolist(), rest_unchanged=not capture)
        if capture:
            if self.has_self_collision(q):
                raise RuntimeError('Current disabled pose collides in the installed model; captured rest was not changed')
            saved = dict(self.saved_poses)
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            saved['Captured rest'] = dict(q=q[:6].tolist(), position=pose[:3, 3].tolist(),
                wxyz=[float(quat[3]), *quat[:3].tolist()], model_fingerprint=self.model_fingerprint,
                source='fresh_disabled_read_only_sdk', physical_timestamp=proof['physical_timestamp'])
            rest = dict(arm_rad=q[:6].tolist(), gripper_normalized=float(q[6]),
                gripper_limits_motor_rad=self.ctrl.gripper_limits.tolist(), live=self.ctrl.live,
                model_fingerprint=self.model_fingerprint, measured_disabled=proof)
            publish_rest_records(self.runtime/'rest.json', self.saved_pose_path, rest, saved)
            self.saved_poses = saved
        self.ctrl.record_disabled_pose(q, capture_rest=capture)
        self.disabled_rest_inspection = proof
        self.message = ('Captured current fresh disabled pose as supported rest' if capture else
                        'Fresh disabled SDK pose inspected; captured rest unchanged')
        return proof

    def dispatch(self, req):
        action = req['action']
        automatic = '_id' in req
        if action in ('rest_inspect', 'rest_capture'):
            if any(not key.startswith('_') and key != 'action' for key in req):
                raise ValueError('Rest inspection/capture accepts no supplied joint target or motion settings')
            return self.inspect_or_capture_rest(capture=action == 'rest_capture')
        if action == 'recover_control':
            return self.recover_control(req)
        if action == 'shutdown' or action == 'disable' and getattr(self, 'exit_after_disable', False):
            return self.shutdown()
        if getattr(self, 'exit_after_disable', False):
            if action == 'mode' and req.get('mode') == 'Hold':
                return self.ctrl.pause_shutdown()
            if (action in ('mode', 'rest', 'enable', 'joint', 'pose', 'pose_recall',
                           'acknowledge_fault', 'gravity_trial', 'gravity_configure',
                           'gravity_feedforward', 'gravity_payload', 'position_speed_limit',
                           'speed_limits', 'motion_limits')
                    or action.startswith('calibration_') and action not in ('calibration_state', 'calibration_cancel')
                    or action.startswith('laser_') and action not in ('laser_state', 'laser_cancel')
                    or action in ('scan_start', 'scan_resume', 'weld_start', 'weld_replay')):
                raise RuntimeError('Shutdown owns robot motion; use Hold to pause or Shutdown to retry')
            if action == 'laser_cancel':
                return self.ctrl.pause_shutdown()
        guard = getattr(self, 'host_guard', None)
        if guard is not None and (
                action in ('enable', 'joint', 'pose', 'pose_recall', 'gravity_trial',
                           'calibration_auto', 'calibration_start', 'calibration_capture',
                           'laser_auto', 'laser_capture', 'scan_start', 'scan_resume', 'weld_start', 'weld_replay',
                           'camera_laser_configure')
                or action == 'mode' and req.get('mode') != 'Hold'
                or action == 'laser_preview' and req.get('enabled')
                or action == 'camera_stream' and req.get('enabled')):
            guard.check(force=action == 'enable')
        if action.startswith('weld_'):
            weld = getattr(self, 'weld', None)
            if weld is None:
                raise RuntimeError('Weld seam service is unavailable')
            if action in ('weld_start', 'weld_replay'):
                if weld.busy:
                    raise RuntimeError('A weld path operation already owns robot motion')
                self.can_start_weld(wait_for_fresh=True)
            return weld.dispatch(req)
        if action.startswith('scan_'):
            if action == 'scan_plan_probe':
                from autoweld.scan_service import ScanPlanProbe
                if getattr(self, 'scan_plan_probe', None) is None:
                    self.scan_plan_probe = ScanPlanProbe(self)
                return self.scan_plan_probe.dispatch(req)
            if action == 'scan_probe':
                return self.probe_scan_pose(req)
            scan = getattr(self, 'scan', None)
            if scan is None:
                raise RuntimeError('Laser scanning service is unavailable')
            if action == 'scan_start':
                if scan.busy:
                    raise RuntimeError('A scan already owns robot motion')
                self.can_start_scan(wait_for_fresh=True)
            return scan.dispatch(req)
        if action.startswith('laser_'):
            laser = getattr(self, 'laser', None)
            if not self.camera or laser is None:
                raise RuntimeError('Laser calibration service is unavailable')
            if action == 'laser_state':
                return self.laser_state()
            return laser.dispatch(req)
        if action.startswith(('camera_', 'calibration_')):
            if not self.camera:
                raise RuntimeError('Camera service is unavailable')
            if action.startswith('calibration_') and self.calibration:
                return self.calibration.dispatch(req)
            if action == 'calibration_capture':
                req = {**req, 'tcp_pose': self.tcp_pose().tolist()}
            return self.camera.dispatch(req)
        if action == 'mode':
            mode = req['mode']
            if mode == 'Gravity compensation':
                self.ctrl.gravity()
            elif mode in ('Hold', 'IK', 'Joint'):
                self.ctrl.hold(automatic=automatic)
                self.ctrl.mode = mode
            else:
                raise ValueError('Unknown operation mode')
        elif action == 'pose':
            pose = np.eye(4)
            pose[:3, 3] = req['position']
            if 'rpy' in req:
                pose[:3, :3] = Rotation.from_euler('xyz', req['rpy']).as_matrix()
            else:
                w, x, y, z = req['wxyz']
                pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            self.move_pose(pose, automatic)
        elif action == 'joint':
            arm = np.asarray(req['q'], dtype=float)
            if arm.shape != (6,):
                raise ValueError('Expected six joint radians')
            self.ctrl.command(np.r_[arm, self.ctrl.gripper_target], 'Hold', automatic=automatic)
        elif action == 'rest':
            self.ctrl.return_rest(automatic=automatic)
        elif action == 'disable':
            self.ctrl.request_disable(automatic=automatic)
        elif action == 'enable':
            self.ctrl.enable()
        elif action == 'acknowledge_fault':
            self.ctrl.acknowledge_fault()
        elif action in ('gravity_configure', 'gravity_feedforward'):
            self.ctrl.configure_gravity(req['factors'], model=req.get('model', 'sdk'))
            profile = {'factors': self.ctrl.gravity_factors.tolist(), 'model': self.ctrl.gravity_model,
                'model_fingerprint': getattr(self, 'model_fingerprint', None)}
            persist = bool(req.get('persist', True))
            if persist:
                atomic_json(self.runtime / 'gravity-config.json', profile)
            self.gravity_tuning = {'status': 'configured' if persist else 'trial',
                'message': 'Gravity factors applied and saved' if persist else 'Temporary commissioning factors; not saved',
                'profile': profile}
            return self.gravity_telemetry()
        elif action == 'gravity_payload':
            self.ctrl.health()
            if (self.ctrl.motion_plan is not None or self.ctrl.disable_pending
                    or getattr(self.ctrl, 'mode_transition', False) or getattr(self.ctrl, 'cooldown_active', False)):
                raise RuntimeError('Stop robot motion before changing payload')
            mass = float(req['mass_kg'])
            com = np.asarray(req['com_flange_m'], float)
            if not np.isfinite(mass) or mass < 0 or com.shape != (3,) or not np.isfinite(com).all():
                raise ValueError('Expected a nonnegative payload mass and three finite COM coordinates in meters')
            self.ctrl.robot.configure_payload(mass, com.tolist())
            self.gravity_payload = {'mass_kg': mass, 'com_flange_m': com.tolist(),
                'model_fingerprint': getattr(self, 'model_fingerprint', None),
                'source': str(req.get('source', 'configured')), 'validated': bool(req.get('validated', False))}
            atomic_json(self.runtime / 'gravity-payload.json', self.gravity_payload)
            return self.gravity_telemetry()
        elif action == 'gravity_tuning_status':
            self.gravity_tuning = {'status': str(req['status']), 'message': str(req.get('message', '')),
                'metrics': req.get('metrics'), 'model_fingerprint': getattr(self, 'model_fingerprint', None),
                'timestamp': time.time()}
            atomic_json(self.runtime / 'gravity-tuning.json', self.gravity_tuning)
            return self.gravity_tuning
        elif action == 'gravity_telemetry':
            return self.gravity_telemetry(include_regressor=req.get('include_regressor', True))
        elif action == 'gravity_test_envelope':
            self.ctrl.configure_gravity_test_envelope(float(req.get('excursion_rad', .49)))
            return {'excursion_rad': self.ctrl.test_excursion, 'persistent': False}
        elif action == 'gravity_test_preflight':
            return self.gravity_test_preflight(req)
        elif action == 'gravity_trial':
            self.ctrl.health()
            if not self.ctrl.ready_for_waypoint():
                raise RuntimeError('Commissioning trial requires healthy stationary idle control')
            self.ctrl._switch_arm_control_mode('MIT')
            try:
                self.ctrl.robot.configure_gravity(self.ctrl.gravity_factors, model='sdk', reference=self.ctrl.rest[:6])
                # A firmware transition may need a brief measured-pose PD
                # settling interval. Never begin a free-drive probe while that
                # hold is still moving, and do not interpret it as a fault.
                self.ctrl.robot.settle_measured_hold()
                self.ctrl.robot.start_gravity_trial(float(req.get('duration_s', .5)),
                    float(req.get('max_excursion_rad', .04)), float(req.get('max_speed_rad_s', .3)))
                self.ctrl.mode = 'Gravity compensation'
            except Exception:
                self.ctrl.hold()
                raise
            return {'status': 'started', 'response': 'SDK automatically restores measured MIT PD hold'}
        elif action == 'position_speed_limit':
            self.ctrl.configure_position_speed_limit(float(req['limit_rad_s']))
        elif action == 'speed_limits':
            return self.configure_speed_limits(req['linear_velocity'], req['joint_velocity'])
        elif action == 'motion_limits':
            if self.ctrl.motion_plan is not None or self.ctrl.disable_pending:
                raise RuntimeError('Wait for motion before changing limits')
            self.planner.limits = MotionLimits(**req['limits'])
            atomic_json(self.runtime / 'motion-limits.json', asdict(self.planner.limits))
        elif action == 'pose_save':
            self.ctrl.health()
            name = str(req['name']).strip()
            if not name or len(name) > 80:
                raise ValueError('Use a pose name of 1–80 characters')
            pose = self.tcp_pose()
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            self.saved_poses[name] = {'q': self.ctrl.positions()[:6].tolist(),
                'position': pose[:3, 3].tolist(), 'wxyz': [float(quat[3]), *quat[:3].tolist()]}
            atomic_json(self.saved_pose_path, self.saved_poses)
        elif action == 'pose_recall':
            saved = self.saved_poses[req['name']]
            self.dispatch({'action': 'joint', 'q': saved['q'], **({'_id': req['_id']} if automatic else {})})
        elif action == 'sync':
            pass
        else:
            raise ValueError(f'Unknown action: {action}')
        self.message = f'{action}: accepted'
        self.fault = None

    def handle_request(self, req):
        action = req['action']
        started = time.monotonic()
        self.notice_recovery_cancellation(req)
        if action in ('mode', 'rest', 'disable', 'shutdown', 'enable', 'joint', 'pose_recall',
                      'acknowledge_fault', 'laser_auto', 'scan_start', 'scan_resume', 'weld_start', 'weld_replay'):
            self.pending_pose = None
        try:
            cutoff = getattr(self, '_recovery_discard_before', None)
            queued_at = req.get('_service_queued_monotonic', float('inf'))
            window = getattr(self, '_recovery_attempt_window', None)
            rejected_window = window is not None and window[0] <= queued_at <= window[1]
            if (recovery_discards(req) and (req.get('_blocked_during_recovery') or rejected_window
                    or cutoff is not None and queued_at <= cutoff)):
                discarded = {'action': action, 'id': req.get('_id'), 'reason': 'admitted before recovery completed'}
                self._recovery_discarded.append(discarded)
                raise RuntimeError('Discarded pre-recovery work; submit a new explicit command after recovery')
            weld = getattr(self, 'weld', None)
            if weld is not None and weld.busy and action != 'shutdown' and not self.exit_after_disable:
                if (action in ('mode', 'rest', 'disable', 'enable', 'joint', 'pose', 'pose_recall',
                               'acknowledge_fault', 'camera_config', 'camera_analysis', 'camera_laser_configure',
                               'camera_laser_cancel', 'motion_limits', 'speed_limits',
                               'position_speed_limit', 'gravity_trial', 'gravity_payload')
                        or action == 'camera_stream' and not req.get('enabled')
                        or action.startswith('calibration_') and action not in ('calibration_state', 'calibration_cancel')
                        or action.startswith('laser_') and action not in ('laser_state', 'laser_cancel')):
                    weld.cancel()
            scan = getattr(self, 'scan', None)
            if scan is not None and scan.busy and action != 'shutdown' and not self.exit_after_disable:
                # Cancel before another controller/camera operation can own a target.
                if (action in ('mode', 'rest', 'disable', 'enable', 'joint', 'pose', 'pose_recall',
                               'acknowledge_fault', 'camera_config', 'camera_analysis', 'camera_laser_configure',
                               'motion_limits', 'speed_limits',
                               'position_speed_limit', 'gravity_trial', 'gravity_payload')
                        or action == 'camera_stream' and not req.get('enabled')
                        or action.startswith('calibration_') and action not in ('calibration_state', 'calibration_cancel')
                        or action.startswith('laser_') and action not in ('laser_state', 'laser_cancel')):
                    scan.cancel()
            laser = getattr(self, 'laser', None)
            if laser is not None and laser.busy and action != 'shutdown' and not self.exit_after_disable:
                if (action in ('mode', 'rest', 'disable', 'enable', 'joint', 'pose', 'pose_recall',
                               'acknowledge_fault', 'camera_config', 'camera_analysis', 'camera_laser_configure',
                               'calibration_auto', 'calibration_start',
                               'calibration_capture', 'calibration_reset', 'calibration_solve')
                        or action == 'camera_stream' and not req.get('enabled')):
                    laser.cancel()
            # An operator control command takes ownership from automatic
            # calibration before any queued TCP target can start another move.
            calibration = getattr(self, 'calibration', None)
            if calibration and action != 'shutdown' and not self.exit_after_disable and (action in ('mode', 'rest', 'disable', 'enable',
                                         'joint', 'pose', 'pose_recall', 'acknowledge_fault', 'camera_config', 'camera_analysis')
                    or action == 'camera_stream' and not req.get('enabled')):
                if calibration.state().get('status') in ('starting', 'moving', 'settling',
                        'capturing', 'returning', 'capturing_manual', 'solving', 'hunting'):
                    calibration.dispatch({'action': 'calibration_cancel'})
            if action == 'pose' and getattr(self, 'host_guard', None) is not None:
                self.host_guard.check()
            if (action == 'pose' and '_id' not in req and not self.ctrl.closed
                    and not self.exit_after_disable
                    and not self.ctrl.disable_pending and not self.ctrl.motion_fault()
                    and (self.ctrl.motion_plan is not None or not self.ctrl.ready_for_waypoint())):
                self.pending_pose = (req.copy(), time.monotonic())
                result = {'ok': True, 'status': 'queued', 'message': 'Latest TCP target queued until arm stops'}
                self.message = result['message']
            else:
                if action == 'pose':
                    self.pending_pose = None
                detail = self.dispatch(req)
                result = {'ok': True, 'status': 'accepted'}
                if detail is not None:
                    result['result'] = detail
                if action == 'recover_control' and detail.get('phase') in ('blocked', 'busy'):
                    result.update(ok=False, status='rejected', error=detail.get('error'))
        except Exception as exc:
            self.message = str(exc)
            result = {'ok': False, 'status': 'rejected', 'error': str(exc)}
            if isinstance(exc, MotionRejection):
                result['diagnostic'] = exc.diagnostic
            elif hasattr(exc, 'diagnostic'):
                result['diagnostic'] = exc.diagnostic
            logging.warning('Request rejected: %s', exc)
        self.request_status = compact_telemetry({'timestamp': time.time(), 'action': action,
            'processing_s': time.monotonic()-started, **result})
        with (self.runtime / 'command-events.jsonl').open('a') as stream:
            stream.write(json.dumps({**self.request_status, 'request': compact_telemetry(req)})+'\n')
        return result

    def process_pending_pose(self):
        if self.pending_pose is None:
            return
        req, queued_at = self.pending_pose
        if self.ctrl.closed or self.ctrl.disable_pending or self.ctrl.motion_fault():
            self.pending_pose = None
        elif time.monotonic()-queued_at > 15:
            self.pending_pose = None
            self.request_status = {'timestamp': time.time(), 'action': 'pose', 'ok': False,
                'status': 'expired', 'error': 'Queued target expired; send again after the arm stops'}
        elif self.ctrl.motion_plan is None and self.ctrl.ready_for_waypoint():
            self.pending_pose = None
            self.handle_request(req)

    def state(self):
        q = self.ctrl.positions()
        pose = self.planner.fk(q).copy()
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        robot = self.ctrl.robot
        temperatures = robot.get_temperature_state() if hasattr(robot, 'get_temperature_state') else None
        errors = robot.get_motor_power_state() if hasattr(robot, 'get_motor_power_state') else {}
        gravity = self.gravity_telemetry()
        camera_state = self.camera.state() if self.camera else None
        sdk = self.ctrl.sdk_status()
        sdk_stopped = sdk['required'] and not sdk['control_alive']
        return {'timestamp': time.time(), 'live': self.ctrl.live, 'mode': self.ctrl.mode,
            'controller': self.ctrl.control_law, 'closed': self.ctrl.closed,
            'arm_control_modes': gravity['arm_control_modes'],
            'mode_transition': gravity['mode_transition'], 'mode_transition_error': gravity['mode_transition_error'],
            'gravity': gravity['gravity'], 'torque': gravity['torque'],
            'gravity_factors': gravity['gravity_factors'], 'gravity_model': gravity['gravity_model'],
            'gravity_tuning': gravity['tuning'], 'gravity_payload': gravity['payload'],
            'model_fingerprint': gravity['model_fingerprint'], 'joint_velocity_rad_s': gravity['vel_rad_s'],
            'reported_effort_nm': gravity['reported_effort_nm'],
            'disable_pending': self.ctrl.disable_pending,
            'rest_verified': self.ctrl.rest_verified and (not sdk['required'] or sdk['feedback_fresh']),
            'shutdown': self.ctrl.shutdown_status,
            'rest_tolerance_deg': float(np.rad2deg(REST_TOL)),
            'at_rest': self.ctrl.at_rest() and (not sdk['required'] or sdk['feedback_fresh']),
            'q_rad': q[:6].tolist(), 'target_rad': self.ctrl.target[:6].tolist(),
            'rest_rad': self.ctrl.rest[:6].tolist(), 'gripper_position_normalized': float(q[6]),
            'error_deg': np.rad2deg(self.ctrl.target[:6]-q[:6]).tolist(),
            'accuracy_tolerance_deg': np.rad2deg(ANGLE_TOL).tolist(),
            'tcp_position_m': pose[:3, 3].tolist(), 'tcp_wxyz': [float(quat[3]), *quat[:3].tolist()],
            'tcp_rpy_rad': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz').tolist(),
            'motion_active': self.ctrl.motion_plan is not None, 'motion_plan': self.ctrl.last_motion,
            'motion_limits': asdict(self.planner.limits),
            'position_speed_limit_rad_s': self.ctrl.position_speed_limit,
            'saved_poses': self.saved_poses, 'last_request': self.request_status,
            'pending_tcp_target': self.pending_pose is not None,
            'motor_errors': errors, 'temperatures': temperatures,
            'cooldown_active': getattr(self.ctrl, 'cooldown_active', False),
            'feedback_warning': getattr(self.ctrl, 'feedback_warning', None),
            'sdk': sdk,
            'host_health': self.host_guard.status() if getattr(self, 'host_guard', None) is not None else None,
            'recovery_host_override': self.recovery_host_override_status(),
            'recovery': ({**self.recovery.state(), 'discarded_work': list(self._recovery_discarded)}
                if getattr(self, 'recovery', None) is not None else None),
            'feedback_source': ('saved rest record' if self.ctrl.cached_rest_display else
                'read-only SDK disabled snapshot; not a streaming owner' if self.ctrl.closed
                    and getattr(self, 'disabled_rest_inspection', None) is not None else
                ('SDK feedback' if sdk['feedback_fresh'] else 'cached SDK feedback; current hardware state unverified')
                if self.ctrl.live else 'ideal simulation'),
            'message': self.message,
            'disabled_rest_inspection': getattr(self, 'disabled_rest_inspection', None),
            'fault': self.fault or ('SDK control stopped; current hardware state unverified' if sdk_stopped else None),
            'motion_fault': self.ctrl.motion_fault(),
            'camera': camera_state,
            'calibration': self.calibration.state() if self.calibration else None,
            'laser': self.laser_state(camera_state=camera_state or {}),
            'scan_plan_probe': (self.scan_plan_probe.state() if getattr(self, 'scan_plan_probe', None) else None),
            'scan': (self.scan.state(camera_state=camera_state or {}) if getattr(self, 'scan', None) else None),
            'weld': (self.weld.state(camera_state=camera_state or {}) if getattr(self, 'weld', None) else None)}

    def report_tick_failure(self, error):
        """Preserve the first failure and rate-limit repeats without claiming torque."""
        self.fault = self.message = str(error)
        sdk = self.ctrl.sdk_status()
        if sdk['required'] and not sdk['control_alive']:
            weld = getattr(self, 'weld', None)
            if weld is not None and weld.busy:
                weld.abort_for_shutdown()
            scan = getattr(self, 'scan', None)
            if scan is not None and scan.busy:
                scan.abort_for_shutdown()  # No Hold or SDK call after owner failure.
            self.pending_pose = None
            self.ctrl.rest_verified = False
            self.ctrl.settled_since = None
            # These are application references only; no SDK command, restart,
            # fault acknowledgement or motor power transition is safe here.
            self.ctrl.motion_plan = None
        signature = (type(error).__name__, str(error))
        previous = getattr(self, '_tick_failure_signature', None)
        now = time.monotonic()
        if signature != previous:
            self._tick_failure_signature = signature
            self._tick_failure_reported_at = now
            if sdk['required'] and not sdk['control_alive']:
                if not getattr(self, '_sdk_failure_persisted', False):
                    try:
                        record = {'timestamp': time.time(), 'error': str(error), 'sdk': sdk,
                                  'cached_q_rad': self.ctrl.positions()[:6].tolist(),
                                  'rest_rad': self.ctrl.rest[:6].tolist()}
                        atomic_json(self.runtime / 'evidence' / f'sdk-failure-{time.time_ns()}.json', record)
                        atomic_json(self.runtime / 'sdk-failure.json', record)
                        self._sdk_failure_persisted = True
                    except Exception:
                        logging.exception('Could not persist SDK failure diagnostics')
                logging.error('Service tick failed; SDK control stopped; motor power and physical pose are unverified',
                              exc_info=error)
            else:
                logging.error('Service tick failed; inspect SDK diagnostics for control and feedback status',
                              exc_info=error)
        elif now - self._tick_failure_reported_at >= 5.:
            self._tick_failure_reported_at = now
            logging.error('Service tick remains blocked: %s (SDK control alive: %s)', error, sdk['control_alive'])

    def publish(self):
        # Dedicated shutdown converts health exceptions into a retryable blocked
        # state. Preserve fatal SDK evidence even when tick therefore returns.
        sdk = self.ctrl.sdk_status()
        if sdk['required'] and not sdk['control_alive'] and not getattr(self, '_sdk_failure_persisted', False):
            detail = (sdk['failure'] or {}).get('message') or 'SDK control stopped; current hardware state unverified'
            self.report_tick_failure(RuntimeError(detail))
        state = self.state()
        if state.get('last_request'):
            self.request_status = compact_telemetry(state['last_request'])
            state['last_request'] = self.request_status
        serialized = json.dumps(state, allow_nan=False)
        with self.snapshot_lock:
            self.snapshot_json = serialized
        # Keep the acquisition payload out of on-disk telemetry.
        if state.get('camera'):
            state['camera'] = {k: v for k, v in state['camera'].items() if k != 'frame_jpeg_base64'}
            if state['camera'].get('laser'):
                state['camera']['laser'] = self._laser_telemetry(state['camera']['laser'])
        if state.get('laser'):
            state['laser'] = self._laser_telemetry(state['laser'])
        if state.get('scan'):
            state['scan'] = self._laser_telemetry(state['scan'])
        if state.get('weld'):
            state['weld'] = self._laser_telemetry(state['weld'])
        atomic_json(self.runtime / 'status.json', state)

    @staticmethod
    def _laser_telemetry(state):
        """Keep raw images and dense point arrays out of periodic disk writes."""
        return compact_telemetry(state)

    def tick(self):
        urgent = []
        inbox = getattr(self, '_urgent_events', None)
        if inbox is not None:
            while not inbox.empty():
                urgent.append(inbox.get_nowait())
        guard = getattr(self, 'host_guard', None)
        override = self.recovery_host_override_status()
        if guard is not None and not self.ctrl.closed and not self.exit_after_disable:
            host = guard.sample()
            if not host['safe'] and override.get('active'):
                warning = '; '.join(host['reasons'])
                self._recovery_host_override['host_warning'] = warning
                now = time.monotonic()
                if now-getattr(self, '_last_recovery_host_warning', 0.) >= 5.:
                    self._last_recovery_host_warning = now
                    logging.warning('Recovered owner retains control despite host pressure: %s', warning)
            elif not host['safe']:
                logging.error('Host pressure requires controlled shutdown: %s', '; '.join(host['reasons']))
                self.handle_request({'action': 'shutdown', 'reason': 'host_pressure', 'host_health': host})
            elif override.get('active'):
                self._recovery_host_override['host_warning'] = None
        for path in sorted((self.runtime / 'requests').glob('*.json')):
            try:
                req = json.loads(path.read_text())
                stamp_admission(req, file_mtime=path.stat().st_mtime)
                self.notice_recovery_cancellation(req)
                req['_id'] = path.stem
                self.events.put(req)
                path.unlink()
            except Exception as exc:
                self.message = str(exc)
                path.rename(path.with_suffix('.invalid'))
        pending = urgent
        while not self.events.empty():
            req = self.events.get_nowait()
            if pending and req['action'] == pending[-1]['action'] == 'pose' and '_id' not in req and '_id' not in pending[-1]:
                pending[-1] = req
            else:
                pending.append(req)
        for req in pending:
            result = self.handle_request(req)
            if '_id' in req:
                atomic_json(self.runtime / 'responses' / f"{req['_id']}.json", result)
        self.ctrl.tick()
        self.process_pending_pose()
        weld = getattr(self, 'weld', None)
        if weld is not None:
            if self.ctrl.cooldown_active and weld.busy:
                weld.abort_for_cooldown()
            weld.tick()
        scan = getattr(self, 'scan', None)
        if scan is not None:
            if self.ctrl.cooldown_active and scan.busy:
                scan.abort_for_cooldown()
            # Continue draining persistence after shutdown; it cannot request motion.
            scan.tick()
        laser = getattr(self, 'laser', None)
        if laser is not None and not self.exit_after_disable:
            if self.ctrl.cooldown_active and laser.busy:
                laser.abort_for_cooldown()
            laser.tick()
        if self.calibration and not self.exit_after_disable:
            if self.ctrl.cooldown_active and self.calibration.status in (
                    'starting', 'hunting', 'moving', 'settling', 'capturing',
                    'capturing_manual', 'solving', 'returning'):
                self.calibration.abort_for_cooldown()
            self.calibration.tick()
        if self.exit_after_disable and self.ctrl.closed:
            self.publish()
            self.running = False


def check_server_port(host, port):
    """Fail before SDK startup when Viser cannot bind its requested endpoint."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, int(port)))
    except OSError as exc:
        raise RuntimeError(f'Cannot serve AutoWeld at {host}:{port}: {exc}; robot was not connected') from exc


def disabled_display_robot(runtime, model_path, model):
    """Construct a saved display only; this path never opens the CAN SDK chain."""
    from i2rt.robots.sim_robot import SimRobot

    previous = json.loads((Path(runtime) / 'rest.json').read_text())
    arm = np.asarray(previous['arm_rad'], float)
    if arm.shape != (6,) or not np.isfinite(arm).all():
        raise ValueError('A valid six-axis captured rest is required for --start-disabled')
    return SimRobot(model_path, n_dofs=6, initial_qpos=arm, compiled_model=model)


def run_application(*, live=False, port=8080, channel='can0', camera_source='', runtime=None,
                    step=None, start_disabled=False):
    """Process entrypoint. Only this service creates SDK connections/devices."""
    import fcntl
    import signal

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    from autoweld.app import App
    from autoweld.joint_calibration import load_joint_calibration, verify_disabled_start
    from autoweld.models import URDF, compiled_model, tool_model

    logging.basicConfig(level=logging.INFO)
    check_server_port('127.0.0.1', port)
    runtime = Path(runtime or ('runtime/live' if live else 'runtime/sim'))
    host_guard = None
    if live:
        from autoweld.host_health import HostHealthGuard
        host_guard = HostHealthGuard()
    hardware_lock = None
    if live:
        hardware_lock = (Path('/tmp') / f'autoweld-{Path(channel).name}.lock').open('a')
        try:
            fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another AutoWeld controller owns this CAN channel') from None
        if list((runtime / 'requests').glob('*.json')):
            raise SystemExit('Unprocessed requests exist; review them before connecting live hardware')
    model_path = tool_model()
    joint_calibration = load_joint_calibration()

    def connect_robot(gripper_limits=None, last_q=None, arm_control_mode='POS_VEL', *,
                      admission_guard=None, enable_deadline=None, on_created=None,
                      recovery_rest_q=None, recovery_joint_limits=None, recovery_near_rest_tolerance_rad=.15):
        check_connection_admission(host_guard, admission_guard=admission_guard,
            recovery_rest_q=recovery_rest_q, recovery_joint_limits=recovery_joint_limits)
        bounds = compiled_model().jnt_range[:6].copy()
        offsets = np.asarray(joint_calibration['fixed_motor_offsets_rad'], float)
        if live:
            check_connection_admission(host_guard, admission_guard=admission_guard,
                recovery_rest_q=recovery_rest_q, recovery_joint_limits=recovery_joint_limits)
            verify_disabled_start(channel, bounds, offsets)
        if admission_guard is not None:
            admission_guard()
        robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM,
            gripper_type=GripperType.NO_GRIPPER, sim=not live, zero_gravity_mode=False,
            use_coulomb_friction=False, enable_auto_recovery=False,
            arm_control_mode='POS_VEL', xml_path_override=model_path, compiled_model=compiled_model(),
            arm_joint_limits_override=bounds, arm_motor_offsets_override=offsets,
            **({'admission_guard': admission_guard, 'enable_deadline': enable_deadline,
                'recovery_rest_q': recovery_rest_q, 'recovery_joint_limits': recovery_joint_limits,
                'recovery_near_rest_tolerance_rad': recovery_near_rest_tolerance_rad}
                if admission_guard is not None else {}))
        if on_created is not None:
            on_created(robot)
        if not live:
            q = np.zeros(6) if last_q is None else np.asarray(last_q)[:6].copy()
            outside = (q < bounds[:, 0]) | (q > bounds[:, 1])
            q[outside] = np.mean(bounds[outside], axis=1)
            robot.command_joint_pos(q)
        fingerprint = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        payload = gravity_record(runtime, 'gravity-payload.json', fingerprint)
        if payload is not None and hasattr(robot, 'configure_payload'):
            if admission_guard is not None:
                admission_guard()
            robot.configure_payload(payload['mass_kg'], payload['com_flange_m'])
        if arm_control_mode != 'POS_VEL':
            robot.switch_arm_control_mode(arm_control_mode)
        return robot

    robot = (disabled_display_robot(runtime, model_path, compiled_model())
             if start_disabled else connect_robot())
    camera = None
    try:
        from autoweld.camera import CameraService
        camera = CameraService.get_instance(runtime)
    except (ImportError, AttributeError) as exc:
        logging.warning('Camera service unavailable: %s', exc)
    def recovery_reader():
        from i2rt.robots.disabled_recovery import DisabledArmRecovery
        return DisabledArmRecovery(channel, offsets_rad=np.asarray(joint_calibration['fixed_motor_offsets_rad']))

    def disabled_reader():
        from i2rt.robots.encoder_teaching import ReadOnlyArmEncoder
        return ReadOnlyArmEncoder(channel, offsets_rad=np.asarray(joint_calibration['fixed_motor_offsets_rad']))

    service = RobotService(robot, model_path, runtime, live=live, robot_factory=connect_robot,
                           start_disabled=start_disabled, camera=camera, host_guard=host_guard,
                           recovery_factory=recovery_reader if live else None,
                           disabled_reader_factory=disabled_reader if live else None)
    try:
        from autoweld.calibration import CalibrationCoordinator
        # Calibration uses the same JSON service/motion planner as manual moves.
        service.calibration = CalibrationCoordinator(camera, service.tcp_pose, service.move_calibration_pose,
            service.ctrl.ready_for_waypoint, service.ctrl.hold, runtime, validate_pose=service.validate_pose)
    except (ImportError, TypeError) as exc:
        logging.warning('Automatic calibration unavailable: %s', exc)
    if camera is not None:
        try:
            from autoweld.laser_calibration import LaserCalibrationCoordinator
            from autoweld.models import camera_from_flange
            service.laser = LaserCalibrationCoordinator(camera, service.tcp_pose,
                lambda pose, automatic=True: service.move_calibration_pose(pose, automatic, owner=service.laser),
                service.ctrl.ready_for_waypoint, service.ctrl.hold, service.move_laser_saved_pose,
                flange_from_nominal_tcp=camera_from_flange(), preempt=service.preempt_for_laser,
                can_start=service.can_start_laser, validate_pose=service.validate_pose)
        except (ImportError, TypeError) as exc:
            logging.warning('Laser calibration unavailable: %s', exc)
        try:
            from autoweld.scanning import LaserScanCoordinator
            service.scan = LaserScanCoordinator(camera, service.tcp_pose, service.move_scan_pose,
                service.ctrl.ready_for_waypoint, service.ctrl.hold, service.move_laser_saved_pose, runtime,
                feedback=service.scan_feedback, can_start=service.can_start_scan,
                validate_pose=service.validate_scan_pose, model_fingerprint=service.model_fingerprint,
                move_is_validated=True)
            from autoweld.scan_service import ContinuousScanService
            service.scan = ContinuousScanService(service, service.scan)
        except (ImportError, TypeError) as exc:
            logging.warning('Laser scanning unavailable: %s', exc)
        try:
            from autoweld.weld_seam_service import WeldSeamService
            service.weld = WeldSeamService(service)
        except (ImportError, TypeError) as exc:
            logging.warning('Weld seam finding unavailable: %s', exc)
    frontend = None
    try:
        frontend = App(JsonConnection(service), URDF, port=port)
    except BaseException:
        logging.exception('Frontend startup failed; returning to verified rest before shutdown')
        service.handle_request({'action': 'shutdown'})
    if step:
        logging.warning('STEP placement is not enabled in the focused calibration UI: %s', step)
    if camera_source:
        logging.info('Camera remains off until Stream camera is selected')

    def stop_requested(*_):
        service.enqueue_signal({'action': 'shutdown'})
        logging.info('Exit requested: dedicated shutdown owns return to rest')

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, stop_requested)
    last_render = 0.
    print(f'AutoWeld ready: https://faraz-precision-3640-tower.tail53558.ts.net:{port} | runtime={runtime}', flush=True)
    try:
        while service.running:
            started = time.monotonic()
            try:
                service.tick()
            except Exception as exc:
                service.report_tick_failure(exc)
            else:
                service._tick_failure_signature = None
            if started-last_render >= .2:
                try:
                    service.publish()
                    if frontend:
                        frontend.render()
                except Exception as exc:
                    logging.exception('UI rendering failed; control continues: %s', exc)
                last_render = started
            time.sleep(max(0., .02-(time.monotonic()-started)))
    finally:
        # An exception must never close the SDK while the arm is away from
        # captured rest. Keep the SDK alive if recovery cannot be verified.
        if live and not service.ctrl.closed:
            try:
                service.handle_request({'action': 'shutdown'})
            except Exception:
                logging.exception('Rest disable request failed; inspect SDK control and motor power status')
            while not service.ctrl.closed:
                try:
                    service.tick()
                    service.publish()
                    if frontend:
                        frontend.render()
                except Exception as exc:
                    service.report_tick_failure(exc)
                time.sleep(.02)
        for resource in (frontend, getattr(service, 'weld', None), getattr(service, 'scan_plan_probe', None),
                         getattr(service, 'scan', None), getattr(service, 'laser', None), service.calibration, camera):
            if resource:
                try:
                    resource.close()
                except Exception:
                    logging.exception('Resource cleanup failed')
        if not live and not service.ctrl.closed:
            service.ctrl.robot.close()
        if hardware_lock:
            hardware_lock.close()
