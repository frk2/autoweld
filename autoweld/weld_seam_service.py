"""Single-owner, stationary weld seam acquisition and replay orchestration.

Weldfinder computes camera-frame seam geometry. This module only associates that
geometry with measured robot feedback, follows it, and retains an ordered path.
All effects execute on the existing service's tick; workers perform file/vision
work only. The replayed tool reference is explicitly the camera optical frame at
150 mm clearance, not an uncalibrated physical welding torch.
"""

import copy
import hashlib
import json
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.calibration import transform


class _OpticalVisibilityUnresolved(ValueError):
    """Only bounded native optical recovery exhaustion may switch branches."""


@dataclass(frozen=True)
class WeldSettings:
    min_distance_mm: float = 25.
    max_distance_each_way_mm: float = 100.
    start_pose: str = 'Weld Finder'
    standoff_mm: float = 150.
    step_mm: float = 1.
    end_confirmations: int = 3

    def validate(self):
        values = (self.min_distance_mm, self.max_distance_each_way_mm, self.standoff_mm, self.step_mm)
        if (any(type(v) not in (int, float) for v in values) or not np.isfinite(values).all()
                or not 1 <= self.min_distance_mm <= 100
                or not max(5., self.min_distance_mm/2) <= self.max_distance_each_way_mm <= 100
                or self.start_pose != 'Weld Finder' or self.standoff_mm != 150. or self.step_mm != 1.
                or type(self.end_confirmations) is not int or self.end_confirmations != 3):
            raise ValueError('Weld finding requires Weld Finder, 1–100 mm minimum, bounded travel, '
                             '1 mm stations, 150 mm standoff and three end confirmations')
        return self


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(_json_value(value), allow_nan=False, indent=2))
    temporary.replace(path)
    return str(path)


def _write_observation(path, payload, previous_files):
    """Bind the exact native file bytes, including rejected/centering views."""
    _atomic_json(path, payload)
    entry = dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                 observation_index=len(previous_files)+1, status=payload['result'].get('status'), disposition='pending')
    files = [*copy.deepcopy(previous_files), entry]
    _atomic_json(Path(path).parent/'observations.json', dict(schema='autoweld.weld-observations.v1',
        native_files=files, native_files_sha256=_digest(files)))
    return entry


def _pose_errors(actual, expected):
    return (float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
            float(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude()))


def preflight_replay(existing_planner, start, poses, bounds, cancelled, progress):
    """Prove the whole replay with worker-private DATA and the one shared MODEL.

    No controller/SDK callback is passed into this worker. The main owner still
    rechecks current health and plans each actual command through its normal
    motion path; this proof prevents a late geometric blocker after replay starts.
    """
    from autoweld.diagnostics import solve_pose
    from autoweld.scan_plan_probe import private_planner, private_self_collision

    planner = private_planner(existing_planner)
    seed, bounds = np.array(start, float, copy=True), np.array(bounds, float, copy=True)
    gripper, reports, started = float(seed[6]), [], time.monotonic()

    def validate(q):
        if (cancelled() or time.monotonic()-started > 300.):
            raise RuntimeError('Weld replay preflight cancelled or exceeded its five-minute bound')
        if (np.asarray(q).shape != (7,) or not np.isfinite(q).all()
                or not np.isclose(q[6], gripper, atol=1e-12, rtol=0)
                or np.any(q[:6] < bounds[:, 0]) or np.any(q[:6] > bounds[:, 1])):
            raise ValueError('Weld replay exceeds fixed gripper or installed joint bounds')

    for index, pose in enumerate(poses):
        validate(seed)
        solution = solve_pose(planner.kin, transform(pose), planner.site_name,
                              planner._configuration(seed), stage='weld_replay_preflight')
        target = np.r_[solution[:6], gripper]
        validate(target)
        plan = planner.plan(seed, target, cartesian=pose, bounds=bounds, validate=validate,
                            collision=lambda q: private_self_collision(planner, q))
        reports.append(plan.report)
        seed = target
        progress(index+1)
    return dict(segment_count=len(poses), reports=reports, start_q=np.asarray(start).tolist(),
                bounds=bounds.tolist(), limits=asdict(existing_planner.limits))


def _preflight_with_native(preflight, planner, start, poses, bounds, cancelled, progress, native_files):
    """Recheck complete native provenance on the worker before any replay move."""
    for entry in native_files:
        if cancelled():
            raise RuntimeError('Weld replay native provenance verification cancelled')
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Weld replay native acquisition bytes differ from their retained SHA256')
    return preflight(planner, start, poses, bounds, cancelled, progress)


class WeldSeamService:
    """Tick-driven coordinator using the live service's sole camera and arm."""

    TERMINAL = frozenset(('idle', 'complete', 'incomplete', 'insufficient_length', 'cancelled', 'interrupted', 'error'))
    QUIET_S = .4
    STATION_TIMEOUT_S = 30.
    MAX_STATIONS = 260

    def __init__(self, service, *, clock=time.monotonic, executor=None, replay_preflight=None):
        self.service, self.clock = service, clock
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-weld')
        self.future = self._future_kind = None
        self._generation = 0
        self._cancel_event = threading.Event()
        self._replay_preflight = replay_preflight or preflight_replay
        self.settings = WeldSettings()
        self.status, self.message, self.error = 'idle', 'Ready to find a seam from Weld Finder', None
        self.session_id = None
        self.revision = 0
        self.records, self.rejections = [], []
        # Display-only native profiles stay separate from the hashed replay path.
        self._profiles = {}
        self._native_files = []
        self.endpoints = {'positive': None, 'negative': None}
        self._identity = self._handeye = self._baseline = self._center = None
        self._goal = self._quiet_pose = self._quiet_since = self._capture_cutoff = None
        self._hold_at = None
        self._after_settle = None
        self._phase_started = self.clock()
        self._direction = 0
        self._distance = {1: 0., -1: 0.}
        self._missing = {1: 0, -1: 0}
        self._absent_records = {1: [], -1: []}
        self._last_found = {1: None, -1: None}
        self._scan_direction = self._reference_direction = None
        self._last_key = None
        self._pending_measurement = None
        self._align_moves = self._stations = 0
        self._replay = []
        self._replay_index = 0
        self._replay_seed = None
        self._result = None
        self._export_path = None
        self._closed = False
        self._configuration_pending = False
        self._configuration_id = None
        self._configuration_submitted = False
        self._ae_wait_started = None
        self._laser_preparation = {}
        self._generation_transitions = []
        self._station_generation_changes = 0
        self._initial_generation = None
        self._manual_exposure_us = None
        self._manual_requested_us = None
        self._manual_configuration_readback = None
        self._manual_requested_gain_db = None
        self._bracket_kind = None
        self._ae_lock_required = True
        self._bracket_candidates = None
        self._bracket_attempts = 0
        self._bracket_history = []
        self._search_stations = []
        self._search_frame = None
        self._search_extents = {1: 0., -1: 0.}
        self._acquisition_gaps = deque(maxlen=128)
        self._acquisition_gap_count = 0
        self._endpoint_retries = []
        self._station_retry = None
        self._station_retry_attempt = 0
        self._endpoint_miss_pose = self._endpoint_miss_since = None
        self._endpoint_miss_samples = 0
        self._station_gate = {}
        self._station_gate_history = deque(maxlen=128)
        self._diagnostic_futures = []
        self._partial_future = None
        self._partial_dirty = False
        self._partial_signature = None
        self._partial_finding_status = self._partial_finding_error = None
        self._save_error = None
        self._restore()

    @property
    def busy(self):
        return (self.status not in self.TERMINAL or self.future is not None or self._configuration_pending
                or self._partial_dirty or self._partial_future is not None and not self._partial_future.done())

    def _can_start(self):
        gate = getattr(self.service, 'can_start_weld', None) or self.service.can_start_scan
        gate()
        if 'Weld Finder' not in self.service.saved_poses:
            raise RuntimeError('Save a Weld Finder pose before finding a seam')

    def _readiness(self):
        try:
            if self.busy:
                raise RuntimeError('Weld finding owns robot motion')
            self._can_start()
            return True, None
        except (ValueError, RuntimeError, KeyError) as error:
            return False, str(error)

    def _replay_identity(self, camera):
        if not self._result or self._result.get('status') not in ('complete', 'incomplete'):
            raise ValueError('Replay needs a retained continuous measured weld section')
        if self._result['status'] == 'incomplete' and (self._result.get('coverage_complete') is not False
                or self._result.get('replay_scope') != 'measured_section_only'):
            raise ValueError('Incomplete weld replay must explicitly cover only the measured section')
        binding = self._result['binding']
        if binding['model_fingerprint'] != self.service.model_fingerprint:
            raise ValueError('Weld path belongs to another installed robot model')
        handeye = camera.get('handeye') or {}
        if (handeye.get('source_id') != binding['source_id']
                or handeye.get('optical_intrinsics_id', handeye.get('calibration_id')) != binding['optical_intrinsics_id']
                or not np.array_equal(np.asarray(handeye.get('tcp_from_camera')), np.asarray(binding['tcp_from_camera']))):
            raise ValueError('Weld replay requires the original measured hand-eye and camera optics')
        active = camera.get('calibration_id')
        if active is not None and active != binding['calibration_id']:
            raise ValueError('Laser calibration changed since the weld path was captured')
        if _digest(self._result['path']) != self._result['path_sha256']:
            raise ValueError('Weld path integrity check failed')
        if (_digest(self._result.get('native_files', [])) != self._result.get('native_files_sha256')):
            raise ValueError('Weld native acquisition manifest integrity check failed')
        path = self._result['path']
        points = np.asarray([record['seam_point_base_m'] for record in path], float)
        if (len(path) < 2 or points.shape != (len(path), 3) or not np.isfinite(points).all()
                or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() <= 1e-11
                or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > .005)):
            raise ValueError('Weld replay needs at least two distinct points without an unobserved interior gap')
        native = {entry['path']: entry for entry in self._result['native_files']}
        for direction in (-1, 1):
            stations = sorted(record['distance_mm'] for record in path if record['direction'] == direction)
            if stations and not np.allclose(np.diff([0., *stations]), 1., atol=1e-8, rtol=0):
                raise ValueError('Measured weld section contains an unobserved interior station')
        for record in path:
            entry = native.get(record.get('native_path')) or {}
            if (entry.get('disposition') != 'path_point' or entry.get('sha256') != record.get('native_sha256')):
                raise ValueError('Weld replay path lacks complete retained native acquisition provenance')
        return binding

    def state(self, camera_state=None):
        if self._result is None and self.records and self.status in self.TERMINAL and not self.busy:
            # A failed/cancelled finding still owns inspectable measured data.
            # This snapshot only builds metadata; it never commands or resumes.
            self._result = self._build_result(partial=True)
        ready, blocker = self._readiness()
        camera = camera_state or {}
        replay_ready, replay_blocker = False, blocker
        if ready:
            try:
                self._replay_identity(camera)
                replay_ready, replay_blocker = True, None
            except (ValueError, KeyError, TypeError) as error:
                replay_blocker = str(error)
        points, lines, poses, profiles = [], [], [], []
        for record in self._ordered_records():
            points.append(record['seam_point_base_m'])
            lines.extend(record['scan_lines_base_m'])
            poses.append(record['torch_pose_base_m'])
            profiles.extend(self._profiles.get(record['native_path'], []))
        return copy.deepcopy(dict(status=self.status, phase=self.status, busy=self.busy,
            message=self.message, error=self.error, settings=asdict(self.settings),
            start_ready=ready, ready=ready, blocker=blocker, replay_ready=replay_ready,
            replay_blocker=replay_blocker, session_id=self.session_id, revision=self.revision,
            captures=len(self.records), point_count=len(self.records),
            distance_mm=self._measured_length_mm(), stations=self._stations, rejected=len(self.rejections),
            search_distance_mm=self._search_extents[1]+self._search_extents[-1],
            search_positive_mm=self._search_extents[1], search_negative_mm=self._search_extents[-1],
            minimum_basis='measured_scan_plane_extent',
            distance_positive_mm=self._distance[1], distance_negative_mm=self._distance[-1],
            endpoints=self.endpoints, export_paths=[self._export_path] if self._export_path else [],
            result=None if self._result is None else {key: self._result.get(key) for key in
                ('status', 'seam_length_mm', 'search_distance_mm', 'minimum_satisfied', 'minimum_basis',
                 'path_sha256', 'replay_reference', 'finding_status', 'finding_error')},
            replay_index=self._replay_index, replay_total=len(self._replay),
            laser_preparation=self._laser_preparation,
            coverage_complete=bool(self._result and self._result.get('coverage_complete')),
            replay_scope=None if self._result is None else self._result.get('replay_scope'),
            acquisition_gap_count=self._acquisition_gap_count,
            recent_acquisition_gaps=list(self._acquisition_gaps)[-8:],
            station_gate=self._station_gate, station_retry=self._station_retry,
            endpoint_retry_count=len(self._endpoint_retries),
            save_status='error' if self._save_error else 'saving' if self._partial_dirty
                or self._partial_future is not None and not self._partial_future.done() else 'saved' if self._export_path else 'idle',
            save_error=self._save_error,
            geometry=dict(units='m', coordinate_frame='robot_base', revision=self.revision,
                seam_points_base_m=points, scan_lines_base_m=lines, torch_poses_base_m=poses,
                profile_points_base_m=profiles),
            replay_reference='camera optical frame at 150 mm torch-reference clearance; no welding actuation'))

    def dispatch(self, request):
        action = request.get('action')
        if action == 'weld_start':
            return self.start(request)
        if action == 'weld_cancel':
            return self.cancel()
        if action == 'weld_replay':
            return self.replay()
        if action == 'weld_state':
            return self.state(camera_state=self.service.camera.state(include_frame=False))
        if action in ('weld_clear', 'weld_export'):
            self._poll_file_saves()
            if self.busy:
                raise RuntimeError('Stop weld finding before clearing or exporting its path')
            if action == 'weld_export':
                if self._result is None:
                    if not self.records:
                        raise ValueError('No weld path is available')
                    self._result = self._build_result(partial=True)
                self._submit('export', self._save_result, copy.deepcopy(self._result))
            else:
                self.records, self.rejections, self._result = [], [], None
                self._profiles.clear()
                self.session_id, self._export_path = None, None
                self.status, self.message = 'idle', 'Captured weld path cleared; archived acquisitions retained'
                self.revision += 1
                # A clear removes the latest pointer, never the raw acquisition archive.
                self._submit('clear', self._clear_latest)
            return self.state()
        raise ValueError('Unsupported weld seam action')

    def start(self, request):
        self._poll_file_saves()  # Consume old-session file results before ownership resets.
        if self.busy:
            raise RuntimeError('Weld finding is already active')
        settings = WeldSettings(**{key: request[key] for key in asdict(WeldSettings()) if key in request}).validate()
        self._can_start()
        self.settings = settings
        self._generation += 1
        self._cancel_event = threading.Event()
        self.session_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        self.records, self.rejections, self._result = [], [], None
        self._profiles.clear()
        self._native_files = []
        self.endpoints = {'positive': None, 'negative': None}
        self.error, self._export_path = None, None
        self._identity = self._handeye = self._baseline = self._center = None
        self._last_found, self._distance, self._missing = {1: None, -1: None}, {1: 0., -1: 0.}, {1: 0, -1: 0}
        self._absent_records = {1: [], -1: []}
        self._direction = self._align_moves = self._stations = 0
        self._scan_direction = self._reference_direction = self._last_key = None
        self._pending_measurement = None
        self._configuration_id = 'weld-'+self.session_id
        self._configuration_pending = self._configuration_submitted = False
        self._ae_wait_started = None
        self._laser_preparation = dict(request_id=self._configuration_id, status='waiting_for_camera')
        self._generation_transitions = []
        self._station_generation_changes = 0
        self._initial_generation = None
        self._manual_exposure_us = self._manual_requested_us = None
        self._manual_configuration_readback = None
        self._manual_requested_gain_db = None
        self._bracket_kind = None
        self._ae_lock_required = True
        self._bracket_candidates = None
        self._bracket_attempts = 0
        self._bracket_history = []
        self._search_stations = []
        self._search_frame = None
        self._search_extents = {1: 0., -1: 0.}
        self._acquisition_gaps.clear()
        self._acquisition_gap_count = 0
        self._endpoint_retries = []
        self._station_retry = None
        self._station_retry_attempt = 0
        self._endpoint_miss_pose = self._endpoint_miss_since = None
        self._endpoint_miss_samples = 0
        self._station_gate = {}
        self._station_gate_history.clear()
        self._partial_dirty = False
        self._partial_signature = None
        self._partial_finding_status = self._partial_finding_error = None
        self._save_error = None
        self.revision += 1
        self.service.camera.dispatch({'action': 'camera_analysis', 'mode': 'laser'})
        self.service.camera.dispatch({'action': 'camera_stream', 'enabled': True})
        self._set_phase('preparing', 'Preparing the sole camera for laser seam finding')
        return self.state()

    def _set_phase(self, phase, message):
        self.status, self.message, self._phase_started = phase, message, self.clock()

    def _submit(self, kind, function, *args, **kwargs):
        if self.future is not None:
            raise RuntimeError('A weld worker operation is already active')
        self._future_kind = kind
        self.future = self.executor.submit(function, *args, **kwargs)

    def _camera_identity(self, camera, *, allow_ae_generation=False, controlled_reason=None, ae_warmup=False):
        identity = tuple(camera.get(key) for key in
            ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id', 'generation'))
        if any(value is None or value == '' for value in identity):
            raise ValueError('Weld finding needs camera, optical, laser and acquisition identities')
        handeye = camera.get('handeye') or {}
        if (handeye.get('source_id') != identity[0]
                or handeye.get('optical_intrinsics_id', handeye.get('calibration_id')) != identity[1]):
            raise ValueError('Measured hand-eye must match the active camera optics')
        correction = transform(handeye['tcp_from_camera'])
        plane = np.asarray((camera.get('intrinsics') or {}).get('laser_plane'), float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
            raise ValueError('Install a calibrated laser plane before weld finding')
        if self._identity is None:
            self._identity, self._handeye = identity, correction
            self._initial_generation = identity[4]
        elif identity[:4] != self._identity[:4] or not np.array_equal(correction, self._handeye):
            raise ValueError('Camera/calibration/acquisition or measured hand-eye changed during weld finding')
        elif identity[4] != self._identity[4]:
            point = ((camera.get('laser') or {}).get('pointcloud_status') or {})
            owned_bracket = controlled_reason == 'owned_exposure_bracket'
            if (type(identity[4]) is not int or type(self._identity[4]) is not int
                    or identity[4] <= self._identity[4] or camera.get('configuring')
                    or not owned_bracket and (not allow_ae_generation
                        or (point.get('auto_exposure') or {}).get('enabled') is not True
                        or self._laser_preparation.get('status') != 'complete')):
                raise ValueError('Unexpected camera generation change outside owned native stripe auto exposure')
            if not owned_bracket and not (ae_warmup and self._ae_lock_required and self._hold_at is not None):
                self._station_generation_changes += 1
                if self._station_generation_changes > 3:
                    raise ValueError('Native stripe exposure changed more than three times at one measured station')
            transition = dict(previous_generation=self._identity[4], generation=identity[4],
                reason=controlled_reason or 'owned_native_stripe_auto_exposure', observed_at=self.clock(),
                controls=copy.deepcopy(camera.get('controls')),
                auto_exposure=copy.deepcopy(point.get('auto_exposure_status')))
            if owned_bracket:
                transition.update(request_id=self._configuration_id, requested_exposure_us=self._manual_requested_us,
                    requested_gain_db=self._manual_requested_gain_db, bracket_kind=self._bracket_kind,
                    bracket_attempt=self._bracket_attempts)
            self._generation_transitions.append(transition)
            self._identity = identity
            self._quiet_pose = self._quiet_since = self._capture_cutoff = None
            self._last_key = None
            return True
        return False

    def _move(self, pose, after='capture'):
        pose = transform(pose)
        if self._baseline is not None and np.linalg.norm(pose[:3, 3]-self._baseline[:3, 3]) > .25:
            raise ValueError('Weld tracking exceeds its 250 mm baseline envelope')
        self.service.move_pose(pose, automatic=False)
        self._begin_settle(pose, after)

    def _begin_settle(self, goal, after):
        self._goal, self._after_settle = transform(goal), after
        self._quiet_pose = self._quiet_since = self._capture_cutoff = self._hold_at = None
        self._ae_wait_started = None
        self._station_generation_changes = 0
        self._bracket_candidates = None
        self._bracket_attempts = 0
        self._station_retry = None
        self._station_retry_attempt = 0
        self._endpoint_miss_pose = self._endpoint_miss_since = None
        self._endpoint_miss_samples = 0
        self._station_gate = {}
        self._station_gate_history.clear()
        self._set_phase('moving', 'Waiting for measured endpoint convergence and quiet feedback')

    def _sample(self):
        # Camera lifecycle/state can block; always sample fresh SDK feedback LAST.
        camera = self.service.camera.state(include_frame=False)
        feedback = self.service.scan_feedback()
        pose = transform(self.service.tcp_pose())
        return camera, feedback, pose

    def _quiet(self, pose, feedback, camera):
        now = self.clock()
        fresh = feedback.get('fresh') and feedback.get('stationary')
        measured = feedback.get('measured_at')
        position, orientation = _pose_errors(pose, self._goal)
        self._station_gate = dict(observed_at=now, feedback=copy.deepcopy(feedback),
            actual_base_from_tcp=pose.tolist(), requested_base_from_tcp=self._goal.tolist(),
            position_error_m=position, orientation_error_rad=orientation,
            hold_monotonic=self._hold_at, station_deadline=self._phase_started+self.STATION_TIMEOUT_S,
            retry_attempt=self._station_retry_attempt)
        if (not fresh or type(measured) not in (float, int) or not np.isfinite(measured)
                or not 0 <= now-measured <= .1 or not self.service.ctrl.ready_for_waypoint()):
            self._station_gate['gate'] = 'fresh_stationary_feedback'
            self._station_gate_history.append(copy.deepcopy(self._station_gate))
            self._quiet_pose = self._quiet_since = None
            self._endpoint_miss_pose = self._endpoint_miss_since = None
            self._endpoint_miss_samples = 0
            return False
        if position > .0005 or orientation > .005:
            self._station_gate['gate'] = 'endpoint_position' if position > .0005 else 'endpoint_orientation'
            self._quiet_pose = self._quiet_since = None
            self._sustained_endpoint_miss(pose, feedback, camera)
            self._station_gate_history.append(copy.deepcopy(self._station_gate))
            return False
        self._endpoint_miss_pose = self._endpoint_miss_since = None
        self._endpoint_miss_samples = 0
        if self._quiet_pose is None:
            self._quiet_pose, self._quiet_since = pose.copy(), now
            self._station_gate['gate'] = 'quiet_dwell'
            return False
        p, r = _pose_errors(pose, self._quiet_pose)
        if p > .0003 or r > .003:
            self._quiet_pose, self._quiet_since = pose.copy(), now
            self._station_gate['gate'] = 'quiet_pose_drift'
            return False
        ready = now-self._quiet_since >= self.QUIET_S
        self._station_gate.update(gate='qualified' if ready else 'quiet_dwell', quiet_seconds=now-self._quiet_since)
        if ready and self._hold_at is not None and self._station_retry is not None and self._station_retry['status'] == 'settling_after_retry':
            self._station_retry.update(status='complete', completed_monotonic=now,
                after_evidence=copy.deepcopy(self._station_gate), replacement_hold_monotonic=self._hold_at)
            self._persist_retry()
        return ready

    def _sustained_endpoint_miss(self, pose, feedback, camera):
        if self._hold_at is None:
            return  # A trajectory endpoint may still be converging before measured Hold.
        if getattr(self.service.ctrl, 'mode', 'Hold') != 'Hold':
            raise RuntimeError('Endpoint retry requires the unchanged measured Hold mode')
        now = self.clock()
        if self._endpoint_miss_pose is None:
            self._endpoint_miss_pose, self._endpoint_miss_since, self._endpoint_miss_samples = pose.copy(), now, 0
        p, r = _pose_errors(pose, self._endpoint_miss_pose)
        if p > .0003 or r > .003:
            self._endpoint_miss_pose, self._endpoint_miss_since, self._endpoint_miss_samples = pose.copy(), now, 0
        self._endpoint_miss_samples += 1
        self._station_gate.update(endpoint_miss_seconds=now-self._endpoint_miss_since,
                                  endpoint_miss_samples=self._endpoint_miss_samples)
        if (self._station_retry_attempt or self._endpoint_miss_samples < 3 or now-self._endpoint_miss_since < .2):
            return
        if self._after_settle == 'replay_next':
            self._replay_identity(camera)
        else:
            self._camera_identity(camera, allow_ae_generation=self._ae_lock_required, ae_warmup=self._ae_lock_required)
        if now >= self._phase_started+self.STATION_TIMEOUT_S:
            return
        self._station_retry_attempt = 1
        self._station_retry = dict(attempt=1, maximum_replans=1, status='replanning',
            reason='Fresh stationary measured Hold persistently missed its unchanged requested endpoint',
            requested_base_from_tcp=self._goal.tolist(), before_evidence=copy.deepcopy(self._station_gate),
            preceding_samples=list(self._station_gate_history), started_monotonic=now,
            station_deadline=self._phase_started+self.STATION_TIMEOUT_S,
            optical_recovery_attempts=self._bracket_attempts, prior_hold_monotonic=self._hold_at,
            diagnostic_path=str(self._directory()/f'endpoint-retry-{len(self._endpoint_retries)+1:04d}.json'))
        self._endpoint_retries.append(self._station_retry)
        self._persist_retry()
        try:
            self.service.move_pose(self._goal.copy(), automatic=False)
        except Exception as error:
            self._station_retry.update(status='failed', error=str(error))
            self._persist_retry()
            raise
        self._station_retry['status'] = 'settling_after_retry'
        self._quiet_pose = self._quiet_since = self._capture_cutoff = self._hold_at = None
        self._endpoint_miss_pose = self._endpoint_miss_since = None
        self._endpoint_miss_samples = 0
        # Do not call _begin_settle: its normal new-station deadline and optical
        # budgets must not be renewed for this one identical-goal retry.
        self.status = 'moving'
        self.message = 'Replanning the identical endpoint once after a fresh quiet measured Hold miss'
        self._persist_retry()

    def _persist_retry(self):
        self._diagnostic_futures.append(self.executor.submit(_atomic_json,
            self._station_retry['diagnostic_path'], copy.deepcopy(self._station_retry)))

    def _settle_tick(self, camera, feedback, pose):
        if self._hold_at is not None and self._ae_lock_required and self._after_settle == 'capture':
            # Warm and freeze the optical controls immediately after the one
            # measured Hold. Post-lock captures still require a NEW continuous
            # quiet dwell; AE generations must not repeatedly erase that dwell.
            self._initial_ae_lock_tick(camera, feedback, pose)
            return
        if not self._quiet(pose, feedback, camera):
            return
        if self._hold_at is None:
            self.service.ctrl.hold(automatic=False)
            self._hold_at = self.clock()
            self._quiet_pose = self._quiet_since = None
            self.status = 'settling'
            return
        if self._capture_cutoff is None:
            self._capture_cutoff = self.clock()
        if self._after_settle == 'return_negative':
            self._direction, self._scan_direction = -1, self._reference_direction.copy()
            self._advance()
        elif self._after_settle == 'replay_next':
            self._replay_index += 1
            if self._replay_index == len(self._replay):
                self._set_phase(self._result['status'], 'Measured weld section replay completed at 150 mm torch-reference clearance'
                    if self._result['status'] == 'incomplete' else
                    'Weld torch-reference path replay completed in measured Hold')
            else:
                self._move(self._replay[self._replay_index], 'replay_next')
        else:
            self.status = 'capturing'
            self._capture(camera, feedback, pose)

    @staticmethod
    def _exposure_interval(cloud):
        timing = cloud.get('timing') or {}
        start, end = timing.get('exposure_host_start_s'), timing.get('exposure_host_end_s')
        uncertainty = timing.get('exposure_time_uncertainty_s')
        if (timing.get('host_clock_domain') != 'host_monotonic'
                or timing.get('motion_pairing_supported') is not True
                or (timing.get('clock_mapping') or {}).get('status') != 'qualified'
                or any(type(v) not in (int, float) for v in (start, end, uncertainty))
                or not np.isfinite([start, end, uncertainty]).all() or not 0 <= uncertainty <= .01
                or end <= start or end-start > .1):
            raise ValueError('Weld capture needs a qualified native exposure interval, not image receipt time')
        return float(start), float(end), float(uncertainty)

    def _capture(self, camera, feedback, pose):
        if not self._quiet(pose, feedback, camera):
            self._capture_cutoff = None
            return
        if self._capture_cutoff is None:
            self._capture_cutoff = self.clock()
        if self._camera_identity(camera, allow_ae_generation=True):
            self.message = 'Native exposure generation changed; restarting quiet qualification at the same Hold'
            return
        ae_status = self._stripe_exposure_ready(camera)
        if ae_status is None:
            return
        cloud = (camera.get('laser') or {}).get('cloud')
        if not isinstance(cloud, dict):
            self.message = 'Waiting at the same Hold for a fresh calibrated laser line'
            return
        if self._manual_exposure_us is not None:
            native_controls = cloud.get('controls') or {}
            if native_controls != self._manual_configuration_readback:
                self.message = 'Waiting for native sensor controls matching the selected manual bracket'
                return
        identity = tuple(cloud.get(key) for key in
            ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id', 'generation'))
        if identity != self._identity:
            raise ValueError('Native laser line identity differs from the current camera')
        key = (cloud.get('acquisition_session_id'), cloud.get('generation'), cloud.get('sequence'))
        if type(key[2]) is not int or key[2] < 0:
            raise ValueError('Native weld exposure needs an integer frame sequence')
        if key == self._last_key:
            return
        if self._last_key is not None and key[2] < self._last_key[2]:
            raise ValueError('Native weld exposure sequence moved backwards')
        acquired = cloud.get('read_started')
        if type(acquired) not in (int, float) or not np.isfinite(acquired) or not 0 <= self.clock()-acquired <= 1.:
            return
        try:
            start, end, uncertainty = self._exposure_interval(cloud)
        except ValueError as error:
            # An individual owned camera frame may lack its hardware event or
            # clock fit. That frame cannot qualify, but it says nothing about
            # physical arm motion or whether a seam ends. Keep this exact Hold
            # and its original cutoff/deadline while awaiting another image.
            self._last_key = key
            self._acquisition_gap_count += 1
            self._acquisition_gaps.append(dict(acquisition_key=list(key), reason=str(error),
                timing=copy.deepcopy(cloud.get('timing')), observed_at=self.clock(),
                direction=self._direction, distance_mm=self._distance.get(self._direction, 0.),
                capture_cutoff=self._capture_cutoff, hold_monotonic=self._hold_at))
            self.message = 'Waiting at the same Hold for the next native frame with qualified exposure timing'
            return
        if start-uncertainty < self._capture_cutoff or feedback['measured_at'] < max(acquired, end+uncertainty):
            self.message = 'Waiting at measured Hold for a wholly post-settle exposure and newer SDK feedback'
            return
        camera_pose = pose @ self._handeye
        previous = self._last_found[self._direction] if self._direction else None
        if previous is None and self._center is not None:
            previous = self._center
        previous_point = None
        if previous is not None:
            expected = np.asarray(previous['seam_point_base_m'], float)
            if self._direction:
                expected += self._direction * self._scan_direction * .001
            previous_point = (camera_pose[:3, :3].T @ (expected-camera_pose[:3, 3])*1000.).tolist()
        measurement = dict(cloud=copy.deepcopy(cloud), base_from_tcp=pose.tolist(),
            captured_at=self.clock(), auto_exposure=copy.deepcopy(ae_status),
            requested_base_from_tcp=self._goal.tolist(),
            base_from_camera=camera_pose.tolist(), feedback=copy.deepcopy(feedback),
            quiet_since=self._quiet_since, quiet_pose=self._quiet_pose.tolist(),
            quiet_duration_s=self.clock()-self._quiet_since,
            goal_error=dict(zip(('position_m', 'orientation_rad'), _pose_errors(pose, self._goal))),
            capture_cutoff=self._capture_cutoff, hold_monotonic=self._hold_at,
            exposure_interval_s=[start, end], exposure_uncertainty_s=uncertainty,
            direction=self._direction, distance_mm=self._distance.get(self._direction, 0.),
            generation=self._generation, acquisition_key=list(key))
        if self._manual_exposure_us is not None:
            measurement['exposure_control'] = dict(mode='owned_manual_bracket',
                requested_us=self._manual_requested_us, actual_us=self._manual_exposure_us,
                requested_gain_db=self._manual_requested_gain_db, bracket_kind=self._bracket_kind,
                readback=copy.deepcopy(camera.get('controls')), limits=self._exposure_bounds(camera),
                configuration_readback=copy.deepcopy(self._manual_configuration_readback),
                request_id=self._configuration_id)
        measurement['generation_transitions'] = copy.deepcopy(self._generation_transitions)
        measurement['station_gate'] = copy.deepcopy(self._station_gate)
        measurement['station_retry'] = copy.deepcopy(self._station_retry)
        self._pending_measurement, self._last_key = measurement, key
        self._submit('analyze', self.service.camera.find_weld_seam, measurement['cloud'],
                     settings={'torch_standoff_mm': 150.}, previous_point_mm=previous_point)
        self.status = 'analyzing'

    def _initial_ae_lock_tick(self, camera, feedback, pose):
        position, orientation = _pose_errors(pose, self._goal)
        if (not feedback.get('fresh') or not feedback.get('stationary')
                or not self.service.ctrl.ready_for_waypoint() or position > .0005 or orientation > .005):
            self._quiet(pose, feedback, camera)
            self.message = 'Waiting at measured Hold for fresh converged feedback before locking stripe exposure'
            return
        self._camera_identity(camera, allow_ae_generation=True, ae_warmup=True)
        feedback_ae = self._stripe_exposure_ready(camera)
        if feedback_ae is None:
            return
        controls = camera.get('controls') or {}
        exposure, gain = controls.get('exposure_us'), controls.get('gain_db')
        limits, gain_bounds = self._exposure_bounds(camera), self._gain_bounds(camera)
        if (type(exposure) not in (int, float) or not np.isfinite(exposure)
                or not limits['min_us'] <= exposure <= limits['max_us']
                or type(gain) not in (int, float) or not np.isfinite(gain)
                or gain_bounds is not None and not gain_bounds[0] <= gain <= gain_bounds[1]):
            raise ValueError('Initial stripe exposure lock requires actual sensor controls within hardware bounds')
        self._manual_requested_us, self._manual_requested_gain_db = float(exposure), float(gain)
        self._bracket_kind = 'ae_lock'
        self._configuration_id = 'weld-'+self.session_id+'-ae-lock'
        self.service.camera.dispatch(dict(action='camera_laser_configure', request_id=self._configuration_id,
            auto_exposure={'enabled': False}, controls={'exposure_us': exposure, 'gain_db': gain}))
        self._configuration_pending = True
        self._bracket_history.append(dict(request_id=self._configuration_id, requested_us=exposure,
            requested_gain_db=gain, kind='ae_lock', attempt=0, limits=limits, gain_bounds_db=gain_bounds,
            gain_ceiling_db=None, direction=self._direction, distance_mm=self._distance.get(self._direction, 0.),
            reason='Freeze reliable native stripe auto-exposure controls before measurement',
            auto_exposure=copy.deepcopy(feedback_ae), started_monotonic=self.clock(),
            measured_hold_monotonic=self._hold_at))
        self.status = 'exposure_recovery'
        self.message = 'Freezing reliable stripe exposure and gain at the same measured Hold'

    def _stripe_exposure_ready(self, camera):
        """Wait at the same Hold for native stripe AE, never brightness-preview AE."""
        point = ((camera.get('laser') or {}).get('pointcloud_status') or {})
        settings = point.get('auto_exposure') or {}
        status = point.get('auto_exposure_status') or {}
        if self._manual_exposure_us is not None:
            actual = (camera.get('controls') or {}).get('exposure_us')
            if settings.get('enabled') is not False or camera.get('controls') != self._manual_configuration_readback:
                raise ValueError('Owned manual exposure readback changed outside the fixed-Hold bracket')
            return dict(state='owned_manual_bracket', reliable=None, exposure_us=actual,
                        request_id=self._configuration_id)
        stamp = status.get('read_started')
        reliable = (settings.get('enabled') is True and status.get('reliable') is True
            and status.get('state') in ('stable', 'limits')
            and status.get('source_id') == self._identity[0]
            and status.get('generation') == self._identity[4]
            and type(stamp) in (int, float) and np.isfinite(stamp) and 0 <= self.clock()-stamp <= 1.)
        self._laser_preparation['auto_exposure'] = copy.deepcopy(status)
        if reliable:
            self._ae_wait_started = None
            return status
        if self._ae_wait_started is None:
            self._ae_wait_started = self.clock()
        if self.clock()-self._ae_wait_started > 5.:
            raise ValueError('Native stripe auto exposure did not reach reliable stable/limit feedback within five seconds')
        self.message = 'Waiting at the same measured Hold for reliable native stripe exposure'
        return None

    @staticmethod
    def _exposure_bounds(camera):
        laser = camera.get('laser') or {}
        limits = laser.get('exposure_limits') or {}
        hardware = camera.get('control_limits') or laser.get('control_limits') or {}
        sensor = limits.get('sensor_us', hardware.get('exposure_us'))
        cap = limits.get('fps_cap_us', hardware.get('fps_exposure_max_us'))
        if (not isinstance(sensor, (tuple, list)) or len(sensor) != 2
                or any(type(value) not in (int, float) for value in (*sensor, cap))
                or not np.isfinite([*sensor, cap]).all() or not 0 < sensor[0] < sensor[1] or cap <= sensor[0]):
            raise ValueError('Native exposure bracket requires actual sensor and maximum-FPS exposure bounds')
        return dict(min_us=float(sensor[0]), max_us=float(min(sensor[1], cap)), fps_cap_us=float(cap))

    @staticmethod
    def _gain_bounds(camera):
        laser = camera.get('laser') or {}
        limits = camera.get('control_limits') or laser.get('control_limits') or {}
        bounds = limits.get('gain_db')
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(value) not in (int, float) for value in bounds)
                or not np.isfinite(bounds).all() or not bounds[0] <= bounds[1]):
            return None
        return [float(value) for value in bounds]

    def _begin_optical_recovery(self, reason):
        camera = self.service.camera.state(include_frame=False)
        self._camera_identity(camera, allow_ae_generation=True)
        limits = self._exposure_bounds(camera)
        if self._bracket_candidates is None:
            high = limits['max_us']
            controls = camera.get('controls') or {}
            current, gain = controls.get('exposure_us'), controls.get('gain_db')
            self._bracket_candidates = [dict(kind='exposure', exposure_us=max(limits['min_us'], high*factor),
                gain_db=gain) for factor in (1., .5, .25)]
            if type(current) in (int, float):
                self._bracket_candidates = [value for value in self._bracket_candidates
                                           if abs(value['exposure_us']-current) > 2.]
            gain_bounds = self._gain_bounds(camera)
            if gain_bounds is not None and type(gain) in (int, float) and np.isfinite(gain):
                ceiling = min(gain_bounds[1], 24.)
                gains = []
                for increment in (6., 12.):
                    selected = min(gain+increment, ceiling)
                    if selected > gain and selected >= gain_bounds[0] and selected not in gains:
                        gains.append(selected)
                self._bracket_candidates.extend(dict(kind='gain', exposure_us=high, gain_db=selected)
                                                for selected in gains)
        if self._bracket_attempts >= 5 or not self._bracket_candidates:
            raise _OpticalVisibilityUnresolved(
                f'Weld optical recovery exhausted its bounded exposure bracket; endpoint is unknown: {reason}')
        selected = self._bracket_candidates.pop(0)
        self._manual_requested_us, self._manual_requested_gain_db = selected['exposure_us'], selected['gain_db']
        self._bracket_kind = selected['kind']
        self._bracket_attempts += 1
        self._configuration_id = f'weld-{self.session_id}-bracket-{self._stations}-{self._bracket_attempts}'
        controls = {'exposure_us': self._manual_requested_us}
        if self._bracket_kind == 'gain':
            controls['gain_db'] = self._manual_requested_gain_db
        self.service.camera.dispatch(dict(action='camera_laser_configure', request_id=self._configuration_id,
            auto_exposure={'enabled': False}, controls=controls))
        self._configuration_pending = True
        self._bracket_history.append(dict(request_id=self._configuration_id, requested_us=self._manual_requested_us,
            direction=self._direction, distance_mm=self._distance.get(self._direction, 0.),
            reason=reason, started_monotonic=self.clock(), limits=limits,
            kind=self._bracket_kind, requested_gain_db=self._manual_requested_gain_db,
            gain_bounds_db=self._gain_bounds(camera), gain_ceiling_db=24.,
            measured_hold_monotonic=self._hold_at, attempt=self._bracket_attempts))
        self.status = 'exposure_recovery'
        self.message = 'Recovering native seam visibility with a bounded exposure bracket at the same Hold'

    def _exposure_recovery_tick(self, camera):
        configuration = camera.get('laser_configuration') or {}
        if configuration.get('request_id') != self._configuration_id:
            return
        status = configuration.get('status')
        if status in ('pending', 'cancelling') or camera.get('configuring'):
            return
        if status in ('complete', 'cancelled', 'error'):
            self._configuration_pending = False
        if status != 'complete':
            raise ValueError(configuration.get('error') or 'Native seam exposure bracket was cancelled')
        readback = configuration.get('readback')
        actual = readback.get('exposure_us') if isinstance(readback, dict) else None
        limits = self._exposure_bounds(camera)
        if (type(actual) not in (int, float) or not np.isfinite(actual)
                or not limits['min_us'] <= actual <= limits['max_us']):
            raise ValueError('Native seam exposure bracket lacks verified sensor readback within its hardware/FPS bounds')
        if camera.get('controls') != readback:
            self.message = 'Waiting at the same Hold for current camera controls matching verified bracket readback'
            return
        gain_bounds = self._gain_bounds(camera)
        gain = readback.get('gain_db')
        if (type(gain) not in (int, float) or not np.isfinite(gain)
                or gain_bounds is not None and not gain_bounds[0] <= gain <= gain_bounds[1]
                or self._bracket_kind == 'gain' and gain > 24.):
            raise ValueError('Native seam gain bracket readback exceeds actual sensor or 24 dB recovery bounds')
        self._manual_exposure_us = float(actual)
        self._manual_configuration_readback = copy.deepcopy(readback)
        self._camera_identity(camera, controlled_reason='owned_exposure_bracket')
        self._quiet_pose = self._quiet_since = self._capture_cutoff = None
        self._last_key = None
        self._ae_wait_started = None
        self._bracket_history[-1].update(actual_us=actual, readback=copy.deepcopy(camera.get('controls')),
            configuration_readback=copy.deepcopy(readback),
            completed_monotonic=self.clock(), generation=camera.get('generation'))
        if self._bracket_kind == 'ae_lock':
            self._ae_lock_required = False
        self.status = 'capturing'

    def _prepare_camera(self, camera):
        if not self._configuration_submitted:
            if (not camera.get('enabled') or any(camera.get(key) for key in
                    ('starting', 'stopping', 'configuring', 'analysis_pending'))):
                return False
            # This adapter schedules the sole camera lifecycle owner. It is not
            # a worker calling the backend concurrently with stream/config work.
            self.service.camera.dispatch(dict(action='camera_laser_configure',
                request_id=self._configuration_id, auto_exposure={'enabled': True}))
            self._configuration_submitted = self._configuration_pending = True
            self._laser_preparation['status'] = 'pending'
            return False
        configuration = camera.get('laser_configuration') or {}
        if configuration.get('request_id') != self._configuration_id:
            return False
        self._laser_preparation.update(copy.deepcopy(configuration))
        state = configuration.get('status')
        if state in ('pending', 'cancelling') or camera.get('configuring'):
            return False
        if state in ('complete', 'cancelled', 'error'):
            self._configuration_pending = False
        if state != 'complete':
            raise ValueError(configuration.get('error') or 'Native laser exposure configuration was cancelled')
        self._laser_preparation['configured_monotonic'] = self.clock()
        return True

    def _drain_configuration(self):
        if not self._configuration_pending:
            return
        camera = self.service.camera.state(include_frame=False)
        configuration = camera.get('laser_configuration') or {}
        if (configuration.get('request_id') == self._configuration_id
                and configuration.get('status') in ('complete', 'cancelled', 'error')
                and not camera.get('configuring')):
            self._configuration_pending = False
            self._laser_preparation.update(copy.deepcopy(configuration))

    def _post_capture(self, measurement):
        camera, feedback, pose = self._sample()
        self._camera_identity(camera, allow_ae_generation=True)
        if not camera.get('enabled') or camera.get('error'):
            raise ValueError('Camera disconnected while the seam exposure was analyzed')
        stamp = feedback.get('measured_at')
        if (not feedback.get('fresh') or type(stamp) not in (int, float) or not np.isfinite(stamp)
                or not 0 <= self.clock()-stamp <= .1 or stamp < measurement['exposure_interval_s'][1]):
            self.message = 'Waiting at the same Hold for fresh post-exposure SDK feedback'
            return False
        if (not feedback.get('stationary') or not self.service.ctrl.ready_for_waypoint()
                or _pose_errors(pose, transform(measurement['base_from_tcp']))[0] > .0003
                or _pose_errors(pose, transform(measurement['base_from_tcp']))[1] > .003):
            raise ValueError('Arm lost continuous stationary pose during native seam analysis')
        measurement['post_capture'] = dict(feedback=copy.deepcopy(feedback), base_from_tcp=pose.tolist(),
            observed_at=self.clock(),
            camera_generation=camera.get('generation'),
            generation_matches=measurement['cloud'].get('generation') == camera.get('generation'),
            goal_error=dict(zip(('position_m', 'orientation_rad'), _pose_errors(pose, self._goal))))
        if not measurement['post_capture']['generation_matches']:
            return 'generation_changed'
        return True

    def _analyzed(self, result):
        measurement = self._pending_measurement
        checked = self._post_capture(measurement)
        if not checked:
            self.status = 'validating'
            return
        self._stations += 1
        if self._stations > self.MAX_STATIONS:
            raise ValueError('Weld finding exhausted its bounded station budget')
        # Save every qualified native observation, including absent, ambiguous,
        # rejected and centering views, before deciding whether to move again.
        if checked == 'generation_changed':
            result = dict(status='reacquire', found=False, reason='Native exposure generation changed during analysis',
                          original_result=copy.deepcopy(result))
        self._pending_result = copy.deepcopy(result)
        self._pending_native_path = self._directory()/f'observation-{self._stations:04d}.json'
        payload = dict(schema='autoweld.weld-native.v1', measurement=measurement, result=result)
        self._submit('save_observation', _write_observation, self._pending_native_path, payload,
                     copy.deepcopy(self._native_files))
        self.status = 'saving'

    def _process_analyzed(self):
        measurement, result = self._pending_measurement, self._pending_result
        if result.get('status') == 'reacquire':
            self._native_files[-1]['disposition'] = 'reacquire'
            self.status = 'capturing'
            self.message = 'Reacquiring a native line at the same Hold after exposure generation changed'
            return
        checked = self._post_capture(measurement)
        if not checked:
            self.status = 'validating_saved'
            return
        if checked == 'generation_changed':
            self._native_files[-1]['disposition'] = 'reacquire'
            self.status = 'capturing'
            self.message = 'Reacquiring at the same Hold after exposure changed while saving; prior result is archived only'
            return
        status = result.get('status')
        if status not in ('found', 'no_seam'):
            self._native_files[-1]['disposition'] = 'rejected'
            # Missing stripe, glare, branch ambiguity and detector faults cannot
            # become physical endpoints or motion directions.
            reason = result.get('reason') or (result.get('quality') or {}).get('reason') or status
            self.rejections.append(dict(direction=self._direction, distance_mm=measurement['distance_mm'],
                                        status=status, reason=reason))
            if status in ('invalid', 'ambiguous'):
                self._begin_optical_recovery(reason)
                return
            raise ValueError(f'Weld observation rejected; seam endpoint is unknown: {reason}')
        if status == 'found':
            camera_pose = transform(measurement['base_from_camera'])
            seam = np.asarray(result['seam_point_mm'], float)/1000.
            direction = np.asarray(result['scan_direction_camera'], float)
            stripe = np.asarray(result['stripe_direction_camera'], float)
            flank = np.asarray(result['flank_lines_mm'], float)/1000.
            torch = transform(result['camera_from_torch'])
            torch[:3, 3] /= 1000.
            if (seam.shape != (3,) or direction.shape != (3,) or stripe.shape != (3,)
                    or flank.shape != (2, 2, 3) or not np.isfinite(np.r_[seam, direction, stripe, flank.ravel()]).all()
                    or abs(np.linalg.norm(direction)-1) > 1e-4 or abs(np.linalg.norm(stripe)-1) > 1e-4):
                raise ValueError('Weldfinder returned invalid camera seam geometry')
            world_direction = camera_pose[:3, :3] @ direction
            if self._reference_direction is not None:
                if world_direction @ self._reference_direction < 0:
                    world_direction *= -1
                if world_direction @ self._reference_direction < .7:
                    raise ValueError('Weld direction changed beyond its bounded tracking view')
            # Keep the seam near the middle of the finite native stripe, not at
            # an invented optical-axis/laser-plane intersection.
            native = np.asarray(measurement['cloud']['xyz_mm'], float)/1000.
            midpoint = .5*(np.min(native @ stripe)+np.max(native @ stripe))
            lateral = float(seam @ stripe-midpoint)
            depth_error = float(seam[2]-.150)
            if abs(lateral) > .0015 or abs(depth_error) > .001:
                self._native_files[-1]['disposition'] = 'centering'
                self._record_search_station(measurement)
                self._align_moves += 1
                if self._align_moves > 40:
                    raise ValueError('Weld centering/standoff did not converge within bounded corrections')
                correction = stripe*np.clip(lateral, -.002, .002)
                correction[2] = np.clip(correction[2]+depth_error, -.005, .005)
                goal = transform(measurement['base_from_tcp'])
                goal[:3, 3] += camera_pose[:3, :3] @ correction
                self._move(goal)
                self.message = 'Centering the measured seam and maintaining 150 mm optical depth'
                return
            self._align_moves = 0
            point = camera_pose[:3, :3] @ seam+camera_pose[:3, 3]
            world_lines = flank @ camera_pose[:3, :3].T+camera_pose[:3, 3]
            world_torch = camera_pose @ torch
            record = dict(direction=self._direction, distance_mm=measurement['distance_mm'],
                seam_point_base_m=point.tolist(), scan_lines_base_m=world_lines.tolist(),
                torch_pose_base_m=world_torch.tolist(), measured_base_from_tcp=measurement['base_from_tcp'],
                acquisition_key=measurement['acquisition_key'], quality=result.get('quality'),
                detector=result, native_path=str(self._pending_native_path),
                native_sha256=self._native_files[-1]['sha256'])
            self._scan_direction = world_direction
            if self._reference_direction is None:
                self._reference_direction = world_direction.copy()
            if self._direction == 0:
                self._center = record
                self._center['measured_base_from_tcp'] = measurement['base_from_tcp']
                self._search_frame = dict(units='m', coordinate_frame='robot_base',
                    center_camera_origin_base_m=camera_pose[:3, 3].tolist(),
                    normal_base=self._reference_direction.tolist(), center_native_path=str(self._pending_native_path),
                    basis='measured_camera_origin_projection_on_fixed_initial_scan_plane_normal')
            else:
                prior = self._last_found[self._direction] or self._center
                separation = np.linalg.norm(point-np.asarray(prior['seam_point_base_m']))
                if separation > .005:
                    raise ValueError('Weld branch association jumped more than 5 mm between 1 mm stations')
                if self._missing[self._direction]:
                    self._native_files[-1]['disposition'] = 'rejected'
                    raise ValueError('Seam reappeared beyond an unobserved interior gap; a continuous replay path is unavailable')
                self._missing[self._direction] = 0
            self._native_files[-1]['disposition'] = 'path_point'
            self._record_search_station(measurement)
            self.records.append(record)
            self._retain_profile(record, measurement)
            if self._direction:
                self._last_found[self._direction] = record
            self.revision += 1
            self._advance()
        else:
            if self._direction == 0:
                raise ValueError('The Weld Finder start pose has a good stripe but no seam valley/discontinuity')
            self._missing[self._direction] += 1
            self._native_files[-1]['disposition'] = 'endpoint_observation'
            self._record_search_station(measurement)
            self._absent_records[self._direction].append(dict(native_path=str(self._pending_native_path),
                acquisition_key=measurement['acquisition_key'], distance_mm=measurement['distance_mm'],
                observation_index=self._native_files[-1]['observation_index']))
            self._advance()

    def _advance(self):
        direction = self._direction
        if direction == 0:
            self._direction = direction = 1
        if (self._missing[direction] >= self.settings.end_confirmations
                and self._search_extents[direction]+1e-8 >= self.settings.min_distance_mm/2):
            name = 'positive' if direction == 1 else 'negative'
            last = self._last_found[direction] or self._center
            self.endpoints[name] = dict(confirmed=True, no_seam_stations=self._missing[direction],
                distance_mm=self._distance[direction], last_seam_point_base_m=last['seam_point_base_m'],
                measured_search_extent_mm=self._search_extents[direction],
                observations=copy.deepcopy(self._absent_records[direction][-self.settings.end_confirmations:]))
            self.revision += 1
            if direction == 1:
                self._move(self._center['measured_base_from_tcp'], 'return_negative')
            else:
                self._finish()
            return
        if self._distance[direction]+self.settings.step_mm > self.settings.max_distance_each_way_mm:
            raise ValueError('Weld seam remains present at bounded travel limit; endpoint is unconfirmed')
        self._distance[direction] += self.settings.step_mm
        goal = transform(self.service.tcp_pose())
        camera_pose = goal @ self._handeye
        camera_pose[:3, 3] += direction*self._scan_direction*.001
        self._move(camera_pose @ np.linalg.inv(self._handeye))
        self.message = f'Following weld {"positive" if direction == 1 else "negative"}: {self._distance[direction]:g} mm'

    def _ordered_records(self):
        return sorted(self.records, key=lambda record: record['direction']*record['distance_mm'])

    def _measured_length_mm(self):
        points = np.asarray([record['seam_point_base_m'] for record in self._ordered_records()])
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()*1000.) if len(points) > 1 else 0.

    def _record_search_station(self, measurement):
        """Measure explored scan-plane displacement independently of seam length.

        A good native line without a valley still observes the searched plane.
        Invalid/reacquired exposures never enter this evidence. Nominal 1 mm
        requests remain diagnostics; measured camera transforms establish span.
        """
        self._search_stations.append(dict(native_path=str(self._pending_native_path),
            acquisition_key=copy.deepcopy(measurement['acquisition_key']),
            direction=measurement['direction'], nominal_distance_mm=measurement['distance_mm'],
            camera_origin_base_m=transform(measurement['base_from_camera'])[:3, 3].tolist()))
        if self._search_frame is None:
            return
        origin = np.asarray(self._search_frame['center_camera_origin_base_m'], float)
        normal = np.asarray(self._search_frame['normal_base'], float)
        self._search_extents = {1: 0., -1: 0.}
        for station in self._search_stations:
            progress = float((np.asarray(station['camera_origin_base_m'])-origin) @ normal*1000.)
            station['signed_progress_mm'] = progress
            direction = station['direction']
            if direction:
                self._search_extents[direction] = max(self._search_extents[direction], direction*progress)

    def _build_result(self, *, partial=False):
        path = self._ordered_records()
        if not path:
            raise ValueError('No measured weld points are available')
        if not partial and len(path) < 2:
            raise ValueError('A replayable weld path needs at least two distinct measured seam points')
        length = self._measured_length_mm()
        if not partial and length <= 1e-8:
            raise ValueError('A replayable weld path cannot contain only coincident seam points')
        search_distance = self._search_extents[1]+self._search_extents[-1]
        minimum = all(self._search_extents[direction]+1e-8 >= self.settings.min_distance_mm/2 for direction in (1, -1))
        coverage_complete = all(bool(endpoint and endpoint.get('confirmed')) for endpoint in self.endpoints.values())
        status = 'incomplete' if partial or not coverage_complete else 'complete' if minimum else 'insufficient_length'
        binding = dict(model_fingerprint=self.service.model_fingerprint,
            source_id=self._identity[0], optical_intrinsics_id=self._identity[1], calibration_id=self._identity[2],
            tcp_from_camera=self._handeye.tolist(), acquisition_session_id=self._identity[3],
            initial_generation=self._initial_generation, final_generation=self._identity[4],
            generation=self._identity[4], generation_policy='owned_stripe_AE_per_station_requalification')
        return dict(schema='autoweld.weld-path.v1', session_id=self.session_id, status=status,
            finding_status=self._partial_finding_status or self.status if partial else status,
            finding_error=self._partial_finding_error or self.error,
            settings=asdict(self.settings), binding=binding, endpoints=self.endpoints,
            seam_length_mm=length, minimum_satisfied=minimum, path=copy.deepcopy(path),
            minimum_basis='measured_scan_plane_extent', search_distance_mm=search_distance,
            coverage_complete=not partial and coverage_complete and minimum,
            replay_scope='full_observed_path' if not partial and coverage_complete and minimum else 'measured_section_only',
            search_positive_mm=self._search_extents[1], search_negative_mm=self._search_extents[-1],
            search_frame=copy.deepcopy(self._search_frame), search_stations=copy.deepcopy(self._search_stations),
            path_sha256=_digest(path), rejections=self.rejections,
            native_files=copy.deepcopy(self._native_files), native_files_sha256=_digest(self._native_files),
            generation_transitions=copy.deepcopy(self._generation_transitions),
            exposure_brackets=copy.deepcopy(self._bracket_history),
            acquisition_gaps=list(self._acquisition_gaps), acquisition_gap_count=self._acquisition_gap_count,
            acquisition_gaps_truncated=max(0, self._acquisition_gap_count-len(self._acquisition_gaps)),
            endpoint_retries=copy.deepcopy(self._endpoint_retries), last_station_gate=copy.deepcopy(self._station_gate),
            replay_reference='camera_optical_frame_at_150mm_torch_reference_clearance',
            units='m', coordinate_frame='robot_base')

    def _finish(self):
        self._result = self._build_result()
        self._submit('finish', self._save_result, copy.deepcopy(self._result))
        self._set_phase('finalizing', 'Saving the ordered, bidirectional weld path')

    def _directory(self):
        if not self.session_id or Path(self.session_id).name != self.session_id:
            raise ValueError('Invalid weld session identity')
        return Path(self.service.runtime)/'weld-paths'/self.session_id

    def _native_path(self, index):
        return self._directory()/f'native-{index:04d}.json'

    def _manifest_path(self):
        return self._directory()/'path.json'

    def _save_result(self, result):
        _atomic_json(self._directory()/'observations.json', dict(schema='autoweld.weld-observations.v1',
            native_files=result['native_files'], native_files_sha256=result['native_files_sha256']))
        path = _atomic_json(self._manifest_path(), result)
        _atomic_json(Path(self.service.runtime)/'weld-paths'/'latest.json', {'path': path})
        return path

    def _poll_file_saves(self):
        pending = []
        for future in self._diagnostic_futures:
            if not future.done():
                pending.append(future)
                continue
            try:
                future.result()
            except Exception as error:
                self._save_error = f'Endpoint diagnostic persistence failed: {error}'
        self._diagnostic_futures = pending
        if self._partial_future is not None and self._partial_future.done():
            future, self._partial_future = self._partial_future, None
            try:
                self._export_path = future.result()
                self._save_error = None
            except Exception as error:
                self._save_error = f'Partial weld path persistence failed: {error}'

    def _schedule_partial_save(self):
        if (not self._partial_dirty or self.future is not None or self._configuration_pending
                or self._partial_future is not None):
            return
        self._partial_dirty = False
        if not self.records:
            return
        result = self._build_result(partial=True)
        signature = _digest(result)
        if signature == self._partial_signature:
            return
        self._result, self._partial_signature = result, signature
        try:
            self._partial_future = self.executor.submit(self._save_result, copy.deepcopy(result))
        except Exception as error:
            self._save_error = f'Partial weld path save could not be submitted: {error}'

    def _clear_latest(self):
        latest = Path(self.service.runtime)/'weld-paths'/'latest.json'
        latest.unlink(missing_ok=True)

    def _retain_profile(self, record, measurement):
        """Register every native return; never join across an unobserved gap."""
        cloud = measurement['cloud']
        if cloud.get('units') != 'mm' or cloud.get('coordinate_frame') != 'camera':
            raise ValueError('Full weld profile requires camera-frame millimetres')
        points = np.asarray(cloud['xyz_mm'], float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('Invalid native weld profile')
        pose = transform(measurement['base_from_camera'])
        world = points/1000. @ pose[:3, :3].T + pose[:3, 3]
        # Micrometre display quantization keeps JSON small; native data is untouched.
        self._profiles[record['native_path']] = world.round(6).tolist()

    def _restore(self):
        latest = Path(self.service.runtime)/'weld-paths'/'latest.json'
        if not latest.exists():
            return
        try:
            result = json.loads(Path(json.loads(latest.read_text())['path']).read_text())
            if result.get('schema') != 'autoweld.weld-path.v1' or _digest(result['path']) != result['path_sha256']:
                raise ValueError('Saved weld path integrity check failed')
            self._result = result
            self._native_files = result.get('native_files', [])
            self._search_frame = result.get('search_frame')
            self._search_stations = result.get('search_stations', [])
            self._search_extents = {1: result.get('search_positive_mm', 0.), -1: result.get('search_negative_mm', 0.)}
            binding = result['binding']
            self._identity = tuple(binding[key] for key in
                ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id'))+(binding['generation'],)
            self._handeye = transform(binding['tcp_from_camera'])
            self._initial_generation = binding.get('initial_generation', binding['generation'])
            self._generation_transitions = result.get('generation_transitions', [])
            self._bracket_history = result.get('exposure_brackets', [])
            self._endpoint_retries = result.get('endpoint_retries', [])
            self._station_gate = result.get('last_station_gate', {})
            self.rejections = result.get('rejections', [])
            self._acquisition_gaps.extend(result.get('acquisition_gaps', []))
            self._acquisition_gap_count = result.get('acquisition_gap_count', 0)
            if result.get('finding_status') in ('error', 'cancelled', 'interrupted'):
                self._partial_finding_status = result['finding_status']
                self._partial_finding_error = result.get('finding_error')
                self._partial_signature = _digest(result)
            self.session_id, self.records = result['session_id'], result['path']
            for record in self.records:
                raw = Path(record['native_path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != record['native_sha256']:
                    raise ValueError('Saved native weld profile integrity check failed')
                self._retain_profile(record, json.loads(raw)['measurement'])
            self.settings = WeldSettings(**result['settings']).validate()
            self.endpoints = result['endpoints']
            self._export_path = str(self._manifest_path())
            self.status, self.message = result['status'], 'Restored weld geometry; no motion was resumed'
            self.error = result.get('finding_error')
            self.revision += 1
        except Exception as error:
            self.status, self.error = 'error', f'Saved weld path could not be restored: {error}'

    def replay(self):
        if self.busy:
            raise RuntimeError('Stop weld finding before replay')
        self._can_start()
        if self._result is None and self.records:
            self._result = self._build_result(partial=True)
        camera = self.service.camera.state(include_frame=False)
        binding = self._replay_identity(camera)
        correction = transform(binding['tcp_from_camera'])
        self._replay = [transform(record['torch_pose_base_m']) @ np.linalg.inv(correction)
                        for record in self._result['path']]
        if len(self._replay) < 2:
            raise ValueError('Weld replay needs at least two measured seam poses')
        self._replay_index = 0
        self._replay_seed = np.array(self.service.ctrl.positions(), float, copy=True)
        self._set_phase('replay_preflight', 'Validating every torch-reference replay segment before movement')
        source = (self.service.ctrl.planner_for_controller() if hasattr(self.service.ctrl, 'planner_for_controller')
                  else self.service.planner)
        source = copy.copy(source)
        source.limits = copy.copy(source.limits)
        self._replay_limits = asdict(source.limits)
        self._replay_bounds = np.array(self.service.ctrl.effective_joint_bounds(), float, copy=True)
        self._cancel_event = threading.Event()
        self._submit('replay_preflight', _preflight_with_native, self._replay_preflight, source, self._replay_seed.copy(),
            [pose.copy() for pose in self._replay], self._replay_bounds.copy(), self._cancel_event.is_set,
            lambda index: setattr(self, '_replay_index', index), copy.deepcopy(self._result['native_files']))
        return self.state(camera_state=camera)

    def _replay_preflight_done(self):
        camera, feedback, _ = self._sample()
        self._replay_identity(camera)
        current_planner = (self.service.ctrl.planner_for_controller()
            if hasattr(self.service.ctrl, 'planner_for_controller') else self.service.planner)
        if (asdict(current_planner.limits) != self._replay_limits
                or not np.array_equal(self.service.ctrl.effective_joint_bounds(), self._replay_bounds)):
            raise ValueError('Motion limits or installed bounds changed during weld replay proof')
        if (not feedback.get('fresh') or not feedback.get('stationary')
                or not self.service.ctrl.ready_for_waypoint()):
            self.status = 'replay_wait_feedback'
            return
        if np.max(abs(self.service.ctrl.positions()[:6]-self._replay_seed[:6])) > .01:
            raise ValueError('Arm moved during weld replay preflight; start a new proof from measured Hold')
        self._replay_index = 0
        self._move(self._replay[0], 'replay_next')

    def _future_tick(self):
        if self.future is None or not self.future.done():
            return False
        future, kind = self.future, self._future_kind
        self.future = self._future_kind = None
        if self.status in self.TERMINAL and kind not in ('export', 'clear'):
            if kind == 'save_observation' and not future.cancelled():
                try:
                    entry = future.result()
                except Exception:
                    pass
                else:
                    # An already-running file write may complete after cancel.
                    # Retain its bytes/index without accepting it or moving.
                    entry['disposition'] = 'reacquire'
                    self._native_files.append(entry)
            return True  # A cancelled/failed generation cannot submit later motion.
        value = future.result()
        if kind == 'analyze':
            self._pending_result = value
            self._analyzed(value)
        elif kind == 'save_observation':
            self._native_files.append(value)
            self._process_analyzed()
        elif kind == 'replay_preflight':
            self._replay_preflight_done()
        elif kind == 'finish':
            self._export_path = value
            self.status = self._result['status']
            self.message = ('Weld path complete; both ends confirmed' if self.status == 'complete' else
                            'Measured weld section retained; at least one endpoint could not be optically confirmed'
                            if self.status == 'incomplete' else
                            'Both ends confirmed, but the measured search extent is shorter than the requested minimum')
            self.revision += 1
        elif kind == 'export':
            self._export_path = value
            self._save_error = None
        return True

    def tick(self):
        if self._closed:
            return
        try:
            self._poll_file_saves()
            if self._future_tick():
                if self.status in self.TERMINAL:
                    self._schedule_partial_save()
                return
            if self.status in self.TERMINAL:
                self._drain_configuration()
                self._schedule_partial_save()
                return
            if self.status not in ('preparing', 'finalizing', 'replay_preflight'):
                if self.clock()-self._phase_started > self.STATION_TIMEOUT_S:
                    gate = self._station_gate.get('gate', self.status)
                    position = self._station_gate.get('position_error_m')
                    orientation = self._station_gate.get('orientation_error_rad')
                    raise TimeoutError('Weld station failed fresh measured convergence/capture within 30 seconds; '
                        f'gate={gate}, position_error_m={position}, orientation_error_rad={orientation}, '
                        f'endpoint_replans={self._station_retry_attempt}')
            if self.status == 'replay_preflight':
                return
            if self.future is not None:
                return
            camera, feedback, pose = self._sample()
            if camera.get('error'):
                raise ValueError(f'Camera failed during weld finding: {camera["error"]}')
            if self.status == 'preparing':
                if self.clock()-self._phase_started > 30.:
                    raise TimeoutError('Camera preparation timed out')
                if not self._prepare_camera(camera):
                    return
                self._camera_identity(camera)
                self._baseline = transform(self.service.move_laser_saved_pose('Weld Finder'))
                self._begin_settle(self._baseline, 'capture')
            elif self.status in ('moving', 'settling'):
                self._settle_tick(camera, feedback, pose)
            elif self.status == 'capturing':
                self._capture(camera, feedback, pose)
            elif self.status == 'exposure_recovery':
                self._exposure_recovery_tick(camera)
            elif self.status == 'validating':
                self._analyzed(self._pending_result)
            elif self.status == 'validating_saved':
                self._process_analyzed()
            elif self.status == 'replay_wait_feedback':
                self._replay_preflight_done()
        except _OpticalVisibilityUnresolved as error:
            try:
                self._visibility_unresolved(str(error))
            except Exception as failure:
                self.error = str(failure)
                self.cancel(reason=self.error, status='error')
        except Exception as error:
            self.error = str(error)
            self.cancel(reason=self.error, status='error')

    def _visibility_unresolved(self, reason):
        """Optical loss never establishes an end; still inspect the other side.

        This is called only for the dedicated optical-exhaustion type. Motion,
        hardware, feedback, identity and IK failures retain the ordinary stop.
        """
        if self._center is None or self._direction not in (-1, 1):
            raise _OpticalVisibilityUnresolved(reason)
        direction = self._direction
        name = 'positive' if direction == 1 else 'negative'
        last = self._last_found[direction] or self._center
        self.endpoints[name] = dict(confirmed=False, reason='visibility_unresolved', detail=reason,
            distance_mm=self._distance[direction], measured_search_extent_mm=self._search_extents[direction],
            last_seam_point_base_m=last['seam_point_base_m'])
        self.revision += 1
        if direction == 1:
            self._move(self._center['measured_base_from_tcp'], 'return_negative')
            self.message = 'Positive endpoint visibility unresolved; returning to the measured center to inspect the other direction'
        else:
            self._finish()

    def cancel(self, *, reason='Weld finding cancelled; measured points retained', status='cancelled', hold=True):
        active = self.status not in self.TERMINAL
        replaying = (self.status in ('replay_preflight', 'replay_wait_feedback')
                     or active and self._after_settle == 'replay_next')
        if self._station_retry is not None and self._station_retry['status'] in ('replanning', 'settling_after_retry'):
            self._station_retry.update(status='failed' if status == 'error' else 'interrupted', error=reason,
                finished_monotonic=self.clock(), after_evidence=copy.deepcopy(self._station_gate))
            self._persist_retry()
        if self.records and not replaying and (active or self._result is None or self._result.get('status') not in ('complete', 'incomplete')
                or self._result.get('finding_status') in ('error', 'cancelled', 'interrupted')):
            self._partial_dirty = True
            if self._partial_finding_status is None:
                self._partial_finding_status, self._partial_finding_error = status, self.error
        self._generation += 1
        self._cancel_event.set()
        self._set_phase(status, reason)
        if self._configuration_pending:
            try:
                self.service.camera.dispatch(dict(action='camera_laser_cancel', request_id=self._configuration_id))
            except Exception as error:
                self.error = f'{reason}; camera configuration cancellation pending: {error}'
        if active and hold:
            try:
                self.service.ctrl.hold(automatic=False)
            except Exception as error:
                self.error = f'{reason}; measured Hold failed: {error}'
        if self.future is not None:
            self.future.cancel()
        self._schedule_partial_save()
        self._poll_file_saves()
        self.revision += 1
        return self.state()

    def abort_for_shutdown(self):
        return self.cancel(reason='Weld finding released motion to shutdown', status='interrupted', hold=False)

    def abort_for_cooldown(self):
        return self.cancel(reason='Weld finding released motion to cooldown', status='interrupted', hold=False)

    def close(self):
        self.abort_for_shutdown()
        self._closed = True
        # The application normally closes this resource after verified motor
        # disable. Drain file-only work then; no cancelled analysis may command.
        wait = bool(getattr(self.service.ctrl, 'closed', True))
        if wait and self.future is not None:
            try:
                self.future.result()
            except Exception:
                pass
            self._future_tick()  # Terminal branch records only a completed raw write.
        self._configuration_pending = False if wait else self._configuration_pending
        self._schedule_partial_save()
        self.executor.shutdown(wait=wait, cancel_futures=False)
        self._poll_file_saves()
