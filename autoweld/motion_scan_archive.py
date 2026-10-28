"""Schema-3 moving scan storage. Pure I/O/arrays; no devices, models or motion.

Every acquisition is preserved independently. Coverage bins describe observed
native points, never fabricated captures at planned grid crossings.
"""
import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported archive JSON type: {type(value).__name__}')


def _array(value, shape, label):
    result = np.asarray(value, float)
    if result.size == 0 and shape[0] is None:
        result = result.reshape(0, *shape[1:])
    if result.ndim != len(shape) or any(n is not None and n != a for n, a in zip(shape, result.shape)) or not np.isfinite(result).all():
        raise ValueError(f'{label} must be finite with shape {shape}')
    return result


def _pose(value):
    pose = _array(value, (4, 4), 'Rigid transform')
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9) or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(pose[:3, :3]) < .999999:
        raise ValueError('Archive pose must be a rigid transform')
    return pose


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, default=_json, allow_nan=False, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path, **values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class MotionScanArchive:
    """One writer with locked snapshots, append-only journal and bounded preview.

    begin metadata uses settings, handeye, baseline_pose (original Tube TCP), and
    identity=[source,optical,laser calibration,acquisition session]. An explicit
    baseline_optical can replace baseline_pose. No stationary gate is applied.
    """
    PREVIEW_LIMIT = 10000
    CHECKPOINT_INTERVAL = 64

    def __init__(self, runtime, max_points=10000):
        if type(max_points) is not int or not 1 <= max_points <= self.PREVIEW_LIMIT:
            raise ValueError('Moving preview must contain between 1 and 10000 points')
        self.directory = Path(runtime)/'motion-scans'
        self.selector = self.directory/'latest-motion.json'
        self.max_points = max_points
        self._lock = threading.RLock()
        self._reset()
        self._restore()

    def _reset(self):
        self._summary = dict(schema_version=3, mode='continuous', status='idle', scan_id=None,
            message='No continuous scan captured', native_frame_count=0, captures=0, visited=0,
            accepted=0, no_return=0, rejected=0, gaps=0, point_count=0, path_fraction=0., settings={})
        self._metadata = {}
        self._identity = None
        self._clock_id = None
        self._seen = set()
        self._surface_cells, self._camera_cells = set(), set()
        self._line_profile = False
        self._line_cells = set()
        self._line_stations = {}
        self._line_progress_mm = None
        self._line_exposure_cells = set()
        self._line_station_outcomes = {}
        self._line_anchors = set()
        self._line_moving_accepted = 0
        self._line_moving_min = self._line_moving_max = None
        self._line_last_exposure_mid = None
        self._preview = np.empty((self.max_points, 3))
        self._preview_size = self._preview_seen = 0
        self._rng = np.random.default_rng(0)
        self._preview_list = []
        self._preview_dirty = False
        self._preview_published = 0.
        self._footprint_config = None
        self._footprint_seen = set()
        self._footprint_total = 0
        self._sequence = 0
        self._last_device_id = None
        self._persistence_error = None

    @property
    def session_directory(self):
        scan_id = self._summary['scan_id']
        if not isinstance(scan_id, str) or Path(scan_id).name != scan_id:
            raise ValueError('No valid moving archive session')
        return self.directory/scan_id

    def _geometry(self, metadata):
        handeye = metadata.get('handeye') or {}
        if handeye.get('units') != 'm' or handeye.get('coordinate_frame') != 'tcp':
            raise ValueError('Measured hand-eye must use explicit TCP-frame metres')
        self._handeye = _pose(handeye.get('tcp_from_camera', metadata.get('tcp_from_camera')))
        self._baseline = (_pose(metadata['baseline_optical']) if 'baseline_optical' in metadata
                          else _pose(metadata['baseline_pose'])@self._handeye)
        settings = metadata.get('settings') or {}
        self._line_profile = metadata.get('scan_profile',settings.get('scan_profile')) == 'vertical_down'
        if self._line_profile:
            self._line_start = _pose(metadata['line_start_base_from_camera'])
            self._line_length_mm = float(metadata.get('sweep_length_mm',settings.get('sweep_length_mm',80.)))
            if not 1 <= self._line_length_mm <= 100 or self._line_length_mm != round(self._line_length_mm):
                raise ValueError('Invalid world-minus-Z line length')
            if metadata.get('direction_base') != [0.,0.,-1.] or metadata.get('scan_method') not in ('fast','precision'):
                raise ValueError('Line profile needs explicit method and world-minus-Z direction')
            expected_end = self._line_start.copy()
            expected_end[2,3] -= self._line_length_mm/1000.
            if not np.allclose(_pose(metadata['line_end_base_from_camera']),expected_end,atol=1e-10,rtol=0):
                raise ValueError('Line endpoint must preserve fixed orientation and exact world-minus-Z length')
            protocol = 'precision_stationary' if metadata['scan_method']=='precision' else 'fast_continuous'
            if metadata.get('scan_protocol') != protocol or not metadata.get('comparison_id') or not metadata.get('comparison_binding'):
                raise ValueError('Line protocol requires matching frozen comparison provenance')
        self._width = float(settings.get('width_mm', 100.))/1000.
        self._height = float(settings.get('height_mm', 100.))/1000.
        if not np.isfinite([self._width, self._height]).all() or not 0 < min(self._width, self._height) or max(self._width, self._height) > .100:
            raise ValueError('Moving archive square extents must be positive and at most 100 mm')
        identity = metadata.get('identity')
        if not isinstance(identity, (list, tuple)) or len(identity) != 4 or any(not isinstance(x, str) or not x for x in identity):
            raise ValueError('Exact source, optical, laser and acquisition-session identity required')
        self._identity = tuple(identity)
        if self._line_profile:
            import hashlib
            binding = metadata['comparison_binding']
            handeye_hash = hashlib.sha256(json.dumps(handeye.get('tcp_from_camera', metadata.get('tcp_from_camera')), sort_keys=True,
                                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if (not isinstance(binding, dict)
                    or any(binding.get(key) != value for key, value in zip(
                        ('source_id', 'optical_intrinsics_id', 'calibration_id'), identity[:3]))
                    or binding.get('sweep_length_mm') != self._line_length_mm
                    or binding.get('handeye_sha256') != handeye_hash):
                raise ValueError('Frozen line comparison source/calibration/length/hand-eye differs')
        tier = metadata.get('registration_reference', 'physical_sample_time')
        if tier not in ('physical_sample_time', 'measured_kernel_rx'):
            raise ValueError('Unknown moving registration reference')
        if tier == 'measured_kernel_rx':
            binding = metadata.get('registration_reference_metadata') or {}
            if (metadata.get('physical_sample_timing_qualified') is not False
                    or 'firmware_sample_age_s' not in metadata or metadata['firmware_sample_age_s'] is not None
                    or not all(isinstance(binding.get(key), str) and binding[key] for key in ('history_id', 'clock_mapping_id'))
                    or any(binding.get(key) != expected for key, expected in
                           zip(('source_id', 'optical_intrinsics_id', 'acquisition_session_id'),
                               (identity[0], identity[1], identity[3])))):
                raise ValueError('Receive-reference metadata must bind actual source/history/clock and preserve unknown firmware timing')
        footprint = metadata.get('scan_footprint')
        if footprint is not None:
            from autoweld.scan_footprint import validate_footprint_config
            validate_footprint_config(footprint)
            if (footprint.get('schema') != 'autoweld.reference-illumination.v1'
                    or footprint.get('native_calibration_id') != identity[2]
                    or footprint.get('source_id') != identity[0]
                    or footprint.get('optical_intrinsics_id') != identity[1]
                    or footprint.get('width_mm') != settings.get('width_mm')
                    or footprint.get('height_mm') != settings.get('height_mm')
                    or not np.allclose(_pose(footprint['baseline_optical']), self._baseline, atol=1e-9, rtol=0)):
                raise ValueError('Illumination footprint source/calibration/original ROI geometry differs')
            self._footprint_config = copy.deepcopy(footprint)
            self._footprint_total = (round(self._width*1000)+1)*(round(self._height*1000)+1)
        for key, expected in (('source_id', identity[0]), ('optical_intrinsics_id', identity[1])):
            actual = handeye.get(key, handeye.get('calibration_id') if key == 'optical_intrinsics_id' else None)
            if actual != expected:
                raise ValueError('Measured hand-eye source/optical identity differs from archive')

    def begin(self, metadata):
        with self._lock:
            if self._summary['status'] == 'scanning':
                raise RuntimeError('Finish the current moving archive before beginning another')
            # Validate before changing the visible session.
            metadata = copy.deepcopy(metadata)
            # Geometry validation assigns attributes. A temporary owner prevents
            # a failed begin from leaking new line/footprint state into the old.
            validator = copy.copy(self)
            validator._geometry(metadata)
            self._reset()
            self._geometry(metadata)
            self._metadata = metadata
            scan_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            self._summary.update(registration_reference=metadata.get('registration_reference', 'physical_sample_time'))
            if metadata.get('registration_reference') == 'measured_kernel_rx':
                self._summary.update(physical_sample_timing_qualified=False, firmware_sample_age_s=None,
                                     physical_spatial_error_bound_m=None)
            self._summary.update(status='scanning', scan_id=scan_id, settings=copy.deepcopy(metadata['settings']),
                                 started_wall_s=time.time(), message=('Recording stationary precision acquisitions'
                                    if metadata.get('scan_method') == 'precision' else 'Recording qualified moving acquisitions'))
            self.session_directory.mkdir(parents=True, exist_ok=False)
            _atomic_json(self.session_directory/'session.json', metadata)
            self._checkpoint()
            return self.state()

    def _local(self, points):
        return (points-self._baseline[:3, 3])@self._baseline[:3, :3]

    def _roi_mask(self, points):
        if self._line_profile:
            return np.ones(len(points),dtype=bool)  # Line exports never use the old100mm XY crop.
        local = self._local(points)
        return (np.abs(local[:, 0]) <= self._width/2+1e-12) & (np.abs(local[:, 1]) <= self._height/2+1e-12)

    def _cells(self, points):
        points = points[self._roi_mask(points)]
        if not len(points):
            return set()
        xy = self._local(points)[:, :2]
        if self._line_profile:
            return {tuple(x) for x in np.floor(xy / .001).astype(int).tolist()}
        sizes = np.ceil(np.array([self._width, self._height])/.001).astype(int)
        bins = np.floor((xy+np.array([self._width, self._height])/2)/.001).astype(int)
        bins = np.minimum(bins, sizes-1)
        return {tuple(x) for x in bins.tolist()}

    def _add_preview(self, world):
        world = world[self._roi_mask(world)]
        for point in world:
            self._preview_seen += 1
            if self._preview_size < self.max_points:
                self._preview[self._preview_size] = point
                self._preview_size += 1
            else:
                index = self._rng.integers(self._preview_seen)
                if index < self.max_points:
                    self._preview[index] = point
        self._preview_dirty = True

    def _validate(self, cloud, registration, selected_indices, status):
        identity = (cloud.get('source_id'), cloud.get('optical_intrinsics_id', cloud.get('calibration_id')),
                    cloud.get('calibration_id'), cloud.get('acquisition_session_id'))
        if identity != self._identity:
            raise ValueError('Acquisition changes source, calibration or camera session')
        for key, expected in zip(('source_id', 'optical_intrinsics_id', 'acquisition_session_id'), (identity[0], identity[1], identity[3])):
            if registration.get(key) != expected:
                raise ValueError('Registrar identity differs from native acquisition')
        sequence, generation = cloud.get('sequence'), cloud.get('generation')
        if type(sequence) is not int or sequence < 0 or type(generation) is not int or generation < 0 or registration.get('sequence') != sequence:
            raise ValueError('Native generation/sequence and registrar sequence must match')
        timing = cloud.get('timing') or {}
        device_id = timing.get('device_frame_id')
        if (type(device_id) is not int or not 0 <= device_id < 2**64
                or timing.get('chunk_frame_id') != device_id
                or timing.get('exposure_end_frame_id') != device_id):
            raise ValueError('Exact native device/chunk/exposure-event frame identity required')
        acquisition = (*identity, device_id)
        if acquisition in self._seen:
            raise ValueError('Duplicate native acquisition device FrameID')
        if self._last_device_id is not None and device_id <= self._last_device_id:
            raise ValueError('Native device FrameID regressed')
        receive_reference = self._metadata.get('registration_reference') == 'measured_kernel_rx'
        if receive_reference:
            binding = self._metadata['registration_reference_metadata']
            if (registration.get('registration_reference') != 'measured_kernel_rx'
                    or registration.get('registration_method') != 'measured_output_receive_reference'
                    or registration.get('reference_registered') is not True
                    or registration.get('reference_timing_qualified') is not True
                    or registration.get('timing_qualified') is not False
                    or registration.get('physical_sample_timing_qualified') is not False
                    or 'firmware_sample_age_s' not in registration or registration['firmware_sample_age_s'] is not None
                    or registration.get('physical_spatial_error_bound_m') is not None
                    or any(registration.get(key) is not None for key in ('timing_uncertainty_s', 'spatial_error_bound_m', 'joint_error_rad'))
                    or registration.get('applied_empirical_delay_s') != 0.
                    or registration.get('applied_delay_s', 0.) != 0.
                    or registration.get('delay_qualification_id') is not None
                    or registration.get('history_id') != binding['history_id']
                    or registration.get('clock_mapping_id') != binding['clock_mapping_id']):
                raise ValueError('Explicit measured kernel-receive reference must preserve unknown physical timing and apply no empirical delay')
        elif (registration.get('timing_qualified') is not True
                or registration.get('registration_reference') == 'measured_kernel_rx'
                or registration.get('registration_method') != 'measured_output_interpolation'):
            raise ValueError('Explicit qualified measured-angle registration without extrapolation required')
        if registration.get('no_extrapolation') is not True:
            raise ValueError('Measured-angle registration must forbid extrapolation')
        timing = cloud.get('timing') or {}
        mapping = timing.get('clock_mapping') or {}
        if timing.get('motion_pairing_supported') is not True or mapping.get('status') != 'qualified' or timing.get('host_clock_domain') != 'host_monotonic':
            raise ValueError('Native exposure mapping is not qualified')
        clock_id = registration.get('clock_mapping_id')
        if not isinstance(clock_id, str) or not clock_id or mapping.get('mapping_id') != clock_id or (self._clock_id is not None and clock_id != self._clock_id):
            raise ValueError('Clock mapping/session identity changed')
        midpoint = float(registration['exposure_monotonic_s'])
        interval = _array(registration['exposure_interval_s'], (2,), 'Actual exposure interval')
        actual_mid = timing.get('exposure_host_mid_s', timing.get('exposure_host_midpoint_s'))
        if actual_mid != midpoint or interval.tolist() != [timing.get('exposure_host_start_s'), timing.get('exposure_host_end_s')] or abs(midpoint-float(np.mean(interval))) > 1e-6 or interval[1] < interval[0]:
            raise ValueError('Registrar exposure differs from actual native midpoint/interval')
        q = _array(registration['joint_pos'], (6,), 'Measured joint angles')
        pose = _pose(registration['base_from_tcp'])
        brackets = _array(registration['robot_brackets_monotonic_s'], (6, 2), 'Measured brackets')
        query = midpoint+float(registration.get('applied_delay_s', 0.))
        if not np.isfinite(query) or np.any(brackets[:, 0] > query) or np.any(brackets[:, 1] < query):
            raise ValueError('Measured pose extrapolates outside its motor brackets')
        # These reference estimates bound interpolation relative to kernel RX;
        # they do not establish the unknown physical encoder sample time.
        limits = ([('reference_timing_uncertainty_s', 'max_reference_timing_uncertainty_s', .002),
                   ('reference_spatial_error_bound_m', 'max_reference_spatial_error_m', .001)]
                  if receive_reference else
                  [('timing_uncertainty_s', 'max_timing_uncertainty_s', .002),
                   ('spatial_error_bound_m', 'max_spatial_error_m', .0005)])
        for value_key, limit_key, ceiling in limits:
            value, limit = float(registration[value_key]), float(registration[limit_key])
            if not np.isfinite([value, limit]).all() or not 0 <= value <= limit <= ceiling:
                raise ValueError('Registration uncertainty/error exceeds its qualified bounds')
        if receive_reference:
            reference_uncertainty = float(registration['reference_timing_uncertainty_s'])
            for value in (registration['reference_clock_uncertainty_s'], timing['exposure_time_uncertainty_s']):
                if not np.isfinite(value) or not 0 <= value <= reference_uncertainty:
                    raise ValueError('Camera/kernel clock uncertainty exceeds receive-reference estimate')
        max_exposure = float(registration['max_exposure_s'])
        if not np.isfinite(max_exposure) or not 0 < max_exposure <= .020 or interval[1]-interval[0] > max_exposure+1e-12:
            raise ValueError('Exposure blur interval exceeds qualified limit')
        max_gap = float(registration['robot_max_gap_s'])
        gap_limit = float(registration['max_reference_gap_s']) if receive_reference else .020
        gap_ceiling = .100 if receive_reference else .020
        if (not np.isfinite([max_gap, gap_limit]).all() or not 0 < gap_limit <= gap_ceiling
                or not 0 <= max_gap <= gap_limit or np.any(np.diff(brackets, axis=1) > max_gap+1e-12)):
            raise ValueError('Measured motor bracket gap exceeds limit')
        joint_error_key = 'reference_joint_error_rad' if receive_reference else 'joint_error_rad'
        if np.any(_array(registration[joint_error_key], (6,), 'Measured joint errors') < 0):
            raise ValueError('Measured joint errors must be nonnegative')
        actual_exposure = timing.get('exposure_duration_us')
        if (timing.get('sensor_shutter_mode') != 'global' or actual_exposure is None
                or not np.isfinite(actual_exposure) or actual_exposure <= 0
                or abs(actual_exposure*1e-6-(interval[1]-interval[0])) > 1e-6):
            raise ValueError('Actual native global-shutter exposure duration is required')
        points = _array(cloud.get('xyz_m'), (None, 3), 'Native XYZ')
        world = _array(registration.get('points_world_m'), (None, 3), 'Registered native XYZ')
        uv = _array(cloud.get('uv'), (None, 2), 'Native pixels')
        strength = _array(cloud.get('strength'), (None,), 'Native strength')
        widths = _array(cloud.get('width_px'), (None,), 'Native widths')
        if any(len(x) != len(points) for x in (world, uv, strength, widths)):
            raise ValueError('Native channels and registered world XYZ must align')
        if cloud.get('coordinate_frame') != 'camera' or cloud.get('xyz_m_units', cloud.get('units')) != 'm' or (cloud.get('frame_product') or {}).get('kind') != 'single_frame':
            raise ValueError('Explicit native single-frame camera XYZ metres required')
        expected = points@(pose@self._handeye)[:3, :3].T+(pose@self._handeye)[:3, 3]
        if not np.allclose(world, expected, atol=1e-9, rtol=0):
            raise ValueError('World XYZ does not match measured pose and original hand-eye')
        indices = np.arange(len(points), dtype=np.int64) if selected_indices is None else np.asarray(selected_indices)
        if indices.size == 0:
            indices = np.empty(0, np.int64)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(points)):
            raise ValueError('Selected native indices must be unique and in bounds')
        if status == 'accepted' and not len(indices):
            raise ValueError('Accepted frame requires selected native returns')
        if status == 'no_return' and len(points):
            raise ValueError('No-return acquisition must contain zero native returns')
        return acquisition, clock_id, points, uv, strength, widths, world, indices, q, pose

    def _line_observation(self, cloud, registration, context):
        """Validate measured line/quiet evidence without changing archive state."""
        from scipy.spatial.transform import Rotation

        camera = _pose(registration['base_from_tcp']) @ self._handeye
        offset = camera[:3, 3] - self._line_start[:3, 3]
        down = -float(offset[2]) * 1000.
        angle = Rotation.from_matrix(camera[:3, :3] @ self._line_start[:3, :3].T).magnitude()
        on_line = (np.linalg.norm(offset[:2]) <= .0005 and angle <= .005
                   and -.5 <= down <= self._line_length_mm + .5)
        midpoint = float(registration['exposure_monotonic_s'])
        if self._line_last_exposure_mid is not None and midpoint <= self._line_last_exposure_mid:
            raise ValueError('Line observations require fresh increasing exposure times')
        context = context or {}
        phase = context.get('acquisition_phase')
        index = None
        if phase in ('stationary_main', 'start_anchor', 'end_anchor'):
            index = context.get('station_index') if phase == 'stationary_main' else (0 if phase == 'start_anchor' else round(self._line_length_mm))
            if type(index) is not int or not 0 <= index <= self._line_length_mm:
                raise ValueError('Stationary line exposure needs a valid frozen station index')
            requested_camera = self._line_start.copy()
            requested_camera[2, 3] -= index / 1000.
            if np.linalg.norm(camera[:3, 3] - requested_camera[:3, 3]) > .0005 or angle > .005:
                raise ValueError('Stationary line exposure misses its actual frozen station pose')
            requested = requested_camera @ np.linalg.inv(self._handeye)
            if not np.allclose(_pose(context.get('requested_base_from_tcp')), requested, atol=1e-9, rtol=0):
                raise ValueError('Stationary target differs from frozen world-minus-Z station')
            settle = context.get('settle') or {}
            names = ('pre_hold_qualified_monotonic', 'hold_monotonic', 'post_hold_stable_since', 'accepted_monotonic')
            times = [settle.get(name) for name in names]
            if (any(type(t) not in (int, float) or not np.isfinite(t) for t in times)
                    or any(a > b for a, b in zip(times, times[1:]))
                    or times[3] - times[2] < .4 - 1e-9 or settle.get('quiet_seconds', 0.) < .4):
                raise ValueError('Stationary exposure lacks ordered single-Hold quiet dwell evidence')
            interval = _array(registration['exposure_interval_s'], (2,), 'Stationary exposure interval')
            if interval[0] < times[3]:
                raise ValueError('Stationary exposure predates the verified quiet Hold')
            for name in ('requested_base_from_tcp', 'measured_base_from_tcp'):
                measured = _pose(settle.get(name))
                error_r = Rotation.from_matrix(measured[:3, :3] @ requested[:3, :3].T).magnitude()
                if np.linalg.norm(measured[:3, 3] - requested[:3, 3]) > .0005 or error_r > .005:
                    raise ValueError('Settled pose differs from frozen line station')
            feedback = context.get('stationary_feedback') or {}
            measured_at, observed_at = feedback.get('measured_at'), feedback.get('observed_monotonic')
            if (feedback.get('fresh') is not True or feedback.get('stationary') is not True
                    or any(type(t) not in (int, float) or not np.isfinite(t) for t in (measured_at, observed_at))
                    or not interval[1] <= measured_at <= observed_at
                    or observed_at - measured_at > .100):
                raise ValueError('Stationary exposure lacks fresh post-exposure quiet SDK feedback')
        return down, on_line, index, phase

    def append(self, cloud, registration, selected_indices=None, *, status='accepted', reason=None, path_fraction=None, acquisition_metadata=None):
        with self._lock:
            if self._summary['status'] != 'scanning':
                raise RuntimeError('Begin a moving scan before appending')
            status = 'accepted' if status == 'captured' else status
            if status not in ('accepted', 'no_return', 'rejected') or (status != 'accepted' and not reason):
                raise ValueError('Frame needs an accepted/no_return/rejected outcome and rejection reason')
            values = self._validate(cloud, registration, selected_indices, status)
            acquisition, clock_id, points, uv, strength, widths, world, indices, q, pose = values
            if self._line_profile:
                self._line_observation(cloud, registration, acquisition_metadata)
            fraction = self._fraction(path_fraction)
            ordinal = self._sequence
            filename = f'frame-{ordinal:08d}.npz'
            raw_path = self.session_directory/filename
            # Native channels live once in the per-frame NPZ, never in JSONL or
            # the embedded provenance JSON. The bridge also exposes xyz_mm.
            dense_fields = {'xyz_m', 'xyz_mm', 'xyz', 'xyz_camera_m', 'xyz_world_m',
                'points_world_m', 'points_base_m', 'uv', 'strength', 'width_px',
                'frame', 'image', 'preview', 'raw_frame', 'native_frame', 'preview_frame',
                'frame_jpeg_base64', 'image_rgb', 'image_gray', 'rgb', 'colors'}
            cloud_metadata = {key: value for key, value in cloud.items() if key not in dense_fields}
            registration_metadata = {key: value for key, value in registration.items() if key != 'points_world_m'}
            illumination = None
            if self._footprint_config is not None:
                from autoweld.scan_footprint import footprint_cells
                illumination = footprint_cells(self._footprint_config, pose@self._handeye)
            record = dict(kind='frame', ordinal=ordinal, status=status, reason=reason, raw_file=filename,
                          acquisition_identity=list(acquisition), cloud=cloud_metadata,
                          registration=registration_metadata, point_count=len(indices) if status == 'accepted' else 0,
                          path_fraction=fraction,
                          roi_point_count=int(self._roi_mask(world[indices]).sum()) if status == 'accepted' else 0)
            if illumination is not None:
                record['reference_illumination'] = illumination
            if acquisition_metadata is not None:
                record['scan_context'] = copy.deepcopy(acquisition_metadata)
            try:
                _atomic_npz(raw_path, xyz_camera_m=points, uv=uv, strength=strength, width_px=widths,
                    xyz_world_m=world, selected_indices=indices, joint_pos=q, base_from_tcp=pose,
                    tcp_from_camera=self._handeye, provenance_json=np.array(json.dumps(record, default=_json, allow_nan=False)))
                self._journal(record)
            except OSError as error:
                self._persistence_error = str(error)
                self._summary.update(status='error', message=f'Frame persistence failed; restore durable journal before retry: {error}')
                raise
            self._observe(record, world=world, indices=indices)
            self._clock_id = clock_id
            self._sequence += 1
            if self._summary['visited'] % self.CHECKPOINT_INTERVAL == 0:
                self._checkpoint()
            return self.state()

    def _fraction(self, value):
        if value is None:
            return self._summary['path_fraction']
        value = float(value)
        if not np.isfinite(value) or not self._summary['path_fraction'] <= value <= 1.:
            raise ValueError('Measured path fraction must be monotonic and between zero and one')
        return value

    def _journal(self, record):
        with (self.session_directory/'frames.jsonl').open('a') as handle:
            handle.write(json.dumps(record, default=_json, allow_nan=False, separators=(',', ':'))+'\n')
            handle.flush()
            os.fsync(handle.fileno())

    def _observe(self, record, *, world=None, indices=None):
        self._summary['path_fraction'] = record['path_fraction']
        if record['kind'] == 'gap':
            self._summary['gaps'] += 1
            return
        if Path(record['raw_file']).name != record['raw_file'] or not record['raw_file'].startswith('frame-'):
            raise ValueError('Invalid native frame journal path')
        if not (self.session_directory/record['raw_file']).is_file():
            raise ValueError('Journal references a missing native frame')
        identity = tuple(record['acquisition_identity'])
        if identity in self._seen:
            raise ValueError('Journal repeats a native acquisition')
        self._seen.add(identity)
        self._last_device_id = int(record['cloud']['timing']['device_frame_id'])
        for key in ('native_frame_count', 'visited'):
            self._summary[key] += 1
        self._summary[record['status']] += 1
        self._summary['point_count'] += record['point_count']
        self._clock_id = record['registration']['clock_mapping_id']
        pose = np.asarray(record['registration']['base_from_tcp'])@self._handeye
        if self._line_profile:
            down, on_line, index, phase = self._line_observation(
                record['cloud'], record['registration'], record.get('scan_context'))
            self._line_last_exposure_mid = float(record['registration']['exposure_monotonic_s'])
            self._line_progress_mm = down
            if on_line:
                cell = int(np.clip(round(down), 0, round(self._line_length_mm)))
                self._line_exposure_cells.add(cell)
                if record['status'] != 'rejected':
                    self._line_cells.add(cell)
            if index is not None:
                outcomes = self._line_station_outcomes.setdefault(index, dict(accepted=0, no_return=0, rejected=0))
                outcomes[record['status']] += 1
                if record['status'] != 'rejected':
                    if phase == 'stationary_main':
                        self._line_stations[index] = self._line_stations.get(index, 0) + 1
                    if index in (0, round(self._line_length_mm)):
                        self._line_anchors.add(index)
            if phase == 'moving_main' and record['status'] == 'accepted' and on_line:
                self._line_moving_accepted += 1
                self._line_moving_min = down if self._line_moving_min is None else min(down, self._line_moving_min)
                self._line_moving_max = down if self._line_moving_max is None else max(down, self._line_moving_max)
        self._camera_cells.update(self._cells(pose[None, :3, 3]))
        if self._footprint_config is not None and record['status'] in ('accepted', 'no_return'):
            illumination = record.get('reference_illumination')
            if illumination is None:
                raise ValueError('Missing measured-pose illumination footprint provenance')
            cells = illumination.get('cells')
            if not isinstance(cells, list) or any(type(index) is not int or not 0 <= index < self._footprint_total for index in cells):
                raise ValueError('Invalid illumination grid-cell provenance')
            self._footprint_seen.update(cells)
        if record['status'] == 'accepted':
            self._summary['captures'] += 1
            if world is None:
                with np.load(self.session_directory/record['raw_file'], allow_pickle=False) as native:
                    world, indices = native['xyz_world_m'], native['selected_indices']
            selected = world[indices]
            self._surface_cells.update(self._cells(selected))
            self._add_preview(selected)

    def record_gap(self, reason, *, start_monotonic_s, end_monotonic_s, path_fraction=None, evidence=None):
        with self._lock:
            if self._summary['status'] != 'scanning':
                raise RuntimeError('No moving scan is active')
            interval = _array([start_monotonic_s, end_monotonic_s], (2,), 'Gap interval')
            if not reason or interval[0] > interval[1]:
                raise ValueError('Coverage gaps require a reason and ordered time interval')
            record = dict(kind='gap', reason=str(reason), interval_monotonic_s=interval.tolist(),
                          path_fraction=self._fraction(path_fraction), evidence=copy.deepcopy(evidence or {}))
            self._journal(record)
            self._observe(record)
            return self.state()

    def _checkpoint(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        summary = copy.deepcopy(self._summary)
        summary.update(metadata_path=str(self.session_directory/'session.json') if summary['scan_id'] else None)
        _atomic_json(self.selector, summary)
        if summary['scan_id']:
            _atomic_json(self.session_directory/'summary.json', summary)
            _atomic_npz(self.session_directory/'preview.npz', points_base_m=self._preview[:self._preview_size])

    def _records(self):
        path = self.session_directory/'frames.jsonl'
        if path.exists():
            with path.open('rb') as handle:
                for line in handle:
                    if not line.endswith(b'\n'):
                        break
                    yield json.loads(line)

    def _restore(self):
        if not self.selector.exists():
            return
        try:
            saved = json.loads(self.selector.read_text())
            if saved.get('schema_version') != 3:
                raise ValueError('Unsupported moving archive schema')
            if not saved.get('scan_id'):
                return
            self._summary.update(scan_id=saved['scan_id'])
            metadata = json.loads((self.session_directory/'session.json').read_text())
            self._geometry(metadata)
            self._metadata = metadata
            self._summary.update(settings=copy.deepcopy(metadata['settings']))
            for record in self._records():
                if self._line_profile and record['kind'] == 'frame':
                    filename = record['raw_file']
                    if Path(filename).name != filename:
                        raise ValueError('Invalid line native frame journal path')
                    with np.load(self.session_directory/filename, allow_pickle=False) as native:
                        if json.loads(str(native['provenance_json'])) != record:
                            raise ValueError('Line journal differs from preserved native frame provenance')
                        cloud = dict(record['cloud'], xyz_m=native['xyz_camera_m'], uv=native['uv'],
                                     strength=native['strength'], width_px=native['width_px'])
                        registration = dict(record['registration'], points_world_m=native['xyz_world_m'])
                        if (not np.array_equal(native['base_from_tcp'], registration['base_from_tcp'])
                                or not np.array_equal(native['tcp_from_camera'], self._handeye)):
                            raise ValueError('Line native pose or hand-eye differs from preserved provenance')
                        self._validate(cloud, registration, native['selected_indices'], record['status'])
                self._observe(record)
                if record['kind'] == 'frame':
                    self._sequence = max(self._sequence, record['ordinal']+1)
            self._summary.update({key: value for key, value in saved.items() if key not in (
                'native_frame_count', 'visited', 'captures', 'accepted', 'no_return', 'rejected', 'gaps', 'point_count', 'path_fraction')})
            if saved.get('status') == 'complete' and self._line_profile:
                self._check_line_completion(saved.get('completion_evidence') or {})
            if saved.get('status') == 'complete' and saved.get('completion_evidence', {}).get('measured_path_complete') is True:
                self._summary['path_fraction'] = 1.
            if self._summary['status'] == 'scanning':
                self._summary.update(status='interrupted', message='Service restarted; moving partial retained')
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._summary.update(status='error', message=f'Moving archive restore failed: {error}')

    def _export(self):
        paths = [self.session_directory/'full-world.ply', self.session_directory/'roi-world.ply']
        counts = [0, 0]
        for record in self._records():
            if record['kind'] == 'frame' and record['status'] == 'accepted':
                counts[0] += record['point_count']
                if 'roi_point_count' in record:
                    counts[1] += record['roi_point_count']
                else:
                    # Backward compatibility for existing schema-3 journals.
                    with np.load(self.session_directory/record['raw_file'], allow_pickle=False) as native:
                        points = native['xyz_world_m'][native['selected_indices']]
                        counts[1] += int(self._roi_mask(points).sum())
        handles = []
        written = [0, 0]
        try:
            for path, count in zip(paths, counts):
                handle = path.with_suffix('.ply.tmp').open('wb')
                handle.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n'.encode('ascii'))
                handles.append(handle)
            for record in self._records():
                if record['kind'] == 'frame' and record['status'] == 'accepted':
                    with np.load(self.session_directory/record['raw_file'], allow_pickle=False) as native:
                        points = native['xyz_world_m'][native['selected_indices']]
                        for index, (handle, selected) in enumerate(zip(handles, (points, points[self._roi_mask(points)]))):
                            # One bounded native view at a time; exact float64
                            # coordinates, with an explicit portable byte order.
                            handle.write(np.asarray(selected, dtype='<f8', order='C').tobytes(order='C'))
                            written[index] += len(selected)
            if written != counts:
                raise ValueError('PLY vertex counts disagree with native frame journal')
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            for path in paths:
                os.replace(path.with_suffix('.ply.tmp'), path)
        finally:
            for handle in handles:
                handle.close()
        report = dict(schema_version=3, full_point_count=counts[0], roi_point_count=counts[1],
                      original_baseline_optical=self._baseline.tolist(), roi_width_m=self._width,
                      roi_height_m=self._height, units='m', coordinate_frame='robot_base',
                      native_views_preserved=True, full_ply=str(paths[0]), roi_ply=str(paths[1]),
                      ply_format='binary_little_endian', point_dtype='<f8', vertex_stride_bytes=24)
        if self._line_profile:
            report.update(scan_profile='vertical_down',roi_mode='all_selected_line_points',
                sweep_length_mm=self._line_length_mm,direction_base=[0.,0.,-1.])
        _atomic_json(self.session_directory/'roi-export.json', report)
        self._summary['roi_point_count'] = counts[1]

    def _check_line_completion(self, evidence):
        if (evidence.get('measured_path_complete') is not True or evidence.get('line_path_complete') is not True
                or not self._summary['accepted'] or evidence.get('path_fraction') != 1.):
            raise ValueError('Line completion needs measured endpoint and actual accepted captures')
        if self._line_anchors != {0, round(self._line_length_mm)}:
            raise ValueError('Line completion needs actual stationary start and end exposures')
        if self._metadata.get('scan_method') == 'precision':
            if any(self._line_stations.get(i, 0) < 3 for i in range(round(self._line_length_mm) + 1)):
                raise ValueError('Precision line needs three fresh captures at all 1 mm stations')
        elif (self._line_moving_accepted < 3 or self._line_moving_min is None
              or self._line_moving_max - self._line_moving_min < self._line_length_mm - 1.):
            raise ValueError('Fast line needs accepted moving exposures spanning the measured line')

    def finish(self, status='complete', message=None, *, completion_evidence=None):
        with self._lock:
            if not self._summary['scan_id']:
                raise RuntimeError('No moving scan to finish')
            if self._persistence_error is not None:
                raise RuntimeError('Frame commit outcome is uncertain; restore durable archive before final export')
            if status not in ('complete', 'error', 'cancelled', 'interrupted'):
                raise ValueError('Invalid moving scan terminal status')
            if status == 'complete':
                if self._footprint_config is not None and len(self._footprint_seen) != self._footprint_total:
                    raise ValueError('Reference-plane illumination coverage is incomplete')
                evidence = copy.deepcopy(completion_evidence or {})
                if self._line_profile:
                    self._check_line_completion(evidence)
                elif (evidence.get('measured_path_complete') is not True or evidence.get('continuous_quality_coverage') is not True
                        or evidence.get('path_fraction') != 1. or (self._footprint_config is None and self._summary['gaps'] and evidence.get('coverage_gaps_resolved') is not True)
                        or not self._summary['native_frame_count'] or not self._summary['accepted']):
                    raise ValueError('Completion requires measured path and continuous quality coverage evidence')
                self._summary['path_fraction'] = 1.
            if completion_evidence is not None:
                self._summary['completion_evidence'] = copy.deepcopy(completion_evidence)
            self._export()
            self._summary.update(status=status, message=message or f'Continuous scan {status}', finished_wall_s=time.time())
            self._checkpoint()
            # The journal IS the complete append-only provenance manifest.
            _atomic_json(self.session_directory/'manifest.json', dict(summary=self._summary,
                metadata=self._metadata, frames_journal='frames.jsonl', all_native_views_preserved=True))
            return self.state()

    def state(self, include_points=True):
        with self._lock:
            result = copy.deepcopy(self._summary)
            if self._preview_dirty and (not self._preview_list or time.monotonic()-self._preview_published >= .25):
                self._preview_list = self._preview[:self._preview_size].tolist()
                self._preview_dirty = False
                self._preview_published = time.monotonic()
            if self._line_profile:
                result.update(scan_profile='vertical_down',scan_method=self._metadata.get('scan_method'),
                    sweep_length_mm=self._line_length_mm,line_progress_mm=self._line_progress_mm,
                    line_samples_observed=len(self._line_cells),station_total=round(self._line_length_mm)+1,
                    stations_complete=sum(count>=3 for count in self._line_stations.values()),
                    station_capture_counts={str(i):count for i,count in self._line_stations.items()},
                    station_outcomes={str(i):dict(counts) for i, counts in self._line_station_outcomes.items()},
                    line_exposure_bins_observed=len(self._line_exposure_cells),
                    line_start_observed=0 in self._line_anchors,
                    line_end_observed=round(self._line_length_mm) in self._line_anchors,
                    moving_main_accepted_count=self._line_moving_accepted,
                    moving_main_axial_min_mm=self._line_moving_min,
                    moving_main_axial_max_mm=self._line_moving_max,
                    moving_main_axial_span_mm=(None if self._line_moving_min is None
                                              else self._line_moving_max - self._line_moving_min))
            result.update(display_point_count=len(self._preview_list), coordinate_frame='robot_base', units='m',
                coverage=dict(observed_1mm_xy_voxel_cells=len(self._surface_cells),
                              observed_camera_xy_cells_1mm=len(self._camera_cells),
                              native_frame_count=self._summary['native_frame_count'],
                              path_fraction=self._summary['path_fraction'], coverage_gaps=self._summary['gaps'],
                              footprint_observed_cells_1mm=len(self._footprint_seen),
                              footprint_total_cells_1mm=self._footprint_total,
                              footprint_complete=bool(self._footprint_config is not None and len(self._footprint_seen)==self._footprint_total),
                              footprint_basis='assumed-on laser intersection with fixed reference plane; not measured surface occupancy',
                              reference_plane_z_m=(self._footprint_config or {}).get('reference_plane_z_m')))
            points = [point[:] for point in self._preview_list] if include_points else []
            result['cloud'] = dict(points_base_m=points, point_count=self._summary['point_count'],
                display_point_count=len(self._preview_list), coordinate_frame='robot_base', units='m', points_are_preview=True)
            if self._summary['scan_id']:
                folder = self.session_directory
                result['export_paths'] = dict(metadata=str(folder/'session.json'), journal=str(folder/'frames.jsonl'),
                    raw_snapshots=str(folder), manifest=str(folder/'manifest.json'), ply=str(folder/'full-world.ply'),
                    roi_ply=str(folder/'roi-world.ply'), roi_metadata=str(folder/'roi-export.json'))
            else:
                result['export_paths'] = {}
            return result

    def footprint_missing_cells(self):
        """Bounded repair-worker query, never a telemetry array of planned poses."""
        with self._lock:
            return [index for index in range(self._footprint_total) if index not in self._footprint_seen]

    def clear(self):
        with self._lock:
            if self._summary['status'] == 'scanning':
                raise RuntimeError('Finish active moving scan before clearing its display')
            self._reset()
            self._checkpoint()
            return self.state()
