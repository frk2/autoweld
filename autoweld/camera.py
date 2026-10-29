"""Process-owned camera bridge. Hardware and vision live in Weldfinder."""

import copy
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class _PublishedCloud(dict):
    """Cached immutable dict; an explicit deepcopy gives the caller private data."""
    def _readonly(self, *args, **kwargs):
        raise TypeError('Published native cloud is immutable')
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


def _freeze_published(value):
    if isinstance(value, _PublishedCloud):
        return value
    if isinstance(value, dict):
        return _PublishedCloud({key: _freeze_published(item) for key,item in value.items()})
    if isinstance(value, (tuple,list)):
        return tuple(_freeze_published(item) for item in value)
    if hasattr(value, 'flags') and hasattr(value, 'shape'):
        value.flags.writeable = False
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, size):
    return isinstance(value, list) and len(value) == size and all(_number(item) for item in value)


def _matrix(value, rows, columns):
    return isinstance(value, list) and len(value) == rows and all(_vector(row, columns) for row in value)


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def _read_json(path):
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def _intrinsic_identity(model, source_id):
    """Validate installed intrinsic JSON without loading the vision backend."""
    if not isinstance(model, dict) or model.get('source_id') != source_id:
        raise ValueError('Installed intrinsics belong to another camera source')
    if model.get('schema_version') != 1 or model.get('synthetic') is not False:
        raise ValueError('Installed intrinsics must be a supported nonsynthetic model')
    size = model.get('image_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError('Invalid intrinsic image resolution')
    matrix, distortion = model.get('camera_matrix'), model.get('distortion')
    if (not _matrix(matrix, 3, 3) or matrix[0][0] <= 0 or matrix[1][1] <= 0
            or any(abs(value-target) > 1e-8 for value, target in zip(matrix[2], (0, 0, 1)))):
        raise ValueError('Invalid intrinsic camera matrix')
    if not isinstance(distortion, list) or len(distortion) not in (4, 5, 8, 12, 14) or not all(_number(v) for v in distortion):
        raise ValueError('Invalid intrinsic distortion coefficients')
    rms = model.get('camera_rms_px')
    if not _number(rms) or rms < 0:
        raise ValueError('Invalid intrinsic reprojection residual')
    board = model.get('board')
    if (not isinstance(board, dict)
            or any(type(board.get(key)) is not int or not 3 <= board[key] <= 40 for key in ('columns', 'rows'))
            or not _number(board.get('square_mm')) or not .1 <= board['square_mm'] <= 500):
        raise ValueError('Invalid intrinsic checkerboard dimensions')
    plane = model.get('laser_plane')
    if plane is not None:
        if not _vector(plane, 4):
            raise ValueError('Invalid intrinsic laser plane')
        normal = math.sqrt(sum(value*value for value in plane[:3]))
        if normal < 1e-8 or abs(plane[3])/normal < .1:
            raise ValueError('Degenerate intrinsic laser plane')
    rvec, tvec = model.get('reference_rvec'), model.get('reference_tvec')
    if ((rvec is None) != (tvec is None)
            or any(value is not None and not _vector(value, 3) for value in (rvec, tvec))
            or tvec is not None and tvec[2] <= 0):
        raise ValueError('Invalid intrinsic reference pose')
    canonical = json.dumps(model, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest(), tuple(size)


def _optical_intrinsics_payload(model):
    return {key: model[key] for key in ('schema_version', 'source_id', 'image_size',
            'camera_matrix', 'distortion', 'board', 'synthetic')}


def _optical_intrinsics_id(model):
    return hashlib.sha256(json.dumps(_optical_intrinsics_payload(model), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _validate_handeye(result, source_id, calibration_id, image_size, optical_id=None):
    if not isinstance(result, dict) or result.get('source_id') != source_id:
        raise ValueError('Hand-eye calibration belongs to another camera source')
    if (result.get('calibration_id') != calibration_id
            and (optical_id is None or result.get('optical_intrinsics_id') != optical_id)):
        raise ValueError('Hand-eye calibration does not match installed intrinsics')
    if result.get('units') != 'm' or result.get('coordinate_frame') != 'tcp':
        raise ValueError('Hand-eye calibration must use meters in the TCP frame')
    if type(result.get('views')) is not int or result['views'] < 20:
        raise ValueError('Hand-eye calibration requires at least 20 accepted views')
    for key, limit in (('translation_rms_mm', 3.), ('rotation_rms_deg', 1.5)):
        value = result.get(key)
        if not _number(value) or not 0 <= value <= limit:
            raise ValueError(f'Invalid or excessive hand-eye residual: {key}')
    for key in ('image_size', 'resolution'):
        if key in result and result[key] != list(image_size):
            raise ValueError('Hand-eye image resolution does not match installed intrinsics')
    matrix = result.get('tcp_from_camera')
    if not _matrix(matrix, 4, 4):
        raise ValueError('Hand-eye transform must be a finite 4x4 matrix')
    if any(abs(value-target) > 1e-8 for value, target in zip(matrix[3], (0, 0, 0, 1))):
        raise ValueError('Hand-eye transform has an invalid homogeneous row')
    rotation = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            if abs(sum(rotation[k][i]*rotation[k][j] for k in range(3))-(1. if i == j else 0.)) > 1e-5:
                raise ValueError('Hand-eye rotation is not orthonormal')
    a, b, c = rotation
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(determinant-1.) > 1e-5:
        raise ValueError('Hand-eye rotation must be proper (determinant +1)')


def _weldfinder_module(name):
    # The sibling checkout is the user-requested home of reusable camera code.
    # Keep OpenCV and SDK loading out of the UI/import startup path.
    source = Path(os.environ.get('WELDFINDER_APP', Path.home() / 'Code/Weld-seam-finder/app'))
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        package = importlib.import_module('weldfinder')
        package_source = str(source / 'weldfinder')
        if source.is_dir() and package_source not in package.__path__:
            package.__path__.insert(0, package_source)
        return importlib.import_module('weldfinder.'+name)
    except ImportError as exc:
        raise RuntimeError(
            'Camera calibration needs the Weldfinder SDK checkout; set WELDFINDER_APP '
            'or install ~/Code/Weld-seam-finder/app in this environment'
        ) from exc


def _backend_class():
    return _weldfinder_module('calibration_camera').CalibrationCamera


class CameraService:
    """Singleton shared by browser clients; construction never opens hardware."""

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, workspace=None, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(workspace, **kwargs)
            return cls._instance

    def __init__(self, workspace=None, *, serial='25440983', backend_factory=None):
        self.workspace = Path(workspace or Path.cwd())
        self.serial = serial
        self._backend_factory = backend_factory
        self._backend = None
        self._lock = threading.RLock()
        self._construction_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autoweld-camera-lifecycle')
        self._operation = None
        self._operation_name = None
        self._laser_configuration = None
        self._next_stream = None
        self._analysis_mode = 'checkerboard'
        self._applied_analysis_mode = None
        self._next_analysis = None
        self._cached_state = self._empty_state()
        self._native_cloud_lock = threading.Lock()
        self._native_cloud_cache = {}
        self._error = None
        self._handeye_count = 0
        self._handeye_result = None
        self._handeye_image_size = None
        self._load_handeye()

    def _intrinsic_path(self):
        source_id = f'spinnaker:{self.serial}'
        key = hashlib.sha256(source_id.encode()).hexdigest()[:16]
        path = self.workspace / 'runtime' / 'flir' / 'data' / f'calibration-{key}.json'
        return path

    def _installed_intrinsics(self):
        source_id = f'spinnaker:{self.serial}'
        calibration_id, image_size = _intrinsic_identity(_read_json(self._intrinsic_path()), source_id)
        return source_id, calibration_id, image_size

    def _prepare_handeye(self, result):
        source_id, calibration_id, image_size = self._installed_intrinsics()
        if not isinstance(result, dict):
            raise ValueError('Hand-eye calibration must be a JSON object')
        # Full paired observations belong in the archive, not 5 Hz telemetry.
        summary = {key: value for key, value in result.items() if key != 'paired_samples'}
        detached = json.loads(json.dumps(summary, allow_nan=False))
        optical_id = None
        if 'optical_intrinsics_id' in detached:
            model = _read_json(self._intrinsic_path())
            optical_id = _optical_intrinsics_id(model)
            if (detached['optical_intrinsics_id'] != optical_id
                    or detached.get('optical_intrinsics') != _optical_intrinsics_payload(model)):
                raise ValueError('Hand-eye optical intrinsics changed or lack verified provenance')
        _validate_handeye(detached, source_id, calibration_id, image_size, optical_id)
        return detached, image_size

    def validate_handeye(self, result):
        """Return a detached validated result without changing active state."""
        return self._prepare_handeye(result)[0]

    def install_handeye(self, result):
        """Install a validated detached result; this never activates the camera."""
        detached, image_size = self._prepare_handeye(result)
        with self._lock:
            self._handeye_result = detached
            self._handeye_count = detached['views']
            self._handeye_image_size = image_size
        return detached

    def _load_handeye(self):
        path = self.workspace / 'handeye.json'
        if not path.exists():
            return
        try:
            self.install_handeye(_read_json(path))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            logging.warning('Ignoring saved hand-eye calibration %s: %s', path, exc)

    def _check_handeye_identity(self, state):
        if self._handeye_result is None:
            return
        # Disconnect clears the active source but does not invalidate an already
        # verified measurement. An explicit identity change always does.
        identity = state.get('calibration_id')
        enabled = state.get('enabled', False)
        if not enabled and identity is None:
            return
        source = state.get('source_id')
        resolution = state.get('resolution')
        optical_match = (self._handeye_result.get('optical_intrinsics_id') is not None
                         and state.get('optical_intrinsics_id') == self._handeye_result['optical_intrinsics_id'])
        changed = ((identity is None or identity != self._handeye_result['calibration_id'] and not optical_match)
            or source is not None and source != self._handeye_result['source_id']
            or resolution is not None and tuple(resolution) != self._handeye_image_size)
        if changed:
            logging.warning('Invalidating hand-eye calibration: active camera source, intrinsics or resolution changed')
            self._handeye_result = None
            self._handeye_count = 0
            self._handeye_image_size = None

    def _get_backend(self):
        with self._construction_lock:
            if self._backend is None:
                backend = (self._backend_factory or _backend_class())(
                    self.workspace, serial=self.serial)
                with self._lock:
                    self._backend = backend
            return self._backend

    @staticmethod
    def _empty_board():
        return dict(found=False, partial_found=False, corner_count=0, corners=[],
                    centroid_px=None, image_center_px=None, offset_px=None,
                    calibration_eligible=False, origin=None, x_axis=None, y_axis=None,
                    read_started=None, sequence=None)

    @staticmethod
    def _empty_state():
        return dict(enabled=False, fresh=False, error=None, source_id=None,
                    sequence=None, timestamp_monotonic=None, read_started=None, read_completed=None,
                    fps=None, resolution=None, board=CameraService._empty_board(),
                    frame_jpeg_base64=None, counts=dict(intrinsic=0, handeye=0),
                    intrinsics=None, handeye=None, starting=False, stopping=False,
                    optical_intrinsics_id=None, acquisition_session_id=None, laser={})

    def _start_operation(self, name, function):
        self._operation_name = name
        self._error = None
        self._operation = self._executor.submit(function)

    def _apply_analysis_mode(self, backend, mode):
        """Runs on the lifecycle worker; never blocks the robot service tick."""
        setter = getattr(backend, 'set_analysis_mode', None)
        if setter is None:
            if mode != 'checkerboard':
                raise RuntimeError('Laser analysis requires the Weldfinder set_analysis_mode API; update the camera SDK')
        else:
            setter(mode)
        with self._lock:
            self._applied_analysis_mode = mode

    def _stream(self, enabled):
        backend = self._get_backend()
        if enabled:
            with self._lock:
                mode = self._analysis_mode
            self._apply_analysis_mode(backend, mode)
        backend.stream(enabled)
        if enabled:
            with self._lock:
                if self._next_analysis == mode and self._analysis_mode == mode:
                    self._next_analysis = None

    def state(self, *, include_frame=True):
        with self._lock:
            if self._operation is not None and self._operation.done():
                try:
                    self._operation.result()
                except Exception as exc:
                    self._error = str(exc)
                self._operation = self._operation_name = None
            if self._operation is None and self._next_stream is not None:
                enabled, self._next_stream = self._next_stream, None
                self._start_operation('starting' if enabled else 'stopping',
                                      lambda: self._stream(enabled))
            if self._operation is None and self._next_analysis is not None:
                mode, self._next_analysis = self._next_analysis, None
                self._start_operation('configuring',
                    lambda: self._apply_analysis_mode(self._get_backend(), mode))
            pending = self._operation is not None
            backend = self._backend
            result = dict(self._cached_state)
            if not pending and backend is not None:
                result = backend.state(include_frame=include_frame)
                self._cached_state = result
                self._check_handeye_identity(result)
            result = {**result, 'counts': dict(result['counts'])}
            if pending:
                # Never wait on backend.stream's lifecycle lock in the motor tick.
                result.update(fresh=False, frame_jpeg_base64=None, read_started=None,
                              read_completed=None, timestamp_monotonic=None, sequence=None,
                              board=self._empty_board())
            result['starting'] = self._operation_name == 'starting'
            result['stopping'] = self._operation_name == 'stopping'
            result['configuring'] = self._operation_name == 'configuring'
            result['analysis_mode'] = self._analysis_mode
            result['analysis_mode_active'] = self._applied_analysis_mode
            result['analysis_pending'] = bool(self._next_analysis is not None
                or pending and self._analysis_mode != self._applied_analysis_mode)
            result['laser_configuration'] = copy.deepcopy(self._laser_configuration)
            if self._error:
                result['error'] = self._error
            result['counts']['handeye'] = self._handeye_count
            result['handeye'] = self._handeye_result
            if (self._handeye_result is not None and result.get('optical_intrinsics_id') is not None
                    and result.get('calibration_id') == self._handeye_result['calibration_id']
                    and 'optical_intrinsics_id' not in self._handeye_result):
                # A verified legacy measurement may advertise the active optical
                # identity before its archive is migrated by an explicit command.
                result['handeye'] = {**self._handeye_result,
                                     'optical_intrinsics_id': result['optical_intrinsics_id']}
            laser = dict(result.get('laser') or {})
            if pending:
                laser.update(cloud=None, frame_jpeg_base64=None, preview_frame=None)
            cloud = laser.get('cloud')
            if cloud is not None:
                points = cloud.get('xyz_mm')
                if (cloud.get('coordinate_frame') != 'camera' or cloud.get('units') != 'mm'
                        or not isinstance(points, list) or not all(_vector(point, 3) for point in points)):
                    raise ValueError('Weldfinder cloud must contain finite camera-frame XYZ in mm')
                laser['cloud'] = {**cloud, 'xyz_m': [[value/1000. for value in point] for point in points],
                                  'xyz_m_units': 'm'}
            result['laser'] = laser
            return result

    def recent_laser_clouds(self, after_key=None):
        """Read published native stripes without lifecycle advancement or SDK calls.

        Cursor is (acquisition_session_id, device_frame_id), or an integer in the
        current camera session. ``overrun``/``session_changed`` must be handled by
        a continuous collector. No main service-state/lifecycle lock is acquired.
        """
        backend = self._backend
        getter = getattr(backend, 'recent_laser_clouds', None) if backend is not None else None
        if getter is None or self._operation is not None or not getattr(backend, 'enabled', True):
            return dict(clouds=(), overrun=False, session_changed=False, oldest_key=None,
                        newest_key=None, retained_keys=(), capacity=128, acquisition_session_id=None,
                        generation=None, unavailable=True)
        native = getter(after_key)
        result = {key: value for key,value in native.items() if key != 'clouds'}
        with self._native_cloud_lock:
            retained = set(tuple(key) for key in native.get('retained_keys', ()))
            if retained:
                for key in tuple(self._native_cloud_cache):
                    if key not in retained:
                        del self._native_cloud_cache[key]
            else:
                self._native_cloud_cache.clear()
            converted = []
            for cloud in native['clouds']:
                key = (cloud.get('acquisition_session_id'), (cloud.get('timing') or {}).get('device_frame_id'))
                cached = self._native_cloud_cache.get(key)
                if cached is None:
                    points = cloud.get('xyz_mm')
                    if cloud.get('coordinate_frame') != 'camera' or cloud.get('units') != 'mm':
                        raise ValueError('Weldfinder cloud must use camera-frame millimetres')
                    if hasattr(points, 'shape'):
                        if len(points.shape) != 2 or points.shape[1] != 3:
                            raise ValueError('Native XYZ must have shape Nx3')
                        if len(points) and not all(math.isfinite(float(v)) for v in (points.min(), points.max())):
                            raise ValueError('Native XYZ must be finite')
                        xyz = points/1000.
                        xyz.flags.writeable = False
                    else:
                        if not isinstance(points, (tuple,list)) or not all(
                                isinstance(point,(tuple,list)) and len(point)==3 and all(_number(v) for v in point)
                                for point in points):
                            raise ValueError('Native XYZ must be finite Nx3 millimetres')
                        xyz = tuple(tuple(v/1000. for v in point) for point in points)
                    cached = _freeze_published({**cloud, 'xyz_m': xyz, 'xyz_m_units': 'm'})
                    self._native_cloud_cache[key] = cached
                converted.append(cached)
            # Also bound fallback backends that omit retained_keys.
            while len(self._native_cloud_cache)>128:
                del self._native_cloud_cache[next(iter(self._native_cloud_cache))]
            result['clouds'] = tuple(converted)
        return result

    def latest_laser_cloud(self):
        """Newest immutable metre cloud, or None; never calls state()/opens camera."""
        clouds = self.recent_laser_clouds()['clouds']
        return clouds[-1] if clouds else None

    def _dispatch_laser_configuration(self, request):
        """Serialize bounded laser preparation on the existing lifecycle owner.

        Cancellation keeps an already-running camera operation owned until it
        finishes. A queued operation checks its cancelled token before any SDK
        call, so a released weld session cannot configure a later camera owner.
        """
        token = request.get('request_id')
        if not isinstance(token, str) or not token or len(token) > 160:
            raise ValueError('Laser configuration requires a bounded request_id')
        with self._lock:
            previous = self._laser_configuration
            if request['action'] == 'camera_laser_cancel':
                if set(request)-{'action', 'request_id'}:
                    raise ValueError('Unknown laser cancellation field')
                if previous is not None and previous['request_id'] == token and previous['status'] in (
                        'pending', 'cancelling'):
                    previous.update(cancel_requested=True, status='cancelling')
                    if self._operation is not None and self._operation.cancel():
                        previous['status'] = 'cancelled'
                return self.state(include_frame=False)
            allowed = {'action', 'request_id', 'detection', 'noise', 'auto_exposure',
                       'depth_range_mm', 'bracket', 'controls', 'preview'}
            if set(request)-allowed:
                raise ValueError('Unknown laser configuration field')
            if previous is not None and previous['request_id'] == token:
                return self.state(include_frame=False)
            if self._operation is not None:
                raise RuntimeError('Wait for the current camera lifecycle operation')
            self._laser_configuration = dict(request_id=token, status='pending',
                                             cancel_requested=False, error=None,
                                             requested_controls=copy.deepcopy(request.get('controls', {})), readback=None)
            payload = {key: copy.deepcopy(value) for key, value in request.items()
                       if key not in ('action', 'request_id')}

            def configure():
                with self._lock:
                    record = self._laser_configuration
                    if record['request_id'] != token or record['cancel_requested']:
                        record['status'] = 'cancelled'
                        return
                try:
                    response = self.laser_command({'action': 'laser_configure', **payload})
                except Exception as error:
                    with self._lock:
                        record.update(status='cancelled' if record['cancel_requested'] else 'error',
                                      error=str(error))
                        if record['cancel_requested']:
                            return
                    raise
                with self._lock:
                    if isinstance(response, dict):
                        # The SDK reports the sensor's value after its write.
                        # Firmware quantization need not equal the requested
                        # float; native image controls must match this readback.
                        record['readback'] = copy.deepcopy(response.get('controls'))
                    record['status'] = 'cancelled' if record['cancel_requested'] else 'complete'

            self._start_operation('configuring', configure)
            return self.state(include_frame=False)

    def dispatch(self, request):
        """JSON entrypoint; camera off/state requests do not import vision."""
        action = request.get('action', request.get('command'))
        if action == 'camera_state':
            return self.state(include_frame=request.get('include_frame', True))
        if action in ('camera_laser_configure', 'camera_laser_cancel'):
            return self._dispatch_laser_configuration(
                {key: value for key, value in request.items() if not key.startswith('_')})
        if action == 'camera_analysis':
            mode = request.get('mode')
            if mode not in ('checkerboard', 'laser'):
                raise ValueError('Camera analysis mode must be checkerboard or laser')
            with self._lock:
                previous, self._analysis_mode = self._analysis_mode, mode
                actual_board = self._cached_state.get('board_detection_enabled')
                actual_mismatch = (type(actual_board) is bool
                    and actual_board != (mode == 'checkerboard'))
                if self._backend is None and self._operation is None:
                    return self.state()
                if self._operation is not None:
                    self._next_analysis = mode
                elif previous != mode or self._applied_analysis_mode != mode or actual_mismatch:
                    self._start_operation('configuring',
                        lambda: self._apply_analysis_mode(self._get_backend(), mode))
            return {**self.state(), 'status': 'queued'}
        if action == 'camera_stream':
            enabled = request.get('enabled')
            if type(enabled) is not bool:
                raise ValueError('Camera enabled must be boolean')
            if not enabled and self._backend is None:
                if self._operation is None:
                    return self.state()
            with self._lock:
                if self._operation is not None:
                    self._next_stream = enabled
                else:
                    self._start_operation('starting' if enabled else 'stopping',
                                          lambda: self._stream(enabled))
            return {**self.state(), 'status': 'queued'}
        if action == 'camera_timing':
            with self._lock:
                if self._operation is not None:
                    raise RuntimeError('Wait for the current camera operation before timing discovery')
                def discover():
                    backend = self._get_backend()
                    getter = getattr(backend, 'timing_capabilities', None)
                    if getter is None:
                        raise RuntimeError('Camera SDK does not expose hardware timing capabilities')
                    return getter()
                self._start_operation('configuring', discover)
            return {**self.state(), 'status': 'queued'}
        if action == 'camera_config':
            settings = request.get('settings', {key: request[key] for key in
                                   ('columns', 'rows', 'square_mm', 'views') if key in request})
            with self._lock:
                if self._operation is not None:
                    raise RuntimeError('Wait for camera startup or shutdown before changing board settings')
                self._start_operation('configuring', lambda: self._get_backend().configure(**settings))
            return {**self.state(), 'status': 'queued'}
        if action in ('calibration_capture', 'calibration_solve', 'calibration_reset'):
            if request.get('kind', 'intrinsic') != 'intrinsic':
                raise ValueError('Hand-eye commands require the calibration coordinator')
            backend = self._get_backend()
            if action == 'calibration_reset':
                return backend.reset()
            if action == 'calibration_capture':
                if 'not_before_monotonic' in request:
                    return backend.capture(request.get('metadata'),
                                           not_before_monotonic=request.get('not_before_monotonic'))
                return backend.capture(request.get('metadata'))
            return backend.solve()
        raise ValueError(f'Unknown camera command: {action}')

    def _bind_handeye_optical_identity(self):
        """Migrate a verified legacy archive before changing its full model hash.

        Original calibration_id and raw paired samples remain untouched. The
        optical snapshot proves what was compared; an arbitrary new full hash
        is never substituted into measured provenance.
        """
        path = self.workspace / 'handeye.json'
        if not path.exists():
            return
        archive = _read_json(path)
        detached, _ = self._prepare_handeye(archive)
        if detached.get('optical_intrinsics_id') is not None:
            return
        model = _read_json(self._intrinsic_path())
        full_id, _ = _intrinsic_identity(model, f'spinnaker:{self.serial}')
        if detached['calibration_id'] != full_id:
            raise ValueError('Cannot bind stale hand-eye data to new optical intrinsics')
        archive['optical_intrinsics_id'] = _optical_intrinsics_id(model)
        archive['optical_intrinsics'] = _optical_intrinsics_payload(model)
        archive['optical_identity_binding'] = dict(schema_version=1,
            verified_full_calibration_id=full_id, reason='verified_before_laser_plane_update')
        temporary = path.with_suffix('.optical.tmp')
        with temporary.open('w') as stream:
            stream.write(json.dumps(archive, indent=2, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck current optics before publishing the migration.
        current = _read_json(self._intrinsic_path())
        if _optical_intrinsics_id(current) != archive['optical_intrinsics_id']:
            raise ValueError('Optical intrinsics changed during hand-eye identity migration')
        temporary.replace(path)
        self.install_handeye(archive)

    def laser_command(self, request):
        """Blocking camera-only work; call from a calibration worker, never UI tick."""
        action = request.get('action')
        allowed = {'laser_new', 'laser_configure', 'laser_capture', 'laser_check',
                   'laser_solve', 'laser_cancel', 'laser_preview', 'pointcloud_configure'}
        if action not in allowed:
            raise ValueError('Unsupported laser camera action')
        if action in ('laser_new', 'laser_configure', 'laser_solve', 'pointcloud_configure'):
            self._bind_handeye_optical_identity()
        backend = self._get_backend()
        result = backend.laser_command(request)
        if action == 'laser_solve' and self._handeye_result is not None:
            # Verify the installed optics again after the plane save.
            self.install_handeye(self._handeye_result)
        return result

    @staticmethod
    def find_weld_seam(cloud, *, settings=None, previous_point_mm=None):
        """Analyze an immutable native acquisition through the camera-only SDK.

        The robot service owns acquisition/pose qualification and world transforms;
        this pure call neither creates a camera nor changes its controls.
        """
        return _weldfinder_module('seam').find_weld_seam(
            cloud, settings=settings, previous_point_mm=previous_point_mm)

    def target_pose(self, max_age_s=.5):
        return self._get_backend().target_pose(max_age_s)

    @property
    def fresh(self):
        return self.state(include_frame=False)['fresh']

    def rgb(self):
        return self._get_backend().rgb()

    def save(self, path):
        return self._get_backend().save(path)

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
        with self._instance_lock:
            if type(self)._instance is self:
                type(self)._instance = None


class Camera:
    """Legacy frame handle sharing the opt-in FLIR service without acquisition."""

    def __init__(self, device=None, *, workspace=None):
        self.service = CameraService.get_instance(workspace)

    @property
    def fresh(self):
        return self.service.fresh

    def rgb(self):
        return self.service.rgb()

    def save(self, path):
        return self.service.save(path)

    def close(self):
        self.service.close()
