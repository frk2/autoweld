"""Single-perspective laser scan geometry and durable, measured captures.

No robot, model or camera is constructed here. Weldfinder supplies native stripe
pixels and calibrated camera XYZ; the service supplies measured stationary poses.
All lengths are metres unless a field is explicitly suffixed ``_mm``.
"""

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoweld.calibration import transform


@dataclass(frozen=True)
class ScanSettings:
    width_mm: float = 10.
    height_mm: float = 10.
    step_mm: float = 1.
    standoff_mm: float = 140.
    depth_tolerance_mm: float = 1.
    max_depth_step_mm: float = 5.
    max_points: int = 200000

    def validate(self):
        values = (self.width_mm, self.height_mm, self.step_mm, self.standoff_mm,
                  self.depth_tolerance_mm, self.max_depth_step_mm)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError('Scan lengths must be finite numbers')
        if not np.isfinite(values).all() or not all(0 <= v <= 100 for v in values[:2]):
            raise ValueError('Horizontal and vertical scan extents must be between 0 and 100 mm')
        if self.step_mm != 1. or self.standoff_mm != 140.:
            raise ValueError('Scanning requires 1 mm capture intervals and 140 mm optical standoff')
        if self.depth_tolerance_mm != 1. or self.max_depth_step_mm != 5.:
            raise ValueError('Scanning requires 1 mm depth tolerance and at most 5 mm depth corrections')
        if any(not np.isclose(v, round(v)) for v in values[:2]):
            raise ValueError('Scan extents must be whole millimetres for uniform 1 mm spacing')
        if (round(self.width_mm)+1)*(round(self.height_mm)+1) > 10201:
            raise ValueError('Scan envelope exceeds the 10201 capture limit')
        if type(self.max_points) is not int or not 1000 <= self.max_points <= 300000:
            raise ValueError('Scan point budget must be an integer between 1000 and 300000')
        return self


def scan_offsets(settings=None):
    """Centered raster in the fixed baseline optical XY plane, 1 mm spacing.

    The first corner is a positioning move. Subsequent adjacent targets differ
    by exactly 1 mm horizontally or vertically; no camera rotation is introduced.
    """
    settings = (settings or ScanSettings()).validate()
    xs = (np.arange(round(settings.width_mm)+1)-settings.width_mm/2)/1000.
    ys = (np.arange(round(settings.height_mm)+1)-settings.height_mm/2)/1000.
    return [[float(x), float(y)] for row, y in enumerate(ys)
            for x in (xs if row % 2 == 0 else xs[::-1])]


def _points(value, dimensions, label):
    points = np.asarray(value, float)
    if points.ndim != 2 or points.shape[1] != dimensions or not np.isfinite(points).all():
        raise ValueError(f'{label} must be finite Nx{dimensions} values')
    return points


def segment_object(cloud, previous_centroid_camera_m=None):
    """Separate the illuminated object using native image and depth continuity.

    These are image-space connected stripe components with measured depth jumps,
    rather than an invented silhouette from a dark laser frame. Similar competing
    surfaces are rejected unless the previously tracked component resolves them.
    No rejected/native point is moved or synthesized.
    """
    if not isinstance(cloud, dict) or cloud.get('coordinate_frame') != 'camera':
        raise ValueError('Scanning requires a calibrated camera-frame stripe')
    if cloud.get('xyz_m_units', cloud.get('units')) != 'm':
        raise ValueError('Scanning XYZ requires explicit metre units')
    points = _points(cloud.get('xyz_m'), 3, 'Stripe XYZ')
    uv = _points(cloud.get('uv'), 2, 'Native stripe pixels')
    strength = np.asarray(cloud.get('strength'), float)
    widths = np.asarray(cloud.get('width_px'), float)
    count = len(points)
    if len(uv) != count or strength.shape != (count,) or widths.shape != (count,):
        raise ValueError('Native stripe pixels, strengths and widths must align with XYZ')
    size = np.asarray(cloud.get('image_size'), float)
    if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0)
            or np.any(uv < 0) or np.any(uv >= size)):
        raise ValueError('Native stripe pixels must lie inside the acquisition image size')
    if (count < 20 or not np.isfinite(strength).all() or not np.isfinite(widths).all()
            or np.any(points[:, 2] <= 0)):
        raise ValueError('A good decoded stripe needs at least 20 finite, positive-depth points')
    rejected = cloud.get('rejected') or {}
    candidates = max(count, int(rejected.get('candidates', count)))
    bad_fraction = (int(rejected.get('saturated', 0))+int(rejected.get('ambiguous', 0)))/candidates
    if bad_fraction > .5:
        raise ValueError('Laser stripe is dominated by saturated or ambiguous native returns')
    product = cloud.get('frame_product') or {}
    if product.get('kind') != 'single_frame':
        raise ValueError('Scanning requires a native single-frame stripe; multi-frame motion is unverified')
    usable = (strength >= 10.) & (widths >= .5) & (widths <= 40.)
    available = np.flatnonzero(usable)
    if len(available) < 20:
        raise ValueError('Laser stripe is too weak or too broad to track')
    axis = int(np.argmax(np.ptp(uv[available], axis=0)))
    ordered = available[np.argsort(uv[available, axis], kind='stable')]
    pixel_delta = np.linalg.norm(np.diff(uv[ordered], axis=0), axis=1)
    depth_delta = np.linalg.norm(np.diff(points[ordered], axis=0), axis=1)
    boundaries = np.flatnonzero((pixel_delta > 12.) | (depth_delta > .005))+1
    components = [indices for indices in np.split(ordered, boundaries) if len(indices) >= 20]
    if not components:
        raise ValueError('No continuous object stripe contains at least 20 good returns')
    components.sort(key=len, reverse=True)
    centers = [np.median(points[indices], axis=0) for indices in components]
    if previous_centroid_camera_m is None:
        if len(components) > 1 and len(components[1]) >= .67*len(components[0]):
            raise ValueError('Multiple disconnected object stripes are ambiguous at scan start')
        selected = 0
    else:
        previous = np.asarray(previous_centroid_camera_m, float)
        if previous.shape != (3,) or not np.isfinite(previous).all():
            raise ValueError('Previous object centroid must be three finite camera metres')
        distances = np.array([np.linalg.norm(center-previous) for center in centers])
        selected = int(np.argmin(distances))
        if distances[selected] > .020:
            raise ValueError('Tracked object stripe was lost; refusing to switch surfaces')
        if len(distances) > 1 and np.partition(distances, 1)[1]-distances[selected] < .003:
            raise ValueError('Tracked stripe association is ambiguous between competing surfaces')
    indices, center = components[selected], centers[selected]
    selected_points = points[indices]
    if np.linalg.norm(np.ptp(selected_points, axis=0)) < .002:
        raise ValueError('Object stripe spans less than 2 mm; insufficient geometric support')
    return {'xyz_m': selected_points.tolist(), 'indices': indices.tolist(),
            'centroid_camera_m': center.tolist(),
            'bounds_camera_m': [selected_points.min(axis=0).tolist(), selected_points.max(axis=0).tolist()],
            'uv_bounds': [uv[indices].min(axis=0).tolist(), uv[indices].max(axis=0).tolist()],
            'quality': {'good': True, 'count': len(indices), 'input_count': count,
                        'components': len(components), 'segmentation': 'native_uv_and_xyz_connectivity',
                        'median_strength': float(np.median(strength[indices])),
                        'median_width_px': float(np.median(widths[indices])),
                        'saturated_or_ambiguous_fraction': float(bad_fraction),
                        'depth_m': float(center[2]), 'median_depth_mm': float(center[2]*1000.),
                        'standoff_mm': float(center[2]*1000.),
                        'depth_error_mm': float(center[2]*1000.-140.)}}


def tracking_target(base_from_tcp, tcp_from_camera, segment, offset_xy_m=(0., 0.), *, settings=None):
    """Translate a measured nominal TCP in optical coordinates; preserve rotation.

    XY is an incremental fixed-perspective raster move. Optical Z corrects the
    measured object depth by at most 5 mm per discrete move. The caller must
    remeasure after settling and must not capture until inside depth tolerance.
    """
    settings = (settings or ScanSettings()).validate()
    pose = transform(base_from_tcp)
    optical = pose @ transform(tcp_from_camera)
    offset = np.asarray(offset_xy_m, float)
    if offset.shape != (2,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .001000001:
        raise ValueError('Optical XY tracking moves must be finite and at most 1 mm')
    center = np.asarray(segment.get('centroid_camera_m'), float)
    if center.shape != (3,) or not np.isfinite(center).all() or not segment.get('quality', {}).get('good'):
        raise ValueError('Tracking requires a good segmented object stripe')
    correction = float(np.clip(center[2]-settings.standoff_mm/1000.,
                               -settings.max_depth_step_mm/1000., settings.max_depth_step_mm/1000.))
    pose[:3, 3] += optical[:3, :3] @ np.array([*offset, correction])
    return pose


class ScanArchive:
    """Append-only native views and full exports, with a bounded display sample.

    ``max_points`` is a legacy preview budget, never a measurement limit. At
    most 10,000 preview points enter service telemetry. Every native view and
    measured transform remains on disk, including explicit empty/rejected cells.
    """

    PREVIEW_LIMIT = 10000
    CHECKPOINT_INTERVAL = 32
    MAX_CELLS = 10201

    def __init__(self, runtime, max_points=200000):
        if type(max_points) is not int or not 1000 <= max_points <= 300000:
            raise ValueError('Scan archive preview budget must be between 1000 and 300000')
        self.directory = Path(runtime)/'laser-scans'
        self.max_points = min(max_points, self.PREVIEW_LIMIT)
        self._preview = np.empty((self.max_points, 3), float)
        self._preview_size = 0
        self._preview_list = []
        self._preview_revision = 0
        self._identity, self._acquisitions, self._cells = None, set(), set()
        self._rng = np.random.default_rng(0)
        self._legacy_captures = None
        self._metadata = self._empty_metadata()
        self._journal_valid_bytes = None
        self._restore_error = None
        self._restore()

    @staticmethod
    def _empty_metadata():
        return {'schema_version': 2, 'status': 'idle', 'message': 'No scan captured',
                'capture_count': 0, 'count': 0, 'cells_visited': 0,
                'cells_no_return': 0, 'cells_rejected': 0, 'next_index': 0}

    def _session_directory(self):
        session = self._metadata.get('session_id')
        if not session or Path(session).name != session:
            raise ValueError('Invalid scan archive session identity')
        return self.directory/session

    def _journal_path(self):
        return self._session_directory()/'cells.jsonl'

    def _iter_records(self):
        if self._legacy_captures is not None:
            for index, record in enumerate(self._legacy_captures):
                yield {**record, 'cell_index': index, 'status': 'captured'}
            return
        path = self._journal_path()
        if not path.exists():
            return
        with path.open('rb') as handle:
            valid_bytes = 0
            while line := handle.readline():
                if not line.endswith(b'\n'):
                    break  # A torn last append cannot qualify as a visited cell.
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('Invalid scan cell journal record')
                valid_bytes = handle.tell()
                yield record
            self._journal_valid_bytes = valid_bytes

    def _load_world(self, record):
        with np.load(record['raw_snapshot_path'], allow_pickle=False) as native:
            if 'xyz_world_m' in native:
                return _points(native['xyz_world_m'], 3, 'Saved world stripe')
            points = _points(native['xyz_camera_m'], 3, 'Saved native stripe')
            selected = native['selected_indices']
            optical = transform(native['base_from_tcp']) @ transform(native['tcp_from_camera'])
            return points[selected] @ optical[:3, :3].T+optical[:3, 3]

    def _add_preview(self, world, previous_count):
        # Reservoir work is O(new points), independent of accumulated raw data.
        available = min(len(world), self.max_points-self._preview_size)
        if available:
            self._preview[self._preview_size:self._preview_size+available] = world[:available]
            self._preview_size += available
        tail = world[available:]
        if len(tail):
            high = previous_count+available+np.arange(len(tail))+1
            positions = self._rng.integers(0, high)
            # Ordered replacement is important when a batch hits one slot twice.
            for point, index in zip(tail, positions):
                if index < self.max_points:
                    self._preview[index] = point

    def _publish_preview(self):
        self._preview_list = self._preview[:self._preview_size].tolist()
        self._preview_revision = self._metadata['cells_visited']

    def _observe_record(self, record, *, replay_preview=False):
        index = record['cell_index']
        if type(index) is not int or not 0 <= index < self.MAX_CELLS or index in self._cells:
            raise ValueError('Scan cell journal has an invalid or duplicate grid index')
        acquisition = tuple(record['acquisition_identity'])
        if acquisition in self._acquisitions:
            raise ValueError('Scan cell journal repeats a native acquisition')
        outcome = record['status']
        if outcome not in ('captured', 'no_return', 'rejected'):
            raise ValueError('Scan cell journal has an invalid outcome')
        if replay_preview and outcome == 'captured':
            self._add_preview(self._load_world(record), self._metadata['count'])
        self._cells.add(index)
        self._acquisitions.add(acquisition)
        self._metadata['cells_visited'] += 1
        self._metadata['count'] += record['count']
        if outcome == 'captured':
            self._metadata['capture_count'] += 1
        else:
            self._metadata['cells_'+outcome] += 1
        while self._metadata['next_index'] in self._cells:
            self._metadata['next_index'] += 1
        self._metadata['last_cell'] = {key: copy.deepcopy(record.get(key)) for key in
                                      ('cell_index', 'offset_xy_m', 'status', 'reason', 'base_from_tcp')}

    def _restore(self):
        path = self.directory/'latest.json'
        if not path.exists():
            return
        try:
            metadata = json.loads(path.read_text())
            legacy = metadata.get('schema_version', 1) < 2
            old_captures = metadata.pop('captures', [])
            metadata.pop('cells', None)
            self._metadata = {**self._empty_metadata(), **metadata}
            self._legacy_captures = old_captures if legacy else None
            checkpoint_cells = int(metadata.get('preview_checkpoint_cells', 0))
            filename = metadata.get('points_file')
            if filename:
                with np.load(self.directory/filename, allow_pickle=False) as data:
                    preview = _points(data['points_world_m'], 3, 'Saved scan preview')
                if legacy:
                    if len(preview) != metadata.get('count', len(preview)):
                        raise ValueError('Legacy scan point count does not match its saved cloud')
                    self._add_preview(preview, 0)
                    checkpoint_cells = len(old_captures)
                else:
                    expected = metadata.get('preview_checkpoint_count', len(preview))
                    if len(preview) != expected:
                        raise ValueError('Saved scan preview count does not match its checkpoint')
                    self._preview_size = min(len(preview), self.max_points)
                    self._preview[:self._preview_size] = preview[:self._preview_size]
                    if metadata.get('preview_rng_state'):
                        self._rng.bit_generator.state = metadata['preview_rng_state']
            for key in ('capture_count', 'count', 'cells_visited', 'cells_no_return', 'cells_rejected', 'next_index'):
                self._metadata[key] = 0
            last_identity = None
            for ordinal, record in enumerate(self._iter_records()):
                self._observe_record(record, replay_preview=ordinal >= checkpoint_cells)
                current_identity = tuple(record.get(key) for key in
                    ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id'))
                if last_identity is not None and current_identity[:3] != last_identity[:3]:
                    raise ValueError('Scan journal changes camera or calibration identity')
                last_identity = current_identity
            identity = metadata.get('identity') or last_identity
            self._identity = None if identity is None else tuple(identity)
            if self._metadata.get('status') == 'scanning':
                self._metadata.update(status='interrupted', message='Service restarted; saved partial scan retained')
            self._publish_preview()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._restore_error = str(error)
            self._metadata.update(status='error', message=f'Saved scan could not be restored: {error}')

    @staticmethod
    def _capture_orientation(metadata, baseline):
        """Validate the fixed capture TCP orientation without rotating the grid."""
        rotation = metadata.get('capture_rotation')
        if rotation is None:
            return baseline[:3, :3]
        rotation = np.asarray(rotation, float)
        if rotation.shape != (3, 3):
            raise ValueError('Scan capture_rotation must be a finite rigid 3x3 rotation')
        pose = np.eye(4)
        pose[:3, :3] = rotation
        return transform(pose)[:3, :3]

    def begin(self, metadata=None):
        metadata = metadata or {}
        if metadata.get('capture_rotation') is not None:
            self._capture_orientation(metadata, np.eye(4))
        self._preview_size, self._preview_list, self._preview_revision = 0, [], 0
        self._identity, self._acquisitions, self._cells = None, set(), set()
        self._legacy_captures = None
        self._journal_valid_bytes = None
        self._restore_error = None
        session = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        self._rng = np.random.default_rng(int(session[-8:], 16))
        self._metadata = {**self._empty_metadata(), 'status': 'scanning',
                          'message': 'Capturing stable object stripes', 'session_id': session,
                          'started_at': time.time(), 'units': 'm', 'coordinate_frame': 'robot_base',
                          'preview_limit': self.max_points, 'metadata': copy.deepcopy(metadata or {})}
        self._session_directory().mkdir(parents=True, exist_ok=True)
        self._journal_path().touch()
        self._checkpoint()
        self._save()
        return self.state()

    def _association(self, cloud, *, now, not_before_monotonic, feedback):
        if self._metadata['status'] != 'scanning':
            raise ValueError('Begin or resume a scan before recording a cell')
        stamp = cloud.get('read_started')
        feedback = feedback or {}
        measured_at, stable_since = feedback.get('measured_at'), feedback.get('stable_since')
        stamps = (now, stamp, not_before_monotonic, measured_at, stable_since)
        if any(v is None or isinstance(v, bool) for v in stamps) or not np.isfinite(stamps).all():
            raise ValueError('Scan capture requires finite acquisition and SDK stationary timestamps')
        if not 0 <= now-stamp <= 1. or stamp < max(not_before_monotonic, stable_since):
            raise ValueError('Scan acquisition is stale or predates the stationary capture interval')
        if (feedback.get('fresh') is not True or feedback.get('stationary') is not True
                or not 0 <= now-measured_at <= .2 or measured_at < stamp):
            raise ValueError('Scan capture requires fresh stationary SDK feedback after acquisition')
        identity = (cloud.get('source_id'), cloud.get('optical_intrinsics_id', cloud.get('calibration_id')),
                    cloud.get('calibration_id'), cloud.get('acquisition_session_id'))
        if any(v is None or v == '' for v in identity):
            raise ValueError('Scan capture requires camera, optical, laser and acquisition session identity')
        if self._identity is not None and identity != self._identity:
            raise ValueError('Camera or calibration identity changed during scanning')
        acquisition = (identity[0], identity[3], cloud.get('generation'), cloud.get('sequence'))
        if any(v is None for v in acquisition) or acquisition in self._acquisitions:
            raise ValueError('Scan capture must have a unique native acquisition identity')
        if cloud.get('frame_product', {}).get('kind') != 'single_frame':
            raise ValueError('Scan captures require native single-frame products')
        if cloud.get('coordinate_frame') != 'camera' or cloud.get('xyz_m_units', cloud.get('units')) != 'm':
            raise ValueError('Scan captures require explicitly calibrated camera XYZ in metres')
        return identity, acquisition

    @staticmethod
    def _native_array(value, dimensions, label):
        array = np.asarray(value, float)
        if array.size == 0:
            return np.empty((0, dimensions))
        return _points(array, dimensions, label)

    def _cell(self, index, offset):
        index = self._metadata['next_index'] if index is None else index
        if type(index) is not int or not 0 <= index < self.MAX_CELLS or index in self._cells:
            raise ValueError('Scan requires a unique grid cell index between 0 and 10200')
        settings = (self._metadata.get('metadata') or {}).get('settings') or {}
        if 'width_mm' in settings and 'height_mm' in settings:
            width, height = settings['width_mm'], settings['height_mm']
            total = (round(width)+1)*(round(height)+1)
            if index >= total:
                raise ValueError('Scan grid cell index exceeds the configured raster')
            if offset is not None:
                columns = round(width)+1
                row, column = divmod(index, columns)
                if row % 2:
                    column = columns-1-column
                expected = np.array([column-width/2, row-height/2])/1000.
                if not np.allclose(offset, expected, atol=1e-9, rtol=0):
                    raise ValueError('Scan grid cell offset does not match its nominal raster index')
        if offset is not None:
            offset = np.asarray(offset, float)
            if offset.shape != (2,) or not np.isfinite(offset).all():
                raise ValueError('Scan grid cell offset must be finite optical XY metres')
            offset = offset.tolist()
        return index, offset

    def append(self, cloud, base_from_tcp, tcp_from_camera, segment, *, now,
               not_before_monotonic, feedback=None, cell_index=None, offset_xy_m=None):
        identity, acquisition = self._association(cloud, now=now,
            not_before_monotonic=not_before_monotonic, feedback=feedback)
        native = self._native_array(cloud.get('xyz_m'), 3, 'Native scan XYZ')
        indices = np.asarray(segment.get('indices'), int)
        if (indices.ndim != 1 or len(indices) < 20 or np.any(indices < 0) or np.any(indices >= len(native))
                or len(np.unique(indices)) != len(indices) or not segment.get('quality', {}).get('good')):
            raise ValueError('Scan capture requires at least 20 unique segmented native stripe points')
        points = native[indices]
        if abs(float(np.median(points[:, 2]))-.140) > .001000001:
            raise ValueError('Scan capture must be within 1 mm of the 140 mm optical standoff')
        return self._record('captured', cloud, native, indices, base_from_tcp, tcp_from_camera,
                            identity, acquisition, feedback, cell_index, offset_xy_m,
                            quality=segment['quality'])

    def record_cell(self, outcome, cloud, base_from_tcp, tcp_from_camera, *, now,
                    not_before_monotonic, feedback=None, cell_index=None, offset_xy_m=None, reason=''):
        if outcome not in ('no_return', 'rejected'):
            raise ValueError('Non-capture cell outcome must be no_return or rejected')
        if not isinstance(reason, str) or len(reason) > 1000:
            raise ValueError('Scan cell reason must be a string of at most 1000 characters')
        identity, acquisition = self._association(cloud, now=now,
            not_before_monotonic=not_before_monotonic, feedback=feedback)
        native = self._native_array(cloud.get('xyz_m'), 3, 'Native scan XYZ')
        if outcome == 'no_return' and len(native):
            raise ValueError('A no-return cell requires an empty calibrated native cloud')
        return self._record(outcome, cloud, native, np.empty(0, int), base_from_tcp, tcp_from_camera,
                            identity, acquisition, feedback, cell_index, offset_xy_m, reason=reason)

    def _record(self, outcome, cloud, native, indices, base_from_tcp, tcp_from_camera,
                identity, acquisition, feedback, cell_index, offset_xy_m, *, quality=None, reason=''):
        index, offset = self._cell(cell_index, offset_xy_m)
        pose, correction = transform(base_from_tcp), transform(tcp_from_camera)
        optical = pose @ correction
        baseline = (self._metadata.get('metadata') or {}).get('baseline_pose')
        saved_handeye = (self._metadata.get('metadata') or {}).get('handeye') or {}
        if saved_handeye.get('tcp_from_camera') is not None and not np.allclose(
                correction, transform(saved_handeye['tcp_from_camera']), atol=1e-10, rtol=0):
            raise ValueError('Scan measured hand-eye transform changed')
        if baseline is not None and offset is not None:
            baseline = transform(baseline)
            baseline_optical = baseline @ correction
            measured_xy = baseline_optical[:3, :3].T @ (optical[:3, 3]-baseline_optical[:3, 3])
            # Camera roll selects one fixed viewing orientation. Grid XY still
            # uses the original Tube optical frame above, never the rolled one.
            capture_rotation = self._capture_orientation(self._metadata.get('metadata') or {}, baseline)
            angle = float(np.arccos(np.clip((np.trace(pose[:3, :3] @ capture_rotation.T)-1)/2, -1, 1)))
            if np.linalg.norm(measured_xy[:2]-offset) > .000500001 or angle > .005000001:
                raise ValueError('Scan cell must match its measured grid endpoint within 0.5 mm / 0.005 rad')
        world = native[indices] @ optical[:3, :3].T+optical[:3, 3]
        capture = {key: copy.deepcopy(cloud.get(key)) for key in
                   ('source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id',
                    'sequence', 'generation', 'read_started', 'read_completed', 'frame_product',
                    'image_size', 'controls', 'timing', 'rejected', 'received')}
        capture.update(cell_index=index, offset_xy_m=offset, status=outcome, reason=reason,
                       acquisition_identity=list(acquisition), base_from_tcp=pose.tolist(),
                       tcp_from_camera=correction.tolist(), feedback=copy.deepcopy(feedback),
                       quality=copy.deepcopy(quality), count=len(world), point_offset=self._metadata['count'])
        raw_path = self._session_directory()/f'{index:05d}.npz'
        raw_temp = raw_path.with_suffix('.npz.tmp')
        uv = self._native_array(cloud.get('uv'), 2, 'Native stripe pixels')
        strength, widths = np.asarray(cloud.get('strength'), float), np.asarray(cloud.get('width_px'), float)
        if len(uv) != len(native) or strength.shape != (len(native),) or widths.shape != (len(native),):
            raise ValueError('Native scan attributes must align with camera XYZ')
        if not np.isfinite(strength).all() or not np.isfinite(widths).all():
            raise ValueError('Native scan attributes must be finite')
        with raw_temp.open('wb') as handle:
            np.savez_compressed(handle, xyz_camera_m=native, xyz_world_m=world, uv=uv,
                                strength=strength, width_px=widths, selected_indices=indices,
                                base_from_tcp=pose, tcp_from_camera=correction)
        os.replace(raw_temp, raw_path)
        capture['raw_snapshot_path'] = str(raw_path)
        encoded = json.dumps(capture, allow_nan=False, separators=(',', ':'))+'\n'
        with self._journal_path().open('a') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            self._journal_valid_bytes = handle.tell()
        self._add_preview(world, self._metadata['count'])
        self._observe_record(capture)
        self._identity = identity
        self._metadata['identity'] = list(identity)
        visited = self._metadata['cells_visited']
        if visited <= 16 or visited % self.CHECKPOINT_INTERVAL == 0:
            self._publish_preview()
        if visited % self.CHECKPOINT_INTERVAL == 0:
            self._checkpoint()
        self._save()
        return capture

    def _checkpoint(self):
        filename = f"{self._metadata.get('session_id', 'empty')}-preview-{self._metadata['cells_visited']}.npz"
        path, previous = self.directory/filename, self._metadata.get('points_file')
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.npz.tmp')
        with temp.open('wb') as handle:
            np.savez_compressed(handle, points_world_m=self._preview[:self._preview_size])
        os.replace(temp, path)
        self._metadata.update(points_file=filename, preview_checkpoint_cells=self._metadata['cells_visited'],
                              preview_checkpoint_count=self._preview_size,
                              preview_rng_state=copy.deepcopy(self._rng.bit_generator.state))
        # Publish reference before deleting the previous current-session checkpoint.
        self._save()
        if previous and previous != filename and previous.startswith(self._metadata.get('session_id', 'empty')+'-preview-'):
            (self.directory/previous).unlink(missing_ok=True)

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {**self._metadata}
        # Historical small-scan readers can still inspect captures in latest.json.
        # Large scans always use the journal/final manifest, never O(N²) rewrites.
        if self._metadata['status'] in ('complete', 'cancelled', 'error', 'interrupted') and self._metadata['cells_visited'] <= 64:
            payload['captures'] = [r for r in self._iter_records() if r['status'] == 'captured']
        path = self.directory/'latest.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')))
        os.replace(temp, path)

    def _migrate_legacy(self):
        if self._legacy_captures is None:
            return
        records = list(self._iter_records())
        self._session_directory().mkdir(exist_ok=True)
        path = self._journal_path()
        with path.open('w') as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        self._legacy_captures = None
        self._metadata['schema_version'] = 2
        self._checkpoint()

    def resume(self, metadata=None, *, acquisition_session_id=None):
        if self._restore_error is not None:
            raise ValueError(f'Saved scan archive requires repair before resume: {self._restore_error}')
        status = self._metadata['status']
        if status == 'scanning' and metadata is None and acquisition_session_id is None:
            return self.state()
        if status not in ('scanning', 'interrupted', 'cancelled', 'error') or not self._metadata.get('session_id'):
            raise ValueError('Only an interrupted, cancelled or failed saved scan can resume')
        original = self._metadata.get('metadata') or {}
        for key, value in (metadata or {}).items():
            previous = original.get(key)
            matches = value == previous
            if key == 'identity' and isinstance(value, (list, tuple)) and isinstance(previous, (list, tuple)):
                matches = list(value[:3]) == list(previous[:3])
            if not matches:
                raise ValueError(f'Resumed scan metadata changed: {key}')
        settings = original.get('settings') or {}
        if 'width_mm' in settings and 'height_mm' in settings:
            total = (round(settings['width_mm'])+1)*(round(settings['height_mm'])+1)
            if self._metadata['next_index'] >= total:
                raise ValueError('Saved scan already visited every configured grid cell')
        self._migrate_legacy()
        if self._journal_valid_bytes is not None:
            with self._journal_path().open('r+b') as handle:
                handle.truncate(self._journal_valid_bytes)
            self._journal_valid_bytes = None
        if acquisition_session_id is not None:
            if not isinstance(acquisition_session_id, str) or not acquisition_session_id:
                raise ValueError('Resumed camera acquisition session must be a nonempty string')
            if self._identity is not None:
                self._identity = (*self._identity[:3], acquisition_session_id)
                self._metadata['identity'] = list(self._identity)
        self._metadata.update(status='scanning', message='Resuming saved grid coverage',
                              resumed_at=time.time(), resume_count=self._metadata.get('resume_count', 0)+1)
        self._metadata.pop('finished_at', None)
        self._save()
        return self.state()

    def finish(self, status='complete', message=''):
        if status not in ('complete', 'cancelled', 'error', 'interrupted'):
            raise ValueError('Invalid terminal scan status')
        if self._metadata.get('session_id'):
            settings = (self._metadata.get('metadata') or {}).get('settings') or {}
            if status == 'complete' and 'width_mm' in settings and 'height_mm' in settings:
                total = (round(settings['width_mm'])+1)*(round(settings['height_mm'])+1)
                if self._metadata['cells_visited'] != total or self._metadata['next_index'] != total:
                    raise ValueError('Complete scan requires every configured grid cell to be visited')
            self._migrate_legacy()
            self._publish_preview()
            self._checkpoint()
            ply = self.directory/(self._metadata['session_id']+'.ply')
            temporary = ply.with_suffix('.ply.tmp')
            with temporary.open('w') as handle:
                handle.write('ply\nformat ascii 1.0\ncomment units meters; frame robot_base\n'
                             f"element vertex {self._metadata['count']}\n"
                             'property float x\nproperty float y\nproperty float z\nend_header\n')
                for record in self._iter_records():
                    if record['status'] == 'captured':
                        np.savetxt(handle, self._load_world(record), fmt='%.9f')
            os.replace(temporary, ply)
            self._metadata.update(ply_path=str(ply), status=status,
                                  message=message or f'Scan {status}', finished_at=time.time())
            self._save()
            manifest = self.directory/(self._metadata['session_id']+'.json')
            temp = manifest.with_suffix('.json.tmp')
            with temp.open('w') as handle:
                payload = json.dumps(self._metadata, allow_nan=False, separators=(',', ':'))
                handle.write(payload[:-1]+',"captures":[')
                separator = ''
                for record in self._iter_records():
                    if record['status'] == 'captured':
                        handle.write(separator+json.dumps(record, allow_nan=False, separators=(',', ':')))
                        separator = ','
                handle.write('],"cells":[')
                separator = ''
                for record in self._iter_records():
                    handle.write(separator+json.dumps(record, allow_nan=False, separators=(',', ':')))
                    separator = ','
                handle.write(']}')
            os.replace(temp, manifest)
        return self.state()

    def state(self, include_points=False):
        result = {key: copy.deepcopy(value) for key, value in self._metadata.items()
                  if key not in ('captures', 'cells', 'preview_rng_state')}
        session = self._metadata.get('session_id')
        settings = (self._metadata.get('metadata') or {}).get('settings') or {}
        grid_total = ((round(settings['width_mm'])+1)*(round(settings['height_mm'])+1)
                      if 'width_mm' in settings and 'height_mm' in settings else None)
        result.update(captures=self._metadata['capture_count'], points=self._metadata['count'],
                      grid_total=grid_total, points_are_preview=True,
                      coordinate_frame='robot_base', units='m', preview_count=len(self._preview_list),
                      metadata_path=str(self.directory/'latest.json'), scan_id=session,
                      cloud_revision=self._preview_revision,
                      export_paths={'metadata': str(self.directory/'latest.json'),
                                    'points': str(self.directory/self._metadata.get('points_file', 'empty.npz')),
                                    'ply': self._metadata.get('ply_path'),
                                    'manifest': str(self.directory/(session+'.json')) if session else None,
                                    'journal': str(self.directory/session/'cells.jsonl') if session else None,
                                    'raw_snapshots': str(self.directory/session) if session else None})
        if include_points:
            result['points_world_m'] = [point[:] for point in self._preview_list]
            result['points_base_m'] = result['points_world_m']
        return result

    def clear(self):
        self._preview_size, self._preview_list, self._preview_revision = 0, [], 0
        self._identity, self._acquisitions, self._cells = None, set(), set()
        self._legacy_captures = None
        self._journal_valid_bytes = None
        self._restore_error = None
        self._metadata = {**self._empty_metadata(), 'message': 'Scan display cleared'}
        self._save()
        return self.state()
