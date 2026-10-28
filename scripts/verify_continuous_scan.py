#!/usr/bin/env python3
"""Independently audit a moving scan using files only; no SDK/model/camera imports.

Streams every journal record, one bounded NPZ view, and matching binary PLY
chunks. Native XYZ-to-world transforms are independently sampled; all journal
identities/timing/poses, all selected NPZ counts/ROI points, and every exported
PLY coordinate are checked. Rounded camera-origin grid coverage is distinct
from object-point occupancy and never filled from planned poses.
"""
import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_POINTS = 20000
MAX_LINE = 2*1024*1024


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def array(value, shape, label):
    result = np.asarray(value, float)
    require(result.shape == shape and np.isfinite(result).all(), f'{label}: invalid/nonfinite shape')
    return result


def pose(value, label):
    result = array(value,(4,4),label)
    require(np.allclose(result[3],[0,0,0,1],rtol=0,atol=1e-9)
        and np.allclose(result[:3,:3].T@result[:3,:3],np.eye(3),rtol=0,atol=1e-6)
        and abs(np.linalg.det(result[:3,:3])-1.) <= 1e-6, f'{label}: not a rigid transform')
    return result


def number(value, label):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),f'{label}: finite number required')
    return float(value)


def records(path):
    with path.open('rb') as handle:
        line_number = 0
        while True:
            line = handle.readline(MAX_LINE+1)
            if not line:
                break
            line_number += 1
            require(len(line) <= MAX_LINE and line.endswith(b'\n'),f'Journal line{line_number}: oversized or uncommitted tail')
            record = json.loads(line)
            require(isinstance(record,dict),f'Journal line{line_number}: object required')
            yield line_number,record


class BinaryPLY:
    def __init__(self,path):
        self.path = path
        self.handle = path.open('rb')
        self.header_bytes = 0
        header = []
        try:
            while self.header_bytes < 65536:
                line = self.handle.readline(4097)
                self.header_bytes += len(line)
                require(line and len(line) <= 4096,f'{path.name}: malformed header')
                text = line.decode('ascii').strip()
                header.append(text)
                if text == 'end_header':
                    break
            require(header[0] == 'ply' and 'format binary_little_endian 1.0' in header,f'{path.name}: binary little-endian PLY required')
            elements = [line for line in header if line.startswith('element ')]
            require(len(elements) == 1 and elements[0].startswith('element vertex '),f'{path.name}: only vertex elements supported')
            self.count = int(elements[0].split()[-1])
            require(0 <= self.count <= 10**9,f'{path.name}: invalid vertex count')
            require([line for line in header if line.startswith('property ')] ==
                ['property double x','property double y','property double z'],f'{path.name}: exact double XYZ properties required')
            require(header[-1] == 'end_header',f'{path.name}: unterminated header')
            self.byte_length = path.stat().st_size
            require(self.byte_length == self.header_bytes+self.count*24,f'{path.name}: header/count/byte length disagree')
            self.read_count = 0
        except Exception:
            self.handle.close()
            raise

    def read(self,count):
        require(0 <= count <= MAX_POINTS,f'{self.path.name}: unbounded per-view read')
        data = self.handle.read(count*24)
        require(len(data) == count*24,f'{self.path.name}: truncated vertex payload')
        points = np.frombuffer(data,dtype='<f8').reshape(-1,3)
        require(np.isfinite(points).all(),f'{self.path.name}: nonfinite XYZ')
        self.read_count += count
        return points

    def finish(self):
        require(self.read_count == self.count and not self.handle.read(1),f'{self.path.name}: exported counts differ from native views')
        return dict(format='binary_little_endian',point_dtype='<f8',vertex_count=self.count,
            byte_length=self.byte_length,header_bytes=self.header_bytes,all_coordinates_match_native=True)

    def close(self):
        self.handle.close()


class Coverage:
    def __init__(self,baseline):
        self.baseline = baseline
        self.grids = {key:np.zeros((101,101),bool) for key in ('accepted','no_return','rejected','object_points')}

    def local(self,points):
        return (points-self.baseline[:3,3])@self.baseline[:3,:3]

    def mark(self,points,key):
        xy = self.local(np.asarray(points,float).reshape(-1,3))[:,:2]
        within = np.all(abs(xy) <= .0505+1e-12,axis=1)
        bins = np.rint((xy[within]+.05)/.001).astype(int)
        bins = bins[np.all((bins >= 0)&(bins <= 100),axis=1)]
        if len(bins):
            self.grids[key][bins[:,1],bins[:,0]] = True

    def roi(self,points):
        return np.all(abs(self.local(points)[:,:2]) <= .05+1e-12,axis=1)

    def state(self):
        good = self.grids['accepted']|self.grids['no_return']
        rows = []
        for index,row in enumerate(good):
            missing = np.flatnonzero(~row)
            if not len(missing):
                continue
            runs = np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
            rows.append(dict(y_mm=index-50,observed_x_bins=int(row.sum()),
                missing_x_ranges_mm=[[int(run[0])-50,int(run[-1])-50] for run in runs]))
        return dict(definition='nearest1mm camera-origin target, per-axis tolerance0.5mm, original measured optical XY',
            required_rows=101,required_x_bins_per_row=101,required_bins=10201,
            good_camera_bins=int(good.sum()),full_rows=int(np.all(good,axis=1).sum()),
            all_grid_positions_observed=bool(good.all()),
            accepted_camera_bins=int(self.grids['accepted'].sum()),
            no_return_camera_bins=int(self.grids['no_return'].sum()),
            rejected_camera_bins=int(self.grids['rejected'].sum()),
            object_point_xy_bins=int(self.grids['object_points'].sum()),incomplete_rows=rows)


def bounded_npz(path):
    require(path.is_file(),f'Missing raw native view{path.name}')
    with zipfile.ZipFile(path) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= 16*1024*1024,f'{path.name}: native view exceeds bounded size')
        shapes = {'xyz_world_m':(None,3),'xyz_camera_m':(None,3),'selected_indices':(None,),
            'joint_pos':(6,),'base_from_tcp':(4,4),'tcp_from_camera':(4,4),'provenance_json':()}
        for name,expected in shapes.items():
            with archive.open(name+'.npy') as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1,0):
                    shape,_,dtype = np.lib.format.read_array_header_1_0(handle)
                else:
                    require(version == (2,0),f'{path.name}: unsupported NumPy header')
                    shape,_,dtype = np.lib.format.read_array_header_2_0(handle)
                valid_shape = len(shape) == len(expected) and all(
                    (0 <= actual <= MAX_POINTS) if required is None else actual == required
                    for actual,required in zip(shape,expected))
                if name == 'selected_indices':
                    valid_dtype = dtype.kind in 'iu' and dtype.itemsize <= 8
                elif name == 'provenance_json':
                    valid_dtype = dtype.kind in 'US' and dtype.itemsize <= MAX_LINE
                else:
                    valid_dtype = dtype.kind == 'f' and dtype.itemsize == 8
                require(valid_shape and valid_dtype and not dtype.hasobject,
                    f'{path.name}: unbounded/invalid native array{name}')
    return np.load(path,allow_pickle=False)


def validate_frame(record,identity,clock_id,history_id,j6_bounds,reference_tier=False):
    cloud,reg = record['cloud'],record['registration']
    actual = [cloud.get(key) for key in ('source_id','optical_intrinsics_id','calibration_id','acquisition_session_id')]
    require(actual == identity,'Source/optical/laser/acquisition identity changed')
    for key in ('source_id','optical_intrinsics_id','acquisition_session_id','sequence'):
        require(reg.get(key) == cloud.get(key),f'Registrar/native{key} mismatch')
    require(cloud.get('coordinate_frame') == 'camera' and cloud.get('xyz_m_units',cloud.get('units')) == 'm'
        and (cloud.get('frame_product') or {}).get('kind') == 'single_frame','Native product/frame/XYZ units invalid')
    timing = cloud['timing']
    device = timing.get('device_frame_id')
    require(type(device) is int and 0 <= device < 2**64 and device == timing.get('chunk_frame_id')
        and device == timing.get('exposure_end_frame_id'),'Physical device/chunk/exposure-event IDs disagree')
    require(record.get('acquisition_identity') == [*identity,device],'Journal acquisition key differs from physical identity')
    require(type(cloud.get('sequence')) is int and cloud['sequence'] >= 0
        and type(cloud.get('generation')) is int and cloud['generation'] >= 0,'Invalid native generation/sequence')
    require(timing.get('motion_pairing_supported') is True and timing.get('sensor_shutter_mode') == 'global'
        and timing.get('host_clock_domain') == 'host_monotonic','Native exposure clock/shutter not qualified')
    mapping = timing['clock_mapping']
    require(mapping.get('status') == 'qualified' and mapping.get('mapping_id') == clock_id
        and reg.get('clock_mapping_id') == clock_id,'Clock mapping identity changed/unqualified')
    require(reg.get('history_id') == history_id and reg.get('no_extrapolation') is True,
        'Measured output history identity or no-extrapolation proof differs')
    if reference_tier:
        require(reg.get('registration_reference') == 'measured_kernel_rx'
            and reg.get('registration_method') == 'measured_output_receive_reference'
            and reg.get('reference_registered') is True and reg.get('reference_timing_qualified') is True
            and reg.get('timing_qualified') is False and reg.get('physical_sample_timing_qualified') is False
            and 'firmware_sample_age_s' in reg and reg['firmware_sample_age_s'] is None
            and reg.get('physical_spatial_error_bound_m') is None
            and all(reg.get(key) is None for key in ('timing_uncertainty_s','spatial_error_bound_m','joint_error_rad'))
            and reg.get('applied_empirical_delay_s') == 0. and reg.get('applied_delay_s',0.) == 0.
            and reg.get('delay_qualification_id') is None,
            'Receive reference must preserve unknown physical timing and apply no empirical delay')
    else:
        require(reg.get('timing_qualified') is True and reg.get('registration_method') == 'measured_output_interpolation',
            'Measured output history identity/registration method unqualified')
    q = array(reg['joint_pos'],(6,),'Measured motor angles')
    require(j6_bounds[0] <= q[5] <= j6_bounds[1],'Measured J6 violates installed fixed encoder branch')
    tcp = pose(reg['base_from_tcp'],'Measured TCP')
    start = number(timing['exposure_host_start_s'],'Exposure start')
    end = number(timing['exposure_host_end_s'],'Exposure end')
    mid = number(timing.get('exposure_host_mid_s',timing.get('exposure_host_midpoint_s')),'Exposure midpoint')
    require(0 < end-start <= number(reg['max_exposure_s'],'Exposure duration limit')+1e-12
        and 0 < reg['max_exposure_s'] <= .020 and abs(mid-(start+end)/2) <= 1e-6,'Invalid actual exposure interval')
    require(abs(number(timing['exposure_duration_us'],'Actual chunk exposure')*1e-6-(end-start)) <= 1e-6,'Chunk exposure/interval mismatch')
    require(reg['exposure_monotonic_s'] == mid and reg['exposure_interval_s'] == [start,end],'Registrar/native exposure mismatch')
    timing_key = 'reference_timing_uncertainty_s' if reference_tier else 'timing_uncertainty_s'
    spatial_key = 'reference_spatial_error_bound_m' if reference_tier else 'spatial_error_bound_m'
    timing_limit = 'max_reference_timing_uncertainty_s' if reference_tier else 'max_timing_uncertainty_s'
    spatial_limit = 'max_reference_spatial_error_m' if reference_tier else 'max_spatial_error_m'
    uncertainty = number(reg[timing_key],'Reference timing uncertainty' if reference_tier else 'Combined timing uncertainty')
    spatial = number(reg[spatial_key],'Reference spatial estimate' if reference_tier else 'Spatial error bound')
    gap = number(reg['robot_max_gap_s'],'Measured motor bracket gap')
    require(0 <= uncertainty <= number(reg[timing_limit],'Timing limit') <= .002,'Combined timing exceeds2ms')
    spatial_ceiling = .001 if reference_tier else .0005
    require(0 <= spatial <= number(reg[spatial_limit],'Spatial limit') <= spatial_ceiling,
        'Reference spatial estimate exceeds1mm' if reference_tier else 'Spatial error exceeds0.5mm')
    gap_limit = number(reg['max_reference_gap_s'],'Reference bracket limit') if reference_tier else .020
    gap_ceiling = .100 if reference_tier else .020
    require(0 < gap_limit <= gap_ceiling and 0 <= gap <= gap_limit,
        'Measured receive-reference bracket gap exceeds100ms' if reference_tier else 'Measured bracket gap exceeds20ms')
    if reference_tier:
        require(0 <= number(reg['reference_clock_uncertainty_s'],'Kernel receive clock uncertainty') <= uncertainty,
            'Kernel clock uncertainty exceeds reference bound')
    require(0 <= number(timing['exposure_time_uncertainty_s'],'Camera timing uncertainty') <= uncertainty,'Camera uncertainty exceeds combined bound')
    brackets = array(reg['robot_brackets_monotonic_s'],(6,2),'Measured motor brackets')
    query = mid+number(reg.get('applied_delay_s',0.),'Applied signed delay')
    require(np.all(brackets[:,0] <= query) and np.all(query <= brackets[:,1])
        and np.all(np.diff(brackets,axis=1) >= 0) and np.all(np.diff(brackets,axis=1) <= gap+1e-12),
        'Exposure+delay extrapolates or crosses invalid measured brackets')
    require(np.all(array(reg['reference_joint_error_rad' if reference_tier else 'joint_error_rad'],(6,),'Joint uncertainty') >= 0),'Negative measured joint uncertainty')
    status = record.get('status')
    require(status in ('accepted','no_return','rejected'),'Unknown native outcome')
    if status != 'accepted':
        require(bool(record.get('reason')),'Missing rejection/no-return reason')
    return device,q,tcp,mid,uncertainty,spatial,gap


def audit(runtime,session=None,sample_native=32):
    runtime = Path(runtime).resolve()
    require(type(sample_native) is int and 0 <= sample_native <= 100000,'sample-native must be in[0,100000]')
    if session is None:
        session = json.loads((runtime/'motion-scans'/'latest-motion.json').read_text())['scan_id']
    require(isinstance(session,str) and session and Path(session).name == session,'Session must be an archive session ID')
    folder = runtime/'motion-scans'/session
    meta = json.loads((folder/'session.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    errors = []
    error_count = 0
    def error(message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 40:
            errors.append(str(message))
    settings = meta['settings']
    require(settings.get('width_mm') == settings.get('height_mm') == 100.,'Audit requires a100x100mm centered square')
    require(settings.get('spacing_mm',settings.get('step_mm')) == 1. and settings.get('standoff_mm') == 140.,'Expected1mm spacing/140mm target')
    handeye = meta['handeye']
    require(handeye.get('units') == 'm' and handeye.get('coordinate_frame') == 'tcp','Measured hand-eye units/frame invalid')
    correction = pose(handeye['tcp_from_camera'],'Hand-eye')
    original = pose(meta['baseline_pose'],'Original measured baseline')@correction
    if 'baseline_optical' in meta:
        require(np.allclose(original,pose(meta['baseline_optical'],'Explicit optical baseline'),atol=1e-9,rtol=0),'Original optical baseline disagrees with measured Tube+hand-eye')
    identity = meta['identity']
    require(isinstance(identity,list) and len(identity) == 4 and all(isinstance(x,str) and x for x in identity),'Exact four-part calibration identity required')
    require(handeye.get('source_id') == identity[0] and handeye.get('optical_intrinsics_id',handeye.get('calibration_id')) == identity[1], 'Hand-eye optical identity mismatch')
    reference_tier = meta.get('registration_reference') == 'measured_kernel_rx'
    qualification = meta.get('timing_qualification') or {}
    if reference_tier:
        qualification = meta.get('registration_reference_metadata') or {}
        clock_id,history_id = qualification.get('clock_mapping_id'),qualification.get('history_id')
        require(meta.get('physical_sample_timing_qualified') is False and 'firmware_sample_age_s' in meta
            and meta['firmware_sample_age_s'] is None and isinstance(clock_id,str) and clock_id
            and isinstance(history_id,str) and history_id,
            'Receive-reference archive must preserve unknown firmware sample timing and actual clock/history identities')
        for key,value in zip(('source_id','optical_intrinsics_id','acquisition_session_id'),(identity[0],identity[1],identity[3])):
            require(qualification.get(key) == value,f'Receive-reference{key} mismatch')
        delay = 0.
    else:
        qualification = meta.get('timing_qualification') or {}
        clock_id = qualification.get('clock_mapping_id')
        history_id = qualification.get('history_id')
        require(qualification.get('status') == 'qualified' and qualification.get('heldout_passed') is True
            and isinstance(clock_id,str) and clock_id and isinstance(history_id,str) and history_id,
            'Missing qualified experimental output/exposure delay identity')
        delay = number(qualification.get('delay_s'),'Qualified signed delay')
        delay_uncertainty = number(qualification.get('uncertainty_s'),'Qualified delay uncertainty')
        ci = array(qualification.get('confidence_interval_s'),(2,),'Delay confidence interval')
        require(0 <= delay_uncertainty <= .002 and ci[0] <= delay <= ci[1]
            and max(delay-ci[0],ci[1]-delay) <= delay_uncertainty+1e-12,'Qualified delay confidence exceeds2ms')
        for key,value in zip(('source_id','optical_intrinsics_id','acquisition_session_id'),(identity[0],identity[1],identity[3])):
            require(qualification.get(key) == value,f'Delay experiment{key} mismatch')
    calibrated = json.loads((ROOT/'assets/camera_mount/joint-calibration.json').read_text())
    require(np.array_equal(np.asarray(calibrated['fixed_motor_offsets_rad']),np.zeros(6)),'Installed fixed encoder offsets are no longer zero')
    j6_bounds = np.asarray(calibrated['j6_physical_endpoints_rad'])+np.array([1.,-1.])*calibrated['clearance_rad']
    coverage = Coverage(original)
    footprint_config = meta.get('scan_footprint')
    footprint_seen = set()
    if footprint_config is not None:
        from autoweld.scan_footprint import validate_footprint_config
        validate_footprint_config(footprint_config)
        require(footprint_config.get('schema') == 'autoweld.reference-illumination.v1'
            and footprint_config.get('source_id') == identity[0]
            and footprint_config.get('optical_intrinsics_id') == identity[1]
            and footprint_config.get('native_calibration_id') == identity[2]
            and footprint_config.get('width_mm') == footprint_config.get('height_mm') == 100.
            and np.allclose(pose(footprint_config['baseline_optical'],'Footprint original baseline'),original,rtol=0,atol=1e-9),
            'Reference illumination source/calibration/original ROI mismatch')
    readers = {}
    for kind,name in (('full','full-world.ply'),('roi','roi-world.ply')):
        try:
            readers[kind] = BinaryPLY(folder/name)
        except (OSError,ValueError) as issue:
            error(issue)
    # Fixed-size sample targets are spread across the claimed archive, including
    # endpoints. Every record and every selected world point is checked anyway.
    claimed_frames = max(1,int(summary.get('native_frame_count',1)))
    targets = set(np.linspace(0,claimed_frames-1,min(sample_native,claimed_frames),dtype=int).tolist())
    counts = dict(native_frames=0,accepted=0,no_return=0,rejected=0,gaps=0,full_points=0,roi_points=0,native_transform_samples=0)
    maxima = dict(timing_uncertainty_s=0.,spatial_error_bound_m=0.,robot_gap_s=0.,camera_xy_step_m=0.)
    previous_device = None
    previous_good = None
    previous_mid = None
    try:
        for line,record in records(folder/'frames.jsonl'):
            if record.get('kind') == 'gap':
                counts['gaps'] += 1
                try:
                    interval = array(record['interval_monotonic_s'],(2,),'Recorded gap')
                    require(interval[0] <= interval[1] and bool(record.get('reason')),'Invalid recorded gap')
                except (ValueError,KeyError,TypeError) as issue:
                    error(f'line{line}:{issue}')
                continue
            ordinal = counts['native_frames']
            counts['native_frames'] += 1
            try:
                require(record.get('kind') == 'frame' and record.get('ordinal') == ordinal,'Invalid frame ordinal/kind')
                device,q,tcp,mid,uncertainty,spatial,gap = validate_frame(record,identity,clock_id,history_id,j6_bounds,reference_tier)
                if not reference_tier:
                    require(record['registration'].get('delay_qualification_id') == qualification.get('qualification_id')
                        and record['registration'].get('applied_delay_s') == delay,'Frame delay differs from qualified experiment')
                require(previous_device is None or device > previous_device,'Physical device FrameID duplicated/regressed')
                previous_device = device
                maxima['timing_uncertainty_s'] = max(maxima['timing_uncertainty_s'],uncertainty)
                maxima['spatial_error_bound_m'] = max(maxima['spatial_error_bound_m'],spatial)
                maxima['robot_gap_s'] = max(maxima['robot_gap_s'],gap)
                status = record['status']
                counts[status] += 1
                optical = tcp@correction
                coverage.mark(optical[None,:3,3],status)
                if footprint_config is not None:
                    from autoweld.scan_footprint import footprint_cells
                    recomputed = footprint_cells(footprint_config,optical)
                    recorded = record.get('reference_illumination')
                    require(isinstance(recorded,dict) and recorded.get('cells') == recomputed['cells']
                        and recorded.get('reference_plane_z_m') == recomputed['reference_plane_z_m']
                        and recorded.get('measured_surface_occupancy') is False
                        and recorded.get('coverage_basis') == recomputed['coverage_basis'],
                        'Recorded reference illumination differs from measured-pose recomputation')
                    require(np.allclose(np.asarray(recorded['segments_original_m']),np.asarray(recomputed['segments_original_m']),
                        rtol=0,atol=1e-9),'Recorded illumination segments differ from calibrated geometry')
                    if status in ('accepted','no_return'):
                        footprint_seen.update(recomputed['cells'])
                if status != 'rejected':
                    if previous_good is not None and 0 < mid-previous_mid < 1.:
                        step = np.linalg.norm(((optical[:3,3]-previous_good)@original[:3,:3])[:2])
                        maxima['camera_xy_step_m'] = max(maxima['camera_xy_step_m'],float(step))
                    previous_good,previous_mid = optical[:3,3].copy(),mid
                raw = record['raw_file']
                require(raw == f'frame-{ordinal:08d}.npz','Native view file does not match unique frame ordinal')
                path = folder/raw
                require(path.resolve().parent == folder.resolve(),'Native path escapes archive')
                with bounded_npz(path) as native:
                    world = np.asarray(native['xyz_world_m'])
                    indices = np.asarray(native['selected_indices'])
                    require(world.ndim == 2 and world.shape[1] == 3 and np.isfinite(world).all(),'Nonfinite native world XYZ')
                    require(indices.ndim == 1 and indices.dtype.kind in 'iu' and len(np.unique(indices)) == len(indices)
                        and np.all(indices >= 0) and np.all(indices < len(world)),'Invalid selected native indices')
                    require(status != 'no_return' or len(world) == 0,'No-return frame contains native points')
                    selected = world[indices] if status == 'accepted' else np.empty((0,3))
                    require(record['point_count'] == len(selected),'Native selected count differs from journal')
                    selected_roi = selected[coverage.roi(selected)]
                    if 'roi_point_count' in record:
                        require(record['roi_point_count'] == len(selected_roi),'Native ROI count differs from journal')
                    counts['full_points'] += len(selected)
                    counts['roi_points'] += len(selected_roi)
                    coverage.mark(selected_roi,'object_points')
                    if status == 'accepted':
                        require(len(selected) > 0,'Accepted acquisition has no selected native returns')
                        for kind,points in (('full',selected),('roi',selected_roi)):
                            if kind in readers:
                                actual = readers[kind].read(len(points))
                                require(np.array_equal(actual,points),f'{kind}PLY coordinates differ from native selected views')
                    if ordinal in targets:
                        camera = np.asarray(native['xyz_camera_m'])
                        require(camera.shape == world.shape and np.isfinite(camera).all(),'Native camera XYZ invalid')
                        expected = camera@optical[:3,:3].T+optical[:3,3]
                        require(np.allclose(expected,world,atol=1e-9,rtol=0),'Native camera→world transform disagrees with measured exposure pose')
                        require(np.array_equal(native['joint_pos'],q) and np.array_equal(native['base_from_tcp'],tcp)
                            and np.array_equal(native['tcp_from_camera'],correction),'NPZ pose/joints/hand-eye differ from journal')
                        require(json.loads(str(native['provenance_json'])) == record,'NPZ acquisition provenance differs from journal')
                        counts['native_transform_samples'] += 1
            except (OSError,ValueError,TypeError,KeyError,IndexError,zipfile.BadZipFile) as issue:
                error(f'line{line}:{issue}')
        ply = {}
        for kind,reader in readers.items():
            try:
                ply[kind] = reader.finish()
            except (ValueError,OSError) as issue:
                error(issue)
        for field,total in (('native_frame_count','native_frames'),('accepted','accepted'),('no_return','no_return'),
                            ('rejected','rejected'),('gaps','gaps'),('point_count','full_points'),('roi_point_count','roi_points')):
            if field in summary and summary[field] != counts[total]:
                error(f'Summary{field} differs from streamed native journal')
        if summary.get('scan_id') != session or summary.get('schema_version') != 3:
            error('Summary schema/session identity mismatch')
        manifest_path = folder/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            committed_summary = {key:value for key,value in summary.items() if key != 'metadata_path'}
            if manifest.get('summary') != committed_summary or manifest.get('metadata') != meta:
                error('Final manifest metadata/summary disagree with committed session')
        elif summary.get('status') == 'complete':
            error('Completed archive has no committed final manifest')
    finally:
        for reader in readers.values():
            reader.close()
    observed = coverage.state()
    if footprint_config is not None:
        observed.update(footprint_observed_cells_1mm=len(footprint_seen),footprint_total_cells_1mm=10201,
            footprint_complete=len(footprint_seen)==10201,
            footprint_definition='assumed-on calibrated laser intersection with fixed reference plane and sensor FoV; every measured pose recomputed',
            reference_plane_z_m=footprint_config['reference_plane_z_m'],
            measured_surface_occupancy_is_separate=True)
    current = {}
    status_path = runtime/'status.json'
    if status_path.exists():
        state = json.loads(status_path.read_text())
        scan = state.get('scan') or {}
        matches = scan.get('scan_id') == session
        age = time.time()-number(state.get('timestamp',0.),'Runtime snapshot timestamp')
        current = dict(matches_session=matches,snapshot_age_s=age,closed=state.get('closed'),
            rest_verified=state.get('rest_verified'),scan_status=scan.get('status'),
            workflow_elapsed_s=scan.get('workflow_elapsed_s',scan.get('elapsed_s')),
            stopped_monotonic=scan.get('stopped_monotonic'),feedback_source=state.get('feedback_source'),
            current_rest_verified=bool(matches and 0 <= age <= 5. and state.get('closed') is True and state.get('rest_verified') is True))
    integrity = error_count == 0
    observed_complete = observed['footprint_complete'] if footprint_config is not None else observed['all_grid_positions_observed']
    complete = bool(integrity and observed_complete and summary.get('status') == 'complete'
        and counts['accepted'] > 0 and counts['full_points'] > 0)
    return dict(schema='autoweld.independent-continuous-audit.v1',session_id=session,archive_status=summary.get('status'),
        integrity_valid=integrity,archive_verified_complete=complete,
        workflow_verified_complete=bool(complete and current.get('current_rest_verified')),
        geometry=dict(width_mm=100,height_mm=100,spacing_mm=1,baseline_optical=original.tolist(),center_definition='measured baseline TCP @ measured hand-eye'),
        counts=counts,limits_observed=({**maxima,'timing_uncertainty_s':None,'spatial_error_bound_m':None} if reference_tier else maxima),
        reference_limits_observed=(dict(reference_timing_uncertainty_s=maxima['timing_uncertainty_s'],
            reference_spatial_error_bound_m=maxima['spatial_error_bound_m']) if reference_tier else None),
        registration_reference='measured_kernel_rx' if reference_tier else 'physical_sample_time',
        physical_sample_timing_qualified=not reference_tier,firmware_sample_age_s=None if reference_tier else qualification.get('firmware_sample_age_s'),
        physical_spatial_error_bound_m=None if reference_tier else maxima['spatial_error_bound_m'],
        coverage=observed,ply=ply,runtime=current,
        error_count=error_count,errors=errors,
        limitations=['File audit verifies recorded measurements and exports; no live device/model is opened.',
            'Camera-origin coverage and object-point occupancy are independent; empty returns do not create object points.']
            + (['Reference-plane illumination assumes the calibrated laser is on and the declared fixed plane; it is separate from actual native surface occupancy.'] if footprint_config is not None else [])
            + (['Measured motor angles are interpolated at kernel receive-reference times. Firmware encoder sample age is unknown; reported reference bounds do not certify physical exposure-pose accuracy.'] if reference_tier else []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',default='runtime/agents-live',type=Path)
    parser.add_argument('--session')
    parser.add_argument('--sample-native',default=32,type=int)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.runtime,args.session,args.sample_native)
    except (OSError,ValueError,KeyError,TypeError) as error:
        result = dict(schema='autoweld.independent-continuous-audit.v1',integrity_valid=False,
            archive_verified_complete=False,workflow_verified_complete=False,error_count=1,errors=[str(error)])
    serialized = json.dumps(result,allow_nan=False,separators=(',',':'))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(serialized+'\n')
    print(serialized)
    return 0 if result['archive_verified_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
