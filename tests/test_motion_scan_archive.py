import json

import numpy as np
import pytest

from autoweld.motion_scan_archive import MotionScanArchive


def read_ply(path):
    with open(path,'rb') as handle:
        lines=[]
        while True:
            line=handle.readline()
            assert line, 'Missing PLY header terminator'
            lines.append(line)
            if line==b'end_header\n':
                break
        header=b''.join(lines).decode('ascii')
        count=int(next(line.split()[2] for line in header.splitlines() if line.startswith('element vertex ')))
        assert 'format binary_little_endian 1.0' in header
        points=np.frombuffer(handle.read(),dtype='<f8')
        assert points.size==count*3
        return points.reshape(-1,3)


def metadata():
    return dict(settings=dict(width_mm=100., height_mm=100., standoff_mm=140.),
        baseline_pose=np.eye(4).tolist(), identity=['camera','optics','laser','session'],
        handeye=dict(tcp_from_camera=np.eye(4).tolist(), units='m', coordinate_frame='tcp',
                     source_id='camera', optical_intrinsics_id='optics'))


def frame(sequence=0, points=None):
    points=np.asarray([[0.,0.,.14], [.08,0.,.14]] if points is None else points,float).reshape(-1,3)
    timing=dict(motion_pairing_supported=True, host_clock_domain='host_monotonic', sensor_shutter_mode='global',
        device_frame_id=1000+sequence, chunk_frame_id=1000+sequence, exposure_end_frame_id=1000+sequence,
        exposure_host_mid_s=10., exposure_host_midpoint_s=10., exposure_host_start_s=9.9995,
        exposure_host_end_s=10.0005, exposure_duration_us=1000.,exposure_time_uncertainty_s=.0001,
        clock_mapping=dict(status='qualified', mapping_id='clock',uncertainty_s=.0001))
    cloud=dict(source_id='camera',optical_intrinsics_id='optics',calibration_id='laser',
        acquisition_session_id='session',generation=3,sequence=sequence,xyz_m=points.tolist(),
        uv=np.zeros((len(points),2)).tolist(),strength=np.ones(len(points)).tolist(),
        width_px=np.ones(len(points)).tolist(), coordinate_frame='camera',units='m',
        frame_product=dict(kind='single_frame'),timing=timing)
    registration=dict(points_world_m=points.copy(), base_from_tcp=np.eye(4), joint_pos=np.arange(6)*.1,
        exposure_monotonic_s=10., exposure_interval_s=[9.9995,10.0005],timing_uncertainty_s=.0002,
        joint_error_rad=np.ones(6)*.0001,spatial_error_bound_m=.0003,
        robot_brackets_monotonic_s=np.tile([9.995,10.005],(6,1)),robot_max_gap_s=.010,
        history_generation=1,history_id='history',registration_method='measured_output_interpolation',
        delay_qualification_id='delay',applied_delay_s=0., clock_mapping_id='clock',
        source_id='camera',optical_intrinsics_id='optics',acquisition_session_id='session',sequence=sequence,
        timing_qualified=True,no_extrapolation=True,max_spatial_error_m=.0005,
        max_timing_uncertainty_s=.002,max_exposure_s=.020)
    return cloud,registration


def completion(**changes):
    return dict(measured_path_complete=True, continuous_quality_coverage=True,path_fraction=1.,**changes)


def test_raw_native_pose_timing_and_roi_preserved_separately(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    cloud,registered=frame()
    state=archive.append(cloud,registered)
    assert state['captures']==state['visited']==state['native_frame_count']==1
    assert state['point_count']==2
    assert len(state['cloud']['points_base_m'])==1  # Preview is original optical ROI.
    raw=next((tmp_path/'motion-scans'/state['scan_id']).glob('frame-*.npz'))
    with np.load(raw,allow_pickle=False) as native:
        np.testing.assert_array_equal(native['xyz_camera_m'],cloud['xyz_m'])
        np.testing.assert_array_equal(native['xyz_world_m'],registered['points_world_m'])
        np.testing.assert_array_equal(native['joint_pos'],registered['joint_pos'])
        provenance=json.loads(str(native['provenance_json']))
        assert provenance['registration']['timing_qualified'] is True
        assert provenance['cloud']['timing']['exposure_host_mid_s']==10.
    finished=archive.finish(completion_evidence=completion())
    assert finished['status']=='complete' and finished['roi_point_count']==1
    np.testing.assert_array_equal(read_ply(finished['export_paths']['ply']),registered['points_world_m'])
    np.testing.assert_array_equal(read_ply(finished['export_paths']['roi_ply']),registered['points_world_m'][:1])
    assert raw.exists()
    restored=MotionScanArchive(tmp_path).state()
    assert restored['captures']==1 and restored['path_fraction']==1. and restored['status']=='complete'


@pytest.mark.parametrize('field,value', [('timing_qualified',False),('no_extrapolation',False),
    ('registration_method','target_pose'),('spatial_error_bound_m',.0006),('timing_uncertainty_s',.003),
    ('source_id','other'),('sequence',99),('clock_mapping_id','other'),('robot_max_gap_s',.03)])
def test_unqualified_or_mismatched_registration_is_not_written(tmp_path,field,value):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    cloud,registered=frame()
    registered[field]=value
    with pytest.raises(ValueError):
        archive.append(cloud,registered)
    assert archive.state()['visited']==0
    assert not list(archive.session_directory.glob('frame-*.npz'))


def test_brackets_forbid_extrapolation_and_world_transform_cannot_be_replaced(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    cloud,registered=frame()
    registered['robot_brackets_monotonic_s'][:]=[9.98,9.99]
    with pytest.raises(ValueError,match='extrapolates'):
        archive.append(cloud,registered)
    cloud,registered=frame()
    registered['points_world_m'][0,0]=.001
    with pytest.raises(ValueError,match='World XYZ'):
        archive.append(cloud,registered)


def test_pose_qualified_empty_and_rejected_frames_do_not_invent_captures(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame(0,[]),status='no_return',reason='Native frame has no returns',path_fraction=.2)
    state=archive.append(*frame(1),status='rejected',reason='Ambiguous surface',path_fraction=.4)
    assert state['visited']==2 and state['native_frame_count']==2
    assert state['no_return']==state['rejected']==1 and state['accepted']==0
    assert state['point_count']==state['coverage']['observed_1mm_xy_voxel_cells']==0
    assert len(list(archive.session_directory.glob('frame-*.npz')))==2
    with pytest.raises(ValueError,match='zero native'):
        archive.append(*frame(2),status='no_return',reason='invalid')
    restored=MotionScanArchive(tmp_path).state()
    assert restored['status']=='interrupted' and restored['visited']==2


def test_completion_needs_path_and_quality_evidence_and_gap_resolution(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame())
    with pytest.raises(ValueError,match='Completion'):
        archive.finish()
    archive.record_gap('Missing measured right bracket',start_monotonic_s=10.,end_monotonic_s=10.1)
    assert archive.state()['gaps']==1 and archive.state()['visited']==1
    with pytest.raises(ValueError,match='Completion'):
        archive.finish(completion_evidence=completion())
    assert archive.finish(completion_evidence=completion(coverage_gaps_resolved=True))['status']=='complete'


def test_bounded_preview_many_frames_and_no_array_rewrite(tmp_path,monkeypatch):
    archive=MotionScanArchive(tmp_path,max_points=7)
    archive.begin(metadata())
    def forbid(*args,**kwargs):
        raise AssertionError('Append must not reload historical NPZ files')
    monkeypatch.setattr(np,'load',forbid)
    for i in range(75):
        archive.append(*frame(i,[[i/10000.,0.,.14]]))
    state=archive.state()
    assert state['native_frame_count']==75 and state['point_count']==75
    assert state['display_point_count']<=7
    lines=(archive.session_directory/'frames.jsonl').read_text().splitlines()
    assert len(lines)==75
    assert len(list(archive.session_directory.glob('frame-*.npz')))==75
    assert 'captures' not in json.loads(archive.selector.read_text()).get('metadata',{})


def test_duplicate_indices_acquisition_and_optical_session_changes_rejected(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    with pytest.raises(ValueError,match='unique'):
        archive.append(*frame(),selected_indices=[0,0])
    archive.append(*frame())
    with pytest.raises(ValueError,match='Duplicate'):
        archive.append(*frame())
    cloud,registered=frame(1)
    cloud['acquisition_session_id']='other'
    with pytest.raises(ValueError,match='session'):
        archive.append(cloud,registered)


def test_clear_preserves_all_existing_scan_files_and_stationary_archive(tmp_path):
    old=tmp_path/'laser-scans'
    old.mkdir()
    (old/'latest.json').write_text('stationary data')
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame())
    folder=archive.session_directory
    with pytest.raises(RuntimeError,match='active'):
        archive.clear()
    archive.finish('cancelled')
    archive.clear()
    assert archive.state()['scan_id'] is None
    assert (folder/'frames.jsonl').exists() and (old/'latest.json').read_text()=='stationary data'
    assert MotionScanArchive(tmp_path).state()['scan_id'] is None


def test_roi_uses_original_rotated_optical_axes(tmp_path):
    meta=metadata()
    baseline=np.eye(4)
    baseline[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    baseline[:3,3]=[.2,.3,0.]
    meta['baseline_optical']=baseline.tolist()
    archive=MotionScanArchive(tmp_path)
    archive.begin(meta)
    points=np.array([[.2,.34,.14],[.2,.36,.14]])
    state=archive.append(*frame(points=points))
    assert len(state['cloud']['points_base_m'])==1
    report=archive.finish(completion_evidence=completion())
    roi=json.loads(open(report['export_paths']['roi_metadata']).read())
    assert roi['original_baseline_optical']==baseline.tolist() and roi['roi_point_count']==1


def test_uncertain_disk_commit_stops_writer_until_restore(tmp_path,monkeypatch):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    original=archive._journal
    def uncertain(record):
        original(record)
        raise OSError('fsync outcome uncertain')
    monkeypatch.setattr(archive,'_journal',uncertain)
    with pytest.raises(OSError):
        archive.append(*frame())
    with pytest.raises(RuntimeError,match='Begin'):
        archive.append(*frame(1))
    with pytest.raises(RuntimeError,match='restore'):
        archive.finish('error')
    restored=MotionScanArchive(tmp_path)
    assert restored.state()['visited']==1 and restored.state()['status']=='interrupted'
    assert restored.finish('error')['point_count']==2


def test_restore_rejects_missing_no_return_native_and_ignores_torn_tail(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame(0,[]),status='no_return',reason='No measured returns')
    journal=archive.session_directory/'frames.jsonl'
    with journal.open('ab') as handle:
        handle.write(b'{"torn":')
    assert MotionScanArchive(tmp_path).state()['visited']==1
    next(archive.session_directory.glob('frame-*.npz')).unlink()
    assert MotionScanArchive(tmp_path).state()['status']=='error'


def test_selected_world_exports_preserve_all_unselected_native_points(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame(),selected_indices=[0])
    done=archive.finish(completion_evidence=completion())
    assert done['point_count']==1
    with np.load(next(archive.session_directory.glob('frame-*.npz')),allow_pickle=False) as native:
        assert len(native['xyz_camera_m'])==len(native['xyz_world_m'])==2
        assert native['selected_indices'].tolist()==[0]
    np.testing.assert_array_equal(read_ply(done['export_paths']['ply']),[[0.,0.,.14]])


def test_measured_preview_has_no_stationary_or_settled_gate(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    cloud,registration=frame()
    registration['joint_vel_rad_s']=[.25]*6
    registration['stationary']=False
    assert archive.append(cloud,registration)['accepted']==1


def test_new_source_sequence_cannot_duplicate_physical_device_acquisition(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    archive.append(*frame(0))
    cloud,registration=frame(1)
    for key in ('device_frame_id','chunk_frame_id','exposure_end_frame_id'):
        cloud['timing'][key]=1000
    with pytest.raises(ValueError,match='Duplicate'):
        archive.append(cloud,registration)
    cloud['timing']['device_frame_id']=1001
    with pytest.raises(ValueError,match='identity'):
        archive.append(cloud,registration)


def test_json_provenance_excludes_duplicate_native_arrays_and_dense_image_aliases(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    for sequence,count in enumerate((1,700)):
        points=np.tile([.012345678901234567,0.,.14000000000000001],(count,1))
        cloud,registration=frame(sequence,points)
        cloud.update(xyz_mm=points*1000.,frame=np.zeros((1024,1024),np.uint8),
                     preview_frame=np.zeros((32,32,3),np.uint8),points_base_m=points.copy())
        archive.append(cloud,registration)
    records=[json.loads(line) for line in (archive.session_directory/'frames.jsonl').read_text().splitlines()]
    for record in records:
        assert not {'xyz_m','xyz_mm','frame','preview_frame','points_base_m','uv','strength','width_px'} & record['cloud'].keys()
        assert record['cloud']['timing']['device_frame_id']==1000+record['ordinal']
        with np.load(archive.session_directory/record['raw_file'],allow_pickle=False) as native:
            embedded=json.loads(str(native['provenance_json']))
            assert embedded['cloud']==record['cloud']
            assert len(native['xyz_camera_m']) in (1,700)
    # Journal size depends on provenance, not on the native stripe point count.
    assert abs(len(json.dumps(records[1]))-len(json.dumps(records[0])))<64


def test_binary_double_exports_retain_exact_float64_geometry_and_old_journals(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    points=np.array([[.012345678901234567,-.012345678901234567,np.nextafter(.14,1.)],
                     [.08765432109876543,0.,np.nextafter(.14,0.)]],dtype=np.float64)
    archive.append(*frame(points=points))
    # Earlier schema-3 views did not record per-frame ROI counts.
    journal=archive.session_directory/'frames.jsonl'
    record=json.loads(journal.read_text())
    record.pop('roi_point_count')
    journal.write_text(json.dumps(record)+'\n')
    state=archive.finish(completion_evidence=completion())
    np.testing.assert_array_equal(read_ply(state['export_paths']['ply']),points)
    np.testing.assert_array_equal(read_ply(state['export_paths']['roi_ply']),points[:1])
    report=json.loads(open(state['export_paths']['roi_metadata']).read())
    assert report['ply_format']=='binary_little_endian' and report['point_dtype']=='<f8'
    assert report['vertex_stride_bytes']==24


def reference_metadata():
    meta=metadata()
    meta.update(registration_reference='measured_kernel_rx',physical_sample_timing_qualified=False,
        firmware_sample_age_s=None,registration_reference_metadata=dict(history_id='history',clock_mapping_id='clock',
            source_id='camera',optical_intrinsics_id='optics',acquisition_session_id='session'))
    return meta


def reference_frame(sequence=0):
    cloud,reg=frame(sequence)
    reg.update(registration_reference='measured_kernel_rx',registration_method='measured_output_receive_reference',
        physical_sample_timing_qualified=False,firmware_sample_age_s=None,physical_spatial_error_bound_m=None,
        applied_empirical_delay_s=0.,applied_delay_s=0.,delay_qualification_id=None,
        timing_qualified=False,reference_registered=True,reference_timing_qualified=True,
        reference_clock_uncertainty_s=.0001,reference_timing_uncertainty_s=.0002,
        reference_spatial_error_bound_m=.0003,reference_joint_error_rad=np.ones(6)*.0001,
        max_reference_timing_uncertainty_s=.002,max_reference_spatial_error_m=.001,max_reference_gap_s=.100,
        timing_uncertainty_s=None,spatial_error_bound_m=None,joint_error_rad=None)
    return cloud,reg


def test_receive_reference_retains_unknown_physical_timing_through_finish_restore(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(reference_metadata())
    archive.append(*reference_frame())
    state=archive.finish(completion_evidence=completion())
    assert state['registration_reference']=='measured_kernel_rx'
    assert state['physical_sample_timing_qualified'] is False
    assert state['firmware_sample_age_s'] is state['physical_spatial_error_bound_m'] is None
    restored=MotionScanArchive(tmp_path).state()
    assert restored['status']=='complete' and restored['physical_sample_timing_qualified'] is False
    record=json.loads((archive.session_directory/'frames.jsonl').read_text())
    assert record['registration']['timing_qualified'] is False
    assert record['registration']['reference_timing_qualified'] is True
    assert record['registration']['timing_uncertainty_s'] is None


@pytest.mark.parametrize('changes',[dict(physical_sample_timing_qualified=True),dict(firmware_sample_age_s=0.),
    dict(timing_qualified=True),dict(applied_empirical_delay_s=.001),dict(applied_delay_s=.001),
    dict(spatial_error_bound_m=.0003),dict(history_id='other'),dict(reference_clock_uncertainty_s=.003)])
def test_receive_reference_rejects_false_physical_claims_or_fitted_delay(tmp_path,changes):
    archive=MotionScanArchive(tmp_path)
    archive.begin(reference_metadata())
    cloud,reg=reference_frame()
    reg.update(changes)
    with pytest.raises(ValueError):
        archive.append(cloud,reg)


def test_receive_reference_must_be_explicit_and_cannot_mix_with_physical_branch(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(metadata())
    with pytest.raises(ValueError):
        archive.append(*reference_frame())
    archive.finish('cancelled')
    archive.begin(reference_metadata())
    with pytest.raises(ValueError):
        archive.append(*frame())


def test_real_receive_registrar_output_roundtrips_archive_without_physical_certification(tmp_path):
    from test_motion_registration import LinearFK, fill, history

    from autoweld.motion_registration import MovingCloudRegistrar
    outputs=fill(history(encoder_sample_age_bound_s=None,reference_clock='kernel_receive'))
    meta=reference_metadata()
    meta['registration_reference_metadata']['history_id']=outputs.history_id
    native,_=frame()
    native['timing'].update(exposure_host_mid_s=10.05,exposure_host_midpoint_s=10.05,
                           exposure_host_start_s=10.0495,exposure_host_end_s=10.0505)
    measured=MovingCloudRegistrar(outputs,LinearFK(),meta['handeye']).register_receive_reference(native)
    archive=MotionScanArchive(tmp_path)
    archive.begin(meta)
    archive.append(native,measured)
    state=archive.finish(completion_evidence=completion())
    assert state['captures']==1 and not state['physical_sample_timing_qualified']


def test_receive_reference_allows_declared_100ms_1mm_estimate_without_physical_claim(tmp_path):
    archive=MotionScanArchive(tmp_path)
    archive.begin(reference_metadata())
    cloud,reg=reference_frame()
    reg.update(robot_max_gap_s=.08,robot_brackets_monotonic_s=np.tile([9.96,10.04],(6,1)),
               reference_spatial_error_bound_m=.0008)
    state=archive.append(cloud,reg)
    assert state['accepted']==1 and state['physical_sample_timing_qualified'] is False
    assert state['physical_spatial_error_bound_m'] is None


@pytest.mark.parametrize('changes',[dict(robot_max_gap_s=.101),dict(max_reference_gap_s=.101),
    dict(reference_spatial_error_bound_m=.00101),dict(max_reference_spatial_error_m=.00101),
    dict(reference_timing_uncertainty_s=.00201)])
def test_receive_reference_still_enforces_declared_reference_clock_gap_error_limits(tmp_path,changes):
    archive=MotionScanArchive(tmp_path)
    archive.begin(reference_metadata())
    cloud,reg=reference_frame()
    reg.update(changes)
    with pytest.raises(ValueError):
        archive.append(cloud,reg)


def footprint_metadata():
    from test_scan_footprint import calibration

    from autoweld.scan_footprint import plan_footprint
    meta=reference_metadata()
    meta['settings']['spacing_mm']=1.
    active=calibration()
    active['camera_matrix'][1][1]=1000.  # Whole100mm reference line is inside sensor.
    cfg=plan_footprint(active,np.eye(4),np.eye(3),reference_plane_z_m=.147)
    cfg.update(native_calibration_id='laser',optical_intrinsics_id='optics')
    meta['scan_footprint']=cfg
    return meta


def footprint_frame(meta,index,x,status='no_return'):
    cloud,reg=reference_frame(index)
    camera=np.eye(4)
    camera[:3,3]=[*np.asarray(meta['scan_footprint']['origin_compensation_xy_m']),.007]
    camera[0,3]+=x
    reg['base_from_tcp']=camera
    if status=='no_return':
        cloud.update(xyz_m=[],uv=[],strength=[],width_px=[])
    reg['points_world_m']=np.asarray(cloud['xyz_m'],float).reshape(-1,3)+camera[:3,3]
    reg['joint_pos'][5]=-3.
    return cloud,reg


def test_footprint_coverage_uses_actual_line_and_no_return_not_planned_origin_bins(tmp_path):
    archive=MotionScanArchive(tmp_path)
    meta=footprint_metadata()
    archive.begin(meta)
    archive.append(*footprint_frame(meta,0,-.05,'accepted'))
    first=archive.state()
    assert first['coverage']['footprint_observed_cells_1mm']==101
    assert first['coverage']['observed_camera_xy_cells_1mm']==1
    assert len(archive.footprint_missing_cells())==10100
    archive.append(*footprint_frame(meta,1,-.049),status='rejected',reason='Bad stripe')
    assert archive.state()['coverage']['footprint_observed_cells_1mm']==101
    with pytest.raises(ValueError,match='illumination coverage is incomplete'):
        archive.finish(completion_evidence=completion())
    for index,x in enumerate(np.arange(-.049,.05001,.001),start=2):
        archive.append(*footprint_frame(meta,index,float(x)),status='no_return',reason='Native no return')
    archive.record_gap('Historical sampling gap repaired by measured illumination',start_monotonic_s=1.,end_monotonic_s=2.)
    state=archive.finish(completion_evidence=completion(actual_footprint_complete=True))
    assert state['coverage']['footprint_complete'] and not archive.footprint_missing_cells()
    assert state['point_count']==2 and state['coverage']['footprint_observed_cells_1mm']==10201
    assert state['native_frame_count']==102  # Never fake10201 distinct camera acquisitions.
    restored=MotionScanArchive(tmp_path).state()
    assert restored['coverage']['footprint_complete'] and restored['point_count']==2


def test_interrupted_finish_preserves_measured_repair_evidence_and_missing_cells(tmp_path):
    archive = MotionScanArchive(tmp_path)
    archive.begin(metadata())
    cloud,registration = frame()
    archive.append(cloud,registration)
    evidence = dict(measured_path_complete=True,actual_footprint_complete=False,
        remaining_footprint_cells=85,unobserved_footprint_cell_indices=[0,1,10200],
        repairs=[dict(kind='reference_footprint_patch',observed_cells_added=10,
            settled_burst=dict(new_qualified_frames=10,one_fixed_hold=True))])
    state = archive.finish(status='interrupted',completion_evidence=evidence)
    assert state['completion_evidence'] == evidence
    summary = json.loads((archive.session_directory/'summary.json').read_text())
    manifest = json.loads((archive.session_directory/'manifest.json').read_text())
    assert summary['completion_evidence'] == evidence
    assert manifest['summary']['completion_evidence'] == evidence
    assert state['status'] == 'interrupted'
    assert state['point_count'] == 2
