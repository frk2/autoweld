"""Independent file audit tests; no hardware, robot model or camera is imported."""
import json
import time

import numpy as np
import pytest
from test_motion_scan_archive import completion, frame, metadata, reference_frame, reference_metadata

from autoweld.motion_scan_archive import MotionScanArchive
from scripts.verify_continuous_scan import Coverage, audit, main


def archive_fixture(tmp_path):
    record = metadata()
    record['settings']['spacing_mm'] = 1.
    record['timing_qualification'] = dict(status='qualified',heldout_passed=True,
        qualification_id='delay',history_id='history',clock_mapping_id='clock',
        source_id='camera',optical_intrinsics_id='optics',acquisition_session_id='session',
        delay_s=0.,uncertainty_s=.0001,confidence_interval_s=[-.0001,.0001])
    archive = MotionScanArchive(tmp_path)
    archive.begin(record)
    for index in range(2):
        cloud,registered = frame(index)
        registered['joint_pos'][5] = -3.
        # Native XYZ spans inside and outside the requested original optical ROI.
        archive.append(cloud,registered,path_fraction=index)
    archive.finish(status='complete',completion_evidence=completion())
    return archive,archive.session_directory


def mutate_journal(folder,change):
    path = folder/'frames.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_partial_observations_never_become_full_grid_from_completion_boolean(tmp_path):
    archive,_ = archive_fixture(tmp_path)
    result = audit(tmp_path)
    assert result['integrity_valid'],result['errors']
    assert result['archive_status'] == 'complete'
    assert not result['archive_verified_complete']
    assert result['coverage']['good_camera_bins'] == 1
    assert result['coverage']['full_rows'] == 0
    assert len(result['coverage']['incomplete_rows']) == 101
    assert result['counts']['full_points'] == 4
    assert result['counts']['roi_points'] == 2
    assert result['counts']['native_transform_samples'] == 2
    assert result['ply']['full']['vertex_count'] == 4
    assert archive.state()['point_count'] == 4


def test_actual_camera_grid_and_object_occupancy_are_independent():
    coverage = Coverage(np.eye(4))
    x,y = np.meshgrid(np.arange(101)/1000.-.05,np.arange(101)/1000.-.05)
    positions = np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    coverage.mark(positions,'object_points')
    coverage.mark([[0.,0.,0.]],'accepted')
    assert not coverage.state()['all_grid_positions_observed']
    assert coverage.state()['object_point_xy_bins'] == 10201
    coverage.mark(positions[::2],'accepted')
    coverage.mark(positions[1::2],'no_return')
    assert coverage.state()['all_grid_positions_observed']
    assert coverage.state()['full_rows'] == 101
    assert coverage.state()['good_camera_bins'] == 10201


@pytest.mark.parametrize('change,reason',[
    (lambda rows:rows[0]['registration'].update(timing_uncertainty_s=.0021),'2ms'),
    (lambda rows:rows[0]['registration'].update(robot_max_gap_s=.021),'20ms'),
    (lambda rows:rows[0]['registration'].update(spatial_error_bound_m=.00051),'0.5mm'),
    (lambda rows:rows[0]['registration']['joint_pos'].__setitem__(5,3.),'J6'),
    (lambda rows:rows[0]['registration']['robot_brackets_monotonic_s'][0].__setitem__(0,10.001),'extrapolates'),
    (lambda rows:rows[1]['cloud']['timing'].update(device_frame_id=1000,chunk_frame_id=1000,exposure_end_frame_id=1000),'identity'),
])
def test_every_journal_record_is_audited_even_without_native_transform_sampling(tmp_path,change,reason):
    _,folder = archive_fixture(tmp_path)
    mutate_journal(folder,change)
    result = audit(tmp_path,sample_native=0)
    assert not result['integrity_valid']
    assert any(reason in error for error in result['errors'])


def test_binary_ply_byte_length_corruption_is_detected(tmp_path):
    _,folder = archive_fixture(tmp_path)
    with (folder/'full-world.ply').open('ab') as handle:
        handle.write(b'12345678')
    result = audit(tmp_path)
    assert not result['integrity_valid']
    assert any('byte length' in error for error in result['errors'])


def test_every_ply_coordinate_matches_npz_not_only_headers_and_counts(tmp_path):
    _,folder = archive_fixture(tmp_path)
    path = folder/'roi-world.ply'
    data = path.read_bytes()
    start = data.index(b'end_header\n')+len(b'end_header\n')
    corrupted = np.frombuffer(data[start:],dtype='<f8').copy()
    corrupted[0] = .051  # Outside the original optical XY ROI.
    path.write_bytes(data[:start]+corrupted.tobytes())
    result = audit(tmp_path)
    assert not result['integrity_valid']
    assert any('roiPLY coordinates differ' in error for error in result['errors'])


def test_sampled_native_transform_detects_consistent_count_pose_tampering(tmp_path):
    _,folder = archive_fixture(tmp_path)
    path = folder/'frame-00000000.npz'
    with np.load(path,allow_pickle=False) as source:
        values = {key:source[key] for key in source.files}
    values['xyz_camera_m'] = values['xyz_camera_m']+np.array([.001,0,0])
    np.savez(path,**values)
    result = audit(tmp_path,sample_native=1)
    assert not result['integrity_valid']
    assert any('transform disagrees' in error for error in result['errors'])


def test_runtime_rest_requires_matching_fresh_session_and_reports_actual_elapsed(tmp_path):
    archive,_ = archive_fixture(tmp_path)
    (tmp_path/'status.json').write_text(json.dumps(dict(timestamp=time.time(),closed=True,
        rest_verified=True,feedback_source='SDK feedback',scan=dict(scan_id=archive.state()['scan_id'],
            status='complete',workflow_elapsed_s=812.3,stopped_monotonic=99.))))
    result = audit(tmp_path)
    assert result['runtime']['current_rest_verified']
    assert result['runtime']['workflow_elapsed_s'] == 812.3
    assert not result['workflow_verified_complete']  # Independent grid is still incomplete.


def test_cli_writes_explicit_failed_evidence_for_missing_archive(tmp_path,capsys):
    output = tmp_path/'audit.json'
    assert main(['--runtime',str(tmp_path),'--output',str(output)]) == 1
    result = json.loads(output.read_text())
    assert not result['archive_verified_complete']
    assert result == json.loads(capsys.readouterr().out)


def test_no_uncommitted_journal_tail_is_silently_ignored(tmp_path):
    _,folder = archive_fixture(tmp_path)
    with (folder/'frames.jsonl').open('ab') as handle:
        handle.write(b'{"kind":"frame"')
    with pytest.raises(ValueError,match='uncommitted'):
        audit(tmp_path)


def test_npz_declared_huge_array_is_rejected_before_loading_or_allocating(tmp_path):
    import io
    import zipfile
    _,folder = archive_fixture(tmp_path)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header,dict(descr='<f8',fortran_order=False,shape=(10**9,3)))
    with zipfile.ZipFile(folder/'frame-00000000.npz','w') as archive:
        archive.writestr('xyz_world_m.npy',header.getvalue())
    result = audit(tmp_path)
    assert not result['integrity_valid']
    assert any('unbounded/invalid native array' in message for message in result['errors'])


def receive_archive_fixture(tmp_path):
    meta=reference_metadata()
    meta['settings']['spacing_mm']=1.
    archive=MotionScanArchive(tmp_path)
    archive.begin(meta)
    cloud,reg=reference_frame()
    reg['joint_pos'][5]=-3.
    archive.append(cloud,reg,path_fraction=1.)
    archive.finish(completion_evidence=completion())
    return archive


def test_receive_reference_audit_preserves_unknown_physical_accuracy(tmp_path):
    receive_archive_fixture(tmp_path)
    result=audit(tmp_path)
    assert result['integrity_valid'],result['errors']
    assert result['registration_reference']=='measured_kernel_rx'
    assert result['physical_sample_timing_qualified'] is False
    assert result['firmware_sample_age_s'] is None
    assert result['physical_spatial_error_bound_m'] is None
    assert result['limits_observed']['timing_uncertainty_s'] is None
    assert result['reference_limits_observed']['reference_timing_uncertainty_s']==.0002
    assert 'Firmware encoder sample age is unknown' in result['limitations'][-1]
    assert not result['archive_verified_complete']  # Actual grid still independently audited.


def test_reference_tier_can_complete_coverage_audit_without_firmware_time_certification(tmp_path,monkeypatch):
    receive_archive_fixture(tmp_path)
    # Coverage geometry has separate exhaustive tests; isolate admission of the
    # honest reference tier to a successful completion result here.
    monkeypatch.setattr(Coverage,'state',lambda self:{'all_grid_positions_observed':True})
    result=audit(tmp_path)
    assert result['archive_verified_complete'] and not result['physical_sample_timing_qualified']


@pytest.mark.parametrize('changes',[dict(timing_qualified=True),dict(firmware_sample_age_s=0.),
    dict(applied_empirical_delay_s=.001),dict(registration_method='measured_output_interpolation')])
def test_reference_file_audit_rejects_invented_physical_qualification_or_delay(tmp_path,changes):
    archive=receive_archive_fixture(tmp_path)
    mutate_journal(archive.session_directory,lambda rows:rows[0]['registration'].update(changes))
    result=audit(tmp_path,sample_native=0)
    assert not result['integrity_valid']
    assert any('Receive reference' in error for error in result['errors'])


def test_receive_reference_audit_accepts_80ms_08mm_estimates_as_reference_only(tmp_path):
    meta=reference_metadata()
    meta['settings']['spacing_mm']=1.
    archive=MotionScanArchive(tmp_path)
    archive.begin(meta)
    cloud,reg=reference_frame()
    reg['joint_pos'][5]=-3.
    reg.update(robot_max_gap_s=.08,robot_brackets_monotonic_s=np.tile([9.96,10.04],(6,1)),
               reference_spatial_error_bound_m=.0008)
    archive.append(cloud,reg)
    archive.finish(completion_evidence=completion())
    result=audit(tmp_path)
    assert result['integrity_valid'],result['errors']
    assert result['reference_limits_observed']['reference_spatial_error_bound_m']==.0008
    assert result['limits_observed']['robot_gap_s']==.08
    assert result['physical_spatial_error_bound_m'] is None and not result['physical_sample_timing_qualified']


@pytest.mark.parametrize('changes',[dict(robot_max_gap_s=.101),dict(max_reference_gap_s=.101),
    dict(reference_spatial_error_bound_m=.00101),dict(reference_timing_uncertainty_s=.00201)])
def test_receive_reference_audit_rejects_values_beyond_exploratory_limits(tmp_path,changes):
    archive=receive_archive_fixture(tmp_path)
    mutate_journal(archive.session_directory,lambda rows:rows[0]['registration'].update(changes))
    result=audit(tmp_path,sample_native=0)
    assert not result['integrity_valid']


def test_independent_audit_recomputes_full_illumination_without_101_origin_rows(tmp_path):
    from test_motion_scan_archive import footprint_frame, footprint_metadata
    meta=footprint_metadata()
    archive=MotionScanArchive(tmp_path)
    archive.begin(meta)
    for index,x in enumerate(np.arange(-.05,.05001,.001)):
        status='accepted' if index==0 else 'no_return'
        archive.append(*footprint_frame(meta,index,float(x),status),status=status,
                       reason=None if status=='accepted' else 'No native return')
    archive.finish(completion_evidence=completion(actual_footprint_complete=True))
    result=audit(tmp_path)
    assert result['integrity_valid'],result['errors']
    assert result['archive_verified_complete']
    assert result['coverage']['footprint_observed_cells_1mm']==10201
    assert not result['coverage']['all_grid_positions_observed']
    assert result['counts']['native_frames']==101 and result['counts']['full_points']==2
    mutate_journal(archive.session_directory,lambda rows:rows[1]['reference_illumination']['cells'].pop())
    failed=audit(tmp_path,sample_native=0)
    assert not failed['integrity_valid']
    assert any('illumination differs' in text for text in failed['errors'])
