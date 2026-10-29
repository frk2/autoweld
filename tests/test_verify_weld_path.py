"""Independent file audit catches association, geometry and timing corruption."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_weld_seam_service import rig as service_rig

SCRIPT = Path(__file__).parents[1]/'scripts'/'verify_weld_path.py'
SPEC = importlib.util.spec_from_file_location('verify_weld_path', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def refresh_hashes(manifest):
    saved = json.loads(manifest.read_text())
    for item in saved.get('native_files', []):
        native = manifest.parent/item['path']
        item['sha256'] = audit.hashlib.sha256(native.read_bytes()).hexdigest()
    saved['native_files_sha256'] = audit.digest(saved.get('native_files', []))
    saved['path_sha256'] = audit.digest(saved['path'])
    save(manifest, saved)


@pytest.fixture
def archive(tmp_path):
    angle = .35
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    handeye = np.eye(4)
    handeye[:3, :3] = rotation.T
    handeye[:3, 3] = [.02, .005, .01]
    binding = dict(source_id='camera-native', optical_intrinsics_id='optical-solve',
        calibration_id='laser-solve', acquisition_session_id='acquisition-owner', generation=3,
        model_fingerprint='installed-model', tcp_from_camera=handeye.tolist())
    path, natives = [], []
    stations = [(0, 0), *[(1, i) for i in range(1, 18)], *[(-1, i) for i in range(1, 18)]]
    for sequence, (direction, distance) in enumerate(stations, 1):
        base = 100.+sequence*2.
        timing = dict(host_clock_domain='host_monotonic', motion_pairing_supported=True,
            sensor_shutter_mode='global', clock_mapping={'status': 'qualified', 'mapping_id': 'clock-map'},
            device_frame_id=10000+sequence, chunk_frame_id=10000+sequence,
            exposure_end_frame_id=10000+sequence, exposure_host_start_s=base+.6,
            exposure_host_end_s=base+.61, exposure_time_uncertainty_s=.0001)
        xyz = np.c_[np.linspace(-10., 10., 101), np.zeros(101), np.full(101, 150.)]
        cloud = dict(**{key: binding[key] for key in audit.IDENTITY},
            units='mm', coordinate_frame='camera', count=len(xyz), xyz_mm=xyz.tolist(),
            xyz_m=(xyz/1000.).tolist(), uv=np.c_[np.arange(101), np.zeros(101)].tolist(),
            sequence=sequence, read_started=base+.62, received=base+.63, timing=timing,
            bounds_mm=[xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()], frame_product={'kind': 'single_frame'})
        camera = np.eye(4)
        camera[:3, :3] = rotation
        camera[:3, 3] = np.array([.2, .1, .05])+rotation @ [0., direction*distance/1000., 0.]
        tcp = camera @ np.linalg.inv(handeye)
        feedback = dict(fresh=True, stationary=True, measured_at=base+.65)
        measurement = dict(cloud=cloud, base_from_tcp=tcp.tolist(), base_from_camera=camera.tolist(),
            requested_base_from_tcp=tcp.tolist(), feedback=feedback, quiet_pose=tcp.tolist(),
            quiet_since=base+.05, capture_cutoff=base+.5, hold_monotonic=base,
            exposure_interval_s=[base+.6, base+.61], exposure_uncertainty_s=.0001,
            captured_at=base+.66, direction=direction, distance_mm=float(distance),
            acquisition_key=[binding['acquisition_session_id'], 3, sequence],
            post_capture=dict(feedback=dict(fresh=True, stationary=True, measured_at=base+.79),
                base_from_tcp=tcp.tolist(), observed_at=base+.8))
        found = distance <= 14
        result = dict(schema_version=1, status='found' if found else 'no_seam', found=found,
            units='mm', coordinate_frame='camera', seam_point_mm=[0., 0., 150.] if found else None,
            camera_from_torch=np.eye(4).tolist() if found else None,
            stripe_direction_camera=[1., 0., 0.] if found else None,
            scan_direction_camera=[0., 1., 0.] if found else None,
            flank_lines_mm=[[[-10., 0., 150.], [-1., 0., 150.]], [[1., 0., 150.], [10., 0., 150.]]] if found else [],
            quality={'good': True}, provenance={key: copy.deepcopy(cloud[key]) for key in (*audit.IDENTITY,
                'sequence', 'read_started', 'timing')})
        filename = f'measured-view-{sequence:03d}.json'
        save(tmp_path/filename, dict(schema='autoweld.weld-native.v1', measurement=measurement, result=result))
        natives.append(dict(path=filename, sha256='', observation_index=sequence, status=result['status']))
        if found:
            seam = camera[:3, :3] @ np.array(result['seam_point_mm'])/1000.+camera[:3, 3]
            flanks = np.asarray(result['flank_lines_mm']) @ camera[:3, :3].T/1000.+camera[:3, 3]
            path.append(dict(direction=direction, distance_mm=float(distance), seam_point_base_m=seam.tolist(),
                scan_lines_base_m=flanks.tolist(), torch_pose_base_m=camera.tolist(),
                measured_base_from_tcp=tcp.tolist(), acquisition_key=measurement['acquisition_key'],
                detector=result, native_path=filename))
    path.sort(key=lambda record: record['direction']*record['distance_mm'])
    endpoints = {}
    for direction, name in ((-1, 'negative'), (1, 'positive')):
        endpoint = next(record for record in path if record['direction'] == direction and record['distance_mm'] == 14.)
        endpoints[name] = dict(confirmed=True, no_seam_stations=3, distance_mm=17.,
            last_seam_point_base_m=endpoint['seam_point_base_m'])
    points = np.asarray([record['seam_point_base_m'] for record in path])
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()*1000.)
    manifest = tmp_path/'path.json'
    save(manifest, dict(schema='autoweld.weld-path.v1', session_id='fixture-session', status='complete',
        settings=dict(start_pose='Weld Finder', min_distance_mm=25., standoff_mm=150., step_mm=1., end_confirmations=3),
        binding=binding, endpoints=endpoints, seam_length_mm=length, minimum_satisfied=True,
        path=path, path_sha256='', units='m', coordinate_frame='robot_base', native_files=natives))
    refresh_hashes(manifest)
    return manifest


def native_for(manifest, signed_distance):
    saved = json.loads(manifest.read_text())
    record = next(item for item in saved['path'] if item['direction']*item['distance_mm'] == signed_distance)
    filename = manifest.parent/record['native_path']
    return filename, json.loads(filename.read_text())


def enable_generation_policy(manifest, *, transition_after_sequence=18):
    """A legitimate owned AE update between two settled stations."""
    saved = json.loads(manifest.read_text())
    transition = dict(previous_generation=3, generation=4, reason='owned_native_stripe_auto_exposure',
        observed_at=100.+transition_after_sequence*2.+1.5, controls={'exposure_us': 500.},
        auto_exposure={'state': 'stable'})
    saved['binding'].update(initial_generation=3, final_generation=4, generation=4,
        generation_policy='owned_stripe_AE_per_station_requalification')
    saved['generation_transitions'] = [transition]
    for entry in saved['native_files']:
        filename = manifest.parent/entry['path']
        native = json.loads(filename.read_text())
        measurement, result = native['measurement'], native['result']
        sequence = measurement['cloud']['sequence']
        generation = 4 if sequence > transition_after_sequence else 3
        measurement['cloud']['generation'] = result['provenance']['generation'] = generation
        measurement['acquisition_key'][1] = generation
        measurement['generation_transitions'] = [transition] if generation == 4 else []
        measurement['post_capture'].update(camera_generation=generation, generation_matches=True)
        entry['disposition'] = 'path_point' if result['found'] else 'endpoint_observation'
        save(filename, native)
        for record in saved['path']:
            if record['native_path'] == entry['path']:
                record['detector'] = copy.deepcopy(result)
                record['acquisition_key'] = measurement['acquisition_key']
    for direction, name in ((-1, 'negative'), (1, 'positive')):
        links = []
        for entry in saved['native_files']:
            native = json.loads((manifest.parent/entry['path']).read_text())
            measurement = native['measurement']
            if measurement['direction'] == direction and native['result']['status'] == 'no_seam':
                links.append(dict(native_path=entry['path'], acquisition_key=measurement['acquisition_key'],
                    distance_mm=measurement['distance_mm'], observation_index=entry['observation_index']))
        saved['endpoints'][name]['observations'] = links[-3:]
    save(manifest, saved)
    refresh_hashes(manifest)
    return transition


def enable_manual_bracket(manifest, *, requested_us=16000., actual_us=16000., maximum_us=16000.):
    transition = enable_generation_policy(manifest)
    saved = json.loads(manifest.read_text())
    transition.update(reason='owned_exposure_bracket', request_id='fixed-hold-bracket-1',
        requested_exposure_us=requested_us, bracket_attempt=1,
        controls={'exposure_us': actual_us, 'auto_exposure': False})
    limits = dict(min_us=10., max_us=maximum_us, fps_cap_us=maximum_us)
    bracket = dict(request_id='fixed-hold-bracket-1', requested_us=requested_us, actual_us=actual_us,
        readback={'exposure_us': actual_us, 'auto_exposure': False}, limits=limits,
        started_monotonic=137.3, completed_monotonic=137.6, measured_hold_monotonic=136.5,
        direction=-1, distance_mm=1., reason='Insufficient native stripe support', attempt=1, generation=4)
    bracket['configuration_readback'] = copy.deepcopy(bracket['readback'])
    saved['exposure_brackets'] = [bracket]
    saved['generation_transitions'] = [transition]
    for entry in saved['native_files']:
        filename = manifest.parent/entry['path']
        native = json.loads(filename.read_text())
        measurement, result = native['measurement'], native['result']
        if measurement['cloud']['generation'] != 4:
            continue
        measurement['generation_transitions'] = [transition]
        measurement['cloud']['controls'] = copy.deepcopy(bracket['readback'])
        measurement['cloud']['timing'].update(camera_exposure_us=actual_us, exposure_duration_us=actual_us)
        measurement['cloud']['timing']['exposure_host_end_s'] += actual_us/1e6-.01
        measurement['exposure_interval_s'][1] += actual_us/1e6-.01
        measurement['exposure_control'] = dict(mode='owned_manual_bracket', requested_us=requested_us, actual_us=actual_us,
            request_id=bracket['request_id'], limits=limits, readback=copy.deepcopy(bracket['readback']),
            configuration_readback=copy.deepcopy(bracket['readback']))
        measurement['auto_exposure'] = dict(state='owned_manual_bracket', reliable=None,
            exposure_us=actual_us, request_id=bracket['request_id'])
        result['provenance']['timing'] = copy.deepcopy(measurement['cloud']['timing'])
        save(filename, native)
        for record in saved['path']:
            if record['native_path'] == entry['path']:
                record['detector'] = copy.deepcopy(result)
    save(manifest, saved)
    refresh_hashes(manifest)


def test_independent_audit_checks_every_point_pose_station_and_native_end_observation(archive):
    result = audit.audit_manifest(archive)
    assert result['passed'] and result['complete'] and result['replay_eligible']
    assert result['native_observations'] == 35 and result['measured_seam_points'] == 29
    assert result['seam_length_mm'] == pytest.approx(28.)
    assert result['raw_observation_hashes_verified'] and result['evidence_limitations'] == []
    assert result['detector_rerun'] is False


def test_path_digest_detects_record_edit_before_geometry_is_trusted(archive):
    saved = json.loads(archive.read_text())
    saved['path'][0]['seam_point_base_m'][0] += .001
    save(archive, saved)
    with pytest.raises(audit.AuditError, match='Path SHA256'):
        audit.audit_manifest(archive)


def test_raw_xyz_hash_detects_interior_native_edit_without_rerunning_detector(archive):
    filename, native = native_for(archive, 0.)
    native['measurement']['cloud']['xyz_mm'][50][2] += .1
    save(filename, native)
    with pytest.raises(audit.AuditError, match='observation SHA256'):
        audit.audit_manifest(archive)


def test_recomputed_transform_catches_world_xyz_edit_even_with_updated_path_digest(archive):
    saved = json.loads(archive.read_text())
    saved['path'][0]['seam_point_base_m'][0] += .001
    save(archive, saved)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='World seam XYZ'):
        audit.audit_manifest(archive)


@pytest.mark.parametrize('field', ['source_id', 'optical_intrinsics_id', 'calibration_id', 'acquisition_session_id', 'generation'])
def test_native_identity_corruption_cannot_be_hidden_by_updated_file_hash(archive, field):
    filename, native = native_for(archive, 0.)
    native['measurement']['cloud'][field] = 'different-owner'
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match=field):
        audit.audit_manifest(archive)


def test_exposure_uncertainty_cannot_overlap_capture_cutoff(archive):
    filename, native = native_for(archive, 0.)
    m = native['measurement']
    m['capture_cutoff'] = m['exposure_interval_s'][0]
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='overlaps or precedes'):
        audit.audit_manifest(archive)


def test_post_exposure_feedback_must_be_fresh_stationary_and_later_than_exposure(archive):
    filename, native = native_for(archive, 0.)
    native['measurement']['feedback']['measured_at'] = native['measurement']['exposure_interval_s'][1]
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='bracket the complete exposure'):
        audit.audit_manifest(archive)


def test_endpoint_needs_three_distinct_valid_no_seam_stations(archive):
    filename = archive.parent/'measured-view-018.json'
    native = json.loads(filename.read_text())
    native['result']['quality']['good'] = False
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='Poor/ambiguous laser'):
        audit.audit_manifest(archive)


def test_zero_length_repeated_geometry_is_not_measured_travel(archive):
    saved = json.loads(archive.read_text())
    saved['seam_length_mm'] = 0.
    save(archive, saved)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='Reported seam length'):
        audit.audit_manifest(archive)


def test_incomplete_file_is_audited_without_claiming_complete_or_replayable(archive):
    saved = json.loads(archive.read_text())
    saved['status'] = 'cancelled'
    saved['endpoints'] = {'positive': None, 'negative': None}
    save(archive, saved)
    refresh_hashes(archive)
    result = audit.audit_manifest(archive)
    assert result['passed'] and not result['complete'] and not result['replay_eligible']


def test_old_schema_evidence_limits_are_explicit_not_fabricated(archive):
    saved = json.loads(archive.read_text())
    saved.pop('native_files')
    saved.pop('native_files_sha256')
    save(archive, saved)
    filename, native = native_for(archive, 0.)
    for key in ('requested_base_from_tcp', 'captured_at', 'post_capture'):
        native['measurement'].pop(key)
    save(filename, native)
    result = audit.audit_manifest(archive)
    assert result['passed'] and not result['raw_observation_hashes_verified']
    assert len(result['evidence_limitations']) == 4


def test_cli_reports_failure_as_json_and_nonzero_without_hardware_imports(archive, tmp_path, capsys):
    saved = json.loads(archive.read_text())
    saved['path_sha256'] = 'bad'
    save(archive, saved)
    output = tmp_path/'audit-report.json'
    assert audit.main(['--manifest', str(archive), '--output', str(output)]) == 1
    assert json.loads(output.read_text())['passed'] is False
    assert json.loads(capsys.readouterr().out)['passed'] is False


def test_legitimate_owned_auto_exposure_transition_preserves_path_and_endpoints(archive):
    enable_generation_policy(archive)
    result = audit.audit_manifest(archive)
    assert result['passed'] and result['complete'] and result['evidence_limitations'] == []


def test_unexpected_native_generation_change_is_rejected_even_if_all_hashes_updated(archive):
    enable_generation_policy(archive)
    filename, native = native_for(archive, -1.)
    native['measurement']['cloud']['generation'] = 5
    native['measurement']['acquisition_key'][1] = 5
    native['result']['provenance']['generation'] = 5
    save(filename, native)
    saved = json.loads(archive.read_text())
    for record in saved['path']:
        if record['native_path'] == filename.name:
            record['detector'] = native['result']
            record['acquisition_key'] = native['measurement']['acquisition_key']
    save(archive, saved)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='explicit owned exposure transition history'):
        audit.audit_manifest(archive)


def test_generation_transition_does_not_excuse_missing_new_quiet_qualification(archive):
    transition = enable_generation_policy(archive)
    filename, native = native_for(archive, -1.)
    native['measurement']['quiet_since'] = transition['observed_at']-.01
    native['measurement']['hold_monotonic'] = transition['observed_at']-.1
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='restart quiet qualification'):
        audit.audit_manifest(archive)


def test_archived_only_reacquire_cannot_qualify_a_path_point_or_endpoint(archive):
    enable_generation_policy(archive)
    saved = json.loads(archive.read_text())
    next(entry for entry in saved['native_files'] if entry['status'] == 'no_seam')['disposition'] = 'reacquire'
    save(archive, saved)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='three fresh contiguous no-seam'):
        audit.audit_manifest(archive)


def test_inflight_old_generation_is_archived_as_reacquire_and_never_used(archive):
    enable_generation_policy(archive, transition_after_sequence=0)
    saved = json.loads(archive.read_text())
    _, original = native_for(archive, 0.)
    old = copy.deepcopy(original)
    measurement, result = old['measurement'], old['result']
    cloud = measurement['cloud']
    cloud.update(generation=3, sequence=0, read_started=101.32, received=101.33)
    cloud['timing'].update(device_frame_id=10000, chunk_frame_id=10000, exposure_end_frame_id=10000,
        exposure_host_start_s=101.3, exposure_host_end_s=101.31)
    result['provenance'] = {key: copy.deepcopy(cloud[key]) for key in (*audit.IDENTITY,
        'sequence', 'read_started', 'timing')}
    measurement.update(captured_at=101.4, quiet_since=100.7, capture_cutoff=101.2, hold_monotonic=100.6,
        feedback=dict(fresh=True, stationary=True, measured_at=101.35),
        exposure_interval_s=[101.3, 101.31], generation_transitions=[],
        acquisition_key=[saved['binding']['acquisition_session_id'], 3, 0])
    measurement['post_capture'].update(observed_at=101.6, camera_generation=4, generation_matches=False,
        feedback=dict(fresh=True, stationary=True, measured_at=101.59))
    old['result'] = dict(status='reacquire', found=False, original_result=result,
        reason='Native exposure generation changed during analysis')
    filename = 'inflight-old-exposure.json'
    save(archive.parent/filename, old)
    saved['native_files'].insert(0, dict(path=filename, sha256='', status='reacquire', disposition='reacquire'))
    for index, entry in enumerate(saved['native_files'], 1):
        entry['observation_index'] = index
    for endpoint in saved['endpoints'].values():
        for link in endpoint['observations']:
            link['observation_index'] += 1
    save(archive, saved)
    refresh_hashes(archive)
    result = audit.audit_manifest(archive)
    assert result['passed'] and result['native_observations'] == 36 and result['measured_seam_points'] == 29


def test_copied_archive_audits_its_local_native_files_even_if_original_paths_still_exist(archive, tmp_path):
    saved = json.loads(archive.read_text())
    for entry in saved['native_files']:
        entry['path'] = str(archive.parent/entry['path'])
    for record in saved['path']:
        record['native_path'] = str(archive.parent/record['native_path'])
    save(archive, saved)
    refresh_hashes(archive)
    copied = tmp_path/'copied-archive'
    copied.mkdir()
    for source in archive.parent.glob('*.json'):
        shutil.copy2(source, copied/source.name)
    assert audit.audit_manifest(copied/'path.json')['passed']
    native = copied/'measured-view-001.json'
    payload = json.loads(native.read_text())
    payload['measurement']['cloud']['xyz_mm'][50][2] += .1
    save(native, payload)
    with pytest.raises(audit.AuditError, match='observation SHA256'):
        audit.audit_manifest(copied/'path.json')


def test_owned_manual_exposure_bracket_uses_sensor_readback_without_claiming_ae_reliability(archive):
    enable_manual_bracket(archive)
    assert audit.audit_manifest(archive)['passed']


def test_tampered_native_manual_controls_rejected_with_even_recomputed_hashes(archive):
    enable_manual_bracket(archive)
    filename, native = native_for(archive, -1.)
    native['measurement']['cloud']['controls']['exposure_us'] = 8000.
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='Native manual controls differ'):
        audit.audit_manifest(archive)


def test_manual_generation_transition_requires_completed_config_proof(archive):
    enable_manual_bracket(archive)
    saved = json.loads(archive.read_text())
    saved['exposure_brackets'] = []
    save(archive, saved)
    with pytest.raises(audit.AuditError, match='completed configuration proof'):
        audit.audit_manifest(archive)


def test_manual_readback_cannot_exceed_fps_cap_or_claim_reliable_auto_exposure(archive):
    enable_manual_bracket(archive)
    saved = json.loads(archive.read_text())
    saved['exposure_brackets'][0]['limits']['fps_cap_us'] = 10000.
    save(archive, saved)
    with pytest.raises(audit.AuditError, match='sensor/FPS bounds'):
        audit.audit_manifest(archive)


def test_manual_generation_requires_per_capture_control_proof(archive):
    enable_manual_bracket(archive)
    filename, native = native_for(archive, -1.)
    native['measurement'].pop('exposure_control')
    save(filename, native)
    refresh_hashes(archive)
    with pytest.raises(audit.AuditError, match='omitted its native exposure'):
        audit.audit_manifest(archive)


def test_verified_firmware_exposure_quantization_is_not_compared_to_nominal_request(archive):
    enable_manual_bracket(archive, requested_us=4140.74, actual_us=4148., maximum_us=16562.96)
    assert audit.audit_manifest(archive)['passed']


@pytest.fixture
def service_archive(tmp_path):
    """Produce the manifest with the real coordinator and hardware-free owners.

    Add realistic immutable camera metadata at the fake owner's publication
    boundary; every manifest, station, disposition, hash and search projection
    is then generated by WeldSeamService itself.
    """
    generator = service_rig.__wrapped__(tmp_path)
    rig = next(generator)
    rig.state['controls']['auto_exposure'] = False
    original_find = rig.service.camera.find_weld_seam
    original_tick = rig.weld.tick

    def find(cloud, **kwargs):
        result = original_find(cloud, **kwargs)
        result.update(units='mm', coordinate_frame='camera', schema_version=1,
            provenance={key: copy.deepcopy(cloud[key]) for key in (*audit.IDENTITY,
                'sequence', 'read_started', 'timing')})
        return result

    def qualified_tick():
        cloud = rig.state['laser'].get('cloud')
        if cloud:
            sequence = cloud['sequence']
            timing = cloud['timing']
            actual_us = cloud['controls']['exposure_us']
            timing.update(sensor_shutter_mode='global', device_frame_id=10000+sequence,
                chunk_frame_id=10000+sequence, exposure_end_frame_id=10000+sequence,
                camera_exposure_us=actual_us, exposure_duration_us=actual_us)
            timing['clock_mapping']['mapping_id'] = 'synthetic-native-clock-map'
            timing['exposure_host_start_s'] = timing['exposure_host_end_s']-actual_us/1e6
            cloud['frame_product'] = {'kind': 'single_frame'}
        original_tick()

    rig.service.camera.find_weld_seam = find
    rig.weld.tick = qualified_tick
    yield rig
    try:
        next(generator)
    except StopIteration:
        pass


def test_real_service_generated_manifest_audits_initial_ae_lock_and_short_seam_with_full_search(service_archive):
    rig = service_archive
    rig.mode['ends_mm'] = 2.
    rig.weld.start({'min_distance_mm': 25.})
    for _ in range(3000):
        rig.advance()
        if rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy:
            break
    assert rig.weld.status == 'complete', rig.weld.error
    manifest = rig.path/'weld-paths'/rig.weld.session_id/'path.json'
    report = audit.audit_manifest(manifest)
    assert report['passed'] and report['coverage_complete'] and report['minimum_satisfied']
    assert report['minimum_basis'] == 'measured_scan_plane_extent'
    assert report['seam_length_mm'] < 5. and report['measured_search']['total_mm'] >= 25.
    assert json.loads(manifest.read_text())['exposure_brackets'][0]['kind'] == 'ae_lock'


def test_real_service_gain_recovery_and_unknown_endpoint_keep_only_measured_section_replay(service_archive):
    rig = service_archive
    rig.state['control_limits'] = {'gain_db': [0., 48.]}
    original_find = rig.service.camera.find_weld_seam

    def find(cloud, **kwargs):
        rig.mode['failure'] = 'invalid' if rig.pose[1, 3] >= .0019 else None
        return original_find(cloud, **kwargs)

    rig.service.camera.find_weld_seam = find
    rig.weld.start({'min_distance_mm': 25.})
    for _ in range(3000):
        rig.advance()
        if rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy:
            break
    assert rig.weld.status == 'incomplete', rig.weld.error
    manifest = rig.path/'weld-paths'/rig.weld.session_id/'path.json'
    saved = json.loads(manifest.read_text())
    assert any(bracket['kind'] == 'gain' for bracket in saved['exposure_brackets'])
    report = audit.audit_manifest(manifest)
    assert report['passed'] and not report['coverage_complete'] and report['replay_eligible']
    assert report['replay_scope'] == 'measured_section_only' and report['unconfirmed_ends'] == ['positive']


def test_ae_freeze_cannot_substitute_another_source_or_claim_missing_native_ae_proof(service_archive):
    rig = service_archive
    rig.weld.start({'min_distance_mm': 1.})
    rig.mode['ends_mm'] = 1.
    for _ in range(1000):
        rig.advance()
        if rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy:
            break
    manifest = rig.path/'weld-paths'/rig.weld.session_id/'path.json'
    saved = json.loads(manifest.read_text())
    saved['exposure_brackets'][0]['auto_exposure']['source_id'] = 'another-camera'
    save(manifest, saved)
    with pytest.raises(audit.AuditError, match='fresh native AE feedback'):
        audit.audit_manifest(manifest)
