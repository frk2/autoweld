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


@pytest.fixture
def coping_archive(archive):
    """Known circular saddle section with inward poses and actual native links."""
    saved = json.loads(archive.read_text())
    selected = [record for record in saved['path'] if record['distance_mm'] <= 1]
    seed_record = next(record for record in selected if record['direction'] == 0)
    seed_payload = json.loads((archive.parent/seed_record['native_path']).read_text())
    frame = np.asarray(seed_payload['measurement']['base_from_camera'])
    handeye = np.asarray(saved['binding']['tcp_from_camera'])
    radius, phase = 12.7, np.pi/4
    origin = np.array([-radius*np.cos(phase), -radius*np.sin(phase), 150.-radius*np.cos(phase)])
    model = dict(units='mm', coordinate_frame='camera', joint_angle_deg=90.,
        axes=[dict(origin_mm=origin.tolist(), direction=[0., 0., 1.], radius_mm=radius),
              dict(origin_mm=origin.tolist(), direction=[1., 0., 0.], radius_mm=radius)])
    names = {record['native_path'] for record in selected}
    for item in list(archive.parent.glob('measured-view-*.json')):
        if item.name not in names:
            item.unlink()
    for record in selected:
        angle = record['direction']*5.
        theta = phase+np.deg2rad(angle)
        point = origin+radius*np.array([np.cos(theta), np.sin(theta), np.cos(theta)])
        centerline = origin+np.array([0., 0., point[2]-origin[2]])
        inward = centerline-point
        inward /= np.linalg.norm(inward)
        tangent = np.array([-np.sin(theta), np.cos(theta), -np.sin(theta)])
        tangent /= np.linalg.norm(tangent)
        torch = np.eye(4)
        torch[:3, :3] = np.column_stack([np.cross(tangent, inward), tangent, inward])
        torch[:3, 3] = (point-150.*inward)/1000.
        world = frame[:3, :3] @ (point/1000.)+frame[:3, 3]
        camera = frame.copy()
        camera[:3, 3] = world-frame[:3, :3] @ [0., 0., .150]
        tcp = camera @ np.linalg.inv(handeye)
        filename = archive.parent/record['native_path']
        payload = json.loads(filename.read_text())
        measurement = payload['measurement']
        measurement.update(base_from_camera=camera.tolist(), base_from_tcp=tcp.tolist(),
            requested_base_from_tcp=tcp.tolist(), quiet_pose=tcp.tolist(),
            station_basis='coped_angular_stations', station_index=record['distance_mm'], angular_theta_deg=angle)
        measurement['post_capture']['base_from_tcp'] = tcp.tolist()
        save(filename, payload)
        record.update(seam_point_base_m=world.tolist(), measured_base_from_tcp=tcp.tolist(),
            scan_lines_base_m=(np.asarray(record['detector']['flank_lines_mm']) @ camera[:3, :3].T/1000.+camera[:3, 3]).tolist(),
            torch_pose_base_m=(frame @ torch).tolist(), torch_orientation_policy='coped_centerline',
            station_basis='coped_angular_stations', station_index=record['distance_mm'], angular_theta_deg=angle,
            requested_angular_theta_deg=angle,
            coping_geometry=dict(model_seed_camera=model, base_from_seed_camera=frame.tolist(), coping_axis=0,
                measured_point_seed_camera_mm=point.tolist(), centerline_closest_point_seed_camera_mm=centerline.tolist(),
                inward_direction_seed_camera=inward.tolist(), radial_residual_mm=[0., 0.], root_sign=1,
                finite_root_ambiguity_band_mm=.2, finite_root_signed_coordinate_mm=float(point[2]-origin[2]),
                association=dict(measured_point_mm=point.tolist(), predicted_point_mm=point.tolist(), residual_mm=0.,
                    associated=True, singularity=None, prediction_is_measurement=False, measured_geometry_preserved=True)))
    selected.sort(key=lambda row: row['direction']*row['distance_mm'])
    saved['native_files'] = [entry for entry in saved['native_files'] if entry['path'] in names]
    for index, entry in enumerate(saved['native_files'], 1):
        entry['observation_index'] = index
        entry['disposition'] = 'path_point'
    saved['settings'].update(joint_kind='coped_tube', branch_diameter_mm=25.4, main_diameter_mm=25.4,
        joint_angle_deg=90., target_arc_deg=180.)
    points = np.asarray([row['seam_point_base_m'] for row in selected])
    saved.update(path=selected, joint_kind='coped_tube', status='incomplete', minimum_satisfied=False,
        coverage_complete=False, replay_scope='measured_section_only', minimum_basis='measured_coped_angular_span',
        endpoints={'positive': {'confirmed': False}, 'negative': {'confirmed': False}},
        seam_length_mm=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()*1000.),
        joint_model={'status': 'qualified', 'model': model}, coping_seed_base_from_camera=frame.tolist(),
        coping_curve={'root_sign': 1, 'predicted_only': True},
        coping_coverage=dict(requested_arc_deg=180., planned_arc_deg=10., observed_arc_deg=10.,
            positive_observed_deg=5., negative_observed_deg=5., measured_station_count=3, coverage_complete=False))
    save(archive, saved)
    refresh_hashes(archive)
    return archive


def test_coping_audit_preserves_measured_seam_and_recomputes_inward_poses_and_angular_span(coping_archive):
    result = audit.audit_manifest(coping_archive)
    assert result['passed'] and result['joint_kind'] == 'coped_tube'
    assert result['measured_angular_coverage']['observed_arc_deg'] == pytest.approx(10.)
    assert result['torch_orientation_policy'] == 'coped_centerline'
    assert result['replay_scope'] == 'measured_section_only' and not result['complete']


@pytest.mark.parametrize('corruption', ['outward_torch', 'angle', 'coverage', 'diameter'])
def test_coping_audit_rejects_false_inward_geometry_and_unmeasured_coverage(coping_archive, corruption):
    saved = json.loads(coping_archive.read_text())
    if corruption == 'outward_torch':
        rotation = np.asarray(saved['path'][0]['torch_pose_base_m'])
        rotation[:3, 0] *= -1
        rotation[:3, 2] *= -1
        saved['path'][0]['torch_pose_base_m'] = rotation.tolist()
    elif corruption == 'angle':
        saved['path'][0]['angular_theta_deg'] -= 3.
    elif corruption == 'coverage':
        saved['coping_coverage']['observed_arc_deg'] = 180.
    else:
        saved['settings']['main_diameter_mm'] = 30.
    save(coping_archive, saved)
    refresh_hashes(coping_archive)
    with pytest.raises(audit.AuditError):
        audit.audit_manifest(coping_archive)


@pytest.mark.parametrize('corruption', [None, 'geometry', 'indices', 'registration'])
def test_coping_surface_fit_samples_are_actual_native_connected_runs(tmp_path, corruption):
    camera = np.eye(4)
    camera[:3, 3] = [.001, .002, .003]
    points = np.column_stack((np.arange(36), np.zeros(36), np.full(36, 150.)))
    uv = np.column_stack((np.arange(36), np.r_[np.zeros(15), np.full(6, 100.), np.zeros(15)]))
    native = tmp_path/'native.json'
    cloud = dict(xyz_mm=points.tolist(), uv=uv.tolist())
    save(native, {'cloud': cloud})
    indices = [*range(15), *range(21, 36)]
    provenance = dict(native_path=str(native), native_sha256=audit.hashlib.sha256(native.read_bytes()).hexdigest(),
        base_from_camera=camera.tolist(), native_point_indices=indices,
        native_point_count=30, raw_native_point_count=36,
        surface_return_selection=dict(parameters=dict(min_run_points=12, max_scan_gap_px=3., max_jump_px=8.),
            native_geometry_preserved=True))
    selected = points[indices]+camera[:3, 3]*1000.
    saved = dict(coping_seed_base_from_camera=np.eye(4).tolist(),
        coping_profiles=[dict(xyz_mm=selected.tolist(), provenance=provenance)])
    if corruption == 'geometry':
        saved['coping_profiles'][0]['xyz_mm'][0][0] += .01
    elif corruption == 'indices':
        provenance['native_point_indices'][14] = 15
    elif corruption == 'registration':
        provenance['base_from_camera'][0][3] += .001
    observations = {native.resolve(): dict(camera=camera, measurement={'cloud': cloud})}
    if corruption:
        with pytest.raises(ValueError, match='Coping surface|Coping fit'):
            audit.validate_coping_surface_profiles(saved, observations, tmp_path)
    else:
        audit.validate_coping_surface_profiles(saved, observations, tmp_path)


@pytest.mark.parametrize('corruption', [None, 'other_frame', 'ambiguous', 'exclusion', 'indices', 'geometry'])
def test_clean_full_profile_fit_is_bound_to_native_rows_and_declared_gap_exclusion(tmp_path, corruption):
    points = np.c_[np.linspace(-6., 6., 121), np.zeros(121), np.full(121, 150.)]
    pixels = np.c_[np.arange(121), np.zeros(121)]
    hint = np.array([0., 0., 150.])
    left = np.flatnonzero((points[:, 0] < 0) & (np.linalg.norm(points-hint, axis=1) >= .5))
    right = np.flatnonzero((points[:, 0] >= 0) & (np.linalg.norm(points-hint, axis=1) >= .5))
    indices = np.r_[left, right].tolist()
    labels = [0]*len(left)+[1]*len(right)
    full = dict(sequence=12, xyz_mm=points.tolist(), uv=pixels.tolist(),
        reflection={'status': 'resolved_optical', 'physical_primary_proven': False})
    cloud = {'sequence': 12, 'full_profile': full}
    native = tmp_path/'native.json'
    save(native, {'cloud': cloud})
    prepared = dict(schema='weldfinder.coped_surface_profile_selection.v1', status='selected',
        native_point_indices=indices, surface_labels=labels, counts_as_measured_seam=False,
        diagnostics=dict(native_geometry_preserved=True, junction_exclusion_mm=.5,
            split_hint_mm=hint.tolist()))
    provenance = dict(native_path=str(native), native_sha256=audit.hashlib.sha256(native.read_bytes()).hexdigest(),
        base_from_camera=np.eye(4).tolist(), native_cloud_path='full_profile', native_point_indices=indices,
        native_point_count=len(indices), raw_native_point_count=len(points), surface_profile_selection=prepared)
    profile = dict(xyz_mm=points[indices].tolist(), surface_labels=labels, provenance=provenance)
    saved = dict(coping_seed_base_from_camera=np.eye(4).tolist(), coping_profiles=[profile])
    if corruption == 'other_frame':
        full['sequence'] += 1
    elif corruption == 'ambiguous':
        full['reflection']['status'] = 'ambiguous'
    elif corruption == 'exclusion':
        prepared['diagnostics']['junction_exclusion_mm'] = 2.
    elif corruption == 'indices':
        provenance['native_point_indices'] = indices[1:]
    elif corruption == 'geometry':
        profile['xyz_mm'][0][2] += .1
    observations = {native.resolve(): dict(camera=np.eye(4), measurement={'cloud': cloud})}
    if corruption:
        with pytest.raises(ValueError):
            audit.validate_coping_surface_profiles(saved, observations, tmp_path)
    else:
        audit.validate_coping_surface_profiles(saved, observations, tmp_path)


@pytest.fixture
def bridged_exposure_policy(tmp_path):
    """A single 1..5 budget interrupted after attempt2 by one measured replan."""
    goal = np.eye(4)
    goal[:3, 3] = [.2, .1, .3]
    missed = goal.copy()
    missed[0, 3] += .0006

    def evidence(observed, *, after=False):
        actual = goal if after else missed
        position, orientation = audit.pose_error(actual, goal)
        result = dict(observed_at=observed,
            feedback=dict(fresh=True, stationary=True, measured_at=observed-.01),
            actual_base_from_tcp=actual.tolist(), requested_base_from_tcp=goal.tolist(),
            position_error_m=position, orientation_error_rad=orientation,
            hold_monotonic=12. if after else 10., station_deadline=39.,
            retry_attempt=1 if after else 0, gate='qualified' if after else 'endpoint_position')
        if after:
            result['quiet_seconds'] = .45
        return result

    before = evidence(11.)
    before.update(endpoint_miss_seconds=.22, endpoint_miss_samples=4)
    retry = dict(attempt=1, maximum_replans=1, status='complete',
        requested_base_from_tcp=goal.tolist(), before_evidence=before,
        preceding_samples=[evidence(stamp) for stamp in (10.78, 10.86, 10.94)],
        started_monotonic=11.01, completed_monotonic=12.5, station_deadline=39.,
        optical_recovery_attempts=2, prior_hold_monotonic=10., replacement_hold_monotonic=12.,
        after_evidence=evidence(12.5, after=True), diagnostic_path=str(tmp_path/'endpoint-retry-0001.json'))
    brackets, transitions = [], []
    for attempt, exposure, started in zip(range(1, 6), (16000., 8000., 4000., 16000., 8000.),
                                         (10.2, 10.5, 12.7, 13., 13.3)):
        controls = dict(exposure_us=exposure, gain_db=12., auto_exposure=False)
        request_id = f'one-station-bracket-{attempt}'
        brackets.append(dict(request_id=request_id, requested_us=exposure, actual_us=exposure,
            configuration_readback=copy.deepcopy(controls), readback=copy.deepcopy(controls),
            limits=dict(min_us=10., max_us=16000., fps_cap_us=16000.), kind='exposure',
            measured_hold_monotonic=10. if attempt <= 2 else 12., started_monotonic=started,
            completed_monotonic=started+.05, attempt=attempt, generation=attempt+1,
            direction=1, distance_mm=2.))
        transitions.append(dict(previous_generation=attempt, generation=attempt+1,
            reason='owned_exposure_bracket', observed_at=started+.025, controls=controls,
            request_id=request_id, bracket_attempt=attempt, requested_exposure_us=exposure))
    saved = dict(binding=dict(source_id='camera', initial_generation=1, final_generation=6, generation=6,
        generation_policy='owned_stripe_AE_per_station_requalification'), exposure_brackets=brackets,
        generation_transitions=transitions, endpoint_retries=[retry])
    save(Path(retry['diagnostic_path']), retry)
    return saved, tmp_path


def test_completed_endpoint_retry_preserves_one_contiguous_bracket_budget(bridged_exposure_policy):
    saved, directory = bridged_exposure_policy
    original = copy.deepcopy(saved)
    result = audit.validate_generation_policy(saved, directory)
    assert len(result['endpoint_retry_bridges']) == 1
    bridge = result['endpoint_retry_bridges'][0]
    assert bridge['prior_hold_monotonic'] == 10. and bridge['replacement_hold_monotonic'] == 12.
    assert bridge['optical_recovery_attempts'] == 2 and bridge['station_deadline'] == 39.
    assert [b['attempt'] for b in result['brackets'].values()] == [1, 2, 3, 4, 5]
    assert saved == original


@pytest.mark.parametrize('corruption', ['missing_retry', 'wrong_hold', 'reset_budget', 'skipped_attempt',
    'extended_budget', 'attempt_count', 'missing_file', 'different_file', 'two_replans',
    'changed_endpoint', 'renewed_deadline', 'stale_before', 'moving_after', 'no_before_miss',
    'bad_after_endpoint', 'short_after_quiet', 'short_miss_history', 'unstable_miss',
    'before_error_disagrees', 'configuration_before_settle', 'configuration_after_original_deadline'])
def test_unproved_or_reset_hold_bridges_fail_closed(bridged_exposure_policy, corruption):
    saved, directory = bridged_exposure_policy
    retry = saved['endpoint_retries'][0]
    if corruption == 'missing_retry':
        saved['endpoint_retries'] = []
    elif corruption == 'wrong_hold':
        retry['replacement_hold_monotonic'] = 12.01
        retry['after_evidence']['hold_monotonic'] = 12.01
    elif corruption == 'reset_budget':
        for bracket, transition in zip(saved['exposure_brackets'][2:], saved['generation_transitions'][2:]):
            bracket['attempt'] -= 2
            transition['bracket_attempt'] -= 2
    elif corruption in ('skipped_attempt', 'extended_budget'):
        saved['exposure_brackets'][2]['attempt'] = 4 if corruption == 'skipped_attempt' else 6
        saved['generation_transitions'][2]['bracket_attempt'] = saved['exposure_brackets'][2]['attempt']
    elif corruption == 'attempt_count':
        retry['optical_recovery_attempts'] = 1
    elif corruption == 'missing_file':
        Path(retry['diagnostic_path']).unlink()
    elif corruption == 'different_file':
        Path(retry['diagnostic_path']).write_text('{}')
    elif corruption == 'two_replans':
        retry['maximum_replans'] = 2
    elif corruption == 'changed_endpoint':
        retry['after_evidence']['requested_base_from_tcp'][0][3] += .001
    elif corruption == 'renewed_deadline':
        retry['station_deadline'] = 45.
        retry['before_evidence']['station_deadline'] = retry['after_evidence']['station_deadline'] = 45.
        for sample in retry['preceding_samples']:
            sample['station_deadline'] = 45.
    elif corruption == 'stale_before':
        retry['before_evidence']['feedback']['measured_at'] -= .2
    elif corruption == 'moving_after':
        retry['after_evidence']['feedback']['stationary'] = False
    elif corruption == 'no_before_miss':
        retry['before_evidence']['actual_base_from_tcp'][0][3] -= .0002
        retry['before_evidence']['position_error_m'] = .0004
    elif corruption == 'bad_after_endpoint':
        retry['after_evidence']['actual_base_from_tcp'][0][3] += .0006
        retry['after_evidence']['position_error_m'] = .0006
    elif corruption == 'short_after_quiet':
        retry['after_evidence']['quiet_seconds'] = .39
    elif corruption == 'short_miss_history':
        retry['preceding_samples'] = retry['preceding_samples'][-1:]
    elif corruption == 'unstable_miss':
        sample = retry['preceding_samples'][0]
        sample['actual_base_from_tcp'][0][3] += .0004
        sample['position_error_m'] += .0004
    elif corruption == 'before_error_disagrees':
        retry['before_evidence']['position_error_m'] = .002
    elif corruption == 'configuration_before_settle':
        saved['exposure_brackets'][2].update(started_monotonic=12.4, completed_monotonic=12.45)
        saved['generation_transitions'][2]['observed_at'] = 12.425
    elif corruption == 'configuration_after_original_deadline':
        saved['exposure_brackets'][-1].update(started_monotonic=39.1, completed_monotonic=39.15)
        saved['generation_transitions'][-1]['observed_at'] = 39.125
    if corruption not in ('missing_file', 'different_file'):
        save(Path(retry['diagnostic_path']), retry)
    with pytest.raises(audit.AuditError):
        audit.validate_generation_policy(saved, directory)


@pytest.mark.parametrize('corruption', [None, 'missing_native', 'wrong_endpoint', 'different_retry', 'renewed_deadline'])
def test_endpoint_retry_bridge_is_bound_to_native_station_evidence(bridged_exposure_policy, corruption):
    saved, directory = bridged_exposure_policy
    policy = audit.validate_generation_policy(saved, directory)
    retry = saved['endpoint_retries'][0]
    old = dict(hold_monotonic=10., captured_at=10.7,
        requested_base_from_tcp=copy.deepcopy(retry['requested_base_from_tcp']),
        station_gate=dict(hold_monotonic=10., retry_attempt=0, station_deadline=39.))
    new = dict(hold_monotonic=12., captured_at=13.,
        requested_base_from_tcp=copy.deepcopy(retry['requested_base_from_tcp']),
        station_gate=dict(hold_monotonic=12., retry_attempt=1, station_deadline=39.),
        station_retry=copy.deepcopy(retry))
    observations = {'old': {'measurement': old}, 'new': {'measurement': new}}
    if corruption == 'missing_native':
        observations.pop('new')
    elif corruption == 'wrong_endpoint':
        new['requested_base_from_tcp'][0][3] += .001
    elif corruption == 'different_retry':
        new['station_retry']['optical_recovery_attempts'] = 0
    elif corruption == 'renewed_deadline':
        new['station_gate']['station_deadline'] += 10.
    if corruption:
        with pytest.raises(audit.AuditError):
            audit.validate_endpoint_retry_observations(saved, policy, observations)
    else:
        audit.validate_endpoint_retry_observations(saved, policy, observations)
        assert policy['endpoint_retry_bridges'][0]['native_observation_evidence_verified']


@pytest.fixture
def roi_generation_policy(tmp_path):
    """ROI transport consumes no part of the ordinary five-configuration budget."""
    controls = dict(exposure_us=16000., gain_db=12., auto_exposure=False)
    anchor = dict(extent=[.2, .2], offset=[0., 0.], native_path='roi-anchor.json',
        acquisition_key=['owner', 2, 1], seam_point_camera_mm=[0., 0., 150.],
        measured_base_from_camera=np.eye(4).tolist())
    camera = np.eye(4)
    camera[:3, 3] = [.01, .02, 0.]
    transport = dict(roi=[.4, .4, .6, .6], basis='fresh_measured_initial_root', projection_only=True,
        point_base_m=[0., 0., .15], point_camera_mm=[-10., -20., 150.],
        measured_base_from_camera=camera.tolist(), anchor=anchor, hold_monotonic=12.)
    base = dict(limits=dict(min_us=10., max_us=16000., fps_cap_us=16000.),
        requested_us=16000., actual_us=16000., readback=copy.deepcopy(controls),
        configuration_readback=copy.deepcopy(controls))
    brackets = [dict(copy.deepcopy(base), request_id='initial-lock', kind='ae_lock', attempt=0, generation=2,
        measured_hold_monotonic=10., started_monotonic=10.1, completed_monotonic=10.15,
        auto_exposure=dict(reliable=True, state='stable', source_id='camera', read_started=9.9, generation=1)),
        dict(copy.deepcopy(base), request_id='roi-only', kind='roi_transport', attempt=0, generation=3,
            requested_gain_db=12., measured_hold_monotonic=12., started_monotonic=12.1,
            completed_monotonic=12.15, roi_transport=transport)]
    for attempt, exposure in enumerate((16000., 8000., 4000., 16000., 8000.), 1):
        readback = dict(controls, exposure_us=exposure)
        brackets.append(dict(base, request_id=f'optical-{attempt}', kind='gain', attempt=attempt,
            requested_us=exposure, actual_us=exposure, readback=readback,
            configuration_readback=copy.deepcopy(readback), requested_gain_db=12.,
            gain_bounds_db=[0., 48.], gain_ceiling_db=24., generation=attempt+3,
            measured_hold_monotonic=12., started_monotonic=12.2+attempt*.1,
            completed_monotonic=12.25+attempt*.1))
    transitions = [dict(previous_generation=b['generation']-1, generation=b['generation'],
        reason='owned_exposure_bracket', request_id=b['request_id'], bracket_kind=b['kind'],
        bracket_attempt=b['attempt'], requested_exposure_us=b['requested_us'],
        controls=copy.deepcopy(b['readback']), observed_at=(b['started_monotonic']+b['completed_monotonic'])/2)
        for b in brackets]
    saved = dict(binding=dict(source_id='camera', initial_generation=1, final_generation=8, generation=8,
        generation_policy='owned_stripe_AE_per_station_requalification'),
        settings=dict(reflective_surface=True), exposure_brackets=brackets, generation_transitions=transitions)
    anchor_path = tmp_path/'roi-anchor.json'
    save(anchor_path, {'fixture': 'original acquisition is independently hashed by audit_manifest'})
    original = dict(result=dict(status='found', found=True, quality=dict(good=True), seam_point_mm=[0., 0., 150.]),
        camera=np.eye(4), measurement=dict(acquisition_key=anchor['acquisition_key'], hold_monotonic=10., captured_at=10.2))
    current = dict(result=dict(status='invalid', found=False), camera=camera,
        measurement=dict(hold_monotonic=12., captured_at=13., requested_base_from_tcp=camera.tolist(),
            seam_search_roi=copy.deepcopy(transport)))
    observations = {anchor_path.resolve(): original, (tmp_path/'current.json').resolve(): current}
    return saved, observations, tmp_path


def test_roi_transport_is_once_per_hold_and_not_an_optical_attempt(roi_generation_policy):
    saved, observations, directory = roi_generation_policy
    policy = audit.validate_generation_policy(saved, directory)
    audit.validate_roi_transport_observations(policy, observations, directory, np.eye(4))
    assert list(policy['roi_transports']) == [12.]
    assert [b['attempt'] for b in policy['brackets'].values() if b['kind'] == 'gain'] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize('corruption', ['bounds', 'empty_extent', 'nonfinite', 'projection', 'basis',
    'point_algebra', 'hold', 'attempt', 'duplicate_hold', 'changed_controls', 'transition_kind'])
def test_roi_transport_rejects_unproved_projection_or_budget_changes(roi_generation_policy, corruption):
    saved, _, directory = roi_generation_policy
    bracket = saved['exposure_brackets'][1]
    transport = bracket['roi_transport']
    if corruption == 'bounds':
        transport['roi'][0] = -.1
    elif corruption == 'empty_extent':
        transport['roi'][2] = transport['roi'][0]
    elif corruption == 'nonfinite':
        transport['roi'][0] = float('nan')
    elif corruption == 'projection':
        transport['projection_only'] = False
    elif corruption == 'basis':
        transport['basis'] = 'nominal_orbit_measurement'
    elif corruption == 'point_algebra':
        transport['point_camera_mm'][0] += .1
    elif corruption == 'hold':
        transport['hold_monotonic'] += .01
    elif corruption == 'attempt':
        bracket['attempt'] = 1
    elif corruption == 'duplicate_hold':
        duplicate = copy.deepcopy(bracket)
        duplicate.update(request_id='duplicate-roi', started_monotonic=12.16, completed_monotonic=12.17)
        saved['exposure_brackets'].insert(2, duplicate)
    elif corruption == 'changed_controls':
        bracket['readback']['gain_db'] = bracket['configuration_readback']['gain_db'] = 13.
    elif corruption == 'transition_kind':
        saved['generation_transitions'][1]['bracket_kind'] = 'gain'
    with pytest.raises(audit.AuditError):
        audit.validate_generation_policy(saved, directory)


@pytest.mark.parametrize('corruption', ['native_hold', 'native_roi', 'anchor_point', 'future_anchor',
    'camera_endpoint', 'measured_claim'])
def test_roi_transport_is_bound_to_actual_same_hold_native_evidence(roi_generation_policy, corruption):
    saved, observations, directory = roi_generation_policy
    current = list(observations.values())[1]['measurement']
    if corruption == 'native_hold':
        current['hold_monotonic'] += .1
    elif corruption == 'native_roi':
        current['seam_search_roi']['roi'][0] += .01
    elif corruption == 'anchor_point':
        observations[next(iter(observations))]['result']['seam_point_mm'][0] += .1
    elif corruption == 'future_anchor':
        observations[next(iter(observations))]['measurement']['captured_at'] = 20.
    elif corruption == 'camera_endpoint':
        current['requested_base_from_tcp'][0][3] += .0006
    elif corruption == 'measured_claim':
        transport = saved['exposure_brackets'][1]['roi_transport']
        transport['point_base_m'][0] += .001
        transport['point_camera_mm'][0] += 1.
        current['seam_search_roi'] = copy.deepcopy(transport)
    policy = audit.validate_generation_policy(saved, directory)
    with pytest.raises(audit.AuditError):
        audit.validate_roi_transport_observations(policy, observations, directory, np.eye(4))


def test_half_or_quarter_exposure_gain_candidates_require_explicit_reflective_mode(roi_generation_policy):
    saved, _, directory = roi_generation_policy
    saved['exposure_brackets'] = [b for b in saved['exposure_brackets'] if b['kind'] != 'roi_transport']
    # Keep a complete chain after removing the independent ROI-only generation.
    saved['generation_transitions'].pop(1)
    for b, transition in zip(saved['exposure_brackets'][1:], saved['generation_transitions'][1:]):
        b['generation'] -= 1
        transition.update(previous_generation=b['generation']-1, generation=b['generation'])
    saved['binding'].update(final_generation=7, generation=7)
    assert audit.validate_generation_policy(saved, directory)
    for value in (False, 1, None):
        saved['settings']['reflective_surface'] = value
        with pytest.raises(audit.AuditError, match='bounded maximum-FPS exposure'):
            audit.validate_generation_policy(saved, directory)


def test_reflective_gain_still_rejects_non_documented_exposure_and_sixth_attempt(roi_generation_policy):
    saved, _, directory = roi_generation_policy
    invalid = copy.deepcopy(saved)
    invalid['exposure_brackets'][3]['requested_us'] = 6000.
    with pytest.raises(audit.AuditError, match='maximum/half/quarter'):
        audit.validate_generation_policy(invalid, directory)
    invalid = copy.deepcopy(saved)
    invalid['exposure_brackets'][-1]['attempt'] = 6
    with pytest.raises(audit.AuditError, match='five attempts'):
        audit.validate_generation_policy(invalid, directory)


@pytest.fixture
def provisional_coping_archive(archive):
    saved = json.loads(archive.read_text())
    center = next(record for record in saved['path'] if record['direction'] == 0)
    native_path = archive.parent/center['native_path']
    native = json.loads(native_path.read_text())
    m, result = native['measurement'], native['result']
    m.update(coping_phase='seed', station_basis='coped_angular_stations', station_index=0, angular_theta_deg=None)
    # A locator is explicitly outside the later 150 mm sensing-station gate.
    result['seam_point_mm'][2] -= 7.5
    result['camera_from_torch'][2][3] -= 7.5
    for flank in result['flank_lines_mm']:
        for point in flank:
            point[2] -= 7.5
    for point in m['cloud']['xyz_mm']:
        point[2] -= 7.5
    points = np.asarray(m['cloud']['xyz_mm'])
    m['cloud']['bounds_mm'] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    if 'xyz_m' in m['cloud']:
        m['cloud']['xyz_m'] = (points/1000.).tolist()
    camera = np.asarray(m['base_from_camera'])
    torch = np.asarray(result['camera_from_torch'])
    torch[:3, 3] /= 1000.
    center.update(provisional_seed_scout=True, station_basis='coping_seed_scout', station_index=0,
        detector=copy.deepcopy(result),
        seam_point_base_m=(camera[:3, :3] @ (np.asarray(result['seam_point_mm'])/1000.)+camera[:3, 3]).tolist(),
        scan_lines_base_m=(np.asarray(result['flank_lines_mm']) @ camera[:3, :3].T/1000.+camera[:3, 3]).tolist(),
        torch_pose_base_m=(camera @ torch).tolist())
    save(native_path, native)
    for file in archive.parent.glob('measured-view-*.json'):
        if file != native_path:
            file.unlink()
    saved['native_files'] = [entry for entry in saved['native_files'] if entry['path'] == center['native_path']]
    saved['native_files'][0].update(observation_index=1, disposition='coping_scout')
    saved['settings'].update(start_pose='Weld Finder 2 glare-safe', joint_kind='coped_tube', target_arc_deg=180.)
    saved.update(path=[center], status='incomplete', minimum_satisfied=False, coverage_complete=False,
        replay_scope='measured_section_only', minimum_basis='measured_coped_angular_span', seam_length_mm=0.,
        joint_model={'status': 'unobservable', 'model': None}, coping_seed_base_from_camera=camera.tolist(),
        coping_coverage=dict(requested_arc_deg=180., planned_arc_deg=0., observed_arc_deg=0.,
            positive_observed_deg=0., negative_observed_deg=0., measured_station_count=0,
            provisional_seed_scout_count=1, coverage_complete=False))
    save(archive, saved)
    refresh_hashes(archive)
    return archive


def test_provisional_coping_locator_is_file_consistent_but_never_observed_or_replayable(provisional_coping_archive):
    result = audit.audit_manifest(provisional_coping_archive)
    assert result['passed'] and not result['complete'] and not result['replay_eligible']
    assert result['native_observations'] == result['provisional_seed_scout_points'] == 1
    assert result['measured_seam_points'] == result['measured_angular_coverage']['measured_station_count'] == 0
    assert result['measured_angular_coverage']['observed_arc_deg'] == result['seam_length_mm'] == 0.
    assert result['torch_orientation_policy'] == 'provisional_local_bisector_only'


@pytest.mark.parametrize('corruption', ['observed', 'planned', 'counted_station', 'complete', 'inward_policy'])
def test_provisional_coping_locator_cannot_be_promoted_to_an_orbit(provisional_coping_archive, corruption):
    saved = json.loads(provisional_coping_archive.read_text())
    if corruption == 'observed':
        saved['coping_coverage']['observed_arc_deg'] = 1.
    elif corruption == 'planned':
        saved['coping_coverage']['planned_arc_deg'] = 10.
    elif corruption == 'counted_station':
        saved['coping_coverage']['measured_station_count'] = 1
    elif corruption == 'complete':
        saved['status'] = 'complete'
    elif corruption == 'inward_policy':
        saved['path'][0]['torch_orientation_policy'] = 'coped_centerline'
    save(provisional_coping_archive, saved)
    refresh_hashes(provisional_coping_archive)
    with pytest.raises(audit.AuditError, match='Provisional'):
        audit.audit_manifest(provisional_coping_archive)


def test_empty_rejected_native_arrays_are_auditable_without_becoming_measurements(archive):
    saved = json.loads(archive.read_text())
    _, native = native_for(archive, 0)
    cloud = native['measurement']['cloud']
    cloud.update(xyz_mm=[], uv=[], xyz_m=[], count=0, bounds_mm=None)
    native['result'].update(status='invalid', found=False, quality=dict(good=False, reason='No returns'))
    observed = audit.validate_observation(native, saved['binding'], np.asarray(saved['binding']['tcp_from_camera']), set())
    assert observed['result']['status'] == 'invalid'
    record = next(row for row in saved['path'] if row['direction'] == 0)
    with pytest.raises(audit.AuditError, match='no good found seam'):
        audit.validate_world_record(record, observed)
    cloud['bounds_mm'] = [[0., 0., 0.], [1., 1., 1.]]
    with pytest.raises(audit.AuditError, match='fabricated bounds'):
        audit.validate_observation(native, saved['binding'], np.asarray(saved['binding']['tcp_from_camera']), set())
