"""Cached command summaries never retain another native cloud or image payload."""

import copy
import json
from threading import Lock
from types import SimpleNamespace

import pytest

from autoweld.backend import RobotService, compact_telemetry


def payload():
    points = [[i*.000001, i*.000002, .140] for i in range(10000)]
    pose = [[1., 0., 0., .02], [0., 1., 0., -.3], [0., 0., 1., .2], [0., 0., 0., 1.]]
    cloud = {'points_base_m': points, 'points_world_m': points, 'xyz_m': points,
             'uv': [[i, i+1] for i in range(10000)], 'strength': [100.]*10000, 'width_px': [3.]*10000,
             'count': 10000, 'total_count': 162000, 'coordinate_frame': 'robot_base', 'units': 'm',
             'scan_id': 'fixture:resumed', 'source_id': 'fixture:camera', 'calibration_id': 'fixture:laser',
             'optical_intrinsics_id': 'fixture:optics', 'acquisition_session_id': 'fixture:session',
             'sequence': 42, 'generation': 4, 'read_started': 100., 'read_completed': 100.01}
    return {'status': 'baseline_settling', 'busy': True, 'visited': 324, 'captures': 197,
        'point_count': 162000, 'cloud': cloud, 'camera': {'frame_jpeg_base64': 'J'*500000,
            'board': {'corner_count': 35, 'corners': [[1., 2.]]*10000,
                      'corners_native_px': [[1., 2.]]*10000, 'raw_corners_native_px': [[1., 2.]]*10000},
            'laser': {'preview_jpeg_base64': 'P'*500000, 'result': {'cloud': cloud}}},
        'geometry': {'outline_base_m': [[0., 0., 0.]]*5, 'frame': pose,
                     'corners': [[0., 0., 0.], [.1, 0., 0.]]},
        'diagnostic': {'requested_pose': pose, 'checks': {'position': {'error': .0002, 'limit': .0005}},
                       'joint_peak_velocity_rad_s': [1.]*6},
        'quality': {'rms_mm': .2, 'count': 460, 'strength': 100., 'width_px': 3.},
        'result': [{'laser': {'cloud': cloud}}]}


@pytest.fixture
def service(tmp_path):
    service = object.__new__(RobotService)
    service.runtime = tmp_path
    service.pending_pose = None
    service.request_status = None
    service.exit_after_disable = False
    service.calibration = service.laser = None
    service.host_guard = None
    service.ctrl = SimpleNamespace(sdk_status=lambda: {'required': False})
    service.snapshot_lock = Lock()
    service.snapshot_json = '{}'
    return service


def test_recursive_summary_preserves_metadata_geometry_and_diagnostics_without_mutating_response():
    source = payload()
    before = json.dumps(source, sort_keys=True)
    compact = compact_telemetry(source)
    assert json.dumps(source, sort_keys=True) == before
    assert len(json.dumps(compact)) < 5000
    for key in ('status', 'busy', 'visited', 'captures', 'point_count', 'geometry', 'diagnostic', 'quality'):
        assert compact[key] == source[key]
    for key in ('count', 'total_count', 'coordinate_frame', 'units', 'scan_id', 'source_id', 'calibration_id',
                'optical_intrinsics_id', 'acquisition_session_id', 'sequence', 'generation', 'read_started', 'read_completed'):
        assert compact['cloud'][key] == source['cloud'][key]
    assert compact['camera']['board'] == {'corner_count': 35}
    assert 'frame_jpeg_base64' not in compact['camera']
    assert compact['result'][0]['laser']['cloud'] == compact['cloud']
    assert len(source['cloud']['points_base_m']) == 10000


def test_scan_resume_full_one_time_response_is_retained_but_cache_and_command_journal_are_bounded(service):
    source = payload()
    service.scan = SimpleNamespace(busy=False, dispatch=lambda _request: source)
    response = service.handle_request({'action': 'scan_resume', '_id': 'fixture-request'})
    assert response['ok'] and response['result'] is source
    assert len(response['result']['cloud']['points_base_m']) == 10000
    assert len(response['result']['camera']['frame_jpeg_base64']) == 500000
    assert len(json.dumps(service.request_status)) < 6000
    event = json.loads((service.runtime/'command-events.jsonl').read_text())
    assert event['action'] == 'scan_resume' and event['request']['_id'] == 'fixture-request'
    assert event['result']['point_count'] == 162000
    assert event['result']['geometry'] == source['geometry']
    assert 'points_base_m' not in event['result']['cloud']
    assert (service.runtime/'command-events.jsonl').stat().st_size < 6500
    response['result']['quality']['rms_mm'] = .8
    assert service.request_status['result']['quality']['rms_mm'] == .2


def test_publish_has_only_current_preview_in_ui_and_keeps_periodic_disk_status_small(service):
    source = payload()
    service.scan = SimpleNamespace(busy=False, dispatch=lambda _request: source)
    service.handle_request({'action': 'scan_resume'})
    preview = {'points_base_m': source['cloud']['points_base_m'], 'count': 10000, 'revision': 8}
    service.state = lambda: {'last_request': service.request_status,
                            'scan': {'cloud': copy.deepcopy(preview), 'point_count': 162000},
                            'camera': {'frame_jpeg_base64': 'current-live-frame'}}
    for _ in range(3):
        service.publish()
        ui = json.loads(service.snapshot_json)
        disk = json.loads((service.runtime/'status.json').read_text())
        assert len(ui['scan']['cloud']['points_base_m']) == 10000  # Actual current visual remains available.
        assert ui['camera']['frame_jpeg_base64'] == 'current-live-frame'
        assert 'frame_jpeg_base64' not in disk['camera']
        assert 'points_base_m' not in ui['last_request']['result']['cloud']
        assert 'points_base_m' not in disk['scan']['cloud']
        assert 'points_base_m' not in disk['last_request']['result']['cloud']
        assert disk['scan']['point_count'] == 162000
        assert ui['last_request']['result']['geometry'] == source['geometry']
        assert (service.runtime/'status.json').stat().st_size < 8000
        assert service.snapshot_json.count('points_base_m') == 1
    response_points = source['cloud']['points_base_m']
    assert len(response_points) == 10000
    assert response_points is source['cloud']['xyz_m']


def test_publish_compacts_old_unsummarized_cached_requests_before_ui_serialization(service):
    source = payload()
    service.request_status = {'action': 'scan_resume', 'result': source}
    service.state = lambda: {'last_request': service.request_status}
    service.publish()
    assert len(service.snapshot_json) < 6000
    assert (service.runtime/'status.json').stat().st_size < 8000
    assert 'points_base_m' not in service.request_status['result']['cloud']
    assert len(source['cloud']['points_base_m']) == 10000


def test_recursive_disk_laser_filter_keeps_nested_results_small_without_touching_live_source():
    source = {'cloud': payload()['cloud'], 'result': payload(), 'frame_jpeg_base64': 'raw',
              'quality': {'rms_mm': .2}}
    compact = RobotService._laser_telemetry(source)
    assert len(json.dumps(compact)) < 6000
    assert compact['quality'] == {'rms_mm': .2}
    assert len(source['cloud']['points_base_m']) == 10000
    assert compact['result']['geometry'] == source['result']['geometry']
