"""Current frontend boundary: serialized commands, shared snapshots, no device imports."""
import ast
import asyncio
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.app import App
from autoweld.backend import JsonConnection, RobotService
from autoweld.control import Controller
from autoweld.diagnostics import MotionRejection


def test_fresh_runtime_gets_installed_live_gravity_profile(tmp_path):
    import hashlib

    from autoweld.backend import gravity_record
    from autoweld.models import tool_model

    fingerprint = hashlib.sha256(Path(tool_model()).read_bytes()).hexdigest()
    profile = gravity_record(tmp_path, 'gravity-config.json', fingerprint)
    payload = gravity_record(tmp_path, 'gravity-payload.json', fingerprint)
    tuning = gravity_record(tmp_path, 'gravity-tuning.json', fingerprint)
    assert profile['factors'] == [1., 1., 1., 1., 1., 1.1]
    assert payload['mass_kg'] == pytest.approx(.45359237)
    assert not payload['validated']  # Physical payload fit was rejected.
    assert tuning['metrics']['validation_probes'] == 3
    assert tuning['status'] == 'limited'
    assert tuning['metrics']['wide_candidate_promoted'] is False


def test_matching_runtime_override_wins_over_installed_profile(tmp_path):
    from autoweld.backend import gravity_record

    override = {'model_fingerprint': 'local-model', 'factors': [1.2] * 6}
    (tmp_path / 'gravity-config.json').write_text(json.dumps(override))
    assert gravity_record(tmp_path, 'gravity-config.json', 'local-model') == override


def test_gravity_records_never_cross_model_fingerprints(tmp_path):
    import hashlib

    from autoweld.backend import gravity_record
    from autoweld.models import tool_model

    (tmp_path / 'gravity-config.json').write_text(json.dumps({
        'model_fingerprint': 'another-model', 'factors': [2.] * 6}))
    fingerprint = hashlib.sha256(Path(tool_model()).read_bytes()).hexdigest()
    assert gravity_record(tmp_path, 'gravity-config.json', fingerprint)['factors'][-1] == 1.1
    assert gravity_record(tmp_path, 'gravity-config.json', 'unknown-model') is None


def test_frontend_imports_no_robot_or_camera_processing():
    source = Path('autoweld/app.py').read_text()
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or '')
    forbidden = ('i2rt', 'mujoco', 'cv2', 'autoweld.control', 'autoweld.camera', 'autoweld.motion')
    assert not any(module.startswith(forbidden) for module in modules)


def test_connections_share_serialized_state_without_devices():
    service = SimpleNamespace(events=queue.Queue(), snapshot_lock=threading.Lock(), snapshot_json='{"mode":"Hold"}')
    one, two = JsonConnection(service), JsonConnection(service)
    assert json.loads(one.snapshot()) == json.loads(two.snapshot()) == {'mode': 'Hold'}
    one.send('{"action":"mode","mode":"IK"}')
    request = service.events.get_nowait()
    assert isinstance(request.pop('_service_queued_monotonic'), float)
    assert request == {'action': 'mode', 'mode': 'IK'}
    with pytest.raises(ValueError):
        two.send('[]')


def test_gizmo_only_sends_released_targets_in_ik():
    sent = []
    app = object.__new__(App)
    app.connection = SimpleNamespace(send=sent.append)
    app.state = {'mode': 'IK', 'closed': False}
    target = SimpleNamespace(position=[.1, .2, .3], wxyz=[1, 0, 0, 0])
    for phase in ('start', 'update', 'end'):
        asyncio.run(app._target_changed(SimpleNamespace(phase=phase, target=target, client_id='operator')))
    assert len(sent) == 1
    assert json.loads(sent[0]) == {'action': 'pose', 'position': [.1, .2, .3], 'wxyz': [1, 0, 0, 0]}
    app.state['mode'] = 'Hold'
    asyncio.run(app._target_changed(SimpleNamespace(phase='end', target=target, client_id='operator')))
    assert len(sent) == 1


@pytest.fixture
def service(tmp_path):
    service = object.__new__(RobotService)
    service.runtime = tmp_path
    service.ctrl = Controller(FakeRobot(), np.tile([-3, 3], (6, 1)))
    service.pending_pose = None
    service.request_status = None
    service.exit_after_disable = False
    service.saved_poses = {}
    service.saved_pose_path = tmp_path / 'poses.json'
    service.tcp_pose = lambda: np.eye(4)
    service.camera = None
    service.calibration = None
    return service


def test_saved_pose_round_trip_uses_backend_controller(service):
    service.ctrl.robot.q[0] = .13
    service.dispatch({'action': 'pose_save', 'name': 'Camera baseline'})
    assert json.loads(service.saved_pose_path.read_text())['Camera baseline']['q'][0] == .13
    service.ctrl.robot.q[0] = .2
    service.dispatch({'action': 'pose_recall', 'name': 'Camera baseline'})
    assert service.ctrl.target[0] == .13


def test_backend_queues_latest_target_and_control_cancels_it(service):
    service.ctrl.motion_plan = object()
    for x in (.1, .2):
        result = service.handle_request({'action': 'pose', 'position': [x, 0, 0], 'rpy': [0, 0, 0]})
        assert result['status'] == 'queued'
    assert service.pending_pose[0]['position'][0] == .2
    service.handle_request({'action': 'mode', 'mode': 'Hold'})
    assert service.pending_pose is None


def test_backend_rejection_retains_diagnostics_in_json(service):
    def reject(_):
        raise MotionRejection('endpoint_validation', 'self_collision', 'Blocked endpoint')
    service.dispatch = reject
    result = service.handle_request({'action': 'pose'})
    assert not result['ok']
    assert result['diagnostic']['code'] == 'self_collision'
    assert json.loads((service.runtime/'command-events.jsonl').read_text())['diagnostic'] == result['diagnostic']


@pytest.mark.parametrize('active_status', ['moving', 'hunting'])
def test_manual_hold_cancels_calibration_before_handling_command(service, active_status):
    actions = []
    service.calibration = SimpleNamespace(state=lambda: {'status': active_status},
        dispatch=lambda req: actions.append(req['action']))
    service.dispatch = lambda req: actions.append(req['action'])
    service.handle_request({'action': 'mode', 'mode': 'Hold'})
    assert actions == ['calibration_cancel', 'mode']


def test_occupied_port_preflight_closes_probe_without_hardware(monkeypatch):
    import socket

    from autoweld.backend import check_server_port
    observed = []
    class Probe:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            observed.append('closed')
        def setsockopt(self, *args):
            pass
        def bind(self, address):
            observed.append(address)
            raise OSError('Address already in use')
    monkeypatch.setattr(socket, 'socket', lambda *args: Probe())
    with pytest.raises(RuntimeError, match='robot was not connected'):
        check_server_port('127.0.0.1', 8080)
    assert observed == [('127.0.0.1', 8080), 'closed']


def test_disabled_display_uses_saved_six_axis_state_without_can(tmp_path):
    from autoweld.backend import disabled_display_robot
    from autoweld.models import compiled_model, tool_model

    arm = [.01, .02, .03, .04, .05, .06]
    (tmp_path / 'rest.json').write_text(json.dumps({'arm_rad': arm}))
    robot = disabled_display_robot(tmp_path, tool_model(), compiled_model())
    try:
        np.testing.assert_allclose(robot.get_joint_pos(), arm)
        assert robot.num_dofs() == 6
        assert robot._model is compiled_model()
        assert not hasattr(robot, 'motor_chain')
    finally:
        robot.close()
    (tmp_path / 'rest.json').write_text(json.dumps({'arm_rad': [0] * 7}))
    with pytest.raises(ValueError, match='six-axis'):
        disabled_display_robot(tmp_path, tool_model(), compiled_model())


def test_gravity_configure_persists_matching_model_profile(service):
    service.model_fingerprint = 'installed-camera-model'
    seen = []
    def configure(factors, model):
        seen.append((factors, model))
        service.ctrl.gravity_factors = np.array(factors)
        service.ctrl.gravity_model = model
    service.ctrl.configure_gravity = configure
    factors = [1., 1.02, .99, 1., 1., 1.]
    result = service.dispatch({'action': 'gravity_configure', 'factors': factors, 'model': 'sdk'})
    assert seen == [(factors, 'sdk')]
    assert result['gravity_factors'] == factors
    assert json.loads((service.runtime / 'gravity-config.json').read_text()) == {
        'factors': factors, 'model': 'sdk', 'model_fingerprint': 'installed-camera-model'}


def test_payload_configure_rejects_motion_before_sdk_write(service):
    writes = []
    service.ctrl.robot.configure_payload = lambda *args: writes.append(args)
    service.ctrl.motion_plan = object()
    with pytest.raises(RuntimeError, match='Stop robot motion'):
        service.dispatch({'action': 'gravity_payload', 'mass_kg': .45, 'com_flange_m': [0., 0., -.05]})
    assert not writes
    assert not (service.runtime / 'gravity-payload.json').exists()


def test_payload_and_tuning_metadata_are_serialized_with_model(service):
    service.model_fingerprint = 'installed-camera-model'
    writes = []
    service.ctrl.robot.configure_payload = lambda *args: writes.append(args)
    service.dispatch({'action': 'gravity_payload', 'mass_kg': .45, 'com_flange_m': [0., 0., -.05],
        'source': 'measured fitting run', 'validated': True})
    assert writes == [(.45, [0., 0., -.05])]
    payload = json.loads((service.runtime / 'gravity-payload.json').read_text())
    assert payload['model_fingerprint'] == 'installed-camera-model'
    assert payload['validated']
    service.dispatch({'action': 'gravity_tuning_status', 'status': 'validated',
        'message': 'Measured residual accepted', 'metrics': {'rms_nm': .02}})
    result = service.gravity_telemetry()
    assert result['payload']['mass_kg'] == .45
    assert result['tuning']['metrics'] == {'rms_nm': .02}


def test_board_settings_sync_preserves_pending_edits_until_backend_changes():
    app = object.__new__(App)
    app.board_columns = SimpleNamespace(value=5)
    app.board_rows = SimpleNamespace(value=7)
    app.square_mm = SimpleNamespace(value=4.)
    settings = {'columns': 5, 'rows': 7, 'square_mm': 4.}
    app._sync_board_settings(settings)
    app.board_columns.value = 9
    app._sync_board_settings(settings)
    assert app.board_columns.value == 9
    app._sync_board_settings({'columns': 6, 'rows': 8, 'square_mm': 4.5})
    assert (app.board_columns.value, app.board_rows.value, app.square_mm.value) == (6, 8, 4.5)


def test_gravity_ui_shows_actual_mode_and_transition_error():
    app = object.__new__(App)
    app.state = {'controller': 'MIT', 'arm_control_modes': ['MIT'] * 6,
        'mode_transition_error': 'Feedback verification failed', 'gravity_factors': [1.] * 6,
        'gravity_model': 'sdk', 'gravity_tuning': {'status': 'validated', 'message': 'Measured fit accepted'}}
    app._gravity_config_signature = None
    app.motor_mode_md = SimpleNamespace(content='')
    app.gravity_status_md = SimpleNamespace(content='')
    app.gravity_torque_md = SimpleNamespace(content='')
    app.gravity_model_select = SimpleNamespace(value='linear')
    app.gravity_inputs = [SimpleNamespace(value=0) for _ in range(6)]
    app.gravity_button = SimpleNamespace(disabled=False)
    app._render_gravity()
    assert 'MIT' in app.motor_mode_md.content
    assert 'Feedback verification failed' in app.motor_mode_md.content
    assert 'Measured fit accepted' in app.gravity_status_md.content
    assert app.gravity_model_select.value == 'sdk'


def test_gravity_regressor_uses_shared_model_and_matched_measurements(service):
    from autoweld.models import compiled_model
    service.planner = SimpleNamespace(model=compiled_model())
    now = 100.
    service.ctrl.robot.get_observations = lambda: {
        'joint_pos': np.arange(6) * .01, 'joint_vel': np.zeros(6),
        'joint_eff': np.arange(6) * .02, 'feedback_timestamp': now,
        'joint_timestamps': np.full(6, now)}
    result = service.dispatch({'action': 'gravity_telemetry'})
    np.testing.assert_allclose(result['q_rad'], np.arange(6) * .01)
    np.testing.assert_allclose(result['reported_effort_nm'], np.arange(6) * .02)
    assert np.asarray(result['regressor']).shape == (6, 4)
    assert np.asarray(result['baseline_nm']).shape == (6,)
    assert service.gravity_data.model is compiled_model()
    scratch = service.gravity_data
    service.dispatch({'action': 'gravity_telemetry'})
    assert service.gravity_data is scratch
    assert result['feedback_timestamp'] == now
    assert not result['feedback_fresh']


def test_partial_checkerboard_display_uses_backend_count_and_offset():
    board = {'found': False, 'partial_found': True, 'corner_count': 12,
        'centroid_px': [412., 203.], 'offset_px': [92., -37.]}
    text = App._board_detection_text(board)
    assert 'partial checkerboard' in text
    assert '12 corners' in text
    assert '(+92.0, -37.0) px' in text
    assert 'full checkerboard' not in text
    assert App._board_detection_text({'found': False, 'corner_count': 0, 'offset_px': None}) == 'no checkerboard corners'


def test_hunt_progress_displays_reported_centroid_without_computing_direction():
    calibration = {'status': 'hunting', 'hunt': {'phase': 'baseline', 'steps': 3,
        'max_steps': 16, 'step_mm': 5, 'corner_count': 12, 'centroid_px': [412., 203.],
        'offset_px': [92., -37.], 'reacquiring': False, 'motion_active': True}}
    text = App._hunt_progress_text(calibration)
    assert 'Baseline checkerboard search' in text
    assert 'step 3 / 16 · 5 mm increments' in text
    assert 'next observation after settling' in text
    assert 'centroid (412.0, 203.0) px' in text
    assert 'offset from image center (+92.0, -37.0) px' in text


def test_hunt_without_corners_reports_undefined_direction_and_clears_when_cancelled():
    calibration = {'status': 'hunting', 'hunt': {'phase': 'capture', 'steps': 4,
        'max_steps': 16, 'corner_count': 0, 'centroid_px': None, 'offset_px': None,
        'reacquiring': True}}
    text = App._hunt_progress_text(calibration)
    assert 'Capture pose checkerboard search' in text
    assert 'Centroid direction unavailable · bounded reacquisition search' in text
    assert 'offset from image center' not in text
    calibration['status'] = 'cancelled'
    assert App._hunt_progress_text(calibration) == ''


@pytest.mark.parametrize('status', ['hunting', 'hunt_moving', 'hunt_settling'])
def test_hunt_camera_panel_prevents_duplicate_start_and_keeps_cancel_available(status):
    app = object.__new__(App)
    app.state = {'closed': False, 'camera': {'enabled': True, 'fresh': True, 'fps': 60.,
        'board': {'found': False, 'partial_found': True, 'corner_count': 8,
                  'centroid_px': [100., 200.], 'offset_px': [-220., -40.]},
        'counts': {'intrinsic': 2, 'handeye': 0}},
        'calibration': {'status': status, 'message': 'Centering checkerboard', 'index': 2,
            'total': 24, 'hunt': {'phase': 'capture', 'steps': 2, 'max_steps': 16,
                'corner_count': 8, 'centroid_px': [100., 200.], 'offset_px': [-220., -40.]}}}
    app._camera_enabled, app._camera_sequence = False, None
    app.camera_on = SimpleNamespace(value=False)
    app.camera_md = SimpleNamespace(content='')
    app.calibration_md = SimpleNamespace(content='')
    app.hunt_md = SimpleNamespace(content='')
    app.auto_button = SimpleNamespace(disabled=False)
    app.capture_button = SimpleNamespace(disabled=False)
    app.solve_button = SimpleNamespace(disabled=False)
    app.cancel_calibration_button = SimpleNamespace(disabled=True)
    app._render_camera()
    assert 'partial checkerboard · 8 corners' in app.camera_md.content
    assert 'Centering checkerboard' in app.calibration_md.content
    assert 'step 2 / 16' in app.hunt_md.content
    assert app.auto_button.disabled and app.capture_button.disabled and app.solve_button.disabled
    assert not app.cancel_calibration_button.disabled


def test_apply_speed_limits_sends_one_atomic_request_above_previous_joint_limit():
    sent = []
    app = object.__new__(App)
    app.connection = SimpleNamespace(send=sent.append)
    app.state = {'motion_limits': {'linear_velocity': 3., 'joint_velocity': .35,
        'linear_acceleration': .15, 'joint_jerk': 5.}, 'position_speed_limit_rad_s': .15}
    app.speed = SimpleNamespace(value=4.)
    app.joint_speed = SimpleNamespace(value=.5)
    app._apply_speed_limits()
    assert [json.loads(message) for message in sent] == [
        {'action': 'speed_limits', 'linear_velocity': 4., 'joint_velocity': .5}]
    assert app.state['motion_limits']['joint_velocity'] == .35
    assert app.state['motion_limits']['linear_acceleration'] == .15


def test_speed_limit_telemetry_and_mode_changes_preserve_pending_operator_edits():
    app = object.__new__(App)
    app.speed = SimpleNamespace(value=0.)
    app.joint_speed = SimpleNamespace(value=0.)
    app.state = {'mode': 'Hold', 'motion_limits': {'linear_velocity': 3.}, 'position_speed_limit_rad_s': .15}
    app._sync_speed_limits()
    assert (app.speed.value, app.joint_speed.value) == (3., .15)
    app.speed.value, app.joint_speed.value = 4., .5
    app.state['mode'] = 'IK'
    app._sync_speed_limits()
    assert (app.speed.value, app.joint_speed.value) == (4., .5)
    # An unrelated backend update must also preserve values awaiting Apply.
    app.state['motion_limits']['linear_velocity'] = 2.
    app._sync_speed_limits()
    assert (app.speed.value, app.joint_speed.value) == (4., .5)
    app.state['motion_limits']['linear_velocity'] = 4.
    app.state['position_speed_limit_rad_s'] = .5
    app._sync_speed_limits()
    assert (app.speed.value, app.joint_speed.value) == (4., .5)
    # Once these values are confirmed, future backend changes are reflected.
    app.state['position_speed_limit_rad_s'] = .6
    app._sync_speed_limits()
    assert (app.speed.value, app.joint_speed.value) == (4., .6)


def cone_preview_fixture():
    start = np.array([.33, -.14, .21])
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    radius = .150 * np.tan(np.deg2rad(7.5))
    azimuth = np.arange(25) * 2 * np.pi / 25
    ring = start + np.column_stack((radius*np.cos(azimuth), radius*np.sin(azimuth),
                                   np.zeros(25))) @ rotation.T
    apex = start + rotation[:, 2] * .150
    geometry = {'axis_start_m': start.tolist(), 'apex_m': apex.tolist(),
        'cone_ring_m': ring.tolist(), 'distance_m': .150, 'radius_m': radius,
        'opening_angle_deg': 15.}
    calls, removed = [], []
    def add(kind, name, **kwargs):
        calls.append((kind, name, kwargs))
        return SimpleNamespace(remove=lambda: removed.append(name))
    app = object.__new__(App)
    app.server = SimpleNamespace(scene=SimpleNamespace(
        add_line_segments=lambda name, **kwargs: add('lines', name, **kwargs),
        add_point_cloud=lambda name, **kwargs: add('points', name, **kwargs)))
    return app, geometry, calls, removed


def test_cone_preview_has_25_discrete_points_and_only_axis_spokes_and_adjacent_edges():
    app, geometry, calls, _ = cone_preview_fixture()
    app._render_cone(geometry)
    assert [kind for kind, _, _ in calls] == ['lines', 'lines', 'points']
    axis, wire, markers = [kwargs for _, _, kwargs in calls]
    ring, apex = np.asarray(geometry['cone_ring_m']), np.asarray(geometry['apex_m'])
    assert axis['points'].shape == (1, 2, 3)
    assert wire['points'].shape == (50, 2, 3)
    assert markers['points'].shape == (25, 3)
    assert len(np.unique(markers['points'], axis=0)) == 25
    np.testing.assert_allclose(markers['points'], ring)
    np.testing.assert_allclose(wire['points'][:25, 0], np.broadcast_to(apex, (25, 3)))
    np.testing.assert_allclose(wire['points'][:25, 1], ring)
    np.testing.assert_allclose(wire['points'][25:, 0], ring)
    np.testing.assert_allclose(wire['points'][25:, 1], np.roll(ring, -1, axis=0))
    assert np.linalg.norm(axis['points'][0, 1]-axis['points'][0, 0]) == pytest.approx(.150)
    np.testing.assert_allclose(np.linalg.norm(ring-np.asarray(geometry['axis_start_m']), axis=1), .019747875, atol=1e-9)
    assert np.max(np.linalg.norm(np.diff(wire['points'], axis=1)[:, 0], axis=1)) < .152
    assert axis['thickness_units'] == wire['thickness_units'] == 'screen'
    assert axis['thickness'] <= 2 and wire['thickness'] <= 2
    assert markers['point_size'] == .003
    assert markers['precision'] == 'float32'


def test_cone_preview_reuses_geometry_then_removes_stale_handles_on_reset():
    app, geometry, calls, removed = cone_preview_fixture()
    app._render_cone(geometry)
    app._render_cone(json.loads(json.dumps(geometry)))
    assert len(calls) == 3 and not removed
    moved = {**geometry, 'apex_m': (np.asarray(geometry['apex_m']) + [.005, 0, 0]).tolist()}
    app._render_cone(moved)
    assert len(calls) == 6 and len(removed) == 3
    app._render_cone(None)
    assert len(removed) == 6
    assert app._cone_handles == []
    app._render_cone(None)
    assert len(removed) == 6


def test_cone_preview_closing_vertex_is_not_an_extra_discrete_point():
    app, geometry, calls, _ = cone_preview_fixture()
    geometry['cone_ring_m'].append(geometry['cone_ring_m'][0])
    app._render_cone(geometry)
    assert calls[-1][2]['points'].shape == (25, 3)
    assert calls[1][2]['points'].shape == (50, 2, 3)


def test_shutdown_button_sends_only_json_shutdown_request():
    sent = []
    app = object.__new__(App)
    app.connection = SimpleNamespace(send=sent.append)
    app._request_shutdown(SimpleNamespace(client_id='operator'))
    assert [json.loads(message) for message in sent] == [{'action': 'shutdown'}]


@pytest.mark.parametrize('other_state', [
    {'disable_pending': True}, {'motion_fault': {'reason': 'Mode fault'}},
    {'motion_active': True}, {'cooldown_active': True}, {'closed': True},
    {'disable_pending': True, 'fault': 'Return failed', 'cooldown_active': True},
])
def test_shutdown_button_remains_enabled_while_disable_fault_busy_or_cooldown(other_state):
    app = object.__new__(App)
    app.state = {**other_state, 'shutdown': {'requested': True, 'active': True, 'phase': 'waiting',
        'error': None, 'retryable': False, 'rest_verified': False}}
    app.shutdown_button = SimpleNamespace(disabled=True, label='')
    app.shutdown_md = SimpleNamespace(content='')
    app._render_shutdown()
    assert not app.shutdown_button.disabled
    assert app.shutdown_button.label == 'Shutdown service at rest'
    assert 'Waiting for fresh feedback and a quiet arm' in app.shutdown_md.content


def test_shutdown_blocked_error_is_visible_and_clears_after_retry():
    app = object.__new__(App)
    app.shutdown_button = SimpleNamespace(disabled=True, label='')
    app.shutdown_md = SimpleNamespace(content='')
    app.state = {'shutdown': {'requested': True, 'active': True, 'phase': 'blocked',
        'error': 'SDK mode transition failed', 'retryable': True, 'rest_verified': False}}
    app._render_shutdown()
    assert app.shutdown_button.label == 'Retry shutdown at rest'
    assert not app.shutdown_button.disabled
    assert '**Shutdown error:** SDK mode transition failed' in app.shutdown_md.content
    app.state['shutdown'].update(phase='returning', error=None, retryable=False)
    app._render_shutdown()
    assert app.shutdown_button.label == 'Shutdown service at rest'
    assert 'Returning to captured rest' in app.shutdown_md.content
    assert 'SDK mode transition failed' not in app.shutdown_md.content
    assert 'Shutdown error' not in app.shutdown_md.content
    app.state['shutdown'].update(phase='disabled', active=False, rest_verified=True)
    app._render_shutdown()
    assert 'Motors disabled at verified rest; stopping the service' in app.shutdown_md.content
    app.state = {'shutdown': {'requested': False, 'active': False, 'phase': 'idle', 'error': None}}
    app._render_shutdown()
    assert app.shutdown_md.content == ''


def test_motion_limit_controls_default_to_three_tcp_two_joint_and_allow_sdk_joint_cap():
    class Widget:
        def __init__(self, label, value=None, **settings):
            self.label, self.value, self.settings = label, value, settings
            self.clicks = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def on_click(self, callback):
            self.clicks.append(callback)
            return callback
        def on_update(self, callback):
            return callback
    class Gui:
        def __init__(self):
            self.markdown = []
        def add_markdown(self, content):
            self.markdown.append(content)
            return Widget(content)
        def __getattr__(self, name):
            assert name.startswith('add_')
            def add(*args, **settings):
                label = settings.pop('label', args[0] if args else '')
                value = args[1] if len(args) > 1 else settings.pop('initial_value', None)
                return Widget(label, value, **settings)
            return add
    sent = []
    app = object.__new__(App)
    gui = Gui()
    app.server = SimpleNamespace(gui=gui)
    app.connection = SimpleNamespace(send=sent.append)
    app._build_gui()
    assert (app.speed.value, app.joint_speed.value) == (3., 2.)
    assert app.joint_speed.settings['max'] == 3.
    assert any('TCP speed default 3 m/s; joint speed default 2 rad/s' in text for text in gui.markdown)
    assert len(app.limit_button.clicks) == 1
    app.limit_button.clicks[0](SimpleNamespace(client_id='operator'))
    assert [json.loads(message) for message in sent] == [
        {'action': 'speed_limits', 'linear_velocity': 3., 'joint_velocity': 2.}]
    # Existing saved/confirmed settings remain authoritative until edited.
    app.state = {'motion_limits': {'linear_velocity': 2.5}, 'position_speed_limit_rad_s': 1.5}
    app._sync_speed_limits()
    assert (app.speed.value, app.joint_speed.value) == (2.5, 1.5)
