"""Laser panel is a JSON client; no camera owner, SDK, model or server is created."""
import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.app import App, LaserPlaneCalibrationUI


class Widget:
    def __init__(self, label, value=None, **settings):
        self.label, self.value, self.settings = label, value, settings
        self.clicks, self.updates = [], []
        self.content, self.disabled, self.visible = '', False, True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def on_click(self, callback):
        self.clicks.append(callback)
        return callback
    def on_update(self, callback):
        self.updates.append(callback)
        return callback
    def click(self):
        for callback in self.clicks:
            callback(SimpleNamespace(client_id='operator'))
    def update(self, value, client_id='operator'):
        self.value = value
        for callback in self.updates:
            callback(SimpleNamespace(client_id=client_id))


class Gui:
    def __init__(self):
        self.widgets = []
    def __getattr__(self, method):
        assert method.startswith('add_')
        def add(*args, **settings):
            label = settings.pop('label', args[0] if args else '')
            value = args[1] if len(args) > 1 else settings.pop('initial_value', None)
            if value is None and settings.get('options'):
                value = settings['options'][0]
            widget = Widget(label, value, **settings)
            widget.kind = method
            self.widgets.append(widget)
            return widget
        return add


class Scene:
    def __init__(self):
        self.calls, self.removed = [], []
    def add_point_cloud(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return SimpleNamespace(remove=lambda: self.removed.append(name))


def app_fixture():
    sent, gui, scene = [], Gui(), Scene()
    app = object.__new__(App)
    app.state = {}
    app.server = SimpleNamespace(gui=gui, scene=scene)
    app.connection = SimpleNamespace(send=lambda message: sent.append(json.loads(message)))
    app._camera_enabled = app._camera_sequence = app._board_settings_signature = None
    app._build_gui()
    return app, sent, gui, scene


def state_fixture():
    return {'closed': False, 'camera': {'enabled': True, 'fresh': True, 'sequence': 1,
        'laser': {'preview_enabled': True, 'captures': 4,
            'quality': {'rms_mm': .08, 'views': 4, 'span_ratio': .07, 'inliers': 450, 'total': 500},
            'pointcloud_status': {'auto_exposure_status': {'state': 'stable', 'stripe_contrast': 120.,
                'saturated_fraction': .02, 'message': 'Stripe visible'},
                'auto_exposure': {'enabled': True}, 'noise': {'enabled': True}}}},
        'laser': {'status': 'idle', 'busy': False, 'message': 'Ready',
            'cloud': {'points_base_m': [[.2, .1, .3], [.21, .1, .3]],
                'points_j6_m': [[99., 99., 99.]], 'sequence': 1,
                'frame': 'j6_tool', 'units': 'm', 'status': 'ready'}}}


def test_separate_selectable_panel_shares_exactly_one_image_and_stream_owner():
    app, sent, gui, _ = app_fixture()
    assert not sent
    assert len([widget for widget in gui.widgets if widget.kind == 'add_image']) == 1
    assert app.camera_calibration_folder.visible and not app.laser_ui.folder.visible
    app.calibration_panel_select.update('Laser plane calibration')
    assert app.laser_ui.folder.visible and not app.camera_calibration_folder.visible
    assert not sent  # Choosing a panel never opens the camera.
    app.laser_ui.stream.update(True)
    assert sent == [{'action': 'camera_stream', 'enabled': True}]
    app.state = state_fixture()
    app._render_camera()
    app.laser_ui.render(app.state, app.server.scene)
    assert app.camera_on.value and app.laser_ui.stream.value
    assert len(sent) == 1
    app.camera_on.update(False)
    assert sent[-1] == {'action': 'camera_stream', 'enabled': False}
    assert not any(widget.label in ('Laser on', 'Laser off', 'Enable laser', 'Disable laser') for widget in gui.widgets
                   if isinstance(widget.label, str))


def test_selected_calibration_panel_displays_its_service_cone():
    app, _, _, _ = app_fixture()
    shown = []
    app._render_cone = shown.append
    app.state = {'camera': {}, 'calibration': {'geometry': {'kind': 'intrinsic'}},
                 'laser': {'geometry': {'kind': 'laser'}}}
    app._render_camera()
    app.calibration_panel_select.update('Laser plane calibration')
    app._render_camera()
    assert shown == [{'kind': 'intrinsic'}, {'kind': 'laser'}]


@pytest.mark.parametrize('button,action,payload', [
    ('auto_button', 'laser_auto', {'go_to_calibration_pose': True}),
    ('check_button', 'laser_check', {}), ('capture_button', 'laser_capture', {}),
    ('solve_button', 'laser_solve', {}), ('reset_button', 'laser_reset', {}),
    ('cancel_button', 'laser_cancel', {}),
])
def test_panel_actions_are_one_service_json_request(button, action, payload):
    app, sent, _, _ = app_fixture()
    getattr(app.laser_ui, button).click()
    assert sent == [{'action': action, **payload}]


def test_supported_detection_exposure_and_noise_settings_stay_backend_owned():
    app, sent, _, scene = app_fixture()
    panel = app.laser_ui
    state = state_fixture()
    status = state['camera']['laser']['pointcloud_status']
    status.update(detection={'threshold': 45., 'relative_threshold': .5, 'min_width': 3, 'max_width': 20},
        noise={'enabled': True, 'min_neighbors': 3, 'max_scan_gap_px': 4, 'max_jump_px': 6.},
        auto_exposure={'enabled': False}, depth_range_mm=[10., 500.])
    panel.render(state, scene)
    assert panel.threshold.value == 45. and panel.max_width.value == 20
    panel.threshold.value = 50.
    panel.render(state, scene)
    assert panel.threshold.value == 50.  # Pending edits survive polling.
    panel.configure_button.click()
    assert sent == [{'action': 'laser_config',
        'detection': {'threshold': 50., 'relative_threshold': .5, 'min_width': 3, 'max_width': 20},
        'auto_exposure': {'enabled': False, 'min_us': 100., 'max_us': 10000.},
        'controls': {}, 'preview': {'gamma': .5, 'brightness': 1.},
        'noise': {'enabled': True, 'min_neighbors': 3, 'max_scan_gap_px': 4, 'max_jump_px': 6.},
        'depth_range_mm': [10., 500.]}]
    panel.preview.update(False)
    assert sent[-1] == {'action': 'laser_preview', 'enabled': False}


@pytest.mark.parametrize('blocker', [
    {'closed': True}, {'motion_active': True}, {'motion_fault': {'reason': 'fault'}},
    {'fault': 'SDK fault'}, {'cooldown_active': True}, {'disable_pending': True},
    {'shutdown': {'requested': True}}, {'mode_transition': True},
    {'calibration': {'status': 'capturing_manual'}}, {'laser': {'status': 'hunting', 'busy': True}},
])
def test_laser_owned_actions_block_unsafe_or_other_ownership_but_cancel_remains_available(blocker):
    app, _, _, scene = app_fixture()
    panel = app.laser_ui
    state = {**state_fixture(), **blocker}
    panel.render(state, scene)
    assert all(handle.disabled for handle in (panel.auto_button, panel.capture_button, panel.check_button, panel.solve_button))
    assert panel.configure_button.disabled is not blocker.get('closed', False)
    assert not panel.cancel_button.disabled


def test_auto_requires_explicit_camera_stream_permission():
    app, _, _, scene = app_fixture()
    state = state_fixture()
    state['camera'].update(enabled=False, fresh=False)
    app.laser_ui.render(state, scene)
    assert app.laser_ui.auto_button.disabled
    assert not app.laser_ui.stream.value
    assert not app.laser_ui.cancel_button.disabled
    assert not scene.calls


def test_world_cloud_is_displayed_without_using_local_j6_points_or_reconstruction():
    app, _, _, scene = app_fixture()
    state = state_fixture()
    app.laser_ui.render(state, scene)
    assert len(scene.calls) == 1
    name, cloud = scene.calls[0]
    assert name == '/laser/pointcloud'
    np.testing.assert_allclose(cloud['points'], state['laser']['cloud']['points_base_m'])
    assert cloud['precision'] == 'float32' and cloud['point_size'] == .002
    assert 'J6 tool coordinates in meters' in app.laser_ui.cloud_md.content
    app.laser_ui.render(state, scene)
    assert len(scene.calls) == 1 and not scene.removed
    state['laser']['cloud']['sequence'] = 2
    app.laser_ui.render(state, scene)
    assert len(scene.calls) == 2 and len(scene.removed) == 1
    state['motion_active'] = True
    app.laser_ui.render(state, scene)
    assert len(scene.removed) == 2
    assert 'stationary, healthy robot' in app.laser_ui.cloud_md.content


@pytest.mark.parametrize('cloud', [
    {'points_j6_m': [[1., 2., 3.]], 'units': 'm'},
    {'points_base_m': [[1., 2., 3.]], 'units': 'mm'},
    {'points_base_m': [[float('nan'), 2., 3.]], 'units': 'm'},
    {'points_base_m': [1., 2., 3.], 'units': 'm'},
    {'points_base_m': [[1., 2., 3.]], 'units': 'm', 'fresh': False},
])
def test_invalid_stale_or_local_only_cloud_never_draws_as_world_coordinates(cloud):
    app, _, _, scene = app_fixture()
    state = state_fixture()
    state['laser']['cloud'] = cloud
    app.laser_ui.render(state, scene)
    assert not scene.calls
    assert 'unavailable' in app.laser_ui.cloud_md.content


def test_pointcloud_preview_has_bounded_display_payload():
    app, _, _, scene = app_fixture()
    state = state_fixture()
    state['laser']['cloud']['points_base_m'] = [[1., 2., 3.]] * (LaserPlaneCalibrationUI.MAX_PREVIEW_POINTS + 1)
    app.laser_ui.render(state, scene)
    assert not scene.calls


def test_quality_visibility_and_failure_reason_clear_with_new_status():
    app, _, _, scene = app_fixture()
    state = state_fixture()
    state['laser'].update(status='error', error='Target left the image')
    app.laser_ui.render(state, scene)
    assert '0.080 mm' in app.laser_ui.quality_md.content
    assert 'Coverage ratio: 0.070' in app.laser_ui.quality_md.content
    assert 'Accepted views: **4**' in app.laser_ui.quality_md.content
    assert 'Target left the image' in app.laser_ui.status_md.content
    assert '120.0' in app.laser_ui.exposure_md.content and '2.0%' in app.laser_ui.exposure_md.content
    state['laser'].update(status='idle', error=None)
    app.laser_ui.render(state, scene)
    assert 'Target left the image' not in app.laser_ui.status_md.content


def test_shared_image_decodes_backend_laser_overlay_when_laser_panel_selected():
    from PIL import Image
    app, _, _, _ = app_fixture()
    stream = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), [250, 10, 20], dtype=np.uint8)).save(stream, format='JPEG')
    encoded = base64.b64encode(stream.getvalue()).decode()
    app.state = state_fixture()
    app.state['camera']['laser'].update(frame_jpeg_base64=encoded, preview_frame={'sequence': 8})
    app.calibration_panel_select.update('Laser plane calibration')
    app._render_camera()
    mean = app.camera_image.image.mean(axis=(0, 1))
    assert mean[0] > 240 and mean[1] < 20 and mean[2] < 30


def test_camera_calibration_waits_for_laser_owner_while_its_cancel_remains_available():
    app, _, _, _ = app_fixture()
    app.state = state_fixture()
    app.state['laser'].update(status='capturing', busy=True)
    app._render_camera()
    assert app.auto_button.disabled and app.capture_button.disabled and app.solve_button.disabled
    assert not app.cancel_calibration_button.disabled


def exposure_state():
    state = state_fixture()
    state['camera']['laser'].update(
        controls={'exposure_us': 8000., 'gain_db': 2.5},
        controls_requested={'exposure_us': 9000., 'gain_db': 2.5},
        control_limits={'exposure_us': [10., 1000000.], 'gain_db': [0., 27.]},
        exposure_limits={'requested_us': [100., 20000.], 'effective_us': [100., 16666.7], 'fps_cap_us': 16666.7},
        preview={'gamma': .5, 'brightness': 1.})
    state['camera']['laser']['pointcloud_status']['auto_exposure'] = {'enabled': True, 'min_us': 100., 'max_us': 16666.7}
    return state


def test_exposure_gain_and_display_settings_emit_one_config_with_requested_microseconds():
    app, sent, _, scene = app_fixture()
    panel = app.laser_ui
    state = exposure_state()
    panel.render(state, scene)
    assert panel.exposure_max.value == 20000.  # Requested, rather than FPS-clamped effective maximum.
    panel.exposure_min.value, panel.exposure_max.value = 250., 25000.
    panel.manual_exposure.value, panel.gain.value = 12000., 4.5
    panel.smart_exposure.value = False
    panel.preview_gamma.value, panel.preview_brightness.value = .3, 1.8
    panel.configure_button.click()
    assert len(sent) == 1 and sent[0]['action'] == 'laser_config'
    assert sent[0]['controls'] == {'exposure_us': 12000., 'gain_db': 4.5}
    assert sent[0]['auto_exposure'] == {'enabled': False, 'min_us': 250., 'max_us': 25000.}
    assert sent[0]['preview'] == {'gamma': .3, 'brightness': 1.8}
    assert 'laser_reset' not in str(sent) and 'laser_new' not in str(sent)


def test_actual_sensor_readback_and_effective_limits_are_visible_without_replacing_requested_values():
    app, _, _, scene = app_fixture()
    panel = app.laser_ui
    panel.render(exposure_state(), scene)
    text = panel.control_readback_md.content
    assert 'Actual exposure: **8000 µs**' in text
    assert 'Actual gain: **2.5 dB**' in text
    assert 'Sensor exposure limits: 10–1e+06 µs' in text
    assert 'Sensor gain limits: 0–27 dB' in text
    assert 'Requested auto exposure: 100–20000 µs' in text
    assert 'Effective auto exposure: 100–16666.7 µs' in text
    assert 'Frame-rate exposure cap: 16666.7 µs' in text
    assert panel.manual_exposure.value == 9000.
    assert (panel.gain.min, panel.gain.max) == (0., 27.)
    assert (panel.manual_exposure.min, panel.manual_exposure.max) == (10., 1000000.)


def test_auto_readback_updates_and_unrelated_changes_preserve_pending_settings_edits():
    app, _, _, scene = app_fixture()
    panel, state = app.laser_ui, exposure_state()
    panel.render(state, scene)
    panel.manual_exposure.value = 14000.
    panel.exposure_max.value = 25000.
    panel.gain.value = 5.
    panel.preview_gamma.value = .25
    state['camera']['laser']['controls']['exposure_us'] = 6000.
    state['camera']['laser']['controls_requested']['gain_db'] = 3.
    state['camera']['laser']['preview']['brightness'] = 1.2
    panel.render(state, scene)
    assert panel.manual_exposure.value == 14000.
    assert panel.exposure_max.value == 25000.
    assert panel.gain.value == 5.
    assert panel.preview_gamma.value == .25
    assert panel.preview_brightness.value == 1.2
    assert 'Actual exposure: **6000 µs**' in panel.control_readback_md.content


def test_camera_settings_are_available_with_disabled_motors_but_capture_and_motion_stay_blocked():
    app, _, _, scene = app_fixture()
    state = exposure_state()
    state['closed'] = True
    app.laser_ui.render(state, scene)
    assert not app.laser_ui.configure_button.disabled
    assert not app.laser_ui.manual_exposure.disabled and not app.laser_ui.gain.disabled
    assert app.laser_ui.auto_button.disabled and app.laser_ui.capture_button.disabled
    assert not app.laser_ui.cancel_button.disabled


def test_unknown_sensor_limits_disable_manual_values_without_inventing_gain_range():
    app, sent, _, scene = app_fixture()
    state = exposure_state()
    state['camera']['laser']['control_limits'] = {'exposure_us': [10., 1000000.], 'gain_db': None}
    app.laser_ui.render(state, scene)
    assert not app.laser_ui.manual_exposure.disabled
    assert app.laser_ui.gain.disabled
    app.laser_ui.configure_button.click()
    assert sent[0]['controls'] == {'exposure_us': 9000.}
    assert 'Sensor gain limits' not in app.laser_ui.control_readback_md.content


def test_display_controls_have_confirmed_backend_ranges_and_do_not_modify_received_jpeg():
    from PIL import Image
    app, _, gui, _ = app_fixture()
    assert any(widget.label == 'Laser line finder settings' for widget in gui.widgets if isinstance(widget.label, str))
    panel = app.laser_ui
    assert panel.preview_gamma.settings['min'] == .1 and panel.preview_gamma.settings['max'] == 3.
    assert panel.preview_brightness.settings['min'] == .1 and panel.preview_brightness.settings['max'] == 4.
    stream = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), [12, 24, 36], dtype=np.uint8)).save(stream, format='JPEG')
    app.state = exposure_state()
    app.state['camera']['laser'].update(frame_jpeg_base64=base64.b64encode(stream.getvalue()).decode(), preview_frame={'sequence': 1})
    app.calibration_panel_select.update('Laser plane calibration')
    app._render_camera()
    original = app.camera_image.image.copy()
    panel.preview_gamma.value, panel.preview_brightness.value = .1, 4.
    app.state['camera']['laser']['preview_frame']['sequence'] = 2
    app._render_camera()
    np.testing.assert_array_equal(app.camera_image.image, original)
    # Disabling the 3D cloud does not hide the backend-brightened laser image.
    app.state['camera']['laser']['preview_enabled'] = False
    panel.render(app.state, app.server.scene)
    app._render_camera()
    assert not panel.preview.value
    np.testing.assert_array_equal(app.camera_image.image, original)
