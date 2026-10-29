"""The weld panel displays service measurements and never advances the robot."""

import numpy as np
import pytest
from test_laser_ui import app_fixture, state_fixture


def weld_state():
    state = state_fixture()
    pose = np.eye(4)
    pose[:3, 3] = [.2, .1, .15]
    state['weld'] = {'start_ready': True, 'replay_ready': True, 'status': 'complete', 'busy': False,
        'point_count': 2, 'distance_mm': 26., 'settings': {'min_distance_mm': 25., 'standoff_mm': 150.},
        'endpoints': {'positive': {'confirmed': True, 'reason': 'Valley disappeared'},
            'negative': {'confirmed': True, 'reason': 'Valley disappeared'}},
        'geometry': {'units': 'm', 'coordinate_frame': 'robot_base',
            'seam_points_base_m': [[.2, .1, .3], [.2, .101, .3]],
            'scan_lines_base_m': [[[.19, .1, .3], [.199, .1, .301]],
                [[.201, .1, .301], [.21, .1, .3]]],
            'torch_poses_base_m': [pose.tolist()]},
        'export_paths': {'json': '/tmp/weld.json', 'csv': '/tmp/weld.csv'}}
    return state


def scene_fixture():
    app, sent, gui, scene = app_fixture()
    scene.add_line_segments = lambda name, **kwargs: scene.add_point_cloud(name, **kwargs)
    return app, sent, gui, scene


def test_weld_panel_selection_uses_shared_preview_and_never_starts_acquisition():
    app, sent, gui, scene = scene_fixture()
    assert not app.weld_ui.folder.visible
    assert app.calibration_panel_select.settings['options'][-2:] == ['Laser scanning', 'Weld seam finding']
    app.calibration_panel_select.update('Weld seam finding')
    assert app.weld_ui.folder.visible
    assert not app.scan_ui.folder.visible and not app.laser_ui.folder.visible
    assert not app.camera_calibration_folder.visible
    assert sent == [{'action': 'camera_analysis', 'mode': 'laser'}]
    assert len([widget for widget in gui.widgets if widget.kind == 'add_image']) == 1
    app.weld_ui.stream.update(True)
    assert sent[-1] == {'action': 'camera_stream', 'enabled': True}
    count = len(sent)
    app.weld_ui.render(weld_state(), scene)
    assert app.weld_ui.stream.value and len(sent) == count


def test_start_is_one_request_with_saved_weld_finder_and_minimum_total_distance():
    app, sent, _, _ = scene_fixture()
    panel = app.weld_ui
    assert panel.minimum.value == 25.
    panel.start_button.click()
    assert sent == [{'action': 'weld_start', 'min_distance_mm': 25., 'start_pose': 'Weld Finder'}]
    panel.minimum.value = 40.
    panel.start_button.click()
    assert sent[-1] == {'action': 'weld_start', 'min_distance_mm': 40., 'start_pose': 'Weld Finder'}
    panel.cancel_button.click()
    assert sent[-1] == {'action': 'weld_cancel'}


@pytest.mark.parametrize('changes', [
    {'closed': True}, {'disable_pending': True}, {'motion_active': True}, {'fault': 'SDK fault'},
    {'motion_fault': {'reason': 'tracking'}}, {'mode_transition': True}, {'cooldown_active': True},
    {'shutdown': {'requested': True}}, {'shutdown': {'active': True}},
    {'camera': {'starting': True}}, {'camera': {'stopping': True}}, {'camera': {'configuring': True}},
    {'scan': {'busy': True}}, {'laser': {'busy': True}}, {'laser': {'status': 'hunting'}},
    {'calibration': {'status': 'hunting'}},
])
def test_start_and_replay_respect_robot_ownership_and_health(changes):
    app, _, _, scene = scene_fixture()
    state = {**weld_state(), **changes}
    app.weld_ui.render(state, scene)
    assert app.weld_ui.start_button.disabled and app.weld_ui.replay_button.disabled
    assert not app.weld_ui.cancel_button.disabled


def test_authoritative_readiness_is_required_for_start_and_replay_separately():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.render(state, scene)
    assert not panel.start_button.disabled and not panel.replay_button.disabled
    state['weld'].pop('start_ready')
    state['weld'].update(replay_ready=False, blocker='No supported valley', replay_blocker='Path incomplete')
    panel.render(state, scene)
    assert panel.start_button.disabled and panel.replay_button.disabled
    assert 'No supported valley' in panel.status_md.content and 'Path incomplete' in panel.quality_md.content
    assert not sent


def test_export_clear_and_replay_are_service_commands_and_busy_cancel_remains_available():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.render(state, scene)
    assert not panel.export_button.disabled and not panel.clear_button.disabled
    assert '/tmp/weld.json' in panel.export_md.content and '/tmp/weld.csv' in panel.export_md.content
    panel.replay_button.click()
    panel.export_button.click()
    panel.clear_button.click()
    assert sent == [{'action': 'weld_replay'}, {'action': 'weld_export'}, {'action': 'weld_clear'}]
    state['weld'].update(status='scanning', busy=True)
    panel.render(state, scene)
    assert all(widget.disabled for widget in (panel.minimum, panel.stream, panel.start_button,
        panel.replay_button, panel.export_button, panel.clear_button))
    assert not panel.cancel_button.disabled


def test_short_measured_section_and_search_extent_are_separate_and_replay_scope_is_visible():
    app, sent, _, scene = scene_fixture()
    state = weld_state()
    state['weld'].update(status='incomplete', distance_mm=12., search_distance_mm=25.4,
                         replay_scope='measured_section_only', coverage_complete=False)
    app.weld_ui.render(state, scene)
    assert not app.weld_ui.replay_button.disabled
    assert 'Measured weld length: **12 mm**' in app.weld_ui.quality_md.content
    assert 'Explored scan span: **25.4 mm**' in app.weld_ui.quality_md.content
    assert 'measured section only' in app.weld_ui.quality_md.content
    assert 'unconfirmed' in app.weld_ui.quality_md.content
    assert not sent


def test_registered_seam_green_scan_lines_red_and_camera_off_preserves_results():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.render(state, scene)
    calls = dict(scene.calls)
    np.testing.assert_allclose(calls['/weld/seam']['points'], state['weld']['geometry']['seam_points_base_m'])
    assert calls['/weld/seam']['colors'] == (30, 235, 80)
    assert calls['/weld/path']['colors'] == (30, 235, 80)
    assert calls['/weld/scan-lines']['colors'] == (245, 55, 55)
    assert calls['/weld/scan-lines']['thickness_units'] == calls['/weld/path']['thickness_units'] == 'screen'
    assert '150' in panel.quality_md.content and '26 mm' in panel.quality_md.content
    assert 'Valley disappeared' in panel.quality_md.content
    assert '200.00, 100.00, 150.00' in panel.pose_md.content
    assert '200.00, 101.00, 300.00' in panel.pose_md.content
    state['camera'].update(enabled=False, fresh=False)
    state.update(motion_active=True, fault='later unrelated fault')
    count = len(scene.calls)
    panel.render(state, scene)
    assert len(scene.calls) == count and not scene.removed
    assert 'retained after the camera stops' in panel.geometry_md.content
    assert not sent


def test_pose_axes_render_only_finite_rigid_service_poses_and_have_a_display_budget():
    app, _, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.show_poses.value = True
    state['weld']['geometry']['torch_poses_base_m'] *= 100
    panel.render(state, scene)
    axes = [kwargs for name, kwargs in scene.calls if name.startswith('/weld/torch-axis-')]
    assert len(axes) == 3 and all(axis['points'].shape == (25, 2, 3) for axis in axes)
    np.testing.assert_allclose(axes[2]['points'][0], [[.2, .1, .15], [.2, .1, .16]])
    state['weld']['geometry']['torch_poses_base_m'][0][0][0] = 2.
    panel.render(state, scene)
    assert '/weld/torch-axis-2' in scene.removed
    assert 'Torch +Z' not in panel.pose_md.content
    assert 'Path endpoint seam' in panel.pose_md.content


@pytest.mark.parametrize('change', [
    {'units': 'mm'}, {'coordinate_frame': 'camera'},
    {'seam_points_base_m': [[float('nan'), 0., 0.]], 'scan_lines_base_m': [], 'torch_poses_base_m': []},
    {'seam_points_base_m': [[0., 0., 0.]]*2001, 'scan_lines_base_m': [], 'torch_poses_base_m': []},
])
def test_unregistered_nonfinite_or_oversized_seam_is_not_drawn(change):
    app, _, _, scene = scene_fixture()
    state = weld_state()
    state['weld']['geometry'].update(change)
    app.weld_ui.render(state, scene)
    assert not scene.calls


def test_visibility_toggles_do_not_change_geometry_or_send_requests():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.render(state, scene)
    panel.show_lines.value = False
    panel.render(state, scene)
    assert '/weld/scan-lines' in scene.removed
    assert [name for name, _ in scene.calls][-2:] == ['/weld/seam', '/weld/path']
    panel.show_seam.value = False
    panel.render(state, scene)
    assert not panel._handles and not sent


def test_weld_busy_blocks_calibration_and_scan_panels_without_switching_camera_analysis():
    app, sent, _, scene = scene_fixture()
    state = weld_state()
    state['scan'] = {'ready': True}
    state['weld'].update(busy=True, status='scanning')
    app.state = state
    app._render_camera()
    app.laser_ui.render(state, scene)
    app.scan_ui.render(state, scene)
    assert app.auto_button.disabled and app.capture_button.disabled and app.solve_button.disabled
    assert app.camera_on.disabled and app.laser_ui.stream.disabled and app.scan_ui.stream.disabled
    assert app.laser_ui.auto_button.disabled and app.laser_ui.configure_button.disabled
    assert app.scan_ui.fast_button.disabled and app.scan_ui.precision_button.disabled
    assert not app.cancel_calibration_button.disabled and not app.laser_ui.cancel_button.disabled
    app.calibration_panel_select.update('Camera calibration')
    assert not sent  # Display navigation must not stop the weld owner's laser processing.


def test_weld_panel_hides_calibration_cone_without_another_camera_image():
    app, _, _, _ = scene_fixture()
    shown = []
    app._render_cone = shown.append
    app.state = weld_state()
    app.calibration_panel_select.update('Weld seam finding')
    app._render_camera()
    assert shown == [None]


def test_full_native_profiles_replace_short_fits_and_do_not_connect_missing_returns():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    profiles = [[.19, .1, .3], [.191, .1, .3], [.210, .1, .3], [.211, .1, .3]]
    state['weld']['geometry']['profile_points_base_m'] = profiles
    panel.render(state, scene)
    calls = dict(scene.calls)
    assert '/weld/scan-lines' not in calls
    cloud = calls['/weld/scan-profiles']
    np.testing.assert_allclose(cloud['points'], profiles)
    assert cloud['points'].shape == (4, 3) and cloud['points'].dtype == np.float32
    assert cloud['colors'] == (245, 55, 55) and cloud['point_size'] == .0005
    assert cloud['point_shading'] == 'flat' and cloud['precision'] == 'float32'
    assert 'full laser profiles' in panel.show_lines.label and 'full laser profiles' in panel.geometry_md.content
    assert not sent


@pytest.mark.parametrize('profiles', [[], None, [0., 1., 2.], [[float('nan'), 0., 0.]],
    [[0., 0., 0.]]*500001])
def test_present_empty_invalid_or_oversized_profiles_never_fall_back_to_short_fits(profiles):
    app, _, _, scene = scene_fixture()
    state = weld_state()
    state['weld']['geometry']['profile_points_base_m'] = profiles
    app.weld_ui.render(state, scene)
    assert not any(name in ('/weld/scan-lines', '/weld/scan-profiles') for name, _ in scene.calls)
    assert '/weld/seam' in dict(scene.calls)


def test_full_profiles_visibility_and_camera_off_keep_the_registered_data():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    state['weld']['geometry']['profile_points_base_m'] = [[.2, .1, .3], [.21, .1, .3]]
    panel.render(state, scene)
    calls = len(scene.calls)
    state['camera'].update(enabled=False, fresh=False)
    panel.render(state, scene)
    assert len(scene.calls) == calls and not scene.removed
    panel.show_lines.value = False
    panel.render(state, scene)
    assert '/weld/scan-profiles' in scene.removed
    assert [name for name, _ in scene.calls][-2:] == ['/weld/seam', '/weld/path']
    assert not sent


def test_profile_cache_uses_service_revision_and_count_without_a_second_large_byte_copy(monkeypatch):
    app, _, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    geometry = state['weld']['geometry']
    geometry.update(revision=1, profile_points_base_m=[[.2, .1, .3], [.21, .1, .3]])
    state['weld']['session_id'] = 'native-profiles'
    original, conversions = panel._array, []

    def convert(value, *args, **kwargs):
        if kwargs.get('dtype') is np.float32:
            conversions.append(1)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(panel, '_array', convert)
    panel.render(state, scene)
    first = panel._profile_cache
    geometry['profile_points_base_m'] = [[.2, .1, .3], [.21, .1, .3]]  # A new JSON snapshot of the same revision.
    panel.render(state, scene)
    assert len(conversions) == 1 and panel._profile_cache is first
    assert len([name for name, _ in scene.calls if name == '/weld/scan-profiles']) == 1
    assert len(panel._profile_signature[1]) == 16  # Digest, not a retained cloud-sized bytes duplicate.
    geometry.update(revision=2, profile_points_base_m=[[.2, .1, .3], [.22, .1, .3]])
    panel.render(state, scene)
    assert len(conversions) == 2 and panel._profile_cache is not first
    assert len([name for name, _ in scene.calls if name == '/weld/scan-profiles']) == 2


def test_full_profile_preview_accepts_half_a_million_finite_points_without_decimation():
    app, _, _, scene = scene_fixture()
    state = weld_state()
    state['weld']['geometry']['profile_points_base_m'] = np.zeros((500000, 3), dtype=np.float32)
    app.weld_ui.render(state, scene)
    assert dict(scene.calls)['/weld/scan-profiles']['points'].shape == (500000, 3)
