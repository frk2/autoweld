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
    panel.joint_kind.update('Flat butt')
    panel.start_button.click()
    assert sent == [{'action': 'weld_start', 'min_distance_mm': 25., 'start_pose': 'Weld Finder', 'reflective_surface': True,
        'joint_kind': 'flat_butt', 'plate_thickness_mm': 2.}]
    panel.minimum.value = 40.
    panel.start_button.click()
    assert sent[-1] == {'action': 'weld_start', 'min_distance_mm': 40., 'start_pose': 'Weld Finder', 'reflective_surface': True,
        'joint_kind': 'flat_butt', 'plate_thickness_mm': 2.}
    panel.cancel_button.click()
    assert sent[-1] == {'action': 'weld_cancel'}


def test_saved_start_pose_prefers_weld_finder_two_and_preserves_operator_selection():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    state['saved_poses'] = {'Captured rest': {}, 'Weld Finder': {}, 'Weld Finder 2': {}}
    panel.render(state, scene)
    assert panel.start_pose.value == 'Weld Finder 2'
    assert 'Start: **Weld Finder 2**' in panel.quality_md.content
    assert panel.start_pose.options == tuple(state['saved_poses'])
    assert not sent
    panel.start_button.click()
    assert sent[-1]['start_pose'] == 'Weld Finder 2'
    assert sent[-1]['reflective_surface'] is True
    panel.reflective.update(False)
    panel.start_pose.update('Weld Finder')
    state['saved_poses']['Another view'] = {}
    panel.render(state, scene)
    assert panel.start_pose.value == 'Weld Finder'
    panel.start_button.click()
    assert sent[-1]['start_pose'] == 'Weld Finder'
    assert sent[-1]['reflective_surface'] is False
    del state['saved_poses']['Weld Finder']
    panel.render(state, scene)
    assert panel.start_pose.value == 'Weld Finder 2'
    state['weld'].update(busy=True)
    panel.render(state, scene)
    assert panel.start_pose.disabled


def test_optional_search_roi_converts_percent_to_normalized_request_and_keeps_full_profile_hint():
    app, sent, _, _ = scene_fixture()
    panel = app.weld_ui
    assert not panel.use_roi.value
    assert all(handle.disabled for handle in (panel.roi_left, panel.roi_top, panel.roi_right, panel.roi_bottom))
    panel.use_roi.update(True)
    panel.roi_left.value, panel.roi_top.value = 20., 10.
    panel.roi_right.value, panel.roi_bottom.value = 60., 30.
    panel.start_button.click()
    assert sent[-1]['reflective_roi'] == [.2, .1, .6, .3]
    panel.use_roi.update(False)
    panel.start_button.click()
    assert 'reflective_roi' not in sent[-1]
    panel.use_roi.update(True)
    panel.reflective.value = False
    panel.start_button.click()
    assert 'reflective_roi' not in sent[-1]


@pytest.mark.parametrize('values', [(70., 0., 30., 100.), (0., 100., 100., 50.),
                                  (0., float('nan'), 100., 100.)])
def test_invalid_search_roi_does_not_submit_weld_finding(values):
    app, sent, _, _ = scene_fixture()
    panel = app.weld_ui
    panel.use_roi.value = True
    for handle, value in zip((panel.roi_left, panel.roi_top, panel.roi_right, panel.roi_bottom), values):
        handle.value = value
    panel.start_button.click()
    assert not sent
    assert 'left < right' in panel.status_md.content


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


def test_joint_type_defaults_to_coped_tube_with_one_inch_tubes_and_half_turn_target():
    app, sent, _, _ = scene_fixture()
    panel = app.weld_ui
    assert panel.joint_kind.value == 'Coped tube'
    assert panel.branch_diameter.value == panel.main_diameter.value == 25.4
    assert panel.joint_angle.value == 90. and panel.target_arc.value == 180.
    assert all(widget.visible for widget in (panel.branch_diameter, panel.main_diameter,
        panel.joint_angle, panel.target_arc))
    assert not panel.minimum.visible and not panel.plate_thickness.visible
    panel.start_button.click()
    assert sent == [{'action': 'weld_start', 'joint_kind': 'coped_tube', 'min_distance_mm': 25.,
        'start_pose': 'Weld Finder', 'reflective_surface': True, 'branch_diameter_mm': 25.4, 'main_diameter_mm': 25.4,
        'joint_angle_deg': 90., 'target_arc_deg': 180., 'max_profile_gap_mm': 5., 'coping_axis': 0}]


@pytest.mark.parametrize('label,kind', [('Flat lap', 'flat_lap'), ('Flat butt', 'flat_butt')])
def test_flat_joint_selection_uses_thickness_and_local_minimum_without_orbit_parameters(label, kind):
    app, sent, _, _ = scene_fixture()
    panel = app.weld_ui
    panel.joint_kind.update(label)
    assert not sent
    assert panel.minimum.visible and panel.plate_thickness.visible
    assert not any(widget.visible for widget in (panel.branch_diameter, panel.main_diameter,
        panel.joint_angle, panel.target_arc))
    panel.plate_thickness.value, panel.minimum.value = 3., 30.
    panel.start_button.click()
    assert sent == [{'action': 'weld_start', 'joint_kind': kind, 'min_distance_mm': 30.,
        'start_pose': 'Weld Finder', 'reflective_surface': True, 'plate_thickness_mm': 3.}]
    assert 'no tube orbit' in panel.joint_help_md.content


@pytest.mark.parametrize('field,value', [('branch_diameter', float('nan')), ('main_diameter', 0.),
    ('joint_angle', float('inf')), ('target_arc', 181.)])
def test_invalid_joint_geometry_does_not_submit_an_automatic_motion_request(field, value):
    app, sent, _, _ = scene_fixture()
    panel = app.weld_ui
    getattr(panel, field).value = value
    panel.start_button.click()
    assert not sent and 'Invalid' in panel.status_md.content


def test_joint_inputs_are_gated_during_service_or_other_motion_ownership():
    app, _, _, scene = scene_fixture()
    state = weld_state()
    for owner in ('weld', 'scan'):
        state.setdefault(owner, {})['busy'] = True
        app.weld_ui.render(state, scene)
        assert all(widget.disabled for widget in (app.weld_ui.joint_kind, app.weld_ui.coping_axis, app.weld_ui.branch_diameter,
            app.weld_ui.main_diameter, app.weld_ui.joint_angle, app.weld_ui.target_arc,
            app.weld_ui.plate_thickness, app.weld_ui.minimum))
        state[owner]['busy'] = False
    assert not app.weld_ui.cancel_button.disabled


def test_coping_fit_coverage_and_reachability_limits_are_distinct_from_predicted_arc():
    app, _, _, scene = scene_fixture()
    state = weld_state()
    state['weld']['settings']['joint_kind'] = 'coped_tube'
    state['weld'].update(joint_model={'status': 'qualified',
        'quality': {'rms_mm': .08, 'reason': 'Two measured cylindrical surfaces'},
        'model': {'kind': 'tube_intersection'}}, coping_coverage={'requested_arc_deg': 180.,
        'planned_arc_deg': 140., 'observed_arc_deg': 83., 'positive_observed_deg': 40.,
        'negative_observed_deg': 43., 'measured_station_count': 51, 'coverage_complete': False,
        'boundaries': {'positive': {'reason': 'joint limit', 'angle_deg': 70.},
            'negative': {'reason': 'visibility unresolved', 'angle_deg': -43.}},
        'preflight': {'status': 'bounded', 'positive_bound_deg': 70., 'negative_bound_deg': -70.},
        'gaps': [{'start_deg': -70., 'end_deg': -43., 'reason': 'visibility unresolved'}]})
    panel = app.weld_ui
    panel.render(state, scene)
    assert 'qualified' in panel.model_md.content and '0.08 mm' in panel.model_md.content
    assert 'Requested cope arc: **180°**' in panel.coverage_md.content
    assert 'Reachable planned arc: **140°**' in panel.coverage_md.content
    assert 'Actually observed arc: **83°**' in panel.coverage_md.content
    assert 'joint limit at 70°' in panel.coverage_md.content
    assert 'visibility unresolved at -43°' in panel.coverage_md.content
    assert 'incomplete' in panel.coverage_md.content and 'predicted geometry does not count' in panel.coverage_md.content
    assert 'no physical seam end is inferred' in panel.quality_md.content
    assert 'Minimum total local scan' not in panel.quality_md.content
    assert 'Positive reachable bound: **70°**' in panel.coverage_md.content
    assert 'Unobserved angular gaps: **1**' in panel.coverage_md.content
    assert '-70° to -43°' in panel.coverage_md.content
    assert 'Sensing +Z inward toward the cope' in panel.pose_md.content
    assert 'Observed cope point' in panel.pose_md.content


def test_predicted_cope_curve_is_opt_in_cyan_never_green_or_counted_as_measurements():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    predicted = [[.2, .1, .3], [.22, .13, .31]]
    state['weld']['geometry']['predicted_seam_points_base_m'] = predicted
    panel.render(state, scene)
    assert '/weld/predicted-cope' not in dict(scene.calls)
    panel.show_predicted.value = True
    panel.render(state, scene)
    calls = dict(scene.calls)
    assert calls['/weld/predicted-cope']['colors'] == (50, 210, 240)
    np.testing.assert_allclose(calls['/weld/predicted-cope']['points'], predicted)
    np.testing.assert_allclose(calls['/weld/seam']['points'], state['weld']['geometry']['seam_points_base_m'])
    assert '**2** measured seam points' in panel.geometry_md.content
    assert 'unmeasured prediction' in panel.geometry_md.content and not sent


def test_legacy_geometry_without_joint_metadata_still_renders_without_coping_claims():
    app, _, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.render(state, scene)
    assert '/weld/seam' in dict(scene.calls) and '/weld/scan-lines' in dict(scene.calls)
    assert 'No fitted joint model yet' in panel.model_md.content and panel.coverage_md.content == ''


def test_fit_role_ambiguity_and_backend_fit_quality_are_visible_without_inventing_a_model():
    app, _, _, scene = scene_fixture()
    state = weld_state()
    state['weld']['joint_model'] = {'status': 'role_ambiguous',
        'quality': {'fit_rms_mm': .12, 'branch_role_ambiguous': True,
            'reason': 'Equal tube diameters need a measured branch-axis hint'}, 'model': None}
    app.weld_ui.render(state, scene)
    assert 'role ambiguous' in app.weld_ui.model_md.content
    assert '0.12 mm' in app.weld_ui.model_md.content
    assert 'Equal tube diameters' in app.weld_ui.model_md.content
    assert 'role assignment is ambiguous' in app.weld_ui.model_md.content


def test_unqualified_coped_model_reports_local_locator_without_dumping_fit_error_or_inventing_coverage():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    weld = state['weld']
    weld['settings']['joint_kind'] = 'coped_tube'
    weld.update(status='error', message='Fit failed: giant_raw_quality_dictionary',
        error='giant_raw_quality_dictionary', replay_ready=False, point_count=1,
        coping_coverage={'requested_arc_deg': 180., 'observed_arc_deg': 0.,
            'measured_station_count': 0, 'provisional_seed_scout_count': 1, 'coverage_complete': False})
    for status in ('unobservable', 'ambiguous', 'error'):
        weld['joint_model'] = {'status': status,
            'quality': {'rms_mm': .24, 'reason': 'Whole-profile held-out validation did not qualify the model.'}}
        panel.render(state, scene)
        assert 'Local seam locator detected; coped-tube tracing is not qualified.' in panel.status_md.content
        assert 'Whole-profile held-out validation did not qualify the model.' in panel.status_md.content
        assert 'giant_raw_quality_dictionary' not in panel.status_md.content
        assert 'Fit RMS: **0.24 mm**' in panel.model_md.content
        assert 'Measured angular stations: **0**' in panel.coverage_md.content
        assert 'Actually observed arc: **0°**' in panel.coverage_md.content
        assert panel.replay_button.disabled and not sent
    weld.update(message='Hardware feedback fault', error='Fresh feedback unavailable')
    weld['coping_coverage']['provisional_seed_scout_count'] = 0
    panel.render(state, scene)
    assert 'Hardware feedback fault' in panel.status_md.content
    assert 'Fresh feedback unavailable' in panel.status_md.content
    assert 'Local seam locator detected' not in panel.status_md.content
    weld['coping_coverage']['provisional_seed_scout_count'] = 1
    state['fault'] = 'SDK reader stopped'
    panel.render(state, scene)
    assert 'Fresh feedback unavailable' in panel.status_md.content
    assert 'Local seam locator detected' not in panel.status_md.content


def test_coped_measured_points_are_not_autojoined_across_missing_angular_stations():
    app, _, _, scene = scene_fixture()
    state = weld_state()
    state['weld']['settings']['joint_kind'] = 'coped_tube'
    app.weld_ui.render(state, scene)
    assert '/weld/seam' in dict(scene.calls) and '/weld/path' not in dict(scene.calls)
    segments = [[[.2, .1, .3], [.2, .101, .3]]]
    state['weld']['geometry']['measured_seam_segments_base_m'] = segments
    app.weld_ui.render(state, scene)
    np.testing.assert_allclose(dict(scene.calls)['/weld/path']['points'], segments)
    state['weld']['geometry']['measured_seam_segments_base_m'] = []
    app.weld_ui.render(state, scene)
    assert '/weld/path' in scene.removed
    assert all(name != '/weld/path' for name, _ in scene.calls[-1:])


def test_revision_geometry_cache_reuses_full_publication_through_camera_off_unchanged_updates():
    app, sent, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    geometry = state['weld']['geometry']
    state['weld']['session_id'] = 'native-session'
    geometry.update(revision=3, profile_points_base_m=[[.19, .1, .3], [.2, .1, .3]])
    panel.render(state, scene)
    full_geometry, profile_array = panel._geometry_cache, panel._profile_cache
    before = len(scene.calls)
    state['camera'].update(enabled=False, fresh=False)
    state['weld']['geometry'] = {'units': 'm', 'coordinate_frame': 'robot_base', 'revision': 3, 'unchanged': True}
    panel.render(state, scene)
    assert len(scene.calls) == before and not scene.removed
    assert panel._geometry_cache is full_geometry and panel._profile_cache is profile_array
    assert 'retained after the camera stops' in panel.geometry_md.content
    assert not sent


def test_revision_geometry_cache_clears_on_unknown_session_or_revision_and_does_not_resurrect():
    app, _, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    state['weld'].update(session_id='old-session')
    state['weld']['geometry'].update(revision=3, profile_points_base_m=[[.2, .1, .3]])
    panel.render(state, scene)
    old_geometry = panel._geometry_cache
    state['weld'].update(session_id='new-session', geometry={
        'units': 'm', 'coordinate_frame': 'robot_base', 'revision': 3, 'unchanged': True})
    panel.render(state, scene)
    assert '/weld/seam' in scene.removed and '/weld/scan-profiles' in scene.removed
    assert panel._geometry_cache is None and panel._profile_cache is None and not panel._handles
    assert 'No registered weld geometry' in panel.geometry_md.content
    state['weld']['session_id'] = 'old-session'
    panel.render(state, scene)
    assert panel._geometry_cache is None and not panel._handles  # The old session cannot reappear from a stub.
    state['weld']['session_id'] = 'new-session'
    state['weld']['geometry'] = {**old_geometry, 'revision': 4}
    panel.render(state, scene)
    assert panel._handles
    state['weld']['geometry'] = {'units': 'm', 'coordinate_frame': 'robot_base', 'revision': 5, 'unchanged': True}
    panel.render(state, scene)
    assert panel._geometry_cache is None and not panel._handles


@pytest.mark.parametrize('replacement', [
    {'units': 'mm', 'coordinate_frame': 'camera', 'revision': 4},
    {'units': 'm', 'coordinate_frame': 'robot_base', 'revision': 4,
        'seam_points_base_m': [], 'profile_points_base_m': [], 'torch_poses_base_m': []},
])
def test_full_invalid_or_clear_publication_invalidates_old_arrays_before_later_stubs(replacement):
    app, _, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    state['weld']['session_id'] = 'session-to-clear'
    state['weld']['geometry'].update(revision=3, profile_points_base_m=[[.2, .1, .3]])
    panel.render(state, scene)
    state['weld']['geometry'] = replacement
    panel.render(state, scene)
    assert not panel._handles
    assert panel._profile_cache is None
    state['weld']['geometry'] = {'units': 'm', 'coordinate_frame': 'robot_base', 'revision': 4, 'unchanged': True}
    panel.render(state, scene)
    assert not panel._handles and '/weld/seam' in scene.removed
    fresh = weld_state()['weld']['geometry']
    fresh.update(revision=5, profile_points_base_m=[[.3, .1, .3]])
    state['weld']['geometry'] = fresh
    panel.render(state, scene)
    assert '/weld/seam' in dict(scene.calls) and panel._profile_cache is not None


def test_unchanged_stub_without_explicit_bound_session_revision_cannot_reuse_legacy_geometry():
    app, _, _, scene = scene_fixture()
    panel, state = app.weld_ui, weld_state()
    panel.render(state, scene)
    assert panel._handles
    state['weld']['geometry'] = {'units': 'm', 'coordinate_frame': 'robot_base', 'unchanged': True}
    panel.render(state, scene)
    assert not panel._handles and panel._geometry_cache is None


def test_starting_profile_branch_selector_is_explicit_coping_only_and_preserves_operator_choice():
    app, sent, _, scene = scene_fixture()
    panel = app.weld_ui
    assert panel.coping_axis.value == 'Left side' and panel.coping_axis.visible
    assert 'starting native laser profile and image' in panel.branch_hint_md.label
    assert 'frozen' in panel.branch_hint_md.label
    panel.coping_axis.update('Right side')
    assert not sent
    panel.start_button.click()
    assert sent[-1]['coping_axis'] == 1
    panel.joint_kind.update('Flat lap')
    assert not panel.coping_axis.visible and not panel.branch_hint_md.visible
    panel.start_button.click()
    assert 'coping_axis' not in sent[-1]
    panel.joint_kind.update('Coped tube')
    panel.render(weld_state(), scene)
    assert panel.coping_axis.value == 'Right side' and panel.coping_axis.visible
    panel.start_button.click()
    assert sent[-1]['coping_axis'] == 1
