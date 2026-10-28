"""Scan controls remain a JSON client; persisted geometry outlives live acquisition."""
import numpy as np
import pytest
from test_laser_ui import app_fixture, state_fixture


def scan_state():
    state = state_fixture()
    state['scan'] = {'ready': True, 'status': 'idle', 'busy': False, 'captures': 3,
        'point_count': 2, 'index': 3, 'total': 121,
        'quality': {'stripe_points': 80, 'standoff_mm': 140.2, 'object_points': 65},
        'cloud': {'units': 'm', 'points_base_m': [[.2, .1, .3], [.201, .1, .3]],
            'revision': 3, 'scan_id': 'tube-1'},
        'export_paths': {'json': '/tmp/scan.json', 'ply': '/tmp/scan.ply'}}
    return state


def test_scan_panel_selection_shares_camera_and_selecting_never_opens_it():
    app, sent, gui, _ = app_fixture()
    app.calibration_panel_select.update('Laser scanning')
    assert app.scan_ui.folder.visible
    assert not app.camera_calibration_folder.visible and not app.laser_ui.folder.visible
    assert sent == [{'action': 'camera_analysis', 'mode': 'laser'}]
    assert len([w for w in gui.widgets if w.kind == 'add_image']) == 1
    app.scan_ui.stream.update(True)
    assert sent == [{'action': 'camera_analysis', 'mode': 'laser'}, {'action': 'camera_stream', 'enabled': True}]
    app.scan_ui.render(scan_state(), app.server.scene)
    assert app.scan_ui.stream.value
    assert len(sent) == 2  # Service readback never sends another camera request.


def test_panel_selection_restores_checkerboard_work_only_for_camera_calibration():
    app, sent, _, _ = app_fixture()
    app.calibration_panel_select.update('Laser scanning')
    app.calibration_panel_select.update('Camera calibration')
    app.calibration_panel_select.update('Laser plane calibration')
    assert sent == [
        {'action': 'camera_analysis', 'mode': 'laser'},
        {'action': 'camera_analysis', 'mode': 'checkerboard'},
        {'action': 'camera_analysis', 'mode': 'laser'},
    ]
    app.calibration_panel_select.update('Camera calibration', client_id=None)
    assert len(sent) == 3  # Server-side UI synchronization never reconfigures acquisition.


def test_fast_and_precision_start_one_service_request_with_default_eighty_mm_downward_profile():
    app, sent, _, _ = app_fixture()
    panel = app.scan_ui
    panel.fast_button.click()
    assert sent == [{'action': 'scan_start', 'scan_method': 'fast', 'scan_profile': 'vertical_down',
        'sweep_length_mm': 80., 'continuous': True, 'max_duration_s': 900., 'feed_mm_s': 10., 'start_pose': 'Tube Scanning',
        'horizontal_mm': 100., 'vertical_mm': 100., 'optical_roll_deg': 0.}]
    panel.start_pose.value = 'Current pose'
    panel.sweep_length.value = 60.
    panel.precision_button.click()
    assert sent[-1] == {'action': 'scan_start', 'scan_method': 'precision', 'scan_profile': 'vertical_down',
        'sweep_length_mm': 60., 'continuous': False, 'max_duration_s': 900., 'feed_mm_s': 10., 'start_pose': 'current',
        'horizontal_mm': 100., 'vertical_mm': 100., 'optical_roll_deg': 0.}
    panel.cancel_button.click()
    assert sent[-1] == {'action': 'scan_cancel'}


def test_optional_area_preserves_hundred_mm_square_and_uses_fast_method_only():
    app, sent, _, scene = app_fixture()
    panel = app.scan_ui
    assert panel.pattern.value == 'Vertical down sweep'
    assert panel.sweep_length.visible and not panel.horizontal.visible and not panel.vertical.visible
    assert panel.fast_button.label == 'Fast scan' and panel.precision_button.label == 'Precision laser scan'
    assert panel.sweep_length.value == 80. and '1 mm intervals' in panel.precision_md.label
    panel.pattern.update('Area')
    assert not sent  # Choosing the scan pattern never reconfigures devices or starts a scan.
    panel.render(scan_state(), scene)
    assert panel.horizontal.visible and panel.vertical.visible and not panel.sweep_length.visible
    assert panel.precision_button.disabled and not panel.fast_button.disabled
    assert panel.optical_roll.value == -45.
    assert panel.horizontal.settings['max'] == panel.vertical.settings['max'] == 100.
    panel.fast_button.click()
    assert sent == [{'action': 'scan_start', 'scan_method': 'fast', 'scan_profile': 'area',
        'sweep_length_mm': 80., 'continuous': True, 'max_duration_s': 900., 'feed_mm_s': 30., 'start_pose': 'Tube Scanning',
        'horizontal_mm': 100., 'vertical_mm': 100., 'optical_roll_deg': -45.}
    ]
    panel.pattern.update('Vertical down sweep')
    assert not panel.precision_button.disabled
    assert panel.optical_roll.value == 0.


def test_pattern_default_roll_preserves_operator_edits_and_both_vertical_methods_share_the_view():
    app, sent, _, _ = app_fixture()
    panel = app.scan_ui
    assert panel.optical_roll.value == 0.
    panel.fast_button.click()
    panel.precision_button.click()
    assert [request['optical_roll_deg'] for request in sent] == [0., 0.]
    panel.optical_roll.value = 12.
    panel.pattern.update('Area')
    assert panel.optical_roll.value == 12.
    panel.pattern.update('Vertical down sweep')
    assert panel.optical_roll.value == 12.


def test_fixed_optical_roll_is_explicit_bounded_start_setting_and_never_overrides_resume():
    app, sent, _, scene = app_fixture()
    panel = app.scan_ui
    assert panel.optical_roll.value == 0.
    assert panel.optical_roll.settings['min'] == -180.
    assert panel.optical_roll.settings['max'] == 180.
    panel.horizontal.value = panel.vertical.value = 100.
    panel.optical_roll.value = -15.
    panel.start_button.click()
    assert sent[-1] == {'action': 'scan_start', 'scan_method': 'fast', 'scan_profile': 'vertical_down',
        'sweep_length_mm': 80., 'continuous': True, 'max_duration_s': 900., 'feed_mm_s': 10., 'start_pose': 'Tube Scanning',
        'horizontal_mm': 100., 'vertical_mm': 100., 'optical_roll_deg': -15.}
    state = scan_state()
    state['scan'].update(busy=True, settings={'optical_roll_deg': -15.})
    panel.render(state, scene)
    assert panel.optical_roll.disabled
    state['scan'].update(busy=False, ready=True, resume_available=True)
    panel.render(state, scene)
    panel.optical_roll.value = 35.
    panel.resume_button.click()
    assert sent[-1] == {'action': 'scan_resume'}


@pytest.mark.parametrize('blocker', [
    {'closed': True}, {'disable_pending': True}, {'motion_active': True},
    {'motion_fault': {'reason': 'tracking guard'}}, {'fault': 'SDK fault'},
    {'cooldown_active': True}, {'shutdown': {'requested': True}}, {'mode_transition': True},
    {'laser': {'status': 'capturing', 'busy': True}}, {'calibration': {'status': 'hunting'}},
    {'scan': {'busy': True, 'ready': True}}, {'scan': {'ready': False, 'blocker': 'No good stripe'}},
])
def test_start_respects_backend_readiness_ownership_and_health_cancel_stays_available(blocker):
    app, _, _, scene = app_fixture()
    panel = app.scan_ui
    panel.render({**scan_state(), **blocker}, scene)
    assert panel.fast_button.disabled and panel.precision_button.disabled
    assert not panel.cancel_button.disabled


def test_readiness_and_quality_are_visible_and_start_needs_authoritative_ready_flag():
    app, _, _, scene = app_fixture()
    state = scan_state()
    app.scan_ui.render(state, scene)
    assert not app.scan_ui.start_button.disabled
    assert '140.2 mm' in app.scan_ui.quality_md.content
    assert '80 points' in app.scan_ui.quality_md.content
    del state['scan']['ready']
    state['scan']['blocker'] = 'Laser optical calibration does not match this camera'
    app.scan_ui.render(state, scene)
    assert app.scan_ui.start_button.disabled
    assert 'Laser optical calibration does not match this camera' in app.scan_ui.status_md.content


def test_scan_uses_backend_native_stripe_readiness_when_checkerboard_observation_is_old():
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['camera']['fresh'] = False
    app.scan_ui.render(state, scene)
    assert not app.scan_ui.start_button.disabled
    state['scan'].update(ready=False, blocker='Native stripe acquisition is stale')
    app.scan_ui.render(state, scene)
    assert app.scan_ui.start_button.disabled


def test_persisted_cloud_survives_motion_camera_stop_fault_and_completion():
    app, _, _, scene = app_fixture()
    state = scan_state()
    panel = app.scan_ui
    panel.render(state, scene)
    assert len(scene.calls) == 1 and scene.calls[0][0] == '/scan/pointcloud'
    np.testing.assert_allclose(scene.calls[0][1]['points'], state['scan']['cloud']['points_base_m'])
    state.update(motion_active=True, fault='later SDK fault')
    state['camera'].update(enabled=False, fresh=False)
    state['scan'].update(status='completed')
    panel.render(state, scene)
    assert len(scene.calls) == 1 and not scene.removed
    assert 'retained after motion and camera stop' in panel.cloud_md.content
    state['scan']['cloud']['points_base_m'].append([.202, .1, .3])
    state['scan']['cloud']['revision'] += 1
    panel.render(state, scene)
    assert len(scene.calls) == 2 and len(scene.removed) == 1
    panel.show_cloud.value = False
    panel.render(state, scene)
    assert len(scene.removed) == 2


def test_export_paths_and_clear_send_service_commands_without_camera_or_file_access():
    app, sent, _, scene = app_fixture()
    panel, state = app.scan_ui, scan_state()
    panel.render(state, scene)
    assert '/tmp/scan.json' in panel.export_md.content and '/tmp/scan.ply' in panel.export_md.content
    assert not panel.export_button.disabled and not panel.clear_button.disabled
    panel.export_button.click()
    panel.clear_button.click()
    assert sent == [{'action': 'scan_export'}, {'action': 'scan_clear'}]
    state['scan']['busy'] = True
    panel.render(state, scene)
    assert panel.export_button.disabled and panel.clear_button.disabled
    assert not panel.cancel_button.disabled


@pytest.mark.parametrize('cloud', [
    {'points_camera_m': [[1., 2., 3.]], 'units': 'm'},
    {'points_base_m': [[1., 2., 3.]], 'units': 'mm'},
    {'points_base_m': [[float('nan'), 2., 3.]], 'units': 'm'},
    {'points_base_m': [1., 2., 3.], 'units': 'm'},
])
def test_invalid_or_unregistered_cloud_is_never_drawn_as_world_coordinates(cloud):
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan']['cloud'] = cloud
    app.scan_ui.render(state, scene)
    assert not scene.calls


def test_scan_ownership_blocks_both_calibration_panels():
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan']['busy'] = True
    app.state = state
    app._render_camera()
    app.laser_ui.render(state, scene)
    assert app.auto_button.disabled and app.capture_button.disabled and app.solve_button.disabled
    assert app.laser_ui.auto_button.disabled and app.laser_ui.configure_button.disabled
    assert not app.cancel_calibration_button.disabled and not app.laser_ui.cancel_button.disabled


def test_scan_selection_hides_calibration_cone_and_displays_shared_laser_preview():
    app, _, _, _ = app_fixture()
    shown = []
    app._render_cone = shown.append
    app.state = scan_state()
    app.calibration_panel_select.update('Laser scanning')
    app._render_camera()
    assert shown == [None]


@pytest.mark.parametrize('ready,resumable,busy,enabled', [
    (True, True, False, True), (False, True, False, False),
    (True, False, False, False), (True, True, True, False),
])
def test_resume_uses_authoritative_session_and_readiness_flags(ready, resumable, busy, enabled):
    app, sent, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(ready=ready, resume_available=resumable, busy=busy)
    app.scan_ui.render(state, scene)
    assert app.scan_ui.resume_button.disabled is not enabled
    assert not app.scan_ui.cancel_button.disabled
    if enabled:
        app.scan_ui.resume_button.click()
        assert sent == [{'action': 'scan_resume'}]


def test_full_grid_progress_distinguishes_accepted_missing_rejected_and_preview_counts():
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(total=10201, visited=5101, accepted=3000, captures=3000,
        no_return=2000, rejected=101, point_count=1200000, display_point_count=2)
    app.scan_ui.render(state, scene)
    text = app.scan_ui.status_md.content
    assert '5,101 / 10,201' in text and '50.0%' in text
    assert 'Accepted: **3,000**' in text
    assert 'No return: **2,000**' in text and 'Rejected: **101**' in text
    assert '2 preview points' in app.scan_ui.cloud_md.content
    assert '1,200,000 stored surface points' in app.scan_ui.cloud_md.content


def test_preflight_progress_is_visible_before_any_grid_movement():
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(status='preflighting', busy=True, total=10201, visited=0,
        preflight={'index': 3500, 'total': 10201})
    app.scan_ui.render(state, scene)
    assert 'Preflight: **3,500 / 10,201**' in app.scan_ui.status_md.content
    assert app.scan_ui.start_button.disabled and app.scan_ui.resume_button.disabled
    assert app.scan_ui.horizontal.disabled and app.scan_ui.vertical.disabled
    assert not app.scan_ui.cancel_button.disabled


def test_no_return_coverage_can_be_exported_without_any_surface_points():
    app, sent, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(visited=20, no_return=20, captures=0, accepted=0, point_count=0,
        cloud={'units': 'm', 'points_base_m': []})
    app.scan_ui.render(state, scene)
    assert not app.scan_ui.export_button.disabled
    assert 'Coverage recorded; no accepted surface points' in app.scan_ui.cloud_md.content
    app.scan_ui.export_button.click()
    assert sent == [{'action': 'scan_export'}]


def geometry():
    return {'units': 'm', 'coordinate_frame': 'robot_base',
        'outline_base_m': [[-.05, -.05, .2], [.05, -.05, .2], [.05, .05, .2],
                           [-.05, .05, .2], [-.05, -.05, .2]],
        'current_base_m': [0., 0., .2]}


def test_service_supplied_envelope_uses_screen_pixels_and_remains_bounded():
    app, _, _, scene = app_fixture()
    scene.add_line_segments = lambda name, **kwargs: scene.add_point_cloud(name, **kwargs)
    state = scan_state()
    state['scan']['geometry'] = geometry()
    app.scan_ui.render(state, scene)
    lines = next(kwargs for name, kwargs in scene.calls if name == '/scan/envelope')
    assert lines['thickness_units'] == 'screen' and lines['thickness'] == 1.5
    assert lines['points'].shape == (4, 2, 3)
    marker = next(kwargs for name, kwargs in scene.calls if name == '/scan/grid-position')
    assert marker['points'].shape == (1, 3)
    calls = len(scene.calls)
    app.scan_ui.render(state, scene)
    assert len(scene.calls) == calls  # Unchanged periodic state never rebuilds geometry.
    app.scan_ui.show_envelope.value = False
    app.scan_ui.render(state, scene)
    assert '/scan/envelope' in scene.removed and '/scan/grid-position' in scene.removed
    assert '/scan/pointcloud' not in scene.removed


@pytest.mark.parametrize('change', [
    {'units': 'mm'}, {'coordinate_frame': 'camera'},
    {'outline_base_m': [[1., 2., 3.]]}, {'current_base_m': [float('nan'), 0., 0.]},
])
def test_invalid_envelope_never_produces_unregistered_or_nonfinite_geometry(change):
    app, _, _, scene = app_fixture()
    scene.add_line_segments = lambda name, **kwargs: scene.add_point_cloud(name, **kwargs)
    state = scan_state()
    state['scan'].update(cloud={}, geometry={**geometry(), **change})
    app.scan_ui.render(state, scene)
    if 'current_base_m' in change:
        assert [name for name, _ in scene.calls] == ['/scan/envelope']
    else:
        assert not scene.calls


def test_preview_draw_budget_is_hard_bounded_independently_of_full_capture_count():
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan']['cloud']['points_base_m'] = [[0., 0., .2]] * 10001
    state['scan']['point_count'] = 10000000
    app.scan_ui.render(state, scene)
    assert not scene.calls


def test_continuous_start_admits_camera_off_and_does_not_send_extra_device_requests():
    app, sent, _, scene = app_fixture()
    state = scan_state()
    state['camera'] = {'enabled': False}
    state['scan'].update(continuous=True, start_ready=True, ready=True)
    app.scan_ui.render(state, scene)
    assert not app.scan_ui.start_button.disabled
    assert not app.scan_ui.precision_button.disabled
    assert not app.scan_ui.resume_button.visible and app.scan_ui.resume_button.disabled
    app.scan_ui.start_button.click()
    assert sent == [{'action': 'scan_start', 'scan_method': 'fast', 'scan_profile': 'vertical_down',
        'sweep_length_mm': 80., 'continuous': True, 'max_duration_s': 900., 'feed_mm_s': 10.,
        'start_pose': 'Tube Scanning', 'horizontal_mm': 100., 'vertical_mm': 100., 'optical_roll_deg': 0.}]


def test_continuous_phase_eta_timing_gaps_and_measured_coverage_are_visible():
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(continuous=True, busy=True, phase='scanning', elapsed_s=180., eta_s=520.,
        validated_duration_s=642.6, timing_uncertainty_s=.0007, visited=1001, total=10201, gaps=4)
    app.scan_ui.render(state, scene)
    status = app.scan_ui.status_md.content
    assert '**scanning**' in status and 'Measured 1 mm coverage' in status
    assert '3m 00s' in status and '8m 40s' in status and '10m 43s' in status
    assert 'Timing/coverage gaps: **4**' in status
    assert '0.7 ms' in app.scan_ui.quality_md.content
    assert app.scan_ui.stream.disabled
    assert not app.scan_ui.resume_button.visible
    assert app.scan_ui.horizontal.disabled and app.scan_ui.optical_roll.disabled
    assert app.scan_ui.pattern.disabled and app.scan_ui.sweep_length.disabled
    assert app.scan_ui.fast_button.disabled and app.scan_ui.precision_button.disabled


@pytest.mark.parametrize('phase', ['idle', 'complete', 'cancelled', 'error', 'interrupted'])
def test_terminal_scan_has_no_remaining_time_estimate(phase):
    app, _, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(continuous=True, busy=False, phase=phase, elapsed_s=180., eta_s=None)
    app.scan_ui.render(state, scene)
    assert 'Elapsed: **3m 00s**' in app.scan_ui.status_md.content
    assert 'Remaining' not in app.scan_ui.status_md.content
    assert 'being estimated' not in app.scan_ui.status_md.content


def test_new_flow_never_resumes_legacy_archive_even_if_flag_accidentally_present():
    app, sent, _, scene = app_fixture()
    state = scan_state()
    state['scan'].update(continuous=True, resume_available=True)
    app.scan_ui.render(state, scene)
    assert app.scan_ui.resume_button.disabled and not app.scan_ui.resume_button.visible
    assert not sent


def test_receive_reference_limits_are_plain_and_never_labeled_physical_timing_proof():
    app,_,_,scene = app_fixture()
    state = scan_state()
    state['scan'].update(continuous=True,registration_reference='measured_kernel_rx',
        physical_sample_timing_qualified=False,firmware_sample_age_s=None,
        reference_timing_uncertainty_s=.0003)
    app.scan_ui.render(state,scene)
    text = app.scan_ui.quality_md.content
    assert 'firmware sample age unknown' in text
    assert 'Receive-reference timing uncertainty: **0.3 ms**' in text
    assert 'Exposure pose timing uncertainty' not in text


def test_line_footprint_area_status_shows_actual_lanes_and_original_grid_cells():
    app,_,_,scene = app_fixture()
    state = scan_state()
    state['scan'].update(continuous=True,coverage_basis='measured_reference_plane_laser_footprint',
        footprint_plan=dict(lanes=6,lane_spacing_mm=20.),visited=9800,total=10201)
    app.scan_ui.render(state,scene)
    assert 'Reference-plane footprint coverage' in app.scan_ui.status_md.content
    assert '9,800 / 10,201' in app.scan_ui.status_md.content
    assert 'Overlapping laser lanes: **6**' in app.scan_ui.quality_md.content
    assert 'lane pitch: **20 mm**' in app.scan_ui.quality_md.content
