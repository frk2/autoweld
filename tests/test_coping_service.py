"""Autonomous coped-view orchestration; synthetic camera and no physical owner."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_weld_seam_service import pump
from test_weld_seam_service import rig as _base_rig

from autoweld.camera import _weldfinder_module
from autoweld.coping_plan import coping_camera_pose
from autoweld.weld_seam_service import WeldSettings

project_coping_seam = _weldfinder_module('coped_joint').project_coping_seam

rig = _base_rig
sample_coping_curve = _weldfinder_module('coped_joint').sample_coping_curve
coping_reference_pose = _weldfinder_module('coped_joint').coping_reference_pose

def model():
    radius = 12.7
    center = [-8., -8., 150.+np.sqrt(radius**2-8.**2)]
    return dict(schema='weldfinder.coped_joint.v1', units='mm', coordinate_frame='camera', model_id='known-model',
        joint_angle_deg=90., center_mm=center, frame_rotation=np.eye(3).tolist(), axis_offset_mm=0.,
        axis_role_ambiguous=True, axes=[dict(origin_mm=center, direction=[1., 0., 0.], radius_mm=radius),
                                     dict(origin_mm=center, direction=[0., 1., 0.], radius_mm=radius)])


@pytest.fixture
def coping(rig, monkeypatch):
    from autoweld import coping_service

    fits, plans = [], []

    def fit(profiles, **kwargs):
        fits.append((copy.deepcopy(profiles), kwargs))
        assert len([profile for profile in profiles if profile['provenance']['fresh_current_run']]) >= 3
        labels, groups = [], []
        for index, profile in enumerate(profiles):
            count = len(profile['xyz_mm'])
            labels.extend([0]*(count//2)+[1]*(count-count//2))
            groups.extend([index]*count)
        return dict(status='qualified', model=model(), quality={'all_point_rms_mm': .05},
            classification={'surface_labels': labels, 'profile_indices': groups,
                'inlier_mask': [True]*len(labels), 'residuals_mm': [[0., 0.]]*len(labels)})

    def preflight(source, start_q, bounds, frame, handeye, plane, curve, **kwargs):
        plans.append(curve)
        rows = []
        for angle, point, tangent, inward in zip(curve['relative_angles_deg'], curve['points_mm'],
                                               curve['tangents'], curve['inward_directions'], strict=True):
            if tangent is None:
                continue
            world = frame[:3, :3] @ (np.asarray(point)/1000.)+frame[:3, 3]
            tangent, inward = frame[:3, :3] @ tangent, frame[:3, :3] @ inward
            pose, camera = coping_camera_pose(world, tangent, inward, plane, handeye, reference_camera=frame)
            rows.append(dict(angle_deg=angle, base_from_tcp=pose.tolist(), base_from_camera=camera.tolist(),
                predicted_point_base_m=world.tolist(), tangent_base=tangent.tolist(), inward_base=inward.tolist(),
                alternatives=[], proof={'normal_full_pose_validation': True}))
        center = next(row for row in rows if abs(row['angle_deg']) < 1e-8)
        positive = sorted([row for row in rows if row['angle_deg'] > 1e-8], key=lambda row: row['angle_deg'])
        negative = sorted([row for row in rows if row['angle_deg'] < -1e-8], key=lambda row: -row['angle_deg'])
        return dict(status='qualified', center=center, branches=dict(positive=positive, negative=negative), boundaries={},
            positive_bound_deg=max(row['angle_deg'] for row in positive),
            negative_bound_deg=min(row['angle_deg'] for row in negative), planned_arc_deg=40.)

    def find(cloud, **kwargs):
        return dict(status='found', found=True, seam_point_mm=[0., 0., 150.],
            camera_from_torch=np.eye(4).tolist(), scan_direction_camera=[0., 1., 0.],
            stripe_direction_camera=[1., 0., 0.], surface_normal_camera=[0., 0., -1.],
            flank_lines_mm=[[[-10., 0., 150.], [-2., 0., 150.]], [[2., 0., 150.], [10., 0., 150.]]],
            quality={'good': True, 'feature_kind': 'inside_corner'})

    rig.service.camera.fit_coped_joint = fit
    rig.service.camera.select_coped_surface_returns = _weldfinder_module('coped_joint').select_coped_surface_returns
    rig.service.camera.sample_coping_curve = sample_coping_curve
    rig.service.camera.coping_reference_pose = coping_reference_pose
    rig.service.camera.project_coping_seam = project_coping_seam
    rig.service.camera.find_weld_seam = find
    rig.service.ctrl.planner_for_controller = lambda: rig.service.planner
    monkeypatch.setattr(coping_service, 'plan_coping_views', preflight)
    return rig, fits, plans


def test_legacy_settings_restore_without_forcing_a_new_joint_type():
    settings = WeldSettings(**dict(min_distance_mm=25., start_pose='Weld Finder')).validate()
    assert settings.joint_kind == 'local'


@pytest.mark.parametrize('values', [dict(joint_kind='unknown'), dict(branch_diameter_mm=1.),
    dict(main_diameter_mm=501.), dict(joint_angle_deg=0.), dict(target_arc_deg=181.), dict(plate_thickness_mm=0.)])
def test_joint_parameters_are_validated(values):
    with pytest.raises(ValueError):
        WeldSettings(**values).validate()


def test_coped_scan_collects_fresh_scouts_rotates_view_and_records_measured_radial_torch_poses(coping):
    rig, fits, plans = coping
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy)
    assert rig.weld.status in ('complete', 'incomplete'), rig.weld.error
    assert len(fits) == 1 and len(plans) == 1
    reference = rig.weld._coping_plan['measured_view_reference']
    assert reference and reference['association']['associated']
    raw = json.loads(Path(reference['native_path']).read_text())
    assert reference['base_from_camera'] == raw['measurement']['base_from_camera']
    assert reference['seam_point_camera_mm'] == raw['result']['seam_point_mm']
    assert len(rig.weld.records) >= 10
    assert any(record['direction'] == -1 for record in rig.weld.records)
    assert all(record['torch_orientation_policy'] == 'coped_centerline' for record in rig.weld.records)
    rotations = [event[1][:3, :3] for event in rig.events if event[0] == 'move']
    assert any(np.linalg.norm(rotation-np.eye(3)) > .1 for rotation in rotations)
    assert any(np.linalg.norm(rotations[0]-rotation) > .1 for rotation in rotations[1:])
    for record in rig.weld.records:
        torch = np.asarray(record['torch_pose_base_m'])
        seam = np.asarray(record['seam_point_base_m'])
        np.testing.assert_allclose(seam-torch[:3, 3], .15*torch[:3, 2], atol=1e-10)
        geometry = record['coping_geometry']
        radial = np.asarray(geometry['centerline_closest_point_seed_camera_mm'])-geometry['measured_point_seed_camera_mm']
        np.testing.assert_allclose(radial/np.linalg.norm(radial), geometry['inward_direction_seed_camera'])
        assert record['detector']['camera_from_torch'] == np.eye(4).tolist()  # Raw local detector remains untouched.
    state = rig.weld.state(camera_state=rig.state)
    assert state['coping_coverage']['observed_arc_deg'] >= 39.9
    assert 'classification' not in state['joint_model']
    assert state['geometry']['predicted_seam_points_base_m']
    assert state['geometry']['profile_points_base_m']
    assert rig.weld._result['joint_model']['classification']['surface_labels']
    assert rig.weld._result['minimum_basis'] == 'measured_coped_angular_span'
    assert state['replay_ready']


def test_no_found_root_never_turns_model_predictions_into_measured_orbit_points(coping):
    rig, _, _ = coping
    original = rig.service.camera.find_weld_seam

    def disappear(cloud, **kwargs):
        if rig.weld._coping_phase == 'orbit' and not rig.weld._coping_current.get('seed_recheck'):
            return dict(status='no_seam', found=False, quality={'good': True})
        return original(cloud, **kwargs)

    rig.service.camera.find_weld_seam = disappear
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status == 'incomplete' and not rig.weld.busy)
    assert len(rig.weld.records) == 1
    assert rig.weld._coping_observed == pytest.approx({1: 0., -1: 0.}, abs=1e-10)
    assert rig.weld._result['coverage_complete'] is False
    assert rig.weld._result['finding_status'] == 'incomplete'
    assert 'Fewer than two' in rig.weld._result['finding_error']
    assert not rig.weld.state(camera_state=rig.state)['replay_ready']


def test_cancellation_during_fitting_cannot_submit_a_later_orbit_move(coping):
    rig, _, _ = coping
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld._future_kind == 'coping_fit')
    rig.weld.abort_for_shutdown()
    before = len(rig.events)
    for _ in range(5):
        rig.advance()
    assert len(rig.events) == before and rig.weld.status == 'interrupted'


def test_coping_orbit_recovery_can_lower_inherited_scout_gain(coping):
    rig, _, _ = coping
    rig.weld.settings = WeldSettings(joint_kind='coped_tube')
    rig.weld._coping_phase = 'orbit'
    rig.weld._coping_scout_baseline_gain_db = 12.
    camera = {'controls': {'gain_db': 24.}, 'control_limits': {'gain_db': [0., 24.]}}
    bracket = rig.weld._joint_surface_bracket(camera, {'min_us': 100., 'max_us': 16000.})
    assert [row['gain_db'] for row in bracket] == [12., 12., 12., 18., 24.]
    assert camera['controls']['gain_db'] == 24.  # Planning a bracket does not configure hardware.


def test_reflective_scout_recovery_uses_frozen_locator_controls_without_gain_or_exposure_drift(coping):
    rig, _, _ = coping
    rig.weld.settings = WeldSettings(joint_kind='coped_tube', reflective_surface=True)
    rig.weld._coping_phase = 'scout'
    rig.weld._coping_scout_baseline_gain_db = 6.
    camera = {'controls': {'gain_db': 24., 'exposure_us': 16000.},
              'control_limits': {'gain_db': [0., 24.]}}
    bracket = rig.weld._joint_surface_bracket(camera, {'min_us': 100., 'max_us': 16000.})
    assert [row['exposure_us'] for row in bracket] == [16000., 16000., 16000., 8000., 16000.]
    assert [row['gain_db'] for row in bracket] == [0., 6., 12., 12., 18.]
    assert camera['controls'] == {'gain_db': 24., 'exposure_us': 16000.}


def test_metadata_only_state_never_constructs_or_deepcopies_retained_geometry(coping, monkeypatch):
    rig, _, _ = coping
    rig.weld._profiles['huge'] = [[1., 2., 3.]]*10000
    rig.weld._coping_extra_profiles = [[4., 5., 6.]]*10000
    monkeypatch.setattr(rig.weld, '_ordered_records', lambda: (_ for _ in ()).throw(AssertionError('Geometry traversed')))
    monkeypatch.setattr(rig.weld, '_measured_length_mm', lambda: 0.)
    state = rig.weld.state(camera_state=rig.state, include_geometry=False)
    assert state['geometry']['unchanged'] is True
    assert 'profile_points_base_m' not in state['geometry']
    assert state['profile_point_count'] == 20000
    assert len(json.dumps(state)) < 10000


def test_explicit_bootstrap_is_restricted_to_runtime_native_archives(coping):
    rig, _, _ = coping
    with pytest.raises(ValueError, match='runtime weld-paths'):
        rig.weld.start({'joint_kind': 'coped_tube', 'bootstrap_manifest': '/tmp/not-an-archive.json'})


def finish(coping):
    rig, _, _ = coping
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy)
    assert rig.weld.status in ('complete', 'incomplete'), rig.weld.error
    return rig


def test_live_completed_coping_replay_uses_its_own_fresh_pose_artifact(coping):
    rig = finish(coping)
    rig.weld.replay()
    pump(rig, lambda: rig.weld.status in ('complete', 'incomplete') and not rig.weld.busy)
    assert rig.weld._replay_index == len(rig.weld.records)
    assert len(rig.plans) == len(rig.weld.records)


@pytest.mark.parametrize('absolute_angle', [84., 88.])
def test_wrong_finite_root_is_rejected_even_when_both_radial_residuals_are_zero(coping, absolute_angle):
    from autoweld.coping_service import CopingObservationRejected

    rig = finish(coping)
    theta = np.deg2rad(absolute_angle)
    m = rig.weld._joint_model['model']
    center = np.asarray(m['center_mm'])
    point = center+np.array([-12.7*np.cos(theta), 12.7*np.cos(theta), 12.7*np.sin(theta)])
    frame = rig.weld._coping_seed_frame
    record = copy.deepcopy(rig.weld.records[len(rig.weld.records)//2])
    record['direction'] = 0
    record['seam_point_base_m'] = (frame[:3, :3] @ (point/1000.)+frame[:3, 3]).tolist()
    with pytest.raises(CopingObservationRejected, match='root sign'):
        rig.weld._coping_orient_record(record)


def test_current_profiles_cannot_be_outvoted_by_a_qualified_archived_fit(coping):
    rig, _, _ = coping
    original = rig.service.camera.fit_coped_joint

    def stale_fit(profiles, **kwargs):
        result = original(profiles, **kwargs)
        result['classification']['inlier_mask'] = [False]*len(result['classification']['inlier_mask'])
        result['classification']['residuals_mm'] = [[10., 10.]]*len(result['classification']['inlier_mask'])
        return result

    rig.service.camera.fit_coped_joint = stale_fit
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status == 'error' and not rig.weld.busy)
    assert rig.weld._joint_model['status'] == 'unobservable'
    assert rig.weld._joint_model['current_validation']['prior_cannot_outvote_current']
    assert not rig.weld._joint_model['current_validation']['qualified']
    assert rig.weld._coping_plan is None


def test_camera_session_change_during_preflight_cannot_submit_an_orbit_pose(coping):
    rig, _, _ = coping
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld._future_kind == 'coping_plan')
    before = len([event for event in rig.events if event[0] == 'move'])
    rig.state['acquisition_session_id'] = 'replacement-camera-session'
    rig.advance()
    assert rig.weld.status == 'error'
    assert len([event for event in rig.events if event[0] == 'move']) == before


def test_actual_angle_duplicate_or_reversal_is_not_added_to_observed_arc(coping):
    from autoweld.coping_service import CopingObservationRejected

    rig = finish(coping)
    record = copy.deepcopy(next(row for row in rig.weld.records if row['direction'] == 1))
    rig.weld._direction = 1
    rig.weld._coping_current = {'angle_deg': 25., 'tangent_base': [0., 1., 0.]}
    with pytest.raises(CopingObservationRejected, match='duplicate, reversal'):
        rig.weld._coping_orient_record(record)


def test_normal_request_returns_to_verified_rest_without_browser_advancement(coping):
    rig, _, _ = coping
    returned = []
    rig.service.ctrl.return_rest = lambda **kwargs: returned.append(kwargs)
    rig.service.ctrl.at_rest = lambda: True
    rig.service.ctrl.rest_verified = False
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status == 'coping_returning_rest')
    assert returned == [{'automatic': False}]
    assert rig.weld.busy and rig.weld._manifest_path().exists()
    rig.service.ctrl.rest_verified = True
    rig.advance()
    assert rig.weld.status in ('complete', 'incomplete') and not rig.weld.busy
    assert 'rest verified' in rig.weld.message


@pytest.mark.parametrize('selected_axis,expected_radii', [(0, (10., 20.)), (1, (20., 10.))])
def test_unequal_diameters_are_assigned_to_ordered_native_sides_using_selected_coping_axis(
        coping, selected_axis, expected_radii):
    rig, fits, _ = coping
    rig.weld.start({'joint_kind': 'coped_tube', 'coping_axis': selected_axis,
                    'branch_diameter_mm': 20., 'main_diameter_mm': 40., 'target_arc_deg': 40.})
    pump(rig, lambda: bool(fits))
    assert fits[0][1]['radii_mm'] == expected_radii
    rig.weld.abort_for_shutdown()


def test_native_133mm_initial_root_is_a_scout_and_first_move_preserves_its_laser_plane(coping):
    rig, fits, _ = coping
    original = rig.service.camera.find_weld_seam
    seed = np.array([1.6185, 0., 133.545])

    def find(cloud, **kwargs):
        result = original(cloud, **kwargs)
        if rig.weld._coping_phase == 'seed':
            result['seam_point_mm'] = seed.tolist()
        return result

    rig.service.camera.find_weld_seam = find
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld._coping_phase == 'scout')
    moves = [event[1] for event in rig.events if event[0] == 'move']
    assert len(moves) == 1 and not fits
    np.testing.assert_allclose(moves[0][:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(moves[0][:3, 3], seed/1000.-[0., 0., .150], atol=1e-12)
    record = rig.weld.records[0]
    assert record['provisional_seed_scout']
    assert rig.weld._native_files[0]['disposition'] == 'coping_scout'
    assert rig.weld._coping_coverage()['measured_station_count'] == 0
    assert rig.weld._coping_previous is None
    # Native root remains the actual first measurement, not a model/150 mm substitute.
    np.testing.assert_allclose(record['seam_point_base_m'], seed/1000.)
    rig.weld.abort_for_shutdown()


def test_previous_result_is_not_silently_reused_as_historical_xyz_bootstrap(coping):
    rig, _, _ = coping
    rig.weld._result = {'records': [{'native_path': '/old-object.json'}]}
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    assert rig.weld._coping_previous is None
    assert not rig.weld._coping_bootstrap['used']
    rig.weld.abort_for_shutdown()


def test_invalid_flat_features_can_supply_distinct_measured_surface_views_without_inventing_roots(coping):
    rig, fits, _ = coping
    original = rig.service.camera.find_weld_seam

    def find(cloud, **kwargs):
        if rig.weld._coping_phase == 'scout':
            return dict(status='invalid', found=False, reason='curved surface is not a qualified flat flank')
        return original(cloud, **kwargs)

    rig.service.camera.find_weld_seam = find
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: bool(fits))
    profiles = fits[0][0]
    assert len(profiles) == 3
    assert len({profile['provenance']['physical_scout_index'] for profile in profiles}) == 3
    assert all('seam_point_mm' not in profile for profile in profiles)
    assert all(profile['provenance']['split_hint_is_measured_seam'] is False for profile in profiles)
    assert all(profile['provenance']['local_feature_status'] == 'invalid' for profile in profiles)
    assert all(len(profile['surface_labels']) == len(profile['xyz_mm']) for profile in profiles)
    assert len(rig.weld.records) == 1 and rig.weld.records[0]['provisional_seed_scout']
    assert rig.weld._coping_coverage()['observed_arc_deg'] == 0.
    assert rig.weld._coping_coverage()['measured_station_count'] == 0
    assert len([row for row in rig.weld._native_files if row['disposition'] == 'coping_surface_profile']) == 3
    rig.weld.abort_for_shutdown()


def test_surface_selection_keeps_best_balanced_native_profile_once_per_physical_view(coping):
    rig, _, _ = coping
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld._coping_phase == 'scout')
    measurement = copy.deepcopy(rig.weld._pending_measurement)
    measurement['base_from_camera'] = copy.deepcopy(rig.weld._coping_current['base_from_tcp'])
    expected = np.asarray(rig.weld._coping_current['predicted_point_base_m'])
    camera = np.asarray(measurement['base_from_camera'])
    hint = camera[:3, :3].T @ (expected-camera[:3, 3])*1000.
    large = hint+np.column_stack((np.linspace(-20., 20., 1001), np.zeros(1001), np.zeros(1001)))
    for path, points, status in [('complete-native', large, 'invalid'),
                                 ('truncated-but-found', large[480:521], 'found')]:
        rig.weld._pending_native_path = rig.path/path
        rig.weld._native_files.append(dict(path=str(rig.weld._pending_native_path), sha256=path))
        measurement['cloud']['xyz_mm'] = points.tolist()
        measurement['cloud']['uv'] = np.column_stack((np.arange(len(points)), np.zeros(len(points)))).tolist()
        rig.weld._joint_scout_surface(measurement, {'status': status, 'seam_point_mm': hint.tolist()})
    assert rig.weld._coping_accept_scout_surface()
    assert not rig.weld._coping_accept_scout_surface()
    profile = rig.weld._coping_profiles[0]
    assert profile['profile_id'].endswith('complete-native')
    np.testing.assert_allclose(profile['xyz_mm'], large)
    assert profile['provenance']['native_point_count'] == 1001
    assert profile['provenance']['local_feature_status'] == 'invalid'
    assert 'seam_point_mm' not in profile
    assert len(rig.weld.records) == 1
    rig.weld.abort_for_shutdown()


def test_surface_membership_profiles_restore_without_a_fabricated_measured_seam(coping):
    from test_weld_seam_service import ImmediateExecutor

    from autoweld.weld_seam_service import WeldSeamService

    rig = finish(coping)
    assert all('seam_point_mm' not in profile for profile in rig.weld._coping_profiles)
    original = rig.weld.state(camera_state=rig.state)['geometry']['profile_points_base_m']
    before = len(rig.events)
    rig.weld.close()
    restored = WeldSeamService(rig.service, clock=lambda: rig.clock[0], executor=ImmediateExecutor())
    assert restored.status in ('complete', 'incomplete'), restored.error
    np.testing.assert_allclose(restored._coping_seed_mm, rig.weld._coping_seed_mm)
    actual = np.asarray(restored.state(camera_state=rig.state)['geometry']['profile_points_base_m'])
    original = np.asarray(original)
    np.testing.assert_allclose(actual[np.lexsort(actual.T)], original[np.lexsort(original.T)])
    assert len(rig.events) == before
    restored.close()


def test_unstable_offset_fit_collects_nine_distinct_surface_views_without_forcing_qualification(coping):
    rig, fits, plans = coping
    original = rig.service.camera.fit_coped_joint

    def unstable(profiles, **kwargs):
        result = original(profiles, **kwargs)
        result.update(status='unobservable', quality={'reason': 'offset changes across independent profile folds'})
        return result

    rig.service.camera.fit_coped_joint = unstable
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status == 'error' and not rig.weld.busy)
    assert [len(profiles) for profiles, _ in fits] == list(range(3, 10))
    assert not plans
    assert [row['scout_offset_mm'] for row in rig.weld._coping_scouts] == [0., 3., -3., 5., -5., 7., -7., 9., -9.]
    assert max(abs(row['scout_view_tilt_deg']) for row in rig.weld._coping_scouts) == 10.
    assert len({profile['provenance']['physical_scout_index'] for profile in fits[-1][0]}) == 9
    assert len(rig.weld.records) == 1 and rig.weld.records[0]['provisional_seed_scout']
    assert rig.weld._coping_coverage()['observed_arc_deg'] == 0.
    assert 'not qualified' in rig.weld.error


def test_each_surface_scout_finishes_bracket_after_found_root_and_resets_gain(coping):
    rig, fits, _ = coping
    rig.state['control_limits'] = {'gain_db': [0., 24.]}
    rig.state['controls']['gain_db'] = 12.
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: bool(fits))
    assert rig.weld._coping_scout_baseline_gain_db == 12.
    assert len(fits[0][0]) == 3
    assert len(rig.weld._native_files) >= 16  # Locator and five configs per view.
    brackets = [row for row in rig.weld._bracket_history if row['kind'] in ('gain', 'exposure')]
    assert len(brackets) == 15
    for offset in (0, 5, 10):
        group = brackets[offset:offset+5]
        assert [row['requested_gain_db'] for row in group] == [12., 12., 12., 18., 24.]
        assert [row['requested_us'] for row in group] == [16000., 8000., 4000., 16000., 16000.]
    assert len([row for row in rig.weld._native_files if row['disposition'] == 'coping_surface_profile']) == 3
    assert len(rig.weld.records) == 1 and rig.weld.records[0]['provisional_seed_scout']
    rig.weld.abort_for_shutdown()


def test_unassociated_orbit_corner_retries_same_hold_before_ending_direction(coping):
    rig, _, _ = coping
    original = rig.service.camera.find_weld_seam
    attempts = []

    def find(cloud, **kwargs):
        result = original(cloud, **kwargs)
        if rig.weld._coping_phase == 'orbit' and not rig.weld._coping_current.get('seed_recheck') and len(attempts) < 2:
            attempts.append(sum(event[0] in ('move', 'hold') for event in rig.events))
            if len(attempts) == 1:
                result['seam_point_mm'] = [10., 0., 150.]
        return result

    rig.service.camera.find_weld_seam = find
    rig.weld.start({'joint_kind': 'coped_tube', 'target_arc_deg': 40.})
    pump(rig, lambda: rig.weld.status in rig.weld.TERMINAL and not rig.weld.busy)
    assert rig.weld.status in ('complete', 'incomplete'), rig.weld.error
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert any(row['disposition'] == 'coping_unassociated_root' for row in rig.weld._native_files)
    assert rig.weld._coping_coverage()['observed_arc_deg'] > 30.
