"""Pure calibrated sensing geometry and worker-owned whole-pose coping proof."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.coping_plan import (
    coping_camera_pose,
    laser_seam_ray,
    measured_angle_deg,
    plan_coping_views,
    transport_measured_camera_pose,
)
from autoweld.motion import MotionLimits


def test_laser_ray_uses_real_plane_offset_and_exact_range():
    plane = [0., 1., .2, -12.]
    ray = laser_seam_ray(plane)
    assert np.linalg.norm(ray) == pytest.approx(150.)
    assert np.dot(plane[:3], ray)+plane[3] == pytest.approx(0., abs=1e-10)
    assert not np.allclose(ray, [0, 0, 150])


@pytest.mark.parametrize('angle', [-70., -20., 0., 30., 80.])
def test_full_camera_pose_reorients_and_keeps_actual_target_on_calibrated_stripe(angle):
    a = np.deg2rad(angle)
    point = np.array([.0127*np.sin(a), .01, .2+.0127*np.cos(a)])
    inward = np.array([-np.sin(a), 0., -np.cos(a)])
    tangent = np.array([np.cos(a), .2, -np.sin(a)])
    plane = [0., 1., .3, -10.]
    h = np.eye(4)
    h[:3, 3] = [.02, -.01, .03]
    tcp, camera = coping_camera_pose(point, tangent, inward, plane, h, normal_sign=1)
    local = camera[:3, :3].T @ (point-camera[:3, 3])*1000.
    assert np.linalg.norm(point-camera[:3, 3]) == pytest.approx(.15)
    assert np.dot(plane[:3], local)+plane[3] == pytest.approx(0., abs=1e-8)
    normal_world = camera[:3, :3] @ (np.asarray(plane[:3])/np.linalg.norm(plane[:3]))
    assert normal_world @ (tangent/np.linalg.norm(tangent)) == pytest.approx(1.)
    np.testing.assert_allclose(tcp @ h, camera)
    assert np.linalg.det(camera[:3, :3]) == pytest.approx(1.)


def test_measured_angle_comes_from_actual_radial_point_not_nominal_station():
    axis = {'origin_mm': [0., 0., 0.], 'direction': [0., 0., 1.], 'radius_mm': 12.7}
    assert measured_angle_deg([0., 12.7, 200.], [12.7, 0., -10.], axis) == pytest.approx(90.)


def test_sensing_view_can_keep_visible_side_while_torch_direction_points_opposite():
    point, tangent, inward = np.array([0., 0., .15]), [0., 1., 0.], np.array([0., 0., -1.])
    plane = [0., 1., .2, -12.]
    _, camera = coping_camera_pose(point, tangent, inward, plane, np.eye(4), normal_sign=1,
        view_direction_base=[0., 0., 1.])
    local = camera[:3, :3].T @ (point-camera[:3, 3])*1000.
    assert np.linalg.norm(local) == pytest.approx(150.)
    assert np.dot(plane[:3], local)+plane[3] == pytest.approx(0., abs=1e-8)
    assert camera[:3, 2] @ inward < -.9
    np.testing.assert_array_equal(inward, [0., 0., -1.])
    # The caller's inward reference remains available for the welding pose.
    _, radial_camera = coping_camera_pose(point, tangent, inward, plane, np.eye(4), normal_sign=1)
    assert radial_camera[:3, 2] @ inward > .9


@pytest.mark.parametrize('angle,orientation', [(0., 1.), (-30., 1.), (30., -1.)])
def test_full_measured_camera_transport_keeps_roll_and_exact_calibrated_range(angle, orientation):
    reference, handeye = np.eye(4), np.eye(4)
    reference[:3, :3] = Rotation.from_euler('xyz', [25., 12., -8.], degrees=True).as_matrix()
    reference[:3, 3] = [.1, -.2, .3]
    handeye[:3, 3] = [.02, -.01, .015]
    axis, point, plane = [0., 0., 1.], np.array([.1, .05, .2]), [0., 1., .2, -12.]
    expected = Rotation.from_rotvec(np.array(axis)*np.deg2rad(angle)*orientation).as_matrix() @ reference[:3, :3]
    # This is a supported oblique cross-section, not a perpendicular one.
    tangent = expected @ np.array([.6, .8, 0.])
    tcp, camera = transport_measured_camera_pose(point, tangent, plane, handeye, reference, axis, angle,
        angular_orientation=orientation)
    np.testing.assert_allclose(camera[:3, :3], expected, atol=1e-15)
    np.testing.assert_allclose(tcp @ handeye, camera, atol=1e-15)
    local = camera[:3, :3].T @ (point-camera[:3, 3])*1000.
    assert np.linalg.norm(local) == pytest.approx(150.)
    assert np.dot(plane[:3], local)+plane[3] == pytest.approx(0., abs=1e-8)
    crossing = (camera[:3, :3] @ (np.array(plane[:3])/np.linalg.norm(plane[:3]))) @ tangent
    assert .5 < crossing < 1.


@pytest.mark.parametrize('tilt', [-5., 5.])
def test_transported_recovery_rotates_about_actual_laser_normal_without_changing_crossing(tilt):
    reference, plane, point = np.eye(4), [0., 1., .2, -12.], [.1, .05, .2]
    reference[:3, :3] = Rotation.from_euler('xyz', [25., 12., -8.], degrees=True).as_matrix()
    normal = reference[:3, :3] @ (np.array(plane[:3])/np.linalg.norm(plane[:3]))
    _, camera = transport_measured_camera_pose(point, normal, plane, np.eye(4), reference, [0., 0., 1.], 0.,
        view_tilt_deg=tilt)
    expected = Rotation.from_rotvec(normal*np.deg2rad(tilt)).as_matrix() @ reference[:3, :3]
    np.testing.assert_allclose(camera[:3, :3], expected, atol=1e-15)
    np.testing.assert_allclose(camera[:3, :3] @ (np.array(plane[:3])/np.linalg.norm(plane[:3])), normal, atol=1e-15)
    assert np.linalg.norm(np.asarray(point)-camera[:3, 3]) == pytest.approx(.15)


def test_measured_camera_transport_rejects_insufficient_laser_seam_crossing():
    with pytest.raises(ValueError, match='less than 30') as error:
        transport_measured_camera_pose([0., 0., .15], [1., 0., 0.], [0., 1., 0., 0.], np.eye(4),
            np.eye(4), [0., 0., 1.], 0.)
    assert error.value.boundary_kind == 'laser_seam_crossing_unobservable'
    assert error.value.diagnostic['minimum_absolute_normal_dot_tangent'] == .5


@pytest.fixture
def planning(monkeypatch):
    from autoweld import diagnostics, scan_plan_probe

    events = []
    options = {'reject_positive_family': False, 'reject_radial_view': False}
    source = SimpleNamespace(limits=MotionLimits())

    def solve(kin, pose, site, seed, **kwargs):
        events.append(('solve', pose.copy()))
        q = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec(), 0.]
        if options['reject_positive_family'] and pose[1, 1] > 0:
            raise ValueError('Installed wrist bounds reject one normal family')
        if options['reject_radial_view'] and pose[2, 2] < 0:
            raise ValueError('Installed reach rejects the opposite-side radial sensing view')
        if 'allowed_view_tilts' in options:
            tilt = np.rad2deg(np.arctan2(pose[0, 2], pose[2, 2]))*np.sign(pose[1, 1])
            if not any(np.isclose(tilt, allowed, atol=1e-10) for allowed in options['allowed_view_tilts']):
                raise ValueError('Installed wrist rejects this nominal or tilted sensing pose')
        return q

    def plan(start, target, **kwargs):
        if options.get('before_plan'):
            options['before_plan']()
        kwargs['validate'](start)
        kwargs['validate'](target)
        assert not kwargs['collision'](target)
        events.append(('plan', kwargs['cartesian'].copy()))
        if options.get('after_plan'):
            options['after_plan']()
        return SimpleNamespace(report={'all_derivatives_and_bounds_checked': True})

    private = SimpleNamespace(limits=source.limits, kin='private-data', model='shared-model', data='private-data', site_name='weld_tcp',
        _configuration=lambda q: q.copy(), fk=lambda q: np.eye(4), plan=plan)
    monkeypatch.setattr(scan_plan_probe, 'private_planner', lambda shared: private)
    monkeypatch.setattr(scan_plan_probe, 'private_self_collision', lambda planner, q: False)
    monkeypatch.setattr(diagnostics, 'solve_pose', solve)
    return source, events, options


def curve():
    return dict(points_mm=[[-3., 0., 150.], [0., 0., 150.], [3., 0., 150.]],
        tangents=[[0., 1., 0.]]*3, inward_directions=[[0., 0., 1.]]*3,
        relative_angles_deg=[-10., 0., 10.], unresolved=[])


def test_preflight_checks_both_continuous_normal_families_and_retains_all_full_pose_proofs(planning):
    source, events, _ = planning
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
                               [0., 1., 0., 0.], curve())
    assert len(result['normal_family_comparison']) == 2
    assert result['planned_arc_deg'] == 20.
    assert result['branches']['positive'][0]['normal_sign'] == result['normal_sign']
    assert result['branches']['negative'][0]['normal_sign'] == result['normal_sign']
    assert all(station['proof']['all_derivatives_and_bounds_checked'] for station in
               [result['center'], *result['branches']['positive'], *result['branches']['negative']])
    assert len([event for event in events if event[0] == 'plan']) == 18  # 2 families × 3 stations × primary/2 tilts.


def test_analytic_cusp_between_sample_nodes_is_a_geometry_boundary_not_robot_limit(planning):
    source, _, _ = planning
    c = curve()
    c['unresolved'] = [{'relative_angle_deg': 5., 'reason': 'crown_cusp'}]
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
                               [0., 1., 0., 0.], c)
    assert result['branches']['positive'] == []
    assert result['boundaries']['positive']['kind'] == 'analytic_intersection_unobservable'
    assert result['boundaries']['positive']['angle_deg'] == 5.
    assert result['negative_bound_deg'] == -10.


def test_cancelled_preflight_cannot_return_an_executable_artifact(planning):
    source, _, _ = planning
    with pytest.raises(TimeoutError):
        plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
                          [0., 1., 0., 0.], curve(), cancelled=lambda: True)


def test_center_ik_failure_tries_the_other_laser_normal_family_without_flipping_later(planning):
    source, _, options = planning
    options['reject_positive_family'] = True
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
                               [0., 1., 0., 0.], curve())
    assert result['normal_sign'] == -1
    assert result['normal_family_failures'][0]['normal_sign'] == 1
    assert all(row['normal_sign'] == -1 for row in result['branches']['positive']+result['branches']['negative'])


@pytest.mark.parametrize('orientation', [1., -1.])
def test_visible_side_is_transported_around_branch_without_replacing_inward_reference(planning, monkeypatch, orientation):
    from autoweld import coping_clearance

    source, events, options = planning
    options['reject_radial_view'] = True
    clearance_calls = []
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance',
        lambda model, data, tubes: clearance_calls.append((model, data, tubes)))
    angles = np.array([-10., 0., 10.])
    phi = np.deg2rad(angles)*orientation
    radial = np.column_stack((np.sin(phi), np.zeros(3), np.cos(phi)))
    tangents = np.column_stack((np.cos(phi), np.zeros(3), -np.sin(phi)))
    c = dict(points_mm=(radial*12.7+[0., 0., 150.]).tolist(),
        inward_directions=(-radial).tolist(), tangents=tangents.tolist(),
        relative_angles_deg=angles.tolist(), unresolved=[], coping_axis=0, orientation=orientation)
    tubes = [dict(origin_m=[0., 0., .15], direction=[0., 1., 0.], radius_m=.0127)]*2
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], c, fitted_tubes=tubes)
    assert result['sensing_view_family'] == 'measured_visible_side'
    assert len(result['normal_family_comparison']) == 10
    assert result['base_view_tilt_deg'] == 0.
    assert all(row['sensing_view_family'] == 'radial' for row in result['normal_family_failures'])
    stations = [result['center'], *result['branches']['positive'], *result['branches']['negative']]
    for station in stations:
        index = int(np.flatnonzero(angles == station['angle_deg'])[0])
        camera = np.asarray(station['base_from_camera'])
        np.testing.assert_allclose(camera[:3, 2], radial[index], atol=1e-12)
        np.testing.assert_allclose(station['inward_base'], -radial[index], atol=1e-12)
        assert station['sensing_view_family'] == 'measured_visible_side'
        assert station['proof']['all_derivatives_and_bounds_checked']
        point = np.asarray(station['predicted_point_base_m'])
        assert np.linalg.norm(point-camera[:3, 3]) == pytest.approx(.15)
        assert (camera[:3, :3].T @ (point-camera[:3, 3]))[1] == pytest.approx(0., abs=1e-12)
    assert clearance_calls
    assert all(call[:2] == ('shared-model', 'private-data') and call[2] is tubes for call in clearance_calls)
    assert len([event for event in events if event[0] == 'plan']) == 78
    assert result['sensing_reference']['angular_orientation'] == orientation


def test_visible_side_still_fails_when_full_workpiece_clearance_proof_rejects(planning, monkeypatch):
    from autoweld import coping_clearance

    source, _, _ = planning
    c = {**curve(), 'coping_axis': 0, 'orientation': 1.}
    tubes = [dict(direction=[0., 0., 1.])]*2
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance', lambda *_: {'clearance_mm': -2.})
    with pytest.raises(ValueError, match='Both coping camera normal families') as error:
        plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
            [0., 1., 0., 0.], c, fitted_tubes=tubes)
    failures = error.value.diagnostic['center_failures']
    assert len(failures) == 12
    assert {row['sensing_view_family'] for row in failures} == {'radial', 'measured_visible_side'}
    assert all(row['diagnostic']['clearance_mm'] == -2. for row in failures)


def test_primary_nonzero_tilt_family_is_tried_when_zero_tilt_endpoint_is_unreachable(planning, monkeypatch):
    from autoweld import coping_clearance

    source, _, options = planning
    options.update(reject_positive_family=True, allowed_view_tilts={-5.})
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance', lambda *_: None)
    c = {**curve(), 'coping_axis': 0, 'orientation': 1.}
    tubes = [dict(direction=[0., 0., 1.])]*2
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], c, fitted_tubes=tubes)
    assert result['sensing_view_family'] == 'measured_visible_side'
    assert result['base_view_tilt_deg'] == -5.
    assert result['normal_sign'] == -1
    assert result['planned_arc_deg'] == 20.
    for station in [result['center'], *result['branches']['positive'], *result['branches']['negative']]:
        assert station['base_view_tilt_deg'] == -5.
        assert station['proof']['all_derivatives_and_bounds_checked']
    assert any(row['sensing_view_family'] == 'measured_visible_side' and row['base_view_tilt_deg'] == 0.
               for row in result['normal_family_failures'])


def test_recovery_tilts_are_relative_to_primary_and_never_exceed_ten_degrees(planning, monkeypatch):
    from autoweld import coping_clearance

    source, _, _ = planning
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance', lambda *_: None)
    families = []
    # Inspect each complete family before selection without changing geometry
    # or bypassing the ordinary planner/validation fixture.
    from autoweld import coping_plan
    original_max = max

    def capture_max(values, **kwargs):
        values = list(values)
        if values and isinstance(values[0], dict):
            families.extend(values)
        return original_max(values, **kwargs)

    monkeypatch.setattr(coping_plan, 'max', capture_max, raising=False)
    c = {**curve(), 'coping_axis': 0, 'orientation': 1.}
    plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], c, fitted_tubes=[dict(direction=[0., 0., 1.])]*2)
    for family in families:
        base = family['base_view_tilt_deg']
        if family['sensing_view_family'] == 'radial':
            assert base == 0.
        for station in [family['center'], *family['branches']['positive'], *family['branches']['negative']]:
            assert station['base_view_tilt_deg'] == base
            for alternate in station['alternatives']:
                assert alternate['relative_view_tilt_deg'] in (-5., 5.)
                assert alternate['view_tilt_deg'] == base+alternate['relative_view_tilt_deg']
                assert abs(alternate['view_tilt_deg']) <= 10.
                assert alternate['proof']['all_derivatives_and_bounds_checked']
            assert len(station['alternatives']) == (1 if abs(base) == 10. else 2)


@pytest.mark.parametrize('expire_after_complete_plan, expected_arc', [(5, 10.), (10, 20.)])
def test_optional_preflight_deadline_retains_only_completed_center_and_station_proofs(
        planning, monkeypatch, expire_after_complete_plan, expected_arc):
    from autoweld import coping_plan

    source, events, options = planning
    clock = {'now': 0.}
    monkeypatch.setattr(coping_plan, 'time', SimpleNamespace(monotonic=lambda: clock['now']))

    def finish_plan():
        if len([event for event in events if event[0] == 'plan']) >= expire_after_complete_plan:
            clock['now'] = 2.

    options['after_plan'] = finish_plan
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], curve(), max_seconds=1.)
    assert result['planned_arc_deg'] == expected_arc
    assert result['preflight_budget_exhausted']
    assert result['center']['proof']['all_derivatives_and_bounds_checked']
    assert all(station['proof']['all_derivatives_and_bounds_checked'] for station in
               result['branches']['positive']+result['branches']['negative'])
    if expected_arc == 10.:
        assert len(result['branches']['positive']) == 1
        assert result['branches']['negative'] == []
        assert result['boundaries']['negative']['kind'] == 'preflight_time_budget'
        assert result['boundaries']['negative']['unfinished_station_included'] is False


def test_deadline_during_primary_motion_excludes_that_unfinished_station(planning, monkeypatch):
    from autoweld import coping_plan

    source, events, options = planning
    clock = {'now': 0.}
    monkeypatch.setattr(coping_plan, 'time', SimpleNamespace(monotonic=lambda: clock['now']))

    def before_plan():
        if len([event for event in events if event[0] == 'plan']) >= 3:
            clock['now'] = 2.

    options['before_plan'] = before_plan
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], curve(), max_seconds=1.)
    assert result['planned_arc_deg'] == 0.
    assert result['branches'] == {'positive': [], 'negative': []}
    assert result['preflight_budget_exhausted']
    assert all(boundary['kind'] == 'preflight_time_budget' for boundary in result['boundaries'].values())


def test_cancellation_after_completed_center_still_rejects_every_artifact(planning):
    source, _, options = planning
    cancelled = {'value': False}
    options['after_plan'] = lambda: cancelled.update(value=True)
    with pytest.raises(TimeoutError, match='cancelled'):
        plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
            [0., 1., 0., 0.], curve(), cancelled=lambda: cancelled['value'])


def test_demonstrated_camera_family_is_preferred_and_does_not_replace_inward_torch_direction(planning, monkeypatch):
    from autoweld import coping_clearance

    source, _, _ = planning
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance', lambda *_: None)
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler('y', 25., degrees=True).as_matrix()
    c = {**curve(), 'coping_axis': 0, 'orientation': -1., 'inward_directions': [[0., 0., -1.]]*3}
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], c, fitted_tubes=[dict(direction=[0., 1., 0.])]*2,
        measured_camera_reference=reference)
    assert result['sensing_view_family'] == 'transport_measured_camera'
    assert result['base_view_tilt_deg'] == 0.
    assert len(result['normal_family_comparison']) == 1
    assert result['planned_arc_deg'] == 20.
    np.testing.assert_allclose(result['center']['base_from_camera'],
        transport_measured_camera_pose([0., 0., .15], [0., 1., 0.], [0., 1., 0., 0.], np.eye(4),
            reference, [0., 1., 0.], 0.)[1])
    for station in [result['center'], *result['branches']['positive'], *result['branches']['negative']]:
        expected = Rotation.from_euler('y', -station['angle_deg'], degrees=True).as_matrix() @ reference[:3, :3]
        np.testing.assert_allclose(np.asarray(station['base_from_camera'])[:3, :3], expected)
        assert station['inward_base'] == [0., 0., -1.]
        assert station['proof']['all_derivatives_and_bounds_checked']
        assert abs(station['laser_plane_normal_dot_seam_tangent']) >= .5
    assert result['sensing_reference']['measured_camera_reference_base_m'] == reference.tolist()


def test_transported_plane_crossing_limit_retains_only_the_proven_observable_side(planning, monkeypatch):
    from autoweld import coping_clearance

    source, _, _ = planning
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance', lambda *_: None)
    c = {**curve(), 'coping_axis': 0, 'orientation': 1.,
         'tangents': [[0., 1., 0.], [0., 1., 0.], [1., 0., 0.]]}
    result = plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
        [0., 1., 0., 0.], c, fitted_tubes=[dict(direction=[0., 0., 1.])]*2,
        measured_camera_reference=np.eye(4))
    assert result['sensing_view_family'] == 'transport_measured_camera'
    assert result['planned_arc_deg'] == 10.
    assert result['branches']['positive'] == []
    assert len(result['branches']['negative']) == 1
    boundary = result['boundaries']['positive']
    assert boundary['kind'] == 'laser_seam_crossing_unobservable'
    assert abs(boundary['diagnostic']['signed_normal_dot_tangent']) < .5


def test_unobservable_measured_reference_can_fall_back_but_never_bypasses_full_clearance(planning, monkeypatch):
    from autoweld import coping_clearance

    source, _, _ = planning
    monkeypatch.setattr(coping_clearance, 'check_tube_clearance', lambda *_: {'clearance_mm': -2.})
    c = {**curve(), 'coping_axis': 0, 'orientation': 1.}
    with pytest.raises(ValueError) as error:
        plan_coping_views(source, np.zeros(7), np.tile([-10., 10.], (6, 1)), np.eye(4), np.eye(4),
            [0., 1., 0., 0.], c, fitted_tubes=[dict(direction=[0., 0., 1.])]*2,
            measured_camera_reference=np.eye(4))
    failures = error.value.diagnostic['center_failures']
    assert len(failures) == 13
    assert failures[0]['sensing_view_family'] == 'transport_measured_camera'
    assert all(row['diagnostic']['clearance_mm'] == -2. for row in failures)
