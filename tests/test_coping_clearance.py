"""Analytic workpiece bounds use fake arrays, never a compiled robot or devices."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.coping_clearance import check_tube_clearance, fitted_tube_clearance


def rig():
    # World decoration/floor, contactless visual mesh, arm mesh, zero-mass mount
    # box, then a contact-enabled plane. Only arm and mount are physical bounds.
    model = SimpleNamespace(geom_rbound=np.array([100., .1, .02, .01, 100.]),
        geom_bodyid=np.array([0, 1, 1, 2, 2]), geom_type=np.array([0, 7, 7, 6, 0]),
        geom_contype=np.array([1, 0, 1, 1, 1]), geom_conaffinity=np.array([1, 0, 1, 0, 1]),
        body_mass=np.array([0., 1., 0.]), geom_names=['floor', 'visual', 'arm_mesh', 'mount_camera', 'plane'],
        body_names=['world', 'arm', 'gripper'])
    data = SimpleNamespace(geom_xpos=np.array([[0., 0., 0.], [0., 0., 0.], [.20, 0., 0.], [.16, 0., 0.], [0., 0., 0.]]))
    tubes = [dict(origin_m=[0., 0., 0.], direction=[0., 0., 1.], radius_m=.0127, label='branch'),
             dict(origin_m=[0., 1., 0.], direction=[1., 0., 0.], radius_m=.0127, label='main')]
    return model, data, tubes


def test_separated_physical_geom_bounds_return_no_workpiece_blocker():
    model, data, tubes = rig()
    assert check_tube_clearance(model, data, tubes) is None
    assert fitted_tube_clearance(model, data, tubes) == (False, None)


def test_zero_mass_collision_mount_is_included_and_reports_a_conservative_bound_not_contact():
    model, data, tubes = rig()
    data.geom_xpos[3] = [.02, 0., .4]
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['geom_name'] == 'mount_camera' and diagnostic['body_name'] == 'gripper'
    assert diagnostic['tube_label'] == 'branch' and diagnostic['checked_geom_count'] == 2
    assert diagnostic['clearance_mm'] == pytest.approx(-4.7)
    assert diagnostic['margin_mm'] == 2.
    assert diagnostic['reason'] == 'conservative fitted tube clearance'
    assert diagnostic['actual_surface_contact_confirmed'] is False
    assert 'infinite' in diagnostic['bound_kind'] and 'physically clear' in diagnostic['limitation']
    assert fitted_tube_clearance(model, data, tubes)[0] is True


def test_collision_flags_not_mass_or_transparency_select_physical_geoms():
    model, data, tubes = rig()
    data.geom_xpos[3] = [0., 0., 0.]
    model.geom_contype[3], model.geom_conaffinity[3] = 0, 1
    assert check_tube_clearance(model, data, tubes)['geom_name'] == 'mount_camera'
    model.geom_conaffinity[3] = 0
    assert check_tube_clearance(model, data, tubes) is None


def test_world_decoration_contactless_visuals_and_plane_geoms_are_excluded():
    model, data, tubes = rig()
    data.geom_xpos[[0, 1, 4]] = [0., 0., 0.]
    assert check_tube_clearance(model, data, tubes) is None


def test_second_infinite_tube_is_checked_even_far_along_its_unseen_extension():
    model, data, tubes = rig()
    data.geom_xpos[3] = [4., 1.01, 0.]
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['tube_index'] == 1 and diagnostic['tube_label'] == 'main'
    assert diagnostic['radial_center_distance_mm'] == pytest.approx(10.)
    assert diagnostic['clearance_mm'] == pytest.approx(-14.7)


def test_margin_is_applied_once_in_metres_and_touching_the_inflated_bound_rejects():
    model, data, tubes = rig()
    data.geom_xpos[3] = [.024, 0., 0.]
    assert check_tube_clearance(model, data, tubes, margin_m=0.) is None
    assert check_tube_clearance(model, data, tubes, margin_m=.002)['clearance_mm'] == pytest.approx(-.7)
    data.geom_xpos[3, 0] = .025
    assert check_tube_clearance(model, data, tubes, margin_m=.002) is None


def test_rotating_and_translating_all_geometry_preserves_the_bound():
    model, data, tubes = rig()
    data.geom_xpos[3] = [.02, 0., .4]
    first = check_tube_clearance(model, data, tubes)
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    translation = np.array([.5, -.25, .7])
    data.geom_xpos = data.geom_xpos @ rotation.T+translation
    for tube in tubes:
        tube['origin_m'] = rotation @ tube['origin_m']+translation
        tube['direction'] = rotation @ tube['direction']
    second = check_tube_clearance(model, data, tubes)
    assert second['clearance_mm'] == pytest.approx(first['clearance_mm'])
    assert second['geom_name'] == first['geom_name'] and second['tube_index'] == first['tube_index']


def test_axis_is_normalized_without_mutating_inputs_and_compatibility_aliases_work():
    model, data, tubes = rig()
    data.geom_xpos[3] = [.02, 0., .4]
    tubes[0]['direction'] = np.array([0., 0., 7.])
    before = tubes[0]['direction'].copy()
    diagnostic = check_tube_clearance(model, data, tubes)
    np.testing.assert_array_equal(tubes[0]['direction'], before)
    assert diagnostic['axis_direction_base'] == [0., 0., 1.]
    aliases = [dict(origin_base_m=tube['origin_m'], axis_base=tube['direction'],
        radius_m=tube['radius_m'], label=tube['label']) for tube in tubes]
    collision, other = fitted_tube_clearance(model, data, aliases)
    assert collision and other['clearance_mm'] == diagnostic['clearance_mm']


@pytest.mark.parametrize('bad', [dict(direction=[0., 0., 0.]), dict(direction=[0., 0., float('nan')]),
    dict(origin_m=[0., 0.]), dict(radius_m=0.), dict(radius_m=-.01), dict(radius_m=float('inf'))])
def test_malformed_axis_or_radius_cannot_silently_pass_clearance(bad):
    model, data, tubes = rig()
    tubes[0].update(bad)
    with pytest.raises(ValueError):
        check_tube_clearance(model, data, tubes)


@pytest.mark.parametrize('margin', [-.001, float('nan'), float('inf'), True])
def test_margin_validation_rejects_invalid_or_boolean_values(margin):
    model, data, tubes = rig()
    with pytest.raises(ValueError):
        check_tube_clearance(model, data, tubes, margin_m=margin)


def test_missing_bounds_nonfinite_physical_state_and_empty_selection_fail_closed():
    model, data, tubes = rig()
    model.geom_rbound[3] = np.nan
    with pytest.raises(ValueError, match='finite measured positions'):
        check_tube_clearance(model, data, tubes)
    model, data, tubes = rig()
    data.geom_xpos[2, 1] = np.nan
    with pytest.raises(ValueError, match='finite measured positions'):
        check_tube_clearance(model, data, tubes)
    model, data, tubes = rig()
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    with pytest.raises(ValueError, match='No physical robot collision geoms'):
        check_tube_clearance(model, data, tubes)
    del model.geom_rbound
    with pytest.raises(ValueError, match='missing required geom clearance arrays'):
        check_tube_clearance(model, data, tubes)


def test_worst_bound_is_reported_without_mutating_shared_model_or_calling_fk():
    model, data, tubes = rig()
    data.geom_xpos[2] = [.01, 0., 0.]
    data.geom_xpos[3] = [.02, 0., 0.]
    before_model, before_data, before_tubes = copy.deepcopy(model), copy.deepcopy(data), copy.deepcopy(tubes)
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['geom_name'] == 'arm_mesh'
    for name in ('geom_rbound', 'geom_contype', 'geom_conaffinity', 'body_mass'):
        np.testing.assert_array_equal(getattr(model, name), getattr(before_model, name))
    np.testing.assert_array_equal(data.geom_xpos, before_data.geom_xpos)
    assert tubes == before_tubes
    source = Path('autoweld/coping_clearance.py').read_text()
    modules = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    modules.extend(alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for alias in node.names)
    assert modules == ['numpy']


def test_invalid_geom_metadata_or_missing_second_axis_is_rejected_before_selection():
    model, data, tubes = rig()
    model.geom_bodyid = model.geom_bodyid.astype(float)
    model.geom_bodyid[3] = np.nan
    with pytest.raises(ValueError, match='integer metadata'):
        check_tube_clearance(model, data, tubes)
    model, data, tubes = rig()
    with pytest.raises(ValueError, match='exactly two'):
        check_tube_clearance(model, data, tubes[:1])


def box_rig():
    model, data, tubes = rig()
    model.geom_rbound[3] = .301
    data.geom_xpos[3] = [.05, 0., 0.]
    model.geom_aabb = np.zeros((5, 6))
    model.geom_aabb[3] = [0., 0., 0., .005, .005, .3]
    data.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (5, 1))
    return model, data, tubes


def test_long_thin_box_parallel_to_tube_proves_clearance_when_enclosing_sphere_overlaps():
    model, data, tubes = box_rig()
    assert check_tube_clearance(model, data, tubes) is None
    del model.geom_aabb
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['clearance_mm'] < -200.
    assert diagnostic['obb_fallback_reason'] and 'bounding_sphere' in diagnostic['bound_kind']


def test_convex_hull_interior_blocks_axis_crossing_even_when_every_corner_is_far_away():
    model, data, tubes = box_rig()
    # Rotate the long box across the tube axis. Projected corners lie 250 mm
    # away, but their convex hull contains the axis and cannot prove clearance.
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    data.geom_xmat[3] = rotation.ravel()
    diagnostic = check_tube_clearance(model, data, tubes)
    assert 'obb_projection' in diagnostic['bound_kind']
    assert diagnostic['obb_radial_distance_lower_bound_mm'] == 0.
    assert diagnostic['clearance_mm'] == pytest.approx(-14.7)
    assert diagnostic['actual_surface_contact_confirmed'] is False


def test_nearest_projected_edge_not_just_vertices_establishes_the_radial_bound():
    model, data, tubes = box_rig()
    model.geom_aabb[3] = [0., 0., 0., .005, .1, .005]
    data.geom_xpos[3] = [.019, 0., 0.]
    diagnostic = check_tube_clearance(model, data, tubes)
    # The nearest edge lies at x=14 mm; vertices have |y|=100 mm and would
    # incorrectly clear the tube if used in place of full polygon distance.
    assert diagnostic['obb_radial_distance_lower_bound_mm'] == pytest.approx(14.-1e-6)
    assert diagnostic['clearance_mm'] == pytest.approx(-.700001)


def test_nonzero_mujoco_local_aabb_center_is_rotated_and_translated():
    model, data, tubes = box_rig()
    data.geom_xpos[3] = [0., 0., 0.]
    model.geom_aabb[3] = [.08, 0., 0., .005, .005, .1]
    assert check_tube_clearance(model, data, tubes) is None
    model.geom_aabb[3, :3] = [0., 0., 0.]
    assert check_tube_clearance(model, data, tubes) is not None


def test_obb_refinement_preserves_rigid_transform_and_normalized_axis_invariance():
    model, data, tubes = box_rig()
    data.geom_xpos[3] = [.019, 0., 0.]
    first = check_tube_clearance(model, data, tubes)
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    translation = np.array([.5, -.25, .7])
    data.geom_xpos = data.geom_xpos @ rotation.T+translation
    data.geom_xmat = (rotation @ data.geom_xmat.reshape(5, 3, 3)).reshape(5, 9)
    for tube in tubes:
        tube['origin_m'] = rotation @ tube['origin_m']+translation
        tube['direction'] = 7.*(rotation @ tube['direction'])
    second = check_tube_clearance(model, data, tubes)
    assert second['clearance_mm'] == pytest.approx(first['clearance_mm'])
    assert 'obb_projection' in second['bound_kind']


@pytest.mark.parametrize('change', ['missing_rotation', 'wrong_bounds_shape', 'negative_halfsize',
    'nonfinite_halfsize', 'zero_box', 'scaled_rotation', 'reflected_rotation', 'nonfinite_rotation'])
def test_missing_or_invalid_native_box_data_keeps_the_original_sphere_rejection(change):
    model, data, tubes = box_rig()
    if change == 'missing_rotation':
        del data.geom_xmat
    elif change == 'wrong_bounds_shape':
        model.geom_aabb = np.zeros((1, 6))
    elif change == 'negative_halfsize':
        model.geom_aabb[3, 3] = -.005
    elif change == 'nonfinite_halfsize':
        model.geom_aabb[3, 3] = np.nan
    elif change == 'zero_box':
        model.geom_aabb[3, 3:] = 0.
    elif change == 'scaled_rotation':
        data.geom_xmat[3, 0] = 2.
    elif change == 'reflected_rotation':
        data.geom_xmat[3, 0] = -1.
    else:
        data.geom_xmat[3, 0] = np.nan
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic is not None and 'bounding_sphere' in diagnostic['bound_kind']
    assert diagnostic['clearance_mm'] == pytest.approx(diagnostic['sphere_clearance_mm'])
    assert diagnostic['obb_fallback_reason']


def test_degenerate_projected_line_and_point_use_segment_and_point_distance():
    model, data, tubes = box_rig()
    model.geom_aabb[3, 3:] = [0., 0., .3]  # Axis-parallel line projects to one point.
    assert check_tube_clearance(model, data, tubes) is None
    model.geom_aabb[3, 3:] = [0., .1, 0.]  # One projected segment at x=14 mm.
    data.geom_xpos[3] = [.014, 0., 0.]
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['clearance_mm'] == pytest.approx(-.700001)


def test_all_sphere_overlaps_are_checked_after_one_obb_proves_separation():
    model, data, tubes = box_rig()
    # The mount has the worst sphere overlap, but its tight box is clear. A
    # second arm geom still intersects and must not be skipped on that basis.
    data.geom_xpos[2] = [.02, 0., 0.]
    model.geom_aabb[2] = [0., 0., 0., .01, .01, .01]
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['geom_name'] == 'arm_mesh' and diagnostic['clearance_mm'] < 0


def test_refinement_does_not_change_existing_model_data_or_margin():
    model, data, tubes = box_rig()
    data.geom_xpos[3] = [.019, 0., 0.]
    before_model, before_data = copy.deepcopy(model), copy.deepcopy(data)
    diagnostic = check_tube_clearance(model, data, tubes)
    assert diagnostic['margin_mm'] == 2.
    for name in ('geom_rbound', 'geom_aabb', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(model, name), getattr(before_model, name))
    np.testing.assert_array_equal(data.geom_xpos, before_data.geom_xpos)
    np.testing.assert_array_equal(data.geom_xmat, before_data.geom_xmat)
