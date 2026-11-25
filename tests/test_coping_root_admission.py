"""Adversarial current-evidence and finite-root admission, without hardware."""

import copy

import numpy as np
import pytest
from test_coping_service import coping as _coping_fixture
from test_coping_service import model
from test_coping_service import rig as _rig_fixture

from autoweld.coping_service import CopingObservationRejected
from autoweld.weld_seam_service import WeldSettings

coping = _coping_fixture
rig = _rig_fixture


def _archived_fit_with_three_fresh(*, fresh_points=100):
    """Thirty-two good prior profiles must not outvote individual fresh views."""
    profiles, labels, groups = [], [], []
    for index in range(35):
        fresh = index >= 32
        count = fresh_points if fresh else 100
        profile_labels = [0]*(count//2)+[1]*(count-count//2)
        profiles.append(dict(xyz_mm=[[float(i), 0., 150.] for i in range(count)], profile_id=index,
                             provenance={'fresh_current_run': fresh}))
        labels.extend(profile_labels)
        groups.extend([index]*count)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    residuals = np.full((len(labels), 2), .8)
    residuals[np.arange(len(labels)), labels] = .01
    fit = dict(status='qualified', model=model(),
               quality={'inlier_fraction': 1., 'train_robust_rms_mm': .01, 'heldout_robust_rms_mm': .01},
               classification=dict(surface_labels=labels.tolist(), profile_indices=groups.tolist(),
                                   inlier_mask=[True]*len(labels), residuals_mm=residuals.tolist()))
    return profiles, fit


def test_one_bad_current_view_cannot_be_outvoted_by_two_good_current_views_and_32_prior_profiles(coping):
    owner, _, _ = coping
    profiles, fit = _archived_fit_with_three_fresh()
    classification = fit['classification']
    groups = np.asarray(classification['profile_indices'])
    bad = groups == 34
    inliers = np.asarray(classification['inlier_mask'])
    inliers[bad] = False
    residuals = np.asarray(classification['residuals_mm'])
    residuals[bad] = 4.
    classification.update(inlier_mask=inliers.tolist(), residuals_mm=residuals.tolist())
    fit['quality']['inlier_fraction'] = float(np.mean(inliers))
    assert fit['quality']['inlier_fraction'] > .97  # An aggregate-only gate would pass.

    owner.weld.settings = WeldSettings(joint_kind='coped_tube')
    owner.weld._coping_profiles = profiles
    owner.weld._coping_scouts = []  # No unmeasured replacement scout may hide this rejection.
    before = copy.deepcopy(owner.events)
    with pytest.raises(ValueError, match='not qualified'):
        owner.weld._joint_future('coping_fit', fit)
    validation = owner.weld._joint_model['current_validation']
    assert not validation['qualified']
    assert validation['prior_cannot_outvote_current']
    assert min(validation['per_surface_inlier_count']) >= 50  # Two good current views are insufficient here.
    failed = next(profile for profile in validation['profiles'] if profile['profile_index'] == 34)
    assert failed['inlier_fraction'] == 0. and failed['all_point_rms_mm'] == 4.
    assert owner.weld._coping_plan is None
    assert owner.events == before


def test_one_current_view_must_independently_observe_both_surfaces(coping):
    owner, _, _ = coping
    profiles, fit = _archived_fit_with_three_fresh()
    classification = fit['classification']
    groups = np.asarray(classification['profile_indices'])
    labels = np.asarray(classification['surface_labels'])
    labels[groups == 34] = 0  # A clear view of only one tube is not two-surface evidence.
    residuals = np.full((len(labels), 2), .8)
    residuals[np.arange(len(labels)), labels] = .01
    classification.update(surface_labels=labels.tolist(), residuals_mm=residuals.tolist())
    owner.weld._coping_profiles = profiles

    validation = owner.weld._coping_validate_fresh(fit)
    assert not validation['qualified']
    assert min(validation['per_surface_inlier_count']) >= 50
    deficient = next(profile for profile in validation['profiles'] if profile['profile_index'] == 34)
    assert deficient['inlier_fraction'] == 1.
    assert deficient['all_point_rms_mm'] == pytest.approx(.01)
    assert deficient['per_surface_inlier_count'] == [100, 0]


def test_prior_points_do_not_satisfy_minimum_current_surface_support(coping):
    owner, _, _ = coping
    profiles, fit = _archived_fit_with_three_fresh(fresh_points=20)
    owner.weld._coping_profiles = profiles
    validation = owner.weld._coping_validate_fresh(fit)
    assert not validation['qualified']
    assert validation['per_surface_inlier_count'] == [30, 30]
    assert all(profile['inlier_fraction'] == 1. for profile in validation['profiles'])
    assert all(profile['per_surface_inlier_count'] == [10, 10] for profile in validation['profiles'])
    assert '50 current inliers per cylinder' in validation['reason']


@pytest.mark.parametrize('heldout_rms_mm', [.2, .35])
def test_correct_root_near_crown_remains_ambiguous_when_fit_cannot_resolve_its_sign(coping, heldout_rms_mm):
    owner, _, _ = coping
    m = model()
    theta = np.deg2rad(88.)
    radial = 12.7*np.cos(theta)
    correct = np.asarray(m['center_mm'])+[radial, radial, 12.7*np.sin(theta)]
    owner.weld.settings = WeldSettings(joint_kind='coped_tube')
    owner.weld._coping_seed_frame = np.eye(4)
    owner.weld._coping_seed_mm = np.array([0., 0., 150.])
    owner.weld._coping_curve = {'root_sign': 1, 'points_mm': [[0., 0., 150.]]}
    owner.weld._joint_model = {'model': m, 'quality': {'heldout_robust_rms_mm': heldout_rms_mm}}
    record = {'direction': 0, 'seam_point_base_m': (correct/1000.).tolist()}
    # This is geometrically the SELECTED root, exactly on both cylinders. Its
    # finite coordinate is nevertheless below the fit-derived ambiguity band.
    assert radial > 0 and radial < 3*heldout_rms_mm
    for axis in m['axes']:
        delta = correct-axis['origin_mm']
        direction = np.asarray(axis['direction'])
        assert np.linalg.norm(delta-direction*np.dot(delta, direction)) == pytest.approx(12.7)
    before = copy.deepcopy(owner.events)
    with pytest.raises(CopingObservationRejected, match='indistinguishable'):
        owner.weld._coping_orient_record(record)
    assert owner.weld.records == [] and owner.weld._coping_observed == {1: 0., -1: 0.}
    assert owner.events == before


def test_same_selected_near_crown_root_can_be_associated_when_geometry_resolves_the_sign(coping):
    owner, _, _ = coping
    m = model()
    theta = np.deg2rad(88.)
    radial = 12.7*np.cos(theta)
    point = np.asarray(m['center_mm'])+[radial, radial, 12.7*np.sin(theta)]
    owner.weld.settings = WeldSettings(joint_kind='coped_tube')
    owner.weld._coping_seed_frame = np.eye(4)
    owner.weld._coping_seed_mm = np.array([0., 0., 150.])
    owner.weld._coping_curve = {'root_sign': 1, 'points_mm': [[0., 0., 150.]]}
    owner.weld._joint_model = {'model': m, 'quality': {'heldout_robust_rms_mm': .05}}
    record = {'direction': 0, 'seam_point_base_m': (point/1000.).tolist()}
    before = copy.deepcopy(record)

    accepted = owner.weld._coping_orient_record(record)
    geometry = accepted['coping_geometry']
    assert geometry['root_sign'] == 1
    assert geometry['finite_root_signed_coordinate_mm'] == pytest.approx(radial)
    assert geometry['finite_root_signed_coordinate_mm'] > geometry['finite_root_ambiguity_band_mm']
    assert geometry['association']['associated']
    np.testing.assert_allclose(geometry['association']['measured_point_mm'], point)
    torch = np.asarray(accepted['torch_pose_base_m'])
    np.testing.assert_allclose(torch[:3, 3]+.15*torch[:3, 2], point/1000.)
    assert record == before  # Qualification does not rewrite original measured geometry.
