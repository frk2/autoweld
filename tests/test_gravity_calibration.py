"""Known physical payload tests using the one shared model and scratch data."""

import copy
import json

import mujoco
import numpy as np
import pytest

from autoweld.gravity_calibration import (
    GravityFitError,
    fit_joint_factors,
    fit_payload,
    gravity_regressor,
    make_sample,
)
from autoweld.models import compiled_model


@pytest.fixture(scope='module')
def shared_model():
    return compiled_model()


def poses():
    # Synthetic geometry only. Root reviews actual physical calibration poses
    # for free-space clearance before taking real measurements.
    rng = np.random.default_rng(103)
    return rng.uniform([-.8, -.7, -.8, -.9, -.8, -1.5], [.8, 1., 1., .9, .8, 1.5], (24, 6))


def fixture_samples(model, *, noise=.0):
    data = mujoco.MjData(model)
    parameters = np.array([.43, .43 * .021, .43 * -.014, .43 * .067])
    rng = np.random.default_rng(77)
    samples = []
    for index, q in enumerate(poses()):
        baseline, matrix = gravity_regressor(model, data, q)
        effort = baseline + matrix @ parameters + rng.normal(0., noise, 6)
        samples.append(make_sample(model, data, q, effort,
                                   split='validation' if index % 4 == 0 else 'train',
                                   free_space=True, settled=True, velocity_rad_s=np.zeros(6),
                                   effort_std_nm=np.ones(6) * noise))
    return parameters, samples


def test_regressor_matches_independent_potential_gradient_and_never_mutates_model(shared_model):
    model = shared_model
    data, energy_data = mujoco.MjData(model), mujoco.MjData(model)
    before = [a.copy() for a in (model.body_mass, model.body_ipos, model.body_inertia)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
    mass, com = .43, np.array([.021, -.014, .067])
    gravity = model.opt.gravity
    def potential(q):
        energy_data.qpos[:] = q
        energy_data.qvel[:] = 0.
        mujoco.mj_forward(model, energy_data)
        # Sum each native body's potential energy, then replace flange payload.
        total = -float(np.sum(model.body_mass[:, None] * energy_data.xipos * gravity))
        total += model.body_mass[body_id] * (energy_data.xipos[body_id] @ gravity)
        payload_com = energy_data.xpos[body_id] + energy_data.xmat[body_id].reshape(3, 3) @ com
        return total - mass * (payload_com @ gravity)
    for q in poses()[:4]:
        baseline, matrix = gravity_regressor(model, data, q)
        analytic = baseline + matrix @ np.r_[mass, mass * com]
        finite = np.array([(potential(q + np.eye(6)[j] * 1e-6)
                            - potential(q - np.eye(6)[j] * 1e-6)) / 2e-6 for j in range(6)])
        np.testing.assert_allclose(analytic, finite, atol=2e-8)
    for expected, actual in zip(before, (model.body_mass, model.body_ipos, model.body_inertia)):
        np.testing.assert_array_equal(actual, expected)
    assert compiled_model() is model


def test_known_physical_payload_fit_and_heldout_prediction(shared_model):
    parameters, samples = fixture_samples(shared_model)
    result = fit_payload(samples)
    np.testing.assert_allclose(result['parameter_vector'], parameters, atol=1e-9)
    np.testing.assert_allclose(result['com_m'], [.021, -.014, .067], atol=1e-9)
    assert result['validation']['rms_nm'] < 1e-10
    assert result['validation']['improvement_fraction'] > .99999
    assert result['rank'] == 4
    assert not result['applied']
    json.dumps(result, allow_nan=False)


def test_robust_fit_limits_isolated_training_feedback_outlier(shared_model):
    parameters, samples = fixture_samples(shared_model, noise=.003)
    sample = next(s for s in samples if s['split'] == 'train')
    sample['effort_nm'][1] += 2.
    result = fit_payload(samples)
    assert abs(result['mass_kg'] - parameters[0]) < .012
    np.testing.assert_allclose(result['com_m'], [.021, -.014, .067], atol=.012)
    assert result['validation']['rms_nm'] < .015
    assert result['largest_training_residual_nm'] > 1.9


@pytest.mark.parametrize('field,value', [('free_space', False), ('settled', False),
                                         ('contact', True), ('at_rest', True)])
def test_contact_rest_and_unsettled_samples_are_never_fitted(shared_model, field, value):
    _, samples = fixture_samples(shared_model)
    samples[0][field] = value
    with pytest.raises(GravityFitError, match='free-space'):
        fit_payload(samples)


def test_quiet_hold_and_independent_validation_gates(shared_model):
    _, samples = fixture_samples(shared_model)
    bad = copy.deepcopy(samples)
    bad[0]['velocity_rad_s'][0] = .2
    with pytest.raises(GravityFitError, match='quiet'):
        fit_payload(bad)
    bad = copy.deepcopy(samples)
    bad[0]['effort_std_nm'][0] = .8
    with pytest.raises(GravityFitError, match='noisy'):
        fit_payload(bad)
    bad = copy.deepcopy(samples)
    for sample in bad:
        sample['split'] = 'train'
    with pytest.raises(GravityFitError, match='independent validation'):
        fit_payload(bad)
    bad = copy.deepcopy(samples)
    bad[0]['q_rad'] = bad[1]['q_rad'][:]
    with pytest.raises(GravityFitError, match='overlap'):
        fit_payload(bad)


def test_wrong_sign_unphysical_com_or_heldout_contact_residual_rejects(shared_model):
    _, samples = fixture_samples(shared_model)
    wrong_sign = copy.deepcopy(samples)
    for sample in wrong_sign:
        sample['effort_nm'] = (2 * np.asarray(sample['baseline_nm']) - sample['effort_nm']).tolist()
    with pytest.raises(GravityFitError, match='mass'):
        fit_payload(wrong_sign)
    absurd = copy.deepcopy(samples)
    for sample in absurd:
        sample['effort_nm'] = (np.asarray(sample['baseline_nm']) +
                               np.asarray(sample['regressor']) @ [.4, .4 * .8, 0., 0.]).tolist()
    with pytest.raises(GravityFitError, match='COM'):
        fit_payload(absurd)
    bad_validation = copy.deepcopy(samples)
    for sample in bad_validation:
        if sample['split'] == 'validation':
            sample['effort_nm'][2] += 1.5
    with pytest.raises(GravityFitError, match='Held-out'):
        fit_payload(bad_validation)


def test_poorly_conditioned_and_duplicate_holds_reject(shared_model):
    _, samples = fixture_samples(shared_model)
    bad = copy.deepcopy(samples)
    for sample in bad:
        matrix = np.asarray(sample['regressor'])
        matrix[:, 3] = matrix[:, 2]
        sample['regressor'] = matrix.tolist()
    with pytest.raises(GravityFitError, match='conditioned'):
        fit_payload(bad)
    bad = copy.deepcopy(samples)
    for sample in bad:
        if sample['split'] == 'train':
            sample['q_rad'] = bad[1]['q_rad'][:]
    with pytest.raises(GravityFitError, match='diverse'):
        fit_payload(bad)


def joint_factor_samples(model, *, noise=0.):
    payload = np.r_[.45359237, .45359237 * np.array([-.02444, .01151, -.04969])]
    factors = np.array([1., 1.12, 1.47, .93, 1.08, 1.])
    data = mujoco.MjData(model)
    rng = np.random.default_rng(391)
    samples = []
    for index, q in enumerate(poses()):
        baseline, matrix = gravity_regressor(model, data, q)
        effort = (baseline + matrix @ payload) * factors + rng.normal(0., noise, 6)
        samples.append(make_sample(model, data, q, effort,
                                   split='validation' if index % 4 == 0 else 'train',
                                   free_space=True, settled=True, effort_std_nm=np.ones(6) * noise))
    return payload, factors, samples


def test_known_payload_joint_factors_recover_bare_arm_mismatch_without_changing_mass(shared_model):
    payload, expected, samples = joint_factor_samples(shared_model)
    before = payload.copy()
    result = fit_joint_factors(samples, payload)
    np.testing.assert_allclose(result['factors'], expected, atol=1e-8)
    np.testing.assert_array_equal(result['parameter_vector'], before)
    np.testing.assert_array_equal(payload, before)
    assert result['mass_kg'] == .45359237
    np.testing.assert_allclose(result['com_m'], [-.02444, .01151, -.04969])
    assert result['unobservable_joints'] == [1, 6]
    assert result['validation']['rms_nm'] < 1e-9
    assert result['validation']['improvement_fraction'] > .9999
    assert result['validation']['correction_rms_nm'][2] > .5
    assert not result['physical_payload_changed'] and not result['applied']
    json.dumps(result, allow_nan=False)


def test_joint_factors_are_robust_and_weak_axis_factors_stay_one(shared_model):
    payload, factors, samples = joint_factor_samples(shared_model, noise=.003)
    train = next(s for s in samples if s['split'] == 'train')
    train['effort_nm'][2] += 2.
    # Inject a large apparent weak-axis gain. It remains unfitted and its entire
    # residual is still included in held-out quality metrics.
    for sample in samples:
        modeled = np.asarray(sample['baseline_nm']) + np.asarray(sample['regressor']) @ payload
        sample['effort_nm'][5] = modeled[5] * 3.
    result = fit_joint_factors(samples, payload)
    assert result['factors'][0] == result['factors'][5] == 1.
    assert abs(result['factors'][2] - factors[2]) < .01
    assert result['validation']['joint_rms_nm'][5] > .05
    assert result['validation']['joint_relative_rms_to_modeled_peak'][5] is None
    assert result['validation']['correction_rms_nm'][5] == 0.


def test_joint_factor_training_is_independent_of_heldout_effort(shared_model):
    payload, _, samples = joint_factor_samples(shared_model)
    expected = fit_joint_factors(samples, payload)
    changed = copy.deepcopy(samples)
    for sample in changed:
        if sample['split'] == 'validation':
            sample['effort_nm'][1] += .1
    result = fit_joint_factors(changed, payload)
    np.testing.assert_array_equal(result['factors'], expected['factors'])
    assert result['validation']['joint_rms_nm'][1] == pytest.approx(.1)


def test_joint_factor_heldout_gates_include_unobservable_axes_and_no_offset_fits(shared_model):
    payload, _, samples = joint_factor_samples(shared_model)
    bad = copy.deepcopy(samples)
    for sample in bad:
        if sample['split'] == 'validation':
            sample['effort_nm'][0] = .8
    with pytest.raises(GravityFitError, match='per joint'):
        fit_joint_factors(bad, payload)
    bad = copy.deepcopy(samples)
    for sample in bad:
        if sample['split'] == 'validation':
            sample['effort_nm'] = (np.asarray(sample['effort_nm']) + .5).tolist()
    with pytest.raises(GravityFitError, match='overall'):
        fit_joint_factors(bad, payload)


def test_joint_factors_do_not_hide_wrong_payload_with_out_of_bounds_gains(shared_model):
    payload, _, samples = joint_factor_samples(shared_model)
    for sample in samples:
        modeled = np.asarray(sample['baseline_nm']) + np.asarray(sample['regressor']) @ payload
        sample['effort_nm'][2] = 2.5 * modeled[2]
    with pytest.raises(GravityFitError, match='Held-out'):
        fit_joint_factors(samples, payload)
    with pytest.raises(ValueError, match='bounds'):
        fit_joint_factors(samples, payload, factor_bounds=(.2, 3.))


def test_joint_factor_fit_retains_contact_quietness_and_heldout_provenance_gates(shared_model):
    payload, _, samples = joint_factor_samples(shared_model)
    samples[0]['at_rest'] = True
    with pytest.raises(GravityFitError, match='free-space'):
        fit_joint_factors(samples, payload)
