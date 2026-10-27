"""Read-only payload gravity identification for the six-joint YAM arm.

The caller supplies the process's existing MuJoCo model and a separate MjData.
No model is compiled, modified, or installed in the robot by this module. Torques
are motor joint torques in Nm, joint coordinates are radians, and payload first
moments/COM use the selected flange body's local axes and origin.
"""

import math

import mujoco
import numpy as np
from scipy.optimize import least_squares


class GravityFitError(ValueError):
    """A candidate payload is unsupported by its measurements or validation."""


def _six(value, name):
    array = np.asarray(value, float)
    if array.shape not in ((6,), (7,)) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain six finite arm values (optional fixed gripper seventh)')
    return array[:6].copy()


def gravity_regressor(model, data, q, body='gripper'):
    """Return bare-arm gravity and a 6x4 payload inertial regressor.

    ``baseline + A @ [mass, mass*cx, mass*cy, mass*cz]`` predicts the static
    holding torque when replacing the selected body's current inertia. The
    existing body's own mass/COM gravity is removed from baseline; its descendants
    remain part of the model. Rotor inertia has no static-gravity contribution.
    The supplied MjData is scratch space and must not be live SDK/planner data.
    """
    q = _six(q, 'Joint position')
    if model.nq != 6 or model.nv != 6:
        raise ValueError('Payload calibration currently requires the six-joint fixed-mount model')
    if data.model is not model:
        raise ValueError('Scratch MjData must belong to the supplied shared model')
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if body_id <= 0:
        raise ValueError(f'Payload body {body!r} does not exist or is the world body')
    gravity = np.asarray(model.opt.gravity, float)
    if not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-6:
        raise ValueError('Gravity calibration requires a finite nonzero model gravity vector')
    data.qpos[:] = q
    data.qvel[:] = 0.
    data.qacc[:] = 0.
    # Forward dynamics computes qfrc_bias (Coriolis + gravity). At zero velocity
    # this is exactly the gravity holding torque, independent of contact forces.
    mujoco.mj_forward(model, data)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jp, jr, body_id)
    rotation = data.xmat[body_id].reshape(3, 3)
    # Gravity wrench is [m*g, (R*h) x g]. Negate its joint projection to obtain
    # compensation torque, using h=m*COM without nonlinear mass/COM fitting.
    matrix = np.column_stack((-jp.T @ gravity,
                              *[-jr.T @ np.cross(rotation[:, axis], gravity) for axis in range(3)]))
    old_parameters = np.r_[model.body_mass[body_id],
                           model.body_mass[body_id] * model.body_ipos[body_id]]
    baseline = data.qfrc_bias[:6].copy() - matrix @ old_parameters
    if not np.isfinite(baseline).all() or not np.isfinite(matrix).all():
        raise ValueError('Model returned non-finite static gravity')
    return baseline, matrix


def make_sample(model, data, q, effort_nm, *, split, free_space, settled, body='gripper',
                velocity_rad_s=None, effort_std_nm=None, contact=False, at_rest=False,
                metadata=None):
    """Build a detached JSON-ready sample without touching hardware or the model.

    ``free_space`` and ``settled`` are explicit caller-reviewed evidence. Rest
    contact, a person supporting the arm, fixtures, and moving poses are excluded.
    Measured effort must use SDK motor feedback with the same joint torque sign
    and units as MuJoCo; commanded feedforward is not a measurement.
    """
    q, effort = _six(q, 'Joint position'), _six(effort_nm, 'Measured effort')
    baseline, matrix = gravity_regressor(model, data, q, body)
    sample = dict(q_rad=q.tolist(), effort_nm=effort.tolist(), baseline_nm=baseline.tolist(),
                  regressor=matrix.tolist(), split=split, free_space=free_space,
                  settled=settled, contact=contact, at_rest=at_rest, body=body,
                  metadata=dict(metadata or {}))
    if velocity_rad_s is not None:
        sample['velocity_rad_s'] = _six(velocity_rad_s, 'Joint velocity').tolist()
    if effort_std_nm is not None:
        sample['effort_std_nm'] = _six(effort_std_nm, 'Effort standard deviation').tolist()
    _validated_sample(sample)
    return sample


def _validated_sample(sample, *, max_velocity_rad_s=.025, max_effort_std_nm=.15):
    if (sample.get('free_space') is not True or sample.get('settled') is not True
            or sample.get('contact', False) is not False or sample.get('at_rest', False) is not False):
        raise GravityFitError('Only settled, verified free-space holds are usable; exclude contact and rest poses')
    if sample.get('split') not in ('train', 'validation'):
        raise GravityFitError('Declare train or validation split before fitting')
    q = _six(sample['q_rad'], 'Joint position')
    effort, baseline = _six(sample['effort_nm'], 'Measured effort'), _six(sample['baseline_nm'], 'Baseline')
    matrix = np.asarray(sample['regressor'], float)
    if matrix.shape != (6, 4) or not np.isfinite(matrix).all():
        raise GravityFitError('Each pose needs a finite 6x4 payload regressor')
    if 'velocity_rad_s' in sample:
        if np.max(np.abs(_six(sample['velocity_rad_s'], 'Joint velocity'))) > max_velocity_rad_s:
            raise GravityFitError('Pose is not quiet enough for static gravity identification')
    if 'effort_std_nm' in sample:
        std = _six(sample['effort_std_nm'], 'Effort standard deviation')
        if np.any(std < 0) or np.max(std) > max_effort_std_nm:
            raise GravityFitError('Measured holding torque is too noisy for gravity identification')
    return q, effort, baseline, matrix


def _metrics(actual, baseline, predicted):
    errors, before = predicted - actual, baseline - actual
    return dict(rms_nm=float(np.sqrt(np.mean(errors**2))),
                baseline_rms_nm=float(np.sqrt(np.mean(before**2))),
                joint_rms_nm=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                joint_baseline_rms_nm=np.sqrt(np.mean(before**2, axis=0)).tolist(),
                max_abs_nm=float(np.max(np.abs(errors))),
                median_abs_nm=float(np.median(np.abs(errors))))


def fit_payload(samples, *, min_train_poses=8, min_validation_poses=3, max_condition=100.,
                min_mass_kg=.005, max_mass_kg=3., max_com_m=.25,
                max_validation_rms_nm=.3, min_improvement_fraction=.05,
                robust_scale_nm=.05, max_velocity_rad_s=.025, max_effort_std_nm=.15):
    """Fit mass and three first moments, then gate independent held-out poses.

    Pose splits must be chosen before fitting. No observations are dropped or
    relabeled on failure. Huber loss limits isolated motor-feedback outliers;
    held-out residual checks remain unweighted and include every validation pose.
    A returned result is a reviewable candidate, never an instruction to apply it.
    """
    samples = list(samples)
    positive_options = (min_train_poses, min_validation_poses, max_condition, min_mass_kg,
                        max_mass_kg, max_com_m, max_validation_rms_nm, robust_scale_nm,
                        max_velocity_rad_s, max_effort_std_nm)
    if (not all(np.isfinite(x) and x > 0 for x in positive_options)
            or min_mass_kg >= max_mass_kg or not 0 <= min_improvement_fraction < 1):
        raise ValueError('Gravity fit thresholds must be finite, positive and physically consistent')
    bodies = {sample.get('body', 'gripper') for sample in samples}
    if len(bodies) != 1:
        raise GravityFitError('All gravity samples must use one payload body frame')
    train, validation = [], []
    for sample in samples:
        row = _validated_sample(sample, max_velocity_rad_s=max_velocity_rad_s,
                                max_effort_std_nm=max_effort_std_nm)
        (train if sample['split'] == 'train' else validation).append(row)
    if len(train) < min_train_poses or len(validation) < min_validation_poses:
        raise GravityFitError(f'Need at least {min_train_poses} training and '
                              f'{min_validation_poses} independent validation holds')
    train_q, validation_q = np.array([s[0] for s in train]), np.array([s[0] for s in validation])
    # Repeated telemetry from a single hold is not independent pose excitation.
    if len(np.unique(np.round(train_q, 4), axis=0)) < min_train_poses:
        raise GravityFitError('Training holds must contain independent diverse poses')
    if len(np.unique(np.round(validation_q, 4), axis=0)) < min_validation_poses:
        raise GravityFitError('Validation holds must contain independent poses')
    for q in validation_q:
        if np.min(np.linalg.norm(train_q - q, axis=1)) < .01:
            raise GravityFitError('Validation poses overlap training poses; reserve held-out configurations')
    actual = np.array([s[1] for s in train])
    baseline = np.array([s[2] for s in train])
    matrices = np.array([s[3] for s in train])
    matrix = matrices.reshape(-1, 4)
    target = (actual - baseline).reshape(-1)
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms < 1e-8):
        raise GravityFitError('Payload parameters are unobservable at these training poses')
    scaled = matrix / norms
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    if np.linalg.matrix_rank(scaled) != 4 or condition > max_condition:
        raise GravityFitError(f'Payload fit is poorly conditioned ({condition:.2f}); vary wrist and arm orientation')
    initial = np.linalg.lstsq(scaled, target, rcond=None)[0]
    fit = least_squares(lambda scaled_parameters: scaled @ scaled_parameters - target,
                        initial, loss='huber', f_scale=robust_scale_nm,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    if not fit.success or not np.isfinite(fit.x).all():
        raise GravityFitError('Robust payload fit did not converge')
    parameters = fit.x / norms
    mass = float(parameters[0])
    if not min_mass_kg <= mass <= max_mass_kg:
        raise GravityFitError(f'Payload mass {mass:.4f}kg is outside [{min_mass_kg}, {max_mass_kg}]kg')
    com = parameters[1:] / mass
    if np.linalg.norm(com) > max_com_m:
        raise GravityFitError(f'Payload COM is {np.linalg.norm(com):.3f}m from flange; check signs and torque units')
    train_metrics = _metrics(actual, baseline, baseline + matrices @ parameters)
    held_actual = np.array([s[1] for s in validation])
    held_baseline = np.array([s[2] for s in validation])
    held_matrices = np.array([s[3] for s in validation])
    held_metrics = _metrics(held_actual, held_baseline, held_baseline + held_matrices @ parameters)
    if (held_metrics['rms_nm'] > max_validation_rms_nm
            or max(held_metrics['joint_rms_nm']) > 2 * max_validation_rms_nm):
        raise GravityFitError(f'Held-out gravity RMS {held_metrics["rms_nm"]:.4f}Nm is too high; '
                              'check contact, friction, effort signs and the bare-arm model')
    before = held_metrics['baseline_rms_nm']
    improvement = 1. - held_metrics['rms_nm'] / before if before > 1e-12 else -math.inf
    if improvement < min_improvement_fraction:
        raise GravityFitError('Payload fit does not improve held-out torque prediction enough')
    return dict(schema_version=1, body=next(iter(bodies)),
                parameter_order=['mass_kg', 'mx_kgm', 'my_kgm', 'mz_kgm'],
                parameter_vector=parameters.tolist(), mass_kg=mass, com_m=com.tolist(),
                first_moment_kgm=parameters[1:].tolist(),
                mass_semantics='replace selected body mass and COM; baseline excludes its current gravity',
                coordinate_frame='payload_body', torque_units='Nm',
                training_poses=len(train), validation_poses=len(validation),
                rank=4, condition_scaled=condition, condition_raw=float(np.linalg.cond(matrix)),
                training=train_metrics, validation={**held_metrics, 'improvement_fraction': improvement},
                robust_scale_nm=robust_scale_nm,
                largest_training_residual_nm=train_metrics['max_abs_nm'],
                applied=False)


def fit_joint_factors(samples, payload_parameters, *, min_train_poses=8, min_validation_poses=3,
                      factor_bounds=(.5, 2.), min_gravity_peak_nm=.15,
                      max_validation_rms_nm=.35, max_joint_validation_rms_nm=.7,
                      robust_scale_nm=.05, min_improvement_fraction=.05,
                      max_velocity_rad_s=.025, max_effort_std_nm=.15):
    """Validate bounded per-joint gravity factors for an unchanged known payload.

    Each observable joint fits ``measured = factor * modeled_gravity`` with
    training-only Huber loss. Modeled gravity includes both the bare arm and the
    supplied physical payload. There are no offsets, friction terms, new payload
    estimates, or fitted gains for weak gravity axes. Every held-out joint remains
    in the residual gates, including unobservable axes whose factors stay 1.0.
    This is an empirical model correction, not a mass/COM measurement.
    """
    samples = list(samples)
    bounds = np.asarray(factor_bounds, float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not .5 <= bounds[0] < 1. < bounds[1] <= 2.):
        raise ValueError('Gravity factor bounds must contain 1 and stay within [0.5, 2]')
    limits = (min_train_poses, min_validation_poses, min_gravity_peak_nm,
              max_validation_rms_nm, max_joint_validation_rms_nm, robust_scale_nm,
              max_velocity_rad_s, max_effort_std_nm)
    if (not all(np.isfinite(value) and value > 0 for value in limits)
            or not np.isfinite(min_improvement_fraction) or not 0 <= min_improvement_fraction < 1):
        raise ValueError('Gravity factor thresholds must be finite and positive')
    if isinstance(payload_parameters, dict):
        payload_parameters = payload_parameters['parameter_vector']
    payload = np.asarray(payload_parameters, float).copy()
    if (payload.shape != (4,) or not np.isfinite(payload).all()
            or not .005 <= payload[0] <= 3.):
        raise GravityFitError('Known payload must have four finite inertial parameters and mass 0.005–3kg')
    com = payload[1:] / payload[0]
    if np.linalg.norm(com) > .25:
        raise GravityFitError('Known payload COM must lie within 0.25m of the payload body origin')
    bodies = {sample.get('body', 'gripper') for sample in samples}
    if len(bodies) != 1:
        raise GravityFitError('All gravity samples must use one payload body frame')
    train, validation = [], []
    for sample in samples:
        row = _validated_sample(sample, max_velocity_rad_s=max_velocity_rad_s,
                                max_effort_std_nm=max_effort_std_nm)
        (train if sample['split'] == 'train' else validation).append(row)
    if len(train) < min_train_poses or len(validation) < min_validation_poses:
        raise GravityFitError(f'Need at least {min_train_poses} training and '
                              f'{min_validation_poses} independent validation holds')
    train_q = np.array([row[0] for row in train])
    held_q = np.array([row[0] for row in validation])
    if len(np.unique(np.round(train_q, 4), axis=0)) < min_train_poses:
        raise GravityFitError('Training holds must contain independent diverse poses')
    if len(np.unique(np.round(held_q, 4), axis=0)) < min_validation_poses:
        raise GravityFitError('Validation holds must contain independent poses')
    if any(np.min(np.linalg.norm(train_q - q, axis=1)) < .01 for q in held_q):
        raise GravityFitError('Validation poses overlap training poses; reserve held-out configurations')

    def values(rows):
        actual = np.array([row[1] for row in rows])
        modeled = np.array([row[2] + row[3] @ payload for row in rows])
        return actual, modeled

    actual, modeled = values(train)
    held_actual, held_modeled = values(validation)
    peak = np.max(np.abs(modeled), axis=0)
    observable = peak >= min_gravity_peak_nm
    factors = np.ones(6)
    for joint in np.flatnonzero(observable):
        basis, target = modeled[:, joint], actual[:, joint]
        initial = np.clip(float(basis @ target / (basis @ basis)), bounds[0], bounds[1])
        fit = least_squares(lambda factor: basis * factor[0] - target, [initial],
                            bounds=(bounds[:1], bounds[1:]), loss='huber', f_scale=robust_scale_nm,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
        if not fit.success or not np.isfinite(fit.x).all():
            raise GravityFitError(f'Gravity factor fit did not converge for J{joint + 1}')
        factors[joint] = fit.x[0]
    training = _metrics(actual, modeled, modeled * factors)
    held = _metrics(held_actual, held_modeled, held_modeled * factors)
    if held['rms_nm'] > max_validation_rms_nm or max(held['joint_rms_nm']) > max_joint_validation_rms_nm:
        raise GravityFitError(f'Held-out joint-factor RMS {held["rms_nm"]:.4f}Nm '
                              f'(worst joint {max(held["joint_rms_nm"]):.4f}Nm) exceeds '
                              f'{max_validation_rms_nm}Nm overall/{max_joint_validation_rms_nm}Nm per joint; '
                              'no factors or physical payload were applied')
    before = held['baseline_rms_nm']
    improvement = 1. - held['rms_nm'] / before if before > 1e-12 else 0.
    if improvement < min_improvement_fraction:
        raise GravityFitError('Joint gravity factors do not improve held-out torque prediction enough')
    held_peak = np.max(np.abs(held_modeled), axis=0)
    before_joint, after_joint = np.array(held['joint_baseline_rms_nm']), np.array(held['joint_rms_nm'])
    joint_improvement = np.zeros(6)
    nonzero = before_joint > 1e-12
    joint_improvement[nonzero] = 1. - after_joint[nonzero] / before_joint[nonzero]
    correction = held_modeled * (factors - 1.)
    held.update(improvement_fraction=float(improvement),
                joint_improvement_fraction=joint_improvement.tolist(),
                joint_relative_rms_to_modeled_peak=[float(after_joint[j] / held_peak[j])
                    if held_peak[j] >= min_gravity_peak_nm else None for j in range(6)],
                correction_rms_nm=np.sqrt(np.mean(correction**2, axis=0)).tolist(),
                correction_peak_nm=np.max(np.abs(correction), axis=0).tolist())
    return dict(schema_version=1, model='known_payload_joint_gravity_factors', body=next(iter(bodies)),
                factors=factors.tolist(), factor_bounds=bounds.tolist(),
                fitted_joints=(np.flatnonzero(observable) + 1).tolist(),
                unobservable_joints=(np.flatnonzero(~observable) + 1).tolist(),
                training_modeled_gravity_peak_nm=peak.tolist(), min_gravity_peak_nm=min_gravity_peak_nm,
                parameter_vector=payload.tolist(), mass_kg=float(payload[0]), com_m=com.tolist(),
                first_moment_kgm=payload[1:].tolist(), physical_payload_changed=False,
                mass_semantics='fixed supplied physical payload; joint factors are empirical gravity corrections',
                coordinate_frame='payload_body', torque_units='Nm',
                training_poses=len(train), validation_poses=len(validation),
                training=training, validation=held, robust_scale_nm=robust_scale_nm,
                validation_limits=dict(overall_rms_nm=max_validation_rms_nm,
                                       joint_rms_nm=max_joint_validation_rms_nm),
                applied=False)
