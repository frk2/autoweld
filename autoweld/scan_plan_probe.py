"""Read-only continuous-raster planning on the running robot's EXISTING model.

All entry arguments are snapshots. Call private_planner/build_scan_plan only in
an asynchronous planning worker, then return a compact report to the main owner.
No function opens hardware, creates a model, enables motors or submits a target.
The SAME builder/collision/constraints can serve autonomous main-raster planning.
"""

import numpy as np

from autoweld.continuous_raster import ContinuousRaster, RasterPlanBuilder, RasterSettings


def private_planner(existing_planner):
    """New private DATA, shared original MODEL; no model compilation/device reads."""
    import mink
    from i2rt.robots.kinematics import Kinematics

    from autoweld.motion import MotionPlanner

    kin = Kinematics.__new__(Kinematics)
    kin._configuration = mink.Configuration(existing_planner.model)
    kin._site_name = existing_planner.site_name
    planner = MotionPlanner(None, site=existing_planner.site_name, kin=kin)
    planner.limits = existing_planner.limits
    return planner


def private_self_collision(planner, q):
    """Match service collision exclusions, using ONLY worker-private data."""
    planner.fk(q)
    return _contacts_collide(planner)


def _contacts_collide(planner):
    import mujoco

    model, data = planner.model, planner.data
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist >= -.001:
            continue
        if any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in (contact.geom1, contact.geom2)):
            continue
        first, second = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if model.body_parentid[first] != second and model.body_parentid[second] != first:
            return True
    return False


def registration_budget(planner, path, tcp_from_camera, *, encoder_quantization_rad=25. / 65535.,
                        target_standoff_m=.140, fov_half_width_m=.025, max_registration_error_m=.0005):
    """Quantization-only diagnostic; not a registration/timing qualification.

    Report both the legacy registrar's conservative separate lever-norm bound
    and a tighter point-Jacobian first-order bound. Neither includes bracket,
    exposure, delay, calibration or other error, and exceeding a conservative
    bound does not prove actual physical error exceeds the accuracy limit.
    """
    handeye = RasterPlanBuilder._transform(tcp_from_camera, 'registration hand-eye')
    values = np.asarray([encoder_quantization_rad, target_standoff_m, fov_half_width_m,
                         max_registration_error_m], float)
    if not np.isfinite(values).all() or encoder_quantization_rad < 0 or np.any(values[1:] <= 0):
        raise ValueError('Expected finite nonnegative quantization and positive registration geometry')
    half = encoder_quantization_rad / 2
    d = fov_half_width_m
    camera_points = np.array([[0., 0., target_standoff_m],
                              [-d, -d, target_standoff_m], [-d, d, target_standoff_m],
                              [d, -d, target_standoff_m], [d, d, target_standoff_m]])
    nominal_points = camera_points @ handeye[:3, :3].T + handeye[:3, 3]
    samples = []
    for name, fraction in (('entry', 0.), ('middle', .5), ('end', 1.)):
        q = path.sample(path.duration * fraction)[0]
        pose = np.asarray(planner.fk(q), float).copy()
        jacobian = np.asarray(planner.jacobian(q), float).copy()
        if pose.shape != (4, 4) or jacobian.shape != (6, 6) or not np.isfinite(pose).all() or not np.isfinite(jacobian).all():
            raise ValueError('Registration diagnostic requires finite pose and 6x6 Jacobian')
        levers = nominal_points @ pose[:3, :3].T
        maximum_lever = float(np.max(np.linalg.norm(levers, axis=1)))
        sensitivity = np.linalg.norm(jacobian[:3], axis=0) + maximum_lever * np.linalg.norm(jacobian[3:], axis=0)
        conservative = float(np.sum(sensitivity) * half)
        exact_bounds = []
        for lever in levers:
            point_jacobian = jacobian[:3] + np.cross(jacobian[3:].T, lever).T
            exact_bounds.append(float(np.sum(np.linalg.norm(point_jacobian, axis=0)) * half))
        samples.append({'phase': name, 'elapsed_s': float(path.duration * fraction),
                        'maximum_tcp_lever_m': maximum_lever,
                        'registrar_quantization_bound_m': conservative,
                        'point_specific_center_bound_m': exact_bounds[0],
                        'point_specific_max_bound_m': max(exact_bounds),
                        'point_specific_bounds_m': exact_bounds,
                        'registrar_sensitivity_m_per_rad': sensitivity.tolist(),
                        'registrar_quantization_budget_passed': conservative <= max_registration_error_m})
    return {'diagnostic_only': True, 'encoder_half_quantization_rad': float(half),
            'active_registrar_bound_model': 'point-specific Jacobian; separate-norm bound retained for comparison',
            'max_registration_error_m': max_registration_error_m,
            'camera_test_points_m': camera_points.tolist(), 'samples': samples,
            'worst_registrar_quantization_bound_m': max(row['registrar_quantization_bound_m'] for row in samples),
            'worst_point_specific_quantization_bound_m': max(row['point_specific_max_bound_m'] for row in samples),
            'excluded_errors': 'Timing, interpolation, exposure blur and calibration uncertainty are not included'}


def build_scan_plan(planner, *, baseline_pose, tcp_from_camera, seed_q, bounds,
                    gripper=0., optical_roll_deg=-45., depth_offset_m=0., depth=None,
                    feed_mm_s=20., width_mm=100., height_mm=100., spacing_mm=1.,
                    max_preflight_s=120., max_total_s=900., cancelled=None, progress=None,
                    model_fingerprint=None, include_registration_budget=True,
                    camera_xy_shift_m=(0., 0.), sweep_axis='x', lane_spacing_mm=None):
    """Build/validate a whole timed raster; planner must have worker-private DATA.

    depth(x,y), if provided, yields original optical-frame camera-Z displacement,
    not absolute world height. Its frozen support/uncertainty remains separate.
    A constant depth_offset_m is diagnostic flat-plane planning only, not evidence
    of object geometry. Feed30 is an explicit candidate; never selected silently.
    """
    from scipy.spatial.transform import Rotation

    from autoweld.diagnostics import MotionRejection

    baseline = RasterPlanBuilder._transform(baseline_pose, 'original nominal Tube')
    handeye = RasterPlanBuilder._transform(tcp_from_camera, 'measured hand-eye')
    limits = np.asarray(bounds, float).copy()
    seed = np.asarray(seed_q, float).copy()
    if seed.shape == (6,):
        seed = np.r_[seed, gripper]
    if seed.shape != (7,) or not np.isfinite(seed).all() or not np.isfinite(gripper):
        raise ValueError('Expected finite six/seven-joint diagnostic seed and gripper')
    if not np.isclose(seed[6], gripper, atol=1e-12, rtol=0):
        raise ValueError('Diagnostic seed must retain captured gripper')
    if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Expected finite installed six-joint bounds')
    if np.any(seed[:6] < limits[:, 0]) or np.any(seed[:6] > limits[:, 1]):
        raise ValueError('Diagnostic IK seed exceeds installed joint bounds')
    values = np.asarray([feed_mm_s, width_mm, height_mm, spacing_mm, optical_roll_deg, depth_offset_m], float)
    if (not np.isfinite(values).all() or not 0 < feed_mm_s <= 30.
            or not 0 < width_mm <= 100. or not 0 < height_mm <= 100.
            or not 0 < spacing_mm <= 25. or abs(optical_roll_deg) > 180. or abs(depth_offset_m) > .060):
        raise ValueError('Expected bounded finite continuous scan feed/square/roll/depth')
    shift = np.asarray(camera_xy_shift_m, float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError('Camera XY footprint compensation must be a finite metre vector')
    if sweep_axis not in ('x', 'y'):
        raise ValueError('Fast sweep axis must be x or y')
    if lane_spacing_mm is not None and (not np.isfinite(lane_spacing_mm) or not 0 < lane_spacing_mm <= 20.):
        raise ValueError('Measured illumination lane spacing must be in (0,20] mm')
    # At larger explicitly requested feed, retain acceleration and reduce jerk
    # by lengthening row ramps rather than raising any configured motion ceiling.
    settings = RasterSettings(width_m=(width_mm if sweep_axis == 'x' else height_mm) / 1000.,
                              height_m=(height_mm if sweep_axis == 'x' else width_mm) / 1000.,
                              spacing_m=spacing_mm / 1000., cruise_m_s=feed_mm_s / 1000.,
                              ramp_s=.6 * max(1., feed_mm_s / 20.),
                              lane_spacing_m=None if lane_spacing_mm is None else lane_spacing_mm / 1000.)
    optical = baseline @ handeye
    fixed_rotation = optical[:3, :3] @ Rotation.from_euler('z', optical_roll_deg, degrees=True).as_matrix()
    position_axes = np.eye(3) if sweep_axis == 'x' else Rotation.from_euler('z', 90., degrees=True).as_matrix()
    position_baseline = optical.copy()
    position_baseline[:3, :3] = optical[:3, :3] @ position_axes
    position_baseline[:3, 3] += optical[:3, :3] @ np.r_[shift, 0.]

    def position_depth(x, y):
        original_xy = (position_axes @ np.array([x, y, 0.]))[:2] + shift
        return depth_offset_m if depth is None else depth(float(original_xy[0]), float(original_xy[1]))

    def validate(q):
        if q.shape != (7,) or not np.isfinite(q).all():
            raise MotionRejection('raster_preflight', 'joint_nonfinite', 'Invalid raster joint reference')
        if not np.isclose(q[6], gripper, atol=1e-12, rtol=0):
            raise MotionRejection('raster_preflight', 'gripper_motion', 'Captured gripper must remain fixed')
        if np.any(q[:6] < limits[:, 0]) or np.any(q[:6] > limits[:, 1]):
            raise MotionRejection('raster_preflight', 'joint_bounds', 'Installed raster joint bounds exceeded')

    # Collision requires full forward dynamics. Its immediately following FK
    # residual query is at EXACTLY the same q; reuse that pose, not another
    # mj_forward. The next collision always performs a fresh forward pass, so
    # intervening Jacobian calls cannot leave stale contacts or stale kinematics.
    collision_q, collision_pose = None, None

    def collision(q):
        nonlocal collision_q, collision_pose
        collision_pose = np.array(planner.fk(q), float, copy=True)
        collision_q = q.copy()
        return _contacts_collide(planner)

    def cached_fk(q):
        if collision_q is not None and np.array_equal(q, collision_q):
            return collision_pose.copy()
        return planner.fk(q)

    builder = RasterPlanBuilder.from_planner(
        planner, ContinuousRaster(settings), seed, baseline_optical=position_baseline,
        fixed_optical_rotation=fixed_rotation, tcp_from_camera=handeye, bounds=limits,
        validate=validate, collision=collision, fk=cached_fk,
        depth=position_depth,
        max_preflight_s=max_preflight_s, max_total_s=max_total_s)
    try:
        plan = builder.run(cancelled=cancelled, progress=progress)
    except Exception as error:
        diagnostic = getattr(error, 'diagnostic', None)
        if isinstance(diagnostic, dict):
            diagnostic['preflight_progress'] = {key: value for key, value in builder.state().items() if key != 'error'}
            diagnostic['reference_duration_s'] = builder.raster.duration
            if include_registration_budget and builder.path is not None:
                try:
                    diagnostic['registration_budget'] = registration_budget(planner, builder.path, handeye)
                except Exception as budget_error:
                    diagnostic['registration_budget_error'] = str(budget_error)
        raise
    plan.report['scan_scope'].update(
        original_roi_baseline_optical=optical.tolist(), roi_width_m=width_mm / 1000.,
        roi_height_m=height_mm / 1000., position_grid_axes=position_axes.tolist(),
        camera_xy_shift_m=shift.tolist(), sweep_axis=sweep_axis,
        lane_spacing_m=None if lane_spacing_mm is None else lane_spacing_mm / 1000.,
        actual_lane_spacing_m=plan.report['actual_lane_spacing_m'])
    plan.report['sweep_axis'] = sweep_axis
    plan.report['camera_xy_shift_m'] = shift.tolist()
    plan.report['model_fingerprint'] = model_fingerprint
    plan.report['feed_mm_s'] = feed_mm_s
    plan.report['optical_roll_deg'] = optical_roll_deg
    plan.report['diagnostic_constant_depth'] = depth is None
    plan.report['entry_pose'] = planner.fk(plan.path.positions[0].copy()).tolist()
    plan.report['entry_q'] = plan.path.positions[0].tolist()
    if include_registration_budget:
        plan.report['registration_budget'] = registration_budget(planner, plan.path, handeye)
    return plan


def compact_plan_summary(plan):
    """No path, dense joint knots or acquisition payload in UI/disk status."""
    return {'read_only': True, 'validated': True, **plan.report}
