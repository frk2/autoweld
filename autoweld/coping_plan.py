"""Calibrated laser illumination and complete pose planning for a coped joint.

Only pure geometry and a worker-private kinematic DATA instance are used. The
installed compiled model is shared; no device, robot writer or model is created.
"""

import copy
import time

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.calibration import transform


class _PreflightDeadline(TimeoutError):
    """Optional search time ended; completed proofs can still be retained."""


def unit(value, label='direction'):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-10:
        raise ValueError(f'Invalid coping {label}')
    return value/np.linalg.norm(value)


def laser_seam_ray(plane_mm, standoff_mm=150.):
    """A real ray on the calibrated laser plane at the requested radial range.

    Its intersection with the range sphere is chosen closest to optical +Z.
    This retains the real stripe offset instead of moving its origin onto +Z.
    """
    plane = np.asarray(plane_mm, float)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError('Coping illumination needs a finite calibrated laser plane')
    normal = unit(plane[:3], 'laser plane normal')
    offset = float(plane[3]/np.linalg.norm(plane[:3]))
    if not np.isfinite(standoff_mm) or not 100 <= standoff_mm <= 200 or abs(offset) >= standoff_mm:
        raise ValueError('Calibrated laser plane cannot illuminate the requested sensing range')
    optical = np.array([0., 0., 1.])
    tangent = unit(optical-normal*np.dot(optical, normal), 'forward laser plane direction')
    ray = -offset*normal+np.sqrt(standoff_mm**2-offset**2)*tangent
    if ray[2] <= 0:
        raise ValueError('Calibrated coping stripe is behind the camera')
    return ray


def coping_camera_pose(point_base_m, tangent_base, inward_base, laser_plane_mm, tcp_from_camera, *,
                       reference_camera=None, view_tilt_deg=0., standoff_mm=150., normal_sign=None,
                       view_direction_base=None):
    """Aim the finite sensing ray at the seam and the laser plane across it.

    The sensing view may differ from the inward torch reference. At a coped
    edge, the visible side of the finite assembly need not face the branch's
    centerline; forcing that direction can move the camera behind the tubes.
    An explicit sensing direction still projects into the seam cross-section,
    illuminates the real calibrated plane and retains the requested range.
    """
    point = np.asarray(point_base_m, float)
    if point.shape != (3,) or not np.isfinite(point).all() or abs(view_tilt_deg) > 10:
        raise ValueError('Invalid coping point or bounded view tilt')
    normal_camera = unit(np.asarray(laser_plane_mm, float)[:3])
    z = np.array([0., 0., 1.])
    optical_in_plane = unit(z-normal_camera*np.dot(normal_camera, z))
    camera_basis = np.column_stack((optical_in_plane, np.cross(normal_camera, optical_in_plane), normal_camera))
    ray = laser_seam_ray(laser_plane_mm, standoff_mm)/1000.
    choices = []
    if normal_sign not in (None, -1, 1):
        raise ValueError('Coping normal family must be explicitly +1 or -1')
    toward = unit(inward_base, 'inward direction')
    view = toward if view_direction_base is None else unit(view_direction_base, 'sensing view')
    for sign in ((1., -1.) if normal_sign is None else (float(normal_sign),)):
        normal_world = unit(tangent_base)*sign
        forward = unit(view-normal_world*np.dot(view, normal_world), 'cross-section sensing view')
        forward = Rotation.from_rotvec(normal_world*np.deg2rad(view_tilt_deg)).apply(forward)
        world_basis = np.column_stack((forward, np.cross(normal_world, forward), normal_world))
        camera = np.eye(4)
        camera[:3, :3] = world_basis @ camera_basis.T
        camera[:3, 3] = point-camera[:3, :3] @ ray
        angle = (0. if reference_camera is None else
                 Rotation.from_matrix(camera[:3, :3] @ transform(reference_camera)[:3, :3].T).magnitude())
        choices.append((angle, camera))
    camera = min(choices, key=lambda item: item[0])[1]
    local = camera[:3, :3].T @ (point-camera[:3, 3])*1000.
    if abs(np.asarray(laser_plane_mm)[:3] @ local+laser_plane_mm[3]) > 1e-7:
        raise ValueError('Coping target is outside its calibrated laser illumination plane')
    return camera @ np.linalg.inv(transform(tcp_from_camera)), camera


def transport_measured_camera_pose(point_base_m, tangent_base, laser_plane_mm, tcp_from_camera,
                                   measured_camera_reference, branch_axis_base, relative_angle_deg, *,
                                   angular_orientation=1., view_tilt_deg=0., standoff_mm=150.):
    """Transport a demonstrated camera orientation without forcing a new roll.

    Exact perpendicularity to the seam is unnecessary: the laser plane must
    cross it by at least 30 degrees. A bounded recovery tilt rotates around the
    world laser-plane normal, preserving that plane's crossing orientation.
    This is sensing geometry only and never replaces the inward torch frame.
    """
    point = np.asarray(point_base_m, float)
    if (point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(relative_angle_deg)
            or angular_orientation not in (-1., 1.) or not np.isfinite(view_tilt_deg) or abs(view_tilt_deg) > 5.):
        raise ValueError('Invalid measured-camera coping transport or bounded recovery tilt')
    reference = transform(measured_camera_reference)
    axis = unit(branch_axis_base, 'measured-camera transport axis')
    normal_camera = unit(np.asarray(laser_plane_mm, float)[:3], 'laser plane normal')
    rotation = Rotation.from_rotvec(axis*np.deg2rad(relative_angle_deg)*angular_orientation).as_matrix() @ reference[:3, :3]
    normal_world = rotation @ normal_camera
    rotation = Rotation.from_rotvec(normal_world*np.deg2rad(view_tilt_deg)).as_matrix() @ rotation
    crossing = float((rotation @ normal_camera) @ unit(tangent_base, 'seam tangent'))
    if abs(crossing) < .5:
        error = ValueError('Transported measured laser plane crosses the seam by less than 30 degrees')
        error.boundary_kind = 'laser_seam_crossing_unobservable'
        error.diagnostic = dict(code='laser_seam_crossing_unobservable',
            signed_normal_dot_tangent=crossing, minimum_absolute_normal_dot_tangent=.5,
            crossing_angle_deg=float(np.rad2deg(np.arcsin(np.clip(abs(crossing), 0., 1.)))),
            minimum_crossing_angle_deg=30., relative_angle_deg=float(relative_angle_deg))
        raise error
    ray = laser_seam_ray(laser_plane_mm, standoff_mm)/1000.
    camera = np.eye(4)
    camera[:3, :3], camera[:3, 3] = rotation, point-rotation @ ray
    local = rotation.T @ (point-camera[:3, 3])*1000.
    if abs(np.asarray(laser_plane_mm)[:3] @ local+laser_plane_mm[3]) > 1e-7:
        raise ValueError('Transported coping target is outside its calibrated laser illumination plane')
    return camera @ np.linalg.inv(transform(tcp_from_camera)), camera


def measured_angle_deg(point_mm, seed_mm, axis):
    origin, direction = np.asarray(axis['origin_mm'], float), unit(axis['direction'])
    seed = np.asarray(seed_mm, float)-origin
    first = unit(seed-direction*np.dot(direction, seed), 'seed radial direction')
    second = np.cross(direction, first)
    radial = np.asarray(point_mm, float)-origin
    radial -= direction*np.dot(direction, radial)
    radial = unit(radial, 'measured radial direction')
    return float(np.rad2deg(np.arctan2(radial @ second, radial @ first)))


def plan_coping_views(existing_planner, start_q, bounds, base_from_seed_camera, tcp_from_camera,
                      laser_plane_mm, curve, *, cancelled=lambda: False, progress=lambda state: None,
                      max_seconds=180., fitted_tubes=None, measured_camera_reference=None):
    """Validate both complete pose branches, retaining maximal safe prefixes."""
    from autoweld.diagnostics import solve_pose
    from autoweld.scan_plan_probe import private_planner, private_self_collision

    planner = private_planner(existing_planner)
    planner.limits = copy.copy(existing_planner.limits)
    frame, handeye = transform(base_from_seed_camera), transform(tcp_from_camera)
    points, inward, angles = (np.asarray(curve[key], float) for key in
        ('points_mm', 'inward_directions', 'relative_angles_deg'))
    tangents = np.asarray([value if value is not None else [np.nan]*3 for value in curve['tangents']], float)
    if (points.ndim != 2 or points.shape[1] != 3 or tangents.shape != points.shape or inward.shape != points.shape
            or angles.shape != (len(points),) or len(points) < 3
            or not np.isfinite(np.r_[points.ravel(), inward.ravel(), angles]).all()):
        raise ValueError('A finite analytic coping curve is required for whole-pose preflight')
    seed_q, bounds = np.asarray(start_q, float).copy(), np.asarray(bounds, float).copy()
    gripper, started = float(seed_q[6]), time.monotonic()
    budget_exhausted = False

    def guard(q):
        if cancelled():
            raise TimeoutError('Coping full-pose preflight cancelled')
        if time.monotonic()-started > max_seconds:
            raise _PreflightDeadline('Coping full-pose preflight exceeded its bounded search deadline')
        if (np.asarray(q).shape != (7,) or not np.isfinite(q).all() or q[6] != gripper
                or np.any(q[:6] < bounds[:, 0]) or np.any(q[:6] > bounds[:, 1])):
            raise ValueError('Coping pose violates the fixed gripper or installed joint bounds')

    def collision(q):
        if private_self_collision(planner, q):
            return True
        if fitted_tubes:
            from autoweld.coping_clearance import check_tube_clearance
            diagnostic = check_tube_clearance(planner.model, planner.data, fitted_tubes)
            if diagnostic is not None:
                error = ValueError('Conservative fitted-tube workpiece clearance boundary')
                error.diagnostic = diagnostic
                error.boundary_kind = 'conservative_fitted_tube_clearance'
                raise error
        return False

    def build(index, last_q, reference, normal_sign, view_family, base_tilt):
        nonlocal budget_exhausted
        guard(last_q)
        point = frame[:3, :3] @ (points[index]/1000.)+frame[:3, 3]
        tangent, toward = frame[:3, :3] @ tangents[index], frame[:3, :3] @ inward[index]
        unit(toward, 'inward direction')
        view = (None if view_family in ('radial', 'transport_measured_camera') else
                Rotation.from_rotvec(transport_axis*np.deg2rad(angles[index])*orientation).apply(visible_reference))

        def make_pose(tilt):
            if view_family == 'transport_measured_camera':
                return transport_measured_camera_pose(point, tangent, laser_plane_mm, handeye, measured_reference,
                    transport_axis, float(angles[index]), angular_orientation=orientation, view_tilt_deg=tilt)
            return coping_camera_pose(point, tangent, toward, laser_plane_mm, handeye,
                reference_camera=reference, normal_sign=normal_sign, view_direction_base=view, view_tilt_deg=tilt)

        pose, camera = make_pose(base_tilt)
        q = solve_pose(planner.kin, pose, planner.site_name, planner._configuration(last_q), stage='coping_pose_ik')
        target = np.r_[q[:6], gripper]
        motion = planner.plan(last_q, target, cartesian=pose, bounds=bounds, validate=guard,
                              collision=collision)
        alternatives = []
        for relative_tilt in (-5., 5.):
            tilt = base_tilt+relative_tilt
            if abs(tilt) > 10.:
                continue
            try:
                guard(target)
                alternate, _ = make_pose(tilt)
                aq = solve_pose(planner.kin, alternate, planner.site_name, planner._configuration(target),
                                stage='coping_view_recovery_ik')
                proof = planner.plan(target, np.r_[aq[:6], gripper], cartesian=alternate, bounds=bounds,
                                      validate=guard, collision=collision)
            except _PreflightDeadline:
                # The primary motion above is already wholly validated. A
                # half-proved optional recovery pose must never enter it.
                budget_exhausted = True
                break
            except (ValueError, RuntimeError):
                continue
            alternatives.append(dict(view_tilt_deg=tilt, relative_view_tilt_deg=relative_tilt,
                base_from_tcp=alternate.tolist(), proof=proof.report))
        crossing = float((camera[:3, :3] @ unit(np.asarray(laser_plane_mm, float)[:3])) @ unit(tangent))
        return dict(angle_deg=float(angles[index]), normal_sign=int(np.sign(crossing)), sensing_view_family=view_family,
            base_view_tilt_deg=base_tilt, base_from_tcp=pose.tolist(),
            base_from_camera=camera.tolist(), predicted_point_base_m=point.tolist(),
            predicted_point_seed_camera_mm=points[index].tolist(), tangent_base=tangent.tolist(),
            inward_base=toward.tolist(), laser_plane_normal_dot_seam_tangent=crossing,
            minimum_absolute_laser_plane_crossing=.5 if view_family == 'transport_measured_camera' else 1.,
            proof=motion.report, alternatives=alternatives), target, camera

    zero = int(np.argmin(abs(angles)))
    if abs(angles[zero]) > 1e-8:
        raise ValueError('Coping curve must contain the seed station at zero degrees')
    original_camera = planner.fk(seed_q).copy() @ handeye
    measured_reference = None if measured_camera_reference is None else transform(measured_camera_reference)
    view_families = ['radial']
    transport_axis = visible_reference = None
    orientation = 1.
    if fitted_tubes and curve.get('coping_axis') in (0, 1):
        # Transport the measured visible side around the selected branch axis.
        # Do not chase the previous planned view, which could drift or change
        # families, and do not substitute this sensing direction for the
        # model's inward torch direction retained in every station.
        transport_axis = unit(fitted_tubes[curve['coping_axis']]['direction'], 'coping transport axis')
        orientation = float(curve.get('orientation', 1.))
        if orientation not in (-1., 1.):
            raise ValueError('Coping angular orientation must be explicitly +1 or -1')
        normal_camera = unit(np.asarray(laser_plane_mm, float)[:3], 'laser plane normal')
        optical = np.array([0., 0., 1.])
        visible_reference = original_camera[:3, :3] @ unit(
            optical-normal_camera*np.dot(normal_camera, optical), 'measured in-plane optical view')
        view_families.insert(0, 'measured_visible_side')
    if measured_reference is not None and transport_axis is None:
        raise ValueError('Measured-camera transport requires the selected fitted branch axis')
    families, failures = [], []
    candidates = [(family, tilt, sign) for family in view_families
        for tilt in ((0., -5., 5., -10., 10.) if family == 'measured_visible_side' else (0.,)) for sign in (1, -1)]
    if measured_reference is not None:
        candidates.insert(0, ('transport_measured_camera', 0., None))
    for view_family, base_tilt, normal_sign in candidates:
        try:
            center, centered_q, centered_camera = build(zero, seed_q, original_camera, normal_sign, view_family, base_tilt)
            result = dict(status='qualified', normal_sign=center['normal_sign'], sensing_view_family=view_family, center=center,
                base_view_tilt_deg=base_tilt,
                branches={'positive': [], 'negative': []}, boundaries={},
                proof_owner='private DATA on shared installed MODEL; normal MotionPlanner for every full pose')
            for sign, name in ((1, 'positive'), (-1, 'negative')):
                last_q, reference = centered_q.copy(), centered_camera.copy()
                indices = sorted(np.flatnonzero(angles*sign > 1e-8), key=lambda index: abs(angles[index]))
                events = [event for event in curve.get('unresolved', []) if event['relative_angle_deg']*sign > 1e-8]
                nearest_event = min(events, key=lambda event: abs(event['relative_angle_deg'])) if events else None
                for index in indices:
                    if nearest_event is not None and abs(angles[index]) >= abs(nearest_event['relative_angle_deg'])-1e-8:
                        result['boundaries'][name] = dict(angle_deg=nearest_event['relative_angle_deg'],
                            reason=nearest_event['reason'], kind='analytic_intersection_unobservable')
                        break
                    try:
                        guard(last_q)
                        progress(dict(status='planning', normal_sign=normal_sign, sensing_view_family=view_family,
                            base_view_tilt_deg=base_tilt, direction=name, angle_deg=float(angles[index])))
                        station, last_q, reference = build(index, last_q, reference, normal_sign, view_family, base_tilt)
                    except _PreflightDeadline as error:
                        budget_exhausted = True
                        result['boundaries'][name] = dict(angle_deg=float(angles[index]), reason=str(error),
                            kind='preflight_time_budget', unfinished_station_included=False)
                        break
                    except (ValueError, RuntimeError) as error:
                        result['boundaries'][name] = dict(angle_deg=float(angles[index]), reason=str(error),
                            diagnostic=getattr(error, 'diagnostic', None),
                            kind=getattr(error, 'boundary_kind', 'kinematic_or_collision_preflight'))
                        break
                    result['branches'][name].append(station)
            result['positive_bound_deg'] = max([0., *[item['angle_deg'] for item in result['branches']['positive']]])
            result['negative_bound_deg'] = min([0., *[item['angle_deg'] for item in result['branches']['negative']]])
            result['planned_arc_deg'] = result['positive_bound_deg']-result['negative_bound_deg']
            result['center_rotation_from_measured_rad'] = float(Rotation.from_matrix(
                centered_camera[:3, :3] @ original_camera[:3, :3].T).magnitude())
            result['preflight_budget_exhausted'] = budget_exhausted
            families.append(result)
            if budget_exhausted or view_family == 'transport_measured_camera' and result['planned_arc_deg'] > 0.:
                break
        except _PreflightDeadline:
            if not families:
                raise
            failures.append(dict(normal_sign=normal_sign, sensing_view_family=view_family,
                base_view_tilt_deg=base_tilt,
                reason='Alternative family exceeded bounded preflight deadline'))
            budget_exhausted = True
            break
        except (ValueError, RuntimeError) as error:
            failures.append(dict(normal_sign=normal_sign, sensing_view_family=view_family,
                base_view_tilt_deg=base_tilt, reason=str(error), diagnostic=getattr(error, 'diagnostic', None)))
    if cancelled():
        raise TimeoutError('Coping full-pose preflight cancelled')
    if not families:
        error = ValueError('Both coping camera normal families failed center pose preflight')
        error.diagnostic = {'center_failures': failures}
        raise error
    supported = [result for result in families
                 if result['sensing_view_family'] == 'transport_measured_camera' and result['planned_arc_deg'] > 0.]
    selected = max(supported or families, key=lambda result: (result['planned_arc_deg'], -result['center_rotation_from_measured_rad']))
    selected['preflight_budget_exhausted'] = budget_exhausted
    selected['normal_family_comparison'] = [dict(normal_sign=result['normal_sign'],
        sensing_view_family=result['sensing_view_family'], base_view_tilt_deg=result['base_view_tilt_deg'],
        planned_arc_deg=result['planned_arc_deg'], positive_bound_deg=result['positive_bound_deg'],
        negative_bound_deg=result['negative_bound_deg']) for result in families]
    selected['normal_family_failures'] = failures
    selected['sensing_reference'] = dict(
        policy='prefer a demonstrated full camera orientation with a nonzero qualified prefix; pure viewing families are fallbacks',
        measured_camera_reference_base_m=None if measured_reference is None else measured_reference.tolist(),
        measured_camera_transport='entire rotation around selected branch axis; center rotation unchanged',
        minimum_measured_camera_crossing_angle_deg=30.,
        transport_axis_base=None if transport_axis is None else transport_axis.tolist(),
        measured_in_plane_view_base=None if visible_reference is None else visible_reference.tolist(),
        angular_orientation=orientation, torch_direction_policy='unchanged model inward direction')
    selected['return_policy'] = 'Reverse the preflighted full-pose station sequence; no end-to-center shortcut'
    selected['fitted_workpiece_obstacles'] = fitted_tubes or []
    return selected
