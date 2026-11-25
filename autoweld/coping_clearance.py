"""Conservative fitted-workpiece bounds using the caller's existing MuJoCo data.

The caller must update ``data.geom_xpos`` for each candidate configuration and
express both tube axes in that same robot-base frame, in metres. No forward
kinematics, model compilation, device access or model mutation occurs here.

Each physical robot geom is enclosed by its MuJoCo ``geom_rbound`` sphere. When
that bound overlaps, a valid MuJoCo local ``geom_aabb`` and world ``geom_xmat``
provide a tighter enclosing box. Its projected convex hull, not just its corners,
is checked against the cylinder. Missing/invalid boxes retain sphere rejection.
Workpieces remain infinite solid cylinders, including unseen axial extensions;
intersection of these bounds is not evidence of actual surface contact.
"""

import numpy as np

# mjtGeom.mjGEOM_PLANE, from the installed MuJoCo public enum. Importing or
# compiling a MuJoCo model is unnecessary for tests or this pure bounds check.
_PLANE = 0
_BOX_SIGNS = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
_DISTANCE_CUSHION_M = 1e-9


def _number(value, label, *, positive=False):
    if (not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f'{label} must be a finite {"positive" if positive else "nonnegative"} number')
    return float(value)


def _vector(value, label):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must contain three finite robot-base coordinates')
    return result


def _axis(tube, index):
    if not isinstance(tube, dict):
        raise ValueError('Fitted tube axes must be dictionaries with origin_m, direction and radius_m')
    origin = _vector(tube.get('origin_m', tube.get('origin_base_m')), 'Tube axis origin')
    direction = _vector(tube.get('direction', tube.get('axis_base')), 'Tube axis direction')
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Tube axis direction must have finite nonzero length')
    return origin, direction/length, _number(tube.get('radius_m'), 'Tube radius', positive=True), str(tube.get('label', index))


def _name(model, kind, index):
    lookup = getattr(model, kind, None)
    if callable(lookup):
        value = getattr(lookup(index), 'name', None)
        if value:
            return str(value)
    names = getattr(model, kind+'_names', None)
    if names is not None and names[index]:
        return str(names[index])
    return f'{kind}_{index}'


def _obb_corners(model, data, geom, count, position):
    """Return trusted native local-box corners in world coordinates, or fallback.

    MuJoCo's compiler stores (local center, half-sizes) in geom_aabb. Its
    collision engine applies geom_xmat to that local center before geom_xpos.
    No model field is inferred from mesh size or an enclosing sphere.
    """
    if not hasattr(model, 'geom_aabb') or not hasattr(data, 'geom_xmat'):
        return None, 'Local geom bounds or world orientation unavailable'
    try:
        bounds = np.asarray(model.geom_aabb, float)
        rotations = np.asarray(data.geom_xmat, float)
        if bounds.shape != (count, 6) or rotations.shape not in ((count, 9), (count, 3, 3)):
            return None, 'Local geom bounds or world orientation shape invalid'
        center, half = bounds[geom, :3], bounds[geom, 3:]
        rotation = rotations[geom].reshape(3, 3)
        if (not np.isfinite(bounds[geom]).all() or np.any(half < 0) or not np.any(half > 0)
                or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
                or abs(np.linalg.det(rotation)-1.) > 1e-6):
            return None, 'Local geom bounds or proper world rotation invalid'
        corners = (center+_BOX_SIGNS*half) @ rotation.T+position
        if not np.isfinite(corners).all():
            return None, 'World geom bounding-box corners are nonfinite'
        return corners, None
    except (TypeError, ValueError, IndexError):
        return None, 'Local geom bounds or world orientation could not be validated'


def _cross2(origin, first, second):
    return ((first[0]-origin[0])*(second[1]-origin[1])
            - (first[1]-origin[1])*(second[0]-origin[0]))


def _origin_hull_distance(points):
    """Exact convex-polygon distance in 2D, with point/segment degeneracies."""
    unique = sorted(set(map(tuple, points.tolist())))
    if len(unique) == 1:
        return float(np.linalg.norm(unique[0]))
    lower, upper = [], []
    for point in unique:
        while len(lower) >= 2 and _cross2(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and _cross2(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1]+upper[:-1], float)
    if len(hull) >= 3 and all(_cross2(first, second, (0., 0.)) >= -1e-12
            for first, second in zip(hull, np.roll(hull, -1, axis=0))):
        return 0.
    distance = np.inf
    for first, second in zip(hull, np.roll(hull, -1, axis=0)):
        delta = second-first
        square = float(delta @ delta)
        fraction = float(np.clip(-(first @ delta)/square, 0., 1.)) if square > 0 else 0.
        distance = min(distance, float(np.linalg.norm(first+fraction*delta)))
    return distance


def _projected_obb_distance(corners, origin, direction):
    """Lower bound on radial geom distance to an infinite fitted tube axis.

    Linear projection preserves inclusion: actual geom ⊆ OBB implies that its
    projection is contained by the convex hull of all eight projected corners.
    An axis passing through the hull has zero distance even if every corner is
    far away. Subtract one nanometre for conservative floating-point roundoff.
    """
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    projected = (corners-origin) @ np.column_stack((first, second))
    if not np.isfinite(projected).all():
        return None
    distance = _origin_hull_distance(projected)
    return max(0., distance-_DISTANCE_CUSHION_M) if np.isfinite(distance) else None


def check_tube_clearance(model, data, axes_base_m, margin_m=.002):
    """Return the worst intersecting bound diagnostic, or ``None`` if separated.

    ``axes_base_m`` contains exactly two ``{origin_m, direction, radius_m}``
    dictionaries, with optional ``label``. Direction is normalized without
    changing the caller's arrays. Compatibility aliases are ``origin_base_m``
    and ``axis_base``. Geometry is selected by collision flags and body ID, never
    mass: fixed, zero-mass receiver/camera collision boxes remain included.

    World-body geoms, planes, and geoms with both contact flags zero are omitted.
    Sites are a separate MuJoCo array and cannot enter this geom-only check.
    Malformed/missing physical bounds raise ``ValueError`` rather than passing.
    """
    margin = _number(margin_m, 'Tube clearance margin')
    try:
        axes = list(axes_base_m)
    except TypeError as error:
        raise ValueError('Clearance needs the two fitted tube axes') from error
    if len(axes) != 2:
        raise ValueError('Clearance needs exactly two fitted tube axes')
    tubes = [_axis(tube, index) for index, tube in enumerate(axes)]
    if (not hasattr(data, 'geom_xpos') or any(not hasattr(model, field) for field in
            ('geom_rbound', 'geom_bodyid', 'geom_type', 'geom_contype', 'geom_conaffinity'))):
        raise ValueError('Existing model/data are missing required geom clearance arrays')
    positions = np.asarray(data.geom_xpos, float)
    radii = np.asarray(model.geom_rbound, float)
    body_ids = np.asarray(model.geom_bodyid)
    types = np.asarray(model.geom_type)
    contact = np.asarray(model.geom_contype)
    affinity = np.asarray(model.geom_conaffinity)
    if radii.ndim != 1:
        raise ValueError('Existing model geom bounds must be one-dimensional')
    count = len(radii)
    if (positions.shape != (count, 3) or any(array.shape != (count,)
            for array in (body_ids, types, contact, affinity)) or count == 0):
        raise ValueError('Existing model/data must provide aligned nonempty geom bounds and world positions')
    if (any(not np.issubdtype(array.dtype, np.integer) for array in (body_ids, types, contact, affinity))
            or np.any(body_ids < 0)):
        raise ValueError('Existing model geom body IDs, types and collision flags must be valid integer metadata')
    physical = (body_ids > 0) & (types != _PLANE) & ((contact != 0) | (affinity != 0))
    indices = np.flatnonzero(physical)
    if not len(indices):
        raise ValueError('No physical robot collision geoms were available for fitted tube clearance')
    selected_positions, selected_radii = positions[indices], radii[indices]
    if (not np.isfinite(selected_positions).all() or not np.isfinite(selected_radii).all()
            or np.any(selected_radii <= 0)):
        raise ValueError('Physical robot geoms need finite measured positions and positive bounding-sphere radii')
    worst, box_cache = None, {}
    for tube_index, (origin, direction, radius, label) in enumerate(tubes):
        radial = np.linalg.norm(np.cross(selected_positions-origin, direction), axis=1)
        clearance = radial-selected_radii-radius-margin
        if not np.isfinite(clearance).all():
            raise ValueError('Fitted tube clearance arithmetic must remain finite')
        for selected in np.flatnonzero(clearance <= 0):
            selected = int(selected)
            geom = int(indices[selected])
            value = sphere_clearance = float(clearance[selected])
            if geom not in box_cache:
                box_cache[geom] = _obb_corners(model, data, geom, count, selected_positions[selected])
            corners, fallback_reason = box_cache[geom]
            refined = None if corners is None else _projected_obb_distance(corners, origin, direction)
            if refined is not None:
                value = refined-radius-margin
                if value > 0:
                    continue  # A tighter independently enclosing box proves separation.
            elif corners is not None:
                fallback_reason = 'Projected local bounding-box distance could not be validated'
            if worst is not None and value >= worst['clearance_m']:
                continue
            worst = dict(reason='conservative fitted tube clearance', code='conservative_fitted_tube_clearance',
                geom_index=geom, geom_name=_name(model, 'geom', geom), body_index=int(body_ids[geom]),
                body_name=_name(model, 'body', int(body_ids[geom])), tube_index=tube_index, tube_label=label,
                clearance_m=value, clearance_mm=value*1000., margin_m=margin, margin_mm=margin*1000.,
                geom_bound_radius_mm=float(selected_radii[selected]*1000.), tube_radius_mm=radius*1000.,
                radial_center_distance_mm=float(radial[selected]*1000.),
                geom_center_base_m=selected_positions[selected].tolist(), axis_origin_base_m=origin.tolist(),
                axis_direction_base=direction.tolist(), checked_geom_count=len(indices),
                bound_kind=('geom_local_obb_projection_vs_infinite_fitted_solid_cylinder' if refined is not None else
                            'geom_bounding_sphere_vs_infinite_fitted_solid_cylinder'),
                sphere_clearance_mm=sphere_clearance*1000.,
                obb_radial_distance_lower_bound_mm=None if refined is None else refined*1000.,
                obb_fallback_reason=fallback_reason,
                actual_surface_contact_confirmed=False,
                limitation='Infinite tube extents and enclosing geom bounds can reject physically clear poses.')
    return worst


def fitted_tube_clearance(model, data, tubes, margin_m=.002):
    """Boolean-plus-diagnostic adapter for planner collision callback composition."""
    diagnostic = check_tube_clearance(model, data, tubes, margin_m=margin_m)
    return diagnostic is not None, diagnostic
