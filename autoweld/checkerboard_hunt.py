"""Pure geometry and bounded checkerboard reacquisition; no devices or models.

The caller supplies fresh detections after each settled move and sends proposed
poses through its existing SDK planner. Poses use camera optical axes: +X right,
+Y down, +Z forward. Detector code owns partial-lattice validation.
"""
import copy

import numpy as np

STEP_M = .005
_EPS = 1e-12
_REVISIT_M = .001


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Expected a finite 4x4 camera pose')
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError('Camera pose must contain a proper rigid rotation')
    return pose.copy()


def _corners(value):
    if value is None:
        return np.empty((0, 2))
    corners = np.asarray(value, dtype=float)
    if corners.size == 0:
        return np.empty((0, 2))
    if corners.ndim == 3 and corners.shape[1:] == (1, 2):
        corners = corners[:, 0]
    if corners.ndim != 2 or corners.shape[1] != 2 or not np.isfinite(corners).all():
        raise ValueError('Expected finite Nx2 checkerboard corners')
    return corners


def centroid_correction(corners_px, image_size, base_from_camera, *, step_m=STEP_M,
                        center_tolerance_px=1.):
    """Return an exact 5 mm image-plane translation and JSON-ready metadata.

    A missing direction returns None, never an invented centroid vector. At least
    four detector-validated lattice corners are required for a correction.
    """
    pose = _pose(base_from_camera)
    size = np.asarray(image_size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Image size must be positive finite [width, height]')
    if not np.isfinite(step_m) or not np.isclose(step_m, STEP_M, atol=1e-12, rtol=0):
        raise ValueError('Checkerboard hunt increments must be exactly 5 mm')
    if not np.isfinite(center_tolerance_px) or center_tolerance_px < 0:
        raise ValueError('Center tolerance must be finite and nonnegative')
    corners = _corners(corners_px)
    centroid = corners.mean(axis=0) if len(corners) else None
    offset = centroid-size/2 if centroid is not None else None
    metadata = {'corner_count': len(corners),
                'centroid_px': centroid.tolist() if centroid is not None else None,
                'offset_px': offset.tolist() if offset is not None else None,
                'step_mm': STEP_M*1000}
    if len(corners) < 4:
        return None, {**metadata, 'centroid_reason': 'insufficient_corners'}
    length = float(np.linalg.norm(offset))
    if length <= center_tolerance_px:
        return None, {**metadata, 'centroid_reason': 'partial_centered'}
    camera_delta = np.r_[STEP_M*offset/length, 0.]
    return pose[:3, :3]@camera_delta, {**metadata, 'centroid_reason': 'centroid'}


def _square_spiral(radius):
    """Integer camera-XY lattice, beginning right, down, left, up."""
    x = y = 0
    length = 1
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    index = 0
    # A complete enclosing square includes every lattice point inside the disk.
    count = (2*radius+1)**2-1
    points = []
    while len(points) < count:
        for _ in range(2):
            dx, dy = directions[index % 4]
            for _ in range(length):
                x, y = x+dx, y+dy
                points.append(np.array([x, y], dtype=float)*STEP_M)
                if len(points) == count:
                    return points
            index += 1
        length += 1
    return points


class CheckerboardHunt:
    """One bounded hunt anchored at a requested baseline/capture camera pose.

    next_step records a returned proposal once. Call it only on a fresh frame
    after that move settles; cancel this hunt if planning/execution is rejected.
    snapshot is read-only and can be used while waiting for motion or frames.
    """
    step_m = STEP_M

    def __init__(self, anchor_pose, started_at, *, max_steps=40, max_radius_m=.03,
                 max_path_m=.20, timeout_s=60., center_tolerance_px=1.):
        self.anchor = _pose(anchor_pose)
        if not np.isfinite(started_at):
            raise ValueError('Hunt start time must be finite')
        if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or not 1 <= max_steps <= 40:
            raise ValueError('Hunt step limit must be an integer in [1, 40]')
        for name, value, ceiling in (('radius', max_radius_m, .03), ('path', max_path_m, .20),
                                     ('timeout', timeout_s, 60.)):
            if not np.isfinite(value) or not 0 < value <= ceiling:
                raise ValueError(f'Hunt {name} limit must be positive and at most {ceiling}')
        if not np.isfinite(center_tolerance_px) or center_tolerance_px < 0:
            raise ValueError('Center tolerance must be finite and nonnegative')
        self.started_at = float(started_at)
        self.max_steps, self.max_radius_m = int(max_steps), float(max_radius_m)
        self.max_path_m, self.timeout_s = float(max_path_m), float(timeout_s)
        self.center_tolerance_px = float(center_tolerance_px)
        self.steps, self.path_m = 0, 0.
        self.visited = [self.anchor[:3, 3].copy()]
        self._spiral = _square_spiral(int(np.ceil(max_radius_m/STEP_M)))
        self._spiral_index = 0
        self._terminal = None
        self._last = {'corner_count': 0, 'centroid_px': None, 'offset_px': None, 'reacquiring': False}

    def snapshot(self, now):
        if not np.isfinite(now) or now < self.started_at:
            raise ValueError('Hunt time must be finite and no earlier than its start')
        return {**copy.deepcopy(self._last), 'steps': self.steps, 'max_steps': self.max_steps,
                'step_mm': STEP_M*1000, 'path_m': float(self.path_m), 'max_path_m': self.max_path_m,
                'max_radius_m': self.max_radius_m, 'elapsed_s': float(now-self.started_at),
                'timeout_s': self.timeout_s}

    def cancel(self, now, reason='cancelled'):
        self._terminal = ('cancelled', str(reason))
        return self._result(now, *self._terminal)

    def _result(self, now, status, reason, *, target=None, delta=None):
        return {**self.snapshot(now), 'status': status, 'reason': reason,
                'target_pose': target.tolist() if target is not None else None,
                'translation_base_m': delta.tolist() if delta is not None else None}

    def _allowed(self, position):
        return (np.linalg.norm(position-self.anchor[:3, 3]) <= self.max_radius_m+_EPS
                and all(np.linalg.norm(position-prior) >= _REVISIT_M for prior in self.visited))

    def _search_delta(self, current):
        """Follow the anchor spiral, bridging off-grid centroid moves in 5 mm.

        An off-grid point can lie less than one step from a spiral vertex. A
        two-step equilateral-circle bridge reaches it without shortening a move
        or bouncing across it; both bridge directions are checked for revisits.
        """
        anchor_rotation = self.anchor[:3, :3]
        xy = (anchor_rotation.T@(current[:3, 3]-self.anchor[:3, 3]))[:2]
        while self._spiral_index < len(self._spiral):
            waypoint = self._spiral[self._spiral_index]
            desired = self.anchor[:3, 3]+anchor_rotation@np.r_[waypoint, 0.]
            if not self._allowed(desired):
                self._spiral_index += 1
                continue
            difference = waypoint-xy
            length = float(np.linalg.norm(difference))
            if length <= _EPS:
                self._spiral_index += 1
                continue
            direction = difference/length
            if length >= STEP_M-_EPS:
                candidates = [xy+STEP_M*direction]
            else:
                normal = np.array([-direction[1], direction[0]])
                height = np.sqrt(max(0., STEP_M**2-(length/2)**2))
                midpoint = (xy+waypoint)/2
                candidates = [midpoint+height*normal, midpoint-height*normal]
            for point in candidates:
                base_delta = anchor_rotation@np.r_[point-xy, 0.]
                # Preserve the current camera depth even with small measured
                # orientation changes from numerical IK/tracking residuals.
                camera_delta = current[:3, :3].T@base_delta
                camera_delta[2] = 0.
                norm = float(np.linalg.norm(camera_delta))
                if norm <= _EPS:
                    continue
                delta = current[:3, :3]@(camera_delta*(STEP_M/norm))
                if self._allowed(current[:3, 3]+delta):
                    return delta
            self._spiral_index += 1
        return None

    def next_step(self, current_pose, corners_px, image_size, *, full_board=False, now):
        if self._terminal is not None:
            return self._result(now, *self._terminal)
        metadata = self.snapshot(now)
        if full_board:
            _, full_metadata = centroid_correction(corners_px, image_size, current_pose,
                                                    center_tolerance_px=self.center_tolerance_px)
            self._last = {**full_metadata, 'reacquiring': False}
            self._terminal = ('complete', 'full_board')
            return self._result(now, *self._terminal)
        limits = ((metadata['elapsed_s'] >= self.timeout_s, 'timeout'),
                  (self.steps >= self.max_steps, 'step_limit'),
                  (self.path_m+STEP_M > self.max_path_m+_EPS, 'path_limit'))
        for exceeded, reason in limits:
            if exceeded:
                self._terminal = ('exhausted', reason)
                return self._result(now, *self._terminal)
        current = _pose(current_pose)
        if np.linalg.norm(current[:3, 3]-self.anchor[:3, 3]) > self.max_radius_m+_EPS:
            self._terminal = ('exhausted', 'radius_limit')
            return self._result(now, *self._terminal)
        delta, metadata = centroid_correction(corners_px, image_size, current,
                                               center_tolerance_px=self.center_tolerance_px)
        self._last = {**metadata, 'reacquiring': False}
        if delta is None or not self._allowed(current[:3, 3]+delta):
            self._last['reacquiring'] = True
            delta = self._search_delta(current)
        if delta is None:
            self._terminal = ('exhausted', 'search_space_exhausted')
            return self._result(now, *self._terminal)
        target = current.copy()
        target[:3, 3] += delta
        self.steps += 1
        self.path_m += float(np.linalg.norm(delta))
        self.visited.append(target[:3, 3].copy())
        return self._result(now, 'move', 'reacquisition' if self._last['reacquiring'] else 'centroid',
                            target=target, delta=delta)
