"""Deterministic native checkerboard association, without vision or hardware I/O."""

import copy
import json

import numpy as np
from scipy.spatial.transform import Rotation


def _rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Hand-eye observation needs a finite rigid transform')
    return t.copy()


class HandeyeAssociation:
    """Associate a compact run to its first visible physical grid origin.

    prepare() never changes accepted state; commit() follows robot post-checks.
    No fitted residual participates in association. A new run needs a new object.
    """

    def __init__(self):
        self.reference = None
        self.accepted = []

    def prepare(self, observation, base_from_tcp):
        o = copy.deepcopy(observation)
        try:
            grid = o['grid']
            columns, rows, square = grid['columns'], grid['rows'], grid['square_mm']
            if (type(columns) is not int or type(rows) is not int or min(columns, rows) < 2
                    or isinstance(square, bool) or not isinstance(square, (int, float))
                    or not np.isfinite(square) or square <= 0):
                raise ValueError('Invalid native checkerboard grid')
            size = np.asarray(o['image_size'], float)
            points = np.asarray(o['corners_native_px'], float)
            if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0)
                    or np.any(size != np.floor(size)) or points.shape != (columns*rows, 2)
                    or not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= size)
                    or len(np.unique(points, axis=0)) != len(points)):
                raise ValueError('Full finite native checkerboard corners required')
            if (o['native_validation'] is not True or o['units'] != 'mm'
                    or o['coordinate_frame'] != 'camera'):
                raise ValueError('Hand-eye requires native camera measurements in mm')
            identity = (o['source_id'], o['calibration_id'], o['generation'], o['acquisition_session_id'],
                        columns, rows, float(square), tuple(size))
            if (any(not isinstance(v, str) or not v for v in (identity[0], identity[1], identity[3]))
                    or type(identity[2]) is not int or identity[2] < 0):
                raise ValueError('Hand-eye needs source, calibration and acquisition identity')
            if type(o['sequence']) is not int or o['sequence'] < 0:
                raise ValueError('Invalid native frame sequence')
            started, completed = o['read_started'], o['read_completed']
            if (isinstance(started, bool) or isinstance(completed, bool)
                    or not isinstance(started, (int, float)) or not isinstance(completed, (int, float))
                    or not np.isfinite([started, completed]).all() or not 0 <= started <= completed):
                raise ValueError('Invalid native acquisition timestamps')
            if not isinstance(o['timing'], dict):
                raise ValueError('Native acquisition timing metadata required')
            rms = o['reprojection_rms_px']
            if not np.isfinite(rms) or not 0 <= rms <= 1.5:
                raise ValueError('Native target pose reprojection RMS exceeds 1.5px')
            raw = _rigid(o['camera_from_target'])
            if raw[2, 3] <= 0:
                raise ValueError('Native target pose must have positive depth')
            raw[:3, 3] /= 1000
            base = _rigid(base_from_tcp)
            axis = points[columns-1] - points[0]
            if np.linalg.norm(axis) < 1:
                raise ValueError('Degenerate checkerboard image axis')
            axis /= np.linalg.norm(axis)
            y_axis = points[(rows-1)*columns]-points[0]
            if np.linalg.norm(y_axis) < 1:
                raise ValueError('Degenerate checkerboard image geometry')
            y_axis /= np.linalg.norm(y_axis)
            lattice = points.reshape(rows, columns, 2)
            if (abs(axis[0]*y_axis[1]-axis[1]*y_axis[0]) < .1
                    or np.any(np.diff(lattice, axis=1) @ axis <= 0)
                    or np.any(np.diff(lattice, axis=0) @ y_axis <= 0)):
                raise ValueError('Invalid native checkerboard lattice geometry')
            reversed_grid, dot, rotation_deg = False, 1., 0.
            if self.reference is not None:
                ref = self.reference
                if identity != ref['identity']:
                    raise ValueError('Hand-eye source/calibration/grid/configuration changed')
                if (o['sequence'] <= self.accepted[-1]['sequence']
                        or started <= self.accepted[-1]['read_started']):
                    raise ValueError('Duplicate or out-of-order native hand-eye frame')
                dot = float(axis @ ref['axis'])
                rotation_deg = float(np.rad2deg(Rotation.from_matrix(ref['base'][:3, :3].T @ base[:3, :3]).magnitude()))
                limit = np.cos(np.deg2rad(20))
                if dot < -limit and rotation_deg < 20:
                    reversed_grid = True
                elif dot <= limit:
                    raise ValueError('Ambiguous checkerboard orientation or large physical rotation')
                if (-y_axis if reversed_grid else y_axis) @ ref['y_axis'] <= limit:
                    raise ValueError('Ambiguous checkerboard Y orientation')
            board180 = np.eye(4)
            board180[:3, :3] = np.diag([-1., -1., 1.])
            board180[:3, 3] = [(columns-1)*square/1000, (rows-1)*square/1000, 0]
            normal = raw @ board180 if reversed_grid else raw.copy()
            normalized_points = points[::-1].copy() if reversed_grid else points.copy()
            sample = dict(units='m', camera_pose_frame='camera_from_target', base_from_tcp=base.tolist(), camera_from_target_m=normal.tolist(),
                          source_id=o['source_id'], calibration_id=o['calibration_id'],
                          sequence=o['sequence'], generation=o['generation'], read_started=started,
                          read_completed=completed, acquisition_session_id=o['acquisition_session_id'],
                          timing=o['timing'], native_validation=True, grid=grid, image_size=o['image_size'],
                          controls=o.get('controls'), reprojection_rms_px=rms,
                          raw_camera_from_target_m=raw.tolist(), raw_corners_native_px=points.tolist(),
                          corners_native_px=normalized_points.tolist(),
                          normalization=dict(index=len(self.accepted), reversed_180=reversed_grid,
                              rule='first-view grid X; antiparallel dot<-cos20deg and TCP rotation<20deg',
                              reference_index=0, image_axis_dot=dot, tcp_rotation_from_reference_deg=rotation_deg,
                              board180_transform_m=board180.tolist()))
            json.dumps(sample, allow_nan=False)
            return sample
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f'Invalid native hand-eye observation: {exc}') from exc

    def commit(self, sample):
        """Commit only after measured post-capture robot checks succeeded."""
        if sample['normalization']['index'] != len(self.accepted):
            raise ValueError('Stale hand-eye association result')
        if self.reference is None:
            points = np.asarray(sample['corners_native_px'])
            grid = sample['grid']
            axis = points[grid['columns']-1]-points[0]
            y_axis = points[(grid['rows']-1)*grid['columns']]-points[0]
            self.reference = dict(axis=axis/np.linalg.norm(axis), y_axis=y_axis/np.linalg.norm(y_axis), base=np.asarray(sample['base_from_tcp']),
                identity=(sample['source_id'], sample['calibration_id'], sample['generation'],
                          sample['acquisition_session_id'], grid['columns'], grid['rows'],
                          float(grid['square_mm']), tuple(sample['image_size'])))
        self.accepted.append(copy.deepcopy(sample))
