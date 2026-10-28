"""Export a completed 101 x 101 raster's original optical-XY ROI.

This offline reader never imports robot/camera owners or changes source files.
Run only after the archive reports complete. Native per-view files and the full
uncropped export remain the measurement record; this adds a separate ROI PLY.
"""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

EXPECTED_CELLS = 10201
EDGE_EPSILON_M = 1e-9


def rigid(value):
    value = np.asarray(value, float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError('Expected a finite rigid transform in metres')
    return value


def records(path, digest):
    with path.open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                raise ValueError('Completed scan has a torn cell journal line')
            digest.update(line)
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('Cell journal contains a non-object record')
            yield record


def view_points(record, session_directory):
    index = record.get('cell_index')
    if type(index) is not int or not 0 <= index < EXPECTED_CELLS:
        raise ValueError('Cell index must be an integer between 0 and 10200')
    # Derive the canonical immutable source path, avoiding CWD-dependent paths.
    path = session_directory/f'{index:05d}.npz'
    if not path.is_file() or Path(record.get('raw_snapshot_path', '')).name != path.name:
        raise ValueError(f'Missing or mismatched native view for cell {index}')
    outcome = record.get('status')
    if outcome not in ('captured', 'no_return', 'rejected'):
        raise ValueError(f'Unknown recorded outcome at cell {index}')
    if type(record.get('count')) is not int or record['count'] < 0:
        raise ValueError(f'Invalid point count at cell {index}')
    if outcome != 'captured':
        if record['count'] != 0:
            raise ValueError('Rejected/no-return cells must not contribute accepted points')
        return np.empty((0, 3))
    with np.load(path, allow_pickle=False) as raw:
        if 'xyz_world_m' in raw:
            points = np.asarray(raw['xyz_world_m'], float)
        else:
            optical = rigid(raw['base_from_tcp']) @ rigid(raw['tcp_from_camera'])
            native = np.asarray(raw['xyz_camera_m'], float)[raw['selected_indices']]
            points = native @ optical[:3, :3].T+optical[:3, 3]
    if points.ndim != 2 or points.shape != (record['count'], 3) or not np.isfinite(points).all():
        raise ValueError(f'Invalid world coordinates at cell {index}')
    return points


def roi_points(points, original_optical):
    # Row-vector inverse of the ORIGINAL Tube optical transform. The chosen
    # capture roll and depth-adjusted camera origin do not redefine the square.
    local = (points-original_optical[:3, 3]) @ original_optical[:3, :3]
    keep = (np.abs(local[:, 0]) <= .050+EDGE_EPSILON_M) & (np.abs(local[:, 1]) <= .050+EDGE_EPSILON_M)
    return points[keep], local[keep]


def export_roi(summary_path, output_directory):
    summary_path = Path(summary_path).resolve()
    encoded_summary = summary_path.read_bytes()
    summary = json.loads(encoded_summary)
    if (summary.get('status') != 'complete' or summary.get('cells_visited') != EXPECTED_CELLS
            or summary.get('next_index') != EXPECTED_CELLS):
        raise ValueError('ROI export requires a completed scan with all 10201 grid cells visited')
    if summary.get('units') != 'm' or summary.get('coordinate_frame') != 'robot_base':
        raise ValueError('ROI export requires metre world coordinates in robot_base')
    metadata = summary['metadata']
    settings = metadata['settings']
    if (settings.get('width_mm') != 100 or settings.get('height_mm') != 100
            or settings.get('step_mm') != 1):
        raise ValueError('ROI export requires the requested 100 x 100 mm, 1 mm raster')
    session = summary.get('session_id')
    if not isinstance(session, str) or not session or Path(session).name != session:
        raise ValueError('Invalid source scan session identity')
    session_directory = summary_path.parent/session
    journal = session_directory/'cells.jsonl'
    original_optical = rigid(metadata['baseline_pose']) @ rigid(metadata['handeye']['tcp_from_camera'])
    output_directory = Path(output_directory).resolve()
    ply = output_directory/(session+'-roi-100x100.ply')
    report = output_directory/(session+'-roi-100x100.json')
    if ply.exists() or report.exists():
        raise FileExistsError('Separate ROI export already exists; existing files will not be overwritten')
    seen = set()
    outcomes = {'captured': 0, 'no_return': 0, 'rejected': 0}
    full_count = roi_count = 0
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    first_digest = hashlib.sha256()
    # First pass validates complete unique coverage and counts native ROI points.
    for record in records(journal, first_digest):
        index = record.get('cell_index')
        if index in seen:
            raise ValueError(f'Duplicate source grid cell {index}')
        points = view_points(record, session_directory)
        seen.add(index)
        outcomes[record['status']] += 1
        selected, local = roi_points(points, original_optical)
        full_count += len(points)
        roi_count += len(selected)
        if len(local):
            minimum = np.minimum(minimum, local.min(0))
            maximum = np.maximum(maximum, local.max(0))
    if seen != set(range(EXPECTED_CELLS)):
        raise ValueError('Source journal does not contain exactly 10201 unique grid cells')
    if full_count != summary.get('count') or outcomes['captured'] != summary.get('capture_count'):
        raise ValueError('Source journal/NPZ totals do not agree with the completed summary')
    if summary_path.read_bytes() != encoded_summary:
        raise ValueError('Source summary changed during ROI counting')
    output_directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temp_ply = output_directory/(ply.name+'.tmp-'+token)
    temp_report = output_directory/(report.name+'.tmp-'+token)
    written = 0
    second_digest = hashlib.sha256()
    try:
        # Second pass streams measured WORLD points. ROI membership never changes
        # those coordinates, selected native indices, or any source view.
        with temp_ply.open('x') as handle:
            handle.write('ply\nformat ascii 1.0\ncomment units meters; frame robot_base\n'
                         'comment ROI is original baseline optical XY, 100 x 100 mm\n'
                         f'element vertex {roi_count}\n'
                         'property double x\nproperty double y\nproperty double z\nend_header\n')
            for record in records(journal, second_digest):
                selected, _ = roi_points(view_points(record, session_directory), original_optical)
                np.savetxt(handle, selected, fmt='%.12f')
                written += len(selected)
            handle.flush()
            os.fsync(handle.fileno())
        if (written != roi_count or first_digest.digest() != second_digest.digest()
                or summary_path.read_bytes() != encoded_summary):
            raise ValueError('Source scan changed during ROI export; refusing to publish it')
        result = {'source_session_id': session, 'source_summary': str(summary_path),
            'source_summary_sha256': hashlib.sha256(encoded_summary).hexdigest(),
            'source_journal': str(journal), 'source_journal_sha256': first_digest.hexdigest(),
            'unique_grid_cells': len(seen), 'cell_outcomes': outcomes,
            'units': 'm', 'coordinate_frame': 'robot_base',
            'roi_frame': 'original_baseline_camera_optical',
            'base_from_original_baseline_camera': original_optical.tolist(),
            'roi_xy_bounds_m': [[-.050, -.050], [.050, .050]],
            'edge_epsilon_m': EDGE_EPSILON_M, 'z_filter': None,
            'source_world_point_count': full_count, 'roi_world_point_count': roi_count,
            'outside_roi_point_count': full_count-roi_count,
            'roi_points_bounds_in_original_optical_m': ([minimum.tolist(), maximum.tolist()] if roi_count else None),
            'native_views_preserved': True, 'source_exports_preserved': True,
            'ply': str(ply), 'report': str(report)}
        with temp_report.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if a destination appeared;
        # unlike replace(), it cannot overwrite an existing export.
        os.link(temp_ply, ply)
        os.link(temp_report, report)
    finally:
        temp_ply.unlink(missing_ok=True)
        temp_report.unlink(missing_ok=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('runtime/agents-live/laser-scans/latest.json'))
    parser.add_argument('--output-directory', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(export_roi(args.summary, args.output_directory), indent=2))
