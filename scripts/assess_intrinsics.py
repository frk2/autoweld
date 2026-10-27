"""Offline native-checkpoint export and intrinsic calibration diagnostics.

This never starts a camera/server, touches robot hardware, or installs a fitted
camera model. A low residual is reported separately from diversity, uncertainty,
held-out corner prediction and focal-length stability.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


def _object_points(board):
    result = np.zeros((board['rows'] * board['columns'], 3), np.float32)
    result[:, :2] = np.mgrid[:board['columns'], :board['rows']].T.reshape(-1, 2) * board['square_mm']
    return result


def _fit(points, objects, image_size, flags, initial=None):
    k, d = (None, None) if initial is None else (np.array(initial['camera_matrix'], float),
                                                np.array(initial['distortion'], float))
    if initial is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    rms, k, d, rotations, translations, std_i, _, per_view = cv2.calibrateCameraExtended(
        [objects] * len(points), points, tuple(image_size), k, d, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-10))
    std_i = std_i.reshape(-1)
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rotations])
    angle = np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1., 1.))).max()
    centered = normals - normals.mean(axis=0)
    spread = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(normals))
    return dict(rms_px=float(rms), camera_matrix=k.tolist(), distortion=d.reshape(-1).tolist(),
                per_view_rms_px=per_view.reshape(-1).tolist(),
                intrinsic_std=std_i.tolist(), focal_relative_std=(std_i[:2] / [k[0, 0], k[1, 1]]).tolist(),
                max_normal_separation_deg=float(angle), normal_spread_singular_values=spread.tolist(),
                normal_second_axis_spread_deg=float(np.rad2deg(np.arcsin(np.clip(spread[1], 0, 1)))),
                board_depth_mm=[float(t.reshape(-1)[2]) for t in translations])


def _heldout_errors(points, objects, fit):
    k, d = np.array(fit['camera_matrix']), np.array(fit['distortion'])
    result = []
    for corners in points:
        ok, r, t = cv2.solvePnP(objects, corners, k, d)
        if not ok or t.reshape(-1)[2] <= 0:
            raise ValueError('Held-out view has no valid positive-depth board pose')
        projected = cv2.projectPoints(objects, r, t, k, d)[0].reshape(-1, 2)
        result.append(float(np.sqrt(np.mean(np.sum((projected - corners.reshape(-1, 2))**2, axis=1)))))
    return result


def _homography_spectrum(points, objects, image_size):
    rows = []
    for corners in points:
        normalized = (corners.reshape(-1, 2) - np.asarray(image_size) / 2) / max(image_size)
        h = cv2.findHomography(objects[:, :2], normalized, method=0)[0]
        if h is None:
            continue
        def v(i, j):
            return np.array([h[0, i] * h[0, j], h[0, i] * h[1, j] + h[1, i] * h[0, j],
                             h[1, i] * h[1, j], h[2, i] * h[0, j] + h[0, i] * h[2, j],
                             h[2, i] * h[1, j] + h[1, i] * h[2, j], h[2, i] * h[2, j]])
        for row in (v(0, 1), v(0, 0) - v(1, 1)):
            rows.append(row / np.linalg.norm(row))
    values = np.linalg.svd(np.asarray(rows), compute_uv=False)
    return dict(singular_values=values.tolist(),
                fifth_to_first=float(values[4] / values[0]) if len(values) >= 5 else None,
                nullspace_gap=float(values[-2] / max(values[-1], 1e-15)) if len(values) >= 6 else None)


def assess_checkpoint(checkpoint_path, *, model='standard', folds=5, initial=None,
                      output_directory=None, export_native=False):
    """Analyze retained measured corners; preserve all source images and captures."""
    checkpoint_path = Path(checkpoint_path)
    data = json.loads(checkpoint_path.read_text())
    settings_path = checkpoint_path.parent / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    required_views = max(20, int(settings.get('views', 20)))
    board, size = data['board'], data['image_size']
    points = [np.asarray(v, np.float32).reshape(-1, 1, 2) for v in data['camera_views']]
    objects = _object_points(board)
    if not points or size is None:
        raise ValueError('Checkpoint has no accepted native camera views')
    if any(p.shape != (len(objects), 1, 2) or not np.isfinite(p).all() for p in points):
        raise ValueError('Checkpoint contains incomplete/non-finite native corner measurements')
    out = Path(output_directory) if output_directory else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack([p.reshape(-1, 2) for p in points])
    centers = np.array([p.reshape(-1, 2).mean(0) for p in points])
    distances = np.full((len(points), len(points)), np.inf)
    for i in range(len(points)):
        for j in range(i):
            a, b = points[i].reshape(-1, 2), points[j].reshape(-1, 2)
            distances[i, j] = distances[j, i] = min(np.linalg.norm(a-b, axis=1).mean(),
                                                   np.linalg.norm(a-b[::-1], axis=1).mean())
    nearest = distances.min(axis=1).tolist() if len(points) > 1 else None
    captures = data.get('captures', {}).get('camera', [])
    image_diagnostics = []
    for index, capture in enumerate(captures):
        encoded = data.get('images', {}).get(capture['id'], {}).get('native')
        if not encoded:
            continue
        raw = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or [image.shape[1], image.shape[0]] != size:
            raise ValueError('Native capture image differs from recorded calibration size')
        image_diagnostics.append(dict(index=index, id=capture['id'],
            brightness_percentiles=np.percentile(image, [1, 5, 50, 95, 99]).tolist(),
            saturation_fraction=float(np.mean(image >= 250)), black_fraction=float(np.mean(image <= 5)),
            metadata=capture.get('robocam')))
        if export_native and out:
            (out / f'native-{index:03d}-{capture["id"]}.png').write_bytes(raw)
    report = dict(source_checkpoint=str(checkpoint_path.resolve()), source_id=data['source_id'],
                  image_size=size, board=board, accepted_views=len(points),
                  required_session_views=required_views,
                  calibration_model=model,
                  measurement_provenance='Retained checkpoint corners; source authenticity not independently verified',
                  image_diagnostics=image_diagnostics,
                  coverage=dict(corner_min_px=all_points.min(0).tolist(), corner_max_px=all_points.max(0).tolist(),
                                corner_span_fraction=(np.ptp(all_points, axis=0) / size).tolist(),
                                center_span_fraction=(np.ptp(centers, axis=0) / size).tolist(),
                                nearest_view_mean_corner_distance_px=nearest),
                  homography_constraints=_homography_spectrum(points, objects, size),
                  warnings=[], installed=False)
    if len(points) < 3:
        report['warnings'].append('Fewer than three views: focal length and working distance cannot be independently estimated')
        report['existing_quality_gates_passed'] = False
    else:
        flags = {'standard': 0, 'fix-k3': cv2.CALIB_FIX_K3,
                 'zero-distortion': cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                                    cv2.CALIB_ZERO_TANGENT_DIST}[model]
        fit = _fit(points, objects, size, flags, initial)
        report['fit'] = fit
        report['existing_quality_gates_passed'] = bool(len(points) >= required_views and fit['rms_px'] <= 1.
                                                       and fit['max_normal_separation_deg'] >= 12.)
        if fit['normal_second_axis_spread_deg'] < 2.:
            report['warnings'].append('Weak second tilt axis: one angular span alone does not establish adequate conditioning')
        if max(fit['focal_relative_std']) > .1:
            report['warnings'].append('Estimated focal-length standard deviation exceeds 10%; inspect conditioning')
        k = np.asarray(fit['camera_matrix'])
        if not (0 < k[0, 2] < size[0] and 0 < k[1, 2] < size[1]):
            report['warnings'].append('Principal point lies outside the sensor image')
        if len(points) >= 25 and folds >= 2:
            fold_results = []
            for fold in range(folds):
                held = [i for i in range(len(points)) if i % folds == fold]
                train = [i for i in range(len(points)) if i not in held]
                if len(train) < 20 or not held:
                    continue
                candidate = _fit([points[i] for i in train], objects, size, flags, initial)
                errors = _heldout_errors([points[i] for i in held], objects, candidate)
                fold_results.append(dict(training_indices=train, heldout_indices=held,
                                         heldout_rms_px=errors, training_rms_px=candidate['rms_px'],
                                         fx=candidate['camera_matrix'][0][0], fy=candidate['camera_matrix'][1][1]))
            if fold_results:
                fxy = np.array([[f['fx'], f['fy']] for f in fold_results])
                errors = [v for f in fold_results for v in f['heldout_rms_px']]
                report['cross_validation'] = dict(folds=fold_results, worst_view_rms_px=max(errors),
                    aggregate_corner_rms_px=float(np.sqrt(np.mean(np.array(errors)**2))),
                    focal_relative_range=(np.ptp(fxy, axis=0) / np.mean(fxy, axis=0)).tolist(),
                    interpretation='Intrinsics are trained without held-out views; each held-out board pose is solved from its corners')
                if max(errors) > 1.:
                    report['warnings'].append('At least one held-out view exceeds 1px RMS')
                if max(report['cross_validation']['focal_relative_range']) > .1:
                    report['warnings'].append('Focal lengths vary by more than 10% across independent training folds')
    if np.any(np.ptp(centers, axis=0) / size < .05):
        report['warnings'].append('View centers cover less than 5% of an image axis; centered ring views may weakly constrain lens distortion')
    report['interpretation'] = ('Existing gates and offline diagnostics are separate. Neither reprojection RMS nor '
                                'these heuristic warnings establish metric accuracy, unchanged optics, or a validated hand-eye transform.')
    if out:
        (out / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--export-native', action='store_true')
    parser.add_argument('--model', choices=['standard', 'fix-k3', 'zero-distortion'], default='standard')
    parser.add_argument('--initial-calibration', type=Path)
    args = parser.parse_args()
    initial = json.loads(args.initial_calibration.read_text()) if args.initial_calibration else None
    cv2.setNumThreads(2)
    report = assess_checkpoint(args.checkpoint, model=args.model, initial=initial,
                               output_directory=args.output, export_native=args.export_native)
    print(json.dumps({k: v for k, v in report.items() if k not in ('image_diagnostics',)}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
