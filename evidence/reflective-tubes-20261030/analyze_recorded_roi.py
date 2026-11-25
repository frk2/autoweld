"""Reprocess recorded native frames only; calibration identity is mandatory.

Example from AutoWeld root:
  .venv/bin/python evidence/reflective-tubes-20261030/analyze_recorded_roi.py \
    --frames evidence/reflective-tubes-20261030/tilt-views/x+5 \
    --roi 300 140 650 260 --threshold 4 --output /tmp/roi-report.json
No hardware is opened, no robot model is loaded and no calibration is modified.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent/'Weld-seam-finder/app'))
from weldfinder.calibration import Calibration
from weldfinder.config import Detection
from weldfinder.geometry import project_camera_points
from weldfinder.optical_identity import optical_intrinsics_id
from weldfinder.pointcloud import PointCloudNoise, find_laser_point_cloud
from weldfinder.seam import SeamSettings, find_weld_seam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=PROJECT/'runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json')
    parser.add_argument('--roi', nargs=4, type=int, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'))
    parser.add_argument('--threshold', type=float, default=4.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    model = Calibration.load(args.calibration)
    payload = model.to_dict()
    calibration_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    optical_id = optical_intrinsics_id(payload)
    roi = (np.asarray(args.roi)/np.tile(model.image_size, 2)).tolist()
    detector = Detection(threshold=args.threshold, min_snr=4., min_width=2, max_width=24, roi=roi)
    noise = PointCloudNoise(enabled=True, min_run_points=12)
    settings = SeamSettings(feature_kinds=('inside_corner',))
    observations = []
    paths = sorted(args.frames.rglob('snapshot.json'))
    if not paths:
        raise ValueError('No native snapshots found')
    for path in paths:
        snapshot = json.loads(path.read_text())
        metadata = snapshot['metadata']
        if (metadata['calibration_id'] != calibration_id
                or metadata['optical_intrinsics_id'] != optical_id
                or metadata['source_id'] != model.source_id
                or snapshot['image']['image_size'] != model.image_size):
            raise ValueError(f'Native frame/calibration identity mismatch: {path}')
        image_path = path.with_name('native.png')
        png = image_path.read_bytes()
        if hashlib.sha256(png).hexdigest() != snapshot['image']['sha256']:
            raise ValueError(f'Native PNG hash mismatch: {path}')
        frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        if (list(frame.shape[1::-1]) != model.image_size
                or hashlib.sha256(frame.tobytes()).hexdigest() != snapshot['image']['native_pixels_sha256']):
            raise ValueError(f'Native pixel bytes mismatch: {path}')
        cloud = find_laser_point_cloud(frame, detector, model, metadata['source_id'],
            metadata={**metadata, 'reprocessed_offline': True}, noise=noise)
        result = find_weld_seam(cloud, settings=settings)
        projection = distances = None
        if result['found']:
            point = np.asarray(result['seam_point_mm'])
            projection = project_camera_points(point[None], model, metadata['source_id'], model.image_size)
            u = projection['uv'][0][0]
            sides = (cloud.uv[:, 0] <= u, cloud.uv[:, 0] > u)
            distances = [float(np.linalg.norm(cloud.xyz_mm[side]-point, axis=1).min()) if side.any() else None
                         for side in sides]
        observations.append(dict(snapshot=str(path), image_sha256=snapshot['image']['sha256'],
            controls=metadata['controls'], metadata=metadata, count=len(cloud.uv),
            result=result, projection=projection, nearest_native_flank_distances_mm=distances))
    report = dict(calibration_path=str(args.calibration), calibration_id=calibration_id,
        optical_intrinsics_id=optical_id, source_id=model.source_id,
        diagnostic_roi_px=args.roi, detector=asdict(detector), noise=asdict(noise),
        settings=asdict(settings), fixture_specific_roi_hint=True,
        reprocessed_offline=True, accuracy_qualified=False, observations=observations)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(calibration_id=calibration_id, optical_intrinsics_id=optical_id,
                         frames=len(observations), found=sum(row['result']['found'] for row in observations),
                         output=str(args.output))))


if __name__ == '__main__':
    main()
