"""Offline retained-image/TCP hand-eye review. Does not install a calibration."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from autoweld.calibration import solve_handeye

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runtime/agents-live/evidence/robocam-1792796400030413285/checkpoint.json'
MODEL = ROOT / 'runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json'
OUT = Path(__file__).resolve().parent
raw = SOURCE.read_bytes()
data = json.loads(raw)
model_raw = MODEL.read_bytes()
model = json.loads(model_raw)
identity = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
k, d = np.array(model['camera_matrix']), np.array(model['distortion'])
b = data['board']
obj = np.zeros((b['columns'] * b['rows'], 3), np.float32)
obj[:, :2] = np.mgrid[:b['columns'], :b['rows']].T.reshape(-1, 2) * b['square_mm']


def pose(corners, flag=cv2.SOLVEPNP_ITERATIVE):
    ok, r, t = cv2.solvePnP(obj, np.asarray(corners, np.float32), k, d, flags=flag)
    if not ok:
        raise ValueError('PnP failed')
    proj = cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - np.array(corners).reshape(-1, 2)) ** 2, axis=1))))
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(r)[0]
    transform[:3, 3] = t.reshape(3) / 1000
    depth = (transform[:3, :3] @ (obj / 1000).T + transform[:3, 3:4])[2]
    if depth.min() <= 0 or rms > 1.5:
        raise ValueError('Native target pose gate failed')
    return transform, rms, float(depth.min())


def residuals(samples, x, reference=None):
    boards = np.array([np.array(s['base_from_tcp']) @ x @ np.array(s['camera_from_target_m']) for s in samples])
    positions = boards[:, :3, 3]
    rs = Rotation.from_matrix(boards[:, :3, :3])
    mean_r = rs.mean() if reference is None else Rotation.from_matrix(reference[:3, :3])
    center = positions.mean(0) if reference is None else reference[:3, 3]
    trans = np.linalg.norm(positions - center, axis=1) * 1000
    angles = np.rad2deg((mean_r.inv() * rs).magnitude())
    mean = np.eye(4)
    mean[:3, :3], mean[:3, 3] = mean_r.as_matrix(), center
    return dict(translation_rms_mm=float(np.sqrt(np.mean(trans**2))),
                translation_max_mm=float(trans.max()), rotation_rms_deg=float(np.sqrt(np.mean(angles**2))),
                rotation_max_deg=float(angles.max()), per_view_translation_mm=trans.tolist(),
                per_view_rotation_deg=angles.tolist()), mean


samples, diagnostics = [], []
for i, (corners, cap) in enumerate(zip(data['camera_views'], data['captures']['camera'], strict=True)):
    assert cap['robocam']['native_validation'] is True
    c, rms, depth = pose(corners)
    sample = dict(base_from_tcp=cap['robocam']['base_from_tcp'], camera_from_target_m=c.tolist(),
                  source_id=data['source_id'], calibration_id=identity, sequence=cap['robocam']['frame_sequence'],
                  capture_id=cap['id'], reprojection_rms_px=rms)
    samples.append(sample)
    ippe = cv2.solvePnPGeneric(obj, np.asarray(corners, np.float32), k, d, flags=cv2.SOLVEPNP_IPPE)
    branches = []
    for r, t in zip(ippe[1], ippe[2], strict=True):
        proj = cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)
        branch_rms = float(np.sqrt(np.mean(np.sum((proj-np.array(corners).reshape(-1, 2))**2, axis=1))))
        branches.append(dict(reprojection_rms_px=branch_rms, depth_mm=float(t[2, 0]),
            iterative_rotation_difference_deg=float(np.rad2deg(Rotation.from_matrix(cv2.Rodrigues(r)[0].T @ c[:3, :3]).magnitude()))))
    arr = np.array(corners).reshape(-1, 2)
    diagnostics.append(dict(index=i, capture_id=cap['id'], created_at=cap['created_at'], sequence=sample['sequence'],
        reprojection_rms_px=rms, minimum_corner_depth_mm=depth*1000,
        first_corner_px=arr[0].tolist(), last_corner_px=arr[-1].tolist(),
        x_axis_image_angle_deg=float(np.rad2deg(np.arctan2(*(arr[b['columns']-1]-arr[0])[::-1]))),
        ippe_branches=branches))

raw_samples = samples
try:
    raw_result = solve_handeye(raw_samples)
except ValueError as exc:
    raw_result = {'rejected': True, 'reason': str(exc)}
# Image-geometry rule selected before any fit: align the grid X direction to view 0.
# The compact cone cannot physically rotate this visible board direction by 180 degrees.
reference_points = np.asarray(data['camera_views'][0]).reshape(-1, 2)
reference_axis = reference_points[b['columns']-1] - reference_points[0]
reversed_indices = []
association = []
threshold = np.cos(np.deg2rad(20))
samples = []
normalized_points = []
for i, (sample, corners) in enumerate(zip(raw_samples, data['camera_views'], strict=True)):
    points = np.array(corners).reshape(-1, 2)
    axis = points[b['columns']-1] - points[0]
    cosine = float(np.dot(axis, reference_axis) / np.linalg.norm(axis) / np.linalg.norm(reference_axis))
    if cosine < -threshold:
        points = points[::-1].copy()
        reversed_indices.append(i)
    else:
        assert cosine > threshold, 'Ambiguous image grid direction'
    normalized_axis = points[b['columns']-1] - points[0]
    normalized_cosine = float(normalized_axis @ reference_axis / np.linalg.norm(normalized_axis) / np.linalg.norm(reference_axis))
    assert normalized_cosine > threshold
    association.append(dict(index=i, raw_axis_dot=cosine, normalized_axis_dot=normalized_cosine,
        centroid_px=points.mean(0).tolist(), normalized_first_corner_px=points[0].tolist(),
        normalized_last_corner_px=points[-1].tolist(),
        quadrilateral_px=points[[0, b['columns']-1, -1, (b['rows']-1)*b['columns']]].tolist()))
    normalized_points.append(points)
    c, rms, _ = pose(points)
    samples.append({**sample, 'camera_from_target_m': c.tolist(), 'reprojection_rms_px': rms})
result = solve_handeye(samples)
x = np.array(result['tcp_from_camera'])
metrics, mean_board = residuals(samples, x)
g = [np.array(s['base_from_tcp']) for s in samples]
pairs = [np.linalg.inv(g[j]) @ g[i] for i in range(len(g)) for j in range(i)]
va = np.array([Rotation.from_matrix(a[:3, :3]).as_rotvec() for a in pairs])
sv = np.linalg.svd(va, compute_uv=False)
mat = np.vstack([a[:3, :3] - np.eye(3) for a in pairs])
mt = np.linalg.svd(mat, compute_uv=False)
robot_max_pair_rotation_deg = float(max(np.rad2deg(Rotation.from_matrix(a[:3, :3]).magnitude()) for a in pairs))
assert robot_max_pair_rotation_deg < 16
# Independent nonlinear rotation fit to rotation-matrix AX=XB equations; no align_vectors.
cp = [np.array(s['camera_from_target_m']) for s in samples]
pair_b = [cp[j] @ np.linalg.inv(cp[i]) for i in range(len(cp)) for j in range(i)]
def rotation_equations(v):
    rr = Rotation.from_rotvec(v).as_matrix()
    return np.concatenate([(a[:3, :3] @ rr - rr @ bb[:3, :3]).reshape(-1) for a, bb in zip(pairs, pair_b, strict=True)])
nonlinear = least_squares(rotation_equations, np.zeros(3), xtol=1e-13, ftol=1e-13, gtol=1e-13)
nr = Rotation.from_rotvec(nonlinear.x).as_matrix()
nt = np.linalg.lstsq(mat, np.concatenate([nr @ bb[:3, 3] - a[:3, 3] for a, bb in zip(pairs, pair_b, strict=True)]), rcond=None)[0]
independent = dict(method='scipy least_squares rotation-matrix AX=XB from identity, independent translation least squares',
    success=bool(nonlinear.success), rotation_difference_deg=float(np.rad2deg(Rotation.from_matrix(nr.T @ x[:3, :3]).magnitude())),
    translation_difference_mm=float(np.linalg.norm(nt-x[:3, 3])*1000))
folds = []
for fold in range(5):
    train = [s for i, s in enumerate(samples) if i % 5 != fold]
    heldout = [s for i, s in enumerate(samples) if i % 5 == fold]
    fit = solve_handeye(train)
    xf = np.array(fit['tcp_from_camera'])
    _, reference = residuals(train, xf)
    errors, _ = residuals(heldout, xf, reference)
    folds.append(dict(fold=fold, train_views=len(train), heldout_views=len(heldout), fit=fit,
        heldout=errors, translation_difference_from_full_mm=float(np.linalg.norm(xf[:3, 3]-x[:3, 3])*1000),
        rotation_difference_from_full_deg=float(np.rad2deg(Rotation.from_matrix(xf[:3, :3].T @ x[:3, :3]).magnitude()))))

# Reversing every grid selects the other physical board origin; X should remain identical.
reversed_samples = []
for s, corners in zip(samples, normalized_points, strict=True):
    c, _, _ = pose(np.array(corners)[::-1].copy())
    reversed_samples.append({**s, 'camera_from_target_m': c.tolist()})
reversed_fit = solve_handeye(reversed_samples)
xr = np.array(reversed_fit['tcp_from_camera'])
mixed = list(samples)
mixed[12] = reversed_samples[12]
try:
    mixed_result = solve_handeye(mixed)
except ValueError as exc:
    mixed_result = {'rejected': True, 'reason': str(exc)}

report = dict(source_checkpoint=str(SOURCE.relative_to(ROOT)), source_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
    installed_intrinsics_path=str(MODEL.relative_to(ROOT)), installed_intrinsics_file_sha256=hashlib.sha256(model_raw).hexdigest(),
    intrinsic_calibration_id=identity, source_id=data['source_id'], board=b, image_size=data['image_size'],
    views=len(samples), raw_ordering_result=raw_result, normalization=dict(rule='Reverse if normalized native grid X image direction dot view-0 grid X image direction < -cos(20deg); otherwise require >cos(20deg); decided from image geometry before fitting', reversed_indices=reversed_indices, original_corners_preserved=True, association=association, robot_max_pair_rotation_deg=robot_max_pair_rotation_deg), result=result, stationary_board=metrics, base_from_target_mean=mean_board.tolist(),
    correction_from_nominal_tcp=dict(translation_mm=(x[:3, 3]*1000).tolist(),
        translation_norm_mm=float(np.linalg.norm(x[:3, 3])*1000),
        rotation_rotvec_deg=np.rad2deg(Rotation.from_matrix(x[:3, :3]).as_rotvec()).tolist(),
        rotation_angle_deg=float(np.rad2deg(Rotation.from_matrix(x[:3, :3]).magnitude()))),
    independent_solver_comparison=independent, excitation=dict(pair_count=len(pairs), rotation_singular_values_rad=sv.tolist(),
        pair_axis_rms_deg=np.rad2deg(sv/np.sqrt(len(pairs))).tolist(),
        translation_design_singular_values=mt.tolist(), translation_design_condition=float(mt[0]/mt[-1])),
    diagnostics=diagnostics, folds=folds,
    consistent_180_reversal=dict(translation_change_mm=float(np.linalg.norm(xr[:3, 3]-x[:3, 3])*1000),
        rotation_change_deg=float(np.rad2deg(Rotation.from_matrix(xr[:3, :3].T @ x[:3, :3]).magnitude()))),
    single_view_180_reversal_test=mixed_result,
    limitations=['No hardware timing is reconstructed: retained metadata has native-validation and sequence but omits read_started/capture_not_before/acquisition_session_id.',
        'TCP is nominal camera optical weld_tcp; result is a correction from that TCP, not flange_from_camera.',
        'Interleaved heldout checks use fixed intrinsics fitted from all 25 images; they test hand-eye stability, not fully independent intrinsic validation.',
        'No active calibration was installed.'])
(OUT/'candidate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
(OUT/'original_paired_samples.json').write_text(json.dumps(raw_samples, indent=2, allow_nan=False)+'\n')
(OUT/'paired_samples.json').write_text(json.dumps(samples, indent=2, allow_nan=False)+'\n')
(OUT/'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ('views','result','stationary_board','correction_from_nominal_tcp','excitation','consistent_180_reversal','single_view_180_reversal_test')}, indent=2))
print('Folds:', json.dumps([dict(fold=f['fold'], heldout=f['heldout'], translation_difference_from_full_mm=f['translation_difference_from_full_mm'], rotation_difference_from_full_deg=f['rotation_difference_from_full_deg']) for f in folds], indent=2))
