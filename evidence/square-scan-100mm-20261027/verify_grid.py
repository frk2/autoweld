"""Read-only verification of the requested square and all durable native views."""
import collections
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

root = Path('runtime/agents-live/laser-scans')
summary = json.loads((root/'latest.json').read_text())
metadata = summary['metadata']
session = root/summary['session_id']
rows = [json.loads(line) for line in (session/'cells.jsonl').read_text().splitlines()]
base = np.asarray(metadata['baseline_pose']) @ np.asarray(metadata['handeye']['tcp_from_camera'])
view = np.asarray(metadata['capture_rotation'])
counts = collections.Counter(row['status'] for row in rows)
expected = [(x/1000., y/1000.) for y in range(-50,51) for x in (range(-50,51) if y%2==0 else range(50,-51,-1))]
xy_errors, rotations, ages, quiet, depths = [],[],[],[],[]
acquisitions, indices, snapshots, point_count = set(),set(),0,0
with Path('evidence/square-scan-100mm-20261027/coverage.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['cell_index','x_mm','y_mm','status','native_points','accepted_points','xy_error_mm','rotation_error_rad','reason'])
 for row in rows:
  index=row['cell_index']; assert index not in indices; indices.add(index)
  assert np.allclose(row['offset_xy_m'],expected[index],atol=1e-12,rtol=0)
  key=tuple(row['acquisition_identity']);assert key not in acquisitions;acquisitions.add(key)
  assert row['source_id']==metadata['identity'][0] and row['optical_intrinsics_id']==metadata['identity'][1] and row['calibration_id']==metadata['identity'][2]
  pose=np.asarray(row['base_from_tcp']); h=np.asarray(row['tcp_from_camera'])
  assert np.allclose(h,metadata['handeye']['tcp_from_camera'],atol=1e-12,rtol=0)
  actual=base[:3,:3].T@((pose@h)[:3,3]-base[:3,3])
  err=float(np.linalg.norm(actual[:2]-row['offset_xy_m'])*1000);xy_errors.append(err)
  angle=float(Rotation.from_matrix(pose[:3,:3]@view.T).magnitude());rotations.append(angle)
  feedback=row['feedback'];assert feedback['fresh'] and feedback['stationary']
  age=feedback['measured_at']-row['read_started'];ages.append(age)
  dwell=row['read_started']-feedback['stable_since'];quiet.append(dwell)
  assert 0<=age<=1 and dwell>=.4 and err<=.500001 and angle<=.00500001
  path=Path(row['raw_snapshot_path']);assert path.is_file()
  with np.load(path,allow_pickle=False) as snap:
   assert np.isfinite(snap['xyz_camera_m']).all() and np.isfinite(snap['xyz_world_m']).all()
   n=len(snap['xyz_world_m']); assert n==row['count'];point_count+=n;snapshots+=1
   if n:
    transformed=snap['xyz_camera_m'][snap['selected_indices']]@(pose@h)[:3,:3].T+(pose@h)[:3,3]
    assert np.allclose(transformed,snap['xyz_world_m'],atol=1e-12,rtol=0)
    depth=row['quality']['median_depth_mm'];depths.append(depth);assert 139<=depth<=141
   writer.writerow([index,*np.asarray(row['offset_xy_m'])*1000,row['status'],len(snap['xyz_camera_m']),n,err,angle,row.get('reason','')])
assert indices==set(range(len(rows)))
report={'session_id':summary['session_id'],'status':summary['status'],'requested_size_mm':[100,100],
 'nominal_spacing_mm':1,'requested_cells':10201,'visited':len(rows),'complete':len(rows)==10201,
 'outcomes':dict(counts),'full_native_snapshots':snapshots,'points':point_count,
 'max_measured_xy_error_mm':max(xy_errors,default=None),'max_rotation_error_rad':max(rotations,default=None),
 'min_quiet_before_acquisition_s':min(quiet,default=None),'max_sdk_after_acquisition_s':max(ages,default=None),
 'accepted_median_depth_range_mm':[min(depths),max(depths)] if depths else None,
 'baseline_pose':metadata['baseline_pose'],'fixed_optical_roll_deg':metadata.get('optical_roll_deg'),
 'acquisition_sessions':sorted({r['acquisition_session_id'] for r in rows})}
if summary['status']=='complete': assert report['complete'] and point_count==summary['count']
Path('evidence/square-scan-100mm-20261027/grid-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
