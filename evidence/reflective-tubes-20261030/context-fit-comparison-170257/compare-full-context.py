import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import ast,copy,hashlib,json,sys,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/faraz/Code/Weld-seam-finder/app')
from weldfinder.coped_joint import fit_coped_joint,select_coped_surface_returns
project=Path('/home/faraz/Code/autoweld');session=project/'runtime/agents-live/weld-paths/20261030-170257-ba242ade';out=project/'evidence/reflective-tubes-20261030/context-fit-comparison-170257';out.mkdir(exist_ok=True)
frozen_manifest=out/'input-observations-frozen.json';manifest_bytes=frozen_manifest.read_bytes()if frozen_manifest.exists()else(session/'observations.json').read_bytes();manifest=json.loads(manifest_bytes);frozen_manifest.write_bytes(manifest_bytes)
rows=[];reference=None;seed=None;groups=[]
for entry in manifest['native_files']:
 path=project/entry['path'];data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256'],path
 record=json.loads(data);m=record['measurement'];c=m['cloud']
 if entry['observation_index']==1:
  reference=np.array(m['base_from_camera']);seed=np.array(record['result']['seam_point_mm']);continue
 if m.get('coping_phase')!='scout':continue
 full=c.get('full_profile');assert full is not None,path
 for key in ('sequence','generation','source_id','acquisition_session_id','calibration_id','optical_intrinsics_id'):
  assert full.get(key)==c.get(key),(str(path),key,full.get(key),c.get(key))
 assert full.get('not_seam_measurement') is True
 goal=np.array(m['requested_base_from_tcp'])
 gid=next((i for i,g in enumerate(groups)if np.max(np.abs(g-goal))<1e-8),None)
 if gid is None:gid=len(groups);groups.append(goal)
 rows.append({'entry':entry,'measurement':m,'group':gid,'full':full})
assert reference is not None and seed is not None
live=project/'runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json';live_id=hashlib.sha256(json.dumps(json.loads(live.read_text()),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert all(r['measurement']['cloud']['calibration_id']==live_id for r in rows)
def profile(row,context):
 m=row['measurement'];raw=row['full']if context else m['cloud'];selection=select_coped_surface_returns(raw);idx=np.array(selection['native_point_indices'],int);xyz=np.array(raw['xyz_mm'],float)
 if context=='anchored':
  uv=np.array(raw['uv']);roi=m['seam_search_roi']['roi'];w,h=raw['image_size'];inside=(uv[:,0]>=roi[0]*w)&(uv[:,0]<=roi[2]*w)&(uv[:,1]>=roi[1]*h)&(uv[:,1]<=roi[3]*h);anchored=[]
  for run in selection['diagnostics']['runs']:
   a,b=run['native_start_index'],run['native_stop_index_exclusive']
   if run['selected'] and inside[a:b].any():anchored.extend(range(a,b))
  idx=np.asarray(anchored,int)
 if len(idx)<16:return None
 xyz=xyz[idx];actual=np.array(m['base_from_camera']);point=np.array(m['seam_search_roi']['point_base_m']);hint=actual[:3,:3].T@(point-actual[:3,3])*1000
 if abs(np.linalg.norm(hint)-150)>2:return None
 split=int(np.argmin(np.linalg.norm(xyz-hint,axis=1)));balance=min(split,len(xyz)-split)
 if balance<8:return None
 frozen=np.linalg.inv(reference)@actual;registered=xyz@frozen[:3,:3].T+frozen[:3,3]*1000
 return {'xyz_mm':registered.tolist(),'surface_labels':(np.arange(len(xyz))>=split).astype(int).tolist(),'coordinate_frame':'camera','units':'mm','profile_id':row['entry']['path'],'provenance':{'native_path':row['entry']['path'],'native_sha256':row['entry']['sha256'],'physical_scout_index':row['group'],'fresh_current_run':True,'context':'ROI-anchored_full_connected_runs'if context=='anchored'else'full_profile'if context else'ROI','raw_native_point_count':len(raw['xyz_mm']),'native_point_count':len(xyz),'native_point_indices':idx.tolist(),'balanced_surface_point_count':balance,'source_id':raw['source_id'],'sequence':raw['sequence'],'generation':raw['generation'],'calibration_id':raw['calibration_id'],'optical_intrinsics_id':raw['optical_intrinsics_id'],'acquisition_session_id':raw['acquisition_session_id'],'base_from_camera':actual.tolist(),'frozen_seed_base_from_camera':reference.tolist(),'split_hint_camera_mm':hint.tolist(),'split_hint_is_measured_seam':False,'membership_basis':'predicted hint initializes membership only','surface_return_selection':selection['diagnostics'],'geometry_role':'actual surfaces for fit diagnostics only; no seam measurement or observed angular coverage'}}
accepted=[r for r in rows if r['entry'].get('disposition')=='coping_surface_profile']
narrow=[p for r in accepted if(p:=profile(r,False))is not None]
full_same=[p for r in accepted if(p:=profile(r,True))is not None]
best={}
for r in rows:
 p=profile(r,True)
 if p is not None and(r['group']not in best or p['provenance']['balanced_surface_point_count']>best[r['group']]['provenance']['balanced_surface_point_count']):best[r['group']]=p
full_balanced=[best[k]for k in sorted(best)]
anchored_same=[p for r in accepted if(p:=profile(r,'anchored'))is not None]
anchored_best={}
for r in rows:
 p=profile(r,'anchored')
 if p is not None and(r['group']not in anchored_best or p['provenance']['balanced_surface_point_count']>anchored_best[r['group']]['provenance']['balanced_surface_point_count']):anchored_best[r['group']]=p
datasets={'narrow_accepted':narrow,'full_same_acquisitions':full_same,'full_best_balanced_all_views':full_balanced,'full_ROI_anchored_same_acquisitions':anchored_same,'full_ROI_anchored_best_balanced':list(anchored_best.values())}
source=ast.parse((project/'autoweld/coping_service.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name=='CopedJointMixin');gate=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_coping_validate_fresh');namespace={'np':np};exec(compile(ast.Module(body=[gate],type_ignores=[]),'<actual-read-only-fresh-gate>','exec'),namespace)
audit={'schema':'offline-full-context-coped-fit-comparison.v1','session_id':session.name,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'calibration_file':str(live.relative_to(project)),'calibration_id':live_id,'calibration_used_for_reconstruction':False,'registration':'provided native XYZ transformed by each immutable actual base_from_camera into obs1 frozen measured camera frame','seed_camera_mm':seed.tolist(),'radius_mm':[12.7,12.7],'angle_deg':90,'locator_profile_excluded':True,'diagnostic_only':True,'no_angular_coverage_or_seam_measurement':True,'selection':'Existing ordered UV connected-run rule only (>=12points, <=3px scan gap, <=8px jump); max balanced per physicalview, no residual-driven removal','datasets':{}}
for name,profiles in datasets.items():
 (out/f'{name}-input.json').write_text(json.dumps({'profiles':profiles,'seed_camera_mm':seed.tolist()})+'\n')
 print(name,'profiles',len(profiles),'points',sum(len(p['xyz_mm'])for p in profiles),'ids',[Path(p['profile_id']).name for p in profiles],flush=True)
 if len(profiles)<3:
  audit['datasets'][name]={'status':'insufficient_profiles'};continue
 started=time.monotonic();fitted=fit_coped_joint(profiles,coordinate_frame='camera',radii_mm=(12.7,12.7),joint_angle_deg=90.,seam_seed_mm=seed.tolist())
 (out/f'{name}-fit.json').write_text(json.dumps(fitted)+'\n')
 summary={'status':fitted['status'],'profile_count':len(profiles),'point_count':sum(len(p['xyz_mm'])for p in profiles),'profiles':[{'observation':Path(p['profile_id']).name,'physical_scout_index':p['provenance']['physical_scout_index'],'balanced_count':p['provenance']['balanced_surface_point_count'],'point_count':len(p['xyz_mm'])}for p in profiles],'quality':fitted.get('quality'),'model_comparison':fitted.get('model_comparison'),'fresh_surface_gate':namespace['_coping_validate_fresh'](SimpleNamespace(_coping_profiles=profiles),fitted),'duration_s':time.monotonic()-started}
 audit['datasets'][name]=summary
 (out/'comparison.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(summary,indent=2),flush=True)
