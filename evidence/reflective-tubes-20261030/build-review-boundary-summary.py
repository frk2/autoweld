import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import hashlib,json
from pathlib import Path
import numpy as np
project=Path('/home/faraz/Code/autoweld');session=project/'runtime/agents-live/weld-paths/20261030-170257-ba242ade';out=project/'evidence/reflective-tubes-20261030';saved=json.loads((session/'path.json').read_text());manifest=json.loads((session/'observations.json').read_text())
profiles={Path(p['provenance']['native_path']).name:p for p in saved.get('coping_profiles',[])}
groups=[];seed=None

def support(cloud):
 xyz=np.asarray(cloud.get('xyz_mm',[]),float);uv=np.asarray(cloud.get('uv',[]),float)
 if xyz.size==0:return dict(point_count=0,uv_bounds_px=None,depth_bounds_mm=None,connected_run_count=0,selected_connected_run_count=0,selected_connected_point_count=0)
 assert xyz.shape==(len(uv),3)and uv.shape==(len(xyz),2)
 ends=np.r_[0,np.flatnonzero((np.diff(uv[:,0])>3)|(abs(np.diff(uv[:,1]))>8))+1,len(uv)]
 sizes=np.diff(ends)
 return dict(point_count=len(xyz),uv_bounds_px=[uv.min(axis=0).tolist(),uv.max(axis=0).tolist()],depth_bounds_mm=[float(xyz[:,2].min()),float(xyz[:,2].max())],connected_run_count=len(sizes),selected_connected_run_count=int((sizes>=12).sum()),selected_connected_point_count=int(sizes[sizes>=12].sum()))

for entry in manifest['native_files']:
 p=project/entry['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256'];obs=json.loads(raw);m,c,r=obs['measurement'],obs['measurement']['cloud'],obs['result']
 if m.get('coping_phase')=='seed':seed={'native_filename':p.name,'native_path':entry['path'],'sha256':entry['sha256'],'role':'initial_locator_only; excluded from 150mm model fit and angular coverage','found':r.get('found'),'seam_point_camera_mm':r.get('seam_point_mm'),'quality':r.get('quality'),'base_from_camera':m.get('base_from_camera'),'same_exposure_native_photo_path':None};continue
 assert m.get('coping_phase')=='scout'
 goal=np.asarray(m['requested_base_from_tcp']);group=next((g for g in groups if np.max(abs(np.asarray(g['requested_base_from_tcp'])-goal))<1e-10),None)
 if group is None:group={'physical_scout_index':len(groups),'requested_base_from_tcp':goal.tolist(),'frames':[]};groups.append(group)
 q=r.get('quality')or{};frame={'observation_index':entry['observation_index'],'native_filename':p.name,'native_path':entry['path'],'sha256':entry['sha256'],'disposition':entry.get('disposition'),'acquisition_sequence':c['sequence'],'generation':c['generation'],'source_id':c['source_id'],'calibration_id':c['calibration_id'],'optical_intrinsics_id':c['optical_intrinsics_id'],'acquisition_session_id':c['acquisition_session_id'],'status':r.get('status'),'found':r.get('found')is True,'quality_good':q.get('good')is True,'reason':q.get('reason'),'feature_kind':q.get('feature_kind'),'flank_rms_mm':q.get('flank_rms_mm'),'roi_profile_support':support(c),'full_context_support':support(c['full_profile']),'actual_controls':c['controls'],'native_rejected':c.get('rejected'),'base_from_camera':m['base_from_camera'],'same_exposure_native_photo_path':None,'photo_status':'No same-exposure native photo retained for this production station. Use exact recorded UV/XYZ plots; any other-pose images must be separately labeled.','selected_for_production_surface_fit':p.name in profiles,'ROI_is_projection_not_measurement':(m.get('seam_search_roi')or{}).get('projection_only')is True}
 if p.name in profiles:
  prov=profiles[p.name]['provenance'];frame['production_fit_support']={'point_count':len(profiles[p.name]['xyz_mm']),'balanced_surface_point_count':prov.get('balanced_surface_point_count'),'surface_label_counts':[int(np.sum(np.asarray(profiles[p.name]['surface_labels'])==k))for k in(0,1)],'native_point_indices':prov.get('native_point_indices'),'selection_rule':prov.get('surface_return_selection',{}).get('parameters'),'role':'actual surface fit inputs only; not measured seam point or angular coverage'}
 group['frames'].append(frame)
assert seed is not None and len(groups)==9
schedule=[(0,0),(3,5),(-3,-5),(5,8),(-5,-8),(7,8),(-7,-8),(9,10),(-9,-10)]
previous={'native_filename':seed['native_filename'],'status':'found','found':True}
for group,(offset,tilt)in zip(groups,schedule):
 frames=group['frames'];group.update(programmed_scout_offset_mm=offset,programmed_camera_view_tilt_deg=tilt,tilt_is_not_observed_seam_orbit_angle=True,frame_count=len(frames),local_feature_found_count=sum(f['found']and f['quality_good']for f in frames),invalid_count=sum(f['status']=='invalid'for f in frames),ambiguous_count=sum(f['status']=='ambiguous'for f in frames))
 best=max(frames,key=lambda f:(f['roi_profile_support']['point_count'],-f['observation_index']))
 group['representative_max_ROI_count']={'native_filename':best['native_filename'],'observation_index':best['observation_index'],'ROI_point_count':best['roi_profile_support']['point_count'],'full_context_point_count':best['full_context_support']['point_count'],'status':best['status'],'reason':best['reason'],'selection_is_not_quality_admission':True,'same_exposure_native_photo_path':None}
 group['first_invalid_or_ambiguous_frame']=next(({'native_filename':f['native_filename'],'status':f['status'],'reason':f['reason']}for f in frames if f['status']not in('found','no_seam')),None)
 group['entry_status_transition']={'previous_native_filename':previous['native_filename'],'previous_status':previous['status'],'current_native_filename':frames[0]['native_filename'],'current_status':frames[0]['status'],'lost_local_feature_on_entry':bool(previous['found']and not frames[0]['found'])}
 group['first_found_to_failure_transition']=None
 for a,b in zip(frames,frames[1:]):
  if a['found']and not b['found']:
   group['first_found_to_failure_transition']={'from':a['native_filename'],'to':b['native_filename'],'to_status':b['status'],'reason':b['reason']};break
 group['production_selected_surface_profiles']=[f['native_filename']for f in frames if f['selected_for_production_surface_fit']]
 previous=frames[-1]
model=saved['joint_model'];candidate=model.get('candidate_model');assert model.get('model')is None and candidate is not None
axes=candidate['axes'];a,b=(np.asarray(axis['direction'])for axis in axes);assert np.allclose([np.linalg.norm(a),np.linalg.norm(b)],[1,1]);assert all(axis['radius_mm']==12.7 for axis in axes);assert abs(a@b)<1e-9
partial={'exists':True,'production_model_status':model['status'],'production_accepted_model_is_none':True,'candidate_model_id':candidate['model_id'],'coordinate_frame':'frozen initial measured seed-camera frame (not current live camera)','base_from_seed_camera':saved['coping_seed_base_from_camera'],'candidate_model':candidate,'fixed_radii_mm':[12.7,12.7],'outside_diameters_mm':[25.4,25.4],'crossing_angle_deg':90.,'quality':model['quality'],'branch_identity':model.get('branch_identity'),'meaning':'Measured partial surface profiles support a rejected candidate pair of infinite known-radius cylinders. Candidate axes/wireframes are predicted fit geometry, not accepted tube reconstruction, weld profile, inward torch pose, or observed seam coverage.','role_ambiguity':'axis_role_ambiguous=true describes indistinguishable physical branch/main roles from equal-diameter geometry. The user explicitly selected left branch/cop ing_axis0; this label is not an independent geometric identification or fit qualification.','fit_failure':'Nominal training robust RMS0.4205mm exceeds unchanged0.4mm fit gate; all-point RMS0.6478mm and several current whole-profile RMS values also exceed0.5mm. Nominal held-out robust RMS0.4407mm is below0.5mm, but does not override failed training/current-data gates.','surface_support_angular_spans_are_not_seam_coverage':True,'numerical_axis_alternatives':model['quality'].get('distinct_alternatives'),'accepted_inward_torch_orientation':False,'observed_seam_angular_span_deg':saved['coping_coverage']['observed_arc_deg']}
report={'schema':'autoweld.reflective-review-boundaries.v1','session_id':session.name,'path_manifest':str((session/'path.json').relative_to(project)),'path_manifest_sha256':hashlib.sha256((session/'path.json').read_bytes()).hexdigest(),'native_observation_manifest_sha256':hashlib.sha256((session/'observations.json').read_bytes()).hexdigest(),'scope':'Read-only summary of already completed production request; no motion, acquisition, calibration or detector rerun','initial_locator':seed,'scout_schedule_not_a_seam_orbit':True,'scouts':groups,'partial_model':partial,'coverage':saved['coping_coverage'],'photo_pairing_warning':'Production scout native UV/XYZ profiles have exact source identities and measured camera poses, but no matched photos retained. Manual ±5deg photos were acquired at different full camera poses and must never be labeled as these production scout images. No matched photo exists for production ±10deg views.','local_feature_found_counts_by_scout':[g['local_feature_found_count']for g in groups],'model_profile_count':len(profiles),'total_native_observations':len(manifest['native_files'])}
(out/'review-boundary-summary.json').write_text(json.dumps(report,indent=2)+'\n')
for g in groups:
 print(g['physical_scout_index'],f"offset{g['programmed_scout_offset_mm']:+}mm tilt{g['programmed_camera_view_tilt_deg']:+}deg",'found',g['local_feature_found_count'],'/6','best',g['representative_max_ROI_count'],'fit',g['production_selected_surface_profiles'])
print('candidate model',partial['exists'],partial['production_model_status'],'radii',partial['fixed_radii_mm'],'span',partial['observed_seam_angular_span_deg'])
