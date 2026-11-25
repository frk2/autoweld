from pathlib import Path
source=Path('/tmp/compare-full-context.py').read_text().split('\naudit=')[0]
exec(compile(source,'<frozen-full-context-setup>','exec'),globals())
def hybrid(row):
 m=row['measurement'];roi=m['cloud'];full=row['full'];roi_select=select_coped_surface_returns(roi);fi=select_coped_surface_returns(full)
 ri=np.asarray(roi_select['native_point_indices'],int)
 if len(ri)<8:return None
 ruv=np.asarray(roi['uv'],float);fuv=np.asarray(full['uv'],float);lookup={int(round(u)):v for u,v in ruv[ri]}
 selected=[];matched_runs=[]
 for run in fi['diagnostics']['runs']:
  a,b=run['native_start_index'],run['native_stop_index_exclusive']
  matched=any(int(round(u))in lookup and abs(v-lookup[int(round(u))])<=2 for u,v in fuv[a:b])
  if run['selected']and matched:selected.extend(range(a,b));matched_runs.append([a,b])
 fi_indices=np.asarray(selected,int);rect=m['seam_search_roi']['roi'];w=roi['image_size'][0]
 if len(fi_indices):fi_indices=fi_indices[(fuv[fi_indices,0]<rect[0]*w)|(fuv[fi_indices,0]>rect[2]*w)]
 rxyz=np.asarray(roi['xyz_mm']);fxyz=np.asarray(full['xyz_mm']);xyz=np.concatenate([rxyz[ri],fxyz[fi_indices]])
 pixels=np.concatenate([ruv[ri],fuv[fi_indices]]);sources=['roi']*len(ri)+['full_context']*len(fi_indices);native_indices=np.r_[ri,fi_indices];order=np.argsort(pixels[:,0],kind='stable');xyz,pixels,native_indices=xyz[order],pixels[order],native_indices[order];sources=[sources[i]for i in order]
 if len(xyz)<16 or np.any(np.diff(pixels[:,0])<=0):return None
 actual=np.asarray(m['base_from_camera']);point=np.asarray(m['seam_search_roi']['point_base_m']);hint=actual[:3,:3].T@(point-actual[:3,3])*1000
 if abs(np.linalg.norm(hint)-150)>2:return None
 split=int(np.argmin(np.linalg.norm(xyz-hint,axis=1)));balance=min(split,len(xyz)-split)
 if balance<8:return None
 frozen=np.linalg.inv(reference)@actual;registered=xyz@frozen[:3,:3].T+frozen[:3,3]*1000
 return {'xyz_mm':registered.tolist(),'surface_labels':(np.arange(len(xyz))>=split).astype(int).tolist(),'coordinate_frame':'camera','units':'mm','profile_id':row['entry']['path'],'provenance':{'native_path':row['entry']['path'],'native_sha256':row['entry']['sha256'],'physical_scout_index':row['group'],'fresh_current_run':True,'context':'ROI-native points inside ROI columns + same-acquisition full connected context outside ROI columns, anchored by actual matching UV','native_point_count':len(xyz),'balanced_surface_point_count':balance,'native_point_sources':sources,'native_point_indices':native_indices.tolist(),'full_component_native_ranges':matched_runs,'roi_native_point_indices':ri.tolist(),'full_context_native_point_indices':fi_indices.tolist(),'cloud_identity':{k:roi[k]for k in('source_id','sequence','generation','calibration_id','acquisition_session_id','optical_intrinsics_id')},'base_from_camera':actual.tolist(),'frozen_seed_base_from_camera':reference.tolist(),'split_hint_camera_mm':hint.tolist(),'split_hint_is_measured_seam':False,'UV_match_max_abs_v_px':2,'selection_scope':'image-space support only; no model residual masks or coordinate changes'}}
fixed=[p for r in accepted if(p:=hybrid(r))is not None];best={}
for r in rows:
 p=hybrid(r)
 if p is not None and(r['group']not in best or p['provenance']['balanced_surface_point_count']>best[r['group']]['provenance']['balanced_surface_point_count']):best[r['group']]=p
report=json.loads((out/'comparison.json').read_text())
for name,profiles in {'hybrid_ROI_priority_same_acquisitions':fixed,'hybrid_ROI_priority_best_balanced':list(best.values())}.items():
 (out/f'{name}-input.json').write_text(json.dumps({'profiles':profiles,'seed_camera_mm':seed.tolist()})+'\n');print(name,len(profiles),sum(len(p['xyz_mm'])for p in profiles),flush=True)
 started=time.monotonic();fitted=fit_coped_joint(profiles,coordinate_frame='camera',radii_mm=(12.7,12.7),joint_angle_deg=90.,seam_seed_mm=seed.tolist());(out/f'{name}-fit.json').write_text(json.dumps(fitted)+'\n')
 gate=namespace['_coping_validate_fresh'](SimpleNamespace(_coping_profiles=profiles),fitted)
 result={'status':fitted['status'],'profile_count':len(profiles),'point_count':sum(len(p['xyz_mm'])for p in profiles),'profiles':[{'observation':Path(p['profile_id']).name,'physical_scout_index':p['provenance']['physical_scout_index'],'balanced_count':p['provenance']['balanced_surface_point_count'],'point_count':len(p['xyz_mm'])}for p in profiles],'quality':fitted.get('quality'),'model_comparison':fitted.get('model_comparison'),'fresh_surface_gate':gate,'duration_s':time.monotonic()-started}
 report['datasets'][name]=result;(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
