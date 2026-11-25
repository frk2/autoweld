import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/faraz/Code/Weld-seam-finder/app')
from weldfinder.seam import find_weld_seam
root=Path('/home/faraz/Code/autoweld/evidence/reflective-tubes-20261030')
c=json.loads((root/'baseline-camera.json').read_text())['laser']['cloud']
p=np.asarray(c['xyz_mm']);uv=np.asarray(c['uv']);st=np.asarray(c['strength']);wi=np.asarray(c['width_px']);order=np.argsort(uv[:,0])
p,uv,st,wi=(x[order] for x in (p,uv,st,wi))
def summary(mask):
 a=p[mask];b=uv[mask]
 return {'count':int(mask.sum()),'uv_min':b.min(axis=0).tolist(),'uv_max':b.max(axis=0).tolist(),'xyz_min_mm':a.min(axis=0).tolist(),'xyz_max_mm':a.max(axis=0).tolist(),'strength_p10_50_90':np.percentile(st[mask],[10,50,90]).tolist(),'width_p10_50_90_px':np.percentile(wi[mask],[10,50,90]).tolist()}
breaks=(np.diff(uv[:,0])>3)|(np.abs(np.diff(uv[:,1]))>8)
starts=np.r_[0,np.flatnonzero(breaks)+1];ends=np.r_[starts[1:],len(uv)]
components=[]
for a,b in zip(starts,ends):
 if b-a<3:continue
 mask=np.zeros(len(p),bool);mask[a:b]=True
 components.append({'first_native_index_sorted':int(a),'last_native_index_sorted':int(b-1),**summary(mask)})
cases={
 'full':np.ones(len(p),bool),
 'exclude_upper_trace_by_image_space':uv[:,1]>150,
 'joint_roi_300_800_180_330':(uv[:,0]>=300)&(uv[:,0]<=800)&(uv[:,1]>=180)&(uv[:,1]<=330),
 'joint_roi_400_750_180_330':(uv[:,0]>=400)&(uv[:,0]<=750)&(uv[:,1]>=180)&(uv[:,1]<=330),
 'joint_roi_300_1000_180_330':(uv[:,0]>=300)&(uv[:,0]<=1000)&(uv[:,1]>=180)&(uv[:,1]<=330),
}
results={}
for name,mask in cases.items():
 candidate=copy.deepcopy(c)
 candidate.update(xyz_mm=p[mask].tolist(),uv=uv[mask].tolist(),strength=st[mask].tolist(),width_px=wi[mask].tolist(),count=int(mask.sum()))
 r=find_weld_seam(candidate)
 results[name]={'selection_unmeasured_diagnostic_only':True,'count':int(mask.sum()),'status':r['status'],'found':r['found'],'quality':r['quality'],'seam_point_mm':r.get('seam_point_mm')}
def fitline(a):
 center=a.mean(axis=0);_,_,v=np.linalg.svd(a-center,full_matrices=False);d=v[0]
 if np.dot(d,a[-1]-a[0])<0:d=-d
 along=(a-center)@d
 rms=np.sqrt(np.mean(np.sum((a-center-along[:,None]*d)**2,axis=1)))
 return center,d,float(rms),along
fits=[]
for left_range,right_range in [((425,474),(542,650)),((435,474),(542,650)),((435,474),(542,711)),((457,474),(542,650))]:
 masks=[(uv[:,0]>=r[0])&(uv[:,0]<=r[1])&(uv[:,1]>180)&(uv[:,1]<280) for r in (left_range,right_range)]
 if min(m.sum() for m in masks)<3:continue
 a,b=[fitline(p[m]) for m in masks];t=np.linalg.lstsq(np.column_stack([a[1],-b[1]]),b[0]-a[0],rcond=None)[0]
 limits=np.array([a[0]+t[0]*a[1],b[0]+t[1]*b[1]]);point=limits.mean(axis=0)
 nearest=[float(np.linalg.norm(p[m]-point,axis=1).min()) for m in masks]
 fits.append({'unmeasured_line_extrapolation_not_a_seam':True,'left_u_range':left_range,'right_u_range':right_range,'counts':[int(m.sum()) for m in masks],'rms_mm':[a[2],b[2]],'line_angle_deg':float(np.rad2deg(np.arccos(np.clip(np.dot(a[1],b[1]),-1,1)))),'closest_lines_gap_mm':float(np.linalg.norm(limits[1]-limits[0])),'predicted_intersection_camera_mm':point.tolist(),'nearest_measured_return_each_side_mm':nearest,'support_span_mm':[float(np.ptp(a[3])),float(np.ptp(b[3]))]})
i=np.flatnonzero(uv[:,0]==474)[0];j=np.flatnonzero(uv[:,0]==542)[0]
report={'schema':'offline-reflective-profile-review.v1','native_sequence':c['sequence'],'native_generation':c['generation'],'source_file_sha256':hashlib.sha256((root/'baseline-camera.json').read_bytes()).hexdigest(),'current_observed_coverage':False,'raw_point_count':len(p),'upper_trace':summary(uv[:,1]<50),'components':components,'roi_diagnostic_results':results,'adjacent_joint_gap':{'left_uv':uv[i].tolist(),'right_uv':uv[j].tolist(),'scan_gap_px':float(uv[j,0]-uv[i,0]),'measured_endpoint_distance_mm':float(np.linalg.norm(p[j]-p[i]))},'unmeasured_flank_extrapolations':fits,'conclusion':'The baseline data cannot establish a measured root at the intended coped junction. A large native return gap separates its two candidate surfaces; a remote supported step is a different feature. Image-space cropping can diagnose upper reflections but must not be treated as measured seam evidence or used to bridge missing returns.'}
(root/'offline-profile-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'upper_trace_count':report['upper_trace']['count'],'joint_gap':report['adjacent_joint_gap'],'crops':{k:{'count':v['count'],'status':v['status'],'reason':v['quality'].get('reason')} for k,v in results.items()},'flank_extrapolations':fits},indent=2))
