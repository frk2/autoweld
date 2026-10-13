"""Offline preflight only; this file never connects to motors."""
import json
from pathlib import Path
import numpy as np
import mujoco
import cadquery as cq
from i2rt.utils.viser_control_interface import ViserControlInterface
from autoweld.app import tool_model
from autoweld.motion import MotionPlanner,MotionLimits
from autoweld.tracking import load_tracks,plan_tracking
root=Path('evidence/long-cruise');s=json.loads(Path('runtime/live/status.json').read_text());rest=np.array(s['rest_rad']);grip=json.loads(Path('runtime/live/rest.json').read_text())['gripper_normalized']
(root/'before-status.json').write_text(json.dumps(s,indent=2))
model=tool_model();p=MotionPlanner(model);p.limits=MotionLimits(**s['motion_limits'])
checker=ViserControlInterface.__new__(ViserControlInterface);checker._model=p.model;checker._check_data=mujoco.MjData(p.model)
collision=lambda q:checker._has_self_collision(q,len(q))
seed=np.r_[rest+[0,.25,.25,.2,0,0],grip];T=p.fk(seed);d=np.array([0.,1.,0.]);length=.5
endpoints=[T[:3,3]-length/2*d,T[:3,3]+length/2*d]
edge=cq.Edge.makeLine(*[cq.Vector(*(e*1000)) for e in endpoints]);asset=root/'assets/line-500mm.step';cq.exporters.export(edge,str(asset));curve=load_tracks(asset)[0]
bounds=p.model.jnt_range[:6].copy();bounds[:,0]=np.minimum(bounds[:,0],rest);bounds[:,1]=np.maximum(bounds[:,1],rest)
records=[]
try:
 for feed in [.02,.03,.04,.05,.06,.08,.1]:
  r={'feed_mm_s':feed*1000,'length_mm':500,'seed_q':seed.tolist(),'endpoints_m':[v.tolist() for v in endpoints]}
  try:
   plan=plan_tracking(p,seed,curve,feed,bounds=bounds,collision=collision,ramp_placement='Inside edge')
   q=plan.motion.path.spline(np.linspace(0,1,1001))[:,:6]
   p.plan(np.r_[rest,grip],seed,bounds=bounds,collision=collision)
   p.plan(seed,plan.motion.path.positions[0],bounds=bounds,collision=collision)
   p.plan(plan.motion.path.positions[-1],np.r_[rest,grip],bounds=bounds,collision=collision)
   r.update(ok=True,excursion_rad=float(max(np.max(abs(q-rest)),np.max(abs(q-seed[:6])),np.max(abs(q-q[0])))),rest_excursion_rad=float(np.max(abs(q-rest))),joint_span_rad=np.ptp(q,axis=0).tolist(),plan=plan.motion.report,start_q=plan.motion.path.positions[0].tolist(),end_q=plan.motion.path.positions[-1].tolist())
   print(feed,'OK',r['excursion_rad'],'cruise',r['plan']['constant_feed_duration_s'],'peakjoint',r['plan']['planned_peaks']['joint_velocity'],flush=True)
  except Exception as e:r.update(ok=False,error=str(e));print(feed,'REJECT',str(e),flush=True)
  records.append(r)
finally:
 Path(model).unlink();(root/'preflight.json').write_text(json.dumps(records,indent=2))
