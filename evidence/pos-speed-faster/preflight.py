import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import cadquery as cq
from autoweld.app import tool_model
from autoweld.motion import MotionPlanner, MotionLimits
from autoweld.tracking import load_tracks, plan_tracking
root=Path('evidence/pos-speed-faster')
s=json.load(open('runtime/live/status.json'))
(root/'before-status.json').write_text(json.dumps(s,indent=2))
manifest=json.load(open('assets/stage3/manifest.json'))
model=tool_model(); planner=MotionPlanner(model)
planner.limits=replace(MotionLimits(**s['motion_limits']),joint_velocity=.15)
bounds=planner.model.jnt_range[:6].copy(); rest=np.array(s['rest_rad'])
bounds[:,0]=np.minimum(bounds[:,0],rest); bounds[:,1]=np.maximum(bounds[:,1],rest)
records=[]
try:
 for length in (.06,.04):
  edges=[]
  for entry in manifest:
   c=np.array(entry['center_m']); d=np.array(entry['line_direction'])
   edges.append(cq.Edge.makeLine(cq.Vector(*((c-length/2*d)*1000)),cq.Vector(*((c+length/2*d)*1000))))
  dest=root/'assets'/f'lines-{length*1000:g}mm.step'
  cq.exporters.export(cq.Compound.makeCompound(edges),str(dest))
  for feed in (.01,.02):
   for i,curve in enumerate(load_tracks(dest)):
    seed=np.array(manifest[i]['seed_q']); seed[6]=json.load(open('runtime/live/rest.json'))['gripper_normalized']
    record={'length_mm':length*1000,'feed_mm_s':feed*1000,'index':i}
    try:
     plan=plan_tracking(planner,seed,curve,feed,bounds=bounds)
     q=np.array([plan.motion.path.spline(u)[:6] for u in np.linspace(0,1,1001)])
     excursion=max(np.max(abs(q-rest)),np.max(abs(q-seed[:6])))
     if excursion>=.49: raise ValueError(f'Automatic excursion {excursion:.4f} rad exceeds bound')
     record.update(ok=True,max_excursion_rad=float(excursion),plan=plan.motion.report)
     print(length,feed,i+1,'OK lead',plan.lead,'duration',plan.motion.path.duration,flush=True)
    except Exception as exc:
     record.update(ok=False,error=str(exc)); print(length,feed,i+1,'REJECT',str(exc)[:120],flush=True)
    records.append(record)
finally:
 Path(model).unlink()
 (root/'preflight.json').write_text(json.dumps(records,indent=2))
