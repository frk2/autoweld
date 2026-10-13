"""Offline only: seek a reviewable 12.7 mm center-facing quarter arc."""
import itertools
import json
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np
import tyro
from i2rt.utils.viser_control_interface import ViserControlInterface
from scipy.spatial.transform import Rotation

from autoweld.app import tool_model
from autoweld.diagnostics import solve_pose
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.tracking import load_tracks, plan_tracking

ROOT=Path('evidence/stage3-welding')


def main(wrist_offset: float = 0.):
    state=json.loads(Path('runtime/live/status.json').read_text())
    assert state['closed']
    rest=np.array(state['rest_rad'])
    grip=json.loads(Path('runtime/live/rest.json').read_text())['gripper_normalized']
    model=tool_model()
    p=MotionPlanner(model)
    p.limits=MotionLimits(**state['motion_limits'])
    checker=ViserControlInterface.__new__(ViserControlInterface)
    checker._model=p.model
    checker._check_data=mujoco.MjData(p.model)
    def collision(q):
        return checker._has_self_collision(q,len(q))
    bounds=p.model.jnt_range[:6].copy()
    bounds[:,0]=np.minimum(bounds[:,0],rest)
    bounds[:,1]=np.maximum(bounds[:,1],rest)
    records=[]
    best=None
    try:
        seeds=list(itertools.product((.15,.3,.5,.8),(.25,.5,.8,1.1),(0.,.35,-.35)))
        for j2,j3,j4 in seeds:
            mid=np.r_[rest+[0,j2,j3,j4,wrist_offset,0],grip]
            T=p.fk(mid)
            center=T[:3,3]-.0127*T[:3,2]
            points=[cq.Vector(*((center+.0127*Rotation.from_rotvec(T[:3,0]*a).apply(T[:3,2]))*1000))
                    for a in (-np.pi/4,0,np.pi/4)]
            asset=ROOT/'assets/review-arc-search.step'
            cq.exporters.export(cq.Edge.makeThreePointArc(*points),str(asset))
            curve=load_tracks(asset)[0]
            item={'mid_q':mid.tolist()}
            try:
                q=p._configuration(mid)
                for f in np.linspace(.5,0,11):
                    q=solve_pose(p.kin,curve.pose(f,T[:3,:3],'Center facing'),p.site_name,q,
                                 stage='arc_midpoint_search',path_fraction=f)
                start_q=np.r_[q[:6],grip]
                qs=[]
                for f in np.linspace(0,1,41):
                    q=solve_pose(p.kin,curve.pose(f,T[:3,:3],'Center facing'),p.site_name,q,
                                 stage='arc_quarter_search',path_fraction=f)
                    if collision(np.r_[q[:6],grip]):
                        raise ValueError('Self collision')
                    qs.append(q[:6].copy())
                qs=np.array(qs)
                excursion=float(max(np.max(abs(qs-rest)),np.max(np.ptp(qs,axis=0))))
                item.update(coarse_ok=True,excursion_rad=excursion)
                if best is None or excursion < best['excursion_rad']:
                    plan=plan_tracking(p,start_q,curve,.01,'Center facing',bounds=bounds,
                                       collision=collision,ramp_placement='Inside edge')
                    p.plan(np.r_[rest,grip],start_q,bounds=bounds,collision=collision)
                    p.plan(plan.motion.path.positions[-1],np.r_[rest,grip],bounds=bounds,collision=collision)
                    dense=plan.motion.path.spline(np.linspace(0,1,1001))[:,:6]
                    excursion=float(max(np.max(abs(dense-rest)),np.max(np.ptp(dense,axis=0))))
                    best={'excursion_rad':excursion,'max_rest_rad':float(np.max(abs(dense-rest))),
                          'max_span_rad':float(np.max(np.ptp(dense,axis=0))),
                          'seed_q':start_q.tolist(),'center_m':curve.center.tolist(),
                          'start_m':curve.start.tolist(),'normal':curve.normal.tolist(),
                          'radius_m':curve.radius,'plan':plan.motion.report,
                          'end_q':plan.motion.path.positions[-1].tolist(),'requires_approval':excursion>=.49}
                    (ROOT/'review-arc.json').write_text(json.dumps(best,indent=2))
                    (ROOT/'assets/review-arc.step').write_bytes(asset.read_bytes())
                    print('BEST',excursion,flush=True)
                    # One demonstrably feasible, bounded candidate is enough for
                    # review; the normal automatic .49 limit is never changed.
                    if excursion<1.8:
                        records.append(item)
                        break
            except Exception as exc:
                item.update(error=str(exc))
            records.append(item)
            print('SEARCH',j2,j3,j4,item.get('coarse_ok',False),item.get('error',''),flush=True)
            (ROOT/'expanded-arc-search.json').write_text(json.dumps(records,indent=2))
        (ROOT/'expanded-arc-search.json').write_text(json.dumps(records,indent=2))
    finally:
        Path(model).unlink()


if __name__=='__main__':
    tyro.cli(main)
