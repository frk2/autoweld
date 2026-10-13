"""Offline radius/angle/pose search with all normal automatic and SDK checks."""
import json
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np
from i2rt.utils.viser_control_interface import ViserControlInterface
from scipy.spatial.transform import Rotation

from autoweld.app import tool_model
from autoweld.diagnostics import solve_pose
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.tracking import load_tracks, plan_tracking

ROOT=Path('evidence/small-oriented-arcs')


def main():
    state=json.loads(Path('runtime/live/status.json').read_text())
    assert state['closed'] and state['automatic_test_excursion_rad']==.49
    (ROOT/'before-status.json').write_text(json.dumps(state,indent=2))
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
    candidates=[]
    offsets=[]
    for j2,j3 in ((.15,.25),(.22,.3),(.3,.35),(.4,.44)):
        for j5 in (0.,.2,-.2):
            for j6 in (0.,.25,-.25):
                for j4 in (0.,.2,-.2):
                    offsets.append([0,j2,j3,j4,j5,j6])
    for degrees in (45,30,20,15,10,8,5,3):
        for radius in (.0127,.025,.05,.075,.10,.15,.20):
            # Permit lower-feed commissioning on short arcs as well.
            if radius*np.deg2rad(degrees)<=.0005:
                continue
            for offset in offsets:
                mid=np.r_[rest+offset,grip]
                T=p.fk(mid)
                normal,radial=T[:3,0],T[:3,2]
                J=p.jacobian(mid)
                dq=np.linalg.lstsq(J,np.r_[radius*np.cross(normal,radial),normal],rcond=1e-5)[0]
                for placement in (0.,-.5,-1.):
                    endpoints=np.array([mid[:6]+dq*np.deg2rad(degrees)*f for f in (placement,placement+1)])
                    risk=max(np.max(abs(endpoints-rest)),np.max(np.ptp(endpoints,axis=0)))
                    if np.isfinite(risk):
                        candidates.append({'radius_m':radius,'angle_deg':degrees,'mid_q':mid.tolist(),
                                           'placement':placement,'linear_prediction_rad':float(risk)})
    # Prefer 45 degrees, then smaller angles; rank plausible poses by joint margin.
    candidates.sort(key=lambda x:(-x['angle_deg'],-x['radius_m'],x['linear_prediction_rad']))
    attempts=[]
    previous=ROOT/'preflight.json'
    successes=json.loads(previous.read_text()) if previous.exists() else []
    old_count=len(successes)
    per_angle={}
    try:
        for c in candidates:
            deg=c['angle_deg']
            group=(deg,c['radius_m'])
            if per_angle.get(group,0)>=3:
                continue
            if any(s['angle_deg']==deg and s['radius_m']==c['radius_m'] for s in successes[old_count:]):
                continue
            per_angle[group]=per_angle.get(group,0)+1
            mid=np.array(c['mid_q'])
            T=p.fk(mid)
            radius=c['radius_m']
            angle=np.deg2rad(deg)
            lo=c['placement']*angle
            center=T[:3,3]-radius*T[:3,2]
            points=[cq.Vector(*((center+radius*Rotation.from_rotvec(T[:3,0]*a).apply(T[:3,2]))*1000)) for a in (lo,lo+angle/2,lo+angle)]
            asset=ROOT/'assets/candidate.step'
            cq.exporters.export(cq.Edge.makeThreePointArc(*points),str(asset))
            curve=load_tracks(asset)[0]
            item=c.copy()
            def validate(q):
                if np.max(abs(q[:6]-rest))>=.475:
                    raise ValueError('Automatic rest-relative margin exceeds .475 rad')
            try:
                q=p._configuration(mid)
                # Continue from the known midpoint pose instead of a large IK jump.
                for a in np.linspace(0,lo,max(2,int(abs(lo)/.02)+1)):
                    fraction=(a-lo)/angle
                    q=solve_pose(p.kin,curve.pose(fraction,T[:3,:3],'Center facing'),p.site_name,q,
                                 stage='small_arc_start',path_fraction=fraction)
                    validate(q)
                seed=np.r_[q[:6],grip]
                qs=[]
                for f in np.linspace(0,1,21):
                    q=solve_pose(p.kin,curve.pose(f,T[:3,:3],'Center facing'),p.site_name,q,
                                 stage='small_arc_search',path_fraction=f)
                    validate(q)
                    qs.append(q[:6].copy())
                if np.max(np.ptp(qs,axis=0))>=.475:
                    raise ValueError('Automatic path span exceeds .475 rad margin')
                if np.max(abs(seed[:6]-rest))>=.475:
                    raise ValueError('Start seed exceeds .475 rad margin')
                feed_errors=[]
                for feed in (.01,.005,.002,.001,.0005,.0002):
                    try:
                        plan=plan_tracking(p,seed,curve,feed,'Center facing',bounds=bounds,
                                           collision=collision,validate=validate,ramp_placement='Inside edge')
                        break
                    except Exception as exc:
                        feed_errors.append({'feed_mm_s':feed*1000,'error':str(exc)})
                else:
                    raise ValueError(str(feed_errors))
                dense=plan.motion.path.spline(np.linspace(0,1,1001))[:,:6]
                if np.max(np.ptp(dense,axis=0))>=.475:
                    raise ValueError('Dense joint span exceeds .475 rad margin')
                p.plan(np.r_[rest,grip],seed,bounds=bounds,collision=collision,validate=validate)
                p.plan(seed,plan.motion.path.positions[0],bounds=bounds,collision=collision,validate=validate)
                p.plan(plan.motion.path.positions[-1],np.r_[rest,grip],bounds=bounds,collision=collision,validate=validate)
                name=f"arc-{len(successes)+1:02}-r{radius*1000:g}-a{deg:g}"
                target=ROOT/f'assets/{name}.step'
                target.write_bytes(asset.read_bytes())
                item.update(ok=True,name=name,asset=str(target),seed_q=seed.tolist(),feed_mm_s=feed*1000,feed_rejections=feed_errors,
                            center_m=curve.center.tolist(),start_m=curve.start.tolist(),normal=curve.normal.tolist(),
                            max_rest_excursion_rad=float(np.max(abs(dense-rest))),
                            max_joint_span_rad=float(np.max(np.ptp(dense,axis=0))),plan=plan.motion.report)
                successes.append(item.copy())
                print('PASS',name,'rest',item['max_rest_excursion_rad'],'span',item['max_joint_span_rad'],'feed',feed*1000,flush=True)
            except Exception as exc:
                item.update(ok=False,error=str(exc))
                print('REJECT',deg,radius,c['linear_prediction_rad'],str(exc),flush=True)
            attempts.append(item)
            (ROOT/'search-wide.json').write_text(json.dumps(attempts,indent=2))
            (ROOT/'preflight.json').write_text(json.dumps(successes,indent=2))
            if len(successes)>=old_count+12:
                break
        print('DONE',len(attempts),'attempts;',len(successes),'accepted paths',flush=True)
    finally:
        Path(model).unlink()


if __name__=='__main__':
    main()
