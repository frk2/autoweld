"""Offline checks for the explicitly requested oriented Stage 3 repeats."""
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

DEST=Path('evidence/stage3-welding')


def main():
    state=json.loads(Path('runtime/live/status.json').read_text())
    assert state['closed'] and state['automatic_test_excursion_rad']==.49
    (DEST/'before-status.json').write_text(json.dumps(state,indent=2))
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
    fixtures=json.loads(Path('assets/stage3/manifest.json').read_text())
    records=[]
    def check(asset, seed, orientation, feed):
        curve=load_tracks(asset)[0]
        def validate(q):
            if max(np.max(abs(q[:6]-rest)),np.max(abs(q[:6]-seed[:6])))>=.475:
                raise ValueError('Automatic excursion exceeds .475 rad preflight margin')
        plan=plan_tracking(p,seed,curve,feed,orientation,bounds=bounds,collision=collision,
                           validate=validate,ramp_placement='Inside edge')
        q=plan.motion.path.spline(np.linspace(0,1,1001))[:,:6]
        if np.max(np.ptp(q,axis=0))>=.475:
            raise ValueError('Path joint span exceeds .475 rad preflight margin')
        p.plan(np.r_[rest,grip],seed,bounds=bounds,collision=collision,validate=validate)
        p.plan(seed,plan.motion.path.positions[0],bounds=bounds,collision=collision,validate=validate)
        p.plan(plan.motion.path.positions[-1],np.r_[rest,grip],bounds=bounds,collision=collision)
        return {'ok':True,'plan':plan.motion.report,'max_rest_excursion_rad':float(np.max(abs(q-rest))),
                'max_joint_span_rad':float(np.max(np.ptp(q,axis=0))),
                'start_q':plan.motion.path.positions[0].tolist(),'end_q':plan.motion.path.positions[-1].tolist()}
    try:
        for i,entry in enumerate(fixtures):
            seed=np.r_[entry['seed_q'][:6],grip]
            asset=DEST/f'assets/perpendicular-line-{i+1:02}.step'
            asset.write_bytes(Path(f'assets/stage3/perpendicular-line-{i+1:02}.step').read_bytes())
            item={'name':f'perpendicular-line-{i+1:02}','suite':'perpendicular-lines','seed_q':seed.tolist(),
                  'feed_mm_s':20,'asset':str(asset),'orientation':'Perpendicular'}
            try:
                item.update(check(asset,seed,'Perpendicular',.02))
            except Exception as exc:
                item.update(ok=False,error=str(exc))
            records.append(item)
            print(item['name'],item['ok'],item.get('error',''),flush=True)
        # Original arcs first, retaining exact diagnostics.
        original=[]
        for i,entry in enumerate(fixtures):
            seed=np.r_[entry['seed_q'][:6],grip]
            asset=Path(f'assets/stage3/center-arc-{i+1:02}.step')
            item={'name':f'original-center-arc-{i+1:02}','seed_q':seed.tolist()}
            try:
                item.update(check(asset,seed,'Center facing',.01))
            except Exception as exc:
                item.update(ok=False,error=str(exc))
            original.append(item)
            print(item['name'],item['ok'],item.get('error',''),flush=True)
        (DEST/'original-arcs.json').write_text(json.dumps(original,indent=2))
        (DEST/'preflight.json').write_text(json.dumps(records,indent=2))
        # Search quarter arcs centered on free-space seed orientations, including
        # both traversal directions. All four wrist/arm bounds remain unchanged.
        attempts=[]
        successes=[]
        for j2,j3,j4,j5 in ((.15,.25,0,0),(.2,.3,.2,.2),(.2,.3,-.2,-.2),
                            (.12,.2,.35,.35),(.25,.2,-.35,.35),(.2,.3,.35,-.35),
                            (.1,.3,-.35,-.35),(.3,.3,0,.4),(.3,.3,0,-.4)):
            seed=np.r_[rest+[0,j2,j3,j4,j5,0],grip]
            T=p.fk(seed)
            for start_angle in (-np.pi/4,0,-np.pi/2):
                for sign in (1,-1):
                    normal=T[:3,0]*sign
                    radial=T[:3,2]
                    center=T[:3,3]-.0127*radial
                    points=[cq.Vector(*((center+.0127*Rotation.from_rotvec(normal*a).apply(radial))*1000))
                            for a in (start_angle,start_angle+np.pi/4,start_angle+np.pi/2)]
                    edge=cq.Edge.makeThreePointArc(*points)
                    asset=DEST/'assets/search-arc.step'
                    cq.exporters.export(edge,str(asset))
                    curve=load_tracks(asset)[0]
                    attempt={'seed_q':seed.tolist(),'start_angle':start_angle,'sign':sign}
                    try:
                        q=p._configuration(seed)
                        for fraction in np.linspace(0,1,21):
                            q=solve_pose(p.kin,curve.pose(fraction,T[:3,:3],'Center facing'),p.site_name,q,
                                         stage='oriented_arc_search',path_fraction=fraction)
                            if np.max(abs(q[:6]-rest))>=.475:
                                raise ValueError(f'Coarse quarter arc exceeds automatic margin at {fraction:.0%}')
                        checked=check(asset,seed,'Center facing',.01)
                        attempt.update(checked)
                        successes.append((asset.read_bytes(),seed.copy(),checked))
                    except Exception as exc:
                        attempt.update(ok=False,error=str(exc))
                    attempts.append(attempt)
                    print('SEARCH',len(attempts),attempt['ok'],attempt.get('error',''),flush=True)
                    (DEST/'arc-search.json').write_text(json.dumps(attempts,indent=2))
                    if len(successes)>=5:
                        break
                if len(successes)>=5:
                    break
            if len(successes)>=5:
                break
        for i,(data,seed,checked) in enumerate(successes):
            asset=DEST/f'assets/center-arc-{i+1:02}.step'
            asset.write_bytes(data)
            records.append({'name':f'center-arc-{i+1:02}','suite':'center-arcs','seed_q':seed.tolist(),
                            'feed_mm_s':10,'orientation':'Center facing','asset':str(asset),**checked})
        (DEST/'preflight.json').write_text(json.dumps(records,indent=2))
    finally:
        Path(model).unlink()


if __name__=='__main__':
    main()
