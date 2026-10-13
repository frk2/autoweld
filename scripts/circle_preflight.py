"""Offline circle search and exact SDK kinematic/derivative preflight; no motor I/O."""
import json
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np
from i2rt.utils.viser_control_interface import ViserControlInterface

from autoweld.app import tool_model
from autoweld.diagnostics import solve_pose
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.tracking import load_tracks, plan_tracking

ROOT = Path('evidence/circle-speed')
FEEDS = (.005, .01, .02, .03, .04, .05, .06, .08, .1)


def main():
    state = json.loads(Path('runtime/live/status.json').read_text())
    assert state['closed'] and state['automatic_test_excursion_rad'] == .49
    (ROOT/'before-status.json').write_text(json.dumps(state, indent=2))
    rest = np.array(state['rest_rad'])
    grip = json.loads(Path('runtime/live/rest.json').read_text())['gripper_normalized']
    model = tool_model()
    planner = MotionPlanner(model)
    planner.limits = MotionLimits(**state['motion_limits'])
    checker = ViserControlInterface.__new__(ViserControlInterface)
    checker._model = planner.model
    checker._check_data = mujoco.MjData(planner.model)
    def collision(q):
        return checker._has_self_collision(q, len(q))
    bounds = planner.model.jnt_range[:6].copy()
    bounds[:, 0] = np.minimum(bounds[:, 0], rest)
    bounds[:, 1] = np.maximum(bounds[:, 1], rest)
    cost = np.full(planner.model.nv, .1)
    cost[:3] = .01
    attempts, selected = [], None
    try:
        for radius in (.06, .05, .04, .03, .025, .02):
            for j2,j3 in ((.20,.25),(.15,.2),(.25,.25),(.15,.3),(.25,.35),(.3,.2)):
                seed = np.r_[rest+[0,j2,j3,.1,0,0], grip]
                T = planner.fk(seed)
                for normal in ([1,0,0],[0,0,1],[0,1,0],[1,0,1],[1,1,0]):
                    item = {'radius_m':radius, 'normal':normal, 'seed_q':seed.tolist(), 'center_m':T[:3,3].tolist()}
                    try:
                        edge = cq.Edge.makeCircle(radius*1000,cq.Vector(*(T[:3,3]*1000)),cq.Vector(*normal))
                        cq.exporters.export(edge,str(ROOT/'assets/candidate.step'))
                        curve = load_tracks(ROOT/'assets/candidate.step')[0]
                        q = planner._configuration(seed)
                        positions=[]
                        for fraction in np.linspace(0,1,65):
                            q = solve_pose(planner.kin,curve.pose(fraction,T[:3,:3],'Unconstrained'),planner.site_name,q,
                                           stage='circle_search',path_fraction=fraction,orientation_required=False,
                                           joint_motion_cost=cost)
                            positions.append(q[:6].copy())
                            if np.any(q[:6]<bounds[:,0]) or np.any(q[:6]>bounds[:,1]):
                                raise ValueError('Joint bound at coarse IK sample')
                            if max(np.max(abs(q[:6]-rest)),np.max(abs(q[:6]-seed[:6]))) >= .465:
                                raise ValueError('Coarse IK exceeds .465 rad margin')
                            if collision(np.r_[q[:6],grip]):
                                raise ValueError('Self collision at coarse IK sample')
                        positions=np.array(positions)
                        if np.max(np.ptp(positions,axis=0)) >= .465:
                            raise ValueError('Circle joint span exceeds .465 rad margin')
                        # Exact path preflight includes interpolated samples and original guards.
                        def validate(q):
                            if max(np.max(abs(q[:6]-rest)),np.max(abs(q[:6]-seed[:6])),
                                   np.max(abs(q[:6]-positions[0]))) >= .475:
                                raise ValueError('Dense automatic excursion exceeds .475 rad margin')
                        plan=plan_tracking(planner,seed,curve,.02,bounds=bounds,collision=collision,
                                           validate=validate,ramp_placement='Inside edge')
                        planner.plan(np.r_[rest,grip],seed,bounds=bounds,collision=collision)
                        planner.plan(seed,plan.motion.path.positions[0],bounds=bounds,collision=collision)
                        planner.plan(plan.motion.path.positions[-1],np.r_[rest,grip],bounds=bounds,collision=collision)
                        selected=item | {'normal':curve.normal.tolist(),'start_m':curve.start.tolist(),
                                          'joint_span_rad':np.ptp(positions,axis=0).tolist()}
                        item['ok']=True
                    except Exception as exc:
                        item.update(ok=False,error=str(exc))
                    attempts.append(item)
                    print('SEARCH',radius,j2,j3,normal,item['ok'],item.get('error',''),flush=True)
                    if selected:
                        (ROOT/'assets/circle.step').write_bytes((ROOT/'assets/candidate.step').read_bytes())
                        break
                if selected:
                    break
            if selected:
                break
        (ROOT/'search.json').write_text(json.dumps(attempts,indent=2))
        if selected is None:
            raise RuntimeError('No sampled circle passed the original automatic bounds')
        (ROOT/'circle.json').write_text(json.dumps(selected,indent=2))
        records=[]
        for feed in FEEDS:
            item={'feed_mm_s':feed*1000,'seed_q':seed.tolist()}
            try:
                plan=plan_tracking(planner,seed,curve,feed,bounds=bounds,collision=collision,
                                   validate=validate,ramp_placement='Inside edge')
                q=plan.motion.path.spline(np.linspace(0,1,1001))[:,:6]
                item.update(ok=True,plan=plan.motion.report,start_q=plan.motion.path.positions[0].tolist(),
                            end_q=plan.motion.path.positions[-1].tolist(),
                            max_rest_excursion_rad=float(np.max(abs(q-rest))),
                            max_joint_span_rad=float(np.max(np.ptp(q,axis=0))))
            except Exception as exc:
                item.update(ok=False,error=str(exc))
            records.append(item)
            print('PREFLIGHT',feed,item['ok'],item.get('error',''),flush=True)
            (ROOT/'preflight.json').write_text(json.dumps(records,indent=2))
    finally:
        Path(model).unlink()


if __name__=='__main__':
    main()
