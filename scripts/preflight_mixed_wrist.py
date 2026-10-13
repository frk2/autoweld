"""Preflight matched line/circle tests in the previously tested circle workspace."""
import json
from dataclasses import replace
from pathlib import Path

import cadquery as cq
import mujoco
import numpy as np
from i2rt.utils.viser_control_interface import ViserControlInterface

from autoweld.app import tool_model
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.tracking import load_tracks, plan_tracking


def main():
    dest = Path('evidence/mixed-wrist')
    (dest/'assets').mkdir(parents=True, exist_ok=True)
    rest = np.r_[json.loads(Path('runtime/live/rest.json').read_text())['arm_rad'], 0.]
    circle = json.loads(Path('evidence/circle-speed/circle.json').read_text())
    seed = np.array(circle['seed_q'])
    seed[6] = 0.
    center = np.array(circle['center_m'])
    a, b = center+[0,0,.06], center-[0,0,.06]
    line = cq.Edge.makeLine(cq.Vector(*(a*1000)), cq.Vector(*(b*1000)))
    cq.exporters.export(line, str(dest/'assets/line120.step'))
    (dest/'assets/circle60.step').write_bytes(Path('evidence/circle-speed/assets/circle.step').read_bytes())
    model = tool_model()
    records = []
    try:
        planner = MotionPlanner(model)
        planner.limits = replace(MotionLimits(**json.loads(Path('runtime/live/motion-limits.json').read_text())), joint_velocity=.25)
        checker = ViserControlInterface.__new__(ViserControlInterface)
        checker._model, checker._check_data = planner.model, mujoco.MjData(planner.model)
        bounds = planner.model.jnt_range[:6].copy()
        bounds[:, 0] = np.minimum(bounds[:, 0], rest[:6])
        bounds[:, 1] = np.maximum(bounds[:, 1], rest[:6])
        def collision(q):
            return checker._has_self_collision(q, len(q))
        def validate(q):
            if max(np.max(abs(q[:6]-rest[:6])), np.max(abs(q[:6]-seed[:6]))) >= .49:
                raise ValueError('Automatic .49 rad excursion exceeded')
        commission = rest + [0,.04,.06,.015,.015,0,0]
        commission_plans = [planner.plan(x,y,bounds=bounds,collision=collision,validate=validate).report
                            for x,y in [(rest,commission),(commission,rest)]]
        for name, feed in [('line120', .02), ('circle60', .05)]:
            curve = load_tracks(dest/'assets'/f'{name}.step')[0]
            item = dict(name=name, feed_m_s=feed, seed_q=seed.tolist(),
                        asset=str(dest/'assets'/f'{name}.step'), cap_rad_s=.25)
            try:
                plan = plan_tracking(planner, seed, curve, feed, bounds=bounds, collision=collision,
                                     validate=validate, ramp_placement='Inside edge')
                qs = plan.motion.path.spline(np.linspace(0,1,1001))
                if np.max(np.ptp(qs[:,:6],axis=0)) >= .49:
                    raise ValueError('Path joint span exceeds automatic bound')
                approaches = [planner.plan(x,y,bounds=bounds,collision=collision,validate=validate).report
                              for x,y in [(rest,seed),(seed,plan.motion.path.positions[0]),
                                          (plan.motion.path.positions[-1],rest)]]
                item.update(ok=True, plan=plan.motion.report, approaches=approaches,
                            max_rest_excursion_rad=float(np.max(abs(qs[:,:6]-rest[:6]))),
                            max_joint_span_rad=float(np.max(np.ptp(qs[:,:6],axis=0))))
            except Exception as exc:
                item.update(ok=False,error=str(exc))
            records.append(item)
            print(name, item.get('ok'), item.get('error', ''), flush=True)
        (dest/'preflight.json').write_text(json.dumps(dict(rest=rest.tolist(), commissioning_q=commission.tolist(),
            commissioning_plans=commission_plans, fixtures=records), indent=2))
        if not all(r['ok'] for r in records):
            raise RuntimeError('A requested path failed preflight; no live command sent')
    finally:
        Path(model).unlink()


if __name__ == '__main__':
    main()
