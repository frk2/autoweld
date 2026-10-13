"""Revalidate the historical guard-stopped fixtures before repeat acquisition."""
import json
import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import tyro
from i2rt.utils.viser_control_interface import ViserControlInterface

from autoweld.app import tool_model
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.tracking import load_tracks, plan_tracking

DEST = Path('evidence/path-deviation-repeat')


def main(manifest_path: str = str(DEST / 'manifest.json'), output_path: str = str(DEST / 'preflight.json')):
    state = json.loads(Path('runtime/live/status.json').read_text())
    assert state['closed']
    rest = np.array(state['rest_rad'])
    grip = json.loads(Path('runtime/live/rest.json').read_text())['gripper_normalized']
    model = tool_model()
    p = MotionPlanner(model)
    operator_limits = MotionLimits(**state['motion_limits'])
    checker = ViserControlInterface.__new__(ViserControlInterface)
    checker._model = p.model
    checker._check_data = mujoco.MjData(p.model)

    def collision(q):
        return checker._has_self_collision(q, len(q))

    bounds = p.model.jnt_range[:6].copy()
    bounds[:, 0] = np.minimum(bounds[:, 0], rest)
    bounds[:, 1] = np.maximum(bounds[:, 1], rest)
    records = []
    try:
        for item in json.loads(Path(manifest_path).read_text()):
            seed = np.r_[item['seed_q'][:6], grip]
            orientation = {'lines': 'Unconstrained', 'circles': 'Unconstrained',
                           'perpendicular-lines': 'Perpendicular', 'center-arcs': 'Center facing'}[item['suite']]
            item = {**item, 'seed_q': seed.tolist(), 'orientation': orientation}
            margin = item['automatic_limit_rad'] - .005

            def validate(q):
                if max(np.max(abs(q[:6] - rest)), np.max(abs(q[:6] - seed[:6]))) >= margin:
                    raise ValueError(f'Automatic preflight margin {margin:g} rad exceeded')

            try:
                curve = load_tracks(item['asset'])[item['index']]
                p.limits = operator_limits
                plan = plan_tracking(p, seed, curve, item['feed_mm_s'] / 1000, orientation,
                                     bounds=bounds, collision=collision, validate=validate,
                                     ramp_placement=item['ramp_placement'])
                cap = max(.15, math.ceil(plan.motion.report['planned_peaks']['joint_velocity'] * 1.25 / .05) * .05)
                p.limits = replace(operator_limits, joint_velocity=min(cap, operator_limits.joint_velocity))
                plan = plan_tracking(p, seed, curve, item['feed_mm_s'] / 1000, orientation,
                                     bounds=bounds, collision=collision, validate=validate,
                                     ramp_placement=item['ramp_placement'])
                dense = plan.motion.path.spline(np.linspace(0, 1, 1001))[:, :6]
                if np.max(np.ptp(dense, axis=0)) >= margin:
                    raise ValueError('Dense joint span exceeds automatic margin')
                p.plan(np.r_[rest, grip], seed, bounds=bounds, collision=collision, validate=validate)
                p.plan(seed, plan.motion.path.positions[0], bounds=bounds, collision=collision, validate=validate)
                p.limits = replace(operator_limits, joint_velocity=.15)
                p.plan(plan.motion.path.positions[-1], np.r_[rest, grip], bounds=bounds,
                       collision=collision, validate=validate)
                item.update(ok=True, cap_rad_s=cap, plan=plan.motion.report,
                            max_rest_excursion_rad=float(np.max(abs(dense - rest))),
                            max_joint_span_rad=float(np.max(np.ptp(dense, axis=0))))
            except Exception as exc:
                item.update(ok=False, error=str(exc))
            records.append(item)
            Path(output_path).write_text(json.dumps(records, indent=2))
            print(item['name'], item['ok'], item.get('error', ''), flush=True)
    finally:
        Path(model).unlink()


if __name__ == '__main__':
    tyro.cli(main)
