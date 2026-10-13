"""Offline bidirectional, free-space gravity calibration fixture checks."""

import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from i2rt.utils.viser_control_interface import ViserControlInterface

from autoweld.app import tool_model
from autoweld.motion import MotionLimits, MotionPlanner

DEST = Path("evidence/mixed-gravity")


def main():
    DEST.mkdir(exist_ok=True)
    rest = np.r_[json.loads(Path("runtime/live/rest.json").read_text())["arm_rad"], 0.0]
    seed = rest + [0, 0.20, 0.25, 0.10, 0, 0, 0]
    centers = [
        ("central", seed),
        ("positive_j5", seed + [0, 0, 0, 0.08, 0.18, 0, 0]),
        ("negative_j5", seed + [0, 0, 0, 0.08, -0.18, 0, 0]),
        ("raised", seed + [0, 0.1, 0.12, 0.06, 0, 0, 0]),
    ]
    dither = np.array([0, 0, 0, 0.03, 0.03, 0, 0])
    model = tool_model()
    try:
        planner = MotionPlanner(model)
        planner.model.opt.gravity[:] = [0, 0, -9.81]
        planner.limits = replace(
            MotionLimits(**json.loads(Path("runtime/live/motion-limits.json").read_text())),
            joint_velocity=0.15,
        )
        checker = ViserControlInterface.__new__(ViserControlInterface)
        checker._model, checker._check_data = planner.model, mujoco.MjData(planner.model)
        bounds = planner.model.jnt_range[:6].copy()
        bounds[:, 0] = np.minimum(bounds[:, 0], rest[:6])
        bounds[:, 1] = np.maximum(bounds[:, 1], rest[:6])

        def collision(q):
            return checker._has_self_collision(q, len(q))

        paths = []
        poses = []
        previous = rest
        for name, center in centers:
            transform = planner.fk(center)
            poses.append(
                dict(
                    name=name,
                    q=center.tolist(),
                    tcp_m=transform[:3, 3].tolist(),
                    gravity_unscaled_nm=planner.data.qfrc_bias[:6].tolist(),
                )
            )
            for side in (-1, 1):
                approach = center + side * dither
                for label, target in [("approach", approach), ("measure", center)]:
                    prior = previous.copy()

                    def validate(q):
                        if max(np.max(abs(q[:6] - rest[:6])), np.max(abs(q[:6] - prior[:6]))) >= 0.49:
                            raise ValueError("Original automatic excursion bound exceeded")

                    plan = planner.plan(prior, target, bounds=bounds, collision=collision, validate=validate)
                    paths.append(dict(pose=name, side=side, kind=label, q=target.tolist(), plan=plan.report))
                    previous = target
        final = planner.plan(previous, rest, bounds=bounds, collision=collision)
        spec = dict(
            rest=rest.tolist(), seed=seed.tolist(), poses=poses, sequence=paths, return_plan=final.report
        )
        (DEST / "calibration-preflight.json").write_text(json.dumps(spec, indent=2))
        # Reuse exactly the previous requested Cartesian geometry and feeds.
        (DEST / "preflight.json").write_bytes(Path("evidence/mixed-wrist/preflight.json").read_bytes())
        for file in ("rest.json", "motion-limits.json", "status.json", "gripper-hold.json"):
            (DEST / ("before-" + file)).write_bytes((Path("runtime/live") / file).read_bytes())
        print(json.dumps(poses, indent=2))
        print("All", len(paths), "calibration moves and the return pass offline checks")
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    main()
