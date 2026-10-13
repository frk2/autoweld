"""Reproduce Stage 2 reference math and live reference/tracking plots."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner

root = Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/stage2")
model = tool_model()
planner = MotionPlanner(model)
start = np.array([0.1, 0.4, 0.8, 0.1, -0.15, 0.2, 0.5])
goal = start + [0.005, 0.01, 0.015, 0, 0, 0, 0]
plan = planner.plan(start, goal, cartesian=planner.fk(goal))
t = np.linspace(0, plan.path.duration, 1802)
joints = np.array([plan.path.sample(x) for x in t])
tcp = np.array([planner.derivatives(plan.path, x) for x in t])
fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
for derivative, unit in enumerate(["rad/s", "rad/s²", "rad/s³"]):
    ax[derivative, 0].plot(t, joints[:, derivative + 1, :6])
    ax[derivative, 0].set_ylabel(unit)
    ax[derivative, 1].plot(t, np.linalg.norm(tcp[:, derivative, :3], axis=1), label="TCP translation")
    limit = list(plan.report["limits"].values())[derivative]
    ax[derivative, 1].axhline(limit, color="red", ls="--", label="limit")
    ax[derivative, 1].set_ylabel(["m/s", "m/s²", "m/s³"][derivative])
    for col in range(2):
        ax[derivative, col].grid(alpha=0.25)
ax[0, 0].set_title("Joint references — Cartesian IK path")
ax[0, 1].set_title("TCP translational reference norms")
ax[-1, 0].set_xlabel("Time (s)")
ax[-1, 1].set_xlabel("Time (s)")
ax[0, 1].legend()
fig.tight_layout()
fig.savefig(root / "numerical-derivatives.png", dpi=160)
plt.close(fig)
(root / "numerical-plan.json").write_text(json.dumps(plan.report, indent=2))
results = []
for source in sorted(root.glob("*.jsonl")):
    rows = [json.loads(x) for x in source.read_text().splitlines()]
    rows = [r for r in rows if r.get("motion_reference")]
    if len(rows) < 5:
        continue
    # New session begins when the accepted SDK reference is installed.
    session = rows[-1]["motion_reference"]["session"]
    rows = [r for r in rows if r["motion_reference"]["session"] == session]
    t = np.array([r["motion_reference"]["elapsed_s"] for r in rows])
    q = np.array([r["q"] for r in rows])
    reference = np.array([r["motion_reference"]["q"][:6] for r in rows])
    goal = np.array(rows[-1]["target"])
    duration = rows[-1]["motion_reference"]["duration_s"]
    error = np.rad2deg(q - goal)
    direction = np.sign(goal - reference[0])
    overshoot = np.maximum(0, np.max(error * direction, axis=0))
    in_band = np.all(abs(error) <= [0.2, 0.2, 0.2, 1, 1, 1], axis=1)
    bad = np.flatnonzero(~in_band)
    settled = int(bad[-1] + 1) if len(bad) else 0
    settle_time = float(t[settled]) if settled < len(t) else None
    speed = np.array([np.linalg.norm(planner.jacobian(r["q"])[:3] @ np.array(r["velocity"])) for r in rows])
    result = {
        "name": source.stem,
        "duration_s": duration,
        "samples": len(rows),
        "settled_from_motion_start_s": settle_time,
        "settled_after_reference_end_s": None if settle_time is None else max(0.0, settle_time - duration),
        "directed_overshoot_deg": overshoot.tolist(),
        "last_error_deg": error[-1].tolist(),
        "max_reference_tracking_error_deg": np.max(abs(np.rad2deg(q - reference)), axis=0).tolist(),
        "measured_joint_peak_speed_rad_s": np.max(
            abs(np.array([r["velocity"] for r in rows])), axis=0
        ).tolist(),
        "estimated_measured_tcp_peak_speed_m_s": float(max(speed)),
        "elapsed_sample_gap_max_s": float(max(np.diff(t))),
    }
    results.append(result)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, np.rad2deg(reference[:, j] - goal[j]), label="Reference")
        ax.plot(t, error[:, j], label="Measured", lw=1)
        tol = 0.2 if j < 3 else 1
        ax.axhspan(-tol, tol, color="green", alpha=0.06)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(duration, color="grey", ls="--")
        ax.set_title(f"J{j + 1}")
        ax.set_ylabel("Angle relative to goal (°)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(source.stem + " — reference vs feedback; dashed line = planned completion")
    fig.tight_layout()
    fig.savefig(root / f"{source.stem}.png", dpi=160)
    plt.close(fig)
(root / "metrics.json").write_text(json.dumps(results, indent=2))
Path(model).unlink()
print(json.dumps(results, indent=2))
