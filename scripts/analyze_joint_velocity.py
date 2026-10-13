"""Compare reference and measured velocities from bounded joint-sweep logs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.velocity import PositionVelocity


def main(root: str = "evidence/velocity"):
    root = Path(root)
    model = tool_model()
    planner = MotionPlanner(model)
    records = []
    try:
        for path in sorted(root.glob("*j1-*.jsonl")):
            if path.stem.endswith("-control"):
                continue
            meta = json.loads(path.with_suffix(".json").read_text())
            session = meta["after"]["motion_reference"]["session"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = [r for r in rows if (r.get("motion_reference") or {}).get("session") == session]
            estimator = PositionVelocity()
            series = []
            for row in rows:
                ref = row["motion_reference"]
                q = np.array(row["q"])
                raw = np.linalg.norm(planner.jacobian(q)[:3] @ np.array(row["velocity"]))
                desired = np.linalg.norm(planner.jacobian(ref["q"])[:3] @ np.array(ref["velocity"])[:6])
                estimated = estimator.update(row["feedback_timestamps"], q, row["timestamp"])
                slope = (
                    np.linalg.norm(planner.jacobian(q)[:3] @ np.array(estimated["joint_velocity_rad_s"]))
                    if estimated
                    else np.nan
                )
                series.append(
                    [ref["elapsed_s"], raw, desired, slope, estimated["delay_s"] if estimated else 0]
                )
            series = np.array(series)
            t, raw, desired, slope, delay = series.T
            duration = meta["after"]["motion_plan"]["duration_s"]
            hold = t > duration + 3
            sdk = [r["sdk_timing"] for r in rows]
            q = np.array([r["q"] for r in rows])
            record = {
                "trial": path.stem,
                "reference_peak_mm_s": float(max(desired) * 1000),
                "raw_peak_mm_s": float(max(raw) * 1000),
                "position_slope_peak_mm_s": float(np.nanmax(slope) * 1000),
                "hold_raw_median_mm_s": float(np.median(raw[hold]) * 1000),
                "hold_position_slope_median_mm_s": float(np.nanmedian(slope[hold]) * 1000),
                "final_error_deg": meta["after"]["error_deg"],
                "fault": meta["after"]["fault"],
                "max_commanded_step_rad": float(
                    max(abs(np.array(meta["payload"]["q"]) - meta["before"]["q_rad"]))
                ),
                "max_measured_rest_excursion_rad": float(
                    np.max(abs(q - np.array(meta["before"]["rest_rad"])))
                ),
                "max_can_cycle_ms": float(max(r["cycle_period_s"] for r in sdk) * 1000),
                "max_host_update_ms": float(max(r.get("host_update_s", 0) for r in sdk) * 1000),
                "all_integral_zero": all(not any(r["integral"]) for r in rows),
                "command_compensation": meta["before"].get("command_quantization_compensation"),
            }
            records.append(record)
            fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            ax[0].plot(t, desired * 1000, "k--", label="Planned TCP speed")
            ax[0].plot(t, raw * 1000, alpha=0.6, label="Raw velocity → Jacobian norm")
            ax[0].plot(t - delay, slope * 1000, label="Position slope (120 ms window; time aligned)")
            ax[0].set_ylabel("mm/s")
            for joint in range(6):
                e = [np.rad2deg(r["motion_reference"]["q"][joint] - r["q"][joint]) for r in rows]
                ax[1].plot(t, e, label=f"J{joint + 1}")
            ax[1].set_ylabel("Reference error (deg)")
            ax[1].set_xlabel("Seconds since waypoint start")
            for a in ax:
                a.grid(alpha=0.25)
                a.legend(fontsize=8)
                a.axvline(duration, color="grey", ls=":")
            fig.suptitle(path.stem + " — native PD + gravity, I = 0; encoder estimates")
            fig.tight_layout()
            fig.savefig(root / f"{path.stem}-speed.png", dpi=150)
            plt.close(fig)
    finally:
        Path(model).unlink()
    (root / "joint-velocity-summary.json").write_text(json.dumps(records, indent=2))
    for record in records:
        print(json.dumps(record))


if __name__ == "__main__":
    tyro.cli(main)
