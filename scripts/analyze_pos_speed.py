"""Analyze recorded POS_VEL trials only; run after disabling hardware."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner


def main():
    root = Path("evidence/pos-speed")
    model = tool_model()
    planner = MotionPlanner(model)
    metrics = []
    try:
        for path in sorted(root.glob("s[12]-*.jsonl")):
            meta = json.loads(path.with_suffix(".json").read_text())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = [r for r in rows if r.get("motion_reference")]
            if not rows:
                continue
            session = rows[-1]["motion_reference"]["session"]
            rows = [r for r in rows if r["motion_reference"]["session"] == session]
            t = np.array([r["motion_reference"]["elapsed_s"] for r in rows])
            q = np.array([r["q"] for r in rows])
            target = np.array(rows[-1]["target"])
            error = np.rad2deg(q - target)
            reference = np.array([r["motion_reference"]["q"][:6] for r in rows])
            direction = np.sign(target - np.array(meta["before"]["q_rad"]))
            overshoot = np.maximum(0, np.max(error * direction, axis=0))
            bad = np.flatnonzero(np.any(abs(error) > [0.2, 0.2, 0.2, 1, 1, 1], axis=1))
            settled = bad[-1] + 1 if len(bad) else 0
            settle_time = float(t[settled]) if settled < len(t) and t[-1] - t[settled] >= 1 else None
            joint_v = np.array([r["velocity"] for r in rows])
            tcp_v = np.array([planner.jacobian(r["q"]) @ np.array(r["velocity"]) for r in rows])
            speed = np.linalg.norm(tcp_v[:, :3], axis=1)
            ref_duration = rows[-1]["motion_reference"]["duration_s"]
            result = dict(
                name=path.stem,
                samples=len(rows),
                duration_s=ref_duration,
                settled_from_motion_start_s=settle_time,
                final_error_deg=error[-1].tolist(),
                max_directed_overshoot_deg=float(max(overshoot)),
                directed_overshoot_deg=overshoot.tolist(),
                joint_peak_speed_rad_s=np.max(abs(joint_v), axis=0).tolist(),
                tcp_peak_speed_m_s=float(max(speed)),
                reference_report=meta["after"]["motion_plan"],
                pass_stage1=bool(settle_time is not None and settle_time < 1 and max(overshoot) < 0.1),
            )
            metrics.append(result)
            fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
            for j, a in enumerate(ax.flat):
                a.plot(t, error[:, j], label="Measured", lw=1)
                a.plot(t, np.rad2deg(reference[:, j] - target[j]), label="Reference", ls="--", lw=1)
                a.axhline(0, color="black", lw=0.5)
                a.axvline(ref_duration, color="gray", ls=":")
                tol = 0.2 if j < 3 else 1
                a.axhspan(-tol, tol, alpha=0.06, color="green")
                a.set_title(f"J{j + 1}")
                a.set_ylabel("Error (°)")
                a.grid(alpha=0.2)
            ax[0, 0].legend()
            ax[-1, 0].set_xlabel("Seconds")
            ax[-1, 1].set_xlabel("Seconds")
            fig.suptitle(path.stem + " — position-speed firmware, original onboard gains")
            fig.tight_layout()
            fig.savefig(root / f"{path.stem}.png", dpi=140)
            plt.close(fig)
            if path.stem.startswith("s2-"):
                # Differentiation of raw quantized feedback is diagnostic, not metrology.
                keep = np.r_[True, np.diff(t) > 1e-6]
                tt = t[keep]
                vv = tcp_v[keep, :3]
                aa = np.gradient(vv, tt, axis=0)
                jj = np.gradient(aa, tt, axis=0)
                quantities = [
                    np.linalg.norm(vv, axis=1),
                    np.linalg.norm(aa, axis=1),
                    np.linalg.norm(jj, axis=1),
                ]
                result["raw_tcp_acceleration_estimate_peak_m_s2"] = float(max(quantities[1]))
                result["raw_tcp_jerk_estimate_peak_m_s3"] = float(max(quantities[2]))
                fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                for i, (a, values, unit, key) in enumerate(
                    zip(
                        axes,
                        quantities,
                        ["m/s", "m/s²", "m/s³"],
                        ["linear_velocity", "linear_acceleration", "linear_jerk"],
                    )
                ):
                    a.plot(tt, values, lw=0.8, label="Raw feedback/model estimate")
                    a.axhline(
                        result["reference_report"]["limits"][key],
                        color="red",
                        ls="--",
                        label="Requested reference limit",
                    )
                    a.set_ylabel(unit)
                    a.grid(alpha=0.2)
                    a.legend(fontsize=8)
                axes[-1].set_xlabel("Seconds from reference start")
                fig.suptitle(
                    path.stem + " — acceleration/jerk differentiation amplifies feedback quantization"
                )
                fig.tight_layout()
                fig.savefig(root / f"{path.stem}-derivatives.png", dpi=140)
                plt.close(fig)
    finally:
        Path(model).unlink()
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pairs = [m for m in metrics if m["name"].startswith("s1-j")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x = np.arange(len(pairs))
    axes[0].bar(x, [m["settled_from_motion_start_s"] or 0 for m in pairs], color="#277da1")
    axes[0].axhline(1, color="red", ls="--")
    axes[0].set_ylabel("Settling (s)")
    axes[1].bar(x, [m["max_directed_overshoot_deg"] for m in pairs], color="#43aa8b")
    axes[1].axhline(0.1, color="red", ls="--")
    axes[1].set_ylabel("Directed overshoot (°)")
    axes[1].set_xticks(x, [m["name"].removeprefix("s1-") for m in pairs], rotation=45, ha="right")
    fig.suptitle("Position–speed free-space joint pairs — no onboard gain changes")
    fig.tight_layout()
    fig.savefig(root / "joint-summary.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            [
                {
                    k: m[k]
                    for k in (
                        "name",
                        "settled_from_motion_start_s",
                        "max_directed_overshoot_deg",
                        "tcp_peak_speed_m_s",
                    )
                }
                for m in metrics
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
