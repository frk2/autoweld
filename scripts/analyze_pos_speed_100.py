"""Analyze recorded feed sweep; no hardware access. Plot only after disable."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("evidence/pos-speed-100")
    records, traces = [], {}
    for path in sorted(root.glob("line120-*.jsonl")):
        if path.stem.endswith("-control"):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows or "speed_m_s" not in rows[0]:
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        result = meta["result"]
        plan = result["plan"]
        t = np.array([r["elapsed_s"] for r in rows])
        raw = np.array([r["speed_m_s"] for r in rows])*1000
        slope = np.array([(r.get("position_velocity") or {}).get("along_speed_m_s", np.nan)
                          for r in rows])*1000
        edge = np.array([r["on_edge"] for r in rows])
        feed = plan["feed_m_s"]*1000
        cruise = plan["constant_feed_duration_s"]
        late = edge & (t >= plan["ramp_s"] + cruise/2)

        def stats(mask):
            if not mask.any():
                return None
            a = raw[mask]
            b = slope[mask & np.isfinite(slope)]
            return {
                "samples": int(mask.sum()),
                "duration_s": float(t[mask][-1]-t[mask][0]),
                "raw_mean_mm_s": float(np.mean(a)),
                "raw_in_band_fraction": float(np.mean(abs(a/feed-1) <= .2)),
                "raw_relative_rmse": float(np.sqrt(np.mean((a/feed-1)**2))),
                "slope_along_mean_mm_s": float(np.mean(b)) if len(b) else None,
                "slope_along_in_band_fraction": float(np.mean(abs(b/feed-1) <= .2)) if len(b) else None,
            }

        sdk = [r["sdk_timing"] for r in rows if r.get("sdk_timing")]
        record = {
            "name": path.stem, "index": meta["index"], "feed_mm_s": feed,
            "joint_ceiling_rad_s": meta["start"]["position_speed_limit_rad_s"],
            "reference_duration_s": plan["duration_s"],
            "ramp_s": plan["ramp_s"], "cruise_duration_s": cruise,
            "completed": result["completed"], "passed": result["passed"], "reason": result["reason"],
            "max_edge_mm": result["max_cross_track_mm"],
            "max_timed_error_mm": result["max_tracking_error_mm"],
            "raw_peak_mm_s": float(max(raw)),
            "slope_along_peak_mm_s": float(np.nanmax(slope)),
            "constant_feed": stats(edge), "late_constant_feed": stats(late),
            "max_sdk_cycle_ms": max((r["cycle_period_s"]*1000 for r in sdk), default=None),
        }
        records.append(record)
        traces[path.stem] = (t, raw, slope, feed, edge, rows, plan)
    (root/"metrics.json").write_text(json.dumps(records, indent=2)+"\n")

    for index in (1, 3, 4):
        selected = [(n,d) for n,d in traces.items() if n.startswith(f"line120-{index:02}-")]
        selected.sort(key=lambda item: item[1][3])
        if not selected:
            continue
        fig, axes = plt.subplots(len(selected), 2, figsize=(13, 2.8*len(selected)), squeeze=False)
        for ax,(name,(t,raw,slope,feed,edge,rows,plan)) in zip(axes,selected):
            x = np.clip(np.minimum(t, plan["duration_s"]-t)/plan["ramp_s"], 0, 1)
            reference = feed*(10*x**3-15*x**4+6*x**5)
            ax[0].plot(t,raw, lw=.8,alpha=.65,label="Raw Jacobian speed norm")
            ax[0].plot(t,slope,lw=1,label="120 ms along-path slope (delayed)")
            ax[0].plot(t,reference,"k--",label="Reference")
            ax[0].fill_between(t[edge],feed*.8,feed*1.2,color="green",alpha=.15,label="±20% cruise band")
            ax[0].set_ylabel("mm/s")
            ax[1].plot(t,[r["cross_track_mm"] for r in rows],label="Finite edge distance")
            ax[1].plot(t,[r["tracking_error_mm"] for r in rows],alpha=.65,label="Timed error (informational)")
            ax[1].axhline(5,color="red",ls="--",label="5 mm edge limit")
            ax[1].set_ylabel("mm")
            for a in ax:
                a.set_title(name)
                a.set_xlabel("Seconds")
                a.grid(alpha=.2)
                a.legend(fontsize=7)
        fig.suptitle("120 mm lines, ramps inside finite edge — encoder estimates")
        fig.tight_layout(rect=(0, 0, 1, .975))
        fig.savefig(root/f"line120-{index:02}-sweep.png",dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for index,label in ((0,"Y"),(2,"Z"),(3,"Y/Z diagonal")):
        data = sorted([r for r in records if r["index"]==index],key=lambda r:r["feed_mm_s"])
        if not data:
            continue
        x=[r["feed_mm_s"] for r in data]
        axes[0].plot(x,[r["slope_along_peak_mm_s"] for r in data],"o-",label=label)
        axes[1].plot(x,[100*r["constant_feed"]["raw_in_band_fraction"] if r['constant_feed'] else 0 for r in data],"o-",label=label)
        axes[2].plot(x,[r["max_edge_mm"] for r in data],"o-",label=label)
    axes[0].plot([20,100],[20,100],"k--",label="Requested feed")
    axes[0].set_ylabel("Peak along-path position slope (mm/s)")
    axes[1].set_ylabel("Raw cruise samples in ±20% band (%)")
    axes[2].axhline(5,color='red',ls='--',label='5 mm limit')
    axes[2].set_ylabel("Maximum finite-edge distance (mm)")
    for a in axes:
        a.set_xlabel("Requested feed (mm/s)")
        a.grid(alpha=.2)
        a.legend(fontsize=8)
    fig.suptitle("Position–speed sweep: short high-speed intervals do not establish sustained tracking")
    fig.tight_layout()
    fig.savefig(root/"sweep-summary.png",dpi=160)
    plt.close(fig)
    for r in records:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
