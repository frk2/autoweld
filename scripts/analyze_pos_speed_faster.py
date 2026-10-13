"""Compare recorded long-path feed trials; run only after hardware is disabled."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("evidence/pos-speed-faster")
    records = []
    traces = {}
    for path in sorted(root.glob("*.jsonl")):
        if path.stem.endswith("-control"):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows or "speed_m_s" not in rows[0]:
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        result = meta["result"]
        plan = result["plan"]
        t = np.array([r["elapsed_s"] for r in rows])
        fraction = np.array([r["edge_fraction"] for r in rows])
        raw = np.array([r["speed_m_s"] for r in rows]) * 1000
        raw_along = np.array([r["raw_along_speed_m_s"] for r in rows]) * 1000
        distance = np.array([r["cross_track_mm"] for r in rows])
        timed_error = np.array([r["tracking_error_mm"] for r in rows])
        slope = np.array([
            (r.get("position_velocity") or {}).get("along_speed_m_s", np.nan) for r in rows
        ]) * 1000
        feed = plan["feed_m_s"] * 1000
        edge = np.array([r["on_edge"] for r in rows])
        # Same elapsed settling allowance and fractional end exclusion at every feed.
        interior = edge & (t >= plan["ramp_s"] + 1) & (fraction <= .85)

        def stats(mask):
            values = raw[mask]
            along = slope[mask & np.isfinite(slope)]
            if not len(values):
                return None
            return {
                "samples": int(mask.sum()),
                "duration_s": float(t[mask][-1] - t[mask][0]),
                "max_edge_mm": float(max(distance[mask])),
                "mean_timed_error_over_feed_s": float(np.mean(timed_error[mask])/feed),
                "raw_norm_mean_mm_s": float(np.mean(values)),
                "raw_norm_p05_mm_s": float(np.percentile(values, 5)),
                "raw_norm_p95_mm_s": float(np.percentile(values, 95)),
                "raw_norm_in_band_fraction": float(np.mean(abs(values / feed - 1) <= .2)),
                "raw_norm_relative_rmse": float(np.sqrt(np.mean((values / feed - 1)**2))),
                "raw_along_mean_mm_s": float(np.mean(raw_along[mask])),
                "slope_along_mean_mm_s": float(np.mean(along)) if len(along) else None,
                "slope_along_in_band_fraction": float(np.mean(abs(along / feed - 1) <= .2)) if len(along) else None,
                "slope_along_relative_rmse": float(np.sqrt(np.mean((along / feed - 1)**2))) if len(along) else None,
            }

        record = {
            "name": path.stem, "feed_mm_s": feed,
            "length_mm": plan["edge_length_m"]*1000,
            "completed": result["completed"], "passed": result["passed"],
            "reason": result["reason"],
            "max_edge_mm": result["max_cross_track_mm"],
            "max_timed_error_mm": result["max_tracking_error_mm"],
            "last_edge_fraction": float(fraction[-1]),
            "last_elapsed_s": float(t[-1]),
            "edge": stats(edge), "interior": stats(interior),
        }
        records.append(record)
        traces[path.stem] = (t, raw, slope, feed, edge, interior, rows, plan)
    (root / "metrics.json").write_text(json.dumps(records, indent=2)+"\n")

    for index in (1, 3, 4, 5):
        selected = [(name, data) for name, data in traces.items() if name.startswith(f"line60-{index:02}-")]
        selected.sort(key=lambda item: item[1][3])
        if not selected:
            continue
        fig, axes = plt.subplots(len(selected), 2, figsize=(13, 3*len(selected)), squeeze=False)
        for ax, (name, (t, raw, slope, feed, edge, interior, rows, plan)) in zip(axes, selected):
            ax[0].plot(t, raw, lw=.7, alpha=.7, label="Raw Jacobian speed norm")
            ax[0].plot(t, slope, lw=1, label="120 ms position slope, along path (delayed)")
            ax[0].axhline(feed, color="black", ls="--", label="Requested feed")
            ax[0].fill_between(t[edge], feed*.8, feed*1.2, color="green", alpha=.15, label="±20% on edge")
            if interior.any():
                ax[0].axvspan(t[interior][0], t[interior][-1], color="orange", alpha=.08, label="Interior analysis")
            ax[0].set_ylabel("mm/s")
            ax[1].plot(t, [r["cross_track_mm"] for r in rows], label="Finite edge distance")
            ax[1].plot(t, [r["tracking_error_mm"] for r in rows], alpha=.6, label="Timed error (informational)")
            ax[1].axhline(5, color="red", ls="--", label="5 mm edge limit")
            ax[1].set_ylabel("mm")
            for a in ax:
                a.set_title(name)
                a.set_xlabel("Seconds since reference start")
                a.grid(alpha=.2)
                a.legend(fontsize=7)
        fig.suptitle("Longer position–speed paths — raw qualification and separate slope diagnostics")
        fig.tight_layout()
        fig.savefig(root / f"line60-{index:02}-comparison.png", dpi=150)
        plt.close(fig)
    for r in records:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
