"""Nine running-average windows, 0.2 to 1.0 s, for recorded straight-line trials."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from scripts.audit_velocity import load, replay
from scripts.plot_velocity_windows import invalid_intervals, running_velocity

WINDOWS = tuple(i / 10 for i in range(2, 11))


def panel(ax, data, window, legend=False):
    avg = data["averages"][window]
    t, _, _, raw, *_ = data["series"]
    plan = data["series"][-1]
    ax.plot(t, raw * 1000, color=".72", lw=0.6, alpha=0.4, label="Raw reported velocity")
    ax.plot(
        avg["center_s"],
        avg["position_mm_s"],
        color="#0072b2",
        lw=1.8,
        label="Measured running avg (position)",
    )
    ax.plot(avg["center_s"], avg["command_mm_s"], "k--", lw=1.2, label="Command avg — same window")
    ax.fill_between(
        avg["center_s"],
        avg["command_mm_s"] * 0.8,
        avg["command_mm_s"] * 1.2,
        color="green",
        alpha=0.1,
        label="±20% of command avg",
    )
    ax.axvspan(plan["ramp_s"], plan["duration_s"] - plan["ramp_s"], color="green", alpha=0.05)
    info = data["metrics"]["windows"][str(window)]
    last = info["final_cruise_avg_mm_s"]
    if last is not None:
        mid = plan["duration_s"] - plan["ramp_s"] - window / 2
        ax.plot(mid, last, "o", color="#0072b2", ms=4)
        label = f"Final cruise avg: {last:.1f} mm/s"
    elif plan["constant_feed_duration_s"] < window:
        i = int(np.nanargmax(avg["position_mm_s"]))
        ax.plot(avg["center_s"][i], avg["position_mm_s"][i], "D", color="#0072b2", ms=4)
        label = f"No full cruise window\nPeak avg: {avg['position_mm_s'][i]:.1f} mm/s"
    elif t[-1] < plan["duration_s"] - plan["ramp_s"]:
        recent = info["last_observed_cruise_avg_mm_s"]
        label = "Run interrupted before cruise end"
        if recent is not None:
            mid = info["last_observed_cruise_window_end_s"] - window / 2
            ax.plot(mid, recent, "s", color="#d55e00", ms=4)
            label += f"\nLast observed avg: {recent:.1f} mm/s"
    else:
        label = "Final window rejected: feedback timing gap"
    ax.text(
        0.98,
        0.04,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
    )
    ax.set_title(f"{window:.1f} s window · available {window / 2:.2f} s after midpoint", fontsize=10)
    ax.set_xlabel("Window midpoint time (s)", fontsize=9)
    ax.set_ylabel("Along-path mm/s", fontsize=9)
    ax.grid(alpha=0.18)
    if legend:
        ax.legend(fontsize=7)


def main(source="evidence/pos-speed-100", dest="evidence/velocity-nine-windows"):
    root = Path(source)
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    model = tool_model()
    planner = MotionPlanner(model)
    records = []
    traces = {}
    try:
        for path in sorted(root.glob("*mms.jsonl")):
            if not path.with_suffix(".json").exists():
                continue
            rows, plan = load(path)
            result = json.loads(path.with_suffix(".json").read_text())["result"]
            if "joint_position_rad" not in rows[0]:
                continue
            _, series = replay(rows, plan, planner)
            t, x, xr, raw, *_ = series
            bad = invalid_intervals(rows)
            averages = {w: running_velocity(t, x, raw, xr, w, invalid=bad) for w in WINDOWS}
            metric = {
                "name": path.stem,
                "feed_mm_s": plan["feed_m_s"] * 1000,
                "length_mm": plan["edge_length_m"] * 1000,
                "cruise_s": plan["constant_feed_duration_s"],
                "completed": result["completed"],
                "stop_reason": result.get("reason"),
                "observed_end_s": float(t[-1]),
                "windows": {},
            }
            a, b = plan["ramp_s"], plan["duration_s"] - plan["ramp_s"]
            for w, avg in averages.items():
                last = None
                if b - a >= w and b <= t[-1]:
                    val = running_velocity(t, x, raw, xr, w, ends=[b], invalid=bad)
                    if val["valid"][0]:
                        last = float(val["position_mm_s"][0])
                mask = avg["valid"] & (avg["end_s"] - w >= a) & (avg["end_s"] <= b)
                recent = int(np.flatnonzero(mask)[-1]) if mask.any() else None
                metric["windows"][str(w)] = {
                    "final_cruise_avg_mm_s": last,
                    "last_observed_cruise_avg_mm_s": float(avg["position_mm_s"][recent])
                    if recent is not None else None,
                    "last_observed_cruise_command_mm_s": float(avg["command_mm_s"][recent])
                    if recent is not None else None,
                    "last_observed_cruise_window_end_s": float(avg["end_s"][recent])
                    if recent is not None else None,
                    "peak_whole_trace_avg_mm_s": float(np.nanmax(avg["position_mm_s"])),
                    "valid_cruise_windows": int(mask.sum()),
                    "cruise_windows_in_20_percent": float(
                        np.mean(abs(avg["position_mm_s"][mask] / metric["feed_mm_s"] - 1) <= 0.2)
                    )
                    if mask.any()
                    else None,
                }
            data = {"series": series, "averages": averages, "metrics": metric}
            traces[path.stem] = data
            records.append(metric)
            fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=True)
            for ax, w in zip(axes.flat, WINDOWS):
                panel(ax, data, w)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, fontsize=9)
            fig.suptitle(
                f"{path.stem} · {metric['length_mm']:.0f} mm path · planned cruise {metric['cruise_s']:.3f} s"
                f" · {'Completed' if metric['completed'] else 'INTERRUPTED'}"
                "\nNine averaging windows; measured and commanded velocity use identical time intervals",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            fig.savefig(out / f"{path.stem}-windows.png", dpi=140)
            plt.close(fig)
        (out / "metrics.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        # One compact figure for every recorded speed, across directions.
        feeds = sorted({r["feed_mm_s"] for r in records})
        fig, axes = plt.subplots(len(feeds), 1, figsize=(12, 2.8 * len(feeds)), squeeze=False)
        for row, feed in zip(axes, feeds):
            ax = row[0]
            ax.axhline(feed, color="black", ls="--", label="Requested feed")
            ax.axhspan(feed * 0.8, feed * 1.2, color="green", alpha=0.08, label="±20% requested feed")
            for r in records:
                if r["feed_mm_s"] != feed:
                    continue
                y = [r["windows"][str(w)]["final_cruise_avg_mm_s"] for w in WINDOWS]
                y = [np.nan if v is None else v for v in y]
                ax.plot(WINDOWS, y, "o-", label=r["name"].rsplit("-", 1)[0])
                if not r["completed"]:
                    y = [r["windows"][str(w)]["last_observed_cruise_avg_mm_s"] for w in WINDOWS]
                    ax.plot(WINDOWS, y, "s--", color="#d55e00", label="Last observed before Hold; interrupted")
            if not any(
                r["windows"][str(w)]["final_cruise_avg_mm_s"] is not None
                for r in records
                if r["feed_mm_s"] == feed
                for w in WINDOWS
            ):
                ax.text(0.5, 0.1, "Planned cruise end was not observed", transform=ax.transAxes, ha="center")
            ax.set_title(f"{feed:g} mm/s requested — final complete cruise window")
            ax.set_ylabel("Running avg mm/s")
            ax.set_xticks(WINDOWS)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=3)
        axes[-1, 0].set_xlabel(
            "Averaging window (s); missing values have insufficient cruise or invalid timing"
        )
        fig.tight_layout()
        fig.savefig(out / "window-comparison.png", dpi=140)
        plt.close(fig)
        print(f"{len(records)} trials, {len(WINDOWS)} windows each. Output: {out}", flush=True)
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="evidence/pos-speed-100")
    parser.add_argument("--dest", default="evidence/velocity-nine-windows")
    args = parser.parse_args()
    main(args.source, args.dest)
