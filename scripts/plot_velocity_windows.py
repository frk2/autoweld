"""Offline, time-weighted running velocity averages from recorded encoder data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from scripts.audit_velocity import ROOT, load, replay

DEST = Path("evidence/velocity-windows")
WINDOWS = (0.12, 0.5, 1.0)
COLORS = {0.12: "#9098a3", 0.5: "#0072b2", 1.0: "#d55e00"}


def integral_at(t, y, at):
    """Exact integral of the piecewise-linear signal, including partial intervals."""
    t, y, at = np.asarray(t, float), np.asarray(y, float), np.asarray(at, float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Need increasing times and matching scalar observations")
    if not np.isfinite(np.r_[t, y, at]).all() or np.any(at < t[0]) or np.any(at > t[-1]):
        raise ValueError("Finite in-range evaluation times required; no extrapolation")
    dt = np.diff(t)
    cumulative = np.r_[0.0, np.cumsum((y[:-1] + y[1:]) * 0.5 * dt)]
    i = np.minimum(np.searchsorted(t, at, side="right") - 1, len(t) - 2)
    h = at - t[i]
    return cumulative[i] + y[i] * h + 0.5 * (y[i + 1] - y[i]) / dt[i] * h * h


def running_velocity(t, position, raw, reference, window, *, ends=None, invalid=()):
    """Full trailing windows; plotted at midpoint; gaps never silently averaged."""
    t = np.asarray(t, float)
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Window must be positive")
    ends = t[t >= t[0] + window] if ends is None else np.asarray(ends, float)
    starts = ends - window
    if np.any(starts < t[0] - 1e-10) or np.any(ends > t[-1] + 1e-10):
        raise ValueError("Full observed windows required")
    starts = np.maximum(starts, t[0])
    ends = np.minimum(ends, t[-1])
    measured = (np.interp(ends, t, position) - np.interp(starts, t, position)) / window
    command = (np.interp(ends, t, reference) - np.interp(starts, t, reference)) / window
    raw_mean = (integral_at(t, raw, ends) - integral_at(t, raw, starts)) / window
    valid = np.ones(len(ends), dtype=bool)
    for a, b in invalid:
        valid &= ~((starts < b) & (ends > a))
    for a in (measured, command, raw_mean):
        a[~valid] = np.nan
    return {
        "center_s": ends - window / 2,
        "end_s": ends,
        "valid": valid,
        "position_mm_s": measured * 1000,
        "raw_mm_s": raw_mean * 1000,
        "command_mm_s": command * 1000,
    }


def invalid_intervals(rows):
    ts = np.array([r["feedback_timestamps"] for r in rows])
    epoch = ts[0, 0]
    relative = ts - epoch
    elapsed = np.array([r["elapsed_s"] for r in rows])
    offset = np.median(elapsed[elapsed > 0] - relative[elapsed > 0, 0])
    good = np.array([r["feedback_age_s"] <= 0.05 and np.ptp(r["feedback_timestamps"]) <= 0.01 for r in rows])
    intervals = []
    for i in range(1, len(rows)):
        dt = ts[i] - ts[i - 1]
        if not (good[i] and good[i - 1] and np.all(dt > 0) and np.max(dt) <= 0.05):
            intervals.append(
                (
                    float(relative[max(0, i - 2), 0] + offset),
                    float(relative[min(len(rows) - 1, i + 1), -1] + offset),
                )
            )
    return intervals


def draw(ax, data, *, raw=True, legend=True):
    t, x, xr, rv, _, fit, plan = data["series"]
    f = plan["feed_m_s"] * 1000
    phase = np.clip(np.minimum(t, plan["duration_s"] - t) / plan["ramp_s"], 0, 1)
    command = f * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    if raw:
        ax.plot(
            t,
            rv * 1000,
            color="#b3b9c1",
            lw=0.7,
            alpha=0.55,
            label="Raw reported velocity (signed along path)",
        )
    ax.plot(t, command, "k--", lw=1.2, label="Instantaneous reference")
    for w in WINDOWS:
        a = data["averages"][w]
        ax.plot(
            a["center_s"],
            a["position_mm_s"],
            color=COLORS[w],
            lw=2 if w == 0.5 else 1.5,
            alpha=0.65 if w == 0.12 else 1,
            label=f"{w:g} s running avg — encoder displacement",
        )
    ax.axvspan(plan["ramp_s"], plan["duration_s"] - plan["ramp_s"], color="green", alpha=0.07)
    ax.axhline(f * 0.8, color="#777777", lw=0.5, ls=":")
    ax.axhline(f * 1.2, color="#777777", lw=0.5, ls=":")
    for w in (0.5, 1.0):
        last = data["record"]["windows"][str(w)]["last_full_cruise_window"]
        label = f"{w:g}s avg"
        marker = "o"
        if last is None and plan["constant_feed_duration_s"] < w:
            avg = data["averages"][w]
            if np.isfinite(avg["position_mm_s"]).any():
                i = int(np.nanargmax(avg["position_mm_s"]))
                last = {k: float(avg[k][i]) for k in ["center_s", "position_mm_s"]}
                label = f"{w:g}s peak avg"
                marker = "D"
        if last:
            ax.plot(last["center_s"], last["position_mm_s"], marker, color=COLORS[w], ms=5)
            ax.annotate(
                f"{label}: {last['position_mm_s']:.1f}",
                (last["center_s"], last["position_mm_s"]),
                xytext=(-4, 18 if w == 0.5 else -23),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=COLORS[w],
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
            )
    ax.set_ylabel("Along-path velocity (mm/s)")
    ax.grid(alpha=0.18)
    if legend:
        ax.legend(fontsize=8, loc="upper left", ncol=2)


def main():
    DEST.mkdir(exist_ok=True)
    model = tool_model()
    planner = MotionPlanner(model)
    records = []
    all_data = {}
    try:
        for path in sorted(ROOT.glob("line120-*.jsonl")):
            if path.stem.endswith("-control"):
                continue
            rows, plan = load(path)
            _, series = replay(rows, plan, planner)
            t, x, xr, raw, _, _, _ = series
            invalid = invalid_intervals(rows)
            averages = {w: running_velocity(t, x, raw, xr, w, invalid=invalid) for w in WINDOWS}
            record = {
                "name": path.stem,
                "feed_mm_s": plan["feed_m_s"] * 1000,
                "cruise_duration_s": plan["constant_feed_duration_s"],
                "windows": {},
            }
            a, b = plan["ramp_s"], plan["duration_s"] - plan["ramp_s"]
            for w, result in averages.items():
                inside = (result["end_s"] - w >= a) & (result["end_s"] <= b) & result["valid"]
                error = result["position_mm_s"][inside] - record["feed_mm_s"]
                last = None
                if b - a >= w:
                    exact = running_velocity(t, x, raw, xr, w, ends=[b], invalid=invalid)
                    if exact["valid"][0]:
                        last = {
                            k: float(exact[k][0])
                            for k in ["center_s", "position_mm_s", "raw_mm_s", "command_mm_s"]
                        }
                record["windows"][str(w)] = {
                    "full_cruise_window_count": int(inside.sum()),
                    "cruise_in_20_percent_fraction": float(np.mean(abs(error) <= record["feed_mm_s"] * 0.2))
                    if len(error)
                    else None,
                    "last_full_cruise_window": last,
                    "note": None
                    if b - a >= w
                    else "Reference cruise shorter than this window; averages mix ramps",
                    "peak_running_avg_mm_s": float(np.nanmax(result["position_mm_s"])),
                }
            data = {"record": record, "series": series, "averages": averages}
            all_data[path.stem] = data
            records.append(record)
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            draw(axes[0], data)
            for ax, w in zip(axes[1:], (0.5, 1.0)):
                avg = averages[w]
                ax.plot(
                    avg["center_s"],
                    avg["position_mm_s"],
                    color=COLORS[w],
                    lw=2,
                    label=f"{w:g} s measured running avg (position)",
                )
                ax.plot(
                    avg["center_s"],
                    avg["command_mm_s"],
                    "k--",
                    label=f"{w:g} s command running avg — identical interval",
                )
                ax.plot(
                    avg["center_s"],
                    avg["raw_mm_s"],
                    color="#777777",
                    ls=":",
                    label=f"{w:g} s raw-velocity running avg (bias retained)",
                )
                ax.fill_between(
                    avg["center_s"],
                    avg["command_mm_s"] * 0.8,
                    avg["command_mm_s"] * 1.2,
                    color="green",
                    alpha=0.1,
                    label="±20% of averaged command",
                )
                ax.axvspan(a, b, color="green", alpha=0.04)
                ax.set_ylabel("mm/s")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8, loc="upper left", ncol=2)
                note = record["windows"][str(w)]["note"]
                if note:
                    ax.text(
                        0.98,
                        0.08,
                        note,
                        transform=ax.transAxes,
                        ha="right",
                        fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
                    )
            axes[-1].set_xlabel("Time at window midpoint (s); estimate available half a window later")
            fig.suptitle(
                f"{path.stem}: larger velocity windows\nGreen: reference cruise; circles: final cruise average; diamonds: whole-trace peak average"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(DEST / f"{path.stem}.png", dpi=150)
            plt.close(fig)
        (DEST / "metrics.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        for index, label in [(1, "Y"), (3, "Z"), (4, "Y/Z diagonal")]:
            data = sorted(
                [d for name, d in all_data.items() if name.startswith(f"line120-{index:02}-")],
                key=lambda d: d["record"]["feed_mm_s"],
            )
            fig, axes = plt.subplots(len(data), 1, figsize=(12, 2.9 * len(data)))
            for ax, d in zip(axes, data):
                draw(ax, d, legend=False)
                ax.set_title(
                    f"{d['record']['feed_mm_s']:g} mm/s requested; cruise {d['record']['cruise_duration_s']:.3f} s",
                    fontsize=11,
                )
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97), fontsize=9)
            fig.suptitle(f"{label}: 0.12 s, 0.5 s and 1 s running velocity averages", fontsize=14)
            axes[-1].set_xlabel("Window midpoint time (s); raw/reference at their own times")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(DEST / f"line120-{index:02}-sweep.png", dpi=140)
            plt.close(fig)
        selected = ["line120-03-20mms", "line120-03-40mms", "line120-03-100mms"]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for ax, name in zip(axes, selected):
            d = all_data[name]
            draw(ax, d, legend=False)
            ax.set_title(
                f"Z line, {d['record']['feed_mm_s']:g} mm/s requested; cruise {d['record']['cruise_duration_s']:.3f} s"
            )
            ax.set_xlabel("Window midpoint time (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96), fontsize=9)
        fig.suptitle("Running average velocity: noise falls, tracking lag remains", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(DEST / "comparison.png", dpi=150)
        plt.close(fig)
        for r in records:
            print(
                r["name"], {w: d["last_full_cruise_window"] for w, d in r["windows"].items() if w != "0.12"}
            )
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    main()
