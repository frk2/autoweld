"""Plot the recorded small-angle PI comparisons without accessing the robot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path("evidence/pi")
    trials = sorted(
        (json.loads(p.read_text()) for p in Path("evidence/pid").glob("pi-*.json")),
        key=lambda x: x["start"],
    )
    groups = [
        (suffix, [r for r in trials if r["name"].endswith("-" + suffix)])
        for suffix in ("out", "return", "reverse")
    ]
    selection = out / "selected-profile.json"
    if selection.exists():
        names = json.loads(selection.read_text())["plot_trials"]
        groups.append(("selected", [r for r in trials if r["name"] in names]))
    for suffix, group in groups:
        if not group:
            continue
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout="constrained")
        for a in axes.flat:
            a.axhspan(-0.1, 0.1, color="#87c9a4", alpha=0.25)
            a.axhline(0, color="#75808d", lw=0.7)
            a.grid(alpha=0.15)
        for trial in group:
            raw = [
                json.loads(s) for s in Path(f"evidence/pid/{trial['name']}.jsonl").read_text().splitlines()
            ]
            t = np.array([r["timestamp"] - trial["start"] for r in raw])
            error = np.rad2deg(np.array(trial["target"]) - np.array([r["q"] for r in raw]))
            for j, a in enumerate(axes.flat):
                a.plot(t, error[:, j], lw=1.1, label=trial["name"].removeprefix("pi-"))
                a.set_title(f"Joint {j + 1}")
                a.set_ylabel("Target − measured (degrees)")
                a.set_xlabel("Seconds since waypoint")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Small-angle PI convergence: {suffix}\nD fixed; deadband 0.06°; shaded band ±0.1°")
        fig.savefig(out / f"convergence-{suffix}.png", dpi=160)
        fig.savefig(out / f"convergence-{suffix}.pdf")
        if suffix == "selected":
            for a in axes.flat:
                a.set_ylim(-0.35, 0.35)
                a.set_xlim(0.5, max(r["duration"] for r in group))
            fig.suptitle(
                "Selected PI settling detail (initial approach clipped)\nD fixed; deadband 0.06°; shaded band ±0.1°"
            )
            fig.savefig(out / "settling-selected.png", dpi=160)
            fig.savefig(out / "settling-selected.pdf")
        plt.close(fig)
    pairs = {}
    for trial in trials:
        prefix, direction = trial["name"].rsplit("-", 1)
        if direction == "reverse":
            pairs.setdefault(trial["name"], {})["out"] = trial
        elif direction in ("out", "return"):
            pairs.setdefault(prefix, {})[direction] = trial
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout="constrained")
    names = [name.removeprefix("pi-") for name in pairs]
    x = np.arange(len(pairs))
    axes[0].bar(
        x, [max(max(r["overshoot_deg"]) for r in pair.values()) for pair in pairs.values()], color="#4477aa"
    )
    axes[0].set_ylabel("Worst joint overshoot (degrees)")
    axes[0].set_title("Worst recorded overshoot across each outbound / return pair")
    for i, pair in enumerate(pairs.values()):
        passed = len(pair) == 2 and all(
            r["settle_and_remain_at_0_1_deg_s"] is not None for r in pair.values()
        )
        duration = max(r["duration"] for r in pair.values())
        value = max(r["settle_and_remain_at_0_1_deg_s"] for r in pair.values()) if passed else duration
        axes[1].bar(i, value, color="#228855" if passed else "#aaaaaa", hatch=None if passed else "//")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Slower six-joint settling of the pair; hatched = incomplete pair or failed in window")
    for a in axes:
        a.set_xticks(x, names, rotation=35, ha="right", fontsize=9)
        a.grid(axis="y", alpha=0.15)
    fig.suptitle("PI comparison — overshoot and sustained ±0.1° convergence")
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    (out / "trials.json").write_text(json.dumps(trials, indent=2))
    print(f"Plotted {len(trials)} trials in {out}")


if __name__ == "__main__":
    main()
