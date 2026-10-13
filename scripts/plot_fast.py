"""Render recorded PI convergence using the requested per-joint tolerances."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path("evidence/fast")
    trials = sorted(
        (json.loads(p.read_text()) for p in Path("evidence/pid").glob("fast-*.json")),
        key=lambda s: s["start"],
    )
    (out / "trials.json").write_text(json.dumps(trials, indent=2))
    selection = out / "selected-profile.json"
    names = json.loads(selection.read_text())["plot_trials"] if selection.exists() else []
    group = [s for s in trials if s["name"] in names]
    if group:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
        for j, ax in enumerate(axes.flat):
            tol = group[0]["accuracy_tolerance_deg"][j]
            ax.axhspan(-tol, tol, color="#51ad7e", alpha=0.18)
            ax.axvline(1, color="#a03d37", ls="--", lw=1)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set(
                title=f"Joint {j + 1} · ±{tol:g}°",
                xlabel="Seconds after command",
                ylabel="Target − measured (°)",
            )
            ax.grid(alpha=0.15)
        for trial in group:
            raw = [
                json.loads(r) for r in Path(f"evidence/pid/{trial['name']}.jsonl").read_text().splitlines()
            ]
            t = np.array([r["timestamp"] - trial["start"] for r in raw])
            error = np.rad2deg(np.array(trial["target"]) - np.array([r["q"] for r in raw]))
            for j, ax in enumerate(axes.flat):
                ax.plot(t, error[:, j], label=trial["name"].removeprefix("fast-"), lw=1.2)
        axes.flat[0].legend(fontsize=8)
        for view, end in (("convergence", 3), ("hold", max(s["duration"] for s in group))):
            for ax in axes.flat:
                ax.set_xlim(0, end)
            fig.suptitle(
                "Live small-angle PI validation\nGreen: requested accuracy · Dashed line: 1-second goal · Factory D unchanged"
            )
            fig.savefig(out / f"{view}.png", dpi=160)
            fig.savefig(out / f"{view}.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(trials) * 0.4), 8), layout="constrained")
    names = [s["name"].removeprefix("fast-") for s in trials]
    for i, s in enumerate(trials):
        settle = s["settle_and_remain_s"]
        axes[0].bar(
            i,
            settle if settle is not None else s["duration"],
            color="#268567" if s["converged_within_1s"] else "#a8afb7",
            hatch=None if settle is not None else "//",
        )
        axes[1].bar(i, max(s["overshoot_deg"]), color="#4577a9")
    axes[0].axhline(1, color="#a03d37", ls="--")
    axes[0].set(
        ylabel="Sustained settling (seconds)",
        title="All six joints: 0.2° on J1–J3, 1° on J4–J6; hatched = did not settle",
    )
    axes[1].set(ylabel="Worst overshoot (degrees)")
    for ax in axes:
        ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    print(f"Plotted {len(trials)} trials")


if __name__ == "__main__":
    main()
