"""Plot recorded low-overshoot PI trials without accessing hardware."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(prefix="os-", output="evidence/overshoot", title="Low-overshoot PI tuning · factory D fixed"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    trials = sorted(
        (r for p in Path("evidence/pid").glob(f"{prefix}*.json") if not (r := json.loads(p.read_text())).get("aborted")), key=lambda r: r["start"]
    )
    (out / "trials.json").write_text(json.dumps(trials, indent=2))
    selection = out / "selected-profile.json"
    names = (
        json.loads(selection.read_text())["plot_trials"]
        if selection.exists()
        else [r["name"] for r in trials[-3:]]
    )
    group = [r for r in trials if r["name"] in names]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for j, ax in enumerate(axes.flat):
        tol = group[0]["accuracy_tolerance_deg"][j]
        ax.axhspan(-0.1, tol, color="#57ab80", alpha=0.16)
        ax.axhline(-0.1, color="#ba4b44", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(1, color="#a16634", ls=":", lw=1)
        ax.set(
            title=f"Joint {j + 1} · accuracy {tol:g}°, overshoot ≤0.1°",
            xlabel="Seconds after command",
            ylabel="Directional error (°)",
        )
        ax.grid(alpha=0.15)
    for trial in group:
        raw = [json.loads(r) for r in Path(f"evidence/pid/{trial['name']}.jsonl").read_text().splitlines()]
        t = np.array([r["timestamp"] - trial["start"] for r in raw])
        error = np.rad2deg(np.array(trial["target"]) - np.array([r["q"] for r in raw])) * np.sign(
            trial["step_deg"]
        )
        for j, ax in enumerate(axes.flat):
            ax.plot(t, error[:, j], lw=1.1, label=trial["name"].removeprefix(prefix))
    axes.flat[0].legend(fontsize=8)
    for label, end in [("convergence", 3), ("hold", max(r["duration"] for r in group))]:
        for ax in axes.flat:
            ax.set_xlim(0, end)
        fig.suptitle(
            f"{title}\nPositive = short of target; negative = past target · Red line = overshoot limit"
        )
        fig.savefig(out / f"{label}.png", dpi=160)
        fig.savefig(out / f"{label}.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.4 * len(trials)), 8), layout="constrained")
    for i, r in enumerate(trials):
        axes[0].bar(i, max(r["overshoot_deg"]), color="#27876a" if r["overshoot_pass"] else "#ac7070")
        axes[1].bar(
            i,
            r["settle_and_remain_s"] if r["settle_and_remain_s"] is not None else r["duration"],
            color="#427aa3",
            hatch=None if r["settle_and_remain_s"] is not None else "//",
        )
    axes[0].axhline(0.1, color="#ba4b44", ls="--")
    axes[0].set(ylabel="Worst overshoot (°)")
    axes[1].axhline(1, color="#ba4b44", ls="--")
    axes[1].set(ylabel="Sustained settling (s)")
    for ax in axes:
        ax.set_xticks(
            range(len(trials)),
            [r["name"].removeprefix(prefix) for r in trials],
            rotation=55,
            ha="right",
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Both limits must pass · 0.1° maximum overshoot and 1 s convergence")
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    print(f"Plotted {len(trials)} trials")


if __name__ == "__main__":
    main()
