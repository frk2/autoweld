"""Plot recorded Stage 3 tip error and feed speed; does not communicate with the arm."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(
    evidence_dir: str = "evidence/stage3",
    pattern: str = "line-*.jsonl",
    output: str = "line-tracking",
    speed_only: bool = False,
):
    root = Path(evidence_dir)
    traces = sorted(root.glob(pattern))
    if not traces:
        raise RuntimeError("No Stage 3 traces")
    fig, axes = plt.subplots(
        len(traces),
        1 if speed_only else 2,
        figsize=(10 if speed_only else 12, max(3, 2.5 * len(traces))),
        squeeze=False,
    )
    for ax, path in zip(axes, traces):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        t = np.array([r["elapsed_s"] for r in rows])
        error = np.array([r["tracking_error_mm"] for r in rows])
        cross = np.array([r["cross_track_mm"] for r in rows])
        speed = np.array([r["speed_m_s"] * 1000 for r in rows])
        target = rows[0]["requested_speed_m_s"] * 1000
        edge = np.array([r["on_edge"] for r in rows])
        if not speed_only:
            edge_metric = rows[0].get("position_metric") == "edge_distance"
            ax[0].plot(t, error, label="Timed error (recorded only)" if edge_metric else "Timed tip error")
            if edge_metric:
                ax[0].plot(t, cross, label="Distance to edge", alpha=0.8)
            elif edge.any():
                ax[0].plot(t[edge], cross[edge], label="Distance to edge", alpha=0.8)
            tolerance = rows[0].get("position_tolerance_mm", 1)
            ax[0].axhline(
                tolerance,
                color="red",
                ls="--",
                label=f"{tolerance:g} mm {'edge' if edge_metric else 'timed error'} limit",
            )
            ax[0].set_ylabel("mm")
        speed_ax = ax[-1]
        speed_ax.plot(t, speed, label="Raw encoder/Jacobian speed")
        metadata = json.loads(path.with_suffix(".json").read_text())["result"]["plan"]
        ramp, duration = metadata["ramp_s"], metadata["duration_s"]
        x = np.clip(np.minimum(t, duration - t) / ramp, 0, 1)
        reference = target * (10 * x**3 - 15 * x**4 + 6 * x**5)
        speed_ax.plot(t, reference, color="black", ls="--", label="Planned ramp/feed")
        if edge.any():
            speed_ax.fill_between(
                t[edge], 0.8 * target, 1.2 * target, color="green", alpha=0.15, label="±20% band on edge only"
            )
        for a in ax:
            a.set_xlabel("Seconds since start")
            a.grid(alpha=0.25)
            a.set_title(path.stem + ("" if edge.any() else " — stopped in lead-in"))
            a.legend(fontsize=7)
        speed_ax.set_ylabel("mm/s")
    fig.suptitle("Stage 3 hardware trials — encoder-model estimates, not external tip metrology")
    fig.tight_layout()
    fig.savefig(root / f"{output}.png", dpi=160)
    fig.savefig(root / f"{output}.svg")
    plt.close(fig)


if __name__ == "__main__":
    tyro.cli(main)
