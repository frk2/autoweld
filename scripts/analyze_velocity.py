"""Raw and delayed encoder-position velocity diagnostics; no robot commands."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(root: str = "evidence/velocity"):
    root = Path(root)
    records = []
    for path in sorted(root.glob("*.jsonl")):
        if path.stem.endswith("-control"):
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if not rows or "speed_m_s" not in rows[0]:
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        report = meta["result"]["plan"]
        t = np.array([r["elapsed_s"] for r in rows])
        raw = np.array([r["speed_m_s"] for r in rows]) * 1000
        slope = np.array([(r.get("position_velocity") or {}).get("speed_m_s", np.nan) for r in rows]) * 1000
        along = (
            np.array([(r.get("position_velocity") or {}).get("along_speed_m_s", np.nan) for r in rows]) * 1000
        )
        delay = np.array([(r.get("position_velocity") or {}).get("delay_s", 0) for r in rows])
        feed = report["feed_m_s"] * 1000
        edge = np.array([r["on_edge"] for r in rows])
        interior = edge & (t > report["ramp_s"] + 1) & (t < report["duration_s"] - report["ramp_s"] - 0.2)

        def metrics(mask):
            a = along[mask & np.isfinite(along)]
            return {
                "raw_peak_mm_s": float(max(raw[mask], default=0)),
                "slope_peak_mm_s": float(np.nanmax(slope[mask])) if mask.any() else None,
                "along_rmse_mm_s": float(np.sqrt(np.mean((a - feed) ** 2))) if len(a) else None,
                "along_in_band_fraction": float(np.mean(abs(a / feed - 1) <= 0.2)) if len(a) else None,
            }

        sdk = [r["sdk_timing"] for r in rows if r.get("sdk_timing")]
        record = {
            "trial": path.stem,
            "completed": meta["result"]["completed"],
            "passed": meta["result"]["passed"],
            "edge": metrics(edge),
            "interior": metrics(interior),
            "max_edge_mm": meta["result"]["max_cross_track_mm"],
            "command_compensation": meta["start"].get("command_quantization_compensation"),
            "timing": {
                k: {
                    "median_ms": float(np.median([r[k] for r in sdk]) * 1000),
                    "max_ms": float(max(r[k] for r in sdk) * 1000),
                }
                for k in ["cycle_period_s", "exchange_s", "command_age_s", "age_s"]
                if sdk
            },
        }
        records.append(record)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        x = np.clip(np.minimum(t, report["duration_s"] - t) / report["ramp_s"], 0, 1)
        ref = feed * (10 * x**3 - 15 * x**4 + 6 * x**5)
        axes[0].plot(t, raw, color="steelblue", alpha=0.65, label="Raw motor velocity → Jacobian norm")
        axes[0].plot(
            t - delay, slope, color="darkorange", label="Position slope norm (120 ms window; time aligned)"
        )
        axes[0].plot(t, ref, "k--", label="Reference")
        axes[0].fill_between(
            t[edge], feed * 0.8, feed * 1.2, color="green", alpha=0.18, label="±20% feed band"
        )
        axes[0].set_ylabel("TCP speed (mm/s)")
        axes[1].plot(t - delay, along, color="purple", label="Position slope along path (signed)")
        cross = [(r.get("position_velocity") or {}).get("cross_speed_m_s", np.nan) * 1000 for r in rows]
        axes[1].plot(t - delay, cross, label="Position slope across path")
        axes[1].plot(t, ref, "k--", label="Reference along path")
        axes[1].set_ylabel("mm/s")
        axes[2].plot(t, [r["cross_track_mm"] for r in rows], label="Finite edge distance")
        axes[2].axhline(5, color="red", ls="--", label="5 mm limit")
        axes[2].set_ylabel("mm")
        axes[2].set_xlabel("Seconds since reference started")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"{path.stem} — encoder diagnostics, not external tip metrology")
        fig.tight_layout()
        fig.savefig(root / f"{path.stem}-speed.png", dpi=150)
        plt.close(fig)
    (root / "velocity-summary.json").write_text(json.dumps(records, indent=2))
    for r in records:
        print(json.dumps(r))


if __name__ == "__main__":
    tyro.cli(main)
