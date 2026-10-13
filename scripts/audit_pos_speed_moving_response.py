"""Offline position-loop response identification from existing recorded paths."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("evidence/path-deviation-repeat")
    dest = Path("evidence/trajectory-audit")
    pooled = [[] for _ in range(6)]
    perrun = []
    for fixture in json.loads((root / "coverage.json").read_text()):
        if not fixture["completed"]:
            continue
        p = Path(fixture["result_path"])
        rec = json.loads(p.read_text())
        tracking = [json.loads(x) for x in p.with_suffix(".jsonl").read_text().splitlines()]
        epoch = float(
            np.median(
                [min(r["feedback_timestamps"]) - r["elapsed_s"] for r in tracking if r["elapsed_s"] > 0]
            )
        )
        session = rec["after"]["motion_reference"]["session"]
        ctrl = [json.loads(x) for x in p.with_name(p.stem + "-control.jsonl").read_text().splitlines()]
        ctrl = [r for r in ctrl if (r.get("motion_reference") or {}).get("session") == session]
        tc = np.array([r["motion_reference"]["elapsed_s"] + epoch for r in ctrl])
        qc = np.array([r["motion_reference"]["q"][:6] for r in ctrl])
        tc, ids = np.unique(tc, return_index=True)
        qc = qc[ids]
        cap = rec["before"]["position_speed_limit_rad_s"]
        duration = rec["result"]["plan"]["duration_s"]
        fit = []
        for j in range(6):
            t = np.array([r["feedback_timestamps"][j] for r in ctrl])
            q = np.array([r["q"][j] for r in ctrl])
            t, ids = np.unique(t, return_index=True)
            q = q[ids]
            mask = (t >= tc[0]) & (t <= tc[-1])
            t = t[mask]
            q = q[mask]
            e = np.interp(t, tc, qc[:, j]) - q
            integ = np.r_[0, np.cumsum((e[1:] + e[:-1]) * np.diff(t) / 2)]
            data = []
            for i in range(len(t)):
                end = t[i] + 0.15
                if end > min(t[-1], epoch + duration):
                    break
                k = np.searchsorted(t, end)
                ee = np.interp(end, t, e)
                if (
                    max(np.diff(t[i : k + 1])) > 0.05
                    or min(abs(e[i]), abs(ee)) < np.deg2rad(0.12)
                    or e[i] * ee <= 0
                ):
                    continue
                h = end - t[k - 1]
                ei = integ[k - 1] + h * (e[k - 1] + ee) / 2
                x = (ei - integ[i]) / 0.15
                y = (np.interp(end, t, q) - q[i]) / 0.15
                if abs(y) > 0.8 * cap:
                    continue
                data.append([x, y])
            pooled[j].extend(data)
            if len(data) >= 10:
                x, y = np.array(data).T
                k = float(x @ y / (x @ x))
                fit.append({"joint": j + 1, "gain_s_inv": k, "tau_s": 1 / k, "windows": len(data)})
        perrun.append({"name": p.stem, "fits": fit})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    summary = []
    for j, (data, ax) in enumerate(zip(pooled, axes.flat)):
        if len(data) < 10:
            summary.append({"joint": j + 1, "windows": len(data)})
            continue
        x, y = np.array(data).T
        k = float(x @ y / (x @ x))
        expected = 54 / (40 if j < 3 else 10)
        r2 = float(1 - np.sum((y - k * x) ** 2) / np.sum((y - y.mean()) ** 2))
        summary.append(
            {
                "joint": j + 1,
                "windows": len(data),
                "effective_gain_s_inv": k,
                "tau_s": 1 / k,
                "r_squared": r2,
                "kp_apr_divided_by_gear_ratio_s_inv": expected,
            }
        )
        ax.scatter(np.rad2deg(x), np.rad2deg(y), s=6, alpha=0.2, label="Moving reference: 0.15 s windows")
        span = np.array([min(0, min(x)), max(0, max(x))])
        ax.plot(np.rad2deg(span), np.rad2deg(k * span), label=f"Fit {k:.2f}/s; τ={1 / k:.3f} s")
        ax.plot(np.rad2deg(span), np.rad2deg(expected * span), "k--", label=f"KP_APR / Gr = {expected:g}/s")
        ax.set(
            title=f"J{j + 1} — R²={r2:.3f}",
            xlabel="Mean position error (degrees)",
            ylabel="Mean measured velocity (degrees/s)",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "POS_VEL effective response during the 16 recorded paths\nPer-motor time alignment; encoder-displacement velocity; firmware gear scaling remains a hypothesis",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(dest / "joint-moving-response.png", dpi=140)
    plt.close(fig)
    output = {
        "method": "Moving-reference fit Δq/Δt=K mean(qref−q), 0.15s overlapping windows, error >0.12°, mean speed <80% cap; reject feedback gaps >50ms. SDK reference aligned to feedback using recorded elapsed time. No hardware I/O. Overlapping windows are correlated; model fits do not prove firmware internals.",
        "pooled": summary,
        "per_run": perrun,
    }
    (dest / "joint-moving-response.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
