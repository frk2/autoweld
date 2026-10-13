"""Estimate constant-target settling response using existing encoder recordings."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("evidence/path-deviation-repeat")
DEST = Path("evidence/trajectory-audit")
WINDOW = 0.15


def main():
    observations = [[] for _ in range(6)]
    summaries = []
    for item in json.loads((ROOT / "coverage.json").read_text()):
        if not item["completed"]:
            continue
        path = Path(item["result_path"])
        record = json.loads(path.read_text())
        session = record["after"]["motion_reference"]["session"]
        duration = record["result"]["plan"]["duration_s"]
        rows = [json.loads(x) for x in path.with_name(path.stem + "-control.jsonl").read_text().splitlines()]
        rows += [
            json.loads(x) for x in path.with_name(path.stem + "-settling.jsonl").read_text().splitlines()
        ]
        rows = [
            r
            for r in rows
            if (r.get("motion_reference") or {}).get("session") == session
            and r["motion_reference"]["elapsed_s"] > duration + 0.08
        ]
        target = np.array(record["after"]["target_rad"])
        assert all(np.max(abs(np.array(r["target"]) - target)) < 1e-8 for r in rows)
        per_joint = []
        for j in range(6):
            t = np.array([r["feedback_timestamps"][j] for r in rows])
            q = np.array([r["q"][j] for r in rows])
            t, ids = np.unique(t, return_index=True)
            q = q[ids]
            e = target[j] - q
            integral = np.r_[0.0, np.cumsum((e[1:] + e[:-1]) * np.diff(t) / 2)]
            data = []
            for i in range(len(t)):
                end = t[i] + WINDOW
                if end > t[-1]:
                    break
                k = np.searchsorted(t, end)
                if max(np.diff(t[i : k + 1])) > 0.05:
                    continue
                end_error = np.interp(end, t, e)
                # Stay well above encoder quantization; use same-sign decay only.
                if min(abs(e[i]), abs(end_error)) < np.deg2rad(0.12) or e[i] * end_error <= 0:
                    continue
                # Trapezoidal integral of linearly interpolated error, including partial end interval.
                h = end - t[k - 1]
                end_integral = integral[k - 1] + h * (e[k - 1] + end_error) / 2
                mean_error = (end_integral - integral[i]) / WINDOW
                mean_velocity = (np.interp(end, t, q) - q[i]) / WINDOW
                data.append([mean_error, mean_velocity])
            survey = record["before"]["motor_mode_survey"][str(j + 1)]
            if data:
                points = np.array(data)
                gain = float(points[:, 0] @ points[:, 1] / (points[:, 0] @ points[:, 0]))
                observations[j].extend(data)
                per_joint.append(
                    {
                        "joint": j + 1,
                        "windows": len(data),
                        "gain_s_inv": gain,
                        "tau_s": 1 / gain if gain > 0 else None,
                        "gear_ratio": survey["Gr"],
                        "kp_apr": survey["KP_APR"],
                    }
                )
        summaries.append({"name": path.stem, "joints": per_joint})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    pooled = []
    for j, (data, ax) in enumerate(zip(observations, axes.flat)):
        if len(data) < 10:
            pooled.append(
                {
                    "joint": j + 1,
                    "windows": len(data),
                    "tau_s": None,
                    "note": "Insufficient above-threshold constant-target settling data",
                }
            )
            ax.set(title=f"J{j + 1}: insufficient settling amplitude")
            ax.text(0.5, 0.5, "No reliable fit above 0.12° error", ha="center", transform=ax.transAxes)
            continue
        points = np.array(data)
        x, y = points.T
        gain = float(x @ y / (x @ x))
        predicted = gain * x
        rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))
        centered_variance = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum((y - predicted) ** 2) / centered_variance)
        expected_gain = 54 / (40 if j < 3 else 10)
        record = {
            "joint": j + 1,
            "windows": len(data),
            "effective_gain_s_inv": gain,
            "tau_s": 1 / gain,
            "rmse_rad_s": rmse,
            "r_squared": r2,
            "kp_apr_divided_by_gear_ratio_s_inv": expected_gain,
        }
        pooled.append(record)
        ax.scatter(np.rad2deg(x), np.rad2deg(y), s=7, alpha=0.25, label="0.15 s observed windows")
        span = np.array([min(0, min(x)), max(0, max(x))])
        ax.plot(np.rad2deg(span), np.rad2deg(span * gain), label=f"Fit: {gain:.2f}/s; τ={1 / gain:.3f} s")
        ax.plot(
            np.rad2deg(span),
            np.rad2deg(span * expected_gain),
            "k--",
            label=f"KP_APR / Gr = {expected_gain:g}/s",
        )
        ax.set(
            title=f"J{j + 1}: constant-target settling",
            xlabel="Mean position error (degrees)",
            ylabel="Measured mean velocity (degrees/s)",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Different effective position-loop responses in recorded POS_VEL settling\n"
        "Encoder displacement, fixed targets, error >0.12°; dashed line is a firmware-scaling hypothesis",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(DEST / "joint-response.png", dpi=140)
    plt.close(fig)
    result = {
        "window_s": WINDOW,
        "method": "Fixed-target settling; fit Δq/Δt = K × mean(target−q). "
        "Overlapping windows are correlated and are not independent trials. No hardware commands.",
        "pooled": pooled,
        "per_run": summaries,
    }
    (DEST / "joint-response.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
