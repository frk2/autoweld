"""Offline paired-approach gravity fit; supported rest is never included."""

import json
from pathlib import Path

import numpy as np

DEST = Path("evidence/mixed-gravity")


def analyze(path):
    run = json.loads(path.read_text())
    rows = [json.loads(x) for x in path.with_name(path.stem + "-control.jsonl").read_text().splitlines()]
    holds = []
    for h in run["holds"]:
        samples = [r for r in rows if h["begin"] <= r["timestamp"] <= h["end"]]
        q = np.array([r["q"] for r in samples])
        error = np.array(h["target"]) - q
        gravity = np.array([r["torque"]["gravity_unscaled_nm"][:6] for r in samples])
        kp = np.array(samples[0]["pid_kp"][:6])
        holds.append(
            dict(
                pose=h["pose"],
                side=h["side"],
                samples=len(samples),
                error_deg=np.rad2deg(error.mean(axis=0)).tolist(),
                position_span_deg=np.rad2deg(np.ptp(q, axis=0)).tolist(),
                correction_nm=(kp * error.mean(axis=0)).tolist(),
                gravity_nm=gravity.mean(axis=0).tolist(),
            )
        )
    pairs = []
    for pose in dict.fromkeys(x["pose"] for x in holds):
        a, b = [x for x in holds if x["pose"] == pose]
        assert a["side"] == -1 and b["side"] == 1
        pairs.append(
            dict(
                pose=pose,
                common_error_deg=((np.array(a["error_deg"]) + b["error_deg"]) / 2).tolist(),
                half_hysteresis_deg=((np.array(a["error_deg"]) - b["error_deg"]) / 2).tolist(),
                correction_nm=((np.array(a["correction_nm"]) + b["correction_nm"]) / 2).tolist(),
                gravity_nm=((np.array(a["gravity_nm"]) + b["gravity_nm"]) / 2).tolist(),
            )
        )
    fits = []
    for j in (3, 4):
        g = np.array([x["gravity_nm"][j] for x in pairs])
        tau = np.array([x["correction_nm"][j] for x in pairs])
        delta = float(g @ tau / (g @ g))
        slope, intercept = np.linalg.lstsq(np.c_[g, np.ones(len(g))], tau, rcond=None)[0]
        fits.append(
            dict(
                joint=j + 1,
                factor=run["requested_factors"][j],
                delta_factor_origin=delta,
                fitted_factor_origin=run["requested_factors"][j] + delta,
                delta_factor_with_intercept=float(slope),
                constant_bias_nm=float(intercept),
                residual_nm=(tau - delta * g).tolist(),
            )
        )
    return dict(name=run["name"], factors=run["requested_factors"], holds=holds, pairs=pairs, fits=fits)


def main():
    results = [
        analyze(p) for p in sorted(DEST.glob("calibration-*.json")) if "holds" in json.loads(p.read_text())
    ]
    (DEST / "hold-analysis.json").write_text(json.dumps(results, indent=2))
    for run in results:
        print(run["name"], "factors", run["factors"][3:5])
        for p in run["pairs"]:
            print(
                p["pose"],
                "common deg",
                np.round(p["common_error_deg"][3:5], 4),
                "half hysteresis deg",
                np.round(p["half_hysteresis_deg"][3:5], 4),
            )
        print("fits", run["fits"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, run in enumerate(results):
        for ax, j in zip(axes, (3, 4)):
            common = np.array([p["common_error_deg"][j] for p in run["pairs"]])
            half = np.abs([p["half_hysteresis_deg"][j] for p in run["pairs"]])
            ax.errorbar(
                np.arange(len(common)) + (i - 1) * 0.16,
                common,
                yerr=half,
                fmt="o",
                capsize=4,
                label=f"J4/J5 factors {run['factors'][3]:g}/{run['factors'][4]:g}",
            )
            ax.set(
                xticks=np.arange(len(common)),
                xticklabels=[p["pose"].replace("_", " ") for p in run["pairs"]],
                ylabel="Target minus measured angle (°)",
                title=f"J{j + 1}: mean and approach-direction range",
            )
            ax.axhline(0, color="black", lw=0.7)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Free-space gravity calibration: average the two approaches\n"
        "Bars connect opposite-direction hold means; they are hysteresis, not confidence intervals"
    )
    fig.savefig(DEST / "gravity-hold-comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
