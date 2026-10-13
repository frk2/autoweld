"""Replay saved POS_VEL measurements; no hardware access or motion commands."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.velocity import PositionVelocity

ROOT = Path("evidence/pos-speed-100")
DEST = Path("evidence/velocity-audit")


def load(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    plan = json.loads(path.with_suffix(".json").read_text())["result"]["plan"]
    return rows, plan


def replay(rows, plan, planner):
    stamps = np.array([r["feedback_timestamps"] for r in rows])
    epoch = stamps[0, 0]
    ts = stamps - epoch
    q = np.array([r["joint_position_rad"] for r in rows])
    raw = np.array([r["joint_velocity_raw_rad_s"] for r in rows])
    # Align all joints by interpolation within their observed ranges, never extrapolate.
    common = ts[:, 0]
    common = common[(common >= max(ts[0])) & (common <= min(ts[-1]))]
    common = np.unique(common)
    qs = np.column_stack([np.interp(common, ts[:, j], q[:, j]) for j in range(6)])
    vs = np.column_stack([np.interp(common, ts[:, j], raw[:, j]) for j in range(6)])
    elapsed = np.array([r["elapsed_s"] for r in rows])
    offset = float(np.median(elapsed[elapsed > 0] - ts[elapsed > 0, 0]))
    t = common + offset
    pos = np.array([planner.fk(a)[:3, 3] for a in qs])
    direction = np.array(rows[-1]["reference_m"]) - rows[0]["reference_m"]
    direction /= np.linalg.norm(direction)
    x = (pos - np.array(rows[0]["reference_m"])) @ direction
    ref = np.array([r["reference_m"] for r in rows])
    xr = np.interp(t, elapsed, (ref - ref[0]) @ direction)
    jacs = np.array([planner.jacobian(a)[:3] for a in qs])
    raw_along = np.einsum("nij,nj,i->n", jacs, vs, direction)
    raw_integrated = cumulative_trapezoid(raw_along, t, initial=0) + x[0]
    # Diagnostic hypothesis only: decoding the center of a truncation bin.
    # Preserve raw SDK velocities and safety/qualification behavior unchanged.
    half_lsb = np.array([10 / 4095] * 3 + [30 / 4095] * 3)
    centered_along = np.einsum("nij,nj,i->n", jacs, vs + half_lsb, direction)
    centered_distance = trapezoid(centered_along, t)
    estimator = PositionVelocity()
    fitted = []
    for r in rows:
        result = estimator.update(
            r["feedback_timestamps"],
            r["joint_position_rad"],
            min(r["feedback_timestamps"]) + r["feedback_age_s"],
        )
        if result:
            J = planner.jacobian(result["joint_position_rad"])[:3]
            v = J @ result["joint_velocity_rad_s"]
            fitted.append(
                [
                    result["timestamp"] - epoch + offset,
                    direction @ v,
                    np.linalg.norm(abs(J) @ np.array(result["joint_quantization_bound_rad_s"])),
                ]
            )
    fitted = np.array(fitted)
    a, b = plan["ramp_s"], plan["duration_s"] - plan["ramp_s"]
    displacement_speed = (np.interp(b, t, x) - np.interp(a, t, x)) / (b - a)
    # Integrate the fitted velocity over the SAME physical interval, including endpoints.
    knots = np.r_[a, fitted[(fitted[:, 0] > a) & (fitted[:, 0] < b), 0], b]
    slope_mean = trapezoid(np.interp(knots, fitted[:, 0], fitted[:, 1]), knots) / (b - a)
    endpoint_bound = sum(
        np.sum(abs(direction @ jacs[i])) * (25 / 65535) / 2
        for i in [np.argmin(abs(t - a)), np.argmin(abs(t - b))]
    ) / (b - a)
    # Fit an effective time constant from nonoverlapping 200ms displacement intervals.
    boundaries = np.arange(max(0.0, t[0]), min(plan["duration_s"] + 0.7, t[-1]), 0.2)
    delta = np.diff(np.interp(boundaries, t, x))
    error = np.interp(boundaries, t, xr - x)
    error_integral = (error[:-1] + error[1:]) / 2 * np.diff(boundaries)
    k = float(np.dot(delta, error_integral) / np.dot(error_integral, error_integral))
    bias = []
    for j in range(6):
        keep = np.r_[True, np.diff(ts[:, j]) > 0]
        tj, qj, vj = ts[keep, j], q[keep, j], raw[keep, j]
        bias.append(float((trapezoid(vj, tj) - (qj[-1] - qj[0])) / (tj[-1] - tj[0])))
    return {
        "feed_mm_s": plan["feed_m_s"] * 1000,
        "cruise_duration_s": b - a,
        "distance_time_cruise_mm_s": float(displacement_speed * 1000),
        "aligned_position_fit_cruise_mm_s": float(slope_mean * 1000),
        "difference_mm_s": float((slope_mean - displacement_speed) * 1000),
        "distance_time_quantization_only_bound_mm_s": float(endpoint_bound * 1000),
        "position_fit_peak_mm_s": float(fitted[:, 1].max() * 1000),
        "raw_integrated_displacement_mm": float((raw_integrated[-1] - raw_integrated[0]) * 1000),
            "encoder_displacement_mm": float((x[-1] - x[0]) * 1000),
            "bin_center_candidate_displacement_mm": float(centered_distance * 1000),
            "bin_center_candidate_error_mm": float((centered_distance-(x[-1]-x[0]))*1000),
        "raw_joint_mean_bias_rad_s": bias,
        "effective_time_constant_s": 1 / k,
        "max_feedback_age_ms": max(r["feedback_age_s"] for r in rows) * 1000,
        "max_feedback_skew_ms": float(np.ptp(stamps, axis=1).max() * 1000),
        "duplicate_snapshots": int(sum(np.any(np.diff(stamps, axis=0) == 0, axis=1))),
        "invalid_timing_snapshots": int(
            sum(r["feedback_age_s"] > 0.05 or np.ptp(r["feedback_timestamps"]) > 0.01 for r in rows)
        ),
    }, (t, x, xr, raw_along, raw_integrated, fitted, plan)


def predict(t, target, initial, tau):
    y = [initial]
    for i, dt in enumerate(np.diff(t)):
        # Exact first-order integration with a linearly interpolated input.
        decay = np.exp(-dt / tau)
        slope = (target[i + 1] - target[i]) / dt
        y.append(target[i + 1] - tau * slope + (y[-1] - target[i] + tau * slope) * decay)
    return np.array(y)


def main():
    DEST.mkdir(exist_ok=True)
    model = tool_model()
    planner = MotionPlanner(model)
    records, series = [], {}
    try:
        # Validate the actual TCP Jacobian against central finite differences of FK.
        rng = np.random.default_rng(15)
        errors = []
        for q in rng.uniform(-0.35, 0.35, (50, 6)):
            J = planner.jacobian(q)[:3]
            fd = np.column_stack(
                [
                    (planner.fk(q + np.eye(6)[j] * 1e-6)[:3, 3] - planner.fk(q - np.eye(6)[j] * 1e-6)[:3, 3])
                    / 2e-6
                    for j in range(6)
                ]
            )
            errors.append(float(abs(J - fd).max()))
        for path in sorted(ROOT.glob("line120-*.jsonl")):
            if path.stem.endswith("-control"):
                continue
            rows, plan = load(path)
            record, data = replay(rows, plan, planner)
            record["name"] = path.stem
            records.append(record)
            series[path.stem] = data
        tau = next(r["effective_time_constant_s"] for r in records if r["name"] == "line120-01-20mms")
        for r in records:
            t, x, xr, _, _, _, _ = series[r["name"]]
            predicted = predict(t, xr, x[0], tau)
            r["held_out_tau_prediction_rmse_mm"] = float(np.sqrt(np.mean((predicted - x) ** 2)) * 1000)
        result = {
            "jacobian_max_absolute_error_m_per_rad": max(errors),
            "raw_velocity_lsb_rad_s": [20 / 4095] * 3 + [60 / 4095] * 3,
            "fit_tau_from_Y_20mm_s": tau,
            "trials": records,
        }
        (DEST / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in ["line120-01-20mms", "line120-03-100mms", "line120-04-100mms"]:
            t, x, xr, raw, integ, fit, plan = series[name]
            fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
            phase = np.clip(np.minimum(t, plan["duration_s"] - t) / plan["ramp_s"], 0, 1)
            reference = plan["feed_m_s"] * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
            axes[0].plot(t, raw * 1000, alpha=0.6, label="Raw signed Jacobian velocity")
            axes[0].plot(fit[:, 0], fit[:, 1] * 1000, label="Position derivative, correct time and pose")
            axes[0].plot(t, reference * 1000, "k--", label="Reference")
            axes[0].fill_between(
                fit[:, 0],
                (fit[:, 1] - fit[:, 2]) * 1000,
                (fit[:, 1] + fit[:, 2]) * 1000,
                color="orange",
                alpha=0.15,
                label="Quantization-only bound",
            )
            axes[0].set_ylabel("Along-path mm/s")
            axes[1].plot(t, x * 1000, label="Encoder position / FK")
            axes[1].plot(t, integ * 1000, label="Integral of raw velocity")
            axes[1].plot(t, xr * 1000, "k--", label="Reference position")
            axes[1].set_ylabel("Displacement mm")
            axes[2].plot(t, x * 1000, label="Encoder position / FK")
            axes[2].plot(
                t,
                predict(t, xr, x[0], tau) * 1000,
                "--",
                label=f"First-order prediction, τ={tau:.3f}s from Y20",
            )
            axes[2].set_ylabel("Displacement mm")
            axes[2].set_xlabel("Feedback time since reference start (s)")
            for a in axes:
                a.axvspan(plan["ramp_s"], plan["duration_s"] - plan["ramp_s"], color="green", alpha=0.06)
                a.grid(alpha=0.2)
                a.legend(fontsize=8)
            fig.suptitle(f"{name}: velocity audit — encoder/model estimates")
            fig.tight_layout()
            fig.savefig(DEST / f"{name}.png", dpi=150)
            plt.close(fig)
        print(json.dumps(result, indent=2))
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    main()
