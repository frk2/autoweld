"""Compare three recorded controllers/conditions on identical valid time masks."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from scripts.analyze_mixed_tcp import FS, filter_valid, load_trial, split_valid, vector_rms

DEST = Path("evidence/mixed-gravity")
CONDITIONS = ("POS_VEL", "MIXED_J45", "MIXED_TUNED")
LABELS = ("Position–speed", "Mixed, previous gravity", "Mixed, calibrated gravity")
COLORS = ("#0072b2", "#d55e00", "#009e73")


def peak(signal, keep, high):
    powers = []
    for a, b in split_valid(keep):
        if b - a >= 200:
            f, p = welch(signal[a:b], fs=FS, nperseg=200, noverlap=100, axis=0)
            if p.ndim == 2:
                p = p.sum(axis=1)
            powers.append((p, b - a))
    if not powers:
        return None
    power = np.average([p for p, _ in powers], weights=[n for _, n in powers], axis=0)
    band = (f >= 2) & (f <= high)
    return float(f[band][np.argmax(power[band])])


def main():
    paths = sorted(Path("evidence/mixed-wrist").glob("*-r[12].json"))
    # Predeclare a complete two-repeat block for the primary comparison. Retain
    # completed paths from interrupted blocks as supplementary evidence.
    excluded = []
    for session in sorted(DEST.glob("tuned-*-session.json")):
        capture = json.loads(session.read_text())
        selected = [DEST / (r["name"] + ".json") for r in capture["runs"]]
        if capture.get("error"):
            excluded.extend(str(p) for p in selected)
        else:
            assert capture["final"]["closed"] and len(selected) == 4
            paths += selected
    (DEST / "supplementary-completed-paths.json").write_text(json.dumps(excluded, indent=2))
    model = tool_model()
    runs = []
    try:
        planner = MotionPlanner(model)
        for path in paths:
            trial = json.loads(path.read_text())
            assert trial["result"]["completed"]
            run = load_trial(path, planner)
            run["condition"] = trial.get("analysis_condition", run["controller"])
            run["source"] = str(path)
            run["gravity_factors"] = trial["before"]["gravity_factors"]
            run["trace"] = [json.loads(x) for x in path.with_suffix(".jsonl").read_text().splitlines()]
            run["result"] = trial["result"]
            for high in (20.0, 10.0):
                joint, trusted = filter_valid(
                    run["t"], np.rad2deg(run["q"] - run["reference_q"]), run["valid"], high
                )
                run["bands"][high]["joint"] = joint
                np.testing.assert_array_equal(trusted, run["bands"][high]["trusted"])
            runs.append(run)
    finally:
        Path(model).unlink()
    metrics, path_metrics = [], []
    for fixture in ("line120", "circle60"):
        matched = [r for r in runs if r["fixture"] == fixture]
        assert len(matched) == 6, "Need exactly two repeats of each condition"
        for run in matched:
            np.testing.assert_array_equal(run["t"], matched[0]["t"])
            trace = [r for r in run["trace"] if 0 <= r["elapsed_s"] <= run["duration_s"]]
            edge = np.array([r["cross_track_mm"] for r in trace])
            keep = (run["t"] >= 0) & (run["t"] <= run["duration_s"]) & run["valid"]
            path_metrics.append(
                dict(
                    name=run["name"],
                    condition=run["condition"],
                    fixture=fixture,
                    source=run["source"],
                    gravity_factors=run["gravity_factors"],
                    max_edge_mm=run["result"]["max_cross_track_mm"],
                    max_moving_edge_mm=float(edge.max()),
                    path_edge_rms_mm=float(np.sqrt(np.mean(edge**2))),
                    path_and_hold_edge_rms_mm=float(
                        np.sqrt(np.mean([r["cross_track_mm"] ** 2 for r in run["trace"]]))
                    ),
                    max_timed_error_mm=run["result"]["max_tracking_error_mm"],
                    mean_joint_error_deg=np.rad2deg(
                        (run["q"] - run["reference_q"])[keep].mean(axis=0)
                    ).tolist(),
                    joint_span_deg=np.rad2deg(np.ptp(run["q"][keep], axis=0)).tolist(),
                    reference_joint_span_deg=np.rad2deg(np.ptp(run["reference_q"][keep], axis=0)).tolist(),
                )
            )
        for high in (20.0, 10.0):
            common = np.logical_and.reduce([r["bands"][high]["trusted"] for r in matched])
            for run in matched:
                t, duration, ramp = run["t"], run["duration_s"], run["ramp_s"]
                for phase, (lo, hi) in {
                    "path": (0, duration),
                    "cruise": (ramp + 1, duration - ramp - 1),
                }.items():
                    keep = common & (t >= lo) & (t <= hi)
                    assert keep.sum() >= 100, "Too little valid matched coverage"
                    signal = run["bands"][high]["filtered"]
                    joint = run["bands"][high]["joint"]
                    values = joint[keep]
                    joint_peaks = [
                        peak(joint[:, j], keep, high) if np.ptp(run["q"][keep, j]) > 1e-8 else None
                        for j in range(6)
                    ]
                    metrics.append(
                        dict(
                            name=run["name"],
                            condition=run["condition"],
                            fixture=fixture,
                            phase=phase,
                            band_hz=[2, high],
                            samples=int(keep.sum()),
                            coverage_s=float(keep.sum() / FS),
                            interval_s=[lo, hi],
                            coverage_fraction=float(min(1, keep.sum() / FS / (hi - lo))),
                            tcp_rms_mm=vector_rms(signal[keep, :3]),
                            mount_rms_mm=vector_rms(signal[keep, 3:6]),
                            tcp_error_rms_mm=vector_rms(signal[keep, 6:9]),
                            reference_rms_mm=vector_rms(signal[keep, 9:12]),
                            tcp_peak_hz=peak(signal[:, :3], keep, high),
                            joint_rms_deg=np.sqrt(np.mean(values**2, axis=0)).tolist(),
                            joint_peak_to_peak_deg=np.ptp(values, axis=0).tolist(),
                            joint_peak_hz=joint_peaks,
                        )
                    )
                    if high == 20 and phase == "path":
                        run["keep"] = keep
    (DEST / "path-metrics.json").write_text(json.dumps(path_metrics, indent=2))
    (DEST / "shaking-metrics.json").write_text(json.dumps(metrics, indent=2))
    (DEST / "timing.json").write_text(
        json.dumps(
            [{k: r[k] for k in ("name", "timing", "max_fk_reproduction_error_mm")} for r in runs], indent=2
        )
    )
    summary = []
    for fixture in ("line120", "circle60"):
        for condition in CONDITIONS:
            m = [
                r
                for r in metrics
                if r["fixture"] == fixture
                and r["condition"] == condition
                and r["phase"] == "path"
                and r["band_hz"] == [2, 20]
            ]
            p = [r for r in path_metrics if r["fixture"] == fixture and r["condition"] == condition]
            c = [
                r
                for r in metrics
                if r["fixture"] == fixture
                and r["condition"] == condition
                and r["phase"] == "cruise"
                and r["band_hz"] == [2, 20]
            ]
            summary.append(
                dict(
                    fixture=fixture,
                    condition=condition,
                    tcp_rms_mm=float(np.mean([x["tcp_rms_mm"] for x in m])),
                    tcp_cruise_rms_mm=float(np.mean([x["tcp_rms_mm"] for x in c])),
                    tcp_peak_hz=[x["tcp_peak_hz"] for x in m],
                    joint_rms_deg=np.mean([x["joint_rms_deg"] for x in m], axis=0).tolist(),
                    joint_peak_hz=[x["joint_peak_hz"] for x in m],
                    edge_max_range_mm=[min(x["max_edge_mm"] for x in p), max(x["max_edge_mm"] for x in p)],
                    moving_edge_max_range_mm=[
                        min(x["max_moving_edge_mm"] for x in p),
                        max(x["max_moving_edge_mm"] for x in p),
                    ],
                    edge_rms_mm=float(np.mean([x["path_edge_rms_mm"] for x in p])),
                    path_and_hold_edge_rms_mm=float(np.mean([x["path_and_hold_edge_rms_mm"] for x in p])),
                )
            )
    (DEST / "comparison-summary.json").write_text(json.dumps(summary, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, fixture in enumerate(("line120", "circle60")):
        for i, (condition, label, color) in enumerate(zip(CONDITIONS, LABELS, COLORS)):
            selected = [r for r in runs if r["fixture"] == fixture and r["condition"] == condition]
            for repeat, run in enumerate(selected):
                trace = run["trace"]
                axes[row, 0].plot(
                    [r["elapsed_s"] for r in trace],
                    [r["cross_track_mm"] for r in trace],
                    color=color,
                    alpha=0.75,
                    lw=1,
                    label=label if repeat == 0 else None,
                )
                keep = run["keep"]
                signal = run["bands"][20.0]["filtered"]
                axes[row, 1].plot(
                    run["t"][keep], np.linalg.norm(signal[keep, :3], axis=1), color=color, alpha=0.6, lw=0.7
                )
            values = [
                m["tcp_rms_mm"]
                for m in metrics
                if m["fixture"] == fixture
                and m["condition"] == condition
                and m["band_hz"] == [2, 20]
                and m["phase"] == "path"
            ]
            axes[row, 2].bar(i, np.mean(values), width=0.6, color=color)
            axes[row, 2].plot([i - 0.06, i + 0.06], values, "ko", ms=4)
            axes[row, 2].text(i, max(values) + 0.008, f"{np.mean(values):.3f}", ha="center")
        axes[row, 0].set(
            title="Line: 120 mm at 20 mm/s" if row == 0 else "Circle: R60 mm at 50 mm/s",
            ylabel="Distance to finite path (mm)",
            xlabel="Reference time (s)",
        )
        axes[row, 1].set(
            title="Encoder-derived TCP shaking, 2–20 Hz",
            ylabel="3D magnitude (mm)",
            xlabel="Reference time (s)",
        )
        axes[row, 2].set(
            title="TCP shaking RMS; dots are repeats",
            ylabel="3D vector RMS (mm)",
            xticks=range(3),
            xticklabels=["Position–\nspeed", "Previous\ngravity", "Calibrated\ngravity"],
        )
        axes[row, 2].set_ylim(0, axes[row, 2].get_ylim()[1] * 1.15)
        axes[row, 0].legend(fontsize=8)
        duration = next(r["duration_s"] for r in runs if r["fixture"] == fixture)
        axes[row, 0].axvspan(duration, duration + 1.01, color="gray", alpha=0.15)
        axes[row, 0].text(
            duration + 0.5,
            0.97,
            "Final\nhold",
            ha="center",
            va="top",
            transform=axes[row, 0].get_xaxis_transform(),
            fontsize=8,
        )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "J4/J5 gravity calibration: path adherence and encoder-derived tip shaking\n"
        "Identical valid time masks across all conditions; virtual TCP 150 mm from mount; no external metrology"
    )
    fig.savefig(DEST / "path-and-shaking-comparison.png", dpi=160)
    plt.close(fig)
    for fixture in ("line120", "circle60"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
        for col, (condition, label) in enumerate(zip(CONDITIONS, LABELS)):
            for run in [r for r in runs if r["fixture"] == fixture and r["condition"] == condition]:
                keep = run["valid"] & (run["t"] >= 0) & (run["t"] <= run["duration_s"])
                for row, j in enumerate((3, 4)):
                    axes[row, col].plot(
                        run["t"][keep], np.rad2deg(run["q"][keep, j]), color="#0072b2", lw=1, alpha=0.7
                    )
                    axes[row, col].plot(
                        run["t"][keep],
                        np.rad2deg(run["reference_q"][keep, j]),
                        "--",
                        color="#d55e00",
                        lw=1,
                        alpha=0.7,
                    )
                    axes[row, col].set(xlabel="Reference time (s)", ylabel=f"J{j + 1} (°)")
            axes[0, col].set_title(label)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        fig.suptitle(f"{fixture}: raw encoder angles (blue), planned reference (orange); two repeats")
        fig.savefig(DEST / f"{fixture}-joint-tracking.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
