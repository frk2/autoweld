"""One bounded live trial through the running app's automatic-test inbox."""

import fcntl
import json
import time
import uuid
from pathlib import Path

import numpy as np
import tyro


def request(root, payload, *, timeout_s=5.):
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('Reply timeout must be finite and positive')
    name = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    path = root / "requests" / f"{name}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload))
    temp.replace(path)
    response = root / "responses" / f"{name}.json"
    deadline = time.monotonic() + timeout_s
    while not response.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("No app reply; inspect state before any further motion")
        time.sleep(0.01)
    result = json.loads(response.read_text())
    if not result["ok"]:
        raise RuntimeError(result)


def main(
    name: str,
    controller: str = "PID",
    ki: tuple[float, ...] = (0, 0, 0, 0, 0, 0),
    delta: tuple[float, ...] = (0, 0, 0.08, 0, 0, 0),
    duration: float = 12.0,
    runtime: str = "runtime/live",
    ik: bool = False,
    deadband_deg: float = 0.06,
    kp: tuple[float, ...] | None = None,
    kd: tuple[float, ...] | None = None,
    accuracy_deg: tuple[float, ...] = (0.2, 0.2, 0.2, 1.0, 1.0, 1.0),
    max_overshoot_deg: float = 0.1,
    gravity_factors: tuple[float, ...] | None = None,
    gravity_model: str | None = None,
    gravity_calibration: bool = False,
    friction_compensation: bool = False,
):
    if not np.isfinite(max_overshoot_deg) or max_overshoot_deg < 0:
        raise ValueError("Overshoot limit must be finite and nonnegative")
    tolerance = np.asarray(accuracy_deg, dtype=float)
    if tolerance.shape != (6,) or np.any(tolerance <= 0) or not np.isfinite(tolerance).all():
        raise ValueError("Expected six positive finite accuracy tolerances")
    root = Path(runtime)
    trial_lock = (root / "tuning.lock").open("a")
    try:
        fcntl.flock(trial_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another tuning trial is still recording; wait for it to finish") from exc
    state = json.loads((root / "status.json").read_text())
    if state.get("closed") or state.get("fault") or time.time() - state["timestamp"] > 1:
        raise RuntimeError("Need healthy enabled app")
    if not state.get("oscillation_guard_active"):
        raise RuntimeError("Live tuning requires the SDK oscillation guard")
    expected_friction = False if gravity_calibration else friction_compensation
    if state.get("gravity") is not None:
        if not state["gravity"]["enabled"] or state["gravity"]["friction_enabled"] != expected_friction:
            raise RuntimeError("Gravity/friction flags do not match the requested test configuration")
    if gravity_calibration and any(ki):
        raise ValueError("Gravity calibration requires I=0 to isolate holding torque")
    if kd is not None:
        d = np.asarray(kd)
        if d.shape != (6,) or not np.isfinite(d).all() or np.any(d <= 0) or np.any(d > 5):
            raise ValueError("Native D must be in (0, 5]")
        if np.any(abs(d / np.asarray(state["pid_kd"]) - 1.0) > 0.2 + 1e-12):
            raise ValueError("D steps must stay within 20%")
    if kp is not None:
        candidate = np.asarray(kp)
        if not np.isfinite(candidate).all() or np.any(candidate <= 0) or np.any(candidate > 500):
            raise ValueError("Test P gains must be within the SDK motor range (0, 500]")
        previous = np.asarray(state["pid_kp"])
        if candidate.shape != (6,) or np.any(abs(candidate / previous - 1.0) > 0.2 + 1e-12):
            raise ValueError("P steps must stay within 20% of the current setting")
    rest = np.array(state["rest_rad"])
    target = rest + np.array(delta)
    if len(delta) != 6 or np.max(np.abs(delta)) >= 0.49:
        raise ValueError("Test excursions must be below .49 rad")
    config = {"action": "controller", "controller": controller, "ki": list(ki), "deadband_deg": deadband_deg}
    if kp is not None:
        config["kp"] = list(kp)
    if kd is not None:
        config["kd"] = list(kd)
    if gravity_factors is not None or gravity_model is not None:
        factors = np.asarray(state["gravity_factors"] if gravity_factors is None else gravity_factors)
        old_factors = np.asarray(state["gravity_factors"])
        if factors.shape != (6,) or not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError("Expected six finite positive gravity factors")
        if np.any(abs(factors / old_factors - 1) > .2 + 1e-12):
            raise ValueError("Automatic gravity tuning steps are limited to 20%")
        request(root, {"action": "gravity_feedforward", "factors": factors.tolist(),
                       "model": gravity_model or state["gravity_model"]})
    request(root, config)
    initial_q = np.array(state["q_rad"])
    payload = {"action": "joint", "q": target.tolist()}
    if ik:
        import mujoco
        from i2rt.robots.kinematics import Kinematics
        from scipy.spatial.transform import Rotation

        from autoweld.app import tool_model

        model_path = tool_model()
        model = mujoco.MjModel.from_xml_path(model_path)
        kinematics = Kinematics(model_path, "weld_tcp")
        goal = np.zeros(model.nq)
        goal[:6] = target
        transform = kinematics.fk(goal)
        payload = {
            "action": "pose",
            "position": transform[:3, 3].tolist(),
            "rpy": Rotation.from_matrix(transform[:3, :3]).as_euler("xyz").tolist(),
        }
    path = root / "control-telemetry.jsonl"
    with path.open() as stream:
        stream.seek(0, 2)
        start = time.time()
        request(root, payload)
        if ik:
            deadline = time.monotonic() + 2.0
            while state["timestamp"] <= start:
                if time.monotonic() > deadline:
                    raise TimeoutError("No updated IK state")
                time.sleep(0.02)
                state = json.loads((root / "status.json").read_text())
            target = np.array(state["target_rad"])
        while time.time() - start < duration:
            time.sleep(0.1)
            state = json.loads((root / "status.json").read_text())
            if state.get("fault") or time.time() - state["timestamp"] > 1:
                out = Path("evidence/pid")
                out.mkdir(exist_ok=True)
                (out / f"{name}-aborted.jsonl").write_text(stream.read())
                (out / f"{name}-aborted.json").write_text(json.dumps({"name": name, "aborted": True, "start": start, "config": config, "request": payload, "state": state}, indent=2))
                raise RuntimeError("Trial interrupted: stale/faulted feedback; partial evidence saved")
            if state.get("gravity") and (not state["gravity"]["enabled"] or
                                         state["gravity"]["friction_enabled"] != expected_friction):
                raise RuntimeError("Compensation changed during the test; leave operator settings intact")
            if np.max(np.abs(np.array(state["q_rad"]) - rest)) >= 0.49:
                request(root, {"action": "rest"})
                raise RuntimeError("Measured test excursion reached .49 rad; returning to rest")
            if not np.allclose(state["target_rad"], target, atol=1e-9):
                # Initial status may predate the command; later mismatch means user intervention.
                if state["timestamp"] > start + 0.3:
                    raise RuntimeError("Target changed during test; leave operator command intact")
        records = [json.loads(line) for line in stream if line.endswith("\n")]
    records = [r for r in records if np.allclose(r["target"], target, atol=1e-9)]
    t = np.array([r["timestamp"] - start for r in records])
    errors = np.rad2deg(target - np.array([r["q"] for r in records]))
    worst = np.max(np.abs(errors) / tolerance, axis=1)
    settle = None
    for i in range(len(t)):
        window = (t >= t[i]) & (t <= t[i] + 1.0)
        if t[-1] >= t[i] + 1.0 and np.all(worst[window] <= 1.0):
            settle = float(t[i])
            break
    last_failure = np.flatnonzero(worst > 1.0)
    stable_index = int(last_failure[-1]) + 1 if len(last_failure) else 0
    stable = float(t[stable_index]) if stable_index < len(t) and t[-1] - t[stable_index] >= 1.0 else None
    tail = t >= t[-1] - 2.0
    positions = np.array([r["q"] for r in records])
    sampled_speed = np.diff(positions, axis=0) / np.diff(t)[:, None]
    step = target - initial_q
    moved = np.abs(step) > np.deg2rad(0.1)
    overshoot = np.maximum(0.0, np.max(-errors * np.sign(step), axis=0))
    overshoot = np.where(moved, overshoot, 0.0)
    summary = {
        "initial_q": initial_q.tolist(),
        "step_deg": np.rad2deg(step).tolist(),
        "overshoot_deg": overshoot.tolist(),
        "max_overshoot_limit_deg": max_overshoot_deg,
        "overshoot_pass": bool(np.all(overshoot < max_overshoot_deg)),
        "speed_and_overshoot_pass": bool(
            stable is not None and stable <= 1.0 and np.all(overshoot < max_overshoot_deg)
        ),
        "overshoot_percent": (100 * overshoot / np.maximum(np.abs(np.rad2deg(step)), 1e-9)).tolist(),
        "name": name,
        "controller": controller,
        "ki": list(ki),
        "kp": state["pid_kp"],
        "kd": state["pid_kd"],
        "gravity_factors": state.get("gravity_factors"),
        "gravity_model": state.get("gravity_model"),
        "gravity": state.get("gravity"),
        "gravity_calibration": gravity_calibration,
        "friction": state.get("friction"),
        "final_torque": records[-1].get("torque"),
        "integral_deadband_deg": state["integral_deadband_deg"],
        "target": target.tolist(),
        "request": payload,
        "start": start,
        "duration": duration,
        "samples": len(records),
        "final_2s_peak_to_peak_deg": np.ptp(errors[tail], axis=0).tolist(),
        "peak_sampled_speed_rad_s": np.max(np.abs(sampled_speed), axis=0).tolist(),
        "accuracy_tolerance_deg": tolerance.tolist(),
        "settle_1s_s": settle,
        "settle_and_remain_s": stable,
        "converged_within_1s": stable is not None and stable <= 1.0,
        "error_at_1s_deg": errors[np.argmin(np.abs(t - 1.0))].tolist(),
        "final_2s_max_abs_error_deg": np.max(np.abs(errors[tail]), axis=0).tolist(),
        "final_2s_mean_error_deg": np.mean(errors[tail], axis=0).tolist(),
        "final_integral_nm": records[-1]["integral"],
        "peak_abs_error_deg": np.max(np.abs(errors), axis=0).tolist(),
    }
    out = Path("evidence/pid")
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2))
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
