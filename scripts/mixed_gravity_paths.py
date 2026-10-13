"""Matched live paths with calibrated free-space J4/J5 gravity factors."""

import fcntl
import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts import mixed_wrist_trial as io
from scripts.kp_path_repeat import check_gains
from scripts.mixed_gravity_calibrate import BASE, DEST, factors_to


def main(label: str, j4: float, j5: float, repetitions: int = 2):
    if not 0.8 <= j4 <= 1.8 or not 0.5 <= j5 <= 2 or repetitions not in (1, 2):
        raise ValueError("Calibration search bounds exceeded")
    io.DEST = DEST
    controller = "MIXED_J45"
    factors = BASE.copy()
    factors[3:5] = [j4, j5]
    name = f"tuned-{label}-{time.time_ns()}"
    spec = json.loads((DEST / "preflight.json").read_text())
    assert all(x["ok"] for x in spec["fixtures"]), "Offline preflight failed"
    record = dict(
        name=name, controller=controller, board_state="removed", runs=[], requested_factors=factors.tolist()
    )
    offset = io.TELEMETRY.stat().st_size
    with (io.ROOT / "tuning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert io.healthy()["closed"], "Start disabled at captured rest"
            io.send(dict(action="controller", controller=controller))
            factors_to(BASE)
            io.send(dict(action="position_speed_limit", limit_rad_s=0.25))
            record["enabled"] = io.send(dict(action="enable"))
            check_gains(record["enabled"])
            assert record["enabled"]["gravity"]["active_arm_joints"] == [4, 5]
            io.observe(3, controller)
            io.send(dict(action="rest"))
            io.settled(controller)
            io.observe(3, controller)
            for repetition in range(repetitions):
                for fixture in spec["fixtures"]:
                    trial_name = f"{name}-{fixture['name']}-r{repetition + 1}"
                    print("TRIAL", trial_name, flush=True)
                    begin_offset = io.TELEMETRY.stat().st_size
                    io.send(dict(action="joint", q=fixture["seed_q"][:6]))
                    io.settled(controller)
                    factors_to(factors)
                    io.settled(controller)
                    io.send(dict(action="track_load", path=fixture["asset"]))
                    payload = dict(
                        edge=0,
                        feed_m_s=fixture["feed_m_s"],
                        orientation="Unconstrained",
                        ramp_placement="Inside edge",
                    )
                    io.send(dict(action="track_approach", **payload))
                    before = io.settled(controller, path_start=True)
                    np.testing.assert_allclose(before["gravity_factors"], factors, atol=1e-12)
                    io.observe(4, controller)
                    running = io.send(dict(action="track_run", **payload))
                    deadline = time.monotonic() + running["motion_plan"]["duration_s"] + 15
                    while time.monotonic() < deadline:
                        after = io.healthy(controller)
                        result = after.get("tracking") or {}
                        if not result.get("running", True) and result.get("trace"):
                            break
                        time.sleep(0.05)
                    else:
                        raise TimeoutError("No finished tracking result")
                    (DEST / f"{trial_name}.jsonl").write_bytes(Path(result["trace"]).read_bytes())
                    trial = dict(
                        name=trial_name,
                        fixture=fixture,
                        before=before,
                        after=after,
                        result=result,
                        analysis_condition="MIXED_TUNED",
                    )
                    record["runs"].append(trial)
                    (DEST / f"{trial_name}.json").write_text(json.dumps(trial, indent=2))
                    if not result["completed"]:
                        raise RuntimeError("Reference interrupted; inspect before another trial")
                    io.observe(5, controller)
                    io.archive_tail(begin_offset, DEST / f"{trial_name}-control.jsonl")
                    print("COMPLETED", trial_name, "edge mm", result["max_cross_track_mm"], flush=True)
                    # The supported rest is excluded from free-space tuning.
                    factors_to(BASE)
                    io.send(dict(action="rest"))
                    io.settled(controller)
            io.disable(name)
            record["final"] = io.state()
        except Exception as exc:
            record["error"] = str(exc)
            record["error_status"] = io.state()
            print("STOPPED", str(exc), flush=True)
            s = io.state()
            if not s["closed"] and not s["fault"] and not s["motion_fault"]:
                if s["motion_active"]:
                    io.send(dict(action="mode", mode="Hold"))
                factors_to(BASE)
                io.disable(name)
            raise
        finally:
            io.archive_tail(offset, DEST / f"{name}-session-control.jsonl")
            (DEST / f"{name}-session.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
